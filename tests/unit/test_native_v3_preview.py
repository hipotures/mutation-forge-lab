from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.native_v3.canonical import canonical_json_bytes
from mutation_forge.native_v3.persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    BOOTSTRAP_ACK_VALUE,
)
from mutation_forge.native_v3.preview import (
    FORBIDDEN_LENGTHS,
    PREVIEW_SLOT_IDS,
    run_persistent_single_ast_cohort,
)
from mutation_forge.native_v3.single_program_ir import (
    BRIEF_OPERATORS,
    SLOT_SPECIFIC_OUTPUT_CONTRACT,
    SLOT_SPECIFIC_SCHEMA_VERSION,
    build_candidate_request,
    slot_specific_contract_sha256,
    slot_specific_schema_hashes,
)
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    TurnError,
)
from mutation_forge.stage3.isolation import IsolatedCapsule

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location("step12e_fake_app_server", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario

_ARTIFACT_SUFFIXES = {
    "codex-profile.json.gz",
    "codex-rpc.jsonl",
    "events.jsonl",
    "output-schema.json.gz",
    "provider-raw.json.gz",
    "request.json.gz",
    "request.md",
    "response.json.gz",
    "response.md",
    "response.raw.txt",
    "stderr.txt",
    "stdout.jsonl",
    "system-prompt.md",
    "transcript.sha256",
    "usage.json.gz",
    "wire.jsonl",
}


def _responses() -> list[str]:
    selector_arguments = {
        "add-edge": {"mode": "min"},
        "remove-edge": {"mode": "min"},
        "relocation": {},
        "fanout": {},
    }
    action_arguments = {
        "add-edge": "edge",
        "remove-edge": "edge",
        "relocation": "relocation",
        "fanout": "fanout",
    }
    result: list[str] = []
    briefs = ("add-edge", "remove-edge", "relocation", "fanout")
    for index, brief_id in enumerate(briefs):
        operators = BRIEF_OPERATORS[brief_id]
        response = {
            "schema_version": SLOT_SPECIFIC_SCHEMA_VERSION,
            "slot_id": f"slot-{index:02d}",
            "brief_id": brief_id,
            "active_forbidden_lengths": list(FORBIDDEN_LENGTHS),
            "design_summary": f"Use the bounded {brief_id} operator family.",
            "hypothesis": "The relation-certified rewrite may improve the graph.",
            "plan": {
                "selector": {
                    "selector_id": operators.selector_id,
                    "arguments": selector_arguments[brief_id],
                },
                "pick": {"mode": "seeded_uniform"},
                "action": {
                    "action_id": operators.action_id,
                    "arguments": {
                        action_arguments[brief_id]: "selected",
                    },
                },
                "terminal": {"kind": "emit"},
            },
        }
        result.append(json.dumps(response, separators=(",", ":")))
    return result


def test_persistent_preview_publishes_one_unique_ast_per_turn(
    tmp_path: Path,
) -> None:
    responses = _responses()
    adapter_calls = 0
    capsule = IsolatedCapsule.create(tmp_path)
    processes: list[Any] = []

    def adapter_factory(
        base_instructions: str,
        selected_capsule: IsolatedCapsule,
        turns_dir: Path,
        prefix: str,
        timeout_seconds: float,
    ) -> CodexAppServerAdapter:
        nonlocal adapter_calls
        assert selected_capsule is capsule
        assert adapter_calls == 0
        scenario = FakeScenario(
            final_texts=[
                json.dumps(
                    {
                        "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                        "ack": BOOTSTRAP_ACK_VALUE,
                    },
                    separators=(",", ":"),
                ),
                *responses,
            ],
            fork_thread_ids=["worker-thread-0", "worker-thread-1"],
        )
        adapter_calls += 1

        def process_factory(*_args: Any, **kwargs: Any) -> Any:
            process = FakeProcess(scenario, **kwargs)
            processes.append(process)
            return process

        return CodexAppServerAdapter(
            capsule=selected_capsule,
            process_factory=process_factory,
            auth_checker=lambda _: True,
            limits=AppServerLimits(
                max_turns=17,
                max_campaigns=3,
                turn_timeout=timeout_seconds,
                usage_grace=0.01,
            ),
            base_instructions=base_instructions,
            artifact_dir=turns_dir,
            artifact_prefix=prefix,
            compress_json_artifacts=True,
            sandbox_mode="read-only",
            copy_rollout_artifact=False,
        )

    def backend_factory() -> Any:
        raise RuntimeError("bounded generation test does not run HEG")

    root = tmp_path / "experiment"
    report = run_persistent_single_ast_cohort(
        root,
        model="gpt-5.6-luna",
        effort="high",
        timeout_seconds=0.1,
        auth_json=tmp_path / "unused-auth.json",
        backend_factory=backend_factory,
        episode_id="test/epoch-0000",
        adapter_factory=adapter_factory,
        capsule_factory=lambda: capsule,
    )

    assert report["status"] == "evaluation_error"
    assert report["communication_mode"] == "persistent_single_ast"
    assert report["valid_slots"] == 4
    assert report["unique_valid_programs"] == 4
    assert report["model_turns"] == 5
    assert report["program_turns"] == 4
    assert report["output_contract"] == SLOT_SPECIFIC_OUTPUT_CONTRACT
    assert report["output_schema_sha256"] == slot_specific_contract_sha256(
        FORBIDDEN_LENGTHS
    )
    assert report["time_to_first_valid_ast_ms"] is not None
    assert report["first_valid_ast_published_before_cohort_complete"] is True
    assert adapter_calls == 1
    state = read_json(root / "native-v3-output/epoch-0000/communication-state.json.gz")
    assert state["status"] == "completed"
    assert state["next_slot"] == 4
    assert len(state["workers"]) == 2
    assert state["specification_thread"]["thread_id"] == "thread-1"
    assert state["specification_thread"]["anchor_turn_id"] == (state["anchor"]["turn_id"])
    assert {worker["fork_parent_turn_id"] for worker in state["workers"]} == {
        state["anchor"]["turn_id"]
    }
    assert len(state["slot_reports"]) == 4
    assert state["provider_attempts"] == 5
    assert state["program_turns"] == 4
    assert state["output_contract"] == SLOT_SPECIFIC_OUTPUT_CONTRACT
    assert state["output_schema_sha256"] == slot_specific_contract_sha256(
        FORBIDDEN_LENGTHS
    )
    assert state["failed_provider_attempts"] == 0
    assert state["provider_retries"] == 0
    assert state["provider_process_restarts"] == 0
    assert state["thread_resume_attempts"] == 0
    assert len(processes) == 1
    manifest = read_json(root / "native-v3-output/epoch-0000/epoch-manifest.json.gz")
    assert manifest["planned_slot_ids"] == list(PREVIEW_SLOT_IDS)
    assert len(manifest["provider_calls"]) == 4
    assert manifest["provider_mode"] == "persistent_single_ast"
    assert manifest["output_contract"] == SLOT_SPECIFIC_OUTPUT_CONTRACT
    assert manifest["output_schema_sha256_by_brief"] == slot_specific_schema_hashes(
        FORBIDDEN_LENGTHS
    )
    methods = [request.get("method") for request in processes[0].received_requests]
    assert methods.count("thread/start") == 1
    assert methods.count("thread/fork") == 2
    assert methods.count("thread/resume") == 0
    turn_starts = [
        request
        for request in processes[0].received_requests
        if request.get("method") == "turn/start"
    ]
    assert len(turn_starts) == 5
    assert [request["params"]["threadId"] for request in turn_starts[1:]] == [
        state["workers"][index % 2]["thread_id"] for index in range(4)
    ]

    turns_dir = root / "provider-turns"
    prefixes = ["00-spec-anchor", "00-worker-00-fork", "00-worker-01-fork"] + [
        f"slot-{index:02d}.initial" for index in range(4)
    ]
    for prefix in prefixes:
        suffixes = {
            path.name.removeprefix(f"{prefix}.")
            for path in turns_dir.iterdir()
            if path.name.startswith(f"{prefix}.")
        }
        assert suffixes == _ARTIFACT_SUFFIXES
    assert not any(path.name.endswith("rollout.jsonl") for path in turns_dir.iterdir())
    schema_hashes = slot_specific_schema_hashes(FORBIDDEN_LENGTHS)
    for index, slot_id in enumerate(PREVIEW_SLOT_IDS):
        record = read_json(
            root / "native-v3-output/epoch-0000/program-records" / f"{slot_id}.json.gz"
        )
        assert record["program"]["program_hash"]
        assert record["lineage"]["worker_index"] == index % 2
        assert record["published_at_ms"] is not None
        assert record["output_contract"] == SLOT_SPECIFIC_OUTPUT_CONTRACT
        brief_id = ("add-edge", "remove-edge", "relocation", "fanout")[index]
        assert record["output_schema_sha256"] == schema_hashes[brief_id]
        output_schema = read_json(turns_dir / f"{slot_id}.initial.output-schema.json.gz")
        assert hashlib.sha256(canonical_json_bytes(output_schema)).hexdigest() == (
            schema_hashes[brief_id]
        )
        prompt = (turns_dir / f"{slot_id}.initial.request.md").read_text(encoding="utf-8")
        assert '"search_memory"' in prompt
        assert '"entry"' not in prompt
        stdout = [
            json.loads(line)
            for line in (turns_dir / f"{slot_id}.initial.stdout.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert sum(message.get("method") == "turn/completed" for message in stdout) == 1
    assert not capsule.root.exists()


def test_persistent_preview_uses_one_replacement_process_for_both_workers(
    tmp_path: Path,
) -> None:
    responses = _responses()
    capsule = IsolatedCapsule.create(tmp_path)
    processes: list[Any] = []
    adapter_calls = 0

    def adapter_factory(
        base_instructions: str,
        selected_capsule: IsolatedCapsule,
        turns_dir: Path,
        prefix: str,
        timeout_seconds: float,
    ) -> CodexAppServerAdapter:
        nonlocal adapter_calls
        assert selected_capsule is capsule
        if adapter_calls == 0:
            scenario = FakeScenario(
                final_texts=[
                    json.dumps(
                        {
                            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                            "ack": BOOTSTRAP_ACK_VALUE,
                        },
                        separators=(",", ":"),
                    ),
                    responses[0],
                    responses[1],
                ],
                terminal_statuses=["completed", "completed", "failed"],
                fork_thread_ids=["worker-thread-0", "worker-thread-1"],
            )
        else:
            scenario = FakeScenario(final_texts=responses[1:])
        adapter_calls += 1

        def process_factory(*_args: Any, **kwargs: Any) -> Any:
            process = FakeProcess(scenario, **kwargs)
            processes.append(process)
            return process

        return CodexAppServerAdapter(
            capsule=selected_capsule,
            process_factory=process_factory,
            auth_checker=lambda _: True,
            limits=AppServerLimits(
                max_turns=17,
                max_campaigns=3,
                turn_timeout=timeout_seconds,
                usage_grace=0.01,
            ),
            base_instructions=base_instructions,
            artifact_dir=turns_dir,
            artifact_prefix=prefix,
            compress_json_artifacts=True,
            sandbox_mode="read-only",
            copy_rollout_artifact=False,
        )

    root = tmp_path / "experiment"
    kwargs = {
        "model": "gpt-5.6-luna",
        "effort": "high",
        "timeout_seconds": 0.1,
        "auth_json": tmp_path / "unused-auth.json",
        "backend_factory": lambda: (_ for _ in ()).throw(
            RuntimeError("bounded generation test does not run HEG")
        ),
        "episode_id": "test/epoch-0000",
        "adapter_factory": adapter_factory,
        "capsule_factory": lambda: capsule,
        "capsule_reopener": lambda _: capsule,
    }

    with pytest.raises(TurnError, match="turn ended with status"):
        run_persistent_single_ast_cohort(root, **kwargs)

    failed_state = read_json(root / "native-v3-output/epoch-0000/communication-state.json.gz")
    assert failed_state["next_slot"] == 1
    assert failed_state["provider_attempts"] == 3
    assert failed_state["failed_provider_attempts"] == 1
    assert failed_state["last_provider_attempt"]["slot_id"] == "slot-01"
    assert failed_state["last_provider_attempt"]["status"] == "failed"

    mismatched_state = dict(failed_state)
    mismatched_state["output_schema_sha256"] = "0" * 64
    write_json(
        root / "native-v3-output/epoch-0000/communication-state.json.gz",
        mismatched_state,
    )
    with pytest.raises(ValueError, match="state is incompatible"):
        run_persistent_single_ast_cohort(root, **kwargs)
    assert adapter_calls == 1
    write_json(
        root / "native-v3-output/epoch-0000/communication-state.json.gz",
        failed_state,
    )

    stored_schema_path = root / "provider-turns/slot-00.initial.output-schema.json.gz"
    write_json(stored_schema_path, {"type": "object"})
    with pytest.raises(ValueError, match="stored provider output-schema"):
        run_persistent_single_ast_cohort(root, **kwargs)
    assert adapter_calls == 1
    write_json(
        stored_schema_path,
        build_candidate_request(
            candidate=SLOT_SPECIFIC_OUTPUT_CONTRACT,
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        ).output_schema,
    )

    report = run_persistent_single_ast_cohort(root, **kwargs)

    assert report["status"] == "evaluation_error"
    assert adapter_calls == 2
    assert len(processes) == 2
    replacement_methods = [request.get("method") for request in processes[1].received_requests]
    assert replacement_methods.count("thread/start") == 0
    assert replacement_methods.count("thread/fork") == 0
    assert replacement_methods.count("thread/resume") == 2
    assert replacement_methods.count("turn/start") == 3
    state = read_json(root / "native-v3-output/epoch-0000/communication-state.json.gz")
    assert state["provider_process_restarts"] == 1
    assert state["thread_resume_attempts"] == 2
    assert state["failed_thread_resume_attempts"] == 0
    assert state["provider_attempts"] == 6
    assert state["failed_provider_attempts"] == 1
