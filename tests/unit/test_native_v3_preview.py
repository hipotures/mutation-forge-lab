from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import read_json
from mutation_forge.native_v3.persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    protocol_hash,
)
from mutation_forge.native_v3.preview import (
    FORBIDDEN_LENGTHS,
    run_persistent_single_ast_cohort,
)
from mutation_forge.native_v3.single_program_contract import (
    validate_single_program_response,
)
from mutation_forge.stage3.app_server import AppServerLimits, CodexAppServerAdapter
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
    fixtures = json.loads(
        Path("tests/fixtures/native_v3_single_program_responses.json").read_text(
            encoding="utf-8"
        )
    )
    result: list[str] = []
    briefs = ("add-edge", "remove-edge", "relocation", "fanout")
    for index in range(8):
        response = copy.deepcopy(fixtures[briefs[index % len(briefs)]])
        entry = response["program"]["entry"]
        for _ in range(index // len(briefs)):
            entry = {
                "op": "try",
                "branches": [
                    entry,
                    {"op": "no_plan", "reason": "NO_MATCH"},
                ],
            }
        response["program"]["entry"] = entry
        encoded = json.dumps(response, separators=(",", ":"))
        validate_single_program_response(
            encoded,
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )
        result.append(encoded)
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
        if adapter_calls == 0:
            scenario = FakeScenario(
                final_text=json.dumps(
                    {
                        "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                        "protocol_hash": protocol_hash(FORBIDDEN_LENGTHS),
                    },
                    separators=(",", ":"),
                ),
                fork_thread_ids=["worker-thread-0", "worker-thread-1"],
            )
            max_campaigns = 3
        else:
            scenario = FakeScenario(final_text=responses[adapter_calls - 1])
            max_campaigns = 1
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
                max_turns=1,
                max_campaigns=max_campaigns,
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
    assert report["valid_slots"] == 8
    assert report["unique_valid_programs"] == 8
    assert report["model_turns"] == 9
    assert adapter_calls == 9
    state = read_json(
        root / "native-v3-output/epoch-0000/communication-state.json.gz"
    )
    assert state["status"] == "completed"
    assert state["next_slot"] == 8
    assert len(state["workers"]) == 2
    assert state["specification_thread"]["thread_id"] == "thread-1"
    assert state["specification_thread"]["anchor_turn_id"] == (
        state["anchor"]["turn_id"]
    )
    assert {
        worker["fork_parent_turn_id"] for worker in state["workers"]
    } == {state["anchor"]["turn_id"]}
    assert len(state["slot_reports"]) == 8
    assert len(processes) == 9
    for index, process in enumerate(processes[1:]):
        resume = next(
            request
            for request in process.received_requests
            if request.get("method") == "thread/resume"
        )
        worker = state["workers"][index % 2]
        assert resume["params"]["threadId"] == worker["thread_id"]
        assert resume["params"]["path"] == worker["thread_path"]

    turns_dir = root / "provider-turns"
    prefixes = (
        ["00-spec-anchor", "00-worker-00-fork", "00-worker-01-fork"]
        + [f"slot-{index:02d}.initial" for index in range(8)]
    )
    for prefix in prefixes:
        suffixes = {
            path.name.removeprefix(f"{prefix}.")
            for path in turns_dir.iterdir()
            if path.name.startswith(f"{prefix}.")
        }
        assert suffixes == _ARTIFACT_SUFFIXES
    assert not any(path.name.endswith("rollout.jsonl") for path in turns_dir.iterdir())
    for index in range(8):
        record = read_json(
            root
            / "native-v3-output/epoch-0000/program-records"
            / f"slot-{index:02d}.json.gz"
        )
        assert record["program"]["program_hash"]
        assert record["lineage"]["worker_index"] == index % 2
        prompt = (
            turns_dir / f"slot-{index:02d}.initial.request.md"
        ).read_text(encoding="utf-8")
        assert '"search_memory"' in prompt
        assert '"entry"' not in prompt
    assert not capsule.root.exists()
