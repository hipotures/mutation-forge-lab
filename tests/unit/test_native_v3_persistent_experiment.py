from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mutation_forge.native_v3.persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    BRIEF_IDS,
    bootstrap_schema,
    protocol_hash,
    run_ab_experiment,
)
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    IsolationError,
    ModelProfile,
    ProtocolError,
    TurnError,
)
from mutation_forge.stage3.isolation import IsolatedCapsule

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location("step12b_fake_app_server", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario

FORBIDDEN_LENGTHS = (4, 8, 16)


def _responses() -> dict[str, dict[str, object]]:
    value = json.loads(
        Path("tests/fixtures/native_v3_single_program_responses.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def _adapter(
    tmp_path: Path,
    scenario: FakeScenario,
    base_instructions: str,
    *,
    capsule: IsolatedCapsule | None = None,
) -> CodexAppServerAdapter:
    return CodexAppServerAdapter(
        capsule=capsule,
        process_factory=lambda *_args, **kwargs: FakeProcess(scenario, **kwargs),
        auth_checker=lambda _capsule: True,
        limits=AppServerLimits(max_turns=5, max_campaigns=1),
        base_instructions=base_instructions,
        artifact_dir=tmp_path,
        artifact_prefix="initial",
        compress_json_artifacts=True,
        sandbox_mode="read-only",
    )


def test_fake_multi_turn_fresh_and_persistent_comparison(tmp_path: Path) -> None:
    fixtures = _responses()
    identity = protocol_hash(FORBIDDEN_LENGTHS)

    def factory(base_instructions: str, prefix: str) -> CodexAppServerAdapter:
        if prefix == "b-bootstrap":
            texts = [
                json.dumps(
                    {
                        "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                        "protocol_hash": identity,
                    },
                    separators=(",", ":"),
                ),
                *[
                    json.dumps(fixtures[brief_id], separators=(",", ":"))
                    for brief_id in BRIEF_IDS
                ],
            ]
        else:
            brief_id = BRIEF_IDS[int(prefix.rsplit("-", 1)[1])]
            texts = [json.dumps(fixtures[brief_id], separators=(",", ":"))]
        return _adapter(
            tmp_path,
            FakeScenario(final_texts=texts),
            base_instructions,
        )

    report = run_ab_experiment(
        tmp_path / "experiment",
        model="gpt-5.6-luna",
        effort="high",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
    )

    a_turns = report["A_fresh_threads"]["turns"]
    b_turns = report["B_persistent_thread"]["turns"]
    assert len(a_turns) == len(b_turns) == 4
    assert report["A_fresh_threads"]["valid_program_rate"] == 1
    assert report["B_persistent_thread"]["valid_program_rate"] == 1
    assert len({turn["thread_id"] for turn in b_turns}) == 1
    assert len({turn["turn_id"] for turn in b_turns}) == 4
    assert report["B_persistent_thread"]["time_to_first_valid_ast_ms"] is not None
    assert report["A_fresh_threads"]["usage"]["totalTokens"] == 20
    assert report["B_persistent_thread"]["usage"]["totalTokens"] == 25
    assert (tmp_path / "experiment" / "ab-report.json.gz").is_file()


def test_bootstrap_schema_uses_provider_accepted_literal_types() -> None:
    identity = protocol_hash(FORBIDDEN_LENGTHS)
    schema = bootstrap_schema(identity)

    assert schema["properties"]["schema_version"] == {
        "type": "string",
        "const": BOOTSTRAP_ACK_SCHEMA_VERSION,
    }
    assert schema["properties"]["protocol_hash"] == {
        "type": "string",
        "const": identity,
    }


def test_one_invalid_program_turn_preserves_other_persistent_results(
    tmp_path: Path,
) -> None:
    fixtures = _responses()
    identity = protocol_hash(FORBIDDEN_LENGTHS)

    def factory(base_instructions: str, prefix: str) -> CodexAppServerAdapter:
        if prefix == "b-bootstrap":
            texts = [
                json.dumps(
                    {
                        "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                        "protocol_hash": identity,
                    },
                    separators=(",", ":"),
                ),
                json.dumps(fixtures["add-edge"], separators=(",", ":")),
                '{"program":{},"design_summary":"Invalid.","hypothesis":"Invalid."}',
                json.dumps(fixtures["relocation"], separators=(",", ":")),
                json.dumps(fixtures["fanout"], separators=(",", ":")),
            ]
        else:
            brief_id = BRIEF_IDS[int(prefix.rsplit("-", 1)[1])]
            texts = [json.dumps(fixtures[brief_id], separators=(",", ":"))]
        return _adapter(tmp_path, FakeScenario(final_texts=texts), base_instructions)

    report = run_ab_experiment(
        tmp_path / "experiment",
        model="gpt-5.6-luna",
        effort="high",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
    )
    turns = report["B_persistent_thread"]["turns"]
    assert [turn["program_hash"] is not None for turn in turns] == [
        True,
        False,
        True,
        True,
    ]
    assert turns[0]["program_hash"] is not None
    assert turns[2]["program_hash"] is not None


def test_terminal_fourth_persistent_turn_preserves_three_programs_and_report(
    tmp_path: Path,
) -> None:
    fixtures = _responses()
    identity = protocol_hash(FORBIDDEN_LENGTHS)

    def factory(base_instructions: str, prefix: str) -> CodexAppServerAdapter:
        if prefix == "b-bootstrap":
            scenario = FakeScenario(
                final_texts=[
                    json.dumps(
                        {
                            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                            "protocol_hash": identity,
                        },
                        separators=(",", ":"),
                    ),
                    *[
                        json.dumps(fixtures[brief], separators=(",", ":"))
                        for brief in BRIEF_IDS
                    ],
                ],
                terminal_statuses=[
                    "completed",
                    "completed",
                    "completed",
                    "completed",
                    "systemError",
                ],
            )
        else:
            brief = BRIEF_IDS[int(prefix.rsplit("-", 1)[1])]
            scenario = FakeScenario(
                final_text=json.dumps(fixtures[brief], separators=(",", ":"))
            )
        return _adapter(tmp_path, scenario, base_instructions)

    report = run_ab_experiment(
        tmp_path / "experiment",
        model="gpt-5.6-luna",
        effort="medium",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
    )

    turns = report["B_persistent_thread"]["turns"]
    assert len(turns) == 4
    assert [turn["terminal_status"] for turn in turns] == [
        "completed",
        "completed",
        "completed",
        "failed",
    ]
    assert report["B_persistent_thread"]["valid_program_rate"] == 0.75
    assert (tmp_path / "experiment" / "ab-report.json.gz").is_file()


def test_fresh_thread_uses_the_production_infrastructure_retry_limit(
    tmp_path: Path,
) -> None:
    fixtures = _responses()
    identity = protocol_hash(FORBIDDEN_LENGTHS)

    def factory(base_instructions: str, prefix: str) -> CodexAppServerAdapter:
        if prefix == "a-slot-00":
            scenario = FakeScenario(error_will_retry=True)
        elif prefix == "b-bootstrap":
            scenario = FakeScenario(
                final_texts=[
                    json.dumps(
                        {
                            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                            "protocol_hash": identity,
                        },
                        separators=(",", ":"),
                    ),
                    *[
                        json.dumps(fixtures[brief], separators=(",", ":"))
                        for brief in BRIEF_IDS
                    ],
                ]
            )
        else:
            slot_text = prefix.split(".retry-", 1)[0]
            brief = BRIEF_IDS[int(slot_text.rsplit("-", 1)[1])]
            scenario = FakeScenario(
                final_text=json.dumps(fixtures[brief], separators=(",", ":"))
            )
        return _adapter(tmp_path, scenario, base_instructions)

    report = run_ab_experiment(
        tmp_path / "experiment",
        model="gpt-5.6-luna",
        effort="high",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
    )

    fresh = report["A_fresh_threads"]
    assert fresh["infrastructure_attempts"] == 5
    assert fresh["failed_attempt_prefixes"] == ["a-slot-00"]
    assert fresh["turns"][0]["prefix"] == "a-slot-00.retry-01"
    failed_names = {
        path.name.removeprefix("a-slot-00.")
        for path in (tmp_path / "experiment" / "provider-turns").iterdir()
        if path.name.startswith("a-slot-00.")
        and not path.name.startswith("a-slot-00.retry-")
    }
    assert failed_names == {
        "codex-profile.json.gz",
        "codex-rpc.jsonl",
        "events.jsonl",
        "output-schema.json.gz",
        "provider-raw.json.gz",
        "request.json.gz",
        "request.md",
        "stderr.txt",
        "stdout.jsonl",
        "system-prompt.md",
        "transcript.sha256",
        "wire.jsonl",
    }


def test_durable_thread_resumes_after_process_restart(tmp_path: Path) -> None:
    capsule = IsolatedCapsule.create(tmp_path)
    profile = ModelProfile("codex", "gpt-5.6-luna", "high")
    first = _adapter(
        tmp_path,
        FakeScenario(final_text="first"),
        "Persistent test.",
        capsule=capsule,
    )
    try:
        thread = first.start_thread(profile, ephemeral=False)
        first_result = first.generate_persistent("first", profile)
        first.close()

        second = _adapter(
            tmp_path,
            FakeScenario(
                final_text="second",
                resume_status_before_response="idle",
                resume_usage_after_response=True,
                resume_usage_turn_id=first_result.turn_id,
            ),
            "Persistent test.",
            capsule=capsule,
        )
        second.resume_thread(
            profile,
            thread_id=str(thread["id"]),
            thread_path=str(thread["path"]),
        )
        second_result = second.generate_persistent("second", profile)
        second.close()

        assert first_result.thread_id == second_result.thread_id == "thread-1"
        assert second_result.text == "second"
    finally:
        capsule.cleanup()


def test_resume_rejects_foreign_and_terminal_status_notifications(
    tmp_path: Path,
) -> None:
    profile = ModelProfile("codex", "gpt-5.6-luna", "high")
    cases = (
        (
            FakeScenario(
                resume_status_before_response="idle",
                resume_status_thread_id="foreign-thread",
            ),
            ProtocolError,
            "foreign thread/resume",
        ),
        (
            FakeScenario(resume_status_before_response="systemError"),
            TurnError,
            "terminal thread/resume status",
        ),
    )
    for index, (scenario, error_type, message) in enumerate(cases):
        parent = tmp_path / f"case-{index}"
        parent.mkdir()
        capsule = IsolatedCapsule.create(parent)
        adapter = _adapter(
            tmp_path,
            scenario,
            "Persistent test.",
            capsule=capsule,
        )
        try:
            with pytest.raises(error_type, match=message):
                adapter.resume_thread(
                    profile,
                    thread_id="thread-1",
                    thread_path=str(capsule.codex_home / "rollout.jsonl"),
                )
        finally:
            adapter.close()
            capsule.cleanup()


def test_cli_0146_dangling_reasoning_is_tolerated_only_experimentally(
    tmp_path: Path,
) -> None:
    profile = ModelProfile("codex", "gpt-5.6-luna", "high")
    strict = _adapter(
        tmp_path,
        FakeScenario(dangling_reasoning=True),
        "Strict test.",
    )
    with pytest.raises(ProtocolError, match="active items"):
        strict.generate("strict", profile)

    experimental = _adapter(
        tmp_path,
        FakeScenario(dangling_reasoning=True),
        "Experimental test.",
    )
    result = experimental.generate_ephemeral_experiment("experimental", profile)
    experimental.close()
    assert result.text == "fixture answer"


def test_server_retry_is_internal_only_for_the_persistent_experiment(
    tmp_path: Path,
) -> None:
    profile = ModelProfile("codex", "gpt-5.6-luna", "high")
    strict = _adapter(
        tmp_path,
        FakeScenario(error_will_retry=True),
        "Strict retry test.",
    )
    with pytest.raises(IsolationError, match="retry is forbidden"):
        strict.generate("strict", profile)

    experimental = _adapter(
        tmp_path,
        FakeScenario(error_will_retry=True),
        "Experimental retry test.",
    )
    with pytest.raises(IsolationError, match="retry is forbidden"):
        experimental.generate_ephemeral_experiment("experimental", profile)

    persistent = _adapter(
        tmp_path,
        FakeScenario(
            error_will_retry=True,
            warning_message="Falling back to another transport.",
        ),
        "Persistent retry test.",
    )
    result = persistent.generate_persistent("persistent", profile)
    persistent.close()
    assert result.text == "fixture answer"


def test_every_experimental_turn_prefix_has_the_same_artifact_names(
    tmp_path: Path,
) -> None:
    fixtures = _responses()
    identity = protocol_hash(FORBIDDEN_LENGTHS)

    def factory(base_instructions: str, prefix: str) -> CodexAppServerAdapter:
        if prefix == "b-bootstrap":
            texts = [
                json.dumps(
                    {
                        "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                        "protocol_hash": identity,
                    },
                    separators=(",", ":"),
                ),
                *[
                    json.dumps(fixtures[brief], separators=(",", ":"))
                    for brief in BRIEF_IDS
                ],
            ]
        else:
            brief = BRIEF_IDS[int(prefix.rsplit("-", 1)[1])]
            texts = [json.dumps(fixtures[brief], separators=(",", ":"))]
        return _adapter(
            tmp_path,
            FakeScenario(
                final_texts=texts,
                thread_id=(
                    "persistent-thread"
                    if prefix == "b-bootstrap"
                    else f"thread-{prefix}"
                ),
            ),
            base_instructions,
        )

    run_ab_experiment(
        tmp_path / "experiment",
        model="gpt-5.6-luna",
        effort="high",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
    )
    directory = tmp_path / "experiment" / "provider-turns"
    prefixes = (
        [f"a-slot-{index:02d}" for index in range(4)]
        + ["b-bootstrap"]
        + [f"b-slot-{index:02d}" for index in range(4)]
    )
    name_sets = []
    for prefix in prefixes:
        name_sets.append(
            {
                path.name.removeprefix(f"{prefix}.")
                for path in directory.iterdir()
                if path.name.startswith(f"{prefix}.")
            }
        )
    assert all(names == name_sets[0] for names in name_sets)
    assert name_sets[0] == {
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
