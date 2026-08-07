from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.native_v3.compaction_experiment import (
    ACK_SCHEMA_VERSION,
    build_reference_manifest,
    compare_manifest,
    retention_manifest_projection,
    run_compaction_experiment,
)
from mutation_forge.native_v3.persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    BOOTSTRAP_ACK_VALUE,
)
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    ModelProfile,
    TurnError,
)

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location("step12c_fake_app_server", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario

FORBIDDEN_LENGTHS = (4, 8, 16)
PROFILE = ModelProfile("codex", "gpt-5.6-luna", "high")


def _candidate_responses() -> dict[str, dict[str, Any]]:
    value = json.loads(
        Path("tests/fixtures/native_v3_single_program_responses.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def _adapter(
    tmp_path: Path,
    scenario: Any,
    *,
    process_holder: list[Any] | None = None,
) -> CodexAppServerAdapter:
    def factory(*_args: Any, **kwargs: Any) -> Any:
        process = FakeProcess(scenario, **kwargs)
        if process_holder is not None:
            process_holder.append(process)
        return process

    return CodexAppServerAdapter(
        process_factory=factory,
        auth_checker=lambda _capsule: True,
        limits=AppServerLimits(
            max_turns=9,
            max_campaigns=1,
            turn_timeout=0.05,
            usage_grace=0.01,
        ),
        base_instructions="Return only the requested structured fixture response.",
        artifact_dir=tmp_path,
        artifact_prefix="initial",
        compress_json_artifacts=True,
        sandbox_mode="read-only",
    )


def _fake_final_texts(
    arm: str,
    reference: dict[str, Any],
    candidate_responses: dict[str, dict[str, Any]],
) -> list[str]:
    checkpoint = (
        "retention-directive" if arm == "directive" else "control-checkpoint"
    )
    return [
        json.dumps(
            {
                "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                "ack": BOOTSTRAP_ACK_VALUE,
            },
            separators=(",", ":"),
        ),
        *[
            json.dumps(
                {
                    "schema_version": ACK_SCHEMA_VERSION,
                    "ack": ack,
                },
                separators=(",", ":"),
            )
            for ack in (
                "fixture-ast-00",
                "fixture-ast-01",
                "evaluation-00",
                "evaluation-01",
                checkpoint,
            )
        ],
        json.dumps(
            retention_manifest_projection(reference),
            separators=(",", ":"),
        ),
        json.dumps(candidate_responses["fanout"], separators=(",", ":")),
    ]


def test_compaction_lifecycle_uses_exact_request_and_correlated_item(
    tmp_path: Path,
) -> None:
    processes: list[Any] = []
    adapter = _adapter(
        tmp_path,
        FakeScenario(
            compaction_usage_before_completion=True,
            compaction_deprecated_notification=True,
        ),
        process_holder=processes,
    )
    try:
        adapter.start_thread(PROFILE, ephemeral=False)
        result = adapter.compact_persistent_thread()
    finally:
        adapter.close()

    assert result.thread_id == "thread-1"
    assert result.turn_id == "compact-1"
    assert result.item_id == "context-compaction-1"
    assert result.usage is not None and result.usage.total_tokens == 5
    requests = [
        request
        for request in processes[0].received_requests
        if request.get("method") == "thread/compact/start"
    ]
    assert requests == [
        {
            "id": result.request_id,
            "method": "thread/compact/start",
            "params": {"threadId": "thread-1"},
        }
    ]


def test_compaction_timeout_is_terminal_and_auditable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, FakeScenario(compaction_hang=True))
    adapter.start_thread(PROFILE, ephemeral=False)

    with pytest.raises(TurnError, match="compaction timed out"):
        adapter.compact_persistent_thread()

    assert adapter.inspect_metadata()["status"] == "failed"


def test_compaction_failed_item_is_not_success(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        FakeScenario(
            compaction_terminal_status="failed",
            compaction_omit_item_completion=True,
        ),
    )
    adapter.start_thread(PROFILE, ephemeral=False)

    with pytest.raises(TurnError, match="compaction ended with status 'failed'"):
        adapter.compact_persistent_thread()


def test_compaction_missing_completion_fails_on_eof(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        FakeScenario(
            compaction_omit_turn_completion=True,
            compaction_close_after_events=True,
        ),
    )
    adapter.start_thread(PROFILE, ephemeral=False)

    with pytest.raises(
        TurnError,
        match="app-server EOF before compaction completion",
    ):
        adapter.compact_persistent_thread()


def test_manifest_comparison_detects_hallucinated_identity_score_and_relation() -> None:
    reference = build_reference_manifest(
        _candidate_responses(),
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    expected = retention_manifest_projection(reference)
    actual = json.loads(json.dumps(expected))
    actual["candidates"][0]["candidate_id"] = "hallucinated-candidate"
    actual["candidates"][1]["score_micros"] = 999999
    actual["candidates"][2]["parent_id"] = "unknown-parent"

    result = compare_manifest(actual, expected)

    assert result["exact"] is False
    assert "candidate_id:hallucinated-candidate" in result["hallucinated"]
    assert any(
        value.startswith("score_micros:g0001-s02:")
        for value in result["hallucinated"]
    )
    assert any(
        value.startswith("parent_id:g0002-s03:")
        for value in result["hallucinated"]
    )


def test_six_repetition_fake_experiment_reports_exact_retention_and_artifacts(
    tmp_path: Path,
) -> None:
    candidate_responses = _candidate_responses()
    reference = build_reference_manifest(
        candidate_responses,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    def factory(
        base_instructions: str,
        arm: str,
        repetition: int,
    ) -> CodexAppServerAdapter:
        return CodexAppServerAdapter(
            process_factory=lambda *_args, **kwargs: FakeProcess(
                FakeScenario(
                    final_texts=_fake_final_texts(
                        arm,
                        reference,
                        candidate_responses,
                    ),
                    thread_id=f"{arm}-thread-{repetition}",
                ),
                **kwargs,
            ),
            auth_checker=lambda _capsule: True,
            limits=AppServerLimits(
                max_turns=9,
                max_campaigns=1,
                turn_timeout=0.1,
                usage_grace=0.01,
            ),
            base_instructions=base_instructions,
            artifact_dir=tmp_path,
            artifact_prefix=f"{arm}-{repetition}",
            compress_json_artifacts=True,
            sandbox_mode="read-only",
        )

    workspace = tmp_path / "experiment"
    report = run_compaction_experiment(
        workspace,
        model="gpt-5.6-luna",
        effort="high",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        candidate_responses=candidate_responses,
        adapter_factory=factory,
    )

    assert report["classification"] == "RELIABLE_FOR_OPTIMIZATION"
    assert report["retention_directive_is_protocol_guarantee"] is False
    assert report["production_compaction_enabled"] is False
    assert len(report["repetitions"]) == 6
    assert all(
        repetition["active_parent_exact_hash_match"] is True
        and repetition["manifest_comparison"]["exact"] is True
        for repetition in report["repetitions"]
    )
    assert (workspace / "compaction-report.json.gz").is_file()
    markdown = (workspace / "compaction-report.md").read_text(encoding="utf-8")
    assert "not a protocol-level guarantee" in markdown

    expected_suffixes = {
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
    for arm in ("directive", "control"):
        turns_dir = workspace / arm / "rep-00" / "provider-turns"
        for prefix in (
            "00-bootstrap",
            "01-fixture-ast-00",
            "02-fixture-ast-01",
            "03-evaluation-00",
            "04-evaluation-01",
            "05-checkpoint",
            "06-compaction",
            "07-manifest-probe",
            "08-parent-probe",
        ):
            suffixes = {
                path.name.removeprefix(f"{prefix}.")
                for path in turns_dir.iterdir()
                if path.name.startswith(f"{prefix}.")
            }
            assert suffixes == expected_suffixes
    directive = (
        workspace
        / "directive/rep-00/provider-turns/05-checkpoint.request.md"
    ).read_text(encoding="utf-8")
    control = (
        workspace / "control/rep-00/provider-turns/05-checkpoint.request.md"
    ).read_text(encoding="utf-8")
    assert "[CONTEXT COMPACTION RETENTION DIRECTIVE]" in directive
    assert "[CONTEXT COMPACTION RETENTION DIRECTIVE]" not in control


def test_parent_probe_failure_preserves_completed_compaction_evidence(
    tmp_path: Path,
) -> None:
    candidate_responses = _candidate_responses()
    reference = build_reference_manifest(
        candidate_responses,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    def factory(
        base_instructions: str,
        arm: str,
        repetition: int,
    ) -> CodexAppServerAdapter:
        terminal_statuses = (
            ["completed"] * 7 + ["failed"]
            if arm == "control" and repetition == 0
            else None
        )
        return CodexAppServerAdapter(
            process_factory=lambda *_args, **kwargs: FakeProcess(
                FakeScenario(
                    final_texts=_fake_final_texts(
                        arm,
                        reference,
                        candidate_responses,
                    ),
                    terminal_statuses=terminal_statuses,
                    thread_id=f"{arm}-thread-{repetition}",
                ),
                **kwargs,
            ),
            auth_checker=lambda _capsule: True,
            limits=AppServerLimits(
                max_turns=9,
                max_campaigns=1,
                turn_timeout=0.1,
                usage_grace=0.01,
            ),
            base_instructions=base_instructions,
            compress_json_artifacts=True,
            sandbox_mode="read-only",
        )

    report = run_compaction_experiment(
        tmp_path / "experiment-late-failure",
        model="gpt-5.6-luna",
        effort="high",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        candidate_responses=candidate_responses,
        adapter_factory=factory,
    )
    failed = next(
        item
        for item in report["repetitions"]
        if item["arm"] == "control" and item["repetition"] == 0
    )

    assert failed["failure_stage"] == "parent-probe"
    assert failed["compaction_status"] == "completed"
    assert isinstance(failed["compaction_turn_id"], str)
    assert isinstance(failed["compaction_item_id"], str)
    assert failed["manifest_comparison"]["exact"] is True
    assert "manifest_probe" in failed["usage_after_compaction"]
