from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge import cli, stage4r
from mutation_forge.stage4 import commands as stage4_commands
from mutation_forge.stage4.generation import GenerationCoordinator, ProviderResult

SOURCE = "def priority(ctx, proposal):\n    return 1.0\n"
CONFIG = Path("configs/stage4-search.toml")
RETAINED = Path("runs/stage4-search/campaign-63eea8c2ddb9")


def _envelope(source: str = SOURCE) -> dict[str, Any]:
    return {
        "schema_version": "stage4.generated_policy.v1",
        "source": source,
        "design_summary": "A bounded deterministic policy.",
        "change_summary": "Use a constant finite priority.",
        "hypothesis": "The contract path remains valid.",
        "used_fields": [],
        "assumptions": ["Inputs satisfy the policy contract."],
        "expected_failure_modes": ["The constant policy may rank poorly."],
    }


def _usage() -> dict[str, Any]:
    return {
        "inputTokens": 10,
        "cachedInputTokens": 2,
        "cacheWriteInputTokens": 0,
        "outputTokens": 4,
        "reasoningOutputTokens": 1,
        "totalTokens": 14,
        "final": True,
        "partial": False,
    }


class FakeProvider:
    def __init__(self, response: Any | None = None) -> None:
        self.response = _envelope() if response is None else response
        self.calls: list[dict[str, Any]] = []
        self.repairs: list[dict[str, Any]] = []

    def generate(self, request: dict[str, Any]) -> ProviderResult:
        self.calls.append(dict(request))
        return ProviderResult(
            response=self.response,
            usage=_usage(),
            request_id="request-1",
            thread_id="thread-1",
            turn_id="turn-1",
        )

    def repair(
        self,
        request: dict[str, Any],
        diagnostics: tuple[dict[str, Any], ...],
    ) -> ProviderResult:
        self.repairs.append({**request, "diagnostics_seen": list(diagnostics)})
        return ProviderResult(
            response=_envelope(),
            usage=_usage(),
            request_id="request-2",
            thread_id="thread-2",
            turn_id="turn-2",
        )


def _doctor() -> dict[str, Any]:
    return {
        "schema_version": "stage4.doctor.v1",
        "status": "completed",
        "decision": "READY",
        "auth": {"ok": True, "authenticated": True},
    }


def test_stage4r_parser_exposes_bounded_commands() -> None:
    parser = cli.build_parser()
    cases = (
        ["stage4r", "canary", "--auth-json", "/tmp/auth.json", "--attempt", "1"],
        ["stage4r", "freeze-search"],
        ["stage4r", "recover", "--auth-json", "/tmp/auth.json"],
        ["stage4r", "freeze-validation"],
        ["stage4r", "validate"],
        ["stage4r", "diagnose-deltas", "--run", "runs/stage4r-search/issue-11"],
    )
    for arguments in cases:
        parsed = parser.parse_args(arguments)
        assert parsed.command == "stage4r"


def test_one_request_uses_one_contract_repair_at_most() -> None:
    provider = FakeProvider(response={"not": "a policy"})
    coordinator = GenerationCoordinator(provider)
    result = coordinator.run_slot(
        0,
        "slot-00",
        "parent",
        allow_repair=True,
        allow_infrastructure_retry=False,
    )
    assert result.status == "accepted"
    assert result.repairs == 1
    assert result.candidate is not None
    assert len(provider.calls) == 1
    assert len(provider.repairs) == 1


def test_one_request_retries_only_proven_uncharged_infrastructure() -> None:
    class RetryProvider(FakeProvider):
        def generate(self, request: dict[str, Any]) -> ProviderResult:
            self.calls.append(dict(request))
            if len(self.calls) == 1:
                return ProviderResult(
                    status="infrastructure",
                    accepted=False,
                    charged=False,
                    content=False,
                    uncharged=True,
                    usage={},
                )
            return ProviderResult(
                response=_envelope(),
                usage=_usage(),
                request_id="request-2",
                thread_id="thread-2",
                turn_id="turn-2",
            )

    provider = RetryProvider()
    result = GenerationCoordinator(provider).run_slot(
        0,
        "slot-00",
        "parent",
        allow_repair=False,
        allow_infrastructure_retry=True,
    )
    assert result.candidate is not None
    assert len(provider.calls) == 2
    assert result.initial["error"] == "infrastructure_retry"


def test_canary_runs_one_turn_and_reindexes_diagnostic_artifact(tmp_path: Path) -> None:
    provider = FakeProvider()
    result = stage4r.canary(
        config_path=CONFIG,
        retained_run=RETAINED,
        run=tmp_path / "run",
        auth_json=tmp_path / "auth.json",
        attempt=1,
        provider=provider,
        doctor_result=_doctor(),
    )
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert len(provider.calls) == 1
    assert result["excluded_from_scientific_archive"] is True
    assert (tmp_path / "run" / "canary-success.json").is_file()


def test_canary_scopes_doctor_artifacts_to_recovery_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}

    def fake_doctor(
        config_path: str | Path,
        *,
        auth_json: str | Path | None,
        check_auth: bool,
        write: bool,
        run_override: str | Path | None,
    ) -> dict[str, Any]:
        observed.update(
            {
                "config_path": config_path,
                "auth_json": auth_json,
                "check_auth": check_auth,
                "write": write,
                "run_override": Path(cast(str | Path, run_override)).resolve(),
            }
        )
        return _doctor()

    monkeypatch.setattr(stage4_commands, "doctor", fake_doctor)
    run = tmp_path / "run"
    result = stage4r.canary(
        config_path=CONFIG,
        retained_run=RETAINED,
        run=run,
        auth_json=tmp_path / "auth.json",
        attempt=1,
        provider=FakeProvider(),
    )
    assert result["passed"] is True
    assert observed["run_override"] == run.resolve()
    assert observed["write"] is False


def test_search_freeze_follows_successful_canary_and_contains_eight_requests(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    stage4r.canary(
        config_path=CONFIG,
        retained_run=RETAINED,
        run=run,
        auth_json=tmp_path / "auth.json",
        attempt=1,
        provider=FakeProvider(),
        doctor_result=_doctor(),
    )
    tracked = tmp_path / "stage4r-search-frozen-v1.json"
    result = stage4r.freeze_search(
        config_path=CONFIG,
        retained_run=RETAINED,
        run=run,
        tracked_freeze=tracked,
    )
    assert result["status"] == "completed"
    assert len(result["requests"]) == 8
    artifact = json.loads((run / "search-freeze.json").read_text(encoding="utf-8"))
    assert len(artifact["private_requests"]) == 8
    assert [request["slot"] for request in artifact["private_requests"]] == [
        f"slot-{index:02d}" for index in range(8)
    ]
    assert json.loads(tracked.read_text(encoding="utf-8"))["results_observed"] is False


def test_scientific_gate_returns_go_only_when_all_issue_11_checks_pass() -> None:
    evidence = {
        "pooled_relative_improvement": 0.02,
        "pooled_bootstrap_lower_bound": 0.001,
        "order_deltas": {"10": 0.0, "12": 0.0},
        "graph_seed_nonnegative_counts": {"10": 3, "12": 3},
        "structural_retention": 0.99,
    }
    replay = {
        "exact": True,
        "canonical_reduction_match": True,
        "metrics_input_match": True,
    }
    summary = {"policies": {}}
    health = {
        "invalid_graphs": 0,
        "worker_failures": 0,
        "selected_plan_only": True,
        "oracle_score_calls": 0,
        "provider_calls": 0,
        "equal_budgets": True,
    }
    gate = stage4r._scientific_gate(
        champion_distinct=True,
        evidence=evidence,
        replay=replay,
        primary_summary=summary,
        replay_summary=summary,
        primary_evidence=evidence,
        replay_evidence=evidence,
        health=health,
        replay_health=health,
    )
    assert gate["decision"] == "GO_TO_STAGE_5"
    failed = stage4r._scientific_gate(
        champion_distinct=True,
        evidence={**evidence, "pooled_relative_improvement": 0.019},
        replay=replay,
        primary_summary=summary,
        replay_summary=summary,
        primary_evidence=evidence,
        replay_evidence=evidence,
        health=health,
        replay_health=health,
    )
    assert failed["decision"] == "NO_GO"


def test_stage4r_delta_diagnostic_reproduces_frozen_pair_and_bootstrap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostics"
    first = stage4r.diagnose_deltas(
        run=Path("runs/stage4r-search/issue-11"),
        output_dir=output,
    )
    second = stage4r.diagnose_deltas(
        run=Path("runs/stage4r-search/issue-11"),
        output_dir=output,
    )
    assert first == second
    assert first["status"] == "completed"
    assert first["historical_decision"] == "NO_GO"
    assert first["primary_replay"]["paired_rows"] == 128
    assert first["bootstrap"]["pooled"]["interval"] == [0.0, 0.03125]
    assert first["quantization"]["delta"]["sign_mass"] == {
        "negative": 32,
        "zero": 30,
        "positive": 66,
        "negative_fraction": 0.25,
        "zero_fraction": 0.234375,
        "positive_fraction": 0.515625,
        "count": 128,
    }
    assert first["quantization"]["mechanism_for_0_03125"]["one_over_32_exemplar"][
        "curve_difference_area_fraction"
    ] == "1"
    assert len(first["power_study"]["summary"]["cells"]) == 54
    assert first["recommendation"]["token"] == "REDESIGN_PRIMARY_METRIC_BEFORE_CONFIRMATION"
    assert (output / "paired-deltas.csv").is_file()
    assert (output / "paired-deltas.jsonl").is_file()
    assert (output / "bootstrap-support.json").is_file()
