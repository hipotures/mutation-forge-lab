from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge import cli
from mutation_forge.experiment.json_io import compress_json_bytes, read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3_python import preview as preview_module
from mutation_forge.native_v3_python import scientific_search as search_module
from mutation_forge.native_v3_python.contracts import (
    PYTHON_EXPERIMENT_PROTOCOL_ID,
)
from mutation_forge.native_v3_python.preview import (
    PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION,
    PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION,
    PythonPreviewWorkspaceError,
    experiment_protocol,
    load_python_preview_config,
    python_preview_bootstrap_status,
    python_preview_status,
    request_python_preview_stop,
    run_python_preview,
)
from mutation_forge.native_v3_python.scientific_search import M10_REPORT_PROTOCOL_ID
from mutation_forge.native_v3_python.search import (
    DevelopmentCaseV1,
    M5OperatorStop,
    M5ProviderContextV1,
    M5ProviderResultV1,
    _evaluation_telemetry_summary,
)
from mutation_forge.native_v3_python.search_provider import CodexM5SearchProvider


def _config(tmp_path: Path, *, exp_id: str = "python-preview") -> Path:
    heg = tmp_path / "heg"
    (heg / "src" / "sglab").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "python-preview.toml"
    path.write_text(
        f'''schema_version = "{PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION}"
protocol = "{PYTHON_EXPERIMENT_PROTOCOL_ID}"
exp_id = "{exp_id}"
workspace = "{(tmp_path / "workspaces").as_posix()}"

[python_preview]
model = "fixture-model"
effort = "high"
timeout_seconds = 30
heg_repo = "{heg.as_posix()}"
''',
        encoding="utf-8",
    )
    return path


def _interval(lower: int, upper: int) -> dict[str, dict[str, int]]:
    return {
        "lower": {"numerator": lower, "denominator": 10},
        "upper": {"numerator": upper, "denominator": 10},
    }


@pytest.mark.parametrize(
    ("child", "parent", "expected"),
    [
        (_interval(6, 7), _interval(3, 5), "proven_better"),
        (_interval(1, 2), _interval(3, 5), "proven_worse"),
        (_interval(4, 6), _interval(3, 5), "neutral"),
    ],
)
def test_same_panel_score_effect_uses_conservative_intervals(
    child: Mapping[str, Any],
    parent: Mapping[str, Any],
    expected: str,
) -> None:
    assert (
        preview_module._same_panel_score_effect(
            candidate={
                "panel_hash": "same-panel",
                "behavior_profile": {"fitness_interval": child},
            },
            reference={
                "panel_hash": "same-panel",
                "profile": {"fitness_interval": parent},
            },
            panel_hash="same-panel",
        )
        == expected
    )


def test_same_panel_score_effect_never_compares_cross_panel_scores() -> None:
    assert (
        preview_module._same_panel_score_effect(
            candidate={
                "panel_hash": "child-panel",
                "behavior_profile": {"fitness_interval": _interval(9, 10)},
            },
            reference={
                "panel_hash": "parent-panel",
                "profile": {"fitness_interval": _interval(0, 1)},
            },
            panel_hash="child-panel",
        )
        == "neutral"
    )


def _scientific_status_config(tmp_path: Path, *, exp_id: str) -> Path:
    heg = tmp_path / "heg"
    (heg / "src" / "sglab").mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{exp_id}.toml"
    path.write_text(
        f'''schema_version = "{PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION}"
protocol = "{PYTHON_EXPERIMENT_PROTOCOL_ID}"
exp_id = "{exp_id}"
workspace = "{(tmp_path / "workspaces").as_posix()}"

[python_preview]
model = "fixture-model"
effort = "medium"
timeout_seconds = 30
heg_repo = "{heg.as_posix()}"

[python_preview.scientific_search]
generation_limit = 2
evaluator_workers = 2
provider_concurrency = 2
wall_seconds = 60
primary_program_slots = 16
repair_turn_limit = 0
provider_total_turn_limit = 16
stop_on_verified = true
resume_enabled = true
replace_terminal_slots = false

[python_preview.scientific_search.evaluation]
graph_mode = "unrestricted_min_degree_3"
order_schedule = "adaptive"
min_order = 8
max_order = 8
orders_per_generation = 1
graph_seeds = [101]
policy_seeds = [17, 19, 23, 29]
horizon = 1
witness_cap = 64
baselines = ["random", "structural"]
replay = false
''',
        encoding="utf-8",
    )
    return path


def test_validated_queue_limits_follow_evaluator_workers(tmp_path: Path) -> None:
    loaded = load_python_preview_config(
        _scientific_status_config(tmp_path, exp_id="derived-queue-limits")
    )

    assert loaded.scientific_search is not None
    assert loaded.scientific_search.validated_queue_target == 4
    assert loaded.scientific_search.validated_queue_capacity == 8


def _source(label: str) -> str:
    return (
        "def propose(ctx, graph, api, seed):\n"
        "    candidates = api.non_edges_legal()\n"
        f'    candidate = api.pick(candidates, seed, "{label}")\n'
        "    if not candidate:\n"
        '        return api.no_plan("NO_MATCH")\n'
        "    api.add_edge(candidate)\n"
        "    return api.emit()\n"
    )


def _envelope(source: str) -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.python_policy_response.v1",
            "source": source,
        },
        separators=(",", ":"),
    )


def _usage() -> dict[str, JsonValue]:
    return {
        "inputTokens": 10,
        "cachedInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "outputTokens": 10,
        "reasoningOutputTokens": 0,
        "totalTokens": 20,
        "final": True,
        "partial": False,
    }


def test_backend_owned_evaluator_uses_numbered_score_worker_names() -> None:
    backend = _Backend()
    owned = preview_module._BackendOwnedEvaluator(
        evaluator=_Evaluator(),
        backend=backend,
    )

    owned.set_worker_name("mforge-eval-03")

    assert backend.score_worker_name == "mforge-eval-03"


class _Backend:
    backend_id = "fixture"
    score_implementation = "fixture"

    def __init__(self) -> None:
        self.closed = False
        self.score_worker_name: str | None = None

    def set_score_worker_name(self, name: str) -> None:
        self.score_worker_name = name

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        assert order == 30
        return (4, 8)

    def close(self) -> None:
        self.closed = True


class _Evaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        self.calls.append((candidate_id, case.case_id))
        digest = hashlib.sha256(f"{source}:{case.case_id}".encode()).hexdigest()
        return {
            "behavior_identity": {
                "protocol_id": "fixture",
                "probe_manifest_sha256": "a" * 64,
                "behavior_signature": digest,
            },
            "worker_telemetry": {
                "calls": 1,
                "failures": 0,
                "rotations": 0,
                "worker_rss_kib": 18_000,
            },
            "scientific_result": {
                "status": "COMPLETE",
                "score_attempts": 2,
                "unique_graph_scores": 2,
                "fitness_interval": {
                    "lower": {"numerator": 1, "denominator": 2},
                    "upper": {"numerator": 1, "denominator": 2},
                },
                "semantic_trace_hash": digest,
                "initial_counterexample": None,
                "initial_evidence": {
                    "components": [
                        {
                            "forbidden_length": 4,
                            "lower_bound": 2,
                            "upper_bound": 2,
                        }
                    ]
                },
                "terminal_evidence": {
                    "components": [
                        {
                            "forbidden_length": 4,
                            "lower_bound": 1,
                            "upper_bound": 1,
                        }
                    ]
                },
                "steps": [
                    {
                        "outcome": "rewrite",
                        "accepted": True,
                        "no_plan_reason": None,
                        "counterexample": None,
                        "interpreter_trace": [
                            {"method": "non_edges_legal"},
                            {"method": "pick"},
                            {"method": "add_edge"},
                            {"method": "emit"},
                        ],
                    }
                ],
            },
            "external_activity": {
                "provider_turns": 0,
                "model_turns": 0,
                "app_server_calls": 0,
            },
        }


class _ProcessFixtureEvaluator:
    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        del source
        return {
            "kind": "candidate",
            "candidate_id": candidate_id,
            "case_id": case.case_id,
            "process_id": os.getpid(),
        }

    def evaluate_baseline(
        self,
        *,
        baseline: str,
        case: DevelopmentCaseV1,
        generation: int,
    ) -> Mapping[str, JsonValue]:
        return {
            "kind": "baseline",
            "baseline": baseline,
            "case_id": case.case_id,
            "generation": generation,
            "process_id": os.getpid(),
        }


def _process_backend_factory(_: preview_module.PythonPreviewConfig) -> _Backend:
    return _Backend()


def _process_evaluator_factory(
    _: preview_module.PythonPreviewConfig,
    __: Any,
) -> _ProcessFixtureEvaluator:
    return _ProcessFixtureEvaluator()


def test_candidate_and_baseline_evaluation_run_in_owned_child_process(
    tmp_path: Path,
) -> None:
    config = load_python_preview_config(_config(tmp_path, exp_id="process-owned"))
    first = preview_module._ProcessOwnedEvaluator(
        config,
        backend_factory=_process_backend_factory,
        evaluator_factory=_process_evaluator_factory,
    )
    second = preview_module._ProcessOwnedEvaluator(
        config,
        backend_factory=_process_backend_factory,
        evaluator_factory=_process_evaluator_factory,
    )
    case = DevelopmentCaseV1("case-00", 30, 101, 17, 1, 64, (4, 8))
    try:
        first_candidate_name = search_module._evaluation_process_name(
            0, "candidate:g0000-slot-01"
        )
        second_candidate_name = search_module._evaluation_process_name(
            1, "candidate:g0000-slot-01"
        )
        first.set_worker_name(first_candidate_name)
        second.set_worker_name(second_candidate_name)
        candidate = first.evaluate(
            source=_source("process"),
            case=case,
            candidate_id="g0000-slot-01",
        )
        assert candidate["process_id"] == first.worker_pid
        assert candidate["process_id"] != os.getpid()
        assert first_candidate_name != second_candidate_name
        assert len(first_candidate_name.encode("ascii")) <= 15
        assert (
            Path(f"/proc/{first.worker_pid}/comm").read_text().strip()
            == first_candidate_name
        )

        baseline_name = search_module._evaluation_process_name(
            0, "baseline:structural"
        )
        first.set_worker_name(baseline_name)
        baseline = first.evaluate_baseline(
            baseline="structural",
            case=case,
            generation=0,
        )
        assert baseline["process_id"] == first.worker_pid
        assert baseline["process_id"] != os.getpid()
        assert "struct" in baseline_name
        assert Path(f"/proc/{first.worker_pid}/comm").read_text().strip() == baseline_name
    finally:
        first.close()
        second.close()
    assert not Path(f"/proc/{first.worker_pid}").exists()
    assert not Path(f"/proc/{second.worker_pid}").exists()


class _FailFirstEvaluator(_Evaluator):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        if not self.failed:
            self.failed = True
            raise RuntimeError("fixture scorer infrastructure failure")
        return super().evaluate(
            source=source,
            case=case,
            candidate_id=candidate_id,
        )


class _Provider:
    model = "fixture-model"
    effort = "high"

    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls: list[tuple[str, int, str]] = []
        self.closed = False
        self.anchor = M5ProviderContextV1(
            "anchor-thread",
            "anchor-turn",
            "/opaque/anchor",
            ("anchor-turn",),
        )

    def ensure_specification_anchor(
        self,
        *,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, artifact_dir
        return M5ProviderResultV1(
            response_text=json.dumps(
                {
                    "schema_version": ("mforge.native.python_m5_specification_ack.v1"),
                    "ack": "specification-retained",
                }
            ),
            context=self.anchor,
            usage=_usage(),
            duration_ms=1,
            warnings=0,
        )

    def _result(
        self,
        *,
        kind: str,
        generation: int,
        slot: str,
        history: tuple[str, ...],
    ) -> M5ProviderResultV1:
        self.calls.append((kind, generation, slot))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("fixture provider failure")
        turn_id = f"turn-{generation}-{slot}-{kind}"
        return M5ProviderResultV1(
            response_text=_envelope(_source(f"{generation}-{slot}-{kind}")),
            context=M5ProviderContextV1(
                f"thread-{generation}-{slot}-{kind}",
                turn_id,
                f"/opaque/{generation}/{slot}/{kind}",
                history + (turn_id,),
            ),
            usage=_usage(),
            duration_ms=1,
            warnings=0,
        )

    def generate_root(
        self,
        *,
        anchor: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, idempotency_key, artifact_dir
        return self._result(
            kind="root",
            generation=generation,
            slot=slot,
            history=anchor.included_turn_ids,
        )

    def generate_child(
        self,
        *,
        parent: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, idempotency_key, artifact_dir
        return self._result(
            kind="child",
            generation=generation,
            slot=slot,
            history=parent.included_turn_ids,
        )

    def repair(
        self,
        *,
        previous: M5ProviderResultV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, idempotency_key, artifact_dir
        return self._result(
            kind="repair",
            generation=generation,
            slot=slot,
            history=previous.context.included_turn_ids,
        )

    def close(self) -> None:
        self.closed = True


def _no_provenance(**_: Any) -> Mapping[str, JsonValue]:
    return {"sha256": "fixture"}


def test_python_preview_config_is_explicit_and_native_v2_stays_default(
    tmp_path: Path,
) -> None:
    native_v2 = tmp_path / "native-v2.toml"
    native_v2.write_text('schema_version = "mforge.experiment.v2"\n')
    preview = _config(tmp_path)

    assert experiment_protocol(native_v2) == "native-v2"
    assert experiment_protocol(preview) == PYTHON_EXPERIMENT_PROTOCOL_ID
    loaded = load_python_preview_config(preview)
    assert loaded.protocol == PYTHON_EXPERIMENT_PROTOCOL_ID
    assert loaded.model == "fixture-model"
    assert loaded.experiment_root.name == "python-preview"


def test_omitted_provider_budgets_follow_mutable_generation_limit(
    tmp_path: Path,
) -> None:
    path = _scientific_status_config(tmp_path, exp_id="derived-provider-budget")
    source = path.read_text(encoding="utf-8")
    source = source.replace("primary_program_slots = 16\n", "")
    source = source.replace("provider_total_turn_limit = 16\n", "")
    path.write_text(source, encoding="utf-8")

    initial = load_python_preview_config(path)
    assert initial.scientific_search is not None
    assert initial.scientific_search.primary_program_slots is None
    assert initial.scientific_search.provider_total_turn_limit is None
    assert initial.scientific_search.current_primary_program_slots == 16
    assert initial.scientific_search.current_provider_total_turn_limit == 16

    path.write_text(
        source.replace("generation_limit = 2", "generation_limit = 5"),
        encoding="utf-8",
    )
    extended = load_python_preview_config(path)
    assert extended.scientific_config_sha256 == initial.scientific_config_sha256
    assert extended.scientific_search is not None
    assert extended.scientific_search.current_primary_program_slots == 40
    assert extended.scientific_search.current_provider_total_turn_limit == 40


def test_explicit_primary_budget_must_match_fresh_generation_limit(
    tmp_path: Path,
) -> None:
    path = _scientific_status_config(tmp_path, exp_id="invalid-provider-budget")
    source = path.read_text(encoding="utf-8")
    path.write_text(
        source.replace("primary_program_slots = 16", "primary_program_slots = 8")
        .replace("provider_total_turn_limit = 16", "provider_total_turn_limit = 8"),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="primary_program_slots must provide exactly eight slots per generation",
    ):
        load_python_preview_config(path)


def test_preview_runs_two_generations_and_status_is_bounded_read_only(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    provider = _Provider()
    evaluator = _Evaluator()
    backend = _Backend()
    result = run_python_preview(
        path,
        provider_factory=lambda *_: provider,
        backend_factory=lambda _: backend,
        evaluator_factory=lambda *_: evaluator,
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    root = load_python_preview_config(path).experiment_root
    protocol = read_json(root / "protocol.json.gz")
    report = read_json(root / "m5-report.json.gz")
    before = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }
    status = python_preview_status(path)
    after = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }

    assert result["state"] == status["state"] == "completed"
    assert isinstance(protocol, dict) and protocol["preview_active"] is True
    assert isinstance(report, dict) and report["preview_active"] is True
    assert result["preview_active"] is True
    assert result["native_v2_default"] is True
    assert result["dsl_runtime_used"] is False
    assert result["scientific_result_kind"] == "DEVELOPMENT_SEARCH_EVIDENCE"
    assert result["scientific_success"] is False
    assert result["counts"] == {
        "planned": 16,
        "terminal": 16,
        "pending": 0,
        "valid": 16,
        "contract_invalid": 0,
        "duplicate": 0,
        "provider_failed": 0,
        "evaluation_infrastructure_failure": 0,
        "evaluated": 16,
        "missing": 0,
        "roots": 12,
        "children": 4,
    }
    assert result["provider"]["turns"] == 17
    assert result["provider"]["forks"] == 0
    assert result["sandbox"]["starts"] == 32
    assert result["policy_invocations"] == 32
    assert result["graph_scores"] == {"attempts": 64, "unique_graphs": 64}
    assert len(result["programs"]) == 16
    assert all("source" not in item for item in result["programs"])
    assert "workspace" not in result
    assert "thread_id" not in json.dumps(result)
    assert "turn_id" not in json.dumps(result)
    assert before == after
    assert backend.closed is True
    assert provider.closed is True


def test_json_dsl_workspace_is_rejected_before_provider_or_backend(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    config = load_python_preview_config(path)
    root = config.experiment_root
    root.mkdir(parents=True)
    write_json(
        root / "v3-state.json.gz",
        {
            "schema_version": "mforge.experiment.status.v3",
            "protocol": "v3",
        },
    )
    before = {
        item.relative_to(root): item.read_bytes() for item in root.rglob("*") if item.is_file()
    }
    calls = {"provider": 0, "backend": 0}

    with pytest.raises(
        PythonPreviewWorkspaceError,
        match="cannot be migrated or reinterpreted",
    ):
        run_python_preview(
            path,
            provider_factory=lambda *_: calls.__setitem__("provider", calls["provider"] + 1),
            backend_factory=lambda _: calls.__setitem__("backend", calls["backend"] + 1),
            provenance_guard=_no_provenance,
            auth_available=lambda _: True,
        )

    after = {
        item.relative_to(root): item.read_bytes() for item in root.rglob("*") if item.is_file()
    }
    assert calls == {"provider": 0, "backend": 0}
    assert before == after


def test_legacy_ranker_prompt_is_rejected_before_provider_or_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _config(tmp_path)
    (
        system_prompt,
        _request_template,
        _specification_prompt,
        policy_schema,
        ack_schema,
    ) = preview_module._prompt_inputs()
    legacy_request = "Generate a ranker defining priority(ctx, proposal)."
    monkeypatch.setattr(
        preview_module,
        "_prompt_inputs",
        lambda: (
            system_prompt,
            legacy_request,
            legacy_request,
            policy_schema,
            ack_schema,
        ),
    )
    calls = {"provider": 0, "backend": 0}

    with pytest.raises(
        ValueError,
        match="does not use the canonical propose contract",
    ):
        run_python_preview(
            path,
            provider_factory=lambda *_: calls.__setitem__("provider", calls["provider"] + 1),
            backend_factory=lambda _: calls.__setitem__("backend", calls["backend"] + 1),
            provenance_guard=_no_provenance,
            auth_available=lambda _: True,
        )

    assert calls == {"provider": 0, "backend": 0}


def test_provider_failure_is_not_scientific_success_and_resume_skips_it(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    first = _Provider(fail_first=True)
    first_result = run_python_preview(
        path,
        provider_factory=lambda *_: first,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert first_result["state"] == "blocked"
    assert first_result["counts"]["provider_failed"] == 1
    assert first_result["counts"]["pending"] == 7
    assert first_result["scientific_success"] is False
    assert first_result["scientific_result_kind"] == "NO_SCIENTIFIC_RESULT"

    resumed = _Provider()
    result = run_python_preview(
        path,
        provider_factory=lambda *_: resumed,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert result["state"] == "completed"
    assert result["counts"]["provider_failed"] == 1
    assert len(resumed.calls) == 15
    assert ("root", 0, "slot-00") not in resumed.calls
    assert result["recovery"]["resume_attempts"] == 1


def test_wall_clock_budget_state_is_resumed_after_legacy_completed_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _config(tmp_path)
    run_python_preview(
        path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    config = load_python_preview_config(path)
    state_path = config.experiment_root / "python-preview-state.json.gz"
    state = read_json(state_path)
    assert isinstance(state, dict)
    state.update(
        {
            "state": "completed",
            "run_terminal": True,
            "terminal_reason": "wall_clock_budget",
        }
    )
    write_json(state_path, state)

    calls = {"provider": 0}

    def resume_probe(**_: Any) -> Mapping[str, JsonValue]:
        raise RuntimeError("resume entered")

    monkeypatch.setattr(preview_module, "run_m5_search", resume_probe)
    result = run_python_preview(
        path,
        provider_factory=lambda *_: calls.__setitem__("provider", calls["provider"] + 1)
        or _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )

    assert calls["provider"] == 1
    assert result["state"] == "blocked"
    assert result["last_error"] == "RuntimeError"


def test_current_provider_error_is_not_masked_by_retained_report(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)

    def missing_provider(*_: Any) -> _Provider:
        raise FileNotFoundError("provider capsule missing")

    result = run_python_preview(
        path,
        provider_factory=missing_provider,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert result["terminal_reason"] == "provider_runtime_missing"

    config = load_python_preview_config(path)
    write_json(
        config.experiment_root / "m10-report.json.gz",
        {
            "protocol_id": M10_REPORT_PROTOCOL_ID,
            "stop_reason": "wall_clock_budget",
            "generation_count": 1,
            "exact_verified": False,
        },
    )
    status = python_preview_status(path)
    assert status["state"] == "blocked"
    assert status["terminal_reason"] == "provider_runtime_missing"

    state_path = config.experiment_root / "python-preview-state.json.gz"
    state = read_json(state_path)
    assert isinstance(state, dict)
    state.update(
        {
            "state": "running",
            "last_error": None,
            "terminal_reason": None,
            "last_boundary": "protocol_persisted",
        }
    )
    write_json(state_path, state)
    assert python_preview_status(path)["state"] == "running"


def test_missing_provider_runtime_fails_closed_without_copying_artifacts(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    run_python_preview(
        path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    base_config = load_python_preview_config(path)
    provider_state = base_config.experiment_root / "provider-runtime" / "provider-state.json.gz"
    provider_state.parent.mkdir(parents=True, exist_ok=True)
    write_json(provider_state, {"capsule_root": str(tmp_path / "missing-capsule")})
    old_candidate = base_config.experiment_root / "generations/generation-0000/slot-00"
    assert old_candidate.is_dir()

    provider_calls = 0

    def provider_factory(config: Any, _: str) -> _Provider:
        nonlocal provider_calls
        del config
        provider_calls += 1
        return _Provider()

    resumed = run_python_preview(
        path,
        provider_factory=provider_factory,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )

    assert resumed["state"] == "failed"
    assert resumed["terminal_reason"] == "provider_runtime_missing"
    assert resumed["resumable"] is False
    assert provider_calls == 0
    assert old_candidate.is_dir()
    marker = base_config.workspace / f".{base_config.exp_id}.active-recovery.json.gz"
    assert not marker.exists()
    assert python_preview_status(path)["state"] == "failed"


def test_resumable_operator_stop_is_consumed_and_can_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _config(tmp_path)
    original_run = preview_module.run_m5_search

    def controlled_stop(**kwargs: Any) -> Mapping[str, JsonValue]:
        requested = request_python_preview_stop(path)
        assert requested["stop_requested"] is True
        assert kwargs["operator_stop"]() is True
        raise M5OperatorStop("operator stop requested")

    monkeypatch.setattr(preview_module, "run_m5_search", controlled_stop)
    stopped = run_python_preview(
        path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert stopped["state"] == "blocked"
    assert stopped["terminal_reason"] == "operator_stop"
    assert stopped["resumable"] is True
    assert stopped["last_error"] is None

    monkeypatch.setattr(preview_module, "run_m5_search", original_run)
    provider = _Provider()
    resumed = run_python_preview(
        path,
        provider_factory=lambda *_: provider,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert resumed["state"] == "completed"
    assert resumed["counts"]["pending"] == 0
    assert resumed["recovery"]["resume_attempts"] == 1
    assert len(provider.calls) == 16


def test_provenance_failure_stops_before_provider_and_is_terminal(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    calls = {"provider": 0, "backend": 0}

    def reject_provenance(**_: Any) -> Mapping[str, JsonValue]:
        raise RuntimeError("dirty mutation-forge worktree")

    result = run_python_preview(
        path,
        provider_factory=lambda *_: calls.__setitem__("provider", calls["provider"] + 1),
        backend_factory=lambda _: calls.__setitem__("backend", calls["backend"] + 1),
        provenance_guard=reject_provenance,
        auth_available=lambda _: True,
    )

    assert result["state"] == "failed"
    assert result["resumable"] is False
    assert result["run_terminal"] is True
    assert result["terminal_reason"] == "provenance_mismatch"
    assert result["scientific_success"] is False
    assert calls == {"provider": 0, "backend": 0}


def test_evaluation_infrastructure_failure_is_distinct_and_consumed_on_resume(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    first_provider = _Provider()
    first = run_python_preview(
        path,
        provider_factory=lambda *_: first_provider,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _FailFirstEvaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert first["state"] == "blocked"
    assert first["counts"]["evaluation_infrastructure_failure"] == 1
    assert first["scientific_success"] is False

    resumed_provider = _Provider()
    resumed = run_python_preview(
        path,
        provider_factory=lambda *_: resumed_provider,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert resumed["state"] == "completed"
    assert resumed["counts"]["evaluation_infrastructure_failure"] == 1
    assert resumed["counts"]["pending"] == 0
    assert len(resumed_provider.calls) == 15
    assert ("root", 0, "slot-00") not in resumed_provider.calls


def test_status_rejects_unknown_private_state_without_echoing_it(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    run_python_preview(
        path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    config = load_python_preview_config(path)
    state_path = config.experiment_root / "python-preview-state.json.gz"
    state = read_json(state_path)
    assert isinstance(state, dict)
    state["source"] = "SOURCE_SECRET"
    state["thread_id"] = "THREAD_SECRET"
    state["workspace"] = "/private/workspace"
    write_json(state_path, state)

    status = python_preview_status(path)
    encoded = json.dumps(status)
    assert status["state"] == "failed"
    assert status["terminal_reason"] == "workspace_mismatch"
    assert "SOURCE_SECRET" not in encoded
    assert "THREAD_SECRET" not in encoded
    assert "/private/workspace" not in encoded


def test_cleanup_runs_once_and_does_not_mask_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _config(tmp_path)

    class FailingProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("provider cleanup secret")

    class TrackingBackend(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            self.closed = True

    provider = FailingProvider()
    backend = TrackingBackend()
    monkeypatch.setattr(
        preview_module,
        "run_m5_search",
        lambda **_: (_ for _ in ()).throw(RuntimeError("primary secret")),
    )

    result = run_python_preview(
        path,
        provider_factory=lambda *_: provider,
        backend_factory=lambda _: backend,
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )
    assert result["state"] == "blocked"
    assert result["last_error"] == "RuntimeError"
    assert "secret" not in json.dumps(result)
    assert provider.close_calls == 1
    assert backend.close_calls == 1


def test_owned_provider_can_preserve_durable_capsule_for_resume() -> None:
    class Adapter:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class Capsule:
        def __init__(self) -> None:
            self.cleanup_calls = 0

        def cleanup(self) -> None:
            self.cleanup_calls += 1

    provider = object.__new__(CodexM5SearchProvider)
    adapter = Adapter()
    capsule = Capsule()
    provider.adapter = adapter  # type: ignore[assignment]
    provider._owns_adapter = True
    provider._cleanup_capsule = True
    provider._capsule = capsule  # type: ignore[assignment]

    provider.close(cleanup_capsule=False)

    assert adapter.close_calls == 1
    assert capsule.cleanup_calls == 0


def _evaluation_payload(
    multiplier: int,
    *,
    timeout: bool = False,
) -> dict[str, Any]:
    return {
        "runtime_profile": {
            "sandbox_wall_seconds": multiplier * 0.5,
            "selector_wall_seconds": multiplier * 0.25,
            "action_wall_seconds": multiplier * 0.125,
        },
        "worker_telemetry": {
            "rotations": multiplier,
            "failures": multiplier,
            "worker_rss_kib": multiplier * 100,
        },
        "scientific_result": {
            "steps": [{} for _ in range(multiplier)],
            "score_attempts": multiplier * 2,
            "unique_graph_scores": multiplier,
            "terminal_evidence": {
                "components": [
                    {
                        "forbidden_length": 4,
                        "wall_time_ns": multiplier * 1_000_000_000,
                    }
                ]
            },
            "failure": {"code": "PROPOSE_TIMEOUT"} if timeout else None,
        },
    }


def test_compact_evaluation_telemetry_preserves_exact_aggregates() -> None:
    candidate_summary = _evaluation_telemetry_summary(
        [_evaluation_payload(1, timeout=True), _evaluation_payload(2)]
    )
    baseline_summary = _evaluation_telemetry_summary(
        [_evaluation_payload(3)]
    )

    telemetry, complete = preview_module._compact_evaluation_telemetry(
        [
            {
                "evaluation_case_count": 2,
                "evaluation_telemetry": candidate_summary,
            },
            {
                "evaluation_case_count": 1,
                "evaluation_telemetry": baseline_summary,
            },
        ]
    )

    assert complete is True
    assert telemetry == {
        "starts": 3,
        "rotations": 6,
        "failures": 6,
        "timeouts": 1,
        "maximum_rss_kib": 300,
        "policy_invocations": 6,
        "graph_score_attempts": 12,
        "unique_graph_scores": 6,
        "sandbox_wall_seconds": 3.0,
        "selector_wall_seconds": 1.5,
        "action_wall_seconds": 0.75,
        "scoring_wall_seconds": 6.0,
    }


def test_bootstrap_and_cold_status_never_load_historical_case_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _scientific_status_config(tmp_path, exp_id="large-history")
    config = load_python_preview_config(config_path)
    preview_module._initialize_workspace(config)
    evaluation_dir = (
        config.experiment_root
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "evaluations"
    )
    evaluation_dir.mkdir(parents=True)
    compressed = compress_json_bytes(b'{"retained":true}')
    for index in range(5_000):
        (evaluation_dir / f"case-{index:04d}.json.gz").write_bytes(compressed)
    write_json(
        evaluation_dir.parent / "candidate.json.gz",
        {
            "candidate_id": "g0000-slot-00",
            "generation": 0,
            "slot": "slot-00",
            "status": "evaluation_infrastructure_failure",
            "evaluation_case_count": 5_000,
        },
    )
    loaded: list[Path] = []
    original = preview_module._load_mapping

    def tracked(path: Path) -> dict[str, Any] | None:
        if path.parent.name == "evaluations":
            loaded.append(path)
        return original(path)

    monkeypatch.setattr(preview_module, "_load_mapping", tracked)

    bootstrap = python_preview_bootstrap_status(config_path)
    first = python_preview_status(config_path)
    for _ in range(20):
        python_preview_status(config_path)

    assert bootstrap["state"] == first["state"] == "ready"
    assert first["profiling_enabled"] is False
    assert loaded == []


def test_status_reconstructs_cumulative_evaluation_progress_after_resume(
    tmp_path: Path,
) -> None:
    config_path = _scientific_status_config(tmp_path, exp_id="cumulative-progress")
    config = load_python_preview_config(config_path)
    preview_module._initialize_workspace(config)
    root = config.experiment_root
    panel = [
        DevelopmentCaseV1(
            case_id=f"case-{index}",
            order=8,
            graph_seed=101,
            policy_seed=17 + index,
            horizon=1,
            witness_cap=64,
            forbidden_lengths=(4,),
        ).as_dict()
        for index in range(4)
    ]
    generation_zero = root / "generations" / "generation-0000"
    generation_one = root / "generations" / "generation-0001"
    write_json(
        generation_zero / "manifest.json.gz",
        {
            "generation": 0,
            "panel": panel,
            "slots": [
                {"slot": "slot-00", "kind": "root"},
                {"slot": "slot-01", "kind": "root"},
            ],
        },
    )
    for slot in ("slot-00", "slot-01"):
        write_json(
            generation_zero / slot / "prepared-candidate.json.gz",
            {"candidate_id": f"g0000-{slot}"},
        )
        write_json(
            generation_zero / slot / "candidate.json.gz",
            {
                "candidate_id": f"g0000-{slot}",
                "generation": 0,
                "slot": slot,
                "status": "evaluated",
                "evaluation_case_count": 4,
            },
        )
    for baseline in ("random", "structural"):
        write_json(
            generation_zero
            / "baselines"
            / baseline
            / search_module.M10_BASELINE_RESULT_FILENAME,
            {
                "evaluation_case_count": 4,
                "profile": {
                    "fitness_interval": {
                        "lower": {"numerator": 0, "denominator": 1},
                        "upper": {"numerator": 0, "denominator": 1},
                    },
                },
            },
        )

    before_resume = python_preview_status(config_path)["evaluation_cases"]
    assert before_resume["baseline"]["completed"] == 8
    assert before_resume["baseline"]["total"] == 8
    assert before_resume["candidate"]["completed"] == 8
    assert before_resume["candidate"]["total"] == 8

    write_json(
        generation_one / "manifest.json.gz",
        {
            "generation": 1,
            "panel": panel,
            "slots": [{"slot": "slot-00", "kind": "root"}],
        },
    )
    write_json(
        root / search_module.M10_RUNTIME_FILENAME,
        {
            "evaluation_progress": {
                "baseline:g0001-random": {
                    "completed": 2,
                    "total": 4,
                    "queued": 2,
                    "running": 0,
                    "state": "queued",
                },
                "baseline:g0001-structural": {
                    "completed": 1,
                    "total": 4,
                    "queued": 3,
                    "running": 0,
                    "state": "queued",
                },
            },
        },
    )

    baseline_snapshot = python_preview_status(config_path)["evaluation_cases"]
    assert baseline_snapshot["baseline"]["completed"] == 11
    assert baseline_snapshot["baseline"]["total"] == 16
    assert baseline_snapshot["candidate"]["completed"] == 8
    assert baseline_snapshot["candidate"]["total"] == 8

    write_json(
        generation_one / "slot-00" / "prepared-candidate.json.gz",
        {"candidate_id": "g0001-slot-00"},
    )
    prepared_snapshot = python_preview_status(config_path)["evaluation_cases"]
    assert prepared_snapshot["candidate"]["completed"] == 8
    assert prepared_snapshot["candidate"]["total"] == 12

    runtime = cast(dict[str, Any], read_json(root / search_module.M10_RUNTIME_FILENAME))
    progress = cast(dict[str, Any], runtime["evaluation_progress"])
    progress["candidate:g0001-slot-00"] = {
        "completed": 2,
        "total": 4,
        "queued": 2,
        "running": 0,
        "state": "queued",
    }
    write_json(root / search_module.M10_RUNTIME_FILENAME, runtime)
    completed_snapshot = python_preview_status(config_path)["evaluation_cases"]
    assert completed_snapshot["candidate"]["completed"] == 10
    assert completed_snapshot["candidate"]["total"] == 12
    assert completed_snapshot["completed"] == 21
    assert completed_snapshot["total"] == 28
    for snapshot in (
        before_resume,
        baseline_snapshot,
        prepared_snapshot,
        completed_snapshot,
    ):
        assert snapshot["baseline"]["completed"] <= snapshot["baseline"]["total"]
        assert snapshot["candidate"]["completed"] <= snapshot["candidate"]["total"]
        assert snapshot["completed"] <= snapshot["total"]
def test_blocked_preview_preserves_owned_capsule_for_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class Capsule:
        def __init__(self) -> None:
            self.cleanup_calls = 0

        def cleanup(self) -> None:
            self.cleanup_calls += 1

    provider = object.__new__(CodexM5SearchProvider)
    adapter = Adapter()
    capsule = Capsule()
    provider.adapter = adapter  # type: ignore[assignment]
    provider._owns_adapter = True
    provider._cleanup_capsule = True
    provider._capsule = capsule  # type: ignore[assignment]
    provider.model = "fixture-model"
    provider.effort = "high"
    monkeypatch.setattr(
        preview_module,
        "run_m5_search",
        lambda **_: (_ for _ in ()).throw(RuntimeError("resumable failure")),
    )

    result = run_python_preview(
        _config(tmp_path),
        provider_factory=lambda *_: provider,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(),
        provenance_guard=_no_provenance,
        auth_available=lambda _: True,
    )

    assert result["state"] == "blocked"
    assert adapter.close_calls == 1
    assert capsule.cleanup_calls == 0


def test_public_cli_routes_python_preview_without_dsl_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "run_python_preview",
        lambda _: calls.append("run") or {"state": "completed"},
    )
    monkeypatch.setattr(
        cli,
        "python_preview_status",
        lambda _: calls.append("status") or {"state": "completed"},
    )
    assert not hasattr(cli, "run_v3")

    assert cli.main(["experiment", "run", "--config", str(path), "--json"]) == 0
    assert cli.main(["experiment", "status", "--config", str(path), "--json"]) == 0
    assert calls == ["run", "status"]
    assert capsys.readouterr().out.count('"state":"completed"') == 2


def test_python_preview_call_path_has_no_json_dsl_runtime_or_ir_compiler() -> None:
    source = inspect.getsource(preview_module)
    forbidden = (
        "native_v3.interpreter",
        "native_v3.single_program_ir",
        "invoke_program(",
        "evaluate_serial_program(",
        "compile_program(",
    )
    assert all(token not in source for token in forbidden)
    protocol_source = (Path(preview_module.__file__).parent / "contracts.py").read_text(
        encoding="utf-8"
    )
    assert "mforge.experiment.v3" in protocol_source
    python_serial_source = (Path(preview_module.__file__).parent / "serial_evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "native_v3.interpreter" not in python_serial_source
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "import mutation_forge.native_v3_python.preview;"
                "assert 'mutation_forge.native_v3.interpreter' not in sys.modules;"
                "assert not any('single_program_ir' in name for name in sys.modules);"
                "import importlib.util;"
                "assert importlib.util.find_spec('mutation_forge.native_v3.interpreter')"
                " is None;"
                "assert importlib.util.find_spec("
                "'mutation_forge.native_v3.single_program_ir') is None"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
