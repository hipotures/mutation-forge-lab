from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mutation_forge import cli
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3_python import preview as preview_module
from mutation_forge.native_v3_python.contracts import (
    PYTHON_EXPERIMENT_PROTOCOL_ID,
)
from mutation_forge.native_v3_python.preview import (
    PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION,
    PythonPreviewWorkspaceError,
    experiment_protocol,
    load_python_preview_config,
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
    owned = preview_module._BackendOwnedEvaluator(
        evaluator=_Evaluator(),
        backend=_Backend(),
    )

    assert owned._process_name("g0000-slot-00") == "mforge-eval-00"
    assert owned._process_name("g0000-slot-07") == "mforge-eval-07"


class _Backend:
    backend_id = "fixture"
    score_implementation = "fixture"

    def __init__(self) -> None:
        self.closed = False

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
