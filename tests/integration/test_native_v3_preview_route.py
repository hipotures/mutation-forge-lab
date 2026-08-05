from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)
from mutation_forge.native_v3.preview import (
    NativeV3PreviewConfig,
    NativeV3PreviewWorkspaceError,
    run_v3_preview,
    v3_preview_status,
)

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location(
    "native_v3_preview_route_fake",
    _FIXTURE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario


def _source() -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.program.v3",
            "entry": {"op": "no_plan", "reason": "EXPLICIT"},
        },
        separators=(",", ":"),
    )


def _response() -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.generated_policy.v1",
            "source": _source(),
            "design_summary": "Return one bounded no-plan program.",
            "hypothesis": "The preview route reaches authoritative evaluation.",
            "used_fields": [],
            "assumptions": [],
            "expected_failure_modes": [],
        },
        separators=(",", ":"),
    )


def _config(tmp_path: Path, *, extra: str = "") -> Path:
    heg_repo = tmp_path / "heg"
    (heg_repo / "src" / "sglab").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "experiment.toml"
    path.write_text(
        f'''schema_version = "mforge.experiment.v3-preview.v1"
protocol = "native-v3-preview"
exp_id = "preview"
workspace = "{(tmp_path / "workspace").as_posix()}"
{extra}

[native_v3_preview]
model = "gpt-5.6-luna"
effort = "high"
timeout_seconds = 30
heg_repo = "{heg_repo.as_posix()}"
''',
        encoding="utf-8",
    )
    return path


def _provider(_: NativeV3PreviewConfig) -> LocalCodexAppServerProvider:
    scenario = FakeScenario(final_text=_response())

    def process_factory(*_: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(scenario, **kwargs)

    return LocalCodexAppServerProvider(
        model="gpt-5.6-luna",
        effort="high",
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=1,
        process_factory=process_factory,
        auth_checker=lambda _: True,
        persist_artifacts=False,
    )


class _Backend:
    backend_id = "preview-recorded-backend"
    score_implementation = "preview-recorded-scorer"

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        return (4,)

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        return GraphState(order, ((0, 1), (1, 2), (2, 3), (0, 3)))

    def validate(self, graph: GraphState) -> GraphValidation:
        return GraphValidation(True)

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: Any = None,
    ) -> GraphScore:
        return GraphScore(True, ((4, 1),), 1, 1, True, (1, 1))

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        raise AssertionError("nonzero score must not be verified")

    def canonical_hash(self, graph: GraphState) -> str:
        return self.state_hash(graph)

    def state_hash(self, graph: GraphState) -> str:
        return hashlib.sha256(repr(graph).encode()).hexdigest()

    def serialize_graph6(self, graph: GraphState) -> str:
        raise AssertionError("nonzero score must not be serialized")

    def deserialize_graph6(self, value: str) -> GraphState:
        raise AssertionError("not used")

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: Any = None,
    ) -> GraphState:
        raise AssertionError("no-plan program must not rewrite")

    def propose_rewrite(self, graph: GraphState, **_: Any) -> RewritePlan:
        raise AssertionError("preview evaluator must not request backend proposals")

    def close(self) -> None:
        return None


def _backend(_: NativeV3PreviewConfig) -> _Backend:
    return _Backend()


def test_preview_completes_one_slot_and_status_is_read_only(tmp_path: Path) -> None:
    path = _config(tmp_path)

    result = run_v3_preview(
        path,
        provider_factory=_provider,
        backend_factory=_backend,
        auth_available=lambda _: True,
    )

    assert result["state"] == "completed"
    assert result["protocol_version"] == "native-v3-preview.v1"
    assert result["provider_turns"] == 1
    assert result["evaluation_count"] == 1
    assert result["latest_scientific_stop_reason"] == "smoke_panel_complete"
    turn = Path(str(result["artifacts"]["provider_turn_directory"]))
    evaluation = Path(str(result["artifacts"]["evaluation_result"]))
    assert evaluation.is_file()
    assert evaluation.parent != turn
    assert not (turn / evaluation.name).exists()

    before = {
        item.relative_to(tmp_path): (item.stat().st_mtime_ns, item.read_bytes())
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    status = v3_preview_status(path)
    after = {
        item.relative_to(tmp_path): (item.stat().st_mtime_ns, item.read_bytes())
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    assert status == result
    assert after == before


def test_auth_preflight_is_blocked_resumable_then_completes(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    provider_calls = 0
    backend_calls = 0

    def provider(config: NativeV3PreviewConfig) -> LocalCodexAppServerProvider:
        nonlocal provider_calls
        provider_calls += 1
        return _provider(config)

    def backend(config: NativeV3PreviewConfig) -> _Backend:
        nonlocal backend_calls
        backend_calls += 1
        return _backend(config)

    blocked = run_v3_preview(
        path,
        provider_factory=provider,
        backend_factory=backend,
        auth_available=lambda _: False,
    )

    assert blocked["state"] == "blocked"
    assert blocked["resumable"] is True
    assert blocked["latest_infrastructure_stop_reason"] == "preflight_failed"
    assert blocked["latest_scientific_stop_reason"] is None
    assert provider_calls == 0
    assert backend_calls == 0

    completed = run_v3_preview(
        path,
        provider_factory=provider,
        backend_factory=backend,
        auth_available=lambda _: True,
    )
    assert completed["state"] == "completed"
    assert provider_calls == 1
    assert backend_calls == 1


def test_v2_workspace_is_rejected_by_preview_without_mutation(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    root = tmp_path / "workspace" / "preview"
    root.mkdir(parents=True)
    (root / "state.sqlite3").write_bytes(b"v2-state-sentinel")
    (root / "experiment.toml").write_text(
        'schema_version = "mforge.experiment.v2"\n',
        encoding="utf-8",
    )
    before = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }

    with pytest.raises(
        NativeV3PreviewWorkspaceError,
        match="never reinterpret a Native v2 workspace",
    ):
        run_v3_preview(
            path,
            provider_factory=_provider,
            backend_factory=_backend,
            auth_available=lambda _: True,
        )
    status = v3_preview_status(path)
    after = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }

    assert status["state"] == "failed"
    assert status["resumable"] is False
    assert after == before


def test_mixed_config_fails_before_workspace_provider_or_backend(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path, extra="\n[search]\npopulation_size = 8\n")
    provider_called = False
    backend_called = False

    def provider(_: NativeV3PreviewConfig) -> LocalCodexAppServerProvider:
        nonlocal provider_called
        provider_called = True
        return _provider(_)

    def backend(_: NativeV3PreviewConfig) -> _Backend:
        nonlocal backend_called
        backend_called = True
        return _Backend()

    with pytest.raises(ValueError, match="cannot contain Native v2 fields"):
        run_v3_preview(
            path,
            provider_factory=provider,
            backend_factory=backend,
            auth_available=lambda _: True,
        )

    assert provider_called is False
    assert backend_called is False
    assert not (tmp_path / "workspace").exists()
