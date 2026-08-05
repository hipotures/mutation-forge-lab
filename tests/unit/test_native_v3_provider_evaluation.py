from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import read_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)
from mutation_forge.native_v3.provider_evaluation import (
    run_provider_evaluation_smoke,
)
from mutation_forge.native_v3.serial_evaluator import SerialEpisodeConfig

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location(
    "native_v3_provider_evaluation_fake",
    _FIXTURE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario


def _response() -> str:
    source = json.dumps(
        {
            "schema_version": "mforge.native.program.v3",
            "entry": {"op": "no_plan", "reason": "EXPLICIT"},
        },
        separators=(",", ":"),
    )
    return json.dumps(
        {
            "schema_version": "mforge.native.generated_policy.v1",
            "source": source,
            "design_summary": "Return a deterministic no-plan program.",
            "hypothesis": "The provider result reaches authoritative evaluation.",
            "used_fields": [],
            "assumptions": [],
            "expected_failure_modes": [],
        },
        separators=(",", ":"),
    )


def _provider(*, authenticated: bool = True) -> LocalCodexAppServerProvider:
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
        auth_checker=lambda _: authenticated,
        persist_artifacts=False,
    )


class _Backend:
    backend_id = "recorded-response-backend"
    score_implementation = "recorded-response-scorer"

    def __init__(self) -> None:
        self.closed = False

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
        raise AssertionError("nonzero recorded score must not be verified")

    def canonical_hash(self, graph: GraphState) -> str:
        return self.state_hash(graph)

    def state_hash(self, graph: GraphState) -> str:
        return hashlib.sha256(repr(graph).encode()).hexdigest()

    def serialize_graph6(self, graph: GraphState) -> str:
        raise AssertionError("nonzero recorded score must not be serialized")

    def deserialize_graph6(self, value: str) -> GraphState:
        raise AssertionError("not used")

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: Any = None,
    ) -> GraphState:
        raise AssertionError("no-plan program must not apply a rewrite")

    def propose_rewrite(self, graph: GraphState, **_: Any) -> RewritePlan:
        raise AssertionError("Native v3 evaluator must not request backend proposals")

    def close(self) -> None:
        self.closed = True


def _config() -> SerialEpisodeConfig:
    return SerialEpisodeConfig(
        order=8,
        graph_seed=101,
        policy_seed=17,
        horizon=1,
        witness_cap=64,
        episode_id="recorded-provider-slot-00",
    )


def test_recorded_provider_response_replays_same_semantic_evaluation(
    tmp_path: Path,
) -> None:
    evaluations: list[dict[str, object]] = []
    for name in ("first", "second"):
        backend = _Backend()
        provider = _provider()
        try:
            report = run_provider_evaluation_smoke(
                provider,
                tmp_path / name,
                backend_factory=lambda backend=backend: backend,
                config=_config(),
            )
        finally:
            provider.close()

        assert report["status"] == "completed"
        assert report["model_turns"] == 1
        assert report["graph_evaluations"] == 1
        assert report["scientific_terminal_result"] is True
        assert backend.closed is True
        evaluation_path = Path(str(report["evaluation_result"]))
        payload = read_json(evaluation_path)
        evaluations.append(payload["evaluation"])
        turn = Path(str(report["provider_turn_directory"]))
        assert evaluation_path.parent != turn
        assert not (turn / evaluation_path.name).exists()
        assert payload["program"]["program_json_raw"] == _response_source()
        assert json.loads(payload["program"]["program_json_canonical"]) == json.loads(
            _response_source()
        )
        assert payload["protocols"]["interpreter"] == (
            "native_v3_graph_interpreter_v1"
        )
        assert payload["backend"]["score_implementation"] == (
            "recorded-response-scorer"
        )

    assert evaluations[0] == evaluations[1]


def _response_source() -> str:
    value = json.loads(_response())
    source = value["source"]
    assert isinstance(source, str)
    return source


def test_provider_failure_creates_no_scientific_terminal_result(
    tmp_path: Path,
) -> None:
    backend_created = False

    def backend_factory() -> _Backend:
        nonlocal backend_created
        backend_created = True
        return _Backend()

    provider = _provider(authenticated=False)
    try:
        report = run_provider_evaluation_smoke(
            provider,
            tmp_path,
            backend_factory=backend_factory,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "provider_error"
    assert report["scientific_terminal_result"] is False
    assert report["graph_evaluations"] == 0
    assert backend_created is False
    assert not (tmp_path / "native-v3-output" / "evaluation-result.json.gz").exists()
