from __future__ import annotations

import json
from collections.abc import Iterable

from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)
from mutation_forge.native_v3.contracts import ValidatedProgram, validate_program
from mutation_forge.native_v3.evaluation import (
    EpisodeStatus,
    ScoreTimeoutWithoutPartial,
    evaluate_episode,
)
from mutation_forge.native_v3.scheduler import EpisodeSpec, EpisodeTask
from mutation_forge.native_v3.scoring import (
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EvidenceStatus,
    ScoreEvidence,
)
from mutation_forge.native_v3.verification import graph_content_hash

IDENTITY = BackendIdentity(
    "test",
    "commit",
    "source",
    "binary",
    "compiler",
    (),
    "platform",
    "arch",
)


def _program(entry: dict[str, object]) -> ValidatedProgram:
    raw = json.dumps(
        {"schema_version": "mforge.native.program.v3", "entry": entry},
        separators=(",", ":"),
    )
    result = validate_program(raw)
    assert result.program is not None, result.diagnostics
    return result.program


class _Backend:
    backend_id = "test"

    def target_forbidden_lengths(self, _order: int) -> tuple[int, ...]:
        return (4,)

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        del seed
        return GraphState(
            order,
            tuple((u, v) for u in range(order) for v in range(u + 1, order)),
        )

    def validate(self, _graph: GraphState) -> GraphValidation:
        return GraphValidation(True)

    def apply_rewrite(
        self, graph: GraphState, rewrite: RewritePlan, **_kwargs: object
    ) -> GraphState:
        edges = set(graph.edges)
        edges.difference_update(rewrite.removed_edges)
        edges.update(rewrite.added_edges)
        return GraphState(graph.order, tuple(edges))

    def score(self, *_args: object, **_kwargs: object) -> GraphScore | None:
        raise NotImplementedError

    def exact_verify(self, _graph: GraphState) -> ExactVerification:
        raise NotImplementedError

    def canonical_hash(self, graph: GraphState) -> str:
        return graph_content_hash(graph)

    state_hash = canonical_hash

    def serialize_graph6(self, _graph: GraphState) -> str:
        raise NotImplementedError

    def deserialize_graph6(self, _value: str) -> GraphState:
        raise NotImplementedError

    def propose_rewrite(self, *_args: object, **_kwargs: object) -> RewritePlan:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _Scorer:
    def __init__(self, *, timeout: bool = False, count: int = 0) -> None:
        self.timeout = timeout
        self.count = count

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        del forbidden_lengths, attempt_kind
        if self.timeout:
            raise ScoreTimeoutWithoutPartial
        component = CycleComponentEvidence(
            4,
            self.count,
            self.count,
            self.count,
            EvidenceStatus.EXACT,
            50_000,
            10,
            1,
            AttemptKind.INITIAL,
            IDENTITY,
        )
        return ScoreEvidence(
            graph_content_hash(graph),
            graph.order,
            len(graph.edges),
            witness_cap,
            (component,),
        )


def _task(program: ValidatedProgram, *, horizon: int = 3) -> EpisodeTask:
    graph = _Backend().generate_seed(order=4, seed=1)
    return EpisodeTask(
        program.program_hash,
        "manifest",
        "protocol",
        graph_content_hash(graph),
        horizon,
        "interpreter",
        "selectors",
        "score",
        "acceptance",
        EpisodeSpec(4, 1, 2),
    )


def test_no_plan_episode_has_horizon_plus_one_curve_and_routes_initial_zero() -> None:
    program = _program({"op": "no_plan", "reason": "EXPLICIT"})
    result = evaluate_episode(
        task=_task(program),
        program=program,
        backend=_Backend(),
        scorer=_Scorer(),
        witness_cap=64,
    )
    assert result.status is EpisodeStatus.COMPLETE
    assert len(result.trajectory) == 4
    assert result.auc is not None
    assert len(result.apparent_zeros) == 1
    assert result.apparent_zeros[0].provenance["source"] == "initial_graph"


def test_initial_timeout_without_partial_fills_full_interval_curve() -> None:
    program = _program({"op": "no_plan", "reason": "EXPLICIT"})
    result = evaluate_episode(
        task=_task(program, horizon=2),
        program=program,
        backend=_Backend(),
        scorer=_Scorer(timeout=True),
        witness_cap=64,
    )
    assert result.status is EpisodeStatus.INCONCLUSIVE_TIMEOUT
    assert len(result.trajectory) == 3
    assert all(interval.lower == 0 and interval.upper == 1 for interval in result.trajectory)
