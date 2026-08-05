from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge.backends.base import ScoringBackendError
from mutation_forge.backends.heg import HegBackend
from mutation_forge.models import GraphState, GraphValidation
from mutation_forge.native_v3.heg_scoring import (
    HegScoreEvidenceAdapter,
    merge_score_evidence,
)
from mutation_forge.native_v3.scoring import AttemptKind, EvidenceStatus


@dataclass
class _Result:
    length: int
    count: int
    complete: bool
    nodes: int
    elapsed_ns: int = 1


@dataclass
class _Response:
    results: tuple[_Result, ...]
    dominated: bool = False


class _Worker:
    binary = Path(__file__)


class _Prepared:
    graph = object()


class _FakeBackend:
    backend_id = "fake-heg"
    commit = "commit"
    repo = Path(__file__).resolve().parents[2]

    def _score_worker(self) -> _Worker:
        return _Worker()

    def validate(self, _graph: GraphState) -> GraphValidation:
        return GraphValidation(True)

    def target_forbidden_lengths(self, _order: int) -> tuple[int, ...]:
        return (4, 5)

    def _prepare(self, _graph: GraphState) -> _Prepared:
        return _Prepared()

    def _worker_response(self, *_args: object, **kwargs: object) -> _Response:
        requested = cast(tuple[int, ...], kwargs["lengths"])
        available = {
            4: _Result(4, 65, False, 100),
            5: _Result(5, 3, False, 50),
        }
        return _Response(tuple(available[length] for length in requested))


def test_adapter_preserves_component_exactness_and_safe_bounds() -> None:
    adapter = HegScoreEvidenceAdapter(cast(HegBackend, _FakeBackend()))
    evidence = adapter.score(
        GraphState(6, ((0, 1),)),
        witness_cap=64,
        attempt_kind=AttemptKind.INITIAL,
    )
    assert evidence.components[0].status is EvidenceStatus.SATURATED_AT_CAP
    assert (evidence.components[0].lower_bound, evidence.components[0].upper_bound) == (
        64,
        64,
    )
    assert evidence.components[1].status is EvidenceStatus.SEARCH_BUDGET_EXHAUSTED
    assert (evidence.components[1].lower_bound, evidence.components[1].upper_bound) == (
        3,
        64,
    )


def test_expanded_evidence_cannot_weaken_an_existing_bound() -> None:
    adapter = HegScoreEvidenceAdapter(cast(HegBackend, _FakeBackend()))
    initial = adapter.score(GraphState(6, ((0, 1),)), witness_cap=64)
    expanded = adapter.score(
        GraphState(6, ((0, 1),)),
        witness_cap=64,
        forbidden_lengths=(4,),
        attempt_kind=AttemptKind.EXPANDED,
    )
    merged = merge_score_evidence(initial, expanded)
    assert tuple(component.forbidden_length for component in merged.components) == (4, 5)


class _FailingWorker:
    def score(self, *_args: object, **_kwargs: object) -> Any:
        raise OSError("worker failed")

    def restart(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_cpp_worker_failure_never_enters_a_python_reference_fallback() -> None:
    repository = Path(__file__).resolve().parents[3] / "heg"
    backend = HegBackend(repository)
    try:
        backend._worker = _FailingWorker()  # noqa: SLF001
        with pytest.raises(ScoringBackendError, match="after one restart"):
            backend._worker_response(  # noqa: SLF001
                object(),
                lengths=(4,),
                limit=65,
                node_budget=50_000,
                cutoff=None,
                recorder=None,
            )
        assert backend.score_implementation == "heg-cpp-score-worker"
        assert not hasattr(backend, "_reference_cycle_counts")
    finally:
        backend.close()


def test_heg_witness_feature_adapter_returns_sampled_vertex_and_edge_loads() -> None:
    repository = Path(__file__).resolve().parents[3] / "heg"
    backend = HegBackend(repository)
    try:
        graph = backend.generate_seed(order=4, seed=7)
        vertex_loads, edge_loads = backend.sampled_forbidden_witness_loads(
            graph,
            relabeling=tuple(reversed(range(graph.order))),
        )
    finally:
        backend.close()
    assert vertex_loads
    assert edge_loads
    assert {length for length, _vertex in vertex_loads} == {4}
    assert {length for length, _edge in edge_loads} == {4}
