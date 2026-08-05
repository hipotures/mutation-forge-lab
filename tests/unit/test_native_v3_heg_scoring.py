from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge.backends.base import ScoringBackendError
from mutation_forge.backends.heg import HegBackend
from mutation_forge.models import GraphState, GraphValidation
from mutation_forge.native_v3.heg_scoring import (
    EXPANDED_NODE_BUDGET,
    INITIAL_NODE_BUDGET,
    HegScoreEvidenceAdapter,
    ScoreContractViolation,
    merge_score_evidence,
    scorer_for_backend,
)
from mutation_forge.native_v3.scoring import (
    AttemptKind,
    EvidenceStatus,
    ScoreTimeoutWithoutPartial,
)


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
    timeout_seconds = 2.0


class _Prepared:
    graph = object()


class _FakeBackend:
    backend_id = "fake-heg"
    commit = "commit"
    repo = Path(__file__).resolve().parents[2]
    score_implementation = "heg-cpp-score-worker"

    def __init__(self) -> None:
        self.node_budgets: list[int] = []
        self.timeouts: list[float] = []
        self.worker = _Worker()

    def _score_worker(self) -> _Worker:
        return self.worker

    def validate(self, _graph: GraphState) -> GraphValidation:
        return GraphValidation(True)

    def target_forbidden_lengths(self, _order: int) -> tuple[int, ...]:
        return (4, 5)

    def state_hash(self, graph: GraphState) -> str:
        return repr(graph)

    def _prepare(self, _graph: GraphState) -> _Prepared:
        return _Prepared()

    def _worker_response(self, *_args: object, **kwargs: object) -> _Response:
        requested = cast(tuple[int, ...], kwargs["lengths"])
        self.node_budgets.append(int(kwargs["node_budget"]))
        self.timeouts.append(self.worker.timeout_seconds)
        available = {
            4: _Result(4, 65, False, 100),
            5: _Result(5, 3, False, 50),
        }
        return _Response(tuple(available[length] for length in requested))


def test_adapter_uses_locked_budgets_and_preserves_sound_bounds() -> None:
    backend = _FakeBackend()
    adapter = HegScoreEvidenceAdapter(cast(HegBackend, backend))
    graph = GraphState(6, ((0, 1),))
    initial = adapter.score_evidence(
        graph,
        witness_cap=64,
        attempt_kind=AttemptKind.INITIAL,
    )
    expanded = adapter.score_evidence(
        graph,
        witness_cap=64,
        forbidden_lengths=(5,),
        attempt_kind=AttemptKind.EXPANDED,
    )

    assert backend.node_budgets == [INITIAL_NODE_BUDGET, EXPANDED_NODE_BUDGET]
    assert backend.timeouts == [5.0, 20.0]
    assert backend.worker.timeout_seconds == 2.0
    assert initial.components[0].status is EvidenceStatus.SATURATED_AT_CAP
    assert initial.components[0].interval.exact
    assert (
        initial.components[1].status
        is EvidenceStatus.SEARCH_BUDGET_EXHAUSTED
    )
    assert (initial.components[1].lower_bound, initial.components[1].upper_bound) == (
        3,
        64,
    )
    merged = merge_score_evidence(initial, expanded)
    assert tuple(component.forbidden_length for component in merged.components) == (
        4,
        5,
    )


def test_non_heg_backend_cannot_silently_fall_back_to_graph_score() -> None:
    class OnlyLegacyScore:
        def score(self, *_args: object, **_kwargs: object) -> object:
            return object()

    with pytest.raises(TypeError, match="no explicit component-evidence"):
        scorer_for_backend(cast(Any, OnlyLegacyScore()))


def test_timeout_infrastructure_and_contract_failures_are_distinct() -> None:
    graph = GraphState(6, ((0, 1),))

    class TimeoutBackend(_FakeBackend):
        def _worker_response(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> _Response:
            raise ScoringBackendError("worker timed out")

    timeout_adapter = HegScoreEvidenceAdapter(
        cast(HegBackend, TimeoutBackend())
    )
    with pytest.raises(ScoreTimeoutWithoutPartial):
        timeout_adapter.score_evidence(graph, witness_cap=64)

    class InfrastructureBackend(_FakeBackend):
        def _worker_response(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> _Response:
            raise ScoringBackendError("worker pipe failed")

    infrastructure_adapter = HegScoreEvidenceAdapter(
        cast(HegBackend, InfrastructureBackend())
    )
    with pytest.raises(ScoringBackendError, match="pipe failed"):
        infrastructure_adapter.score_evidence(graph, witness_cap=64)

    class ContractBackend(_FakeBackend):
        def _worker_response(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> _Response:
            return _Response((_Result(4, 1, True, 1),))

    contract_adapter = HegScoreEvidenceAdapter(
        cast(HegBackend, ContractBackend())
    )
    with pytest.raises(ScoreContractViolation, match="omitted"):
        contract_adapter.score_evidence(graph, witness_cap=64)


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
                node_budget=INITIAL_NODE_BUDGET,
                cutoff=None,
                recorder=None,
            )
        assert backend.score_implementation == "heg-cpp-score-worker"
        assert not hasattr(backend, "_reference_cycle_counts")
    finally:
        backend.close()
