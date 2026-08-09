"""Native v3 evidence adapter for the mandatory HEG C++ scorer."""

from __future__ import annotations

import hashlib
import platform
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from mutation_forge.backends.base import GraphBackend, ScoringBackendError
from mutation_forge.backends.heg import HegBackend
from mutation_forge.models import GraphState

from .scoring import (
    EXPANDED_NODE_BUDGET,
    INITIAL_NODE_BUDGET,
    SCORE_PROTOCOL_ID,
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EvidenceStatus,
    ScoreEvidence,
    ScoreEvidenceCache,
    ScoreEvidenceCacheKey,
    ScoreTimeoutWithoutPartial,
)

INITIAL_SCORE_WALL_SECONDS = 5.0
EXPANDED_SCORE_WALL_SECONDS = 20.0


class ScoreContractViolation(RuntimeError):
    """The scorer returned evidence that violates its locked contract."""


@runtime_checkable
class ScoreEvidenceScorer(Protocol):
    raw_graph_score_calls: int
    unique_graph_scores: int

    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence: ...


def _source_tree_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    source_root = repo / "src"
    for path in sorted(
        item
        for item in source_root.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts
    ):
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def backend_identity(backend: HegBackend) -> BackendIdentity:
    worker = backend._score_worker()  # noqa: SLF001 - protocol adapter boundary
    binary = Path(worker.binary)
    if not binary.is_file():
        raise ScoringBackendError(f"mandatory C++ score worker is missing: {binary}")
    return BackendIdentity(
        backend_id=backend.backend_id,
        heg_commit=backend.commit,
        source_tree_sha256=_source_tree_hash(backend.repo),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        compiler_identity="content-addressed-binary",
        build_flags=(),
        platform=platform.platform(),
        architecture=platform.machine(),
    )


class HegScoreEvidenceAdapter:
    """Expose sound component evidence without a scoring fallback."""

    def __init__(self, backend: HegBackend) -> None:
        if backend.score_implementation != "heg-cpp-score-worker":
            raise ScoringBackendError(
                "Native v3 requires the mandatory HEG C++ score worker"
            )
        self.backend = backend
        self.identity = backend_identity(backend)
        self.cache = ScoreEvidenceCache()
        self.raw_graph_score_calls = 0
        self.unique_graph_scores = 0

    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        if witness_cap <= 0:
            raise ValueError("witness cap must be positive")
        validation = self.backend.validate(graph)
        if not validation.valid:
            raise ScoreContractViolation(
                f"cannot score an invalid graph: {validation.errors}"
            )
        target_lengths = self.backend.target_forbidden_lengths(graph.order)
        requested = (
            target_lengths
            if forbidden_lengths is None
            else tuple(sorted(set(forbidden_lengths)))
        )
        if not requested or not set(requested).issubset(target_lengths):
            raise ValueError(
                "requested forbidden lengths are not valid for this graph order"
            )
        node_budget = (
            INITIAL_NODE_BUDGET
            if attempt_kind is AttemptKind.INITIAL
            else EXPANDED_NODE_BUDGET
        )
        graph_hash = self.backend.state_hash(graph)
        cache_key = ScoreEvidenceCacheKey(
            graph_hash,
            requested,
            witness_cap,
            node_budget,
            attempt_kind,
            SCORE_PROTOCOL_ID,
            self.identity.canonical_key(),
        )
        self.raw_graph_score_calls += 1
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        self.unique_graph_scores += 1
        worker = self.backend._score_worker()  # noqa: SLF001
        previous_timeout = worker.timeout_seconds
        worker.timeout_seconds = (
            INITIAL_SCORE_WALL_SECONDS
            if attempt_kind is AttemptKind.INITIAL
            else EXPANDED_SCORE_WALL_SECONDS
        )
        prepared = self.backend._prepare(graph)  # noqa: SLF001
        started_ns = time.perf_counter_ns()
        try:
            response = self.backend._worker_response(  # noqa: SLF001
                prepared.graph,
                lengths=requested,
                limit=witness_cap + 1,
                node_budget=node_budget,
                cutoff=None,
                recorder=None,
            )
        except ScoringBackendError as error:
            if _is_timeout(error):
                raise ScoreTimeoutWithoutPartial(str(error)) from error
            raise
        finally:
            worker.timeout_seconds = previous_timeout
        elapsed_ns = time.perf_counter_ns() - started_ns
        if response.dominated:
            raise ScoreContractViolation("uncut component request returned dominated")
        by_length = {int(result.length): result for result in response.results}
        if set(by_length) != set(requested):
            raise ScoreContractViolation(
                "scorer omitted or added a forbidden length"
            )
        components: list[CycleComponentEvidence] = []
        for length in requested:
            result = by_length[length]
            raw_count = int(result.count)
            nodes_visited = int(result.nodes)
            result_elapsed_ns = int(result.elapsed_ns)
            if raw_count < 0 or nodes_visited < 0:
                raise ScoreContractViolation("scorer returned a negative counter")
            observed = min(raw_count, witness_cap)
            if raw_count >= witness_cap:
                status = EvidenceStatus.SATURATED_AT_CAP
                lower = upper = witness_cap
            elif bool(result.complete):
                status = EvidenceStatus.EXACT
                lower = upper = raw_count
            else:
                status = EvidenceStatus.SEARCH_BUDGET_EXHAUSTED
                lower, upper = raw_count, witness_cap
            components.append(
                CycleComponentEvidence(
                    forbidden_length=length,
                    observed_count=observed,
                    lower_bound=lower,
                    upper_bound=upper,
                    status=status,
                    node_budget=node_budget,
                    nodes_visited=nodes_visited,
                    wall_time_ns=(
                        result_elapsed_ns
                        if result_elapsed_ns >= 0
                        else elapsed_ns
                    ),
                    attempt_kind=attempt_kind,
                    backend_identity=self.identity,
                )
            )
        evidence = ScoreEvidence(
            graph_content_hash=graph_hash,
            order=graph.order,
            edge_count=len(graph.edges),
            witness_cap=witness_cap,
            components=tuple(components),
        )
        self.cache.put(cache_key, evidence)
        return evidence


def _is_timeout(error: BaseException) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        text = str(cause).lower()
        if "timeout" in text or "timed out" in text:
            return True
        cause = cause.__cause__
    return False


def merge_score_evidence(
    initial: ScoreEvidence,
    expanded: ScoreEvidence,
) -> ScoreEvidence:
    """Merge selected expanded components without weakening prior bounds."""

    identity = (
        initial.graph_content_hash,
        initial.order,
        initial.edge_count,
        initial.witness_cap,
    )
    if identity != (
        expanded.graph_content_hash,
        expanded.order,
        expanded.edge_count,
        expanded.witness_cap,
    ):
        raise ScoreContractViolation("cannot merge evidence for different graphs")
    merged = {
        component.forbidden_length: component
        for component in initial.components
    }
    for component in expanded.components:
        previous = merged.get(component.forbidden_length)
        if previous is None:
            raise ScoreContractViolation(
                "expanded evidence introduced an unrequested component"
            )
        if (
            component.lower_bound < previous.lower_bound
            or component.upper_bound > previous.upper_bound
        ):
            raise ScoreContractViolation("expanded evidence weakened a sound bound")
        merged[component.forbidden_length] = component
    return ScoreEvidence(
        graph_content_hash=initial.graph_content_hash,
        order=initial.order,
        edge_count=initial.edge_count,
        witness_cap=initial.witness_cap,
        components=tuple(merged[length] for length in sorted(merged)),
    )


def scorer_for_backend(backend: GraphBackend) -> ScoreEvidenceScorer:
    """Return only an explicit Native v3 evidence scorer; never infer a fallback."""

    if isinstance(backend, HegBackend):
        return HegScoreEvidenceAdapter(backend)
    if isinstance(backend, ScoreEvidenceScorer):
        return backend
    raise TypeError(
        "Native v3 backend has no explicit component-evidence scorer"
    )
