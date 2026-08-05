"""Native v3 component-evidence adapter for the mandatory HEG C++ scorer."""

from __future__ import annotations

import hashlib
import platform
import time
from collections.abc import Iterable
from pathlib import Path

from mutation_forge.backends.base import ScoringBackendError
from mutation_forge.backends.heg import HegBackend
from mutation_forge.models import GraphState

from .scoring import (
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
from .verification import graph_content_hash

INITIAL_NODE_BUDGET = 50_000
EXPANDED_NODE_BUDGET = 200_000
INITIAL_SCORE_WALL_SECONDS = 5.0
EXPANDED_SCORE_WALL_SECONDS = 20.0


class ScoreContractViolation(RuntimeError):
    """The scorer returned internally inconsistent component evidence."""


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
    worker = backend._score_worker()  # noqa: SLF001 - local protocol adapter boundary
    binary = Path(worker.binary)
    if not binary.is_file():
        raise ScoringBackendError(f"mandatory C++ score worker is missing: {binary}")
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    return BackendIdentity(
        backend_id=backend.backend_id,
        heg_commit=backend.commit,
        source_tree_sha256=_source_tree_hash(backend.repo),
        binary_sha256=binary_sha256,
        compiler_identity="content-addressed-binary",
        build_flags=(),
        platform=platform.platform(),
        architecture=platform.machine(),
        score_protocol_id=SCORE_PROTOCOL_ID,
    )


class HegScoreEvidenceAdapter:
    """Expose selected-length sound evidence without a Python score fallback."""

    def __init__(self, backend: HegBackend) -> None:
        self.backend = backend
        self.identity = backend_identity(backend)
        self.cache = ScoreEvidenceCache()
        self.raw_graph_score_calls = 0
        self.unique_graph_scores = 0

    def score(
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
            raise ValueError(f"cannot score an invalid graph: {validation.errors}")
        target_lengths = self.backend.target_forbidden_lengths(graph.order)
        requested = (
            target_lengths if forbidden_lengths is None else tuple(sorted(set(forbidden_lengths)))
        )
        if not requested or not set(requested).issubset(target_lengths):
            raise ValueError("requested forbidden lengths are not valid for this graph order")
        node_budget = (
            INITIAL_NODE_BUDGET if attempt_kind is AttemptKind.INITIAL else EXPANDED_NODE_BUDGET
        )
        self.raw_graph_score_calls += 1
        content_hash = graph_content_hash(graph)
        cache_key = ScoreEvidenceCacheKey(
            content_hash,
            requested,
            witness_cap,
            node_budget,
            attempt_kind,
            SCORE_PROTOCOL_ID,
            self.identity.canonical_key(),
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        self.unique_graph_scores += 1
        worker = self.backend._score_worker()  # noqa: SLF001
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
            cause: BaseException | None = error
            timeout = False
            while cause is not None:
                if "timeout" in str(cause).lower() or "timed out" in str(cause).lower():
                    timeout = True
                    break
                cause = cause.__cause__
            if timeout:
                raise ScoreTimeoutWithoutPartial(str(error)) from error
            raise
        wall_time_ns = time.perf_counter_ns() - started_ns
        if response.dominated:
            raise ScoreContractViolation("uncut component request returned dominated")
        by_length = {int(result.length): result for result in response.results}
        if set(by_length) != set(requested):
            raise ScoreContractViolation("scorer omitted or added a forbidden length")
        components: list[CycleComponentEvidence] = []
        for length in requested:
            result = by_length[length]
            raw_count = int(result.count)
            if raw_count < 0 or int(result.nodes) < 0:
                raise ScoreContractViolation("scorer returned a negative counter")
            capped = min(raw_count, witness_cap)
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
                    observed_count=capped,
                    lower_bound=lower,
                    upper_bound=upper,
                    status=status,
                    node_budget=node_budget,
                    nodes_visited=int(result.nodes),
                    wall_time_ns=(
                        int(result.elapsed_ns) if int(result.elapsed_ns) >= 0 else wall_time_ns
                    ),
                    attempt_kind=attempt_kind,
                    backend_identity=self.identity,
                )
            )
        evidence = ScoreEvidence(
            graph_content_hash=content_hash,
            order=graph.order,
            edge_count=len(graph.edges),
            witness_cap=witness_cap,
            components=tuple(components),
        )
        self.cache.put(cache_key, evidence)
        return evidence


def merge_score_evidence(
    initial: ScoreEvidence,
    expanded: ScoreEvidence,
) -> ScoreEvidence:
    """Merge one selected-length expanded attempt into prior sound evidence."""

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
    merged = {component.forbidden_length: component for component in initial.components}
    for component in expanded.components:
        previous = merged.get(component.forbidden_length)
        if previous is not None and (
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
