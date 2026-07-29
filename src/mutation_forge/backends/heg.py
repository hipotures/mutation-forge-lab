from __future__ import annotations

import hashlib
import importlib
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mutation_forge.backends.base import (
    DeepProposalProfileRecorder,
    ProposalTimingRecorder,
)
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)

OPERATOR_MAP = {
    "heg_uniform_two_switch": "uniform_two_edge_switch",
    "heg_forbidden_cycle_break": "forbidden_cycle_break_switch",
}


class HegBackend:
    backend_id = "heg-erdos-gyarfas-connected-cubic"

    def __init__(self, repo: Path, *, score_timeout_seconds: float = 2.0) -> None:
        self.repo = repo.resolve()
        source = self.repo / "src"
        if not (source / "sglab").is_dir():
            raise ValueError(f"HEG Python package not found at {source}")
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        self._model = importlib.import_module("sglab.model")
        target = importlib.import_module("sglab.targets.erdos_gyarfas")
        worker_module = importlib.import_module("sglab.score_worker")
        self._plugin = target.PLUGIN
        self._worker_class = worker_module.PersistentScoreWorker
        self._worker_error = worker_module.ScoreWorkerError
        self._cycle_count_result = worker_module.CycleCountResult
        self._worker: Any | None = None
        self._worker_disabled = False
        self.score_implementation = "heg-cpp-score-worker"
        self._score_timeout_seconds = score_timeout_seconds
        self.commit = self._git("rev-parse", "HEAD")
        self.dirty = bool(self._git("status", "--short"))

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    def _to_heg(self, graph: GraphState) -> Any:
        return self._model.BitGraph.from_edges(graph.order, graph.edges)

    def _from_heg(self, graph: Any) -> GraphState:
        return GraphState(order=graph.n, edges=tuple(graph.edges()))

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        graph = self._plugin.generate_seed(
            random.Random(seed), {"order": order, "mode": "cubic_first"}
        )
        return self._from_heg(graph)

    def validate(self, graph: GraphState) -> GraphValidation:
        try:
            heg_graph = self._to_heg(graph)
        except (TypeError, ValueError) as error:
            return GraphValidation(False, (str(error),))
        result = self._plugin.validate_graph(heg_graph)
        errors: list[str] = [] if result.valid else [result.message]
        if any(heg_graph.degree(vertex) != 3 for vertex in range(heg_graph.n)):
            errors.append("connected-cubic mode requires degree 3 at every vertex")
        return GraphValidation(not errors, tuple(errors))

    def _score_worker(self) -> Any:
        if self._worker_disabled:
            raise self._worker_error("score worker disabled after protocol failure")
        if self._worker is None:
            self._worker = self._worker_class(
                timeout_seconds=self._score_timeout_seconds,
                memory_limit_bytes=64 * 1024 * 1024,
            )
        return self._worker

    def _reference_cycle_counts(
        self, graph: Any, lengths: tuple[int, ...], *, limit: int, node_budget: int
    ) -> tuple[Any, ...]:
        results = []
        for length in lengths:
            started = time.perf_counter_ns()
            witnesses, complete = self._model.find_cycles_of_length_bounded(
                graph, length, limit, node_budget
            )
            results.append(
                self._cycle_count_result(
                    length=length,
                    count=len(witnesses),
                    complete=complete,
                    nodes=0,
                    elapsed_ns=time.perf_counter_ns() - started,
                )
            )
        self.score_implementation = "heg-python-bounded-reference"
        return tuple(results)

    def score(self, graph: GraphState, *, witness_cap: int) -> GraphScore:
        heg_graph = self._to_heg(graph)
        validation = self.validate(graph)
        if not validation.valid:
            return GraphScore(False, (), 10**9, 10**9, True, (1, 10**9, 10**9))
        lengths = tuple(self._plugin.forbidden_lengths(heg_graph.n))
        limit = witness_cap + 1
        node_budget = max(4096, min(50_000, witness_cap * 1024))
        try:
            response = self._score_worker().score(
                heg_graph,
                lengths=lengths,
                limit=limit,
                node_budget=node_budget,
            )
            if response.dominated:
                raise RuntimeError(
                    "HEG score worker returned an unexpected dominated response"
                )
            cycle_results = response.results
        except self._worker_error:
            if self._worker is not None:
                self._worker.close()
                self._worker = None
            self._worker_disabled = True
            cycle_results = self._reference_cycle_counts(
                heg_graph,
                lengths,
                limit=limit,
                node_budget=node_budget,
            )
        result = self._plugin.score_from_cycle_counts(
            heg_graph, witness_cap, cycle_results, None
        )
        counts = tuple((int(length), int(count)) for length, count in result.witness_counts)
        return GraphScore(
            valid=bool(result.valid),
            capped_cycle_counts=counts,
            total_capped_witnesses=sum(count for _, count in counts),
            weighted_penalty=int(result.weighted_penalty),
            complete=bool(result.complete),
            ordering_key=tuple(int(item) for item in result.ordering_key),
        )

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        result = self._plugin.exact_verify(self._to_heg(graph))
        witnesses = tuple(
            (str(witness.kind), tuple(int(vertex) for vertex in witness.vertices))
            for witness in result.witnesses
        )
        return ExactVerification(
            status=str(result.status),
            complete=bool(result.complete),
            message=str(result.message),
            implementation=str(result.implementation),
            witnesses=witnesses,
            elapsed_seconds=float(result.elapsed_seconds),
        )

    def canonical_hash(self, graph: GraphState) -> str:
        canonical = self._plugin.canonical_key(self._to_heg(graph))
        return hashlib.sha256(canonical).hexdigest()

    def state_hash(self, graph: GraphState) -> str:
        return str(self._to_heg(graph).stable_hash())

    def serialize_graph6(self, graph: GraphState) -> str:
        return str(self._to_heg(graph).to_graph6())

    def deserialize_graph6(self, value: str) -> GraphState:
        return self._from_heg(self._model.BitGraph.from_graph6(value))

    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState:
        if len(rewrite.removed_edges) > 2 or len(rewrite.added_edges) > 2:
            raise ValueError("Stage 1 rewrites are limited to two removed and added edges")
        if len(set(rewrite.removed_edges)) != len(rewrite.removed_edges):
            raise ValueError("rewrite contains duplicate removed edges")
        if len(set(rewrite.added_edges)) != len(rewrite.added_edges):
            raise ValueError("rewrite contains duplicate added edges")
        current = set(graph.edges)
        removed = set(rewrite.removed_edges)
        if not removed.issubset(current):
            raise ValueError("rewrite removes a missing edge")
        remaining = current.difference(removed)
        if any(u == v for u, v in rewrite.added_edges):
            raise ValueError("rewrite adds a loop")
        if set(rewrite.added_edges).intersection(remaining):
            raise ValueError("rewrite adds an existing edge")
        candidate = GraphState(
            graph.order,
            tuple(sorted(remaining.union(rewrite.added_edges))),
        )
        if candidate.order != graph.order:
            raise ValueError("rewrite changed graph order")
        validation = self.validate(candidate)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return candidate

    def propose_rewrite(
        self,
        graph: GraphState,
        *,
        operator_family: str,
        policy_seed: int,
        evaluation: int,
        record_timing: ProposalTimingRecorder | None = None,
        record_deep_profile: DeepProposalProfileRecorder | None = None,
    ) -> RewritePlan:
        phase_started_ns = time.perf_counter_ns() if record_timing is not None else 0
        try:
            heg_operator = OPERATOR_MAP[operator_family]
        except KeyError as error:
            raise ValueError(f"unsupported HEG operator family: {operator_family}") from error
        rng = random.Random((policy_seed << 32) ^ evaluation)
        if record_timing is not None:
            record_timing("rng_setup", time.perf_counter_ns() - phase_started_ns)

        phase_started_ns = time.perf_counter_ns() if record_timing is not None else 0
        heg_graph = self._to_heg(graph)
        if record_timing is not None:
            record_timing(
                "graph_materialization",
                time.perf_counter_ns() - phase_started_ns,
            )

        mutation_config: dict[str, Any] = {
            "mode": "cubic_first",
            "mutation_operator": heg_operator,
        }
        deep_profile = None
        if record_deep_profile is not None:
            profile_factory = getattr(self._plugin, "new_mutation_profile", None)
            if profile_factory is None:
                raise RuntimeError(
                    "configured HEG repository does not support deep mutation profiling"
                )
            deep_profile = profile_factory()
            mutation_config["mutation_profile"] = deep_profile

        measure_operator = (
            record_timing is not None or record_deep_profile is not None
        )
        phase_started_ns = time.perf_counter_ns() if measure_operator else 0
        result = self._plugin.mutate_with_delta(
            heg_graph,
            rng,
            mutation_config,
        )
        operator_elapsed_ns = (
            time.perf_counter_ns() - phase_started_ns if measure_operator else 0
        )
        if record_timing is not None:
            record_timing("operator_search", operator_elapsed_ns)
        if record_deep_profile is not None:
            assert deep_profile is not None
            deep_profile.record_operator(heg_operator, operator_elapsed_ns)
            record_deep_profile(
                operator_family,
                deep_profile.payload(cache_enabled=False),
            )

        phase_started_ns = time.perf_counter_ns() if record_timing is not None else 0
        rewrite = RewritePlan(
            removed_edges=tuple(result.removed_edges),
            added_edges=tuple(result.added_edges),
            operator_family=operator_family,
            metadata={"evaluation": evaluation},
        )
        if record_timing is not None:
            record_timing(
                "proposal_packaging",
                time.perf_counter_ns() - phase_started_ns,
            )
        return rewrite

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
