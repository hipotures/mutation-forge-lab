from __future__ import annotations

import hashlib
import importlib
import random
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutation_forge.backends.base import (
    DeepProposalProfileRecorder,
    InvalidRewriteError,
    ProposalTimingRecorder,
    ScoreProfileRecorder,
    ScoringBackendError,
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
PREPARED_GRAPH_CACHE_SIZE = 2
HEG_GRAPH_MODE = "unrestricted_min_degree_3"
HEG_GRAPH_MODES = frozenset(
    {
        "cubic_first",
        "minimal_structure_mixed_degree",
        "unrestricted_min_degree_3",
    }
)


@dataclass(slots=True)
class _PreparedGraph:
    graph: Any
    validation: GraphValidation | None = None
    validation_context: Any | None = None


@dataclass(slots=True)
class _PreparedProposal:
    source: GraphState
    rewrite: RewritePlan
    removed_edges: tuple[tuple[int, int], ...]
    added_edges: tuple[tuple[int, int], ...]
    operator_family: str
    evaluation: int
    graph: Any


class HegBackend:
    backend_id = "heg-erdos-gyarfas-min-degree-3"

    def __init__(
        self,
        repo: Path,
        *,
        score_timeout_seconds: float = 2.0,
        mutation_witness_cache_enabled: bool = True,
        score_cutoff_enabled: bool = True,
        prepared_graph_cache_enabled: bool = True,
        prepared_proposal_handoff_enabled: bool = True,
        score_longest_first_enabled: bool = True,
        score_compact_dominated_enabled: bool = True,
        score_prepared_request_cache_enabled: bool = True,
        graph_mode: str = HEG_GRAPH_MODE,
    ) -> None:
        if graph_mode not in HEG_GRAPH_MODES:
            raise ValueError(f"unsupported HEG graph mode: {graph_mode}")
        self.graph_mode = graph_mode
        self.backend_id = f"heg-erdos-gyarfas-{graph_mode}"
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
        target_base = importlib.import_module("sglab.targets.base")
        self._plugin = target.PLUGIN
        self._worker_class = worker_module.PersistentScoreWorker
        self._worker_error = worker_module.ScoreWorkerError
        self._validation_context_class = target_base.GraphValidationContext
        self._validation_result_class = target_base.ValidationResult
        self._worker: Any | None = None
        self._score_worker_name: str | None = None
        self._score_worker_link_dir: tempfile.TemporaryDirectory[str] | None = None
        self._worker_disabled = False
        self.score_implementation = "heg-cpp-score-worker"
        self._score_timeout_seconds = score_timeout_seconds
        self._score_longest_first_enabled = score_longest_first_enabled
        self._score_compact_dominated_enabled = score_compact_dominated_enabled
        self._score_prepared_request_cache_enabled = score_prepared_request_cache_enabled
        context_factory = getattr(self._plugin, "new_mutation_context", None)
        if context_factory is None:
            raise RuntimeError(
                "configured HEG repository does not support mutation witness caching"
            )
        self._mutation_witness_cache_enabled = mutation_witness_cache_enabled
        self._mutation_witness_context = context_factory(
            cache_enabled=mutation_witness_cache_enabled
        )
        self._proposal_graph_state: GraphState | None = None
        self._proposal_heg_graph: Any | None = None
        self._score_cutoff_enabled = score_cutoff_enabled
        self._prepared_graph_cache_enabled = prepared_graph_cache_enabled
        self._prepared_graphs: OrderedDict[GraphState, _PreparedGraph] = OrderedDict()
        self._prepared_proposal_handoff_enabled = prepared_proposal_handoff_enabled
        self._prepared_proposal: _PreparedProposal | None = None
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

    @staticmethod
    def _record(
        recorder: ScoreProfileRecorder | None,
        event: str,
        payload: Mapping[str, int],
    ) -> None:
        if recorder is not None:
            recorder(event, payload)

    def _store_prepared(
        self,
        graph: GraphState,
        prepared: _PreparedGraph,
    ) -> None:
        if not self._prepared_graph_cache_enabled:
            return
        self._prepared_graphs.pop(graph, None)
        self._prepared_graphs[graph] = prepared
        while len(self._prepared_graphs) > PREPARED_GRAPH_CACHE_SIZE:
            self._prepared_graphs.popitem(last=False)

    def _cached_prepared(self, graph: GraphState) -> _PreparedGraph | None:
        if not self._prepared_graph_cache_enabled:
            return None
        prepared = self._prepared_graphs.pop(graph, None)
        if prepared is not None:
            self._prepared_graphs[graph] = prepared
        return prepared

    def _prepare(
        self,
        graph: GraphState,
        recorder: ScoreProfileRecorder | None = None,
    ) -> _PreparedGraph:
        prepared = self._cached_prepared(graph)
        self._record(
            recorder,
            "prepared_cache",
            {
                "lookups": int(self._prepared_graph_cache_enabled),
                "hits": int(prepared is not None),
                "misses": int(self._prepared_graph_cache_enabled and prepared is None),
            },
        )
        if prepared is not None:
            return prepared

        started_ns = time.perf_counter_ns() if recorder is not None else 0
        heg_graph = self._to_heg(graph)
        self._record(
            recorder,
            "graph_materialization",
            {
                "calls": 1,
                "elapsed_ns": (time.perf_counter_ns() - started_ns if recorder is not None else 0),
            },
        )
        prepared = _PreparedGraph(heg_graph)
        self._store_prepared(graph, prepared)
        return prepared

    def _validate_prepared(
        self,
        prepared: _PreparedGraph,
        recorder: ScoreProfileRecorder | None = None,
    ) -> GraphValidation:
        if prepared.validation is not None:
            self._record(
                recorder,
                "validation_cache",
                {"lookups": 1, "hits": 1, "misses": 0},
            )
            return prepared.validation
        self._record(
            recorder,
            "validation_cache",
            {"lookups": 1, "hits": 0, "misses": 1},
        )
        started_ns = time.perf_counter_ns() if recorder is not None else 0
        result = self._plugin.validate_graph(prepared.graph)
        errors: list[str] = [] if result.valid else [result.message]
        prepared.validation = GraphValidation(not errors, tuple(errors))
        prepared.validation_context = self._validation_context_class(
            prepared.graph,
            result,
        )
        self._record(
            recorder,
            "validation",
            {
                "calls": 1,
                "elapsed_ns": (time.perf_counter_ns() - started_ns if recorder is not None else 0),
            },
        )
        return prepared.validation

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        self._prepared_proposal = None
        graph = self._plugin.generate_seed(
            random.Random(seed), {"order": order, "mode": self.graph_mode}
        )
        state = self._from_heg(graph)
        self._store_prepared(state, _PreparedGraph(graph))
        return state

    def validate(self, graph: GraphState) -> GraphValidation:
        try:
            prepared = self._prepare(graph)
        except (TypeError, ValueError) as error:
            return GraphValidation(False, (str(error),))
        return self._validate_prepared(prepared)

    def _score_worker(self) -> Any:
        if self._worker_disabled:
            raise self._worker_error("score worker disabled after protocol failure")
        if self._worker is None:
            self._worker = self._worker_class(
                timeout_seconds=self._score_timeout_seconds,
                memory_limit_bytes=64 * 1024 * 1024,
                cutoff_longest_first=self._score_longest_first_enabled,
                prepared_request_cache_enabled=(self._score_prepared_request_cache_enabled),
            )
            self._configure_score_worker_binary()
        return self._worker

    def set_score_worker_name(self, name: str) -> None:
        """Set the Linux process name used by this backend's scorer."""

        if not isinstance(name, str) or not name:
            raise ValueError("score-worker process name must be non-empty")
        # Linux exposes at most 15 bytes through /proc/<pid>/comm.  The
        # evaluator names are ASCII, so truncating characters is sufficient.
        self._score_worker_name = name[:15]
        self._configure_score_worker_binary()

    def _configure_score_worker_binary(self) -> None:
        name = self._score_worker_name
        worker = self._worker
        if (
            name is None
            or worker is None
            or not sys.platform.startswith("linux")
            or getattr(worker, "process", None) is not None
        ):
            return
        binary = getattr(worker, "binary", None)
        if not isinstance(binary, Path) or not binary.is_file():
            return
        target = binary.resolve()
        if self._score_worker_link_dir is not None:
            self._score_worker_link_dir.cleanup()
        self._score_worker_link_dir = tempfile.TemporaryDirectory(
            prefix="mforge-score-worker-"
        )
        named_binary = Path(self._score_worker_link_dir.name) / name
        named_binary.symlink_to(target)
        worker.binary = named_binary

    def _cutoff_tuple(
        self,
        graph: GraphState,
        cutoff: GraphScore | None,
        recorder: ScoreProfileRecorder | None,
    ) -> tuple[int, int, int] | None:
        if cutoff is None:
            return None
        key = cutoff.ordering_key
        supported = (
            self._score_cutoff_enabled
            and cutoff.valid
            and cutoff.total_capped_witnesses > 0
            and graph.order == 30
            and len(graph.edges) == 45
            and len(key) == 5
            and key[0] == 0
            and key[1] == cutoff.total_capped_witnesses
            and key[2] == cutoff.weighted_penalty
            and key[3] == 0
            and key[4] == len(graph.edges)
        )
        self._record(
            recorder,
            "cutoff",
            {
                "requests": 1,
                "applied": int(supported),
                "disabled": int(not supported),
            },
        )
        if not supported:
            return None
        return int(key[1]), int(key[2]), int(key[4])

    def _record_worker_response(
        self,
        results: tuple[Any, ...],
        dominated: bool,
        timing: Any | None,
        recorder: ScoreProfileRecorder | None,
    ) -> None:
        if recorder is None:
            return
        payload: dict[str, int] = {
            "calls": 1,
            "full_results": int(not dominated),
            "dominated_results": int(dominated),
            "request_packing_ns": (int(timing.request_packing_ns) if timing is not None else 0),
            "request_write_ns": (int(timing.request_write_ns) if timing is not None else 0),
            "response_read_ns": (int(timing.response_read_ns) if timing is not None else 0),
            "response_parsing_ns": (int(timing.response_parsing_ns) if timing is not None else 0),
            "worker_roundtrip_ns": (int(timing.worker_roundtrip_ns) if timing is not None else 0),
        }
        for index, result in enumerate(results):
            prefix = f"cycle_{int(result.length)}"
            payload[f"{prefix}_calls"] = 1
            payload[f"{prefix}_elapsed_ns"] = int(result.elapsed_ns)
            payload[f"{prefix}_nodes"] = int(result.nodes)
            payload[f"{prefix}_complete"] = int(result.complete)
            payload[f"{prefix}_cutoff"] = int(dominated and index == len(results) - 1)
        self._record(recorder, "worker_response", payload)

    def _worker_response(
        self,
        graph: Any,
        *,
        lengths: tuple[int, ...],
        limit: int,
        node_budget: int,
        cutoff: tuple[int, int, int] | None,
        recorder: ScoreProfileRecorder | None,
    ) -> Any:
        if self._worker_disabled:
            raise ScoringBackendError(
                "mandatory C++ score worker is disabled after a prior failure"
            )
        last_error: BaseException | None = None
        for attempt in range(2):
            started_ns = time.perf_counter_ns() if recorder is not None else 0
            try:
                worker = self._score_worker()
                if self._score_worker_name is not None:
                    # Start the child before the first request so top/ps sees
                    # its evaluator identity throughout the scoring call.
                    worker.start()
                response = worker.score(
                    graph,
                    lengths=lengths,
                    limit=limit,
                    node_budget=node_budget,
                    cutoff=cutoff,
                    cutoff_inclusive=cutoff is not None,
                    compact_dominated=(
                        self._score_compact_dominated_enabled
                        and recorder is None
                        and cutoff is not None
                    ),
                    profile_timing=recorder is not None,
                )
            except self._worker_error as error:
                last_error = error
                self._record(
                    recorder,
                    "worker_failure",
                    {
                        "calls": 1,
                        "elapsed_ns": (
                            time.perf_counter_ns() - started_ns if recorder is not None else 0
                        ),
                    },
                )
                if attempt == 0 and self._worker is not None:
                    self._record(
                        recorder,
                        "worker_restart",
                        {"attempts": 1, "successes": 0},
                    )
                    try:
                        self._worker.restart()
                    except self._worker_error:
                        self._record(
                            recorder,
                            "worker_failure",
                            {"calls": 1, "elapsed_ns": 0},
                        )
                        break
                    self._record(
                        recorder,
                        "worker_restart",
                        {"attempts": 0, "successes": 1},
                    )
                    continue
                break
            self._record_worker_response(
                response.results,
                bool(response.dominated),
                response.timing,
                recorder,
            )
            return response

        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if self._score_worker_link_dir is not None:
            self._score_worker_link_dir.cleanup()
            self._score_worker_link_dir = None
        self._worker_disabled = True
        detail = str(last_error) if last_error is not None else "unknown worker failure"
        raise ScoringBackendError(
            f"mandatory C++ score worker failed after restart: {detail}"
        ) from last_error

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: ScoreProfileRecorder | None = None,
    ) -> GraphScore | None:
        prepared = self._prepare(graph, record_profile)
        validation = self._validate_prepared(prepared, record_profile)
        if not validation.valid:
            return GraphScore(False, (), 10**9, 10**9, True, (1, 10**9, 10**9))
        heg_graph = prepared.graph
        lengths = tuple(self._plugin.forbidden_lengths(heg_graph.n))
        limit = witness_cap + 1
        node_budget = max(4096, min(50_000, witness_cap * 1024))
        cutoff_tuple = self._cutoff_tuple(graph, cutoff, record_profile)
        response = self._worker_response(
            heg_graph,
            lengths=lengths,
            limit=limit,
            node_budget=node_budget,
            cutoff=cutoff_tuple,
            recorder=record_profile,
        )
        if response.dominated:
            return None
        cycle_results = response.results
        assembly_started_ns = time.perf_counter_ns() if record_profile is not None else 0
        assert prepared.validation_context is not None
        result = self._plugin.score_from_cycle_counts(
            heg_graph,
            witness_cap,
            cycle_results,
            None,
            validation_context=prepared.validation_context,
        )
        self._record(
            record_profile,
            "score_assembly",
            {
                "calls": 1,
                "elapsed_ns": (
                    time.perf_counter_ns() - assembly_started_ns
                    if record_profile is not None
                    else 0
                ),
            },
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

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        return tuple(int(length) for length in self._plugin.forbidden_lengths(order))

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        result = self._plugin.exact_verify(self._prepare(graph).graph)
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
        canonical = self._plugin.canonical_key(self._prepare(graph).graph)
        return hashlib.sha256(canonical).hexdigest()

    def state_hash(self, graph: GraphState) -> str:
        return str(self._prepare(graph).graph.stable_hash())

    def serialize_graph6(self, graph: GraphState) -> str:
        return str(self._prepare(graph).graph.to_graph6())

    def deserialize_graph6(self, value: str) -> GraphState:
        self._prepared_proposal = None
        heg_graph = self._model.BitGraph.from_graph6(value)
        graph = self._from_heg(heg_graph)
        self._store_prepared(graph, _PreparedGraph(heg_graph))
        return graph

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: ScoreProfileRecorder | None = None,
    ) -> GraphState:
        if len(rewrite.removed_edges) > 4 or len(rewrite.added_edges) > 4:
            raise InvalidRewriteError(
                "rewrites are limited to four removed and added edges"
            )
        if len(set(rewrite.removed_edges)) != len(rewrite.removed_edges):
            raise InvalidRewriteError("rewrite contains duplicate removed edges")
        if len(set(rewrite.added_edges)) != len(rewrite.added_edges):
            raise InvalidRewriteError("rewrite contains duplicate added edges")
        prepared_proposal = self._prepared_proposal
        self._prepared_proposal = None
        if (
            prepared_proposal is not None
            and graph is prepared_proposal.source
            and rewrite is prepared_proposal.rewrite
            and rewrite.removed_edges == prepared_proposal.removed_edges
            and rewrite.added_edges == prepared_proposal.added_edges
            and rewrite.operator_family == prepared_proposal.operator_family
            and len(rewrite.metadata) == 1
            and rewrite.metadata.get("evaluation") == prepared_proposal.evaluation
        ):
            candidate = self._from_heg(prepared_proposal.graph)
            if candidate.order != graph.order:
                raise ValueError("rewrite changed graph order")
            validation_result = self._validation_result_class(
                True,
                "valid HEG mutation result",
            )
            self._store_prepared(
                candidate,
                _PreparedGraph(
                    prepared_proposal.graph,
                    GraphValidation(True),
                    self._validation_context_class(
                        prepared_proposal.graph,
                        validation_result,
                    ),
                ),
            )
            return candidate
        current = set(graph.edges)
        removed = set(rewrite.removed_edges)
        if not removed.issubset(current):
            raise InvalidRewriteError("rewrite removes a missing edge")
        remaining = current.difference(removed)
        if any(u == v for u, v in rewrite.added_edges):
            raise InvalidRewriteError("rewrite adds a loop")
        if set(rewrite.added_edges).intersection(remaining):
            raise InvalidRewriteError("rewrite adds an existing edge")
        candidate = GraphState(
            graph.order,
            tuple(sorted(remaining.union(rewrite.added_edges))),
        )
        if candidate.order != graph.order:
            raise ValueError("rewrite changed graph order")
        prepared = self._prepare(candidate, record_score_profile)
        validation = self._validate_prepared(
            prepared,
            record_score_profile,
        )
        if not validation.valid:
            raise InvalidRewriteError("; ".join(validation.errors))
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
        self._prepared_proposal = None
        phase_started_ns = time.perf_counter_ns() if record_timing is not None else 0
        try:
            heg_operator = OPERATOR_MAP[operator_family]
        except KeyError as error:
            raise ValueError(f"unsupported HEG operator family: {operator_family}") from error
        rng = random.Random((policy_seed << 32) ^ evaluation)
        if record_timing is not None:
            record_timing("rng_setup", time.perf_counter_ns() - phase_started_ns)

        if graph is self._proposal_graph_state:
            assert self._proposal_heg_graph is not None
            heg_graph = self._proposal_heg_graph
        else:
            prepared = self._cached_prepared(graph)
            if prepared is not None:
                heg_graph = prepared.graph
            else:
                phase_started_ns = time.perf_counter_ns() if record_timing is not None else 0
                heg_graph = self._to_heg(graph)
                self._store_prepared(graph, _PreparedGraph(heg_graph))
                if record_timing is not None:
                    record_timing(
                        "graph_materialization",
                        time.perf_counter_ns() - phase_started_ns,
                    )
            self._proposal_graph_state = graph
            self._proposal_heg_graph = heg_graph

        mutation_config: dict[str, Any] = {
            "mode": self.graph_mode,
            "mutation_operator": heg_operator,
        }
        if heg_operator == "forbidden_cycle_break_switch":
            mutation_config["forbidden_witness_context"] = self._mutation_witness_context
        deep_profile = None
        if record_deep_profile is not None:
            profile_factory = getattr(self._plugin, "new_mutation_profile", None)
            if profile_factory is None:
                raise RuntimeError(
                    "configured HEG repository does not support deep mutation profiling"
                )
            deep_profile = profile_factory()
            mutation_config["mutation_profile"] = deep_profile

        measure_operator = record_timing is not None or record_deep_profile is not None
        phase_started_ns = time.perf_counter_ns() if measure_operator else 0
        result = self._plugin.mutate_with_delta(
            heg_graph,
            rng,
            mutation_config,
        )
        operator_elapsed_ns = time.perf_counter_ns() - phase_started_ns if measure_operator else 0
        if record_timing is not None:
            record_timing("operator_search", operator_elapsed_ns)
        if record_deep_profile is not None:
            assert deep_profile is not None
            deep_profile.record_operator(heg_operator, operator_elapsed_ns)
            record_deep_profile(
                operator_family,
                deep_profile.payload(cache_enabled=self._mutation_witness_cache_enabled),
            )

        phase_started_ns = time.perf_counter_ns() if record_timing is not None else 0
        rewrite = RewritePlan(
            removed_edges=tuple(result.removed_edges),
            added_edges=tuple(result.added_edges),
            operator_family=operator_family,
            metadata={"evaluation": evaluation},
        )
        if self._prepared_proposal_handoff_enabled:
            self._prepared_proposal = _PreparedProposal(
                source=graph,
                rewrite=rewrite,
                removed_edges=rewrite.removed_edges,
                added_edges=rewrite.added_edges,
                operator_family=rewrite.operator_family,
                evaluation=evaluation,
                graph=result.graph,
            )
        if record_timing is not None:
            record_timing(
                "proposal_packaging",
                time.perf_counter_ns() - phase_started_ns,
            )
        return rewrite

    def close(self) -> None:
        self._mutation_witness_context.invalidate()
        self._proposal_graph_state = None
        self._proposal_heg_graph = None
        self._prepared_proposal = None
        self._prepared_graphs.clear()
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        if self._score_worker_link_dir is not None:
            self._score_worker_link_dir.cleanup()
            self._score_worker_link_dir = None
