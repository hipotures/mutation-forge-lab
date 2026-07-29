from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
type Edge = tuple[int, int]


def normalized_edge(edge: Edge) -> Edge:
    u, v = edge
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True, slots=True)
class GraphState:
    order: int
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        normalized = tuple(sorted(normalized_edge(edge) for edge in self.edges))
        if normalized != self.edges:
            object.__setattr__(self, "edges", normalized)


@dataclass(frozen=True, slots=True)
class GraphValidation:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExactVerification:
    status: str
    complete: bool
    message: str
    implementation: str
    witnesses: tuple[tuple[str, tuple[int, ...]], ...] = ()
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RewritePlan:
    removed_edges: tuple[Edge, ...]
    added_edges: tuple[Edge, ...]
    operator_family: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphScore:
    valid: bool
    capped_cycle_counts: tuple[tuple[int, int], ...]
    total_capped_witnesses: int
    weighted_penalty: int
    complete: bool
    ordering_key: tuple[int, ...]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "valid": self.valid,
            "capped_cycle_counts": [list(item) for item in self.capped_cycle_counts],
            "total_capped_witnesses": self.total_capped_witnesses,
            "weighted_penalty": self.weighted_penalty,
            "complete": self.complete,
            "ordering_key": list(self.ordering_key),
        }


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    entry_id: str
    order: int
    graph_seed: int
    graph6: str
    graph_hash: str
    generator_version: str
    backend_id: str
    heg_commit: str
    split: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "entry_id": self.entry_id,
            "order": self.order,
            "graph_seed": self.graph_seed,
            "graph6": self.graph6,
            "graph_hash": self.graph_hash,
            "generator_version": self.generator_version,
            "backend_id": self.backend_id,
            "heg_commit": self.heg_commit,
            "split": self.split,
        }


@dataclass(frozen=True, slots=True)
class EpisodeTimingProfile:
    measured_total_ns: int
    scoring_ns: int
    proposal_generation_ns: int
    rewrite_application_ns: int
    duplicate_detection_ns: int
    controller_ns: int
    exact_verification_ns: int
    progress_reporting_ns: int
    finalization_ns: int
    proposal_rng_setup_ns: int
    proposal_graph_materialization_ns: int
    proposal_operator_search_ns: int
    proposal_packaging_ns: int
    proposal_generation_calls: int
    proposal_rng_setup_calls: int
    proposal_graph_materialization_calls: int
    proposal_operator_search_calls: int
    proposal_packaging_calls: int

    def phase_nanoseconds(self) -> dict[str, int]:
        return {
            "scoring": self.scoring_ns,
            "proposal_generation": self.proposal_generation_ns,
            "rewrite_application": self.rewrite_application_ns,
            "duplicate_detection": self.duplicate_detection_ns,
            "controller": self.controller_ns,
            "exact_verification": self.exact_verification_ns,
            "progress_reporting": self.progress_reporting_ns,
            "finalization": self.finalization_ns,
        }

    def proposal_phase_nanoseconds(self) -> dict[str, int]:
        return {
            "rng_setup": self.proposal_rng_setup_ns,
            "graph_materialization": self.proposal_graph_materialization_ns,
            "operator_search": self.proposal_operator_search_ns,
            "proposal_packaging": self.proposal_packaging_ns,
        }

    def proposal_breakdown_nanoseconds(self) -> dict[str, int]:
        phases = self.proposal_phase_nanoseconds()
        phases["other"] = max(
            0,
            self.proposal_generation_ns - sum(phases.values()),
        )
        return phases

    def proposal_phase_calls(self) -> dict[str, int]:
        return {
            "rng_setup": self.proposal_rng_setup_calls,
            "graph_materialization": self.proposal_graph_materialization_calls,
            "operator_search": self.proposal_operator_search_calls,
            "proposal_packaging": self.proposal_packaging_calls,
        }

    @property
    def accounted_ns(self) -> int:
        return sum(self.phase_nanoseconds().values())

    @property
    def unattributed_ns(self) -> int:
        return max(0, self.measured_total_ns - self.accounted_ns)

    def as_dict(self) -> dict[str, JsonValue]:
        phases = self.phase_nanoseconds()
        dominant_phase, dominant_ns = max(phases.items(), key=lambda item: item[1])
        return {
            "phase_seconds": {
                phase: nanoseconds / 1_000_000_000
                for phase, nanoseconds in phases.items()
            },
            "phase_children_seconds": {
                "proposal_generation": {
                    phase: nanoseconds / 1_000_000_000
                    for phase, nanoseconds in (
                        self.proposal_breakdown_nanoseconds().items()
                    )
                }
            },
            "phase_calls": {
                "proposal_generation": self.proposal_generation_calls,
            },
            "phase_children_calls": {
                "proposal_generation": {
                    **self.proposal_phase_calls(),
                    "other": None,
                }
            },
            "measured_total_seconds": self.measured_total_ns / 1_000_000_000,
            "accounted_seconds": self.accounted_ns / 1_000_000_000,
            "unattributed_seconds": self.unattributed_ns / 1_000_000_000,
            "unattributed_fraction": self.unattributed_ns
            / max(1, self.measured_total_ns),
            "dominant_phase": dominant_phase,
            "dominant_seconds": dominant_ns / 1_000_000_000,
        }


@dataclass(frozen=True, slots=True)
class DeepOperatorTimingProfile:
    operator_counters: Mapping[str, Mapping[str, int]]

    @staticmethod
    def _node(
        nanoseconds: int,
        *,
        calls: int | None,
        children: dict[str, JsonValue] | None = None,
        counters: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        node: dict[str, JsonValue] = {
            "seconds": nanoseconds / 1_000_000_000,
            "calls": calls,
        }
        if children:
            node["children"] = children
        if counters:
            node["counters"] = counters
        return node

    def as_dict(self) -> dict[str, JsonValue]:
        operators: dict[str, JsonValue] = {}
        for operator, counters in self.operator_counters.items():
            operator_ns = counters.get("operator_search_ns", 0)
            witness_search_ns = counters.get("witness_search_ns", 0)
            materialization_ns = counters.get(
                "witness_edge_materialization_ns", 0
            )
            switch_children_ns = {
                "partner_edge_sampling": counters.get(
                    "partner_edge_sampling_ns", 0
                ),
                "candidate_construction": counters.get(
                    "candidate_construction_ns", 0
                ),
                "connectivity_validation": counters.get(
                    "connectivity_validation_ns", 0
                ),
                "graph_family_validation": counters.get(
                    "graph_family_validation_ns", 0
                ),
            }
            switch_ns = sum(switch_children_ns.values())

            switch_children: dict[str, JsonValue] = {
                phase: self._node(nanoseconds, calls=None)
                for phase, nanoseconds in switch_children_ns.items()
            }
            operator_children: dict[str, JsonValue] = {}
            if witness_search_ns:
                operator_children["witness_search"] = self._node(
                    witness_search_ns,
                    calls=counters.get("witness_searches", 0),
                )
            if materialization_ns:
                operator_children["witness_edge_materialization"] = self._node(
                    materialization_ns,
                    calls=None,
                )
            if counters.get("switch_attempts", 0):
                operator_children["switch_attempts"] = self._node(
                    switch_ns,
                    calls=counters.get("switch_attempts", 0),
                    children=switch_children,
                    counters={"timing_scope": "measured children"},
                )
            other_ns = max(
                0,
                operator_ns - witness_search_ns - materialization_ns - switch_ns,
            )
            if other_ns:
                operator_children["other"] = self._node(other_ns, calls=None)

            operators[operator] = self._node(
                operator_ns,
                calls=counters.get("operator_search_calls", 0),
                children=operator_children,
                counters={
                    "witness_cache_lookups": counters.get(
                        "witness_cache_lookups", 0
                    ),
                    "witness_cache_hits": counters.get("witness_cache_hits", 0),
                    "witness_cache_misses": counters.get(
                        "witness_cache_misses", 0
                    ),
                },
            )
        return {"operators": operators}


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    baseline: str
    entry_id: str
    graph_seed: int
    policy_seed: int
    evaluations: int
    initial_score: GraphScore
    best_score: GraphScore
    final_score: GraphScore
    best_curve: tuple[int, ...]
    normalized_best_auc: float
    first_improvement_evaluation: int | None
    exact_zero_submissions: int
    exact_verified_count: int
    exact_verification_failures: int
    legal_proposals: int
    invalid_proposals: int
    noop_proposals: int
    duplicate_proposals: int
    score_failures: int
    timed_out: bool
    policy_call_ms: float
    elapsed_seconds: float
    final_graph6: str
    final_graph_hash: str
    timing_profile: EpisodeTimingProfile | None = None
    deep_operator_profile: DeepOperatorTimingProfile | None = None

    def as_dict(self, *, include_timing: bool = True) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "baseline": self.baseline,
            "entry_id": self.entry_id,
            "graph_seed": self.graph_seed,
            "policy_seed": self.policy_seed,
            "evaluations": self.evaluations,
            "initial_score": self.initial_score.as_dict(),
            "best_score": self.best_score.as_dict(),
            "final_score": self.final_score.as_dict(),
            "best_curve": list(self.best_curve),
            "normalized_best_auc": self.normalized_best_auc,
            "first_improvement_evaluation": self.first_improvement_evaluation,
            "exact_zero_submissions": self.exact_zero_submissions,
            "exact_verified_count": self.exact_verified_count,
            "exact_verification_failures": self.exact_verification_failures,
            "legal_proposals": self.legal_proposals,
            "invalid_proposals": self.invalid_proposals,
            "noop_proposals": self.noop_proposals,
            "duplicate_proposals": self.duplicate_proposals,
            "score_failures": self.score_failures,
            "timed_out": self.timed_out,
            "final_graph6": self.final_graph6,
            "final_graph_hash": self.final_graph_hash,
        }
        if include_timing:
            result["policy_call_ms"] = self.policy_call_ms
            result["elapsed_seconds"] = self.elapsed_seconds
            if self.timing_profile is not None:
                result["timing_profile"] = self.timing_profile.as_dict()
            if self.deep_operator_profile is not None:
                result["deep_operator_profile"] = (
                    self.deep_operator_profile.as_dict()
                )
        return result
