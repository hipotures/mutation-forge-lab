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
        return result
