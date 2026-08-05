from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)

type ProposalTimingRecorder = Callable[[str, int], None]
type DeepProposalProfileRecorder = Callable[[str, Mapping[str, int | float | bool]], None]
type ScoreProfileRecorder = Callable[[str, Mapping[str, int]], None]


class GraphBackend(Protocol):
    backend_id: str

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]: ...

    def generate_seed(self, *, order: int, seed: int) -> GraphState: ...

    def validate(self, graph: GraphState) -> GraphValidation: ...

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: ScoreProfileRecorder | None = None,
    ) -> GraphScore | None: ...

    def exact_verify(self, graph: GraphState) -> ExactVerification: ...

    def canonical_hash(self, graph: GraphState) -> str: ...

    def state_hash(self, graph: GraphState) -> str: ...

    def serialize_graph6(self, graph: GraphState) -> str: ...

    def deserialize_graph6(self, value: str) -> GraphState: ...

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: ScoreProfileRecorder | None = None,
    ) -> GraphState: ...

    def propose_rewrite(
        self,
        graph: GraphState,
        *,
        operator_family: str,
        policy_seed: int,
        evaluation: int,
        record_timing: ProposalTimingRecorder | None = None,
        record_deep_profile: DeepProposalProfileRecorder | None = None,
    ) -> RewritePlan: ...

    def close(self) -> None: ...
