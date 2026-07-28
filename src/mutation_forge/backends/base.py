from __future__ import annotations

from typing import Protocol

from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)


class GraphBackend(Protocol):
    backend_id: str

    def generate_seed(self, *, order: int, seed: int) -> GraphState: ...

    def validate(self, graph: GraphState) -> GraphValidation: ...

    def score(self, graph: GraphState, *, witness_cap: int) -> GraphScore: ...

    def exact_verify(self, graph: GraphState) -> ExactVerification: ...

    def canonical_hash(self, graph: GraphState) -> str: ...

    def state_hash(self, graph: GraphState) -> str: ...

    def serialize_graph6(self, graph: GraphState) -> str: ...

    def deserialize_graph6(self, value: str) -> GraphState: ...

    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState: ...

    def propose_rewrite(
        self,
        graph: GraphState,
        *,
        operator_family: str,
        policy_seed: int,
        evaluation: int,
    ) -> RewritePlan: ...

    def close(self) -> None: ...
