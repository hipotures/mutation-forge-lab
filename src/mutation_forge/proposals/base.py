from __future__ import annotations

from typing import Protocol

from mutation_forge.models import GraphState, RewritePlan


class ProposalSource(Protocol):
    operator_family: str

    def propose(
        self, graph: GraphState, *, policy_seed: int, evaluation: int
    ) -> RewritePlan: ...
