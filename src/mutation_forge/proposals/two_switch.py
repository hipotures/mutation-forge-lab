from __future__ import annotations

from dataclasses import dataclass

from mutation_forge.backends.base import GraphBackend
from mutation_forge.models import GraphState, RewritePlan


@dataclass(frozen=True, slots=True)
class TwoSwitchProposalSource:
    backend: GraphBackend
    operator_family: str

    def propose(
        self, graph: GraphState, *, policy_seed: int, evaluation: int
    ) -> RewritePlan:
        return self.backend.propose_rewrite(
            graph,
            operator_family=self.operator_family,
            policy_seed=policy_seed,
            evaluation=evaluation,
        )
