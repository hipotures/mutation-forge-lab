from __future__ import annotations

from dataclasses import dataclass

from mutation_forge.backends.base import (
    DeepProposalProfileRecorder,
    GraphBackend,
    ProposalTimingRecorder,
)
from mutation_forge.models import GraphState, RewritePlan


@dataclass(frozen=True, slots=True)
class TwoSwitchProposalSource:
    backend: GraphBackend
    operator_family: str

    def propose(
        self,
        graph: GraphState,
        *,
        policy_seed: int,
        evaluation: int,
        record_timing: ProposalTimingRecorder | None = None,
        record_deep_profile: DeepProposalProfileRecorder | None = None,
    ) -> RewritePlan:
        return self.backend.propose_rewrite(
            graph,
            operator_family=self.operator_family,
            policy_seed=policy_seed,
            evaluation=evaluation,
            record_timing=record_timing,
            record_deep_profile=record_deep_profile,
        )
