from __future__ import annotations

import time
from dataclasses import dataclass

from mutation_forge.models import JsonValue
from mutation_forge.proposals.k_switch import ProposalCandidate, ProposalPool
from mutation_forge.sandbox.contracts import (
    SandboxLimits,
    ScientificContext,
)
from mutation_forge.sandbox.errors import (
    ProtocolError,
    WorkerCrashError,
    WorkerTimeoutError,
)
from mutation_forge.sandbox.validation import ProgramIdentity, validate_policy
from mutation_forge.sandbox.worker import PolicyWorker


@dataclass(frozen=True, slots=True)
class RankedProposal:
    proposal_id: str
    priority: int | float
    k: int
    operator_family: str
    selector_tags: tuple[str, ...]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "proposal_id": self.proposal_id,
            "priority": self.priority,
            "k": self.k,
            "operator_family": self.operator_family,
            "selector_tags": list(self.selector_tags),
        }


@dataclass(frozen=True, slots=True)
class RankResult:
    policy_id: str
    pool_hash: str
    ranked: tuple[RankedProposal, ...]
    selected_proposal_id: str | None
    elapsed_ns: int
    exception: bool
    timeout: bool
    crash: bool
    protocol: bool
    error: dict[str, JsonValue] | None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "policy_id": self.policy_id,
            "pool_hash": self.pool_hash,
            "ranked": [item.as_dict() for item in self.ranked],
            "rank_order": [item.proposal_id for item in self.ranked],
            "selected_proposal_id": self.selected_proposal_id,
            "elapsed_ns": self.elapsed_ns,
            "exception": self.exception,
            "timeout": self.timeout,
            "crash": self.crash,
            "protocol": self.protocol,
            "error": self.error,
        }


class SourceRanker:
    def __init__(
        self,
        policy_id: str,
        source: str,
        limits: SandboxLimits,
    ) -> None:
        validation = validate_policy(source, limits)
        if not validation.valid:
            raise ValueError(f"invalid {policy_id} source: {validation.as_dict()}")
        self.policy_id = policy_id
        self.source = source
        self.identity: ProgramIdentity = validation.identity
        self._worker = PolicyWorker(source, limits)

    def rank(
        self,
        context: ScientificContext,
        pool: ProposalPool,
    ) -> RankResult:
        started = time.perf_counter_ns()
        ranked: list[RankedProposal] = []
        error: dict[str, JsonValue] | None = None
        exception = timeout = crash = protocol = False
        for candidate in pool.candidates:
            try:
                result = self._worker.call(context, candidate.payload)
            except WorkerTimeoutError as caught:
                timeout = True
                error = {"code": "worker_timeout", "message": str(caught)}
                break
            except WorkerCrashError as caught:
                crash = True
                error = {"code": "worker_crash", "message": str(caught)}
                break
            except ProtocolError as caught:
                protocol = True
                error = {"code": "worker_protocol", "message": str(caught)}
                break
            if result.status != "ok" or result.priority is None:
                exception = True
                error = result.error
                break
            ranked.append(
                RankedProposal(
                    proposal_id=candidate.proposal_id,
                    priority=result.priority,
                    k=candidate.payload["k"],
                    operator_family=candidate.payload["operator_family"],
                    selector_tags=tuple(candidate.payload["selector_tags"]),
                )
            )
        ranked.sort(key=lambda item: (-item.priority, item.proposal_id))
        return RankResult(
            policy_id=self.policy_id,
            pool_hash=pool.pool_hash,
            ranked=tuple(ranked),
            selected_proposal_id=ranked[0].proposal_id if ranked else None,
            elapsed_ns=time.perf_counter_ns() - started,
            exception=exception,
            timeout=timeout,
            crash=crash,
            protocol=protocol,
            error=error,
        )

    def candidate(
        self,
        pool: ProposalPool,
        proposal_id: str | None,
    ) -> ProposalCandidate | None:
        if proposal_id is None:
            return None
        return next(
            (
                candidate
                for candidate in pool.candidates
                if candidate.proposal_id == proposal_id
            ),
            None,
        )

    def telemetry(self) -> dict[str, JsonValue]:
        return self._worker.telemetry()

    def close(self) -> None:
        self._worker.close()

    def __enter__(self) -> SourceRanker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
