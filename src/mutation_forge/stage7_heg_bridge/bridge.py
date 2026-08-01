# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mutation_forge.backends.heg import HegBackend
from mutation_forge.models import GraphScore, GraphState, JsonValue
from mutation_forge.proposals.k_switch import (
    FeatureLimits,
    KSwitchPoolGenerator,
    PoolLimits,
    ProposalCandidate,
    ProposalPool,
    make_scientific_context,
)
from mutation_forge.sandbox.contracts import SandboxLimits, ScientificContext
from mutation_forge.sandbox.errors import ProtocolError, WorkerCrashError, WorkerTimeoutError
from mutation_forge.stage2b.rankers import SourceRanker
from mutation_forge.stage7_heg_bridge.contract import (
    CATALOG_ID,
    FAILURE_POLICY,
    FROZEN_CYCLE_NODE_BUDGET,
    FROZEN_DISTANCE_QUERY_BUDGET,
    FROZEN_FORBIDDEN_LENGTHS,
    FROZEN_IDENTITY,
    FROZEN_LOCAL_RISK_BUDGET,
    FROZEN_WITNESS_SAMPLE_CAP,
    HEG_COMMIT,
    TIE_BREAKING_RULE,
    ContractViolation,
    catalog_source,
    source_identity,
    verify_heg_checkout,
)


class BridgeError(RuntimeError):
    """A fail-closed bridge error; callers must not choose a fallback operator."""


@dataclass(slots=True)
class BridgeTelemetry:
    policy_call_count: int = 0
    invalid_result_count: int = 0
    timeout_count: int = 0
    crash_count: int = 0
    protocol_count: int = 0
    exception_count: int = 0
    selection_latency_ns_sum: int = 0
    pool_generation_ns: int = 0
    feature_computation_ns: int = 0
    tie_count: int = 0
    selected_k_counts: Counter[str] = field(default_factory=Counter)
    selector_counts: Counter[str] = field(default_factory=Counter)
    scorer_calls: int = 0
    m4_calls: int = 0

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "policy_call_count": self.policy_call_count,
            "invalid_result_count": self.invalid_result_count,
            "timeout_count": self.timeout_count,
            "crash_count": self.crash_count,
            "protocol_count": self.protocol_count,
            "exception_count": self.exception_count,
            "selection_latency_ns_sum": self.selection_latency_ns_sum,
            "pool_generation_ns": self.pool_generation_ns,
            "feature_computation_ns": self.feature_computation_ns,
            "tie_count": self.tie_count,
            "selected_k_counts": dict(sorted(self.selected_k_counts.items())),
            "selector_counts": dict(sorted(self.selector_counts.items())),
            "scorer_calls": self.scorer_calls,
            "m4_calls": self.m4_calls,
        }


@dataclass(frozen=True, slots=True)
class Selection:
    catalog_id: str
    policy_id: str
    pool_hash: str
    selected_proposal_id: str
    selected_k: int
    selected_operator_family: str
    selected_selector_tags: tuple[str, ...]
    rank_order: tuple[str, ...]
    telemetry: dict[str, JsonValue]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "catalog_id": self.catalog_id,
            "policy_id": self.policy_id,
            "pool_hash": self.pool_hash,
            "selected_proposal_id": self.selected_proposal_id,
            "selected_k": self.selected_k,
            "selected_operator_family": self.selected_operator_family,
            "selected_selector_tags": list(self.selected_selector_tags),
            "rank_order": list(self.rank_order),
            "telemetry": self.telemetry,
        }


def _pool_hash(pool: ProposalPool) -> str:
    candidates = [
        {
            "proposal": candidate.payload,
            "removed_edges": candidate.rewrite.removed_edges,
            "added_edges": candidate.rewrite.added_edges,
        }
        for candidate in pool.candidates
    ]
    return hashlib.sha256(
        json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class HegPolicyBridge:
    """Reference-only host boundary for the reviewed Stage 4R policy.

    The constructor intentionally has no source/path argument.  A caller can
    choose only ``mutation_forge_stage4r_v1`` and the source is resolved from
    the reviewed package fixture.  The bridge never invokes a scorer while
    ranking a pool and never invokes M4.
    """

    def __init__(
        self,
        heg_repo: Path,
        *,
        catalog_id: str = CATALOG_ID,
        pool_limits: PoolLimits | None = None,
        feature_limits: FeatureLimits | None = None,
        sandbox_limits: SandboxLimits | None = None,
    ) -> None:
        if catalog_id != CATALOG_ID:
            raise ContractViolation("only the reviewed catalog ID may be activated")
        self.heg_repo = heg_repo.resolve()
        self.heg_identity = verify_heg_checkout(self.heg_repo)
        self.catalog_id = catalog_id
        self.source = catalog_source(catalog_id)
        self.identity = source_identity(self.source)
        self.pool_limits = pool_limits or PoolLimits()
        self.feature_limits = feature_limits or FeatureLimits(
            forbidden_lengths=FROZEN_FORBIDDEN_LENGTHS,
            witness_sample_cap=FROZEN_WITNESS_SAMPLE_CAP,
            cycle_node_budget=FROZEN_CYCLE_NODE_BUDGET,
            distance_query_budget=FROZEN_DISTANCE_QUERY_BUDGET,
            local_risk_budget=FROZEN_LOCAL_RISK_BUDGET,
        )
        self.sandbox_limits = sandbox_limits or SandboxLimits()
        self.backend = HegBackend(self.heg_repo)
        if self.backend.commit != HEG_COMMIT or self.backend.dirty:
            self.backend.close()
            raise ContractViolation("HEG changed after the read-only pin check")
        self.generator = KSwitchPoolGenerator(
            self.backend,
            pool_limits=self.pool_limits,
            feature_limits=self.feature_limits,
        )
        self.ranker = SourceRanker(
            FROZEN_IDENTITY.policy_id,
            self.source,
            self.sandbox_limits,
        )
        self.telemetry = BridgeTelemetry()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise BridgeError("bridge is closed")

    @staticmethod
    def _candidate_by_id(pool: ProposalPool, proposal_id: str) -> ProposalCandidate:
        for candidate in pool.candidates:
            if candidate.proposal_id == proposal_id:
                return candidate
        raise BridgeError("worker selected a proposal not present in the host pool")

    def validate_pool(self, graph: GraphState, pool: ProposalPool) -> None:
        self._ensure_open()
        if pool.schema_version != "stage2b.pool.v1":
            raise BridgeError("unsupported proposal-pool schema")
        if not pool.candidates:
            raise BridgeError("empty proposal pool is not selectable")
        if len(pool.candidates) > 64:
            raise BridgeError("proposal pool exceeds the bounded contract")
        ids = [candidate.proposal_id for candidate in pool.candidates]
        if len(ids) != len(set(ids)):
            raise BridgeError("duplicate proposal IDs are rejected")
        if pool.pool_hash != _pool_hash(pool):
            raise BridgeError("proposal pool hash does not match canonical contents")
        for candidate in pool.candidates:
            payload = candidate.payload
            k = payload.get("k")
            if k not in (2, 3, 4):
                raise BridgeError("the frozen policy contract requires k=2,3,4")
            if payload.get("operator_family") != f"legal_{k}_switch":
                raise BridgeError("operator family is not the canonical k-switch label")
            if payload.get("proposal_id") != candidate.proposal_id:
                raise BridgeError("proposal ID is not stable")
            try:
                self.backend.apply_rewrite(graph, candidate.rewrite)
            except ValueError as error:
                raise BridgeError(f"host rejected a supposedly legal proposal: {error}") from error

    def generate_pool(self, graph: GraphState, *, policy_seed: int, step: int) -> ProposalPool:
        self._ensure_open()
        started = time.perf_counter_ns()
        pool = self.generator.generate(graph, policy_seed=policy_seed, step=step)
        elapsed = time.perf_counter_ns() - started
        self.telemetry.pool_generation_ns += elapsed
        self.telemetry.feature_computation_ns += pool.feature_elapsed_ns
        self.validate_pool(graph, pool)
        return pool

    def context_for_graph(
        self,
        graph: GraphState,
        *,
        step: int,
        remaining_steps: int,
        score: GraphScore | None = None,
    ) -> ScientificContext:
        self._ensure_open()
        if score is None:
            score_started = time.perf_counter_ns()
            score = self.backend.score(graph, witness_cap=32)
            self.telemetry.scorer_calls += 1
            if score is None or self.backend.score_implementation != "heg-cpp-score-worker":
                raise BridgeError("context score did not use the mandatory HEG scorer")
            self.telemetry.feature_computation_ns += time.perf_counter_ns() - score_started
        return make_scientific_context(
            graph,
            score,
            forbidden_lengths=self.feature_limits.forbidden_lengths,
            step=step,
            remaining_steps=remaining_steps,
        )

    def select(
        self,
        context: ScientificContext,
        pool: ProposalPool,
        *,
        graph: GraphState | None = None,
        apply_selected: bool = False,
    ) -> Selection:
        """Rank a host-generated pool and return only the selected identity/telemetry."""
        self._ensure_open()
        if graph is not None:
            self.validate_pool(graph, pool)
        started = time.perf_counter_ns()
        try:
            ranking = self.ranker.rank(context, pool)
        except WorkerTimeoutError as error:
            self.telemetry.timeout_count += 1
            raise BridgeError(f"{FAILURE_POLICY}: policy timeout") from error
        except WorkerCrashError as error:
            self.telemetry.crash_count += 1
            raise BridgeError(f"{FAILURE_POLICY}: policy crash") from error
        except ProtocolError as error:
            self.telemetry.protocol_count += 1
            raise BridgeError(f"{FAILURE_POLICY}: policy protocol error") from error
        self.telemetry.selection_latency_ns_sum += time.perf_counter_ns() - started
        self.telemetry.policy_call_count += len(pool.candidates)
        if ranking.exception or ranking.timeout or ranking.crash or ranking.protocol:
            self.telemetry.exception_count += int(ranking.exception)
            raise BridgeError(f"{FAILURE_POLICY}: policy returned an invalid result: {ranking.error}")
        if ranking.selected_proposal_id is None or len(ranking.ranked) != len(pool.candidates):
            self.telemetry.invalid_result_count += 1
            raise BridgeError("policy did not rank every legal proposal")
        priorities = [item.priority for item in ranking.ranked]
        self.telemetry.tie_count += sum(
            1 for left, right in zip(priorities, priorities[1:], strict=False) if left == right
        )
        selected = self._candidate_by_id(pool, ranking.selected_proposal_id)
        self.telemetry.selected_k_counts[str(selected.payload["k"])] += 1
        for tag in selected.payload["selector_tags"]:
            self.telemetry.selector_counts[tag] += 1
        if apply_selected:
            if graph is None:
                raise BridgeError("selected-plan application requires the source graph")
            candidate_graph = self.backend.apply_rewrite(graph, selected.rewrite)
            selected_score = self.backend.score(candidate_graph, witness_cap=32)
            self.telemetry.scorer_calls += 1
            if selected_score is None or self.backend.score_implementation != "heg-cpp-score-worker":
                raise BridgeError("selected-plan score did not use the mandatory HEG scorer")
        return Selection(
            catalog_id=self.catalog_id,
            policy_id=FROZEN_IDENTITY.policy_id,
            pool_hash=pool.pool_hash,
            selected_proposal_id=selected.proposal_id,
            selected_k=int(selected.payload["k"]),
            selected_operator_family=str(selected.payload["operator_family"]),
            selected_selector_tags=tuple(selected.payload["selector_tags"]),
            rank_order=tuple(item.proposal_id for item in ranking.ranked),
            telemetry=self.telemetry.as_dict(),
        )

    def select_for_graph(
        self,
        graph: GraphState,
        *,
        policy_seed: int,
        step: int,
        remaining_steps: int,
        apply_selected: bool = False,
    ) -> Selection:
        pool = self.generate_pool(graph, policy_seed=policy_seed, step=step)
        context = self.context_for_graph(
            graph,
            step=step,
            remaining_steps=remaining_steps,
        )
        return self.select(context, pool, graph=graph, apply_selected=apply_selected)

    def canonical_selection_json(self, selection: Selection) -> bytes:
        return json.dumps(
            selection.as_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.ranker.close()
        self.backend.close()

    def __enter__(self) -> HegPolicyBridge:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def assert_selection_contract(selection: Selection) -> None:
    if selection.catalog_id != CATALOG_ID or selection.policy_id != FROZEN_IDENTITY.policy_id:
        raise ContractViolation("selection identity drift")
    if not selection.selected_proposal_id:
        raise ContractViolation("selection must contain a proposal identity")
    if selection.rank_order[0] != selection.selected_proposal_id:
        raise ContractViolation("selection is not the first stable ranked proposal")
    if selection.telemetry.get("m4_calls") != 0:
        raise ContractViolation("policy bridge must never call M4")
    if cast(int, selection.telemetry.get("scorer_calls", 0)) < 0:
        raise ContractViolation("invalid scorer telemetry")
    if TIE_BREAKING_RULE not in "descending_priority_then_lexicographic_proposal_id":
        raise ContractViolation("tie-breaking contract drift")
