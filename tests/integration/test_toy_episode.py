from __future__ import annotations

import time

import pytest

from mutation_forge.backends.base import (
    DeepProposalProfileRecorder,
    ProposalTimingRecorder,
    ScoreProfileRecorder,
)
from mutation_forge.backends.toy import ToyBackend
from mutation_forge.evaluation.episode import run_episode
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    RewritePlan,
)
from mutation_forge.policies.baselines import HEG_UNIFORM_TWO_SWITCH


class CountingToyBackend(ToyBackend):
    def __init__(self) -> None:
        self.state_hash_calls = 0
        self.canonical_hash_calls = 0

    def state_hash(self, graph: GraphState) -> str:
        self.state_hash_calls += 1
        return super().state_hash(graph)

    def canonical_hash(self, graph: GraphState) -> str:
        self.canonical_hash_calls += 1
        return super().canonical_hash(graph)


class RepeatedCandidateBackend(ToyBackend):
    def __init__(self, *, fail_first_candidate: bool = False) -> None:
        self.score_calls = 0
        self.fail_first_candidate = fail_first_candidate

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
        return super().propose_rewrite(
            graph,
            operator_family=operator_family,
            policy_seed=policy_seed,
            evaluation=1,
        )

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: ScoreProfileRecorder | None = None,
    ) -> GraphScore | None:
        self.score_calls += 1
        if self.fail_first_candidate and self.score_calls == 2:
            raise RuntimeError("synthetic score failure")
        return GraphScore(
            valid=True,
            capped_cycle_counts=((4, 1),),
            total_capped_witnesses=1,
            weighted_penalty=16,
            complete=True,
            ordering_key=(0, 1, 16, 0, len(graph.edges)),
        )


class ZeroScoreBackend(ToyBackend):
    def __init__(self) -> None:
        self.cutoffs: list[GraphScore | None] = []
        self.exact_calls = 0

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: ScoreProfileRecorder | None = None,
    ) -> GraphScore | None:
        self.cutoffs.append(cutoff)
        return GraphScore(
            valid=True,
            capped_cycle_counts=((4, 0),),
            total_capped_witnesses=0,
            weighted_penalty=0,
            complete=True,
            ordering_key=(0, 0, 0, 0, len(graph.edges)),
        )

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        self.exact_calls += 1
        return ExactVerification(
            status="VERIFIED",
            complete=True,
            message="synthetic exact zero",
            implementation="test",
        )


def test_toy_episode_is_deterministic() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=30, seed=101)
    kwargs = {
        "backend": backend,
        "initial_graph": graph,
        "entry_id": "toy",
        "graph_seed": 101,
        "policy_seed": 1,
        "baseline": HEG_UNIFORM_TWO_SWITCH,
        "evaluations": 40,
        "witness_cap": 64,
    }
    first = run_episode(**kwargs, deadline=time.monotonic() + 30)
    second = run_episode(**kwargs, deadline=time.monotonic() + 30)
    assert first.as_dict(include_timing=False) == second.as_dict(include_timing=False)
    assert backend.validate(backend.deserialize_graph6(first.final_graph6)).valid


def test_episode_profiling_preserves_trajectory_and_accounts_time() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=30, seed=101)
    kwargs = {
        "backend": backend,
        "initial_graph": graph,
        "entry_id": "toy",
        "graph_seed": 101,
        "policy_seed": 1,
        "baseline": HEG_UNIFORM_TWO_SWITCH,
        "evaluations": 40,
        "witness_cap": 64,
    }
    profiled = run_episode(
        **kwargs,
        deadline=time.monotonic() + 30,
        profiling_enabled=True,
    )
    unprofiled = run_episode(
        **kwargs,
        deadline=time.monotonic() + 30,
        profiling_enabled=False,
    )

    assert profiled.as_dict(include_timing=False) == unprofiled.as_dict(
        include_timing=False
    )
    assert unprofiled.timing_profile is None
    assert "timing_profile" not in unprofiled.as_dict()
    assert profiled.timing_profile is not None
    profile = profiled.timing_profile
    assert profile.measured_total_ns >= profile.accounted_ns > 0
    assert profile.unattributed_ns >= 0
    serialized = profile.as_dict()
    assert serialized["dominant_phase"] in profile.phase_nanoseconds()
    assert serialized["measured_total_seconds"] > 0
    proposal_children = serialized["phase_children_seconds"]["proposal_generation"]
    assert proposal_children["other"] > 0
    assert sum(proposal_children.values()) == pytest.approx(
        serialized["phase_seconds"]["proposal_generation"]
    )
    assert serialized["phase_calls"]["proposal_generation"] == 40
    assert serialized["phase_children_calls"]["proposal_generation"] == {
        "rng_setup": 0,
        "graph_materialization": 0,
        "operator_search": 0,
        "proposal_packaging": 0,
        "other": None,
    }


def test_episode_uses_exact_graph_state_in_hot_loop() -> None:
    backend = CountingToyBackend()
    graph = backend.generate_seed(order=30, seed=101)
    result = run_episode(
        backend=backend,
        initial_graph=graph,
        entry_id="toy",
        graph_seed=101,
        policy_seed=1,
        baseline=HEG_UNIFORM_TWO_SWITCH,
        evaluations=40,
        witness_cap=64,
        deadline=time.monotonic() + 30,
    )
    assert result.evaluations == 40
    assert backend.state_hash_calls == 0
    assert backend.canonical_hash_calls == 1


def test_episode_score_cache_reuses_full_duplicate_scores() -> None:
    results = {}
    calls = {}
    for cache_enabled in (False, True):
        backend = RepeatedCandidateBackend()
        result = run_episode(
            backend=backend,
            initial_graph=backend.generate_seed(order=30, seed=101),
            entry_id="duplicate-cache",
            graph_seed=101,
            policy_seed=1,
            baseline=HEG_UNIFORM_TWO_SWITCH,
            evaluations=12,
            witness_cap=64,
            deadline=time.monotonic() + 30,
            deep_profiling_enabled=True,
            score_cache_enabled=cache_enabled,
        )
        results[cache_enabled] = result.as_dict(include_timing=False)
        calls[cache_enabled] = backend.score_calls
        assert result.duplicate_proposals == 11
        assert result.deep_score_profile is not None

    assert results[False] == results[True]
    assert calls[False] == 13
    assert calls[True] == 2


def test_episode_score_cache_does_not_cache_failures() -> None:
    backend = RepeatedCandidateBackend(fail_first_candidate=True)
    result = run_episode(
        backend=backend,
        initial_graph=backend.generate_seed(order=30, seed=101),
        entry_id="failure-cache",
        graph_seed=101,
        policy_seed=1,
        baseline=HEG_UNIFORM_TWO_SWITCH,
        evaluations=4,
        witness_cap=64,
        deadline=time.monotonic() + 30,
        deep_profiling_enabled=True,
        score_cache_enabled=True,
    )

    assert result.score_failures == 1
    assert backend.score_calls == 3
    assert result.deep_score_profile is not None
    counters = result.deep_score_profile.as_dict()["counters"]
    assert counters["score_cache_hits"] == 2
    assert counters["score_cache_misses"] == 2
    assert counters["score_result_failures"] == 1


def test_episode_disables_cutoff_for_zero_incumbent() -> None:
    backend = ZeroScoreBackend()
    result = run_episode(
        backend=backend,
        initial_graph=backend.generate_seed(order=30, seed=101),
        entry_id="zero-cutoff",
        graph_seed=101,
        policy_seed=1,
        baseline=HEG_UNIFORM_TWO_SWITCH,
        evaluations=1,
        witness_cap=64,
        deadline=time.monotonic() + 30,
    )

    assert backend.cutoffs == [None, None]
    assert backend.exact_calls == 1
    assert result.exact_zero_submissions == 1
    assert result.exact_verified_count == 1


def test_episode_stops_at_wall_deadline() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=30, seed=101)
    result = run_episode(
        backend=backend,
        initial_graph=graph,
        entry_id="toy",
        graph_seed=101,
        policy_seed=1,
        baseline=HEG_UNIFORM_TWO_SWITCH,
        evaluations=1000,
        witness_cap=64,
        deadline=time.monotonic() - 1,
    )
    assert result.timed_out
    assert result.evaluations == 0
