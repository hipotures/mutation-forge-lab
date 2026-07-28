from __future__ import annotations

import time

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.evaluation.episode import run_episode
from mutation_forge.models import GraphState
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


def test_episode_uses_fast_hash_in_hot_loop() -> None:
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
    assert backend.state_hash_calls == 1 + result.legal_proposals
    assert backend.canonical_hash_calls == 1


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
