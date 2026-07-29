from __future__ import annotations

import random
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from mutation_forge.backends.heg import HegBackend
from mutation_forge.evaluation.episode import run_episode
from mutation_forge.models import GraphState
from mutation_forge.policies.baselines import HEG_FORBIDDEN_CYCLE_BREAK


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_heg_seed_score_and_graph6_parity(heg_repo: Path) -> None:
    before = (_git(heg_repo, "rev-parse", "HEAD"), _git(heg_repo, "status", "--short"))
    backend = HegBackend(heg_repo)
    try:
        graph = backend.generate_seed(order=30, seed=101)
        assert backend.validate(graph).valid
        encoded = backend.serialize_graph6(graph)
        assert backend.deserialize_graph6(encoded) == graph

        direct = backend._plugin.generate_seed(  # noqa: SLF001
            random.Random(101), {"order": 30, "mode": "cubic_first"}
        )
        assert direct.to_graph6() == encoded
        score = backend.score(graph, witness_cap=64)
        assert score.valid
        assert score.total_capped_witnesses == sum(
            count for _, count in score.capped_cycle_counts
        )
        assert score.ordering_key[0] == 0
    finally:
        backend.close()
    after = (_git(heg_repo, "rev-parse", "HEAD"), _git(heg_repo, "status", "--short"))
    assert after == before


def test_both_heg_baselines_preserve_validity(heg_repo: Path) -> None:
    backend = HegBackend(heg_repo)
    try:
        graph = backend.generate_seed(order=30, seed=101)
        for operator in (
            "heg_uniform_two_switch",
            "heg_forbidden_cycle_break",
        ):
            proposal_timings: list[tuple[str, int]] = []
            deep_profiles: list[tuple[str, dict[str, int | float | bool]]] = []

            def record_timing(
                phase: str,
                elapsed_ns: int,
                timings: list[tuple[str, int]] = proposal_timings,
            ) -> None:
                timings.append((phase, elapsed_ns))

            def record_deep_profile(
                family: str,
                payload: Mapping[str, int | float | bool],
                profiles: list[
                    tuple[str, dict[str, int | float | bool]]
                ] = deep_profiles,
            ) -> None:
                profiles.append((family, dict(payload)))

            plain_rewrite = backend.propose_rewrite(
                graph,
                operator_family=operator,
                policy_seed=1,
                evaluation=1,
            )
            rewrite = backend.propose_rewrite(
                graph,
                operator_family=operator,
                policy_seed=1,
                evaluation=1,
                record_timing=record_timing,
                record_deep_profile=record_deep_profile,
            )
            assert rewrite == plain_rewrite
            assert [phase for phase, _ in proposal_timings] == [
                "rng_setup",
                "operator_search",
                "proposal_packaging",
            ]
            assert all(elapsed_ns > 0 for _, elapsed_ns in proposal_timings)
            assert len(deep_profiles) == 1
            family, payload = deep_profiles[0]
            assert family == operator
            if operator == "heg_uniform_two_switch":
                assert payload["uniform_evaluations"] == 1
                assert payload["uniform_ns"] > 0
            else:
                assert payload["targeted_evaluations"] == 1
                assert payload["targeted_ns"] > 0
                assert payload["witness_cache_lookups"] == 1
                assert payload["witness_cache_hits"] == 1
                assert payload["witness_cache_misses"] == 0
                assert payload["witness_searches"] == 0
            candidate = backend.apply_rewrite(graph, rewrite)
            assert backend.validate(candidate).valid
    finally:
        backend.close()


def test_forbidden_witness_cache_tracks_current_graph_identity(
    heg_repo: Path,
) -> None:
    backend = HegBackend(heg_repo)
    uncached = HegBackend(heg_repo, mutation_witness_cache_enabled=False)
    try:
        graph = backend.generate_seed(order=30, seed=101)
        equal_graph = GraphState(graph.order, graph.edges)
        profiles: list[dict[str, int | float | bool]] = []
        materializations: list[int] = []

        def record_deep_profile(
            _family: str,
            payload: Mapping[str, int | float | bool],
        ) -> None:
            profiles.append(dict(payload))

        def record_timing(phase: str, _elapsed_ns: int) -> None:
            if phase == "graph_materialization":
                materializations.append(1)

        rewrites = [
            backend.propose_rewrite(
                current,
                operator_family="heg_forbidden_cycle_break",
                policy_seed=1,
                evaluation=1,
                record_timing=record_timing,
                record_deep_profile=record_deep_profile,
            )
            for current in (graph, graph, equal_graph)
        ]
        assert rewrites[0] == rewrites[1] == rewrites[2]
        assert len(materializations) == 2
        assert [
            (
                profile["witness_cache_lookups"],
                profile["witness_cache_hits"],
                profile["witness_cache_misses"],
                profile["witness_searches"],
            )
            for profile in profiles
        ] == [
            (1, 0, 1, 1),
            (1, 1, 0, 0),
            (1, 0, 1, 1),
        ]

        uncached_graph = uncached.generate_seed(order=30, seed=101)
        uncached_profiles: list[dict[str, int | float | bool]] = []
        for _ in range(2):
            uncached.propose_rewrite(
                uncached_graph,
                operator_family="heg_forbidden_cycle_break",
                policy_seed=1,
                evaluation=1,
                record_deep_profile=lambda _family, payload: (
                    uncached_profiles.append(dict(payload))
                ),
            )
        assert all(
            profile["witness_cache_lookups"] == 1
            and profile["witness_cache_hits"] == 0
            and profile["witness_cache_misses"] == 1
            and profile["witness_searches"] == 1
            for profile in uncached_profiles
        )
    finally:
        backend.close()
        uncached.close()


def test_forbidden_witness_cache_preserves_episode_trajectory(
    heg_repo: Path,
) -> None:
    cached = HegBackend(heg_repo)
    uncached = HegBackend(heg_repo, mutation_witness_cache_enabled=False)
    try:
        kwargs = {
            "entry_id": "cache-parity",
            "graph_seed": 101,
            "policy_seed": 1,
            "baseline": HEG_FORBIDDEN_CYCLE_BREAK,
            "evaluations": 40,
            "witness_cap": 64,
        }
        cached_result = run_episode(
            backend=cached,
            initial_graph=cached.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            **kwargs,
        )
        uncached_result = run_episode(
            backend=uncached,
            initial_graph=uncached.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            **kwargs,
        )
        assert cached_result.as_dict(
            include_timing=False
        ) == uncached_result.as_dict(include_timing=False)
    finally:
        cached.close()
        uncached.close()
