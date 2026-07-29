from __future__ import annotations

import random
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mutation_forge.backends.base import ScoreProfileRecorder
from mutation_forge.backends.heg import HegBackend
from mutation_forge.evaluation.episode import run_episode
from mutation_forge.models import GraphScore, GraphState, RewritePlan
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
        assert score is not None
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


def test_prepared_proposal_handoff_is_exact_bounded_and_fail_closed(
    heg_repo: Path,
) -> None:
    backend = HegBackend(heg_repo, prepared_graph_cache_enabled=False)
    try:
        graph = backend.generate_seed(order=30, seed=101)
        rewrite = backend.propose_rewrite(
            graph,
            operator_family="heg_uniform_two_switch",
            policy_seed=1,
            evaluation=1,
        )
        assert backend._prepared_proposal is not None  # noqa: SLF001
        with (
            patch.object(
                backend,
                "_to_heg",
                wraps=backend._to_heg,  # noqa: SLF001
            ) as materialize,
            patch.object(
                backend._plugin,  # noqa: SLF001
                "validate_graph",
                wraps=backend._plugin.validate_graph,  # noqa: SLF001
            ) as validate,
        ):
            candidate = backend.apply_rewrite(graph, rewrite)
        assert materialize.call_count == 0
        assert validate.call_count == 0
        assert backend.validate(candidate).valid
        assert backend._prepared_proposal is None  # noqa: SLF001

        stale_rewrite = backend.propose_rewrite(
            candidate,
            operator_family="heg_uniform_two_switch",
            policy_seed=1,
            evaluation=2,
        )
        current_rewrite = backend.propose_rewrite(
            candidate,
            operator_family="heg_uniform_two_switch",
            policy_seed=1,
            evaluation=3,
        )
        assert backend._prepared_proposal is not None  # noqa: SLF001
        assert backend._prepared_proposal.rewrite is current_rewrite  # noqa: SLF001
        with patch.object(
            backend._plugin,  # noqa: SLF001
            "validate_graph",
            wraps=backend._plugin.validate_graph,  # noqa: SLF001
        ) as validate:
            backend.apply_rewrite(candidate, stale_rewrite)
        assert validate.call_count == 1
        assert backend._prepared_proposal is None  # noqa: SLF001

        rewrite = backend.propose_rewrite(
            candidate,
            operator_family="heg_uniform_two_switch",
            policy_seed=1,
            evaluation=4,
        )
        copied_rewrite = RewritePlan(
            rewrite.removed_edges,
            rewrite.added_edges,
            rewrite.operator_family,
            dict(rewrite.metadata),
        )
        with patch.object(
            backend._plugin,  # noqa: SLF001
            "validate_graph",
            wraps=backend._plugin.validate_graph,  # noqa: SLF001
        ) as validate:
            backend.apply_rewrite(candidate, copied_rewrite)
        assert validate.call_count == 1

        rewrite = backend.propose_rewrite(
            candidate,
            operator_family="heg_uniform_two_switch",
            policy_seed=1,
            evaluation=5,
        )
        equal_source = GraphState(candidate.order, candidate.edges)
        with patch.object(
            backend._plugin,  # noqa: SLF001
            "validate_graph",
            wraps=backend._plugin.validate_graph,  # noqa: SLF001
        ) as validate:
            backend.apply_rewrite(equal_source, rewrite)
        assert validate.call_count == 1

        backend.propose_rewrite(
            candidate,
            operator_family="heg_uniform_two_switch",
            policy_seed=1,
            evaluation=6,
        )
        backend.deserialize_graph6(backend.serialize_graph6(candidate))
        assert backend._prepared_proposal is None  # noqa: SLF001
    finally:
        backend.close()
    assert backend._prepared_proposal is None  # noqa: SLF001


@pytest.mark.parametrize("policy_seed", [1, 2, 3])
def test_prepared_proposal_handoff_preserves_episode_trajectory(
    heg_repo: Path,
    policy_seed: int,
) -> None:
    enabled = HegBackend(heg_repo)
    disabled = HegBackend(
        heg_repo,
        prepared_proposal_handoff_enabled=False,
    )
    try:
        kwargs = {
            "entry_id": "prepared-proposal-parity",
            "graph_seed": 101,
            "policy_seed": policy_seed,
            "run_seed": 7,
            "baseline": HEG_FORBIDDEN_CYCLE_BREAK,
            "evaluations": 80,
            "witness_cap": 64,
            "profiling_enabled": False,
        }
        enabled_result = run_episode(
            backend=enabled,
            initial_graph=enabled.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            **kwargs,
        )
        disabled_result = run_episode(
            backend=disabled,
            initial_graph=disabled.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            **kwargs,
        )
        assert enabled_result.as_dict(
            include_timing=False
        ) == disabled_result.as_dict(include_timing=False)
    finally:
        enabled.close()
        disabled.close()


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

        with patch.object(
            backend,
            "_cached_prepared",
            wraps=backend._cached_prepared,  # noqa: SLF001
        ) as prepared_lookup:
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
        assert prepared_lookup.call_count == 2
        assert len(materializations) == 0
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
            (1, 1, 0, 0),
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


def test_prepared_graph_cache_reuses_materialization_and_validation(
    heg_repo: Path,
) -> None:
    cached = HegBackend(heg_repo)
    uncached = HegBackend(heg_repo, prepared_graph_cache_enabled=False)
    try:
        cached_graph = cached.generate_seed(order=30, seed=101)
        uncached_graph = uncached.generate_seed(order=30, seed=101)
        cached_counters: dict[str, int] = {}
        uncached_counters: dict[str, int] = {}

        def recorder(target: dict[str, int]) -> ScoreProfileRecorder:
            def record(event: str, payload: Mapping[str, int]) -> None:
                for name, value in payload.items():
                    key = f"{event}_{name}"
                    target[key] = target.get(key, 0) + value

            return record

        cached_scores = [
            cached.score(
                GraphState(cached_graph.order, cached_graph.edges),
                witness_cap=64,
                record_profile=recorder(cached_counters),
            )
            for _ in range(2)
        ]
        uncached_scores = [
            uncached.score(
                GraphState(uncached_graph.order, uncached_graph.edges),
                witness_cap=64,
                record_profile=recorder(uncached_counters),
            )
            for _ in range(2)
        ]

        assert cached_scores == uncached_scores
        assert all(score is not None for score in cached_scores)
        assert cached_counters["prepared_cache_hits"] == 2
        assert cached_counters.get("graph_materialization_calls", 0) == 0
        assert cached_counters["validation_calls"] == 1
        assert cached_counters["validation_cache_hits"] == 1
        assert uncached_counters["prepared_cache_hits"] == 0
        assert uncached_counters["graph_materialization_calls"] == 2
        assert uncached_counters["validation_calls"] == 2
    finally:
        cached.close()
        uncached.close()


def test_heg_score_cutoff_is_inclusive_and_fail_closed(
    heg_repo: Path,
) -> None:
    backend = HegBackend(heg_repo)
    cutoff_disabled = HegBackend(heg_repo, score_cutoff_enabled=False)
    try:
        graph = backend.generate_seed(order=30, seed=101)
        full_score = backend.score(graph, witness_cap=64)
        assert full_score is not None
        assert full_score.total_capped_witnesses > 0
        counters: dict[str, int] = {}

        def record(event: str, payload: Mapping[str, int]) -> None:
            for name, value in payload.items():
                key = f"{event}_{name}"
                counters[key] = counters.get(key, 0) + value

        dominated = backend.score(
            graph,
            witness_cap=64,
            cutoff=full_score,
            record_profile=record,
        )
        assert dominated is None
        assert counters["cutoff_applied"] == 1
        assert counters["worker_response_dominated_results"] == 1

        disabled_graph = cutoff_disabled.generate_seed(order=30, seed=101)
        disabled_score = cutoff_disabled.score(
            disabled_graph,
            witness_cap=64,
            cutoff=full_score,
            record_profile=record,
        )
        assert disabled_score == full_score
        assert counters["cutoff_disabled"] == 1

        zero_cutoff = GraphScore(
            valid=True,
            capped_cycle_counts=((4, 0), (8, 0), (16, 0)),
            total_capped_witnesses=0,
            weighted_penalty=0,
            complete=True,
            ordering_key=(0, 0, 0, 0, 45),
        )
        assert (
            backend.score(
                graph,
                witness_cap=64,
                cutoff=zero_cutoff,
                record_profile=record,
            )
            == full_score
        )
        assert counters["cutoff_disabled"] == 2
    finally:
        backend.close()
        cutoff_disabled.close()


def test_heg_score_worker_restarts_after_one_crash(heg_repo: Path) -> None:
    backend = HegBackend(heg_repo)
    try:
        graph = backend.generate_seed(order=30, seed=101)
        expected = backend.score(graph, witness_cap=64)
        assert expected is not None
        worker = backend._worker  # noqa: SLF001
        assert worker is not None
        assert worker.process is not None
        worker.process.kill()
        worker.process.wait(timeout=2)
        counters: dict[str, int] = {}

        def record(event: str, payload: Mapping[str, int]) -> None:
            for name, value in payload.items():
                key = f"{event}_{name}"
                counters[key] = counters.get(key, 0) + value

        recovered = backend.score(
            graph,
            witness_cap=64,
            record_profile=record,
        )
        assert recovered == expected
        assert backend.score_implementation == "heg-cpp-score-worker"
        assert counters["worker_failure_calls"] == 1
        assert counters["worker_restart_attempts"] == 1
        assert counters["worker_restart_successes"] == 1
        assert counters.get("python_fallback_calls", 0) == 0
    finally:
        backend.close()


def test_heg_score_worker_falls_back_after_restart_failure(
    heg_repo: Path,
) -> None:
    backend = HegBackend(heg_repo)
    reference = HegBackend(heg_repo)
    worker_error = backend._worker_error  # noqa: SLF001

    class AlwaysFailWorker:
        def score(self, *args: Any, **kwargs: Any) -> None:
            raise worker_error("synthetic worker failure")

        def restart(self) -> None:
            raise worker_error("synthetic restart failure")

        def close(self) -> None:
            return None

    try:
        graph = backend.generate_seed(order=30, seed=101)
        expected = reference.score(
            reference.generate_seed(order=30, seed=101),
            witness_cap=64,
        )
        assert expected is not None
        backend._worker = AlwaysFailWorker()  # noqa: SLF001
        counters: dict[str, int] = {}

        def record(event: str, payload: Mapping[str, int]) -> None:
            for name, value in payload.items():
                key = f"{event}_{name}"
                counters[key] = counters.get(key, 0) + value

        recovered = backend.score(
            graph,
            witness_cap=64,
            record_profile=record,
        )
        assert recovered == expected
        assert backend.score_implementation == "heg-python-bounded-reference"
        assert counters["worker_failure_calls"] == 2
        assert counters["worker_restart_attempts"] == 1
        assert counters.get("worker_restart_successes", 0) == 0
        assert counters["python_fallback_calls"] == 1
    finally:
        backend.close()
        reference.close()


@pytest.mark.parametrize("policy_seed", [1, 2, 3])
def test_score_optimizations_preserve_episode_trajectory(
    heg_repo: Path,
    policy_seed: int,
) -> None:
    optimized = HegBackend(heg_repo)
    baseline = HegBackend(
        heg_repo,
        score_cutoff_enabled=False,
        prepared_graph_cache_enabled=False,
    )
    try:
        kwargs = {
            "entry_id": "score-optimization-parity",
            "graph_seed": 101,
            "policy_seed": policy_seed,
            "run_seed": 7,
            "baseline": HEG_FORBIDDEN_CYCLE_BREAK,
            "evaluations": 80,
            "witness_cap": 64,
            "profiling_enabled": False,
        }
        optimized_result = run_episode(
            backend=optimized,
            initial_graph=optimized.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            score_cache_enabled=True,
            **kwargs,
        )
        baseline_result = run_episode(
            backend=baseline,
            initial_graph=baseline.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            score_cache_enabled=False,
            **kwargs,
        )
        assert optimized_result.as_dict(
            include_timing=False
        ) == baseline_result.as_dict(include_timing=False)
    finally:
        optimized.close()
        baseline.close()


@pytest.mark.parametrize("policy_seed", [1, 2, 3])
def test_score_longest_first_preserves_episode_trajectory(
    heg_repo: Path,
    policy_seed: int,
) -> None:
    longest_first = HegBackend(heg_repo)
    increasing = HegBackend(
        heg_repo,
        score_longest_first_enabled=False,
    )
    try:
        kwargs = {
            "entry_id": "score-order-parity",
            "graph_seed": 101,
            "policy_seed": policy_seed,
            "run_seed": 7,
            "baseline": HEG_FORBIDDEN_CYCLE_BREAK,
            "evaluations": 80,
            "witness_cap": 64,
            "profiling_enabled": False,
        }
        longest_result = run_episode(
            backend=longest_first,
            initial_graph=longest_first.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            **kwargs,
        )
        increasing_result = run_episode(
            backend=increasing,
            initial_graph=increasing.generate_seed(order=30, seed=101),
            deadline=time.monotonic() + 30,
            **kwargs,
        )
        assert longest_result.as_dict(
            include_timing=False
        ) == increasing_result.as_dict(include_timing=False)
    finally:
        longest_first.close()
        increasing.close()
