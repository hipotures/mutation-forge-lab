from __future__ import annotations

import random
import subprocess
from collections.abc import Mapping
from pathlib import Path

from mutation_forge.backends.heg import HegBackend


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
                "graph_materialization",
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
                assert payload["witness_searches"] == 1
                assert payload["witness_search_ns"] > 0
            candidate = backend.apply_rewrite(graph, rewrite)
            assert backend.validate(candidate).valid
    finally:
        backend.close()
