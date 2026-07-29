from __future__ import annotations

import pytest

from mutation_forge.evaluation.profiling import (
    DeepOperatorTimingAccumulator,
    aggregate_deep_operator_profiles,
)


def _profile():
    accumulator = DeepOperatorTimingAccumulator()
    accumulator.record(
        "heg_forbidden_cycle_break",
        {
            "targeted_evaluations": 1,
            "targeted_ns": 1_000_000_000,
            "witness_cache_lookups": 0,
            "witness_cache_hits": 0,
            "witness_cache_misses": 0,
            "witness_searches": 1,
            "witness_search_ns": 400_000_000,
            "witness_edge_materialization_ns": 100_000_000,
            "switch_attempts": 4,
            "partner_edge_sampling_ns": 50_000_000,
            "candidate_construction_ns": 200_000_000,
            "connectivity_validation_ns": 100_000_000,
            "graph_family_validation_ns": 0,
        },
    )
    return accumulator.finish()


def test_deep_operator_profile_builds_non_overlapping_tree() -> None:
    operators = _profile().as_dict()["operators"]
    assert isinstance(operators, dict)
    operator = operators["heg_forbidden_cycle_break"]
    assert isinstance(operator, dict)
    children = operator["children"]
    assert isinstance(children, dict)

    assert operator["seconds"] == 1.0
    assert operator["calls"] == 1
    assert sum(
        child["seconds"]
        for child in children.values()
        if isinstance(child, dict)
    ) == pytest.approx(1.0)

    witness = children["witness_search"]
    assert isinstance(witness, dict)
    assert witness["calls"] == 1

    switch = children["switch_attempts"]
    assert isinstance(switch, dict)
    assert switch["calls"] == 4
    assert set(switch["children"]) == {
        "partner_edge_sampling",
        "candidate_construction",
        "connectivity_validation",
        "graph_family_validation",
    }


def test_deep_operator_profiles_aggregate_exact_counters() -> None:
    profile = _profile()
    aggregate = aggregate_deep_operator_profiles(
        (profile, profile),
        enabled=True,
    )
    assert aggregate["enabled"] is True
    assert aggregate["profiled_episodes"] == 2
    operators = aggregate["operators"]
    assert isinstance(operators, dict)
    operator = operators["heg_forbidden_cycle_break"]
    assert isinstance(operator, dict)
    assert operator["seconds"] == 2.0
    assert operator["calls"] == 2
