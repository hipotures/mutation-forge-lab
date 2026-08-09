from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from mutation_forge.native_v3.scoring import (
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EnergyScale,
    EvidenceStatus,
    IntegerInterval,
    RationalInterval,
    ScoreEvidence,
    candidate_fitness,
    conservative_fitness_key,
    episode_auc,
    proved_strict_energy_improvement,
)

IDENTITY = BackendIdentity(
    backend_id="heg",
    heg_commit="a" * 40,
    source_tree_sha256="b" * 64,
    binary_sha256="c" * 64,
    compiler_identity="c++ test",
    build_flags=("-O3",),
    platform="linux",
    architecture="x86_64",
)


def _component(
    length: int,
    *,
    lower: int,
    upper: int,
    status: EvidenceStatus,
    attempt_kind: AttemptKind = AttemptKind.INITIAL,
) -> CycleComponentEvidence:
    return CycleComponentEvidence(
        forbidden_length=length,
        observed_count=lower,
        lower_bound=lower,
        upper_bound=upper,
        status=status,
        node_budget=(
            50_000 if attempt_kind is AttemptKind.INITIAL else 200_000
        ),
        nodes_visited=1_000,
        wall_time_ns=2_000,
        attempt_kind=attempt_kind,
        backend_identity=IDENTITY,
    )


def test_cap_saturation_is_exact_and_exhaustion_is_only_a_lower_bound() -> None:
    evidence = ScoreEvidence(
        "graph",
        8,
        12,
        64,
        (
            _component(
                4,
                lower=64,
                upper=64,
                status=EvidenceStatus.SATURATED_AT_CAP,
            ),
            _component(
                8,
                lower=7,
                upper=64,
                status=EvidenceStatus.SEARCH_BUDGET_EXHAUSTED,
            ),
        ),
    )

    assert evidence.total_witness_interval == IntegerInterval(71, 128)
    assert evidence.scientifically_bounded
    assert not evidence.complete_under_cap


def test_safe_and_unsafe_timeout_statuses_are_scientifically_distinct() -> None:
    safe = ScoreEvidence(
        "safe",
        8,
        12,
        64,
        (
            _component(
                4,
                lower=3,
                upper=64,
                status=EvidenceStatus.SEARCH_TIMEOUT_WITH_SAFE_PARTIAL,
            ),
        ),
    )
    unsafe = ScoreEvidence(
        "unsafe",
        8,
        12,
        64,
        (
            _component(
                4,
                lower=0,
                upper=64,
                status=EvidenceStatus.SEARCH_TIMEOUT_WITHOUT_PARTIAL,
            ),
        ),
    )

    assert safe.scientifically_bounded
    assert not unsafe.scientifically_bounded
    scale = EnergyScale.build(order=8, forbidden_lengths=(4,), witness_cap=64)
    assert scale.interval(safe)
    with pytest.raises(ValueError, match="scientific"):
        scale.interval(unsafe)


def test_exact_status_rejects_non_point_bounds() -> None:
    with pytest.raises(ValueError, match="point"):
        _component(4, lower=1, upper=2, status=EvidenceStatus.EXACT)


def test_attempt_kind_rejects_an_unlocked_node_budget() -> None:
    with pytest.raises(ValueError, match="locked node budget"):
        CycleComponentEvidence(
            forbidden_length=4,
            observed_count=1,
            lower_bound=1,
            upper_bound=1,
            status=EvidenceStatus.EXACT,
            node_budget=49_999,
            nodes_visited=1,
            wall_time_ns=1,
            attempt_kind=AttemptKind.INITIAL,
            backend_identity=IDENTITY,
        )


def test_mixed_radix_preserves_lexicographic_order() -> None:
    scale = EnergyScale.build(order=8, forbidden_lengths=(4, 8), witness_cap=64)
    tuples = [
        (0, 0, scale.edge_min),
        (0, 0, scale.edge_min + 1),
        (0, 1, scale.edge_min),
        (1, 0, scale.edge_min),
        (1, scale.weighted_max, scale.edge_max),
    ]
    encoded = [
        scale.encode(total=total, weighted=weighted, edge_count=edges)
        for total, weighted, edges in tuples
    ]

    assert encoded == sorted(encoded)
    assert len(encoded) == len(set(encoded))


def test_interval_utility_auc_and_fitness_are_hand_calculated_rationals() -> None:
    scale = EnergyScale.build(order=8, forbidden_lengths=(4,), witness_cap=4)
    uncertain = scale.utility(IntegerInterval(0, scale.energy_max))
    exact_best = scale.utility(IntegerInterval(0, 0))
    auc = episode_auc([uncertain, exact_best, exact_best], horizon=2)

    assert auc == RationalInterval(Fraction(2, 3), Fraction(1))
    fitness = candidate_fitness(
        {
            8: [RationalInterval(Fraction(1, 2), Fraction(3, 4))],
            10: [RationalInterval(Fraction(1, 4), Fraction(1, 2))],
        }
    )
    assert fitness == RationalInterval(Fraction(3, 8), Fraction(5, 8))


def test_unproved_overlap_is_never_a_strict_improvement() -> None:
    incumbent = IntegerInterval(10, 20)
    overlap = IntegerInterval(5, 10)
    proved = IntegerInterval(5, 9)

    assert not proved_strict_energy_improvement(overlap, incumbent)
    assert proved_strict_energy_improvement(proved, incumbent)


def test_conservative_fitness_prefers_proved_lower_bound_then_exactness() -> None:
    exact = RationalInterval(Fraction(3, 4), Fraction(3, 4))
    uncertain = RationalInterval(Fraction(3, 4), Fraction(1))
    stronger = RationalInterval(Fraction(4, 5), Fraction(1))

    assert conservative_fitness_key(exact, "b") < conservative_fitness_key(
        uncertain, "a"
    )
    assert conservative_fitness_key(stronger, "c") < conservative_fitness_key(
        exact, "b"
    )


def test_identical_evidence_has_identical_semantic_hash() -> None:
    component = _component(
        4,
        lower=3,
        upper=64,
        status=EvidenceStatus.SEARCH_BUDGET_EXHAUSTED,
    )
    first = ScoreEvidence("graph", 8, 12, 64, (component,))
    second = ScoreEvidence(
        "graph",
        8,
        12,
        64,
        (replace(component, wall_time_ns=component.wall_time_ns + 1),),
    )

    assert first.semantic_hash == second.semantic_hash
    assert len(first.semantic_hash) == 64
