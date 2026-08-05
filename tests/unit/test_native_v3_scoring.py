from __future__ import annotations

from fractions import Fraction

import pytest

from mutation_forge.native_v3.scoring import (
    ACCEPTANCE_PROTOCOL_ID,
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EnergyScale,
    EvidenceStatus,
    IntegerInterval,
    RationalInterval,
    ScoreEvidence,
    ScoreEvidenceCache,
    ScoreEvidenceCacheKey,
    acceptance_seed64,
    aggregate_order_balanced,
    episode_auc,
    metropolis_accepts,
    metropolis_threshold,
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
) -> CycleComponentEvidence:
    return CycleComponentEvidence(
        forbidden_length=length,
        observed_count=lower,
        lower_bound=lower,
        upper_bound=upper,
        status=status,
        node_budget=50_000,
        nodes_visited=1_000,
        wall_time_ns=2_000,
        attempt_kind=AttemptKind.INITIAL,
        backend_identity=IDENTITY,
    )


def test_cap_saturation_is_exact_and_budget_exhaustion_is_interval() -> None:
    saturated = _component(
        4,
        lower=64,
        upper=64,
        status=EvidenceStatus.SATURATED_AT_CAP,
    )
    exhausted = _component(
        8,
        lower=7,
        upper=64,
        status=EvidenceStatus.SEARCH_BUDGET_EXHAUSTED,
    )
    evidence = ScoreEvidence("graph", 8, 12, 64, (saturated, exhausted))
    assert evidence.total_witness_interval == IntegerInterval(71, 128)
    assert evidence.scientifically_bounded
    assert not evidence.complete_under_cap


def test_exact_status_rejects_non_point_bounds() -> None:
    with pytest.raises(ValueError, match="point"):
        _component(4, lower=1, upper=2, status=EvidenceStatus.EXACT)


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


def test_interval_utility_and_auc_are_exact_rationals() -> None:
    scale = EnergyScale.build(order=8, forbidden_lengths=(4,), witness_cap=4)
    first = scale.utility(IntegerInterval(0, scale.energy_max))
    exact_best = scale.utility(IntegerInterval(0, 0))
    auc = episode_auc([first, exact_best, exact_best], horizon=2)
    assert auc.lower == Fraction(2, 3)
    assert auc.upper == 1

    aggregate = aggregate_order_balanced(
        {
            8: [RationalInterval(Fraction(1, 2), Fraction(3, 4))],
            10: [RationalInterval(Fraction(1, 4), Fraction(1, 2))],
        }
    )
    assert aggregate == RationalInterval(Fraction(3, 8), Fraction(5, 8))


def test_score_cache_requires_complete_protocol_key() -> None:
    evidence = ScoreEvidence(
        "graph",
        8,
        12,
        64,
        (_component(4, lower=1, upper=64, status=EvidenceStatus.SEARCH_BUDGET_EXHAUSTED),),
    )
    key_50k = ScoreEvidenceCacheKey(
        "graph",
        (4,),
        64,
        50_000,
        AttemptKind.INITIAL,
        "score",
        IDENTITY.canonical_key(),
    )
    key_200k = ScoreEvidenceCacheKey(
        "graph",
        (4,),
        64,
        200_000,
        AttemptKind.EXPANDED,
        "score",
        IDENTITY.canonical_key(),
    )
    cache = ScoreEvidenceCache()
    cache.put(key_50k, evidence)
    assert cache.get(key_50k) is evidence
    assert cache.get(key_200k) is None
    assert cache.hit_rate == Fraction(1, 2)


def test_metropolis_kernel_has_frozen_boundary_vectors() -> None:
    assert metropolis_threshold(Fraction(0), Fraction(1, 64)) == 1 << 64
    threshold = metropolis_threshold(Fraction(1, 100), Fraction(1, 64))
    assert threshold == 9726828398328049861
    seed = acceptance_seed64(
        protocol_id=ACCEPTANCE_PROTOCOL_ID,
        program_hash="a" * 64,
        episode_id="episode",
        policy_seed=7,
        step_index=3,
        ast_path="/entry/choose",
        repeat_indices=(1, 2),
        invocation_ordinal=4,
        draw_ordinal=0,
    )
    assert seed == 17721263687745471401
    assert metropolis_accepts(
        delta=Fraction(1, 100),
        temperature=Fraction(1, 64),
        seed64=seed,
    ) == (False, 9726828398328049861, 17649913427034394247)
