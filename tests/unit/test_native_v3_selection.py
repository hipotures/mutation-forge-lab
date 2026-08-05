from __future__ import annotations

from fractions import Fraction

import pytest

from mutation_forge.native_v3.scoring import RationalInterval
from mutation_forge.native_v3.selection import (
    IncomparableFitness,
    ProgramFitness,
    development_order,
    freeze_promotion_shortlist,
    missing_current_manifest_evaluations,
    validated_global_best,
)


def _fitness(
    program_hash: str,
    lower: int,
    *,
    manifest: str = "development",
    signature: str = "same",
) -> ProgramFitness:
    return ProgramFitness(
        program_hash,
        manifest,
        "protocol",
        RationalInterval(Fraction(lower, 10), Fraction(lower + 1, 10)),
        8,
        8,
        signature,
    )


def test_development_fitness_is_never_compared_across_manifest_hashes() -> None:
    with pytest.raises(IncomparableFitness):
        development_order((_fitness("a", 5), _fitness("b", 9, manifest="other")))


def test_promotion_shortlist_is_frozen_and_prefers_diversity_without_pruning() -> None:
    shortlist = freeze_promotion_shortlist(
        epoch_id="epoch",
        values=(
            _fitness("a", 9, signature="x"),
            _fitness("b", 8, signature="x"),
            _fitness("c", 7, signature="y"),
            _fitness("d", 6, signature="z"),
            _fitness("e", 5, signature="q"),
        ),
    )
    assert shortlist.program_hashes == ("a", "c", "d", "e")
    assert len(shortlist.program_hashes) == 4


def test_validated_global_best_only_compares_completed_locked_validation() -> None:
    best = validated_global_best(
        (
            _fitness("development-best", 10),
            _fitness("validation-a", 4, manifest="validation"),
            _fitness("validation-b", 7, manifest="validation"),
        ),
        validation_manifest_hash="validation",
        protocol_bundle_hash="protocol",
    )
    assert best.program_hash == "validation-b"


def test_retained_parents_and_baselines_are_planned_once_per_current_manifest() -> None:
    cached = _fitness("parent", 5)
    missing = missing_current_manifest_evaluations(
        program_hashes=("parent", "baseline", "parent"),
        manifest_hash="development",
        protocol_bundle_hash="protocol",
        cache={cached.cache_key: cached},
    )
    assert missing == ("baseline",)
