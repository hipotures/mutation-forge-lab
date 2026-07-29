from __future__ import annotations

from mutation_forge.stage2c.metrics import (
    curve_summary,
    oracle_summary,
    rank_correlation,
    top_k_overlap,
    top_tie,
)


def test_rank_correlation_handles_ties_and_constant_priorities() -> None:
    left = {"a": 3.0, "b": 2.0, "c": 2.0, "d": 1.0}
    right = {"a": 9.0, "b": 5.0, "c": 5.0, "d": 0.0}
    assert rank_correlation(left, right) == 1.0
    assert rank_correlation(
        {"a": 1.0, "b": 1.0},
        {"a": 2.0, "b": 1.0},
    ) is None
    assert top_tie(left) == (1, 0.25, 3)


def test_same_order_top_k_overlap_is_bounded() -> None:
    left = ("a", "b", "c", "d")
    right = ("a", "c", "b", "d")
    assert top_k_overlap(left, right, 1) == 1.0
    assert top_k_overlap(left, right, 2) == 0.5
    assert top_k_overlap(left, right, 5) == 1.0


def test_oracle_hit_regret_and_priority_association() -> None:
    result = oracle_summary(
        {"a": 2, "b": 2, "c": -1},
        "c",
        "b",
        {"a": 1.0, "b": 0.0, "c": 2.0},
        {"a": 1.0, "b": 2.0, "c": 0.0},
        (1, 2),
    )
    assert result["any_improving_proposal"]
    assert result["best_immediate_score_delta"] == 2
    assert result["improving_fraction"] == 2 / 3
    assert result["random"]["regret"] == 3
    assert not result["random"]["best_tie_hit"]
    assert result["structural"]["best_tie_hit"]
    assert not result["structural"]["top_k_hits"]["1"]
    assert result["structural"]["top_k_hits"]["2"]


def test_curve_diagnostics_cover_saturation_quantization_and_degenerate_denominator() -> None:
    result = curve_summary(
        [4, 4, 0, 4],
        [
            [4, 4, 4],
            [2, 2, 2],
            [0, 0, 0],
            [4, 2, 2],
        ],
        [
            [0, 0, 0],
            [4, 4, 4],
            [0, 0, 0],
            [2, 2, 2],
        ],
    )
    assert result["no_possible_denominator_fraction"] == 0.25
    assert result["random"]["improvement_timing_fraction"] == {
        "step_1": 0.25,
        "later": 0.25,
        "never": 0.5,
    }
    assert result["structural"]["improvement_timing_fraction"] == {
        "step_1": 0.5,
        "later": 0.0,
        "never": 0.5,
    }
    assert result["random"]["distinct_normalized_auc_values"] >= 3
    assert (
        result["median_auc_sensitivity_to_one_unit_one_step_improvement"]
        > 0.0
    )
