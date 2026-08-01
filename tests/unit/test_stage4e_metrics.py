from __future__ import annotations

from fractions import Fraction

import pytest

from mutation_forge.stage4e_metrics import (
    DEFAULT_BOOTSTRAP_SEED,
    FROZEN_PERCENTILE_RULE,
    PairedAreaEpisode,
    bootstrap_paired_theta,
    paired_area_delta,
    summarize_paired_areas,
    terminal_gate_checks,
    terminal_gate_passes,
)


def _episodes() -> list[PairedAreaEpisode]:
    return [
        PairedAreaEpisode(10, 101, "o10-g101-p1", (0.6, 0.8), (0.4, 0.6)),
        PairedAreaEpisode(10, 101, "o10-g101-p2", (0.7, 0.9), (0.5, 0.7)),
        PairedAreaEpisode(10, 102, "o10-g102-p1", (0.8, 0.8), (0.4, 0.4)),
        PairedAreaEpisode(10, 102, "o10-g102-p2", (0.7, 0.9), (0.3, 0.5)),
        PairedAreaEpisode(12, 201, "o12-g201-p1", (0.5, 0.7), (0.4, 0.5)),
        PairedAreaEpisode(12, 201, "o12-g201-p2", (0.6, 0.8), (0.5, 0.6)),
        PairedAreaEpisode(12, 202, "o12-g202-p1", (0.5, 0.7), (0.3, 0.5)),
        PairedAreaEpisode(12, 202, "o12-g202-p2", (0.6, 0.6), (0.4, 0.4)),
    ]


def test_paired_area_hierarchy_is_exact_and_equal_order_weighted() -> None:
    summary = summarize_paired_areas(_episodes())

    assert paired_area_delta((0.6, 0.8), (0.4, 0.6)) == Fraction(1, 5)
    assert [graph.delta_mean for graph in summary.graphs] == [
        Fraction(1, 5),
        Fraction(2, 5),
        Fraction(3, 20),
        Fraction(1, 5),
    ]
    assert [order.delta_mean for order in summary.orders] == [Fraction(3, 10), Fraction(7, 40)]
    assert summary.theta == Fraction(19, 80)
    assert summary.mu_b == Fraction(37, 80)
    assert summary.relative_improvement == Fraction(19, 37)
    assert len(summary.episodes) == 8
    assert len(summary.graphs) == 4


def test_point_bootstrap_and_relative_effect_share_the_paired_area_estimand() -> None:
    summary = summarize_paired_areas(_episodes())
    bootstrap = bootstrap_paired_theta(summary, samples=16, seed=DEFAULT_BOOTSTRAP_SEED)

    # The bootstrap targets theta, not a separate median-AUC statistic, and
    # the reported relative effect is that same theta divided by mu_B.
    assert bootstrap.estimand == "paired_transition_aware_area_delta_theta"
    assert bootstrap.observed_theta == summary.theta == Fraction(19, 80)
    assert summary.relative_improvement == bootstrap.observed_theta / summary.mu_b
    assert bootstrap == bootstrap_paired_theta(summary, samples=16, seed=DEFAULT_BOOTSTRAP_SEED)
    assert bootstrap.percentile_rule == FROZEN_PERCENTILE_RULE
    assert bootstrap.negative_count + bootstrap.zero_count + bootstrap.positive_count == 16
    assert bootstrap.support.order_count == 2
    assert bootstrap.support.graph_count == 4
    assert bootstrap.support.episode_count == 8
    assert [item.episodes_per_graph for item in bootstrap.support.by_order] == [(2, 2), (2, 2)]
    assert bootstrap.interval == (Fraction(3, 16), Fraction(189, 640))


def test_terminal_metric_gate_uses_relative_effect_and_bootstrap_lower_bound() -> None:
    summary = summarize_paired_areas(_episodes())
    bootstrap = bootstrap_paired_theta(summary, samples=16, seed=DEFAULT_BOOTSTRAP_SEED)

    checks = terminal_gate_checks(
        summary,
        bootstrap,
        minimum_relative_improvement=Fraction(1, 2),
        minimum_bootstrap_lower_bound=Fraction(3, 20),
    )
    assert checks == {
        "relative_improvement_at_least_threshold": True,
        "bootstrap_lower_bound_positive": True,
        "bootstrap_lower_bound_at_least_threshold": True,
    }
    assert terminal_gate_passes(
        summary,
        bootstrap,
        minimum_relative_improvement=Fraction(1, 2),
        minimum_bootstrap_lower_bound=Fraction(3, 20),
    )
    assert not terminal_gate_passes(
        summary,
        bootstrap,
        minimum_relative_improvement=Fraction(20, 37),
        minimum_bootstrap_lower_bound=Fraction(3, 20),
    )


def test_invalid_normalized_or_unpaired_curves_fail_closed() -> None:
    with pytest.raises(ValueError, match="same horizon"):
        paired_area_delta((0.1,), (0.1, 0.2))
    with pytest.raises(ValueError, match="best-so-far"):
        summarize_paired_areas(
            [PairedAreaEpisode(10, 1, "bad", (0.5, 0.4), (0.1, 0.2))]
        )
