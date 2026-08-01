"""Deterministic statistics for the Stage 4E paired-area estimand.

This module reduces already-recorded normalized best-so-far curves.  It never
executes a policy, generates a graph, or contacts a provider.  All reduction
arithmetic is held as :class:`fractions.Fraction` until a caller explicitly
serializes a result, so the point estimate, bootstrap statistic, and relative
effect share one definition.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026080102
DEFAULT_CONFIDENCE_LEVEL = Fraction(95, 100)
FROZEN_PERCENTILE_RULE = "linear_interpolation_at_p_times_n_minus_1"

type CurveValue = int | float | Decimal | Fraction


def fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _fraction(value: CurveValue, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be a finite number")
        return Fraction(value)
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        # Decimal text is the public value carried by JSON evidence.  Do not
        # smuggle a binary floating-point approximation into the estimand.
        return Fraction(str(value))
    raise ValueError(f"{name} must be a finite number")


def _probability(value: CurveValue, name: str) -> Fraction:
    result = _fraction(value, name)
    if not Fraction(0) < result < Fraction(1):
        raise ValueError(f"{name} must be in (0, 1)")
    return result


def _mean(values: Sequence[Fraction], name: str) -> Fraction:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    return sum(values, Fraction(0)) / len(values)


def _curve_area(curve: Sequence[CurveValue], name: str) -> Fraction:
    if isinstance(curve, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    values = tuple(_fraction(value, name) for value in curve)
    if not values:
        raise ValueError(f"{name} must not be empty")
    previous = Fraction(0)
    for index, value in enumerate(values):
        if not Fraction(0) <= value <= Fraction(1):
            raise ValueError(f"{name}[{index}] must be normalized to [0, 1]")
        if index and value < previous:
            raise ValueError(f"{name} must be best-so-far (nondecreasing)")
        previous = value
    return _mean(values, name)


def paired_area_delta(
    candidate_curve: Sequence[CurveValue], comparator_curve: Sequence[CurveValue]
) -> Fraction:
    """Return exact ``d(e)``: paired candidate-minus-comparator curve area."""
    if len(candidate_curve) != len(comparator_curve):
        raise ValueError("paired curves must have the same horizon")
    candidate = _curve_area(candidate_curve, "candidate curve")
    comparator = _curve_area(comparator_curve, "comparator curve")
    return candidate - comparator


@dataclass(frozen=True, slots=True)
class PairedAreaEpisode:
    """One policy-seed pair on one fixed graph."""

    order: int
    graph_seed: int
    episode_id: str
    candidate_curve: Sequence[CurveValue]
    comparator_curve: Sequence[CurveValue]


@dataclass(frozen=True, slots=True)
class EpisodeAreaSummary:
    order: int
    graph_seed: int
    episode_id: str
    candidate_area: Fraction
    comparator_area: Fraction
    delta: Fraction

    def as_dict(self) -> dict[str, int | str | float]:
        return {
            "order": self.order,
            "graph_seed": self.graph_seed,
            "episode_id": self.episode_id,
            "candidate_area": float(self.candidate_area),
            "comparator_area": float(self.comparator_area),
            "d_e": float(self.delta),
        }


@dataclass(frozen=True, slots=True)
class GraphAreaSummary:
    order: int
    graph_seed: int
    episode_count: int
    candidate_mean: Fraction
    comparator_mean: Fraction
    delta_mean: Fraction

    def as_dict(self) -> dict[str, int | float]:
        return {
            "order": self.order,
            "graph_seed": self.graph_seed,
            "episode_count": self.episode_count,
            "candidate_mean": float(self.candidate_mean),
            "comparator_mean": float(self.comparator_mean),
            "delta_mean": float(self.delta_mean),
        }


@dataclass(frozen=True, slots=True)
class OrderAreaSummary:
    order: int
    graph_count: int
    episode_count: int
    candidate_mean: Fraction
    comparator_mean: Fraction
    delta_mean: Fraction

    def as_dict(self) -> dict[str, int | float]:
        return {
            "order": self.order,
            "graph_count": self.graph_count,
            "episode_count": self.episode_count,
            "candidate_mean": float(self.candidate_mean),
            "comparator_mean": float(self.comparator_mean),
            "delta_mean": float(self.delta_mean),
        }


@dataclass(frozen=True, slots=True)
class PairedAreaSummary:
    """The hierarchy graph means -> equal-order means -> theta."""

    episodes: tuple[EpisodeAreaSummary, ...]
    graphs: tuple[GraphAreaSummary, ...]
    orders: tuple[OrderAreaSummary, ...]
    theta: Fraction
    mu_b: Fraction
    relative_improvement: Fraction

    @property
    def mu_B(self) -> Fraction:
        """Compatibility spelling for the preregistered comparator mean."""
        return self.mu_b

    def as_dict(self) -> dict[str, object]:
        return {
            "estimand": "paired_transition_aware_area_delta",
            "episode_count": len(self.episodes),
            "graph_count": len(self.graphs),
            "order_count": len(self.orders),
            "theta": float(self.theta),
            "mu_B": float(self.mu_b),
            "relative_improvement": float(self.relative_improvement),
            "orders": [order.as_dict() for order in self.orders],
            "graphs": [graph.as_dict() for graph in self.graphs],
            "episodes": [episode.as_dict() for episode in self.episodes],
        }


def summarize_paired_areas(episodes: Iterable[PairedAreaEpisode]) -> PairedAreaSummary:
    """Reduce paired curves with equal weight for each graph and each order."""
    records = tuple(episodes)
    if not records:
        raise ValueError("paired-area summary requires at least one episode")
    by_graph: dict[tuple[int, int], list[EpisodeAreaSummary]] = {}
    seen: set[tuple[int, int, str]] = set()
    for record in records:
        if isinstance(record.order, bool) or not isinstance(record.order, int) or record.order < 1:
            raise ValueError("episode order must be a positive integer")
        if isinstance(record.graph_seed, bool) or not isinstance(record.graph_seed, int):
            raise ValueError("episode graph_seed must be an integer")
        if not isinstance(record.episode_id, str) or not record.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        key = (record.order, record.graph_seed, record.episode_id)
        if key in seen:
            raise ValueError("duplicate paired-area episode")
        seen.add(key)
        if len(record.candidate_curve) != len(record.comparator_curve):
            raise ValueError("paired curves must have the same horizon")
        candidate_area = _curve_area(record.candidate_curve, "candidate curve")
        comparator_area = _curve_area(record.comparator_curve, "comparator curve")
        episode = EpisodeAreaSummary(
            order=record.order,
            graph_seed=record.graph_seed,
            episode_id=record.episode_id,
            candidate_area=candidate_area,
            comparator_area=comparator_area,
            delta=candidate_area - comparator_area,
        )
        by_graph.setdefault((record.order, record.graph_seed), []).append(episode)

    episode_summaries = tuple(
        sorted(
            (episode for values in by_graph.values() for episode in values),
            key=lambda item: (item.order, item.graph_seed, item.episode_id),
        )
    )
    graph_summaries: list[GraphAreaSummary] = []
    for (order, graph_seed), values in sorted(by_graph.items()):
        graph_summaries.append(
            GraphAreaSummary(
                order=order,
                graph_seed=graph_seed,
                episode_count=len(values),
                candidate_mean=_mean([item.candidate_area for item in values], "graph candidate"),
                comparator_mean=_mean(
                    [item.comparator_area for item in values], "graph comparator"
                ),
                delta_mean=_mean([item.delta for item in values], "graph delta"),
            )
        )
    by_order: dict[int, list[GraphAreaSummary]] = {}
    for graph in graph_summaries:
        by_order.setdefault(graph.order, []).append(graph)
    order_summaries = tuple(
        OrderAreaSummary(
            order=order,
            graph_count=len(graphs),
            episode_count=sum(graph.episode_count for graph in graphs),
            candidate_mean=_mean([graph.candidate_mean for graph in graphs], "order candidate"),
            comparator_mean=_mean(
                [graph.comparator_mean for graph in graphs], "order comparator"
            ),
            delta_mean=_mean([graph.delta_mean for graph in graphs], "order delta"),
        )
        for order, graphs in sorted(by_order.items())
    )
    theta = _mean([order.delta_mean for order in order_summaries], "theta")
    mu_b = _mean([order.comparator_mean for order in order_summaries], "mu_B")
    if mu_b == 0:
        raise ValueError("mu_B is zero; relative improvement is undefined")
    return PairedAreaSummary(
        episodes=episode_summaries,
        graphs=tuple(graph_summaries),
        orders=order_summaries,
        theta=theta,
        mu_b=mu_b,
        relative_improvement=theta / mu_b,
    )


def _percentile(values: Sequence[Fraction], probability: Fraction) -> Fraction:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = position.numerator // position.denominator
    upper = (position.numerator + position.denominator - 1) // position.denominator
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _sample_rng(seed: int, sample_index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{sample_index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _bootstrap_theta_draws(summary: PairedAreaSummary, samples: int, seed: int) -> list[Fraction]:
    by_order: dict[int, list[GraphAreaSummary]] = {}
    episodes_by_graph: dict[tuple[int, int], list[EpisodeAreaSummary]] = {}
    for graph in summary.graphs:
        by_order.setdefault(graph.order, []).append(graph)
    for episode in summary.episodes:
        episodes_by_graph.setdefault((episode.order, episode.graph_seed), []).append(episode)
    draws: list[Fraction] = []
    for sample_index in range(samples):
        rng = _sample_rng(seed, sample_index)
        order_means: list[Fraction] = []
        for order in sorted(by_order):
            graphs = by_order[order]
            resampled_graph_means: list[Fraction] = []
            for _ in graphs:
                graph = graphs[rng.randrange(len(graphs))]
                graph_episodes = episodes_by_graph[(graph.order, graph.graph_seed)]
                resampled_graph_means.append(
                    _mean(
                        [
                            graph_episodes[rng.randrange(len(graph_episodes))].delta
                            for _ in graph_episodes
                        ],
                        "bootstrap graph delta",
                    )
                )
            order_means.append(_mean(resampled_graph_means, "bootstrap order delta"))
        draws.append(_mean(order_means, "bootstrap theta"))
    return draws


@dataclass(frozen=True, slots=True)
class BootstrapOrderSupport:
    order: int
    graph_count: int
    episode_count: int
    episodes_per_graph: tuple[int, ...]

    def as_dict(self) -> dict[str, int | list[int]]:
        return {
            "order": self.order,
            "graph_count": self.graph_count,
            "episode_count": self.episode_count,
            "episodes_per_graph": list(self.episodes_per_graph),
        }


@dataclass(frozen=True, slots=True)
class BootstrapSupport:
    order_count: int
    graph_count: int
    episode_count: int
    by_order: tuple[BootstrapOrderSupport, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "order_count": self.order_count,
            "graph_count": self.graph_count,
            "episode_count": self.episode_count,
            "by_order": [item.as_dict() for item in self.by_order],
        }


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    estimand: str
    samples: int
    seed: int
    confidence_level: Fraction
    percentile_rule: str
    observed_theta: Fraction
    interval: tuple[Fraction, Fraction]
    negative_count: int
    zero_count: int
    positive_count: int
    support: BootstrapSupport
    draw_support: tuple[tuple[Fraction, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "estimand": self.estimand,
            "samples": self.samples,
            "seed": self.seed,
            "confidence_level": float(self.confidence_level),
            "percentile_rule": self.percentile_rule,
            "observed_theta": float(self.observed_theta),
            "interval": [float(value) for value in self.interval],
            "interval_fraction": [fraction_text(value) for value in self.interval],
            "sign_counts": {
                "negative": self.negative_count,
                "zero": self.zero_count,
                "positive": self.positive_count,
            },
            "support": self.support.as_dict(),
            "draw_support": [
                {"value": float(value), "value_fraction": fraction_text(value), "count": count}
                for value, count in self.draw_support
            ],
        }


def bootstrap_paired_theta(
    summary: PairedAreaSummary,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: CurveValue = DEFAULT_CONFIDENCE_LEVEL,
) -> BootstrapSummary:
    """Graph-then-policy cluster bootstrap for the exact paired-area ``theta``."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    confidence = _probability(confidence_level, "confidence_level")
    draws = _bootstrap_theta_draws(summary, samples, seed)
    alpha = (Fraction(1) - confidence) / 2
    by_order: list[BootstrapOrderSupport] = []
    for order in summary.orders:
        graph_episodes = [
            graph.episode_count for graph in summary.graphs if graph.order == order.order
        ]
        by_order.append(
            BootstrapOrderSupport(
                order=order.order,
                graph_count=order.graph_count,
                episode_count=order.episode_count,
                episodes_per_graph=tuple(graph_episodes),
            )
        )
    return BootstrapSummary(
        estimand="paired_transition_aware_area_delta_theta",
        samples=samples,
        seed=seed,
        confidence_level=confidence,
        percentile_rule=FROZEN_PERCENTILE_RULE,
        observed_theta=summary.theta,
        interval=(_percentile(draws, alpha), _percentile(draws, Fraction(1) - alpha)),
        negative_count=sum(value < 0 for value in draws),
        zero_count=sum(value == 0 for value in draws),
        positive_count=sum(value > 0 for value in draws),
        support=BootstrapSupport(
            order_count=len(summary.orders),
            graph_count=len(summary.graphs),
            episode_count=len(summary.episodes),
            by_order=tuple(by_order),
        ),
        draw_support=tuple(sorted(Counter(draws).items(), key=lambda item: item[0])),
    )


def terminal_gate_checks(
    summary: PairedAreaSummary,
    bootstrap: BootstrapSummary,
    *,
    minimum_relative_improvement: CurveValue,
    minimum_bootstrap_lower_bound: CurveValue = 0,
) -> dict[str, bool]:
    """Return the metric-only terminal checks for a preregistered threshold."""
    relative_threshold = _fraction(minimum_relative_improvement, "relative threshold")
    lower_threshold = _fraction(minimum_bootstrap_lower_bound, "bootstrap threshold")
    if bootstrap.observed_theta != summary.theta:
        raise ValueError("bootstrap observed theta does not match paired-area summary")
    return {
        "relative_improvement_at_least_threshold": (
            summary.relative_improvement >= relative_threshold
        ),
        "bootstrap_lower_bound_positive": bootstrap.interval[0] > 0,
        "bootstrap_lower_bound_at_least_threshold": bootstrap.interval[0] >= lower_threshold,
    }


def terminal_gate_passes(
    summary: PairedAreaSummary,
    bootstrap: BootstrapSummary,
    *,
    minimum_relative_improvement: CurveValue,
    minimum_bootstrap_lower_bound: CurveValue = 0,
) -> bool:
    return all(
        terminal_gate_checks(
            summary,
            bootstrap,
            minimum_relative_improvement=minimum_relative_improvement,
            minimum_bootstrap_lower_bound=minimum_bootstrap_lower_bound,
        ).values()
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "FROZEN_PERCENTILE_RULE",
    "BootstrapOrderSupport",
    "BootstrapSummary",
    "BootstrapSupport",
    "CurveValue",
    "EpisodeAreaSummary",
    "GraphAreaSummary",
    "OrderAreaSummary",
    "PairedAreaEpisode",
    "PairedAreaSummary",
    "bootstrap_paired_theta",
    "paired_area_delta",
    "summarize_paired_areas",
    "terminal_gate_checks",
    "terminal_gate_passes",
    "fraction_text",
]
