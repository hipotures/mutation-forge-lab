"""Frozen hierarchical paired-area reduction and bootstrap for Stage 5.

Only compact, already-recorded curves enter this module.  It performs no graph
generation, policy execution, provider access, or network I/O.  Every policy
mean and every pairwise effect is reduced through the same hierarchy:
16 policy seeds -> 2 relabelings -> 16 graphs per order -> equal order mean.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from .stage5_config import CHAMPION_ID, RANDOM_ID, STAGE3_COMPARATOR_ID, STRUCTURAL_ID

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2_026_080_103
FROZEN_PERCENTILE_RULE = "linear_interpolation_at_p_times_n_minus_1"
EFFECT_STAGE3 = "C_vs_stage3"
EFFECT_RANDOM = "C_vs_random"
EFFECT_STRUCTURAL = "C_vs_structural"
EFFECTS = (EFFECT_STAGE3, EFFECT_RANDOM, EFFECT_STRUCTURAL)

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
        return Fraction(str(value))
    raise ValueError(f"{name} must be a finite number")


def _mean(values: Sequence[Fraction], name: str) -> Fraction:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    return sum(values, Fraction(0)) / len(values)


def curve_area(curve: Sequence[CurveValue], name: str = "curve") -> Fraction:
    if isinstance(curve, (str, bytes)) or not curve:
        raise ValueError(f"{name} must be a non-empty sequence")
    values = tuple(_fraction(item, name) for item in curve)
    previous = Fraction(0)
    for index, value in enumerate(values):
        if not Fraction(0) <= value <= Fraction(1):
            raise ValueError(f"{name}[{index}] must be normalized to [0, 1]")
        if index and value < previous:
            raise ValueError(f"{name} must be best-so-far (nondecreasing)")
        previous = value
    return _mean(values, name)


def paired_area_delta(candidate_curve: Sequence[CurveValue], comparator_curve: Sequence[CurveValue]) -> Fraction:
    if len(candidate_curve) != len(comparator_curve):
        raise ValueError("paired curves must have the same horizon")
    return curve_area(candidate_curve, "candidate curve") - curve_area(comparator_curve, "comparator curve")


@dataclass(frozen=True, slots=True)
class PolicyAreaEpisode:
    order: int
    graph_seed: int
    relabeling_seed: int
    policy_seed: int
    episode_id: str
    areas: Mapping[str, Fraction]


@dataclass(frozen=True, slots=True)
class RelabelAreaSummary:
    order: int
    graph_seed: int
    relabeling_seed: int
    policy_means: Mapping[str, Fraction]
    episode_count: int


@dataclass(frozen=True, slots=True)
class GraphAreaSummary:
    order: int
    graph_seed: int
    policy_means: Mapping[str, Fraction]
    relabel_count: int
    episode_count: int


@dataclass(frozen=True, slots=True)
class OrderAreaSummary:
    order: int
    policy_means: Mapping[str, Fraction]
    graph_count: int
    episode_count: int


@dataclass(frozen=True, slots=True)
class PairEffectSummary:
    effect: str
    candidate: str
    comparator: str
    theta: Fraction
    order_deltas: Mapping[int, Fraction]
    graph_deltas: Mapping[tuple[int, int], Fraction]
    relabel_deltas: Mapping[tuple[int, int, int], Fraction]
    stratum_deltas: Mapping[tuple[int, int], Fraction]
    sign_counts: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True, slots=True)
class Stage5Summary:
    policies: tuple[str, ...]
    episodes: tuple[PolicyAreaEpisode, ...]
    relabels: tuple[RelabelAreaSummary, ...]
    graphs: tuple[GraphAreaSummary, ...]
    orders: tuple[OrderAreaSummary, ...]
    policy_means: Mapping[str, Fraction]
    effects: Mapping[str, PairEffectSummary]
    structural_retention: Fraction

    @property
    def theta(self) -> Fraction:
        return self.effects[EFFECT_STAGE3].theta

    @property
    def mu_stage3(self) -> Fraction:
        return self.policy_means[STAGE3_COMPARATOR_ID]

    @property
    def relative_improvements(self) -> Mapping[str, Fraction]:
        return {
            EFFECT_STAGE3: self.effects[EFFECT_STAGE3].theta / self.mu_stage3,
            EFFECT_RANDOM: self.effects[EFFECT_RANDOM].theta / self.policy_means[RANDOM_ID],
            EFFECT_STRUCTURAL: self.effects[EFFECT_STRUCTURAL].theta / self.policy_means[STRUCTURAL_ID],
        }


def _effect(candidate: Mapping[str, Fraction], comparator: Mapping[str, Fraction]) -> Fraction:
    return candidate[CHAMPION_ID] - comparator[next(iter(comparator))]


def summarize_stage5(episodes: Iterable[PolicyAreaEpisode], policies: Sequence[str]) -> Stage5Summary:
    policy_ids = tuple(policies)
    if set(policy_ids) != {CHAMPION_ID, STAGE3_COMPARATOR_ID, RANDOM_ID, STRUCTURAL_ID}:
        raise ValueError("Stage 5 requires the frozen four-policy roster")
    records = tuple(episodes)
    if not records:
        raise ValueError("Stage 5 summary requires episodes")
    by_relabel: dict[tuple[int, int, int], list[PolicyAreaEpisode]] = {}
    seen: set[tuple[int, int, int, int]] = set()
    for record in records:
        key = (record.order, record.graph_seed, record.relabeling_seed, record.policy_seed)
        if key in seen:
            raise ValueError("duplicate Stage 5 episode identity")
        seen.add(key)
        if set(record.areas) != set(policy_ids):
            raise ValueError("Stage 5 policy roster mismatch")
        by_relabel.setdefault(key[:3], []).append(record)
    relabel_summaries: list[RelabelAreaSummary] = []
    for relabel_key, policy_seed_values in sorted(by_relabel.items()):
        relabel_summaries.append(
            RelabelAreaSummary(
                order=relabel_key[0],
                graph_seed=relabel_key[1],
                relabeling_seed=relabel_key[2],
                policy_means={
                    policy: _mean([item.areas[policy] for item in policy_seed_values], "policy-seed mean")
                    for policy in policy_ids
                },
                episode_count=len(policy_seed_values),
            )
        )
    by_graph: dict[tuple[int, int], list[RelabelAreaSummary]] = {}
    for item in relabel_summaries:
        by_graph.setdefault((item.order, item.graph_seed), []).append(item)
    graph_summaries: list[GraphAreaSummary] = []
    for graph_key, relabel_values in sorted(by_graph.items()):
        graph_summaries.append(
            GraphAreaSummary(
                order=graph_key[0],
                graph_seed=graph_key[1],
                policy_means={policy: _mean([item.policy_means[policy] for item in relabel_values], "relabel mean") for policy in policy_ids},
                relabel_count=len(relabel_values),
                episode_count=sum(item.episode_count for item in relabel_values),
            )
        )
    by_order: dict[int, list[GraphAreaSummary]] = {}
    for graph_summary in graph_summaries:
        by_order.setdefault(graph_summary.order, []).append(graph_summary)
    order_summaries: list[OrderAreaSummary] = []
    for order, graph_values in sorted(by_order.items()):
        order_summaries.append(
            OrderAreaSummary(
                order=order,
                policy_means={policy: _mean([item.policy_means[policy] for item in graph_values], "graph mean") for policy in policy_ids},
                graph_count=len(graph_values),
                episode_count=sum(item.episode_count for item in graph_values),
            )
        )
    policy_means = {
        policy: _mean([item.policy_means[policy] for item in order_summaries], "order mean")
        for policy in policy_ids
    }
    pair_specs = (
        (EFFECT_STAGE3, STAGE3_COMPARATOR_ID),
        (EFFECT_RANDOM, RANDOM_ID),
        (EFFECT_STRUCTURAL, STRUCTURAL_ID),
    )
    effects: dict[str, PairEffectSummary] = {}
    for effect_name, comparator in pair_specs:
        order_deltas = {
            item.order: item.policy_means[CHAMPION_ID] - item.policy_means[comparator]
            for item in order_summaries
        }
        graph_deltas = {
            (item.order, item.graph_seed): item.policy_means[CHAMPION_ID] - item.policy_means[comparator]
            for item in graph_summaries
        }
        relabel_deltas = {
            (item.order, item.graph_seed, item.relabeling_seed): item.policy_means[CHAMPION_ID] - item.policy_means[comparator]
            for item in relabel_summaries
        }
        stratum_values: dict[tuple[int, int], list[Fraction]] = {}
        for (order, _graph, relabel), value in relabel_deltas.items():
            stratum_values.setdefault((order, relabel), []).append(value)
        stratum_deltas = {
            key: _mean(values, "order-relabel stratum mean")
            for key, values in sorted(stratum_values.items())
        }
        episode_deltas = [item.areas[CHAMPION_ID] - item.areas[comparator] for item in records]
        sign_counts = {
            "episode": _sign_counts(episode_deltas),
            "relabel": _sign_counts(list(relabel_deltas.values())),
            "stratum": _sign_counts(list(stratum_deltas.values())),
            "graph": _sign_counts(list(graph_deltas.values())),
            "order": _sign_counts(list(order_deltas.values())),
        }
        effects[effect_name] = PairEffectSummary(
            effect=effect_name,
            candidate=CHAMPION_ID,
            comparator=comparator,
            theta=_mean(list(order_deltas.values()), "theta"),
            order_deltas=order_deltas,
            graph_deltas=graph_deltas,
            relabel_deltas=relabel_deltas,
            stratum_deltas=stratum_deltas,
            sign_counts=sign_counts,
        )
    structural = policy_means[STRUCTURAL_ID]
    if structural == 0:
        raise ValueError("structural mean is zero; retention is undefined")
    return Stage5Summary(
        policies=policy_ids,
        episodes=tuple(sorted(records, key=lambda item: item.episode_id)),
        relabels=tuple(relabel_summaries),
        graphs=tuple(graph_summaries),
        orders=tuple(order_summaries),
        policy_means=policy_means,
        effects=effects,
        structural_retention=policy_means[CHAMPION_ID] / structural,
    )


def _sign_counts(values: Sequence[Fraction]) -> dict[str, int]:
    return {
        "negative": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive": sum(value > 0 for value in values),
    }


def _percentile(values: Sequence[Fraction], probability: Fraction) -> Fraction:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = position.numerator // position.denominator
    upper = (position.numerator + position.denominator - 1) // position.denominator
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _sample_rng(seed: int, sample_index: int) -> random.Random:
    digest = hashlib.sha256(f"stage5:{seed}:{sample_index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True, slots=True)
class BootstrapSupport:
    order_count: int
    graph_count_per_order: int
    relabel_count_per_graph: int
    policy_seed_count_per_relabel: int
    episode_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "order_count": self.order_count,
            "graph_count_per_order": self.graph_count_per_order,
            "relabel_count_per_graph": self.relabel_count_per_graph,
            "policy_seed_count_per_relabel": self.policy_seed_count_per_relabel,
            "episode_count": self.episode_count,
        }


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    samples: int
    seed: int
    confidence_level: Fraction
    percentile_rule: str
    observed: Mapping[str, Fraction]
    intervals: Mapping[str, tuple[Fraction, Fraction]]
    sign_counts: Mapping[str, Mapping[str, int]]
    support: BootstrapSupport
    draw_support: Mapping[str, tuple[tuple[Fraction, int], ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimand": "stage5_hierarchical_paired_area_theta",
            "samples": self.samples,
            "seed": self.seed,
            "confidence_level": float(self.confidence_level),
            "percentile_rule": self.percentile_rule,
            "observed": {key: fraction_payload(value) for key, value in self.observed.items()},
            "intervals": {key: [fraction_payload(value) for value in interval] for key, interval in self.intervals.items()},
            "sign_counts": dict(self.sign_counts),
            "support": self.support.as_dict(),
            "draw_support": {
                key: [{"value": float(value), "value_fraction": fraction_text(value), "count": count} for value, count in values]
                for key, values in self.draw_support.items()
            },
        }


def fraction_payload(value: Fraction) -> dict[str, float | str]:
    return {"value": float(value), "fraction": fraction_text(value)}


def _bootstrap_draw(summary: Stage5Summary, rng: random.Random) -> dict[str, Fraction]:
    by_order_graph: dict[int, dict[int, dict[int, dict[int, PolicyAreaEpisode]]]] = {}
    for item in summary.episodes:
        by_order_graph.setdefault(item.order, {}).setdefault(item.graph_seed, {}).setdefault(item.relabeling_seed, {})[item.policy_seed] = item
    policy_means: dict[str, list[Fraction]] = {policy: [] for policy in summary.policies}
    for order in sorted(by_order_graph):
        graph_ids = sorted(by_order_graph[order])
        order_graph_means: dict[str, list[Fraction]] = {policy: [] for policy in summary.policies}
        for _ in graph_ids:
            graph_seed = graph_ids[rng.randrange(len(graph_ids))]
            relabels = by_order_graph[order][graph_seed]
            graph_relabel_means: dict[str, list[Fraction]] = {policy: [] for policy in summary.policies}
            relabel_ids = sorted(relabels)
            for _ in relabel_ids:
                relabel_seed = relabel_ids[rng.randrange(len(relabel_ids))]
                policy_map = relabels[relabel_seed]
                seed_ids = sorted(policy_map)
                sampled = [seed_ids[rng.randrange(len(seed_ids))] for _ in seed_ids]
                for policy in summary.policies:
                    graph_relabel_means[policy].append(_mean([policy_map[seed].areas[policy] for seed in sampled], "bootstrap policy mean"))
            for policy in summary.policies:
                order_graph_means[policy].append(_mean(graph_relabel_means[policy], "bootstrap relabel mean"))
        for policy in summary.policies:
            order_mean = _mean(order_graph_means[policy], "bootstrap graph mean")
            policy_means[policy].append(order_mean)
    theta = {
        EFFECT_STAGE3: _mean(policy_means[CHAMPION_ID], "bootstrap theta") - _mean(policy_means[STAGE3_COMPARATOR_ID], "bootstrap theta"),
        EFFECT_RANDOM: _mean(policy_means[CHAMPION_ID], "bootstrap theta") - _mean(policy_means[RANDOM_ID], "bootstrap theta"),
        EFFECT_STRUCTURAL: _mean(policy_means[CHAMPION_ID], "bootstrap theta") - _mean(policy_means[STRUCTURAL_ID], "bootstrap theta"),
    }
    return theta


def bootstrap_stage5(
    summary: Stage5Summary,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: CurveValue = Fraction(95, 100),
) -> BootstrapSummary:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    confidence = _fraction(confidence_level, "confidence_level")
    if not Fraction(0) < confidence < Fraction(1):
        raise ValueError("confidence_level must be in (0, 1)")
    draws: dict[str, list[Fraction]] = {effect: [] for effect in EFFECTS}
    for sample_index in range(samples):
        draw = _bootstrap_draw(summary, _sample_rng(seed, sample_index))
        for effect in EFFECTS:
            draws[effect].append(draw[effect])
    alpha = (Fraction(1) - confidence) / 2
    intervals = {effect: (_percentile(draws[effect], alpha), _percentile(draws[effect], Fraction(1) - alpha)) for effect in EFFECTS}
    observed = {effect: summary.effects[effect].theta for effect in EFFECTS}
    return BootstrapSummary(
        samples=samples,
        seed=seed,
        confidence_level=confidence,
        percentile_rule=FROZEN_PERCENTILE_RULE,
        observed=observed,
        intervals=intervals,
        sign_counts={effect: _sign_counts(draws[effect]) for effect in EFFECTS},
        support=BootstrapSupport(
            order_count=len(summary.orders),
            graph_count_per_order=len([item for item in summary.graphs if item.order == summary.orders[0].order]),
            relabel_count_per_graph=len([item for item in summary.relabels if item.order == summary.orders[0].order and item.graph_seed == summary.graphs[0].graph_seed]),
            policy_seed_count_per_relabel=len([item for item in summary.episodes if item.order == summary.orders[0].order and item.graph_seed == summary.graphs[0].graph_seed and item.relabeling_seed == summary.relabels[0].relabeling_seed]),
            episode_count=len(summary.episodes),
        ),
        draw_support={effect: tuple(sorted(Counter(draws[effect]).items(), key=lambda item: item[0])) for effect in EFFECTS},
    )


def gate_checks(
    summary: Stage5Summary,
    bootstrap: BootstrapSummary,
    *,
    champion_stage3_threshold: CurveValue,
    champion_random_threshold: CurveValue,
    structural_retention_threshold: CurveValue,
) -> dict[str, bool]:
    if any(bootstrap.observed[effect] != summary.effects[effect].theta for effect in EFFECTS):
        raise ValueError("bootstrap observed effects do not match the paired-area summary")
    relative = summary.relative_improvements
    stage3_order = all(value >= 0 for value in summary.effects[EFFECT_STAGE3].order_deltas.values())
    stage3_strata = all(value >= 0 for value in summary.effects[EFFECT_STAGE3].stratum_deltas.values())
    return {
        "relative_improvement_C_vs_stage3_at_least_threshold": relative[EFFECT_STAGE3] >= _fraction(champion_stage3_threshold, "stage3 threshold"),
        "bootstrap_C_vs_stage3_lower_bound_positive": bootstrap.intervals[EFFECT_STAGE3][0] > 0,
        "C_vs_stage3_nonnegative_each_order": stage3_order,
        "C_vs_stage3_nonnegative_all_six_order_relabel_strata": stage3_strata,
        "relative_improvement_C_vs_random_at_least_threshold": relative[EFFECT_RANDOM] >= _fraction(champion_random_threshold, "random threshold"),
        "bootstrap_C_vs_random_lower_bound_positive": bootstrap.intervals[EFFECT_RANDOM][0] > 0,
        "structural_retention_at_least_threshold": summary.structural_retention >= _fraction(structural_retention_threshold, "structural threshold"),
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EFFECTS",
    "EFFECT_RANDOM",
    "EFFECT_STAGE3",
    "EFFECT_STRUCTURAL",
    "FROZEN_PERCENTILE_RULE",
    "BootstrapSummary",
    "BootstrapSupport",
    "GraphAreaSummary",
    "OrderAreaSummary",
    "PairEffectSummary",
    "PolicyAreaEpisode",
    "RelabelAreaSummary",
    "Stage5Summary",
    "bootstrap_stage5",
    "curve_area",
    "fraction_payload",
    "fraction_text",
    "gate_checks",
    "paired_area_delta",
    "summarize_stage5",
]
