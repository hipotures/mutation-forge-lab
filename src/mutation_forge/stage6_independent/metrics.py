# ruff: noqa: E501, UP040
"""Independent recomputation of the frozen Stage 5 area metrics.

This module intentionally has no dependency on the Stage 5 implementation.  It
accepts the compact records emitted by an execution run, validates and reduces
their curves with :class:`fractions.Fraction`, and exposes the same frozen
hierarchical estimand (policy seed -> relabel -> graph -> order).  It is useful
for an independent verification process and performs no execution or I/O.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, TypeAlias

# These identifiers are duplicated deliberately: importing Stage 5 config (or
# any other Stage 5 module) would defeat the purpose of an independent path.
CHAMPION_ID = "program-d5ad1c8203e0d9f25f03aabd"
STAGE3_COMPARATOR_ID = "candidate-slot-04"
RANDOM_ID = "random"
STRUCTURAL_ID = "structural"
POLICY_IDS = (CHAMPION_ID, STAGE3_COMPARATOR_ID, RANDOM_ID, STRUCTURAL_ID)

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2_026_080_103
# Explicit name used by the Stage 5 comparison envelope and by callers that
# want to assert the frozen draw count.
BOOTSTRAP_SAMPLES = DEFAULT_BOOTSTRAP_SAMPLES
DEFAULT_CONFIDENCE_LEVEL = Fraction(95, 100)
DEFAULT_CHAMPION_STAGE3_THRESHOLD = Fraction(2, 100)
DEFAULT_CHAMPION_RANDOM_THRESHOLD = Fraction(5, 100)
DEFAULT_STRUCTURAL_RETENTION_THRESHOLD = Fraction(99, 100)
FROZEN_PERCENTILE_RULE = "linear_interpolation_at_p_times_n_minus_1"
EFFECT_STAGE3 = "C_vs_stage3"
EFFECT_RANDOM = "C_vs_random"
EFFECT_STRUCTURAL = "C_vs_structural"
EFFECTS = (EFFECT_STAGE3, EFFECT_RANDOM, EFFECT_STRUCTURAL)

CurveValue: TypeAlias = int | float | Decimal | Fraction


def fraction_text(value: Fraction) -> str:
    """Return the canonical ``numerator`` or ``numerator/denominator`` text."""

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
        # Decimal text conversion is the frozen Stage 5 rule for floats.
        return Fraction(str(value))
    raise ValueError(f"{name} must be a finite number")


def _mean(values: Sequence[Fraction], name: str) -> Fraction:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    return sum(values, Fraction(0)) / len(values)


def curve_area(curve: Sequence[CurveValue], name: str = "curve") -> Fraction:
    """Validate a normalized best-so-far curve and return its exact mean area."""

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "graph_seed": self.graph_seed,
            "relabeling_seed": self.relabeling_seed,
            "policy_seed": self.policy_seed,
            "episode_id": self.episode_id,
            "areas": {key: fraction_payload(value) for key, value in self.areas.items()},
        }


@dataclass(frozen=True, slots=True)
class RelabelAreaSummary:
    order: int
    graph_seed: int
    relabeling_seed: int
    policy_means: Mapping[str, Fraction]
    episode_count: int

    def as_dict(self) -> dict[str, Any]:
        return {"order": self.order, "graph_seed": self.graph_seed, "relabeling_seed": self.relabeling_seed,
                "policy_means": {k: fraction_payload(v) for k, v in self.policy_means.items()},
                "episode_count": self.episode_count}


@dataclass(frozen=True, slots=True)
class GraphAreaSummary:
    order: int
    graph_seed: int
    policy_means: Mapping[str, Fraction]
    relabel_count: int
    episode_count: int

    def as_dict(self) -> dict[str, Any]:
        return {"order": self.order, "graph_seed": self.graph_seed,
                "policy_means": {k: fraction_payload(v) for k, v in self.policy_means.items()},
                "relabel_count": self.relabel_count, "episode_count": self.episode_count}


@dataclass(frozen=True, slots=True)
class OrderAreaSummary:
    order: int
    policy_means: Mapping[str, Fraction]
    graph_count: int
    episode_count: int

    def as_dict(self) -> dict[str, Any]:
        return {"order": self.order, "policy_means": {k: fraction_payload(v) for k, v in self.policy_means.items()},
                "graph_count": self.graph_count, "episode_count": self.episode_count}


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "effect": self.effect, "candidate": self.candidate, "comparator": self.comparator,
            "theta": fraction_payload(self.theta),
            "order_deltas": {str(k): fraction_payload(v) for k, v in self.order_deltas.items()},
            "graph_deltas": {f"{k[0]},{k[1]}": fraction_payload(v) for k, v in self.graph_deltas.items()},
            "relabel_deltas": {f"{k[0]},{k[1]},{k[2]}": fraction_payload(v) for k, v in self.relabel_deltas.items()},
            "stratum_deltas": {f"{k[0]},{k[1]}": fraction_payload(v) for k, v in self.stratum_deltas.items()},
            "sign_counts": {k: dict(v) for k, v in self.sign_counts.items()},
        }


@dataclass(frozen=True, slots=True)
class MetricsSummary:
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "policies": list(self.policies), "episodes": [e.as_dict() for e in self.episodes],
            "relabels": [e.as_dict() for e in self.relabels], "graphs": [e.as_dict() for e in self.graphs],
            "orders": [e.as_dict() for e in self.orders],
            "policy_means": {k: fraction_payload(v) for k, v in self.policy_means.items()},
            "effects": {k: v.as_dict() for k, v in self.effects.items()},
            "structural_retention": fraction_payload(self.structural_retention),
            "relative_improvements": {k: fraction_payload(v) for k, v in self.relative_improvements.items()},
        }


# Backwards-friendly aliases make the result pleasant to consume without
# tying callers to the Stage 5 class name.
Summary = MetricsSummary
Stage5Summary = MetricsSummary
Episode = PolicyAreaEpisode
RelabelSummary = RelabelAreaSummary
GraphSummary = GraphAreaSummary
OrderSummary = OrderAreaSummary
EffectSummary = PairEffectSummary


def _sign_counts(values: Sequence[Fraction]) -> dict[str, int]:
    return {"negative": sum(v < 0 for v in values), "zero": sum(v == 0 for v in values), "positive": sum(v > 0 for v in values)}


def _policy_curve(value: object, policy: str) -> Sequence[CurveValue]:
    # Compact records use {normalized_best_so_far_curve: [...]}; accepting a
    # direct sequence also makes recomputation convenient for hand-authored data.
    if isinstance(value, Mapping):
        for key in ("normalized_best_so_far_curve", "curve", "values"):
            if key in value:
                value = value[key]
                break
        else:
            raise ValueError(f"{policy} policy row has no normalized curve")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{policy} curve must be a sequence")
    return value


def parse_metrics_episodes(rows: Iterable[Mapping[str, Any]], policy_ids: Sequence[str] = POLICY_IDS) -> tuple[PolicyAreaEpisode, ...]:
    """Parse compact execution rows into exact-area episodes.

    ``rows`` are not mutated.  Each row must contain hierarchy keys and a
    ``policies`` mapping whose values are either curve arrays or objects with a
    ``normalized_best_so_far_curve`` array.
    """

    ids = tuple(policy_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("policy_ids must be a non-empty sequence of unique IDs")
    parsed: list[PolicyAreaEpisode] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"metrics row {index} must be an object")
        # Some evidence records wrap the compact payload under ``metrics_input``;
        # accept that representation while retaining the same validation.
        nested = row.get("metrics_input")
        source = nested if isinstance(nested, Mapping) else row
        policies = row.get("policies", row.get("policy_curves"))
        if policies is None:
            policies = source.get("policies", source.get("policy_curves"))
        if not isinstance(policies, Mapping):
            raise ValueError("compact metrics row has no policy rows")
        if set(policies) != set(ids):
            raise ValueError("policy roster mismatch")
        try:
            order = int(row["order"] if "order" in row else source["order"])
            graph_seed = int(row["graph_seed"] if "graph_seed" in row else source["graph_seed"])
            relabeling_seed = int(row["relabeling_seed"] if "relabeling_seed" in row else source["relabeling_seed"])
            policy_seed = int(row["policy_seed"] if "policy_seed" in row else source["policy_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("metrics row is missing hierarchy identity") from exc
        episode_id = str(row.get("episode_id", source.get(
            "episode_id", f"o{order:02d}-g{graph_seed:04d}-r{relabeling_seed:04d}-p{policy_seed:04d}")))
        areas = {policy: curve_area(_policy_curve(policies[policy], policy), f"{policy} curve") for policy in ids}
        parsed.append(PolicyAreaEpisode(order, graph_seed, relabeling_seed, policy_seed, episode_id, areas))
    return tuple(parsed)


def summarize(episodes: Iterable[PolicyAreaEpisode], policy_ids: Sequence[str] = POLICY_IDS) -> MetricsSummary:
    """Reduce episodes through policy-seed, relabel, graph, and order means."""

    ids = tuple(policy_ids)
    if set(ids) != set(POLICY_IDS) or len(ids) != len(POLICY_IDS):
        raise ValueError("the frozen four-policy roster is required")
    records = tuple(episodes)
    if not records:
        raise ValueError("summary requires episodes")
    by_relabel: dict[tuple[int, int, int], list[PolicyAreaEpisode]] = {}
    seen: set[tuple[int, int, int, int]] = set()
    for record in records:
        key = (record.order, record.graph_seed, record.relabeling_seed, record.policy_seed)
        if key in seen:
            raise ValueError("duplicate episode identity")
        seen.add(key)
        if set(record.areas) != set(ids):
            raise ValueError("policy roster mismatch")
        by_relabel.setdefault(key[:3], []).append(record)
    relabel_summaries = [RelabelAreaSummary(o, g, r,
        {p: _mean([x.areas[p] for x in vals], "policy-seed mean") for p in ids}, len(vals))
        for (o, g, r), vals in sorted(by_relabel.items())]
    by_graph: dict[tuple[int, int], list[RelabelAreaSummary]] = {}
    for value in relabel_summaries:
        by_graph.setdefault((value.order, value.graph_seed), []).append(value)
    graph_summaries = [GraphAreaSummary(o, g,
        {p: _mean([x.policy_means[p] for x in vals], "relabel mean") for p in ids}, len(vals),
        sum(x.episode_count for x in vals)) for (o, g), vals in sorted(by_graph.items())]
    by_order: dict[int, list[GraphAreaSummary]] = {}
    for value in graph_summaries:
        by_order.setdefault(value.order, []).append(value)
    order_summaries = [OrderAreaSummary(o,
        {p: _mean([x.policy_means[p] for x in vals], "graph mean") for p in ids}, len(vals),
        sum(x.episode_count for x in vals)) for o, vals in sorted(by_order.items())]
    policy_means = {p: _mean([x.policy_means[p] for x in order_summaries], "order mean") for p in ids}
    effects: dict[str, PairEffectSummary] = {}
    for effect_name, comparator in ((EFFECT_STAGE3, STAGE3_COMPARATOR_ID), (EFFECT_RANDOM, RANDOM_ID), (EFFECT_STRUCTURAL, STRUCTURAL_ID)):
        order_deltas = {x.order: x.policy_means[CHAMPION_ID] - x.policy_means[comparator] for x in order_summaries}
        graph_deltas = {(x.order, x.graph_seed): x.policy_means[CHAMPION_ID] - x.policy_means[comparator] for x in graph_summaries}
        relabel_deltas = {(x.order, x.graph_seed, x.relabeling_seed): x.policy_means[CHAMPION_ID] - x.policy_means[comparator] for x in relabel_summaries}
        strata: dict[tuple[int, int], list[Fraction]] = {}
        for (o, _g, r), delta in relabel_deltas.items():
            strata.setdefault((o, r), []).append(delta)
        stratum_deltas = {key: _mean(vals, "order-relabel stratum mean") for key, vals in sorted(strata.items())}
        signs = {"episode": _sign_counts([x.areas[CHAMPION_ID] - x.areas[comparator] for x in records]),
                 "relabel": _sign_counts(list(relabel_deltas.values())), "stratum": _sign_counts(list(stratum_deltas.values())),
                 "graph": _sign_counts(list(graph_deltas.values())), "order": _sign_counts(list(order_deltas.values()))}
        effects[effect_name] = PairEffectSummary(effect_name, CHAMPION_ID, comparator, _mean(list(order_deltas.values()), "theta"),
                                                   order_deltas, graph_deltas, relabel_deltas, stratum_deltas, signs)
    structural = policy_means[STRUCTURAL_ID]
    if structural == 0:
        raise ValueError("structural mean is zero; retention is undefined")
    return MetricsSummary(ids, tuple(sorted(records, key=lambda x: x.episode_id)), tuple(relabel_summaries), tuple(graph_summaries), tuple(order_summaries), policy_means, effects, policy_means[CHAMPION_ID] / structural)


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
        return {"order_count": self.order_count, "graph_count_per_order": self.graph_count_per_order,
                "relabel_count_per_graph": self.relabel_count_per_graph,
                "policy_seed_count_per_relabel": self.policy_seed_count_per_relabel,
                "episode_count": self.episode_count}


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
        return {"estimand": "stage5_hierarchical_paired_area_theta", "samples": self.samples, "seed": self.seed,
                "confidence_level": fraction_payload(self.confidence_level), "percentile_rule": self.percentile_rule,
                "observed": {k: fraction_payload(v) for k, v in self.observed.items()},
                "intervals": {k: [fraction_payload(v) for v in pair] for k, pair in self.intervals.items()},
                "sign_counts": {k: dict(v) for k, v in self.sign_counts.items()}, "support": self.support.as_dict(),
                "draw_support": {k: [{"value": float(v), "value_fraction": fraction_text(v), "count": n} for v, n in vals] for k, vals in self.draw_support.items()}}


def _bootstrap_draw(summary: MetricsSummary, rng: random.Random) -> dict[str, Fraction]:
    hierarchy: dict[int, dict[int, dict[int, dict[int, PolicyAreaEpisode]]]] = {}
    for item in summary.episodes:
        hierarchy.setdefault(item.order, {}).setdefault(item.graph_seed, {}).setdefault(item.relabeling_seed, {})[item.policy_seed] = item
    means: dict[str, list[Fraction]] = {p: [] for p in summary.policies}
    for order in sorted(hierarchy):
        graph_ids = sorted(hierarchy[order])
        graph_means: dict[str, list[Fraction]] = {p: [] for p in summary.policies}
        for _ in graph_ids:
            graph = hierarchy[order][graph_ids[rng.randrange(len(graph_ids))]]
            relabel_ids = sorted(graph)
            relabel_means: dict[str, list[Fraction]] = {p: [] for p in summary.policies}
            for _ in relabel_ids:
                relabel = graph[relabel_ids[rng.randrange(len(relabel_ids))]]
                seeds = sorted(relabel)
                sampled = [seeds[rng.randrange(len(seeds))] for _ in seeds]
                for p in summary.policies:
                    relabel_means[p].append(_mean([relabel[s].areas[p] for s in sampled], "bootstrap policy mean"))
            for p in summary.policies:
                graph_means[p].append(_mean(relabel_means[p], "bootstrap relabel mean"))
        for p in summary.policies:
            means[p].append(_mean(graph_means[p], "bootstrap graph mean"))
    c = _mean(means[CHAMPION_ID], "bootstrap theta")
    return {EFFECT_STAGE3: c - _mean(means[STAGE3_COMPARATOR_ID], "bootstrap theta"),
            EFFECT_RANDOM: c - _mean(means[RANDOM_ID], "bootstrap theta"),
            EFFECT_STRUCTURAL: c - _mean(means[STRUCTURAL_ID], "bootstrap theta")}


def bootstrap(summary: MetricsSummary, *, samples: int = DEFAULT_BOOTSTRAP_SAMPLES, seed: int = DEFAULT_BOOTSTRAP_SEED,
              confidence_level: CurveValue = DEFAULT_CONFIDENCE_LEVEL) -> BootstrapSummary:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    confidence = _fraction(confidence_level, "confidence_level")
    if not Fraction(0) < confidence < Fraction(1):
        raise ValueError("confidence_level must be in (0, 1)")
    draws = {effect: [] for effect in EFFECTS}
    for index in range(samples):
        draw = _bootstrap_draw(summary, _sample_rng(seed, index))
        for effect in EFFECTS:
            draws[effect].append(draw[effect])
    alpha = (Fraction(1) - confidence) / 2
    return BootstrapSummary(samples, seed, confidence, FROZEN_PERCENTILE_RULE,
        {e: summary.effects[e].theta for e in EFFECTS},
        {e: (_percentile(draws[e], alpha), _percentile(draws[e], Fraction(1) - alpha)) for e in EFFECTS},
        {e: _sign_counts(draws[e]) for e in EFFECTS},
        _bootstrap_support(summary),
        {e: tuple(sorted(Counter(draws[e]).items(), key=lambda x: x[0])) for e in EFFECTS})


def _bootstrap_support(summary: MetricsSummary) -> BootstrapSupport:
    if not summary.orders or not summary.graphs or not summary.relabels or not summary.episodes:
        raise ValueError("summary requires hierarchy support")
    first_order = summary.orders[0].order
    first_graph = next(x for x in summary.graphs if x.order == first_order)
    first_relabel = next(x for x in summary.relabels if x.order == first_order and x.graph_seed == first_graph.graph_seed)
    return BootstrapSupport(len(summary.orders), sum(x.order == first_order for x in summary.graphs),
                            sum(x.order == first_order and x.graph_seed == first_graph.graph_seed for x in summary.relabels),
                            sum(x.order == first_order and x.graph_seed == first_graph.graph_seed and x.relabeling_seed == first_relabel.relabeling_seed for x in summary.episodes),
                            len(summary.episodes))


def gates(summary: MetricsSummary, bootstrap_summary: BootstrapSummary, *,
          champion_stage3_threshold: CurveValue = DEFAULT_CHAMPION_STAGE3_THRESHOLD,
          champion_random_threshold: CurveValue = DEFAULT_CHAMPION_RANDOM_THRESHOLD,
          structural_retention_threshold: CurveValue = DEFAULT_STRUCTURAL_RETENTION_THRESHOLD) -> dict[str, bool]:
    if any(bootstrap_summary.observed[e] != summary.effects[e].theta for e in EFFECTS):
        raise ValueError("bootstrap observed effects do not match the paired-area summary")
    relative = summary.relative_improvements
    return {
        "relative_improvement_C_vs_stage3_at_least_threshold": relative[EFFECT_STAGE3] >= _fraction(champion_stage3_threshold, "stage3 threshold"),
        "bootstrap_C_vs_stage3_lower_bound_positive": bootstrap_summary.intervals[EFFECT_STAGE3][0] > 0,
        "C_vs_stage3_nonnegative_each_order": all(v >= 0 for v in summary.effects[EFFECT_STAGE3].order_deltas.values()),
        "C_vs_stage3_nonnegative_all_six_order_relabel_strata": all(v >= 0 for v in summary.effects[EFFECT_STAGE3].stratum_deltas.values()),
        "relative_improvement_C_vs_random_at_least_threshold": relative[EFFECT_RANDOM] >= _fraction(champion_random_threshold, "random threshold"),
        "bootstrap_C_vs_random_lower_bound_positive": bootstrap_summary.intervals[EFFECT_RANDOM][0] > 0,
        "structural_retention_at_least_threshold": summary.structural_retention >= _fraction(structural_retention_threshold, "structural threshold"),
    }


def fraction_payload(value: Fraction) -> dict[str, float | str]:
    return {"value": float(value), "fraction": fraction_text(value)}


def as_dict(value: Any) -> dict[str, Any]:
    """Convert any metrics dataclass to its canonical JSON-ready mapping."""

    method = getattr(value, "as_dict", None)
    if method is None or not callable(method):
        raise TypeError("value does not provide an as_dict helper")
    return method()


# Familiar Stage 5 names are aliases, while all computation remains local.
summarize_stage5 = summarize
bootstrap_stage5 = bootstrap
gate_checks = gates

__all__ = ["CHAMPION_ID", "STAGE3_COMPARATOR_ID", "RANDOM_ID", "STRUCTURAL_ID", "POLICY_IDS", "EFFECTS",
           "EFFECT_STAGE3", "EFFECT_RANDOM", "EFFECT_STRUCTURAL", "BOOTSTRAP_SAMPLES", "DEFAULT_BOOTSTRAP_SAMPLES", "DEFAULT_BOOTSTRAP_SEED",
           "DEFAULT_CONFIDENCE_LEVEL", "DEFAULT_CHAMPION_STAGE3_THRESHOLD", "DEFAULT_CHAMPION_RANDOM_THRESHOLD",
           "DEFAULT_STRUCTURAL_RETENTION_THRESHOLD", "FROZEN_PERCENTILE_RULE", "CurveValue", "PolicyAreaEpisode", "Episode", "RelabelAreaSummary",
           "GraphAreaSummary", "GraphSummary", "OrderAreaSummary", "OrderSummary", "PairEffectSummary", "EffectSummary", "MetricsSummary", "Summary", "Stage5Summary", "BootstrapSupport", "BootstrapSummary",
           "curve_area", "paired_area_delta", "parse_metrics_episodes", "summarize", "summarize_stage5", "bootstrap", "bootstrap_stage5",
           "gates", "gate_checks", "fraction_text", "fraction_payload", "as_dict"]
