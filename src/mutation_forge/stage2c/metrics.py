from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import cast

from mutation_forge.models import JsonValue
from mutation_forge.proposals.k_switch import ProposalCandidate
from mutation_forge.stage2b.rankers import RankResult

ScalarFeature = float | str | None


def _average_ranks(values: dict[str, float], *, descending: bool) -> dict[str, float]:
    ordered = sorted(
        values.items(),
        key=lambda item: ((-item[1]) if descending else item[1], item[0]),
    )
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for key, _ in ordered[index:end]:
            result[key] = average
        index = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / denominator


def rank_correlation(
    left: dict[str, float],
    right: dict[str, float],
    *,
    left_descending: bool = True,
    right_descending: bool = True,
) -> float | None:
    shared = sorted(set(left).intersection(right))
    if len(shared) < 2:
        return None
    left_ranks = _average_ranks(
        {key: left[key] for key in shared},
        descending=left_descending,
    )
    right_ranks = _average_ranks(
        {key: right[key] for key in shared},
        descending=right_descending,
    )
    return pearson(
        [left_ranks[key] for key in shared],
        [right_ranks[key] for key in shared],
    )


def top_tie(priority_by_id: dict[str, float]) -> tuple[int, float, int]:
    if not priority_by_id:
        return 0, 0.0, 0
    best = max(priority_by_id.values())
    count = sum(value == best for value in priority_by_id.values())
    return count, count / len(priority_by_id), len(set(priority_by_id.values()))


def top_k_overlap(
    left_order: tuple[str, ...],
    right_order: tuple[str, ...],
    k: int,
) -> float:
    limit = min(k, len(left_order), len(right_order))
    if limit == 0:
        return 0.0
    return len(set(left_order[:limit]).intersection(right_order[:limit])) / limit


def rank_result_maps(rank: RankResult) -> tuple[tuple[str, ...], dict[str, float]]:
    return (
        tuple(item.proposal_id for item in rank.ranked),
        {item.proposal_id: float(item.priority) for item in rank.ranked},
    )


def oracle_summary(
    oracle_delta: dict[str, int],
    random_selected_id: str,
    structural_selected_id: str,
    random_priority: dict[str, float],
    structural_priority: dict[str, float],
    top_k_values: tuple[int, ...],
) -> dict[str, JsonValue]:
    ordered = tuple(
        key for key, _ in sorted(oracle_delta.items(), key=lambda item: (-item[1], item[0]))
    )
    best_delta = max(oracle_delta.values())
    improving = sum(delta > 0 for delta in oracle_delta.values())

    def policy(selected_id: str, priorities: dict[str, float]) -> dict[str, JsonValue]:
        selected_delta = oracle_delta[selected_id]
        return {
            "selected_delta": selected_delta,
            "regret": best_delta - selected_delta,
            "top_1_hit": selected_id == ordered[0],
            "best_tie_hit": selected_delta == best_delta,
            "top_k_hits": {
                str(k): selected_id in ordered[: min(k, len(ordered))]
                for k in top_k_values
            },
            "oracle_rank_correlation": rank_correlation(priorities, {
                key: float(value) for key, value in oracle_delta.items()
            }),
        }

    return {
        "any_improving_proposal": improving > 0,
        "best_immediate_score_delta": best_delta,
        "improving_count": improving,
        "improving_fraction": improving / len(oracle_delta),
        "oracle_order": list(ordered),
        "oracle_deltas": {
            key: oracle_delta[key] for key in sorted(oracle_delta)
        },
        "random": policy(random_selected_id, random_priority),
        "structural": policy(structural_selected_id, structural_priority),
    }


def flatten_proposal_features(
    candidate: ProposalCandidate,
    forbidden_lengths: tuple[int, ...],
) -> dict[str, ScalarFeature]:
    payload = candidate.payload
    result: dict[str, ScalarFeature] = {
        "k": float(payload["k"]),
        "operator_family": payload["operator_family"],
        "selector_tags": "+".join(payload["selector_tags"]),
        "anchor_forbidden_length": (
            float(payload["anchor_forbidden_length"])
            if payload["anchor_forbidden_length"] is not None
            else None
        ),
        "minimum_distance_between_removed_edges": float(
            payload["minimum_distance_between_removed_edges"]
        ),
        "mean_distance_between_removed_edges": float(
            payload["mean_distance_between_removed_edges"]
        ),
        "minimum_preexisting_distance_for_new_edges": float(
            payload["minimum_preexisting_distance_for_new_edges"]
        ),
        "mean_preexisting_distance_for_new_edges": float(
            payload["mean_preexisting_distance_for_new_edges"]
        ),
        "local_triangle_risk": float(payload["local_triangle_risk"]),
        "local_c4_risk": float(payload["local_c4_risk"]),
        "reconnection_span": float(payload["reconnection_span"]),
    }
    vector_names = (
        "broken_sampled_witnesses_by_length",
        "removed_edge_load_sum_by_length",
        "removed_edge_load_max_by_length",
    )
    payload_mapping = cast(dict[str, object], payload)
    for name in vector_names:
        values = cast(list[int], payload_mapping[name])
        for length, value in zip(forbidden_lengths, values, strict=True):
            result[f"{name}:{length}"] = float(value)
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(slots=True)
class _GroupMoments:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.total / self.count,
        }


@dataclass(slots=True)
class _NumericFeature:
    sample_cap: int
    distinct_cap: int
    total: int = 0
    missing: int = 0
    samples: list[tuple[float, float, float, float]] = field(default_factory=list)
    distinct: set[float] = field(default_factory=set)
    distinct_overflow: bool = False
    by_k: dict[str, _GroupMoments] = field(
        default_factory=lambda: defaultdict(_GroupMoments)
    )
    by_selector: dict[str, _GroupMoments] = field(
        default_factory=lambda: defaultdict(_GroupMoments)
    )

    def add(
        self,
        value: float | None,
        *,
        delta: float,
        improvement: float,
        structural_priority: float,
        k: int,
        selector: str,
    ) -> None:
        self.total += 1
        if value is None:
            self.missing += 1
            return
        if len(self.samples) < self.sample_cap:
            self.samples.append((value, delta, improvement, structural_priority))
        if len(self.distinct) < self.distinct_cap:
            self.distinct.add(value)
        elif value not in self.distinct:
            self.distinct_overflow = True
        self.by_k[str(k)].add(value)
        self.by_selector[selector].add(value)

    def as_dict(self, *, near_constant_epsilon: float) -> dict[str, JsonValue]:
        values = [item[0] for item in self.samples]
        if not values:
            return {
                "kind": "numeric",
                "count": self.total,
                "missing_rate": 1.0,
                "constant": True,
                "near_constant_rate": 1.0,
                "distinct_value_count": 0,
                "distinct_value_count_capped": False,
            }
        minimum = min(values)
        maximum = max(values)
        median = statistics.median(values)
        tolerance = near_constant_epsilon * max(1.0, maximum - minimum)
        near_rate = sum(abs(value - median) <= tolerance for value in values) / len(values)
        return {
            "kind": "numeric",
            "count": self.total,
            "sample_count": len(values),
            "missing_rate": self.missing / self.total,
            "constant": len(self.distinct) == 1 and not self.distinct_overflow,
            "near_constant_rate": near_rate,
            "distinct_value_count": len(self.distinct),
            "distinct_value_count_capped": self.distinct_overflow,
            "range": [minimum, maximum],
            "quantiles": {
                "q05": _quantile(values, 0.05),
                "q25": _quantile(values, 0.25),
                "q50": _quantile(values, 0.50),
                "q75": _quantile(values, 0.75),
                "q95": _quantile(values, 0.95),
            },
            "correlation_with_oracle_delta": pearson(
                values,
                [item[1] for item in self.samples],
            ),
            "correlation_with_improvement_status": pearson(
                values,
                [item[2] for item in self.samples],
            ),
            "correlation_with_structural_priority": pearson(
                values,
                [item[3] for item in self.samples],
            ),
            "by_k": {
                key: value.as_dict() for key, value in sorted(self.by_k.items())
            },
            "by_selector": {
                key: value.as_dict()
                for key, value in sorted(self.by_selector.items())
            },
        }


@dataclass(slots=True)
class _CategoricalGroup:
    count: int = 0
    delta_total: float = 0.0
    improving: int = 0
    priority_total: float = 0.0

    def add(self, *, delta: float, improving: bool, priority: float) -> None:
        self.count += 1
        self.delta_total += delta
        self.improving += improving
        self.priority_total += priority

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "count": self.count,
            "mean_oracle_delta": self.delta_total / self.count,
            "improvement_rate": self.improving / self.count,
            "mean_structural_priority": self.priority_total / self.count,
        }


@dataclass(slots=True)
class _CategoricalFeature:
    distinct_cap: int
    total: int = 0
    missing: int = 0
    groups: dict[str, _CategoricalGroup] = field(
        default_factory=lambda: defaultdict(_CategoricalGroup)
    )
    overflow: bool = False

    def add(
        self,
        value: str | None,
        *,
        delta: float,
        structural_priority: float,
    ) -> None:
        self.total += 1
        if value is None:
            self.missing += 1
            return
        if value not in self.groups and len(self.groups) >= self.distinct_cap:
            self.overflow = True
            return
        self.groups[value].add(
            delta=delta,
            improving=delta > 0,
            priority=structural_priority,
        )

    def as_dict(self) -> dict[str, JsonValue]:
        observed = self.total - self.missing
        dominant = max(
            (group.count for group in self.groups.values()),
            default=0,
        )
        return {
            "kind": "categorical",
            "count": self.total,
            "missing_rate": self.missing / self.total,
            "constant": len(self.groups) == 1 and not self.overflow,
            "near_constant_rate": dominant / observed if observed else 1.0,
            "distinct_value_count": len(self.groups),
            "distinct_value_count_capped": self.overflow,
            "categories": {
                key: value.as_dict() for key, value in sorted(self.groups.items())
            },
        }


@dataclass(slots=True)
class _Polarity:
    pools: int = 0
    hits: int = 0
    regret_total: float = 0.0

    def add(self, selected_delta: float, best_delta: float) -> None:
        self.pools += 1
        self.hits += selected_delta == best_delta
        self.regret_total += best_delta - selected_delta

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "pools": self.pools,
            "oracle_best_tie_hit_rate": self.hits / self.pools,
            "mean_regret": self.regret_total / self.pools,
        }


class FeatureAnalyzer:
    def __init__(
        self,
        *,
        forbidden_lengths: tuple[int, ...],
        sample_cap: int,
        distinct_cap: int,
        near_constant_epsilon: float,
    ) -> None:
        self.forbidden_lengths = forbidden_lengths
        self.sample_cap = sample_cap
        self.distinct_cap = distinct_cap
        self.near_constant_epsilon = near_constant_epsilon
        self.numeric: dict[str, _NumericFeature] = {}
        self.categorical: dict[str, _CategoricalFeature] = {}
        self.polarity: dict[str, dict[str, _Polarity]] = defaultdict(
            lambda: {"ascending": _Polarity(), "descending": _Polarity()}
        )

    def add_pool(
        self,
        candidates: tuple[ProposalCandidate, ...],
        oracle_delta: dict[str, int],
        structural_priority: dict[str, float],
    ) -> None:
        flattened = {
            candidate.proposal_id: flatten_proposal_features(
                candidate,
                self.forbidden_lengths,
            )
            for candidate in candidates
        }
        for candidate in candidates:
            proposal_id = candidate.proposal_id
            delta = float(oracle_delta[proposal_id])
            priority = structural_priority[proposal_id]
            k = candidate.payload["k"]
            selector = candidate.payload["selector_tags"][0]
            for name, value in flattened[proposal_id].items():
                if isinstance(value, float):
                    numeric_feature = self.numeric.setdefault(
                        name,
                        _NumericFeature(self.sample_cap, self.distinct_cap),
                    )
                    numeric_feature.add(
                        value,
                        delta=delta,
                        improvement=float(delta > 0),
                        structural_priority=priority,
                        k=k,
                        selector=selector,
                    )
                else:
                    categorical_feature = self.categorical.setdefault(
                        name,
                        _CategoricalFeature(self.distinct_cap),
                    )
                    categorical_feature.add(
                        value,
                        delta=delta,
                        structural_priority=priority,
                    )
        best_delta = float(max(oracle_delta.values()))
        for name in sorted(self.numeric):
            present = [
                (cast(float, flattened[candidate.proposal_id][name]), candidate.proposal_id)
                for candidate in candidates
                if isinstance(flattened[candidate.proposal_id].get(name), float)
            ]
            if not present:
                continue
            ascending_id = min(present, key=lambda item: (item[0], item[1]))[1]
            descending_id = min(present, key=lambda item: (-item[0], item[1]))[1]
            self.polarity[name]["ascending"].add(
                float(oracle_delta[ascending_id]),
                best_delta,
            )
            self.polarity[name]["descending"].add(
                float(oracle_delta[descending_id]),
                best_delta,
            )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "numeric": {
                name: feature.as_dict(
                    near_constant_epsilon=self.near_constant_epsilon,
                )
                for name, feature in sorted(self.numeric.items())
            },
            "categorical": {
                name: feature.as_dict()
                for name, feature in sorted(self.categorical.items())
            },
            "univariate_polarity_screen": {
                name: {
                    direction: accumulator.as_dict()
                    for direction, accumulator in sorted(directions.items())
                }
                for name, directions in sorted(self.polarity.items())
            },
            "sampling": {
                "deterministic_first_observations": True,
                "sample_cap_per_numeric_feature": self.sample_cap,
                "distinct_value_cap_per_feature": self.distinct_cap,
            },
        }


@dataclass(slots=True)
class RankAggregate:
    top_k_values: tuple[int, ...]
    pools: int = 0
    same_selection: int = 0
    disagreements_with_headroom: int = 0
    headroom_pools: int = 0
    agreement_improvements: int = 0
    agreement_pools: int = 0
    disagreement_random_improvements: int = 0
    disagreement_structural_improvements: int = 0
    disagreement_pools: int = 0
    random_tie_pools: int = 0
    structural_tie_pools: int = 0
    random_top_tie_total: int = 0
    structural_top_tie_total: int = 0
    correlations: list[float] = field(default_factory=list)
    top_k_overlap_totals: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    random_regret_total: float = 0.0
    structural_regret_total: float = 0.0
    random_oracle_hits: int = 0
    structural_oracle_hits: int = 0

    def add(self, record: dict[str, JsonValue]) -> None:
        self.pools += 1
        same = cast(bool, record["same_selection"])
        self.same_selection += same
        headroom = cast(bool, record["any_improving_proposal"])
        self.headroom_pools += headroom
        if not same and headroom:
            self.disagreements_with_headroom += 1
        random_delta = cast(int, record["random_selected_delta"])
        structural_delta = cast(int, record["structural_selected_delta"])
        if same:
            self.agreement_pools += 1
            self.agreement_improvements += random_delta > 0
        else:
            self.disagreement_pools += 1
            self.disagreement_random_improvements += random_delta > 0
            self.disagreement_structural_improvements += structural_delta > 0
        random_ties = cast(int, record["random_top_tie_count"])
        structural_ties = cast(int, record["structural_top_tie_count"])
        self.random_tie_pools += random_ties > 1
        self.structural_tie_pools += structural_ties > 1
        self.random_top_tie_total += random_ties
        self.structural_top_tie_total += structural_ties
        correlation = record["rank_correlation"]
        if isinstance(correlation, int | float) and not isinstance(correlation, bool):
            self.correlations.append(float(correlation))
        overlaps = cast(dict[str, JsonValue], record["top_k_overlap"])
        for key, value in overlaps.items():
            self.top_k_overlap_totals[key] += float(cast(int | float, value))
        self.random_regret_total += cast(int, record["random_regret"])
        self.structural_regret_total += cast(int, record["structural_regret"])
        self.random_oracle_hits += cast(bool, record["random_best_tie_hit"])
        self.structural_oracle_hits += cast(bool, record["structural_best_tie_hit"])

    def as_dict(self) -> dict[str, JsonValue]:
        if self.pools == 0:
            return {
                "status": "disabled",
                "pool_count": 0,
                "reason": "diagnostic oracle is opt-in",
            }
        return {
            "pool_count": self.pools,
            "same_selection_rate": self.same_selection / self.pools,
            "policy_disagreement_rate": 1.0 - self.same_selection / self.pools,
            "top_k_overlap": {
                key: value / self.pools
                for key, value in sorted(self.top_k_overlap_totals.items())
            },
            "tie_frequency": {
                "random": self.random_tie_pools / self.pools,
                "structural": self.structural_tie_pools / self.pools,
            },
            "mean_top_tie_size": {
                "random": self.random_top_tie_total / self.pools,
                "structural": self.structural_top_tie_total / self.pools,
            },
            "rank_correlation": {
                "count": len(self.correlations),
                "minimum": min(self.correlations) if self.correlations else None,
                "median": (
                    statistics.median(self.correlations)
                    if self.correlations
                    else None
                ),
                "maximum": max(self.correlations) if self.correlations else None,
            },
            "policy_disagreement_rate_given_headroom": (
                self.disagreements_with_headroom / self.headroom_pools
                if self.headroom_pools
                else None
            ),
            "improvement_rate_after_agreement": (
                self.agreement_improvements / self.agreement_pools
                if self.agreement_pools
                else None
            ),
            "improvement_rate_after_disagreement": {
                "random": (
                    self.disagreement_random_improvements / self.disagreement_pools
                    if self.disagreement_pools
                    else None
                ),
                "structural": (
                    self.disagreement_structural_improvements
                    / self.disagreement_pools
                    if self.disagreement_pools
                    else None
                ),
            },
            "oracle": {
                "headroom_rate": self.headroom_pools / self.pools,
                "mean_regret": {
                    "random": self.random_regret_total / self.pools,
                    "structural": self.structural_regret_total / self.pools,
                },
                "best_tie_hit_rate": {
                    "random": self.random_oracle_hits / self.pools,
                    "structural": self.structural_oracle_hits / self.pools,
                },
            },
        }


def curve_summary(
    initial_scores: list[int],
    random_raw_curves: list[list[int]],
    structural_raw_curves: list[list[int]],
) -> dict[str, JsonValue]:
    if not (
        len(initial_scores) == len(random_raw_curves) == len(structural_raw_curves)
    ):
        raise ValueError("curve inputs must align")

    def policy(curves: list[list[int]]) -> dict[str, JsonValue]:
        raw_auc = [statistics.fmean(curve) for curve in curves]
        normalized_curves = [
            [
                (initial - value) / max(1, initial)
                for value in curve
            ]
            for initial, curve in zip(initial_scores, curves, strict=True)
        ]
        normalized_auc = [statistics.fmean(curve) for curve in normalized_curves]
        first = later = never = 0
        for initial, curve in zip(initial_scores, curves, strict=True):
            improving_steps = [index for index, value in enumerate(curve) if value < initial]
            if improving_steps and improving_steps[0] == 0:
                first += 1
            elif improving_steps:
                later += 1
            else:
                never += 1
        return {
            "distinct_raw_auc_values": len(set(raw_auc)),
            "distinct_normalized_auc_values": len(set(normalized_auc)),
            "raw_auc_values": cast(list[JsonValue], sorted(set(raw_auc))),
            "normalized_auc_values": cast(
                list[JsonValue],
                sorted(set(normalized_auc)),
            ),
            "improvement_timing_fraction": {
                "step_1": first / len(curves),
                "later": later / len(curves),
                "never": never / len(curves),
            },
            "floor_frequency": sum(value <= 0.0 for value in normalized_auc)
            / len(normalized_auc),
            "ceiling_frequency": sum(value >= 1.0 for value in normalized_auc)
            / len(normalized_auc),
        }

    horizons = [len(curve) for curve in random_raw_curves]
    sensitivities = [
        1.0 / (max(1, initial) * horizon)
        for initial, horizon in zip(initial_scores, horizons, strict=True)
    ]
    denominators = [max(1, value) for value in initial_scores]
    return {
        "initial_score_distribution": dict(
            sorted(Counter(str(value) for value in initial_scores).items())
        ),
        "no_possible_denominator_fraction": sum(value == 0 for value in initial_scores)
        / len(initial_scores),
        "normalization_denominator_distribution": dict(
            sorted(Counter(str(value) for value in denominators).items())
        ),
        "normalization_quantum_distribution": dict(
            sorted(Counter(str(1.0 / value) for value in denominators).items())
        ),
        "effectively_quantized_normalization_fraction": sum(
            value <= 4 for value in denominators
        )
        / len(denominators),
        "median_auc_sensitivity_to_one_unit_one_step_improvement": statistics.median(
            sensitivities
        ),
        "random": policy(random_raw_curves),
        "structural": policy(structural_raw_curves),
    }
