from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from typing import cast

from mutation_forge.models import JsonValue
from mutation_forge.stage2d.config import Stage2DConfig


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sample_rng(seed: int, sample_index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{sample_index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _group_differences(
    episodes: list[dict[str, JsonValue]],
) -> dict[int, dict[int, list[float]]]:
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        order = cast(int, episode["order"])
        graph_seed = cast(int, episode["graph_seed"])
        random_record = cast(dict[str, JsonValue], episode["random"])
        structural_record = cast(dict[str, JsonValue], episode["structural"])
        grouped[order][graph_seed].append(
            cast(float, structural_record["auc"])
            - cast(float, random_record["auc"])
        )
    return grouped


def _resampled_differences(
    grouped: dict[int, list[float]],
    rng: random.Random,
) -> list[float]:
    graph_seeds = sorted(grouped)
    result: list[float] = []
    for _ in graph_seeds:
        sampled_graph = graph_seeds[rng.randrange(len(graph_seeds))]
        values = grouped[sampled_graph]
        result.extend(values[rng.randrange(len(values))] for _ in values)
    return result


def hierarchical_bootstrap(
    episodes: list[dict[str, JsonValue]],
    config: Stage2DConfig,
    *,
    workers: int = 1,
) -> dict[str, JsonValue]:
    if workers < 1 or workers > config.resources.max_concurrent_shards:
        raise ValueError("bootstrap workers exceed frozen bounds")
    grouped = _group_differences(episodes)
    expected_orders = set(config.experiment.orders)
    if set(grouped) != expected_orders:
        raise ValueError("bootstrap order coverage mismatch")
    estimates: dict[str, list[tuple[int, float]]] = {
        str(order): [] for order in config.experiment.orders
    }
    estimates["pooled_stratified"] = []
    for worker in range(workers):
        for sample_index in range(
            worker,
            config.statistics.bootstrap_samples,
            workers,
        ):
            rng = _sample_rng(config.statistics.bootstrap_seed, sample_index)
            pooled: list[float] = []
            for order in config.experiment.orders:
                sampled = _resampled_differences(grouped[order], rng)
                estimates[str(order)].append(
                    (sample_index, statistics.median(sampled))
                )
                pooled.extend(sampled)
            estimates["pooled_stratified"].append(
                (sample_index, statistics.median(pooled))
            )
    alpha = (1.0 - config.statistics.confidence_level) / 2.0
    result: dict[str, JsonValue] = {}
    for name, indexed in estimates.items():
        values = [value for _, value in sorted(indexed)]
        result[name] = {
            "samples": len(values),
            "seed": config.statistics.bootstrap_seed,
            "confidence_level": config.statistics.confidence_level,
            "median": statistics.median(values),
            "interval": [
                _percentile(values, alpha),
                _percentile(values, 1.0 - alpha),
            ],
        }
    return result


def _policy_values(
    episodes: list[dict[str, JsonValue]],
    order: int,
    policy: str,
    field: str,
) -> list[int | float]:
    return [
        cast(
            int | float,
            cast(dict[str, JsonValue], episode[policy])[field],
        )
        for episode in episodes
        if episode["order"] == order
    ]


def _policy_json_values(
    episodes: list[dict[str, JsonValue]],
    order: int,
    policy: str,
    field: str,
) -> list[JsonValue]:
    return [
        cast(dict[str, JsonValue], episode[policy])[field]
        for episode in episodes
        if episode["order"] == order
    ]


def _policy_score_values(
    episodes: list[dict[str, JsonValue]],
    order: int,
    policy: str,
    field: str,
) -> list[JsonValue]:
    return [
        cast(
            dict[str, JsonValue],
            cast(dict[str, JsonValue], episode[policy])["best_score"],
        )[field]
        for episode in episodes
        if episode["order"] == order
    ]


def summarize_episodes(
    episodes: list[dict[str, JsonValue]],
    config: Stage2DConfig,
    *,
    bootstrap_workers: int,
) -> dict[str, JsonValue]:
    expected_count = (
        len(config.experiment.orders)
        * len(config.experiment.graph_seeds)
        * len(config.experiment.policy_seeds)
    )
    if len(episodes) != expected_count:
        raise ValueError("Stage 2D episode count mismatch")
    order_metrics: dict[str, JsonValue] = {}
    nonnegative_primary_graph_seeds = 0
    for order in config.experiment.orders:
        random_auc = [
            float(value) for value in _policy_values(episodes, order, "random", "auc")
        ]
        structural_auc = [
            float(value)
            for value in _policy_values(episodes, order, "structural", "auc")
        ]
        random_best = [
            int(value)
            for value in _policy_values(
                episodes, order, "random", "best_total_witnesses"
            )
        ]
        structural_best = [
            int(value)
            for value in _policy_values(
                episodes, order, "structural", "best_total_witnesses"
            )
        ]
        differences = [
            structural - random_value
            for structural, random_value in zip(
                structural_auc, random_auc, strict=True
            )
        ]
        median_random = statistics.median(random_auc)
        median_structural = statistics.median(structural_auc)
        by_graph_seed: dict[str, JsonValue] = {}
        for graph_seed in config.experiment.graph_seeds:
            graph_differences = [
                cast(
                    float,
                    cast(dict[str, JsonValue], episode["structural"])["auc"],
                )
                - cast(
                    float,
                    cast(dict[str, JsonValue], episode["random"])["auc"],
                )
                for episode in episodes
                if episode["order"] == order
                and episode["graph_seed"] == graph_seed
            ]
            graph_median = statistics.median(graph_differences)
            by_graph_seed[str(graph_seed)] = {
                "paired_episodes": len(graph_differences),
                "median_auc_delta": graph_median,
            }
            if order == config.statistics.primary_order and graph_median >= 0.0:
                nonnegative_primary_graph_seeds += 1
        order_metrics[str(order)] = {
            "paired_episodes": len(random_auc),
            "median_random_auc": median_random,
            "median_structural_auc": median_structural,
            "median_auc_delta": statistics.median(differences),
            "relative_median_improvement": (
                (median_structural - median_random)
                / max(abs(median_random), 1.0e-12)
            ),
            "median_random_best_total_witnesses": statistics.median(random_best),
            "median_structural_best_total_witnesses": statistics.median(
                structural_best
            ),
            "random_best_total_witness_distribution": cast(
                list[JsonValue], sorted(random_best)
            ),
            "structural_best_total_witness_distribution": cast(
                list[JsonValue], sorted(structural_best)
            ),
            "random_final_capped_cycle_counts": _policy_score_values(
                episodes,
                order,
                "random",
                "capped_cycle_counts",
            ),
            "structural_final_capped_cycle_counts": _policy_score_values(
                episodes,
                order,
                "structural",
                "capped_cycle_counts",
            ),
            "random_final_weighted_penalties": _policy_score_values(
                episodes,
                order,
                "random",
                "weighted_penalty",
            ),
            "structural_final_weighted_penalties": _policy_score_values(
                episodes,
                order,
                "structural",
                "weighted_penalty",
            ),
            "random_final_ordering_keys": _policy_score_values(
                episodes,
                order,
                "random",
                "ordering_key",
            ),
            "structural_final_ordering_keys": _policy_score_values(
                episodes,
                order,
                "structural",
                "ordering_key",
            ),
            "random_evaluations_to_first_improvement": _policy_json_values(
                episodes,
                order,
                "random",
                "evaluations_to_first_improvement",
            ),
            "structural_evaluations_to_first_improvement": _policy_json_values(
                episodes,
                order,
                "structural",
                "evaluations_to_first_improvement",
            ),
            "random_acceptance_counts": _policy_json_values(
                episodes,
                order,
                "random",
                "accepted_count",
            ),
            "structural_acceptance_counts": _policy_json_values(
                episodes,
                order,
                "structural",
                "accepted_count",
            ),
            "random_rejection_counts": _policy_json_values(
                episodes,
                order,
                "random",
                "rejected_count",
            ),
            "structural_rejection_counts": _policy_json_values(
                episodes,
                order,
                "structural",
                "rejected_count",
            ),
            "random_duplicate_counts": _policy_json_values(
                episodes,
                order,
                "random",
                "duplicate_count",
            ),
            "structural_duplicate_counts": _policy_json_values(
                episodes,
                order,
                "structural",
                "duplicate_count",
            ),
            "by_graph_seed": by_graph_seed,
        }
    bootstrap = hierarchical_bootstrap(
        episodes,
        config,
        workers=bootstrap_workers,
    )
    invalid_graphs = sum(cast(int, episode["invalid_graphs"]) for episode in episodes)
    policy_failures = sum(cast(int, episode["policy_failures"]) for episode in episodes)
    initial_score_calls = sum(
        cast(int, episode["initial_score_calls"]) for episode in episodes
    )
    selected_score_calls = sum(
        cast(int, episode["selected_score_calls"]) for episode in episodes
    )
    oracle_score_calls = sum(
        cast(int, episode["oracle_score_calls"]) for episode in episodes
    )
    expected_selected_calls = len(episodes) * 2 * config.experiment.horizon
    primary = cast(
        dict[str, JsonValue],
        order_metrics[str(config.statistics.primary_order)],
    )
    secondary = cast(
        dict[str, JsonValue],
        order_metrics[str(config.statistics.secondary_order)],
    )
    primary_bootstrap = cast(
        dict[str, JsonValue],
        bootstrap[str(config.statistics.primary_order)],
    )
    pooled_bootstrap = cast(
        dict[str, JsonValue],
        bootstrap["pooled_stratified"],
    )
    primary_interval = cast(list[float], primary_bootstrap["interval"])
    pooled_interval = cast(list[float], pooled_bootstrap["interval"])
    gate_without_replay: dict[str, JsonValue] = {
        "primary_relative_median_at_least_10_percent": (
            cast(float, primary["relative_median_improvement"])
            >= config.statistics.relative_median_threshold
        ),
        "primary_bootstrap_lower_bound_above_zero": primary_interval[0] > 0.0,
        "pooled_stratified_bootstrap_lower_bound_above_zero": (
            pooled_interval[0] > 0.0
        ),
        "secondary_median_delta_nonnegative": (
            cast(float, secondary["median_auc_delta"]) >= 0.0
        ),
        "structural_witness_count_no_worse_each_order": all(
            cast(
                int | float,
                cast(dict[str, JsonValue], order_metrics[str(order)])[
                    "median_structural_best_total_witnesses"
                ],
            )
            <= cast(
                int | float,
                cast(dict[str, JsonValue], order_metrics[str(order)])[
                    "median_random_best_total_witnesses"
                ],
            )
            for order in config.experiment.orders
        ),
        "at_least_six_primary_graph_seeds_nonnegative": (
            nonnegative_primary_graph_seeds
            >= config.statistics.minimum_nonnegative_graph_seeds
        ),
        "graph_validity_100_percent": invalid_graphs == 0,
        "policy_failure_rate_zero": policy_failures == 0,
        "selected_plan_only_scoring_no_oracle": (
            initial_score_calls == len(episodes)
            and selected_score_calls == expected_selected_calls
            and oracle_score_calls == 0
        ),
    }
    return {
        "episode_count": len(episodes),
        "orders": order_metrics,
        "bootstrap": bootstrap,
        "accounting": {
            "initial_score_calls": initial_score_calls,
            "selected_score_calls": selected_score_calls,
            "expected_selected_score_calls": expected_selected_calls,
            "oracle_score_calls": oracle_score_calls,
            "invalid_graphs": invalid_graphs,
            "policy_failures": policy_failures,
        },
        "primary_nonnegative_graph_seed_count": nonnegative_primary_graph_seeds,
        "gate_without_replay": gate_without_replay,
    }
