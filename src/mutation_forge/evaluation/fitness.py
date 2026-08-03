from __future__ import annotations

import statistics

from mutation_forge.models import EpisodeResult, JsonValue

FITNESS_SCHEMA_VERSION = "mforge.experiment.evaluation.v2"


def aggregate_fitness(episodes: list[EpisodeResult]) -> dict[str, JsonValue]:
    if not episodes:
        raise ValueError("cannot aggregate an empty episode set")
    normalized_best_total = [
        episode.best_score.total_capped_witnesses
        / max(1, episode.initial_score.total_capped_witnesses)
        for episode in episodes
    ]
    normalized_best_weighted = [
        episode.best_score.weighted_penalty / max(1, episode.initial_score.weighted_penalty)
        for episode in episodes
    ]
    failures = sum(episode.timed_out or episode.score_failures > 0 for episode in episodes)
    attempted = sum(episode.evaluations for episode in episodes)
    illegal_or_noop = sum(
        episode.invalid_proposals + episode.noop_proposals for episode in episodes
    )
    policy_call_ms = [episode.policy_call_ms / max(1, episode.evaluations) for episode in episodes]
    key: list[JsonValue] = [
        failures,
        statistics.median(normalized_best_total),
        statistics.median(normalized_best_weighted),
        statistics.median(episode.normalized_best_auc for episode in episodes),
        sum(episode.timed_out for episode in episodes) / len(episodes),
        illegal_or_noop / max(1, attempted),
        statistics.median(policy_call_ms),
        0,
        0,
    ]
    return {
        "fitness_schema_version": FITNESS_SCHEMA_VERSION,
        "episodes": len(episodes),
        "failure_episode_count": failures,
        "median_normalized_best_total_witnesses": key[1],
        "median_normalized_best_weighted_penalty": key[2],
        "median_normalized_best_so_far_auc": key[3],
        "timeout_rate": key[4],
        "illegal_or_noop_rate": key[5],
        "median_policy_call_ms": key[6],
        "normalized_ast_node_count": 0,
        "ordering_key": key,
    }
