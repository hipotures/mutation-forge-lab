"""Deterministic descriptive statistics and the Stage 3 evidence gate."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Iterable, Mapping
from typing import Any, cast

GO_TO_STAGE_4 = "GO_TO_STAGE_4"
NO_GO = "NO_GO"
INCONCLUSIVE_INFRASTRUCTURE_FAILURE = "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"
GATE_NAMES = (
    "dependency_provenance",
    "protocol_safety",
    "campaign_authority",
    "exact_usage",
    "minimum_unique",
    "baseline_ast_distinct",
    "champion_random_relative",
    "champion_structural_relative",
    "primary_replay_exact",
    "zero_invalid_and_worker_failures",
    "selected_only_equal_bounded_parity",
    "repository_and_heg_validation",
)


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else float("nan")


def _auc(record: Mapping[str, Any]) -> float:
    if "auc" in record:
        return float(record["auc"])
    curve = record.get("normalized_best_so_far_curve", record.get("curve", []))
    values = [float(x) for x in curve]
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap(
    values: Mapping[str, list[float]] | list[float],
    *,
    samples: int = 10_000,
    seed: int = 2026072909,
    confidence: float = 0.95,
) -> dict[str, float | int | list[float]]:
    """Bootstrap a paired sample with a stable, independent RNG stream."""
    if isinstance(values, Mapping):
        data = [float(v) for v in values.get("values", [])]
    else:
        data = [float(v) for v in values]
    if not data:
        raise ValueError("bootstrap requires at least one observation")
    estimates: list[float] = []
    for i in range(samples):
        digest = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        estimates.append(float(statistics.median(data[rng.randrange(len(data))] for _ in data)))
    alpha = (1.0 - confidence) / 2.0
    ordered = sorted(estimates)

    def pct(p: float) -> float:
        pos = p * (len(ordered) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    return {
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence,
        "median": _median(data),
        "interval": [pct(alpha), pct(1.0 - alpha)],
    }


def _policy_record(episode: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    container = episode.get("policies", episode)
    value = container.get(policy) if isinstance(container, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"episode missing policy {policy}")
    return value


def summarize_development(
    episodes: list[Mapping[str, Any]],
    policies: Iterable[str] | None = None,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026072909,
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    ordered_episodes = sorted(episodes, key=lambda e: str(e.get("episode_id", "")))
    first_container = ordered_episodes[0].get("policies", ordered_episodes[0])
    names = list(
        policies
        or [k for k, v in first_container.items() if isinstance(v, Mapping) and k not in {"steps"}]
    )
    if "random" not in names or "structural" not in names:
        raise ValueError("random and structural baselines are required")
    result: dict[str, Any] = {"policies": {}, "episode_count": len(episodes)}
    episodes = ordered_episodes
    orders = sorted({int(e["order"]) for e in episodes})
    for name in names:
        records = [_policy_record(e, name) for e in episodes]
        by_order: dict[str, Any] = {}
        for order in orders:
            selected = [_policy_record(e, name) for e in episodes if int(e["order"]) == order]
            aucs = [_auc(r) for r in selected]
            witnesses = [
                float(r.get("best_total_witnesses", r.get("best_total", 0))) for r in selected
            ]
            by_order[str(order)] = {
                "episodes": len(selected),
                "median_auc": _median(aucs),
                "median_best_witnesses": _median(witnesses),
                "auc_values": aucs,
                "best_witness_values": witnesses,
            }
        pooled_auc = [_auc(r) for r in records]
        pooled_witnesses = [
            float(r.get("best_total_witnesses", r.get("best_total", 0))) for r in records
        ]

        def median_field(
            field: str,
            policy_records: list[Mapping[str, Any]] = records,
        ) -> float:
            return _median(float(r.get(field, 0) or 0) for r in policy_records)

        result["policies"][name] = {
            "by_order": by_order,
            "pooled_median_auc": _median(pooled_auc),
            "pooled_median_best_witnesses": _median(pooled_witnesses),
            "median_accepted": median_field("accepted_count"),
            "median_rejected": median_field("rejected_count"),
            "median_duplicates": median_field("duplicate_count"),
            "median_nonimproving": median_field("nonimproving_count"),
            "median_divergence": median_field("divergence_count"),
            "median_failures": median_field("failure_count"),
            "median_evaluations_to_first_improvement": median_field(
                "evaluations_to_first_improvement"
            ),
            "median_first_improvement_ns": median_field("first_improvement_ns"),
        }
    for name in names:
        if name in {"random", "structural"}:
            continue
        deltas: dict[str, Any] = {}
        for baseline in ("random", "structural"):
            paired = [
                _auc(_policy_record(e, name)) - _auc(_policy_record(e, baseline)) for e in episodes
            ]
            baseline_median = _median([_auc(_policy_record(e, baseline)) for e in episodes])
            policy_median = _median([_auc(_policy_record(e, name)) for e in episodes])
            deltas[baseline] = {
                "median_auc_delta": _median(paired),
                "relative_median_auc": (policy_median - baseline_median)
                / max(abs(baseline_median), 1e-12),
                "bootstrap": paired_bootstrap(
                    paired, samples=bootstrap_samples, seed=bootstrap_seed
                ),
            }
        result["policies"][name]["paired_deltas"] = deltas
    return result


def select_champion(
    summary: Mapping[str, Any], identities: Mapping[str, Mapping[str, str]] | None = None
) -> str | None:
    policies = cast(Mapping[str, Mapping[str, Any]], summary.get("policies", {}))
    candidates = [name for name in policies if name not in {"random", "structural"}]
    if not candidates:
        return None

    def key(name: str) -> tuple[float, float, float, str, str]:
        metrics = policies[name]
        by_order = cast(Mapping[str, Mapping[str, Any]], metrics.get("by_order", {}))
        order10 = float(by_order.get("10", {}).get("median_auc", float("-inf")))
        witness = float(metrics.get("pooled_median_best_witnesses", float("inf")))
        ast = (identities or {}).get(name, {}).get("normalized_ast_sha256", "")
        return (
            -float(metrics.get("pooled_median_auc", float("-inf"))),
            -order10,
            witness,
            ast,
            name,
        )

    return min(candidates, key=key)


def evaluate_gate(
    summary: Mapping[str, Any],
    *,
    champion: str | None = None,
    infrastructure: Mapping[str, Any] | None = None,
) -> str:
    """Return the only three legal Stage 3 decisions.

    Scientific failures are NO_GO; missing/invalid infrastructure is inconclusive.
    Exactly twelve named checks are evaluated, making gate reports auditable.
    """
    policies = cast(Mapping[str, Mapping[str, Any]], summary.get("policies", {}))
    champ = champion or select_champion(summary)
    infra = infrastructure or {}
    if bool(infra.get("failure")) or any(
        bool(infra.get(key))
        for key in (
            "auth_failure",
            "protocol_failure",
            "usage_failure",
            "campaign_failure",
        )
    ):
        return INCONCLUSIVE_INFRASTRUCTURE_FAILURE
    checks = _gate_checks(summary, policies, champ)
    if any(not value for value in checks.values()):
        infra_keys = {
            "dependency_provenance",
            "protocol_safety",
            "campaign_authority",
            "exact_usage",
            "primary_replay_exact",
            "zero_invalid_and_worker_failures",
            "selected_only_equal_bounded_parity",
            "repository_and_heg_validation",
        }
        if any(not checks[key] for key in infra_keys):
            return INCONCLUSIVE_INFRASTRUCTURE_FAILURE
        return NO_GO
    return GO_TO_STAGE_4


def _gate_checks(
    summary: Mapping[str, Any],
    policies: Mapping[str, Mapping[str, Any]],
    champion: str | None,
) -> dict[str, bool]:
    random_relative = (
        float(summary["champion_random_relative"])
        if "champion_random_relative" in summary
        else float(
            policies.get(champion, {})
            .get("paired_deltas", {})
            .get("random", {})
            .get("relative_median_auc", 0.0)
        )
        if champion
        else 0.0
    )
    structural_relative = (
        float(summary["champion_structural_relative"])
        if "champion_structural_relative" in summary
        else (
            float(policies.get(champion, {}).get("pooled_median_auc", 0.0))
            / max(
                abs(float(policies.get("structural", {}).get("pooled_median_auc", 0.0))),
                1e-12,
            )
        )
        if champion
        else 0.0
    )
    unique = int(
        summary.get(
            "minimum_unique",
            len([name for name in policies if name not in {"random", "structural"}]),
        )
    )
    return {
        "dependency_provenance": bool(summary.get("dependency_provenance", False)),
        "protocol_safety": bool(summary.get("protocol_safety", False)),
        "campaign_authority": bool(summary.get("campaign_authority", False)),
        "exact_usage": bool(summary.get("exact_usage", False)),
        "minimum_unique": unique >= 4,
        "baseline_ast_distinct": bool(summary.get("baseline_ast_distinct", False)),
        "champion_random_relative": random_relative >= 0.05,
        "champion_structural_relative": structural_relative >= 0.90,
        "primary_replay_exact": bool(summary.get("primary_replay_exact", False)),
        "zero_invalid_and_worker_failures": (
            int(summary.get("invalid_records", 1)) == 0
            and int(summary.get("worker_failures", 1)) == 0
        ),
        "selected_only_equal_bounded_parity": bool(
            summary.get("selected_only_equal_bounded_parity", False)
        ),
        "repository_and_heg_validation": bool(summary.get("repository_and_heg_validation", False)),
    }


def gate_report(
    summary: Mapping[str, Any],
    *,
    champion: str | None = None,
    infrastructure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persistable gate artifact containing all named checks and decision."""
    policies = cast(Mapping[str, Mapping[str, Any]], summary.get("policies", {}))
    champ = champion or select_champion(summary)
    checks = _gate_checks(summary, policies, champ)
    return {
        "champion": champ,
        "checks": checks,
        "decision": evaluate_gate(summary, champion=champion, infrastructure=infrastructure),
    }


# Compatibility aliases used by early experiment scripts.
summarize = summarize_development
gate = evaluate_gate
summarize_episodes = summarize_development
development_gate = evaluate_gate
evaluate_development_gate = evaluate_gate
stage3_gate = evaluate_gate
select_champion_policy = select_champion


def hierarchical_bootstrap(
    episodes: list[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 2026072909,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Graph-seed then policy-seed paired bootstrap, independent of completion order."""
    candidate = (
        "development"
        if "development" in (episodes[0].get("policies", episodes[0]))
        else "candidate"
    )
    grouped: dict[int, dict[int, list[float]]] = {}
    for episode in episodes:
        order = int(episode["order"])
        graph = int(episode["graph_seed"])
        delta = _auc(_policy_record(episode, candidate)) - _auc(_policy_record(episode, "random"))
        grouped.setdefault(order, {}).setdefault(graph, []).append(delta)

    def sample(values: dict[int, list[float]], index: int) -> float:
        digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        graphs = sorted(values)
        draws: list[float] = []
        for _ in graphs:
            graph = graphs[rng.randrange(len(graphs))]
            policies = values[graph]
            draws.extend(policies[rng.randrange(len(policies))] for _ in policies)
        return float(statistics.median(draws))

    def interval(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        alpha = (1.0 - confidence) / 2.0

        def pct(p: float) -> float:
            pos = p * (len(ordered) - 1)
            lo, hi = math.floor(pos), math.ceil(pos)
            return (
                ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
            )

        return {
            "samples": samples,
            "seed": seed,
            "confidence_level": confidence,
            "median": _median(values),
            "interval": [pct(alpha), pct(1.0 - alpha)],
        }

    by_order = {
        str(order): interval([sample(values, i) for i in range(samples)])
        for order, values in grouped.items()
    }
    pooled_values = [
        value for values in grouped.values() for policies in values.values() for value in policies
    ]
    return {
        "by_order": by_order,
        "pooled": paired_bootstrap(
            pooled_values, samples=samples, seed=seed, confidence=confidence
        ),
    }
