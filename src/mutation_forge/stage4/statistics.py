"""Deterministic Stage 4 reductions, bootstrap and terminal gate.

The reducer deliberately accepts the small JSON mappings emitted by the Stage 3
and Stage 4 runners.  It does not execute policies or consult the HEG; all
inputs are treated as evidence and validated before a decision is made.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

GO_TO_STAGE_5 = "GO_TO_STAGE_5"
NO_GO = "NO_GO"
INCONCLUSIVE_INFRASTRUCTURE_FAILURE = "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"

# The names are part of the persisted report contract.  Keep this tuple in
# issue order (a--l).
GATE_NAMES = (
    "dependency_import_provenance_heg",
    "four_generations_exact_usage",
    "minimum_unique_offspring",
    "champion_distinct",
    "pooled_validation_improvement",
    "bootstrap_lower_bound",
    "order_deltas_nonnegative",
    "graph_seed_deltas_nonnegative",
    "structural_retention",
    "primary_replay_exact",
    "validity_worker_health",
    "archive_lineage_repository",
)

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026073004
DEFAULT_CONFIDENCE = 0.95
EXPECTED_ORDERS = (10, 12)
EXPECTED_GRAPH_SEEDS_PER_ORDER = 4
EXPECTED_POLICY_SEEDS_PER_GRAPH = 16


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _median(values: Iterable[float], name: str = "metric") -> float:
    vals = [_finite(v, name) for v in values]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


def _policy_record(episode: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    container = episode.get("policies", episode)
    value = container.get(policy) if isinstance(container, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"episode missing policy {policy}")
    return value


def _auc(record: Mapping[str, Any]) -> float:
    for key in ("normalized_best_so_far_auc", "normalized_best_so_far_AUC", "auc"):
        if key in record and record[key] is not None:
            return _nonnegative(record[key], "AUC")
    curve = record.get("normalized_best_so_far_curve", record.get("curve"))
    if curve is None:
        raise ValueError("policy record missing normalized best-so-far curve/AUC")
    if isinstance(curve, (str, bytes)) or not isinstance(curve, Iterable):
        raise ValueError("AUC curve must be iterable")
    vals = [_nonnegative(v, "AUC curve value") for v in curve]
    if not vals:
        raise ValueError("AUC curve must not be empty")
    return float(sum(vals) / len(vals))


def _metric(record: Mapping[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    for key in names:
        if key in record and record[key] is not None:
            return _nonnegative(record[key], key)
    return default


def _identity(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    identities = summary.get("policy_identities", {})
    if isinstance(identities, Mapping) and isinstance(identities.get(name), Mapping):
        return cast(Mapping[str, Any], identities[name])
    policies = summary.get("policies", {})
    item = policies.get(name) if isinstance(policies, Mapping) else None
    return cast(Mapping[str, Any], item) if isinstance(item, Mapping) else {}


def paired_bootstrap(
    values: Iterable[float],
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Median paired bootstrap with deterministic per-sample RNG streams."""
    vals = [_finite(v, "bootstrap value") for v in values]
    if not vals:
        raise ValueError("bootstrap requires at least one observation")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    seed = _integer(seed, "bootstrap seed")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be in (0,1)")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    estimates: list[float] = []
    for i in range(samples):
        digest = hashlib.sha256(f"{int(seed)}:{i}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        estimates.append(float(statistics.median(vals[rng.randrange(len(vals))] for _ in vals)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "samples": samples,
        "seed": seed,
        "confidence_level": confidence,
        "median": _median(vals, "bootstrap value"),
        "interval": [_percentile(estimates, alpha), _percentile(estimates, 1.0 - alpha)],
    }


def _matrix_rows(
    episodes: Sequence[Mapping[str, Any]], candidate: str
) -> dict[int, dict[int, dict[int, Mapping[str, Any]]]]:
    """Return order -> graph seed -> policy seed -> episode mapping."""
    grouped: dict[int, dict[int, dict[int, Mapping[str, Any]]]] = {}
    for episode in episodes:
        try:
            order = _integer(episode["order"], "order")
            graph = _integer(episode["graph_seed"], "graph_seed")
            policy_seed = _integer(episode["policy_seed"], "policy_seed")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("episode is missing integer order/graph_seed/policy_seed") from exc
        _policy_record(episode, candidate)
        _policy_record(episode, "random")
        cell = grouped.setdefault(order, {}).setdefault(graph, {})
        if policy_seed in cell:
            raise ValueError("duplicate policy seed in balanced matrix")
        cell[policy_seed] = episode
    if set(grouped) != set(EXPECTED_ORDERS):
        raise ValueError("bootstrap order coverage mismatch")
    for order in EXPECTED_ORDERS:
        if len(grouped[order]) != EXPECTED_GRAPH_SEEDS_PER_ORDER:
            raise ValueError("bootstrap requires four graph seeds per order")
        for _graph, policies in grouped[order].items():
            if len(policies) != EXPECTED_POLICY_SEEDS_PER_GRAPH:
                raise ValueError("bootstrap requires sixteen policy seeds per graph")
    return grouped


def _delta(episode: Mapping[str, Any], policy: str, baseline: str) -> float:
    return _auc(_policy_record(episode, policy)) - _auc(_policy_record(episode, baseline))


def hierarchical_bootstrap(
    episodes: Sequence[Mapping[str, Any]],
    *,
    policy: str | None = None,
    baseline: str = "random",
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Graph-seed then policy-seed bootstrap, retaining policy pairing.

    The matrix is intentionally strict: two orders, four graph seeds per
    order, and sixteen policy seeds per graph.  This prevents accidental
    weighting changes from incomplete searches.
    """
    if not episodes:
        raise ValueError("episodes must not be empty")
    first = episodes[0].get("policies", episodes[0])
    if policy is None:
        policy = "stage4" if isinstance(first, Mapping) and "stage4" in first else "candidate"
    grouped = _matrix_rows(episodes, policy)
    seed = _integer(seed, "bootstrap seed")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be in (0,1)")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if baseline not in {"random", "structural", "stage3_champion"}:
        raise ValueError("unsupported bootstrap baseline")

    def draw_values(order: int, sample_index: int) -> list[float]:
        digest = hashlib.sha256(f"{seed}:{sample_index}:{order}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        graph_ids = sorted(grouped[order])
        selected: list[float] = []
        for _ in graph_ids:
            graph = graph_ids[rng.randrange(len(graph_ids))]
            by_policy = grouped[order][graph]
            pids = sorted(by_policy)
            for _ in pids:
                pid = pids[rng.randrange(len(pids))]
                selected.append(_delta(by_policy[pid], policy, baseline))
        return selected

    by_order: dict[str, Any] = {}
    order_draws: dict[int, list[float]] = {}
    for order in EXPECTED_ORDERS:
        draws = [float(statistics.median(draw_values(order, i))) for i in range(samples)]
        order_draws[order] = draws
        alpha = (1.0 - confidence) / 2.0
        observed = [
            _delta(ep, policy, baseline)
            for graph in grouped[order].values()
            for ep in graph.values()
        ]
        by_order[str(order)] = {
            "samples": samples,
            "seed": seed,
            "confidence_level": float(confidence),
            "median": _median(observed, "paired delta"),
            "interval": [_percentile(draws, alpha), _percentile(draws, 1.0 - alpha)],
        }
    pooled_observed = [
        _delta(ep, policy, baseline)
        for order in EXPECTED_ORDERS
        for graph in grouped[order].values()
        for ep in graph.values()
    ]
    pooled_draws = [
        float(
            statistics.median(
                draw_values(EXPECTED_ORDERS[0], i) + draw_values(EXPECTED_ORDERS[1], i)
            )
        )
        for i in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "policy": policy,
        "baseline": baseline,
        "by_order": by_order,
        "pooled": {
            "samples": samples,
            "seed": seed,
            "confidence_level": float(confidence),
            "median": _median(pooled_observed, "paired delta"),
            "interval": [
                _percentile(pooled_draws, alpha),
                _percentile(pooled_draws, 1.0 - alpha),
            ],
        },
    }


def _curve_compatible(records: Sequence[Mapping[str, Any]]) -> bool:
    lengths: set[int] = set()
    missing = False
    for record in records:
        curve = record.get("normalized_best_so_far_curve", record.get("curve"))
        if curve is None:
            missing = True
            continue
        if isinstance(curve, (str, bytes)) or not isinstance(curve, Iterable):
            return False
        values = list(curve)
        if not values:
            return False
        lengths.add(len(values))
    # A complete curve is present for every record and all curves have the
    # same horizon.  AUC-only legacy fixtures therefore report incompatibility
    # rather than silently claiming complete trajectory evidence.
    return bool(lengths) and not missing and len(lengths) == 1


def summarize_development(
    episodes: Sequence[Mapping[str, Any]],
    policies: Iterable[str] | None = None,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    ordered = sorted(episodes, key=lambda e: str(e.get("episode_id", "")))
    container = ordered[0].get("policies", ordered[0])
    if not isinstance(container, Mapping):
        raise ValueError("episodes must contain policy mappings")
    names = list(policies or [str(k) for k, v in container.items() if isinstance(v, Mapping)])
    if "random" not in names or "structural" not in names:
        raise ValueError("random and structural baselines are required")
    result: dict[str, Any] = {"episode_count": len(ordered), "policies": {}}
    orders = sorted({_integer(e["order"], "order") for e in ordered})
    for name in names:
        records = [_policy_record(e, name) for e in ordered]
        auc_values = [_auc(r) for r in records]
        witnesses = [_metric(r, ("best_total_witnesses", "best_total")) for r in records]
        by_order: dict[str, Any] = {}
        for order in orders:
            selected = [
                _policy_record(e, name) for e in ordered if _integer(e["order"], "order") == order
            ]
            vals = [_auc(r) for r in selected]
            ws = [_metric(r, ("best_total_witnesses", "best_total")) for r in selected]
            by_order[str(order)] = {
                "episodes": len(selected),
                "median_auc": _median(vals, "AUC"),
                "median_normalized_best_so_far_auc": _median(vals, "AUC"),
                "median_best_total_witness": _median(ws, "best_total_witnesses"),
                "median_best_witnesses": _median(ws, "best_total_witnesses"),
                "complete_curve_compatible": _curve_compatible(selected),
                "median_accepted": _median(
                    (_metric(r, ("accepted_count",)) for r in selected), "accepted_count"
                ),
                "median_rejected": _median(
                    (_metric(r, ("rejected_count",)) for r in selected), "rejected_count"
                ),
                "median_duplicate": _median(
                    (_metric(r, ("duplicate_count",)) for r in selected), "duplicate_count"
                ),
                "median_nonimproving": _median(
                    (_metric(r, ("nonimproving_count",)) for r in selected), "nonimproving_count"
                ),
                "median_divergence": _median(
                    (_metric(r, ("divergence_count", "divergence")) for r in selected),
                    "divergence_count",
                ),
                "median_evaluations_to_first_improvement": _median(
                    (_metric(r, ("evaluations_to_first_improvement",)) for r in selected),
                    "evaluations_to_first_improvement",
                ),
                "median_first_improvement_ns": _median(
                    (
                        _metric(r, ("first_improvement_ns", "time_to_first_improvement_ns"))
                        for r in selected
                    ),
                    "first_improvement_ns",
                ),
            }

        def med(
            fields: Sequence[str], policy_records: Sequence[Mapping[str, Any]] = records
        ) -> float:
            return _median((_metric(r, fields) for r in policy_records), fields[0])

        result["policies"][name] = {
            "by_order": by_order,
            "pooled_median_auc": _median(auc_values, "AUC"),
            "pooled_median_normalized_best_so_far_auc": _median(auc_values, "AUC"),
            "pooled_median_best_total_witness": _median(witnesses, "best_total_witnesses"),
            "pooled_median_best_total_witnesses": _median(witnesses, "best_total_witnesses"),
            "pooled_median_best_witnesses": _median(witnesses, "best_total_witnesses"),
            "complete_curve_compatible": _curve_compatible(records),
            "median_accepted": med(("accepted_count",)),
            "median_rejected": med(("rejected_count",)),
            "median_duplicate": med(("duplicate_count",)),
            "median_duplicates": med(("duplicate_count",)),
            "median_nonimproving": med(("nonimproving_count",)),
            "median_divergence": med(("divergence_count", "divergence")),
            "median_evaluations_to_first_improvement": med(("evaluations_to_first_improvement",)),
            "median_first_improvement_ns": med(("first_improvement_ns",)),
            "median_time_to_first_improvement": med(
                ("time_to_first_improvement", "first_improvement_ns")
            ),
            "phase_timing": {
                "median": _phase_medians(records),
            },
            "worker_health": _worker_health(records),
        }
    # Compute paired deltas for every non-baseline policy against all useful
    # baselines.  The Stage 3 champion is optional in small fixtures.
    for name, metrics in result["policies"].items():
        if name in {"random", "structural"}:
            continue
        deltas: dict[str, Any] = {}
        for baseline in ("stage3_champion", "random", "structural"):
            if baseline == "stage3_champion" and not all(
                isinstance(e.get("policies", e), Mapping)
                and "stage3_champion" in e.get("policies", e)
                for e in ordered
            ):
                continue
            paired = [_delta(e, name, baseline) for e in ordered]
            baseline_vals = [_auc(_policy_record(e, baseline)) for e in ordered]
            candidate_median = cast(float, metrics["pooled_median_auc"])
            baseline_median = _median(baseline_vals, "AUC")
            deltas[baseline] = {
                "median_auc_delta": _median(paired, "paired delta"),
                "relative_median_auc": (candidate_median - baseline_median)
                / max(abs(baseline_median), 1e-12),
                "bootstrap": paired_bootstrap(
                    paired, samples=bootstrap_samples, seed=bootstrap_seed
                ),
            }
        metrics["paired_deltas"] = deltas
    return result


def _phase_medians(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys: set[str] = set()
    for record in records:
        value = record.get("phase_timing", record.get("phase_timings", {}))
        if isinstance(value, Mapping):
            keys.update(str(k) for k in value)
    return {
        key: _median(
            (
                _metric(
                    cast(Mapping[str, Any], r.get("phase_timing", r.get("phase_timings", {}))),
                    (key,),
                )
                for r in records
            ),
            key,
        )
        for key in sorted(keys)
    }


def _worker_health(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [r.get("worker_health", r.get("worker_status")) for r in records]
    failures = sum(1 for value in values if value in {False, "failed", "failure"})
    return {"records": len(values), "failures": failures, "healthy": failures == 0}


def select_champion(
    summary: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    generation: int = 4,
    seed_ast_hashes: Iterable[str] = (),
    baseline_ast_hashes: Iterable[str] = (),
) -> str | None:
    """Select exactly one new Stage 4 policy using the frozen four-key order."""
    if isinstance(generation, bool) or generation < 4:
        return None
    policy_map = summary.get("policies", {})
    if not isinstance(policy_map, Mapping):
        return None
    identities = identities or cast(
        Mapping[str, Mapping[str, Any]], summary.get("policy_identities", {})
    )
    forbidden = set(seed_ast_hashes) | set(baseline_ast_hashes)
    forbidden |= set(cast(Iterable[str], summary.get("seed_ast_hashes", ())))
    forbidden |= set(cast(Iterable[str], summary.get("baseline_ast_hashes", ())))
    names: list[str] = []
    for name, metrics in policy_map.items():
        if name in {"random", "structural", "stage3_champion"} or not isinstance(metrics, Mapping):
            continue
        identity = identities.get(str(name), {}) if isinstance(identities, Mapping) else {}
        origin = str(identity.get("origin", identity.get("stage", "stage4"))).lower()
        if identity.get("is_stage4") is False or origin in {"seed", "baseline", "stage3"}:
            continue
        ast = identity.get("normalized_ast_sha256", metrics.get("normalized_ast_sha256", ""))
        if not isinstance(ast, str) or not ast or ast in forbidden:
            continue
        names.append(str(name))
    if not names:
        return None

    def key(name: str) -> tuple[float, float, float, str]:
        metrics = cast(Mapping[str, Any], policy_map[name])
        by_order = metrics.get("by_order", {})
        order10 = by_order.get("10", {}) if isinstance(by_order, Mapping) else {}
        pooled = _nonnegative(metrics.get("pooled_median_auc"), "pooled median AUC")
        auc10 = (
            _nonnegative(order10.get("median_auc", 0.0), "order-10 median AUC")
            if isinstance(order10, Mapping)
            else 0.0
        )
        witness = _nonnegative(
            metrics.get(
                "pooled_median_best_total_witness", metrics.get("pooled_median_best_witnesses", 0.0)
            ),
            "best witness",
        )
        identity = identities.get(name, {}) if isinstance(identities, Mapping) else {}
        ast = identity.get("normalized_ast_sha256", metrics.get("normalized_ast_sha256", ""))
        return (-pooled, -auc10, witness, str(ast))

    ordered = sorted(names, key=key)
    return ordered[0]


def _bool(summary: Mapping[str, Any], *keys: str) -> bool:
    for key in keys:
        if key in summary:
            return bool(summary[key])
    return False


def _count_at_least(summary: Mapping[str, Any], threshold: int, *keys: str) -> bool:
    for key in keys:
        if key in summary:
            value = summary[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            return math.isfinite(float(value)) and float(value) >= threshold
    return False


def _validity(summary: Mapping[str, Any]) -> bool:
    graph_validity = summary.get(
        "graph_validity_100_percent", summary.get("invalid_graphs", 1) == 0
    )
    worker = summary.get("worker_failures_zero", summary.get("worker_failures", 1) == 0)
    selected = summary.get(
        "selected_plan_only", summary.get("selected_only_equal_bounded_parity", False)
    )
    oracle = summary.get("oracle_score_calls_zero", summary.get("oracle_score_calls", 1) == 0)
    equal = summary.get("equal_budgets", summary.get("equal_bounded_budgets", False))
    return bool(graph_validity) and bool(worker) and bool(selected) and bool(oracle) and bool(equal)


def _gate_checks(summary: Mapping[str, Any], champion: str | None) -> dict[str, bool]:
    policies = summary.get("policies", {})
    champ_metrics = policies.get(champion, {}) if isinstance(policies, Mapping) and champion else {}
    if not isinstance(champ_metrics, Mapping):
        champ_metrics = {}
    stage3 = (
        _finite(
            summary.get("stage3_champion_median_auc", summary.get("stage3_baseline_auc", 0.0)),
            "stage3 AUC",
        )
        if ("stage3_champion_median_auc" in summary or "stage3_baseline_auc" in summary)
        else None
    )
    if stage3 is None and isinstance(policies, Mapping):
        stage3_record = policies.get("stage3_champion", policies.get("champion"))
        if isinstance(stage3_record, Mapping) and "pooled_median_auc" in stage3_record:
            stage3 = _finite(stage3_record["pooled_median_auc"], "stage3 AUC")
    pooled = _finite(champ_metrics.get("pooled_median_auc", -1.0), "champion AUC")
    structural_auc = (
        _finite(
            (policies.get("structural", {}) or {}).get("pooled_median_auc", 0.0), "structural AUC"
        )
        if isinstance(policies, Mapping)
        else 0.0
    )
    rel = summary.get("pooled_relative_improvement")
    if rel is None:
        rel = (
            (pooled - stage3) / max(abs(stage3), 1e-12)
            if stage3 is not None
            else summary.get("champion_random_relative", -1.0)
        )
    relative = _finite(rel, "pooled relative improvement")
    bootstrap = summary.get("bootstrap", summary.get("pooled_bootstrap", {}))
    lower = summary.get("pooled_bootstrap_lower_bound")
    if lower is None and isinstance(bootstrap, Mapping):
        pooled_boot = bootstrap.get("pooled", bootstrap)
        if isinstance(pooled_boot, Mapping):
            interval = pooled_boot.get("interval", [])
            lower = interval[0] if isinstance(interval, Sequence) and interval else -1.0
    lower_value = _finite(lower if lower is not None else -1.0, "bootstrap lower bound")
    order_deltas = summary.get("order_deltas", summary.get("order_median_deltas", {}))
    order_ok = (
        all(
            _finite(order_deltas.get(str(o), order_deltas.get(o, -1.0)), "order delta") >= 0
            for o in EXPECTED_ORDERS
        )
        if isinstance(order_deltas, Mapping)
        else bool(summary.get("order_deltas_nonnegative", False))
    )
    graph_counts = summary.get(
        "graph_seed_nonnegative_counts",
        summary.get("nonnegative_graph_seeds", summary.get("graph_seed_deltas", {})),
    )
    if isinstance(graph_counts, Sequence) and not isinstance(graph_counts, (str, bytes)):
        count = sum(1 for value in graph_counts if _finite(value, "graph delta") >= 0)
        graph_counts = {str(order): count for order in EXPECTED_ORDERS}
    graph_ok = (
        all(int(graph_counts.get(str(o), graph_counts.get(o, 0))) >= 3 for o in EXPECTED_ORDERS)
        if isinstance(graph_counts, Mapping)
        else bool(summary.get("graph_seed_deltas_nonnegative", False))
    )
    structural_ratio = _finite(
        summary.get("structural_retention", pooled / max(abs(structural_auc), 1e-12)),
        "structural retention",
    )
    replay_parts = (
        "primary_replay_records_exact",
        "primary_replay_hashes_exact",
        "primary_replay_metrics_exact",
        "primary_replay_bootstrap_exact",
        "primary_replay_gate_exact",
        "primary_replay_aggregate_exact",
    )
    replay_present = [key for key in replay_parts if key in summary]
    replay_ok = (
        all(bool(summary[key]) for key in replay_present)
        if replay_present
        else _bool(summary, "primary_replay_exact", "replay_exact", "primary_replay_hashes_match")
    )
    return {
        GATE_NAMES[0]: _bool(
            summary,
            "dependency_import_provenance_heg",
            "dependency_provenance",
            "dependency_import_provenance",
        ),
        GATE_NAMES[1]: (
            (
                _bool(summary, "four_generations_exact_usage")
                or _bool(summary, "four_generations")
                or summary.get("generation_count") == 4
                or summary.get("generations") == 4
            )
            and (
                _bool(summary, "initial_turns_exact_32", "exactly_32_initial_turns")
                or _bool(summary, "protocol_safety")
                or summary.get("initial_turns") == 32
            )
            and _bool(summary, "exact_usage", "exact_budget_usage")
            and not bool(summary.get("unauthorized_tool_approval", False))
        ),
        GATE_NAMES[2]: _count_at_least(
            summary,
            16,
            "new_unique_valid_offspring",
            "new_unique_offspring",
            "minimum_unique",
        ),
        GATE_NAMES[3]: bool(champion)
        and bool(summary.get("champion_distinct", summary.get("baseline_ast_distinct", False))),
        GATE_NAMES[4]: relative >= 0.02,
        GATE_NAMES[5]: lower_value > 0.0,
        GATE_NAMES[6]: order_ok,
        GATE_NAMES[7]: graph_ok,
        GATE_NAMES[8]: structural_ratio >= 0.99,
        GATE_NAMES[9]: replay_ok and not bool(summary.get("aggregate_hash_mismatch", False)),
        GATE_NAMES[10]: _validity(summary),
        GATE_NAMES[11]: _bool(
            summary,
            "archive_lineage_repository",
            "archive_lineage_checkpoint_reindex_bounds_rich_json_repository_heg",
            "repository_and_heg_validation",
        ),
    }


def evaluate_gate(
    summary: Mapping[str, Any],
    *,
    champion: str | None = None,
    infrastructure: Mapping[str, Any] | None = None,
) -> str:
    """Return GO, NO_GO, or inconclusive infrastructure failure."""
    infra = infrastructure or {}
    incomplete = bool(
        summary.get("incomplete", summary.get("status") in {"incomplete", "interrupted"})
    )
    exhausted = bool(summary.get("exhausted", summary.get("resource_exhausted", False)))
    if incomplete and exhausted and any(bool(v) for v in infra.values()):
        return INCONCLUSIVE_INFRASTRUCTURE_FAILURE
    try:
        checks = _gate_checks(summary, champion)
    except (TypeError, ValueError, KeyError):
        # Numeric evidence is fail-closed.  Malformed science cannot be
        # upgraded to an infrastructure exception merely by omission.
        return NO_GO
    return GO_TO_STAGE_5 if all(checks.values()) else NO_GO


def gate_report(
    summary: Mapping[str, Any],
    *,
    champion: str | None = None,
    infrastructure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        checks = _gate_checks(summary, champion)
    except (TypeError, ValueError, KeyError):
        checks = {name: False for name in GATE_NAMES}
    decision = evaluate_gate(summary, champion=champion, infrastructure=infrastructure)
    report: dict[str, Any] = {"champion": champion, "checks": checks, "decision": decision}
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    report["canonical_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return report


# Compatibility names used by analysis scripts.
summarize = summarize_development
summarize_episodes = summarize_development
summarize_search = summarize_development
gate = evaluate_gate
stage4_gate = evaluate_gate
terminal_decision = evaluate_gate
select_champion_policy = select_champion
