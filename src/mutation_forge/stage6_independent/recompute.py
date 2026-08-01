"""Independent reduction of raw Stage 5 shard artifacts.

Only the raw gzip shards and the retained result JSON are consumed here.  No
Stage 5 implementation is imported: parsing, integrity checks, metrics,
bootstrap, and field-level comparison all use the independent package.
"""

# ruff: noqa: E501

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .metrics import (
    BOOTSTRAP_SAMPLES,
    EFFECTS,
    POLICY_IDS,
    bootstrap,
    fraction_payload,
    gates,
    parse_metrics_episodes,
    summarize,
)
from .persistence import canonical_record_hash, timing_stripped

STAGE5_BOOTSTRAP_SEED = 2_026_080_103
STAGE5_RESULT_NAME = "stage5-terminal.json"
STAGE5_SUMMARY_NAME = "stage5-summary.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_from_gzip(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("shard row must be an object")
                    rows.append(cast(dict[str, Any], value))
    except (OSError, EOFError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid shard {path}: {error}") from error
    return rows


def load_stage5_pass(
    evidence: str | Path, pass_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and independently validate one preserved Stage 5 pass."""

    if pass_name not in {"primary", "replay"}:
        raise ValueError("pass_name must be primary or replay")
    root = Path(evidence).resolve() / pass_name
    summaries = sorted(root.glob(f"*-{pass_name}-summary.json"))
    if len(summaries) != 1:
        raise ValueError(f"expected one {pass_name} summary")
    summary = _json(summaries[0])
    entries = summary.get("shards")
    if not isinstance(entries, list) or len(entries) != 24:
        raise ValueError("Stage 5 pass must contain exactly 24 shard entries")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"shard {index} entry is not an object")
        name = str(raw_entry.get("path", ""))
        if Path(name).name != name or not name.endswith(".jsonl.gz"):
            raise ValueError("unsafe shard path")
        path = root / name
        if not path.is_file() or _sha(path) != str(raw_entry.get("file_sha256")):
            raise ValueError(f"shard {index} file hash mismatch")
        rows = _rows_from_gzip(path)
        expected_ids = [str(item) for item in raw_entry.get("episode_ids", [])]
        actual_ids = [str(row.get("episode_id", "")) for row in rows]
        if len(rows) != int(raw_entry.get("record_count", -1)) or actual_ids != expected_ids:
            raise ValueError(f"shard {index} roster mismatch")
        if seen.intersection(actual_ids):
            raise ValueError("duplicate episode identity across shards")
        seen.update(actual_ids)
        for row in rows:
            declared = row.get("canonical_episode_sha256")
            if declared is not None and str(declared) != canonical_record_hash(row):
                raise ValueError(f"canonical record hash mismatch: {row.get('episode_id')}")
        records.extend(rows)
    expected = [str(item) for item in summary.get("manifest_episode_ids", [])]
    if len(records) != 1536 or sorted(seen) != sorted(expected):
        raise ValueError("Stage 5 pass roster is incomplete")
    return summary, sorted(records, key=lambda row: str(row["episode_id"]))


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_payload_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _payload_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _effect_payload(summary: Any, effect: str) -> dict[str, Any]:
    value = summary.effects[effect]
    return {
        "theta": fraction_payload(value.theta),
        "relative_improvement": fraction_payload(summary.relative_improvements[effect]),
        "order_deltas": {str(k): fraction_payload(v) for k, v in value.order_deltas.items()},
        "relabel_deltas": {
            f"{o}-{g}-{r}": fraction_payload(v) for (o, g, r), v in value.relabel_deltas.items()
        },
        "stratum_deltas": {
            f"{o}-{r}": fraction_payload(v) for (o, r), v in value.stratum_deltas.items()
        },
        "sign_counts": {key: dict(counts) for key, counts in value.sign_counts.items()},
    }


def independent_result_payload(summary: Any, bootstrap_summary: Any) -> dict[str, Any]:
    return {
        "policy_means": {
            key: fraction_payload(value) for key, value in summary.policy_means.items()
        },
        "effects": {effect: _effect_payload(summary, effect) for effect in EFFECTS},
        "structural_retention": fraction_payload(summary.structural_retention),
        "bootstrap": bootstrap_summary.as_dict(),
    }


def compare_retained_result(
    independent: Mapping[str, Any], retained: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every scientific field represented in the retained result."""

    expected = retained.get("metrics")
    if not isinstance(expected, Mapping):
        return {"exact": False, "differences": [{"path": "$.metrics", "reason": "missing"}]}
    differences: list[dict[str, Any]] = []

    def check(path: str, left: Any, right: Any) -> None:
        if not _payload_equal(left, right):
            differences.append({"path": path, "independent": left, "retained": right})

    check("$.metrics.policy_means", independent.get("policy_means"), expected.get("policy_means"))
    check("$.metrics.effects", independent.get("effects"), expected.get("effects"))
    check(
        "$.metrics.structural_retention",
        independent.get("structural_retention"),
        expected.get("structural_retention"),
    )

    left_boot = independent.get("bootstrap", {})
    right_boot = expected.get("bootstrap", {})
    for field in (
        "samples",
        "seed",
        "percentile_rule",
        "observed",
        "intervals",
        "sign_counts",
        "support",
    ):
        check(f"$.metrics.bootstrap.{field}", left_boot.get(field), right_boot.get(field))
    # Draw support is lossless and is the canonical 10,000-draw comparison.
    check(
        "$.metrics.bootstrap.draw_support",
        left_boot.get("draw_support"),
        right_boot.get("draw_support"),
    )
    return {"exact": not differences, "differences": differences}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _all_rows(rows: list[dict[str, Any]], predicate: Any) -> bool:
    return all(bool(predicate(row)) for row in rows)


def _recompute_stage5_gates(
    primary_summary: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    primary_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    freeze: Mapping[str, Any],
    replay_exact: bool,
    metric_gates: Mapping[str, bool],
) -> dict[str, bool]:
    frozen_policies = freeze.get("policies", {})
    policy_provenance = isinstance(frozen_policies, Mapping) and _all_rows(
        primary_rows + replay_rows,
        lambda row: set(row.get("policies", {})) == set(POLICY_IDS)
        and row.get("policy_source_sha256")
        == {
            policy: cast(Mapping[str, Any], frozen_policies[policy]).get("source_sha256")
            for policy in POLICY_IDS
        },
    )
    primary_ids = [str(row.get("episode_id", "")) for row in primary_rows]
    replay_ids = [str(row.get("episode_id", "")) for row in replay_rows]
    manifest_disjointness = freeze.get("manifest_disjointness", {})
    manifest_complete = (
        len(primary_rows) == 1536
        and len(replay_rows) == 1536
        and len(primary_ids) == len(set(primary_ids))
        and len(replay_ids) == len(set(replay_ids))
        and sorted(primary_ids) == sorted(str(item) for item in primary_summary.get("manifest_episode_ids", []))
        and sorted(replay_ids) == sorted(str(item) for item in replay_summary.get("manifest_episode_ids", []))
        and isinstance(manifest_disjointness, Mapping)
        and all(bool(manifest_disjointness.get(key)) for key in ("orders_disjoint", "base_identities_disjoint", "complete_identities_disjoint"))
    )
    budgets = _all_rows(
        primary_rows + replay_rows,
        lambda row: row.get("evaluation_count") == 128
        and row.get("selected_score_calls") == 128
        and row.get("horizon") == 32
        and isinstance(row.get("metrics_input"), Mapping)
        and set(cast(Mapping[str, Any], row["metrics_input"]).get("policies", {})) == set(POLICY_IDS),
    )
    graph_valid = _all_rows(
        primary_rows + replay_rows,
        lambda row: row.get("invalid_graphs") == 0
        and isinstance(row.get("relabel_proof"), Mapping)
        and _valid_permutation(cast(Mapping[str, Any], row["relabel_proof"]).get("permutation"), row.get("order"))
        and cast(Mapping[str, Any], row["relabel_proof"]).get("base_graph_hash") == row.get("base_graph_hash")
        and cast(Mapping[str, Any], row["relabel_proof"]).get("relabeled_graph_hash") == row.get("relabeled_graph_hash")
        and cast(Mapping[str, Any], row["relabel_proof"]).get("canonical_unlabeled_hash") == row.get("canonical_unlabeled_hash"),
    )
    worker_clean = _all_rows(
        primary_rows + replay_rows,
        lambda row: row.get("terminal_status") == "completed"
        and row.get("policy_failures") == 0
        and all(
            not bool(cast(Mapping[str, Any], trace).get(flag))
            for step in cast(list[Any], row.get("steps", []))
            for trace in cast(Mapping[str, Any], step).get("policies", {}).values()
            for flag in ("exception", "timeout", "crash", "protocol")
        ),
    )
    selected_only = _all_rows(
        primary_rows + replay_rows,
        lambda row: row.get("oracle_score_calls") == 0
        and row.get("selected_score_calls") == row.get("evaluation_count") == 128,
    )
    zero_calls = _all_rows(
        primary_rows + replay_rows,
        lambda row: all(row.get(key) == 0 for key in ("model_calls", "app_server_calls", "runtime_network_calls"))
        and row.get("oracle_score_calls") == 0,
    )
    artifact_provenance = (
        primary_summary.get("status") == "completed"
        and replay_summary.get("status") == "completed"
        and primary_summary.get("shard_count") == replay_summary.get("shard_count") == 24
        and primary_summary.get("record_count") == replay_summary.get("record_count") == 1536
        and freeze.get("provider_calls_allowed") is False
        and freeze.get("stage6_started") is False
        and freeze.get("freeze_sha256") == _canonical_hash({key: value for key, value in freeze.items() if key != "freeze_sha256"})
    )
    return {
        "1_policy_provenance_exact": policy_provenance,
        "2_manifest_complete_and_disjoint": manifest_complete,
        "3_primary_and_replay_complete_equal_budgets": budgets and primary_summary.get("status") == "completed" and replay_summary.get("status") == "completed",
        "4_timing_stripped_replay_identity_exact": replay_exact,
        "5_graph_validity_100_percent": graph_valid,
        "6_zero_worker_failures_crashes_timeouts_protocol_violations": worker_clean,
        "7_selected_plan_only_zero_oracle": selected_only,
        "8_zero_model_app_server_runtime_network_calls": zero_calls,
        "9_C_vs_stage3_relative_improvement_ge_2_percent": bool(metric_gates.get("relative_improvement_C_vs_stage3_at_least_threshold")),
        "10_C_vs_stage3_bootstrap_lower_bound_positive": bool(metric_gates.get("bootstrap_C_vs_stage3_lower_bound_positive")),
        "11_C_vs_stage3_nonnegative_each_order": bool(metric_gates.get("C_vs_stage3_nonnegative_each_order")),
        "12_C_vs_stage3_nonnegative_all_six_order_relabel_strata": bool(metric_gates.get("C_vs_stage3_nonnegative_all_six_order_relabel_strata")),
        "13_C_vs_random_threshold_and_lower_bound": bool(metric_gates.get("relative_improvement_C_vs_random_at_least_threshold")) and bool(metric_gates.get("bootstrap_C_vs_random_lower_bound_positive")),
        "14_structural_retention_ge_99_percent": bool(metric_gates.get("structural_retention_at_least_threshold")),
        "15_artifact_provenance_preservation_repository_verified": artifact_provenance,
    }


def _valid_permutation(value: Any, order: Any) -> bool:
    if not isinstance(value, list):
        return False
    try:
        expected = list(range(int(order)))
    except (TypeError, ValueError):
        return False
    return len(value) == len(expected) and sorted(value) == expected


def recompute_stage5(evidence: str | Path) -> dict[str, Any]:
    """Recompute Stage 5 science from raw primary shards and compare it."""

    primary_summary, primary_rows = load_stage5_pass(evidence, "primary")
    replay_summary, replay_rows = load_stage5_pass(evidence, "replay")
    primary_by_id = {str(row["episode_id"]): timing_stripped(row) for row in primary_rows}
    replay_by_id = {str(row["episode_id"]): timing_stripped(row) for row in replay_rows}
    replay_exact = primary_by_id == replay_by_id
    episodes = parse_metrics_episodes(primary_rows, POLICY_IDS)
    reduced = summarize(episodes, POLICY_IDS)
    draws = bootstrap(reduced, samples=BOOTSTRAP_SAMPLES, seed=STAGE5_BOOTSTRAP_SEED)
    scientific = independent_result_payload(reduced, draws)
    retained_path = Path(evidence).resolve() / STAGE5_RESULT_NAME
    retained = _json(retained_path)
    comparison = compare_retained_result(scientific, retained)
    metric_gates = gates(
        reduced,
        draws,
        champion_stage3_threshold=0.02,
        champion_random_threshold=0.05,
        structural_retention_threshold=0.99,
    )
    freeze = _json(Path(evidence).resolve() / "stage5-generalization-freeze-v1.json")
    recomputed_gates = _recompute_stage5_gates(
        primary_summary,
        replay_summary,
        primary_rows,
        replay_rows,
        freeze,
        replay_exact,
        metric_gates,
    )
    retained_gates = retained.get("gates")
    gates_exact = _payload_equal(recomputed_gates, retained_gates)
    decision = "GO_TO_STAGE_6" if all(recomputed_gates.values()) else "NO_GO"
    comparison["differences"].append({"path": "$.gates", "independent": recomputed_gates, "retained": retained_gates}) if not gates_exact else None
    comparison["differences"].append({"path": "$.decision", "independent": decision, "retained": retained.get("decision")}) if decision != retained.get("decision") else None
    comparison["exact"] = not comparison["differences"]
    return {
        "schema_version": "stage6.verification.recomputation.v1",
        "status": "passed" if comparison["exact"] and replay_exact else "failed",
        "record_count": len(primary_rows),
        "primary_run_identity": primary_summary.get("run_identity_sha256"),
        "replay_run_identity": replay_summary.get("run_identity_sha256"),
        "replay_exact": replay_exact,
        "comparison": comparison,
        "metrics": scientific,
        "metric_gates": metric_gates,
        "gates": recomputed_gates,
        "terminal_decision": decision,
        "retained_result_sha256": _sha(retained_path),
    }


__all__ = [
    "STAGE5_BOOTSTRAP_SEED",
    "compare_retained_result",
    "independent_result_payload",
    "load_stage5_pass",
    "recompute_stage5",
]
