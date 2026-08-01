"""Retained-only Stage 4R paired-delta diagnostics.

Issue #12 deliberately consumes the compact final-validation shards emitted by
Stage 4R.  It never loads a policy, starts a worker, contacts the App Server,
or evaluates a graph.  Every reduction is deterministic and every input hash
is checked before a result is written.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from mutation_forge.stage4.replay import verify_replay

DIAGNOSTIC_SCHEMA_VERSION = "stage4r.delta-diagnostic.v1"
EXPECTED_MANIFEST_SHA256 = "1d5f1b2bd4e7978337b9351fd050b0ea0069f4b30bed8cc830247724c42a777b"
EXPECTED_MANIFEST_FILE_SHA256 = (
    "87f5b6298e4c312feac2d9c4f6bafea63b70a3b29c0104a0aef33d4b91dcc91e"
)
EXPECTED_CONFIG_SHA256 = "63eea8c2ddb9318e84a161a909d18200cb9898b1e6f0e22ad4052ff71c37e179"
EXPECTED_REDUCTION_SHA256 = "de3b808e6c682db38f565a78255699752eb2ee6603cf7fbbdec380262e58e539"
EXPECTED_METRICS_INPUT_SHA256 = "879be3d6c309c7db094af632627fe2a80e39fee571c6cb5b132777a9f007f7d1"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026073004
BOOTSTRAP_CONFIDENCE = 0.95
POWER_REPLICATES = 256
POWER_SEED = 2026080101
POWER_GRAPH_COUNTS = (4, 8, 12, 16, 24, 32)
POWER_POLICY_COUNTS = (8, 16, 32)
ORDERS = (10, 12)
GRAPH_SEEDS = (451, 452, 453, 454)
POLICY_SEEDS = tuple(range(4501, 4517))
HORIZON = 32

PAIR_JSONL_NAME = "paired-deltas.jsonl"
PAIR_CSV_NAME = "paired-deltas.csv"
PAIR_HASH_NAME = "paired-deltas.sha256.json"
BOOTSTRAP_NAME = "bootstrap-support.json"
CLUSTER_NAME = "cluster-summary.json"
POWER_NAME = "power-study.json"
SUMMARY_NAME = "diagnostic-summary.json"

_STAGE3_POLICY = "stage3-candidate-slot-04"
_CHAMPION_POLICY = "champion"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(value) + b"\n")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _fraction(value: Any, name: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        try:
            parsed = Fraction(value)
            return parsed if "/" in value else parsed.limit_denominator(100_000)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{name} must be a valid rational") from exc
    number = _finite(value, name)
    result = Fraction(str(number)).limit_denominator(100_000)
    if abs(float(result) - number) > 1e-10:
        raise ValueError(f"{name} is not a stable rational decimal")
    return result


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _fraction_gcd(values: Iterable[Fraction]) -> Fraction:
    materialized = [abs(value) for value in values if value]
    if not materialized:
        return Fraction(0)
    denominator = 1
    for value in materialized:
        denominator = math.lcm(denominator, value.denominator)
    numerator = 0
    for value in materialized:
        numerator = math.gcd(numerator, abs(value.numerator * (denominator // value.denominator)))
    return Fraction(numerator, denominator)


def _median(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("median requires values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _mean(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("mean requires values")
    return sum(values, Fraction(0)) / len(values)


def _percentile(values: Sequence[Fraction], probability: float) -> tuple[Fraction, int, int, float]:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    fraction = Fraction(str(position - low)).limit_denominator(1_000_000)
    value = (
        ordered[low]
        if low == high
        else ordered[low] + (ordered[high] - ordered[low]) * fraction
    )
    return value, low, high, position


def _project(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _project(item)
            for key, item in value.items()
            if key != "timing_ns"
            and key != "timing"
            and not str(key).endswith("_ns")
            and key != "path"
        }
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    return value


def _validation_root(run: Path) -> Path:
    run = run.resolve()
    if list(run.glob("*-primary-summary.json")) and list(run.glob("*-replay-summary.json")):
        return run
    candidate = run / "evaluations" / "final-validation"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"retained final-validation artifacts not found below {run}")


def _one_summary(root: Path, pass_name: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(root.glob(f"*-{pass_name}-summary.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {pass_name} summary in {root}")
    path = candidates[0]
    return path, _read_json(path)


def _safe_relative(root: Path, value: str) -> Path:
    candidate = Path(value)
    result = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes final-validation root: {value}") from exc
    return result


def _load_rows(root: Path, summary: Mapping[str, Any], pass_name: str) -> dict[str, dict[str, Any]]:
    if summary.get("schema_version") != "stage4.evaluation.v1":
        raise ValueError(f"unexpected {pass_name} summary schema")
    if summary.get("pass") != pass_name:
        raise ValueError(f"{pass_name} summary pass mismatch")
    for key, expected in (
        ("manifest_sha256", EXPECTED_MANIFEST_SHA256),
        ("config_sha256", EXPECTED_CONFIG_SHA256),
        ("canonical_reduction_sha256", EXPECTED_REDUCTION_SHA256),
        ("metrics_input_sha256", EXPECTED_METRICS_INPUT_SHA256),
    ):
        if summary.get(key) != expected:
            raise ValueError(f"{pass_name} summary {key} drifted")
    if summary.get("record_count") != 128 or summary.get("shard_count") != 8:
        raise ValueError(f"{pass_name} summary record/shard count mismatch")
    counts = summary.get("counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(key) != expected
        for key, expected in (
            ("episodes", 128),
            ("model_calls", 0),
            ("app_server_calls", 0),
            ("oracle_score_calls", 0),
        )
    ):
        raise ValueError(f"{pass_name} summary contains forbidden calls or count drift")
    shards = summary.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)) or len(shards) != 8:
        raise ValueError(f"{pass_name} summary shard metadata mismatch")
    rows: dict[str, dict[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, Mapping) or not isinstance(shard.get("path"), str):
            raise ValueError(f"{pass_name} summary contains malformed shard metadata")
        path = _safe_relative(root, str(shard["path"]))
        if not path.is_file() or _sha_file(path) != shard.get("file_sha256"):
            raise ValueError(f"{pass_name} shard hash mismatch: {path.name}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            shard_rows = [json.loads(line) for line in handle if line.strip()]
        if len(shard_rows) != shard.get("record_count"):
            raise ValueError(f"{pass_name} shard record count mismatch: {path.name}")
        for row in shard_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("episode_id"), str):
                raise ValueError(f"{pass_name} contains malformed episode row")
            episode_id = str(row["episode_id"])
            if episode_id in rows:
                raise ValueError(f"{pass_name} contains duplicate episode {episode_id}")
            rows[episode_id] = cast(dict[str, Any], row)
    if len(rows) != 128:
        raise ValueError(f"{pass_name} has {len(rows)} rows, expected 128")
    expected_ids = summary.get("manifest_episode_ids")
    actual_ids = sorted(rows)
    expected_id_list = (
        sorted(str(value) for value in expected_ids)
        if isinstance(expected_ids, Sequence)
        else []
    )
    if expected_id_list != actual_ids:
        raise ValueError(f"{pass_name} episode identity list mismatch")
    return rows


def _find_repo_root(run: Path) -> Path:
    for candidate in (Path.cwd().resolve(), *run.resolve().parents):
        if (candidate / "configs" / "manifests" / "stage4-validation-v1.json").is_file():
            return candidate
    raise FileNotFoundError("could not locate repository root for frozen manifest")


def _validate_manifest(
    repo_root: Path, run_root: Path, primary: Mapping[str, Any]
) -> dict[str, Any]:
    freeze_path = run_root / "validation-freeze.json"
    freeze = _read_json(freeze_path) if freeze_path.is_file() else {}
    manifest_meta = freeze.get("final_validation_manifest")
    manifest_path = repo_root / "configs" / "manifests" / "stage4-validation-v1.json"
    if isinstance(manifest_meta, Mapping) and isinstance(manifest_meta.get("path"), str):
        candidate = Path(str(manifest_meta["path"]))
        if candidate.is_file():
            manifest_path = candidate
    if _sha_file(manifest_path) != EXPECTED_MANIFEST_FILE_SHA256:
        raise ValueError("frozen validation manifest file hash drifted")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "stage4.manifest.v1":
        raise ValueError("unexpected validation manifest schema")
    if manifest.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("frozen validation manifest content hash drifted")
    if primary.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("evaluation summary and frozen manifest hashes differ")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, Sequence) or len(episodes) != 128:
        raise ValueError("frozen validation manifest episode count drifted")
    ids = [str(item.get("episode_id")) for item in episodes if isinstance(item, Mapping)]
    if len(ids) != 128 or len(set(ids)) != 128:
        raise ValueError("frozen validation manifest episode identities are invalid")
    if tuple(int(value) for value in manifest.get("orders", ())) != ORDERS:
        raise ValueError("frozen validation manifest order coverage drifted")
    if tuple(int(value) for value in manifest.get("graph_seeds", ())) != GRAPH_SEEDS:
        raise ValueError("frozen validation manifest graph coverage drifted")
    if tuple(int(value) for value in manifest.get("policy_seeds", ())) != POLICY_SEEDS:
        raise ValueError("frozen validation manifest policy coverage drifted")
    if manifest.get("horizon") != HORIZON:
        raise ValueError("frozen validation manifest horizon drifted")
    return {
        "path": str(manifest_path),
        "file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "episode_count": len(ids),
        "orders": list(ORDERS),
        "graph_seeds": list(GRAPH_SEEDS),
        "policy_seeds": list(POLICY_SEEDS),
        "horizon": HORIZON,
    }


def _source_identity(
    repo_root: Path, run_root: Path, validation: Mapping[str, Any]
) -> dict[str, Any]:
    freeze = _read_json(run_root / "validation-freeze.json")
    champion = freeze.get("champion", validation.get("champion", {}))
    if not isinstance(champion, Mapping):
        raise ValueError("validation champion identity is missing")
    stage3_path = (
        repo_root
        / "runs"
        / "stage3-development"
        / "stage3-generation-1f7f0784e37c-attempt-01"
        / "revalidation"
        / "slots"
        / "slot-04"
        / "source.py"
    )
    stage3_hash = _sha_file(stage3_path) if stage3_path.is_file() else None
    stage3_ast: str | None = None
    identity_path = stage3_path.with_name("identity.json")
    if identity_path.is_file():
        identity = _read_json(identity_path)
        stage3_ast = str(identity.get("normalized_ast_sha256"))
    return {
        "champion": {
            "program_id": champion.get("program_id"),
            "source_sha256": champion.get("source_sha256"),
            "normalized_ast_sha256": champion.get("normalized_ast_sha256"),
            "origin": champion.get("origin"),
        },
        "stage3_champion": {
            "policy_id": _STAGE3_POLICY,
            "source_sha256": stage3_hash,
            "normalized_ast_sha256": stage3_ast,
        },
    }


def _policy_curve(row: Mapping[str, Any], policy: str) -> tuple[Fraction, ...]:
    policies = row.get("policies")
    if not isinstance(policies, Mapping):
        metrics = row.get("metrics_input")
        policies = metrics.get("policies") if isinstance(metrics, Mapping) else None
    if not isinstance(policies, Mapping) or not isinstance(policies.get(policy), Mapping):
        raise ValueError(f"episode is missing policy {policy}")
    record = cast(Mapping[str, Any], policies[policy])
    curve = record.get("normalized_best_so_far_curve")
    if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes)) or len(curve) != HORIZON:
        raise ValueError(f"{policy} curve must contain {HORIZON} values")
    return tuple(_fraction(value, f"{policy} curve") for value in curve)


def _policy_record(row: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    metrics = row.get("metrics_input")
    policies = metrics.get("policies") if isinstance(metrics, Mapping) else None
    if not isinstance(policies, Mapping) or not isinstance(policies.get(policy), Mapping):
        raise ValueError(f"metrics input is missing policy {policy}")
    return cast(Mapping[str, Any], policies[policy])


def _paired_rows(
    primary: Mapping[str, Mapping[str, Any]],
    replay: Mapping[str, Mapping[str, Any]],
    source_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if set(primary) != set(replay):
        raise ValueError("primary/replay episode identity sets differ")
    result: list[dict[str, Any]] = []
    for episode_id in sorted(primary):
        left, right = primary[episode_id], replay[episode_id]
        if _project(left) != _project(right):
            raise ValueError(f"primary/replay timing-stripped row mismatch: {episode_id}")
        if left.get("canonical_episode_sha256") != right.get("canonical_episode_sha256"):
            raise ValueError(f"primary/replay canonical episode hash mismatch: {episode_id}")
        metrics = left.get("metrics_input")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"episode metrics input missing: {episode_id}")
        champion_curve = _policy_curve(left, _CHAMPION_POLICY)
        stage3_curve = _policy_curve(left, _STAGE3_POLICY)
        champion_auc = sum(champion_curve, Fraction(0)) / HORIZON
        stage3_auc = sum(stage3_curve, Fraction(0)) / HORIZON
        champion_record = _policy_record(left, _CHAMPION_POLICY)
        stage3_record = _policy_record(left, _STAGE3_POLICY)
        champion_raw = champion_record.get("raw_best_so_far_curve")
        stage3_raw = stage3_record.get("raw_best_so_far_curve")
        if (
            not isinstance(champion_raw, Sequence)
            or isinstance(champion_raw, (str, bytes))
            or not isinstance(stage3_raw, Sequence)
            or isinstance(stage3_raw, (str, bytes))
            or len(champion_raw) != HORIZON
            or len(stage3_raw) != HORIZON
        ):
            raise ValueError(f"raw witness curves are incomplete: {episode_id}")
        difference = tuple(
            champion - baseline
            for champion, baseline in zip(champion_curve, stage3_curve, strict=True)
        )
        transitions = [
            {
                "step": index,
                "champion_raw": int(champion_raw[index]),
                "stage3_raw": int(stage3_raw[index]),
                "champion_normalized": float(champion_curve[index]),
                "stage3_normalized": float(stage3_curve[index]),
                "normalized_difference": _fraction_text(difference),
            }
            for index, difference in enumerate(difference)
            if difference
        ]
        initial_score = left.get("initial_score")
        initial_witnesses = (
            int(initial_score.get("total_capped_witnesses", 0))
            if isinstance(initial_score, Mapping)
            else 0
        )
        for record, value, policy in (
            (champion_record, champion_auc, _CHAMPION_POLICY),
            (stage3_record, stage3_auc, _STAGE3_POLICY),
        ):
            metric_auc = _fraction(record.get("auc"), f"{policy} auc")
            if metric_auc != value:
                raise ValueError(f"{policy} AUC does not match its curve: {episode_id}")
        difference = tuple(a - b for a, b in zip(champion_curve, stage3_curve, strict=True))
        first_difference = next((index for index, value in enumerate(difference) if value), None)
        result.append(
            {
                "episode_id": episode_id,
                "order": int(left["order"]),
                "graph_seed": int(left["graph_seed"]),
                "policy_seed": int(left["policy_seed"]),
                "horizon": HORIZON,
                "champion_auc": float(champion_auc),
                "stage3_auc": float(stage3_auc),
                "delta": float(champion_auc - stage3_auc),
                "champion_auc_fraction": _fraction_text(champion_auc),
                "stage3_auc_fraction": _fraction_text(stage3_auc),
                "delta_fraction": _fraction_text(champion_auc - stage3_auc),
                "champion_best_total_witnesses": int(
                    champion_record.get("best_total_witnesses", 0)
                ),
                "stage3_best_total_witnesses": int(stage3_record.get("best_total_witnesses", 0)),
                "champion_accepted_count": int(champion_record.get("accepted_count", 0)),
                "stage3_accepted_count": int(stage3_record.get("accepted_count", 0)),
                "champion_duplicate_count": int(champion_record.get("duplicate_count", 0)),
                "stage3_duplicate_count": int(stage3_record.get("duplicate_count", 0)),
                "champion_first_improvement_step": champion_record.get("first_improvement_step"),
                "stage3_first_improvement_step": stage3_record.get("first_improvement_step"),
                "first_difference_step": first_difference,
                "curve_difference_area_fraction": _fraction_text(sum(difference, Fraction(0))),
                "curve_difference_steps": sum(value != 0 for value in difference),
                "initial_total_capped_witnesses": initial_witnesses,
                "witness_transition_summary": transitions,
                "canonical_episode_sha256": left.get("canonical_episode_sha256"),
                "champion_program_id": source_identity["champion"].get("program_id"),
                "champion_source_sha256": source_identity["champion"].get("source_sha256"),
                "champion_normalized_ast_sha256": source_identity["champion"].get(
                    "normalized_ast_sha256"
                ),
                "stage3_source_sha256": source_identity["stage3_champion"].get("source_sha256"),
                "stage3_normalized_ast_sha256": source_identity["stage3_champion"].get(
                    "normalized_ast_sha256"
                ),
            }
        )
    if len(result) != 128:
        raise ValueError("paired table must contain exactly 128 rows")
    return result


def _support(values: Sequence[Fraction]) -> list[dict[str, Any]]:
    counts = Counter(values)
    return [
        {"value": _fraction_text(value), "decimal": float(value), "count": counts[value]}
        for value in sorted(counts)
    ]


def _sign_mass(values: Sequence[Fraction]) -> dict[str, Any]:
    negative = sum(value < 0 for value in values)
    zero = sum(value == 0 for value in values)
    positive = sum(value > 0 for value in values)
    count = len(values)
    return {
        "negative": negative,
        "zero": zero,
        "positive": positive,
        "negative_fraction": negative / count,
        "zero_fraction": zero / count,
        "positive_fraction": positive / count,
        "count": count,
    }


def _quantization(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    champion = [_fraction(str(row["champion_auc_fraction"]), "champion AUC") for row in rows]
    stage3 = [_fraction(str(row["stage3_auc_fraction"]), "stage3 AUC") for row in rows]
    deltas = [_fraction(str(row["delta_fraction"]), "delta") for row in rows]
    curve_values: list[Fraction] = []
    for row in rows:
        # Difference area is sufficient for the diagnostic lattice; the global
        # normalized curve support is recorded separately below.
        curve_values.append(_fraction(str(row["curve_difference_area_fraction"]), "curve area"))
    area_step = _fraction_gcd(curve_values)
    auc_support = sorted(set(champion + stage3))
    auc_step = _fraction_gcd([b - a for a, b in zip(auc_support, auc_support[1:], strict=False)])
    exemplar = next((row for row in rows if row["delta_fraction"] == "1/32"), None)
    mechanism: dict[str, Any] = {
        "horizon": HORIZON,
        "area_step_fraction": _fraction_text(area_step),
        "auc_lattice_step_fraction": _fraction_text(auc_step),
        "explanation": (
            "The 0.03125 endpoint is 1/32: in a 32-step AUC, one full normalized "
            "witness unit of cumulative area separates the two trajectories. "
            "The exemplar records the witness-transition pattern and its exact area."
        ),
    }
    if exemplar is not None:
        mechanism["one_over_32_exemplar"] = {
            "episode_id": exemplar["episode_id"],
            "initial_total_capped_witnesses": exemplar["initial_total_capped_witnesses"],
            "curve_difference_area_fraction": exemplar["curve_difference_area_fraction"],
            "curve_difference_steps": exemplar["curve_difference_steps"],
            "first_difference_step": exemplar["first_difference_step"],
            "witness_transition_summary": exemplar["witness_transition_summary"],
        }
    return {
        "champion_auc": {
            "support": _support(champion),
            "lattice_step_fraction": _fraction_text(_fraction_gcd(champion)),
        },
        "stage3_auc": {
            "support": _support(stage3),
            "lattice_step_fraction": _fraction_text(_fraction_gcd(stage3)),
        },
        "delta": {
            "support": _support(deltas),
            "lattice_step_fraction": _fraction_text(_fraction_gcd(deltas)),
            "sign_mass": _sign_mass(deltas),
        },
        "mechanism_for_0_03125": mechanism,
    }


def _cluster_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clusters: dict[tuple[int, int], list[Fraction]] = defaultdict(list)
    for row in rows:
        clusters[(int(row["order"]), int(row["graph_seed"]))].append(
            _fraction(str(row["delta_fraction"]), "delta")
        )
    cluster_rows: list[dict[str, Any]] = []
    for key in sorted(clusters):
        values = clusters[key]
        if len(values) != 16:
            raise ValueError(f"cluster {key} does not contain 16 policy seeds")
        cluster_rows.append(
            {
                "order": key[0],
                "graph_seed": key[1],
                "count": len(values),
                "median_delta": float(_median(values)),
                "median_delta_fraction": _fraction_text(_median(values)),
                "mean_delta": float(_mean(values)),
                "mean_delta_fraction": _fraction_text(_mean(values)),
                "population_stddev": statistics.pstdev(float(value) for value in values),
                "sign_mass": _sign_mass(values),
                "first_difference_steps": {
                    ("none" if step is None else str(step)): count
                    for step, count in sorted(
                        Counter(
                            row["first_difference_step"]
                            for row in rows
                            if int(row["order"]) == key[0]
                            and int(row["graph_seed"]) == key[1]
                        ).items(),
                        key=lambda item: (
                            item[0] is not None,
                            item[0] if item[0] is not None else -1,
                        ),
                    )
                },
            }
        )
    values = [_fraction(str(row["delta_fraction"]), "delta") for row in rows]
    by_order: dict[str, Any] = {}
    for order in ORDERS:
        order_values = [
            value
            for row, value in zip(rows, values, strict=True)
            if row["order"] == order
        ]
        order_clusters = [item for item in cluster_rows if item["order"] == order]
        graph_means = [float(item["mean_delta"]) for item in order_clusters]
        within = [
            statistics.pvariance(
                float(value)
                for value in clusters[(order, int(item["graph_seed"]))]
            )
            for item in order_clusters
        ]
        by_order[str(order)] = {
            "count": len(order_values),
            "median_delta": float(_median(order_values)),
            "mean_delta": float(_mean(order_values)),
            "population_stddev": statistics.pstdev(float(value) for value in order_values),
            "graph_mean_between_variance": statistics.pvariance(graph_means),
            "mean_within_graph_variance": statistics.fmean(within),
            "graph_median_sign_mass": _sign_mass(
                [
                    _fraction(str(item["median_delta_fraction"]), "cluster median")
                    for item in order_clusters
                ]
            ),
        }
    pooled_means = [float(item["mean_delta"]) for item in cluster_rows]
    within_all = [
        statistics.pvariance(float(value) for value in values_for_cluster)
        for values_for_cluster in clusters.values()
    ]
    return {
        "clusters": cluster_rows,
        "by_order": by_order,
        "pooled": {
            "count": len(values),
            "median_delta": float(_median(values)),
            "mean_delta": float(_mean(values)),
            "population_stddev": statistics.pstdev(float(value) for value in values),
            "graph_mean_between_variance": statistics.pvariance(pooled_means),
            "mean_within_graph_variance": statistics.fmean(within_all),
            "sign_mass": _sign_mass(values),
            "graph_median_sign_mass": _sign_mass(
                [
                    _fraction(str(item["median_delta_fraction"]), "cluster median")
                    for item in cluster_rows
                ]
            ),
        },
    }


def _rng(seed: int, *parts: object) -> random.Random:
    material = ":".join(str(part) for part in (seed, *parts)).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _bootstrap_draws(
    rows: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> dict[str, list[Fraction]]:
    grouped: dict[int, dict[int, dict[int, Fraction]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[int(row["order"])][int(row["graph_seed"])][int(row["policy_seed"])] = _fraction(
            str(row["delta_fraction"]), "delta"
        )
    result: dict[str, list[Fraction]] = {"10": [], "12": [], "pooled": []}
    for index in range(samples):
        by_order: dict[int, list[Fraction]] = {}
        for order in ORDERS:
            generator = _rng(seed, index, order)
            graph_ids = sorted(grouped[order])
            values: list[Fraction] = []
            for _ in graph_ids:
                graph = graph_ids[generator.randrange(len(graph_ids))]
                policy_ids = sorted(grouped[order][graph])
                for _ in policy_ids:
                    policy = policy_ids[generator.randrange(len(policy_ids))]
                    values.append(grouped[order][graph][policy])
            by_order[order] = values
            result[str(order)].append(_median(values))
        result["pooled"].append(_median(by_order[10] + by_order[12]))
    return result


def _bootstrap_entry(observed: Sequence[Fraction], draws: Sequence[Fraction]) -> dict[str, Any]:
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    low, low_i, low_j, low_position = _percentile(draws, alpha)
    high, high_i, high_j, high_position = _percentile(draws, 1.0 - alpha)
    mean = _mean(list(draws))
    values = [float(value) for value in draws]
    stddev = statistics.pstdev(values)
    skew = 0.0
    if stddev:
        skew = statistics.fmean(((value - float(mean)) / stddev) ** 3 for value in values)
    return {
        "samples": len(draws),
        "seed": BOOTSTRAP_SEED,
        "confidence_level": BOOTSTRAP_CONFIDENCE,
        "observed_median": float(_median(observed)),
        "observed_median_fraction": _fraction_text(_median(observed)),
        "interval": [float(low), float(high)],
        "interval_fraction": [_fraction_text(low), _fraction_text(high)],
        "support": _support(list(draws)),
        "sign_mass": _sign_mass(list(draws)),
        "mean": float(mean),
        "population_stddev": stddev,
        "skewness": skew,
        "percentile_indices": {
            "lower": {
                "position": low_position,
                "low_index": low_i,
                "high_index": low_j,
                "low_value": _fraction_text(sorted(draws)[low_i]),
                "high_value": _fraction_text(sorted(draws)[low_j]),
            },
            "upper": {
                "position": high_position,
                "low_index": high_i,
                "high_index": high_j,
                "low_value": _fraction_text(sorted(draws)[high_i]),
                "high_value": _fraction_text(sorted(draws)[high_j]),
            },
        },
    }


def _bootstrap_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [_fraction(str(row["delta_fraction"]), "delta") for row in rows]
    by_order = {
        order: [
            value
            for row, value in zip(rows, deltas, strict=True)
            if int(row["order"]) == order
        ]
        for order in ORDERS
    }
    draws = _bootstrap_draws(rows, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    result = {
        "method": "frozen hierarchical graph-seed replacement then policy-seed replacement",
        "policy": _CHAMPION_POLICY,
        "baseline": _STAGE3_POLICY,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": BOOTSTRAP_CONFIDENCE,
        "by_order": {
            str(order): _bootstrap_entry(by_order[order], draws[str(order)]) for order in ORDERS
        },
        "pooled": _bootstrap_entry(deltas, draws["pooled"]),
    }
    expected = [0.0, 0.03125]
    pooled = cast(Mapping[str, Any], result["pooled"])
    interval = cast(list[float], pooled["interval"])
    if interval != expected:
        raise RuntimeError(f"frozen pooled bootstrap interval drifted: {interval}")
    return cast(dict[str, Any], result)


def _generic_bootstrap(
    values: Sequence[float], *, statistic: str, samples: int, seed: int
) -> dict[str, Any]:
    if not values:
        raise ValueError("sensitivity bootstrap requires values")
    draws: list[float] = []
    for index in range(samples):
        generator = _rng(seed, statistic, index)
        sample = [values[generator.randrange(len(values))] for _ in values]
        draws.append(statistics.mean(sample) if statistic == "mean" else statistics.median(sample))
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    ordered = sorted(draws)
    low = _float_percentile(ordered, alpha)
    high = _float_percentile(ordered, 1.0 - alpha)
    return {
        "statistic": statistic,
        "samples": samples,
        "seed": seed,
        "estimate": statistics.mean(values) if statistic == "mean" else statistics.median(values),
        "interval": [low, high],
    }


def _float_percentile(values: Sequence[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (position - low))


def _randomization(values: Sequence[float], *, samples: int, seed: int) -> dict[str, Any]:
    observed = statistics.median(values)
    null_medians: list[float] = []
    for index in range(samples):
        generator = _rng(seed, index)
        null_medians.append(
            statistics.median(value if generator.randrange(2) else -value for value in values)
        )
    p_value = (1 + sum(abs(value) >= abs(observed) for value in null_medians)) / (samples + 1)
    return {
        "estimand": "paired sign-randomization null for the episode median",
        "samples": samples,
        "seed": seed,
        "observed_median": observed,
        "two_sided_p_value": p_value,
        "null_interval": [
            _float_percentile(sorted(null_medians), 0.025),
            _float_percentile(sorted(null_medians), 0.975),
        ],
    }


def _sensitivity(rows: Sequence[Mapping[str, Any]], clusters: Mapping[str, Any]) -> dict[str, Any]:
    values = [float(row["delta"]) for row in rows]
    cluster_values = [
        float(item["median_delta"])
        for item in cast(Sequence[Mapping[str, Any]], clusters["clusters"])
    ]
    larger = _generic_bootstrap(values, statistic="median", samples=20_000, seed=2026080102)
    return {
        "estimands": {
            "episode_mean": "mean paired AUC delta over 128 episodes",
            "episode_median": "median paired AUC delta over 128 episodes",
            "cluster_median": "median of the eight order/graph cluster medians",
            "sign": "directional sign mass; zero is retained as a third outcome",
        },
        "episode": {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sign_mass": _sign_mass([_fraction(str(value), "delta") for value in values]),
            "mean_bootstrap": _generic_bootstrap(
                values, statistic="mean", samples=BOOTSTRAP_SAMPLES, seed=2026080103
            ),
            "median_bootstrap": _generic_bootstrap(
                values, statistic="median", samples=BOOTSTRAP_SAMPLES, seed=2026080104
            ),
            "larger_draw_median_bootstrap": larger,
        },
        "clusters": {
            "count": len(cluster_values),
            "mean": statistics.mean(cluster_values),
            "median": statistics.median(cluster_values),
            "sign_mass": _sign_mass(
                [_fraction(str(value), "cluster delta") for value in cluster_values]
            ),
            "median_bootstrap": _generic_bootstrap(
                cluster_values, statistic="median", samples=BOOTSTRAP_SAMPLES, seed=2026080105
            ),
        },
        "paired_randomization": _randomization(values, samples=BOOTSTRAP_SAMPLES, seed=2026080106),
        "graph_cluster_randomization": _randomization(
            cluster_values, samples=BOOTSTRAP_SAMPLES, seed=2026080107
        ),
        "interpretation": (
            "Episode resampling treats trajectories as exchangeable; cluster resampling "
            "treats the eight order/graph cells as the independent units. The latter is "
            "the conservative sensitivity because graph heterogeneity is the scientific concern."
        ),
    }


def _power_study(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clusters: list[list[float]] = []
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["order"]), int(row["graph_seed"]))].append(float(row["delta"]))
    clusters = [grouped[key] for key in sorted(grouped)]
    if len(clusters) != 8 or any(len(values) != 16 for values in clusters):
        raise ValueError("power study requires eight observed 16-policy clusters")
    scenarios = {
        "observed_effect": {"scale": 1.0, "null_sign_flip": False},
        "conservative_effect": {"scale": 0.5, "null_sign_flip": False},
        "zero_effect": {"scale": 1.0, "null_sign_flip": True},
    }
    cells: list[dict[str, Any]] = []
    for graph_count in POWER_GRAPH_COUNTS:
        for policy_count in POWER_POLICY_COUNTS:
            for scenario, spec in scenarios.items():
                hits = 0
                effects: list[float] = []
                generator = _rng(POWER_SEED, graph_count, policy_count, scenario)
                for _ in range(POWER_REPLICATES):
                    observations: list[float] = []
                    for _graph in range(graph_count):
                        source = clusters[generator.randrange(len(clusters))]
                        for _policy in range(policy_count):
                            value = source[generator.randrange(len(source))] * float(spec["scale"])
                            if bool(spec["null_sign_flip"]) and generator.randrange(2) == 0:
                                value = -value
                            observations.append(value)
                    estimate = statistics.median(observations)
                    effects.append(estimate)
                    hits += int(estimate > 0.0)
                rate = hits / POWER_REPLICATES
                standard_error = math.sqrt(rate * (1.0 - rate) / POWER_REPLICATES)
                cells.append(
                    {
                        "graph_seed_count_per_order": graph_count,
                        "policy_seed_count_per_graph": policy_count,
                        "scenario": scenario,
                        "replicates": POWER_REPLICATES,
                        "directional_positive_rate": rate,
                        "false_positive_rate": rate if scenario == "zero_effect" else None,
                        "monte_carlo_standard_error": standard_error,
                        "monte_carlo_interval_95": [
                            max(0.0, rate - 1.96 * standard_error),
                            min(1.0, rate + 1.96 * standard_error),
                        ],
                        "median_simulated_effect": statistics.median(effects),
                        "mean_simulated_effect": statistics.mean(effects),
                    }
                )
    return {
        "method": (
            "Synthetic retained-distribution study: sample observed order/graph clusters "
            "with replacement, then policy deltas within each sampled cluster. Observed "
            "keeps the retained deltas, conservative scales them by one half, and zero "
            "randomizes signs while preserving magnitudes. A positive pooled median is "
            "the directional detection rule; this is not a new scientific gate."
        ),
        "seed": POWER_SEED,
        "replicates": POWER_REPLICATES,
        "orders": list(ORDERS),
        "graph_seed_counts_per_order": list(POWER_GRAPH_COUNTS),
        "policy_seed_counts_per_graph": list(POWER_POLICY_COUNTS),
        "cells": cells,
        "priority": (
            "Independent graph seeds are the priority: the retained four graph cells "
            "cannot reveal unseen graph heterogeneity. More policy seeds reduce within-graph "
            "noise but do not substitute for new graph seeds."
        ),
    }


def _write_table(output: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    jsonl = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    columns = [
        "episode_id",
        "order",
        "graph_seed",
        "policy_seed",
        "horizon",
        "champion_auc",
        "stage3_auc",
        "delta",
        "champion_auc_fraction",
        "stage3_auc_fraction",
        "delta_fraction",
        "champion_best_total_witnesses",
        "stage3_best_total_witnesses",
        "champion_accepted_count",
        "stage3_accepted_count",
        "champion_duplicate_count",
        "stage3_duplicate_count",
        "champion_first_improvement_step",
        "stage3_first_improvement_step",
        "first_difference_step",
        "curve_difference_area_fraction",
        "curve_difference_steps",
        "initial_total_capped_witnesses",
        "witness_transition_summary",
        "canonical_episode_sha256",
        "champion_program_id",
        "champion_source_sha256",
        "champion_normalized_ast_sha256",
        "stage3_source_sha256",
        "stage3_normalized_ast_sha256",
    ]
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False
    ) as text_handle:
        temporary_name = text_handle.name
        writer = csv.DictWriter(text_handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values: dict[str, Any] = {column: row.get(column) for column in columns}
            values["witness_transition_summary"] = _canonical_bytes(
                row.get("witness_transition_summary", [])
            ).decode("utf-8")
            writer.writerow(values)
        text_handle.flush()
        os.fsync(text_handle.fileno())
    try:
        csv_bytes = Path(temporary_name).read_bytes()
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    _atomic_bytes(output / PAIR_JSONL_NAME, jsonl)
    _atomic_bytes(output / PAIR_CSV_NAME, csv_bytes)
    hashes = {
        "rows": len(rows),
        "jsonl_sha256": _sha_bytes(jsonl),
        "csv_sha256": _sha_bytes(csv_bytes),
    }
    _atomic_json(output / PAIR_HASH_NAME, hashes)
    return {
        **hashes,
        "jsonl": PAIR_JSONL_NAME,
        "csv": PAIR_CSV_NAME,
        "hash": PAIR_HASH_NAME,
    }


def diagnose_deltas(
    run: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete retained-only Stage 4R delta diagnostic."""

    run_root = Path(run).resolve()
    validation_summary_path = run_root / "validation-summary.json"
    if not validation_summary_path.is_file():
        raise FileNotFoundError(f"retained validation summary not found: {validation_summary_path}")
    validation_summary = _read_json(validation_summary_path)
    if validation_summary.get("schema_version") != "stage4r.validation.result.v1":
        raise ValueError("unexpected Stage 4R validation summary schema")
    if validation_summary.get("decision") != "NO_GO":
        raise ValueError("historical Stage 4R NO_GO decision is required and immutable")
    if validation_summary.get("final_validation_results_observed") is not True:
        raise ValueError("final-validation result is not marked observed")
    root = _validation_root(run_root)
    primary_path, primary_summary = _one_summary(root, "primary")
    replay_path, replay_summary = _one_summary(root, "replay")
    replay_check = verify_replay(primary_summary, replay_summary)
    if replay_check.get("exact") is not True:
        raise ValueError("primary/replay summary replay contract is not exact")
    primary_rows = _load_rows(root, primary_summary, "primary")
    replay_rows = _load_rows(root, replay_summary, "replay")
    repo_root = _find_repo_root(run_root)
    manifest = _validate_manifest(repo_root, run_root, primary_summary)
    source_identity = _source_identity(repo_root, run_root, validation_summary)
    rows = _paired_rows(primary_rows, replay_rows, source_identity)
    clusters = _cluster_summary(rows)
    quantization = _quantization(rows)
    bootstrap = _bootstrap_summary(rows)
    sensitivity = _sensitivity(rows, clusters)
    power = _power_study(rows)
    destination = Path(output_dir).resolve() if output_dir is not None else run_root / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    table = _write_table(destination, rows)
    _atomic_json(destination / BOOTSTRAP_NAME, bootstrap)
    _atomic_json(destination / CLUSTER_NAME, clusters)
    _atomic_json(destination / POWER_NAME, power)
    artifact_names = {
        PAIR_JSONL_NAME,
        PAIR_CSV_NAME,
        PAIR_HASH_NAME,
        BOOTSTRAP_NAME,
        CLUSTER_NAME,
        POWER_NAME,
    }
    artifact_bytes = sum(
        path.stat().st_size
        for path in destination.iterdir()
        if path.is_file() and path.name in artifact_names
    )
    if artifact_bytes >= 16 * 1024 * 1024:
        raise RuntimeError("Stage 4R diagnostic artifacts exceed the 16 MiB bound")
    artifact_manifest = {
        path.name: {"sha256": _sha_file(path), "bytes": path.stat().st_size}
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name in artifact_names
    }
    stage2c = {
        "report": "docs/reports/STAGE2C_DIAGNOSTIC_REPORT.md",
        "diagnosis": "BENCHMARK_SATURATION",
        "historical_control": {
            "random_median_auc": 0.5,
            "structural_median_auc": 0.5,
            "relative_median_improvement": 0.0,
            "paired_bootstrap_interval": [0.0, 0.03125],
        },
        "comparison": (
            "Stage 2C found coarse order-8 score levels and a saturated 0.5 median "
            "despite 93.75% policy disagreement and pool headroom. Stage 4R separates "
            "the point medians, but the episode lattice, 30 exact ties, 32 negative "
            "deltas, and only four graph cells leave the primary lower bound at zero."
        ),
    }
    recommendation = {
        "token": "REDESIGN_PRIMARY_METRIC_BEFORE_CONFIRMATION",
        "reason": (
            "The positive point estimate is not stable at the episode level and the "
            "frozen graph-cluster bootstrap has a zero lower bound. Redesign the primary "
            "estimand before spending new graph-seed budget."
        ),
        "proposed_confirmatory_design": {
            "frozen_policies": ["stage4r_champion", "stage3-candidate-slot-04"],
            "unseen_graph_seeds_per_order": 16,
            "orders": [10, 12, 16],
            "policy_seeds_per_graph": 32,
            "horizon": HORIZON,
            "primary_estimand": (
                "paired transition-aware area delta, with episode and graph-cluster "
                "summaries reported separately from the existing normalized AUC"
            ),
            "inference": (
                "10,000 graph-cluster bootstrap draws with graph replacement then "
                "paired policy replacement"
            ),
            "threshold": (
                "pre-register positive 95% lower bound and at least 2% relative improvement"
            ),
            "primary_replay_requirements": (
                "two exact timing-stripped passes with identical rows, hashes, and reductions"
            ),
            "equal_cpu_budget": (
                "128 evaluations per episode, identical worker limits and affinity accounting"
            ),
            "artifact_budget": "diagnostic artifacts below 16 MiB",
            "model_calls": 0,
        },
    }
    result: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "completed",
        "command": "stage4r diagnose-deltas",
        "run": str(run_root),
        "diagnostics_dir": str(destination),
        "retained_only": True,
        "decision": "NO_GO",
        "historical_decision": "NO_GO",
        "manifest": manifest,
        "primary_replay": {
            "exact": True,
            "primary_summary": primary_path.name,
            "replay_summary": replay_path.name,
            "primary_summary_sha256": _sha_file(primary_path),
            "replay_summary_sha256": _sha_file(replay_path),
            **replay_check,
            "paired_rows": len(rows),
        },
        "source_identity": source_identity,
        "artifacts": artifact_manifest,
        "paired_table": table,
        "quantization": quantization,
        "clusters": clusters,
        "bootstrap": bootstrap,
        "sensitivity": sensitivity,
        "stage2c_comparison": stage2c,
        "power_study": {
            "artifact": POWER_NAME,
            "summary": power,
        },
        "recommendation": recommendation,
        "health": {
            "provider_calls": 0,
            "model_calls": 0,
            "new_graph_evaluations": 0,
            "stage5_started": False,
            "historical_decision_changed": False,
        },
    }
    _atomic_json(destination / SUMMARY_NAME, result)
    return result


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "diagnose_deltas",
]
