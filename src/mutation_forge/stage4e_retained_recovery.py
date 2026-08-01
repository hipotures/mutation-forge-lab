"""Recover a frozen Stage 4E result from retained primary/replay artifacts.

This module is deliberately an artifact reducer, not an evaluator.  It reads
two completed passes, removes the one explicitly timing-only field allowed by
the Stage 4E result contract (``timing_ns``), and then performs all identity
and metric work from the preserved files.  There is no import of the Stage 3
evaluator or the Stage 4E execution module here.
"""
# ruff: noqa: E501

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.stage3.manifest import canonical_bytes, sha256

from .stage4e_config import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CHAMPION_AST_SHA256,
    CHAMPION_ID,
    CHAMPION_SOURCE_SHA256,
    COMPARATOR_AST_SHA256,
    COMPARATOR_ID,
    COMPARATOR_SOURCE_SHA256,
    CONFIDENCE_LEVEL,
    HEG_COMMIT,
    RELATIVE_THRESHOLD,
    START_COMMIT,
)
from .stage4e_metrics import (
    BootstrapSummary,
    PairedAreaEpisode,
    PairedAreaSummary,
    bootstrap_paired_theta,
    fraction_text,
    summarize_paired_areas,
    terminal_gate_checks,
)

RECOVERY_SCHEMA = "stage4e.retained_recovery.v1"
EXECUTION_SCHEMA = "stage4e.execution.v1"
TIMING_ONLY_FIELDS = frozenset({"timing_ns"})
EXPECTED_MANIFEST_SHA256 = "d80164cc4e0f26e2a2999adb7b1f8ff4b40a194e6f2576962190bd7b7bd22a34"
EXPECTED_CANONICAL_REDUCTION_SHA256 = (
    "c10e135df06963014be00a5cb262dce1260906b26ba5d84d8a9d79c282121282"
)
EXPECTED_METRICS_INPUT_SHA256 = (
    "92247c893a6e347925dec06cefec7c8d17b898b3ab7a23322e20c26e8f302bdd"
)
NON_TIMING_MISMATCH = "INCONCLUSIVE_NON_TIMING_REPLAY_MISMATCH"
RETAINED_EVIDENCE_FAILURE = "INCONCLUSIVE_RETAINED_EVIDENCE_FAILURE"
POLICY_SOURCE_SHA256 = {
    CHAMPION_ID: CHAMPION_SOURCE_SHA256,
    COMPARATOR_ID: COMPARATOR_SOURCE_SHA256,
}
POLICY_AST_SHA256 = {
    CHAMPION_ID: CHAMPION_AST_SHA256,
    COMPARATOR_ID: COMPARATOR_AST_SHA256,
}
FORBIDDEN_COUNTERS = (
    "model_calls",
    "app_server_calls",
    "oracle_score_calls",
    "runtime_network_calls",
)


class RecoveryError(ValueError):
    """A retained artifact cannot satisfy the frozen evidence contract."""


def timing_stripped_projection(value: Any) -> Any:
    """Recursively remove only the frozen timing-only ``timing_ns`` field.

    No suffix matching or heuristic field filtering is used.  In particular,
    scientific fields that happen to differ are retained and therefore fail
    canonical comparison.
    """

    if isinstance(value, Mapping):
        return {
            str(key): timing_stripped_projection(item)
            for key, item in value.items()
            if str(key) not in TIMING_ONLY_FIELDS
        }
    if isinstance(value, list | tuple):
        return [timing_stripped_projection(item) for item in value]
    return value


def _diff_paths(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[dict[str, Any]] = []
        keys = sorted({str(key) for key in left} | {str(key) for key in right})
        left_by_key = {str(key): item for key, item in left.items()}
        right_by_key = {str(key): item for key, item in right.items()}
        for key in keys:
            child = f"{path}.{key}"
            if key not in left_by_key:
                differences.append({"path": child, "primary": None, "replay": right_by_key[key]})
            elif key not in right_by_key:
                differences.append({"path": child, "primary": left_by_key[key], "replay": None})
            else:
                differences.extend(_diff_paths(left_by_key[key], right_by_key[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(
                {"path": f"{path}.length", "primary": len(left), "replay": len(right)}
            )
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            differences.extend(_diff_paths(left_item, right_item, f"{path}[{index}]"))
        return differences
    if left != right:
        return [{"path": path, "primary": left, "replay": right}]
    return []


def compare_canonical_rows(
    primary: Mapping[str, Mapping[str, Any]], replay: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare every paired row after the explicit timing-only projection."""

    primary_ids = set(primary)
    replay_ids = set(replay)
    missing_primary = sorted(replay_ids - primary_ids)
    missing_replay = sorted(primary_ids - replay_ids)
    differences: list[dict[str, Any]] = []
    row_hashes: list[dict[str, str]] = []
    for episode_id in sorted(primary_ids & replay_ids):
        primary_projection = timing_stripped_projection(primary[episode_id])
        replay_projection = timing_stripped_projection(replay[episode_id])
        row_hash = hashlib.sha256(canonical_bytes(primary_projection)).hexdigest()
        row_hashes.append({"episode_id": episode_id, "sha256": row_hash})
        row_differences = _diff_paths(primary_projection, replay_projection)
        if row_differences:
            differences.append(
                {"episode_id": episode_id, "fields": row_differences}
            )
    row_hashes_hash = sha256(row_hashes)
    return {
        "projection": {
            "name": "recursive_timing_stripped",
            "excluded_fields": sorted(TIMING_ONLY_FIELDS),
        },
        "primary_episode_count": len(primary_ids),
        "replay_episode_count": len(replay_ids),
        "missing_primary_episode_ids": missing_primary,
        "missing_replay_episode_ids": missing_replay,
        "non_timing_differences": differences,
        "canonical_row_hashes_sha256": row_hashes_hash,
        "rows_exact": not missing_primary and not missing_replay and not differences,
    }


@dataclass(frozen=True, slots=True)
class _ShardData:
    index: int
    path: Path
    episode_ids: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class _PassData:
    pass_name: str
    summary_path: Path
    summary: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    shards: tuple[_ShardData, ...]
    canonical_shards_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def _hash_without(value: Mapping[str, Any], excluded: str) -> str:
    return sha256({key: item for key, item in value.items() if key != excluded})


def _load_frozen_context(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = project_root / "configs/manifests/stage4e-confirmation-v1.json"
    freeze_path = project_root / "configs/stage4e-confirmation-freeze-v1.json"
    manifest = _read_json(manifest_path)
    freeze = _read_json(freeze_path)
    if manifest.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RecoveryError("frozen manifest SHA-256 does not match the preregistered value")
    if _hash_without(manifest, "manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RecoveryError("frozen manifest content hash is invalid")
    if freeze.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RecoveryError("freeze does not bind the expected manifest")
    freeze_hash = freeze.get("freeze_sha256")
    if not isinstance(freeze_hash, str) or _hash_without(freeze, "freeze_sha256") != freeze_hash:
        raise RecoveryError("frozen preregistration hash is invalid")
    if freeze.get("stage4e_results_observed") is not False:
        raise RecoveryError("frozen preregistration was marked observed")
    if freeze.get("start_commit") != START_COMMIT or freeze.get("heg_commit") != HEG_COMMIT:
        raise RecoveryError("frozen repository pins differ")
    if freeze.get("policy_ids") != [CHAMPION_ID, COMPARATOR_ID]:
        raise RecoveryError("frozen policy IDs differ")
    if freeze.get("policy_source_sha256") != POLICY_SOURCE_SHA256:
        raise RecoveryError("frozen policy source identities differ")
    if freeze.get("policy_ast_sha256") != POLICY_AST_SHA256:
        raise RecoveryError("frozen policy AST identities differ")
    bootstrap = freeze.get("bootstrap")
    if bootstrap != {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "percentile_rule": "linear_interpolation_at_p_times_n_minus_1",
    }:
        raise RecoveryError("frozen bootstrap differs")
    gate = freeze.get("gate")
    if gate != {
        "relative_improvement_at_least": RELATIVE_THRESHOLD,
        "bootstrap_lower_bound_strictly_positive": True,
        "order_effects_nonnegative": True,
    }:
        raise RecoveryError("frozen gate differs")
    episodes = manifest.get("episodes")
    shards = manifest.get("shards")
    if not isinstance(episodes, list) or not isinstance(shards, list):
        raise RecoveryError("frozen manifest has no episode or shard roster")
    if (
        manifest.get("episode_count") != 1536
        or manifest.get("shard_count") != 24
        or manifest.get("episodes_per_shard") != 64
        or len(episodes) != 1536
        or len(shards) != 24
    ):
        raise RecoveryError("frozen manifest dimensions differ")
    expected: dict[str, dict[str, Any]] = {}
    for item in episodes:
        if not isinstance(item, Mapping) or not isinstance(item.get("episode_id"), str):
            raise RecoveryError("frozen manifest episode row is invalid")
        episode_id = str(item["episode_id"])
        if episode_id in expected:
            raise RecoveryError(f"duplicate frozen episode identity: {episode_id}")
        expected[episode_id] = cast(dict[str, Any], dict(item))
    if len(expected) != 1536:
        raise RecoveryError("frozen manifest episode identities are not unique")
    return manifest, freeze, expected


def _summary_path(pass_root: Path, pass_name: str) -> Path:
    paths = sorted(pass_root.glob(f"*-{pass_name}-summary.json"))
    if len(paths) != 1:
        raise RecoveryError(
            f"expected exactly one {pass_name} summary in {pass_root}, found {len(paths)}"
        )
    return paths[0]


def _validate_row(
    row: dict[str, Any], expected: Mapping[str, Any], freeze: Mapping[str, Any]
) -> None:
    episode_id = str(row.get("episode_id", ""))
    if episode_id != str(expected.get("episode_id")):
        raise RecoveryError("episode ID does not match frozen roster")
    for key in ("order", "graph_seed", "policy_seed", "horizon"):
        if row.get(key) != expected.get(key):
            raise RecoveryError(f"{episode_id}: frozen field {key} differs")
    if row.get("terminal_status") != "completed":
        raise RecoveryError(f"{episode_id}: terminal status is not completed")
    if row.get("policy_source_sha256") != freeze.get("policy_source_sha256"):
        raise RecoveryError(f"{episode_id}: policy source identity differs")
    policies = row.get("policies")
    if not isinstance(policies, Mapping) or set(str(key) for key in policies) != {
        CHAMPION_ID,
        COMPARATOR_ID,
    }:
        raise RecoveryError(f"{episode_id}: policy roster differs")
    for counter in FORBIDDEN_COUNTERS:
        if int(row.get(counter, 0)) != 0:
            raise RecoveryError(f"{episode_id}: forbidden counter {counter} is nonzero")
    if int(row.get("invalid_graphs", 0)) != 0:
        raise RecoveryError(f"{episode_id}: invalid graph count is nonzero")
    if int(row.get("policy_failures", 0)) != 0:
        raise RecoveryError(f"{episode_id}: policy failure count is nonzero")
    if not isinstance(row.get("canonical_episode_sha256"), str) or len(row["canonical_episode_sha256"]) != 64:
        raise RecoveryError(f"{episode_id}: canonical episode hash is missing")
    if not isinstance(row.get("metrics_input"), Mapping):
        raise RecoveryError(f"{episode_id}: metrics input is missing")


def _canonical_shard_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(
        canonical_bytes(timing_stripped_projection(row)) + b"\n" for row in rows
    )
    return hashlib.sha256(payload).hexdigest()


def _read_pass(
    run_root: Path,
    pass_name: str,
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
) -> _PassData:
    pass_root = run_root / pass_name
    if not pass_root.is_dir():
        raise RecoveryError(f"retained {pass_name} directory is missing: {pass_root}")
    summary_path = _summary_path(pass_root, pass_name)
    summary = _read_json(summary_path)
    if summary.get("schema_version") != EXECUTION_SCHEMA or summary.get("status") != "completed":
        raise RecoveryError(f"retained {pass_name} summary is not a completed Stage 4E execution")
    if summary.get("pass") != pass_name:
        raise RecoveryError(f"retained summary pass label differs: {summary_path}")
    if summary.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RecoveryError(f"retained {pass_name} summary manifest hash differs")
    if summary.get("record_count") != 1536 or summary.get("shard_count") != 24 or summary.get("episodes_per_shard") != 64:
        raise RecoveryError(f"retained {pass_name} summary dimensions differ")
    if (
        set(cast(list[Any], summary.get("policy_ids", [])))
        != {CHAMPION_ID, COMPARATOR_ID}
        or summary.get("policy_source_sha256") != POLICY_SOURCE_SHA256
    ):
        raise RecoveryError(f"retained {pass_name} policy identity differs")
    if summary.get("canonical_reduction_sha256") != EXPECTED_CANONICAL_REDUCTION_SHA256:
        raise RecoveryError(f"retained {pass_name} canonical reduction hash differs")
    if summary.get("metrics_input_sha256") != EXPECTED_METRICS_INPUT_SHA256:
        raise RecoveryError(f"retained {pass_name} metrics-input hash differs")
    counts = summary.get("counts")
    if not isinstance(counts, Mapping) or counts.get("episodes") != 1536 or any(
        int(counts.get(counter, -1)) != 0 for counter in FORBIDDEN_COUNTERS
    ):
        raise RecoveryError(f"retained {pass_name} summary call counters are not zero")
    manifest_ids = [str(item["episode_id"]) for item in cast(list[Mapping[str, Any]], manifest["episodes"])]
    if summary.get("manifest_episode_ids") != manifest_ids:
        raise RecoveryError(f"retained {pass_name} manifest episode roster differs")
    expected_shards = cast(list[Mapping[str, Any]], manifest["shards"])
    if summary.get("manifest_shards") != [
        [str(item) for item in cast(list[Any], shard["episode_ids"])] for shard in expected_shards
    ]:
        raise RecoveryError(f"retained {pass_name} manifest shard roster differs")
    entries = summary.get("shards")
    if not isinstance(entries, list) or len(entries) != 24:
        raise RecoveryError(f"retained {pass_name} shard list is incomplete")
    rows: dict[str, dict[str, Any]] = {}
    shard_data: list[_ShardData] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise RecoveryError(f"retained {pass_name} shard entry {index} is invalid")
        entry = cast(Mapping[str, Any], raw_entry)
        path_text = entry.get("path")
        if not isinstance(path_text, str):
            raise RecoveryError(f"retained {pass_name} shard {index} has no path")
        relative = Path(path_text)
        if relative.is_absolute() or relative.name != path_text:
            raise RecoveryError(f"retained {pass_name} shard path escapes its directory")
        shard_path = pass_root / relative
        expected_ids = tuple(str(item) for item in cast(list[Any], expected_shards[index]["episode_ids"]))
        entry_ids = tuple(str(item) for item in cast(list[Any], entry.get("episode_ids", [])))
        if entry_ids != expected_ids or entry.get("record_count") != 64:
            raise RecoveryError(f"retained {pass_name} shard {index} roster differs")
        if not shard_path.is_file():
            raise RecoveryError(f"retained {pass_name} shard is missing: {shard_path}")
        raw_bytes = shard_path.read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != entry.get("file_sha256"):
            raise RecoveryError(f"retained {pass_name} shard bytes fail their recorded hash")
        shard_rows: list[dict[str, Any]] = []
        try:
            with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise RecoveryError(f"retained {pass_name} shard row is not an object")
                        shard_rows.append(cast(dict[str, Any], item))
        except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecoveryError(f"retained {pass_name} shard is corrupt: {shard_path}: {error}") from error
        if [str(row.get("episode_id", "")) for row in shard_rows] != list(expected_ids):
            raise RecoveryError(f"retained {pass_name} shard {index} row roster differs")
        for row in shard_rows:
            episode_id = str(row["episode_id"])
            if episode_id in rows:
                raise RecoveryError(f"retained {pass_name} has duplicate episode {episode_id}")
            if episode_id not in expected:
                raise RecoveryError(f"retained {pass_name} has an unpaired episode {episode_id}")
            _validate_row(row, expected[episode_id], freeze)
            rows[episode_id] = row
        shard_data.append(
            _ShardData(
                index=index,
                path=shard_path,
                episode_ids=expected_ids,
                rows=tuple(shard_rows),
                canonical_sha256=_canonical_shard_hash(shard_rows),
            )
        )
    if len(rows) != 1536 or set(rows) != set(expected):
        raise RecoveryError(f"retained {pass_name} does not contain exactly 1,536 paired episodes")
    ordered_ids = sorted(rows)
    if sha256([rows[episode_id]["canonical_episode_sha256"] for episode_id in ordered_ids]) != EXPECTED_CANONICAL_REDUCTION_SHA256:
        raise RecoveryError(f"retained {pass_name} canonical reduction recomputation differs")
    if sha256([rows[episode_id]["metrics_input"] for episode_id in ordered_ids]) != EXPECTED_METRICS_INPUT_SHA256:
        raise RecoveryError(f"retained {pass_name} metrics-input recomputation differs")
    canonical_shards = [
        {"index": shard.index, "episode_ids": list(shard.episode_ids), "sha256": shard.canonical_sha256}
        for shard in shard_data
    ]
    return _PassData(
        pass_name=pass_name,
        summary_path=summary_path,
        summary=summary,
        rows=rows,
        shards=tuple(shard_data),
        canonical_shards_sha256=sha256(canonical_shards),
    )


def _compare_shards(primary: _PassData, replay: _PassData) -> dict[str, Any]:
    shard_differences: list[dict[str, Any]] = []
    primary_hashes: list[dict[str, Any]] = []
    replay_hashes: list[dict[str, Any]] = []
    for left, right in zip(primary.shards, replay.shards, strict=True):
        primary_hashes.append({"index": left.index, "sha256": left.canonical_sha256})
        replay_hashes.append({"index": right.index, "sha256": right.canonical_sha256})
        if left.episode_ids != right.episode_ids or left.canonical_sha256 != right.canonical_sha256:
            shard_differences.append(
                {
                    "index": left.index,
                    "episode_ids_equal": left.episode_ids == right.episode_ids,
                    "primary_sha256": left.canonical_sha256,
                    "replay_sha256": right.canonical_sha256,
                }
            )
    return {
        "primary": primary_hashes,
        "replay": replay_hashes,
        "primary_aggregate_sha256": primary.canonical_shards_sha256,
        "replay_aggregate_sha256": replay.canonical_shards_sha256,
        "differences": shard_differences,
        "exact": not shard_differences and primary.canonical_shards_sha256 == replay.canonical_shards_sha256,
    }


def _fraction_payload(value: Any) -> dict[str, float | str]:
    return {"value": float(value), "fraction": fraction_text(value)}


def _paired_payload(summary: PairedAreaSummary) -> dict[str, Any]:
    return {
        "estimand": "transition-aware paired area; graph mean, equal-order mean",
        "episode_count": len(summary.episodes),
        "graph_count": len(summary.graphs),
        "order_count": len(summary.orders),
        "theta": _fraction_payload(summary.theta),
        "mu_B": _fraction_payload(summary.mu_B),
        "relative_improvement": _fraction_payload(summary.relative_improvement),
        "episodes": [
            {
                "order": item.order,
                "graph_seed": item.graph_seed,
                "episode_id": item.episode_id,
                "candidate_area": _fraction_payload(item.candidate_area),
                "comparator_area": _fraction_payload(item.comparator_area),
                "delta": _fraction_payload(item.delta),
            }
            for item in summary.episodes
        ],
        "graphs": [
            {
                "order": item.order,
                "graph_seed": item.graph_seed,
                "episode_count": item.episode_count,
                "candidate_mean": _fraction_payload(item.candidate_mean),
                "comparator_mean": _fraction_payload(item.comparator_mean),
                "delta_mean": _fraction_payload(item.delta_mean),
            }
            for item in summary.graphs
        ],
        "orders": [
            {
                "order": item.order,
                "graph_count": item.graph_count,
                "episode_count": item.episode_count,
                "candidate_mean": _fraction_payload(item.candidate_mean),
                "comparator_mean": _fraction_payload(item.comparator_mean),
                "delta_mean": _fraction_payload(item.delta_mean),
            }
            for item in summary.orders
        ],
        "sign_counts": {
            "episode": {
                "negative": sum(item.delta < 0 for item in summary.episodes),
                "zero": sum(item.delta == 0 for item in summary.episodes),
                "positive": sum(item.delta > 0 for item in summary.episodes),
            },
            "graph": {
                "negative": sum(item.delta_mean < 0 for item in summary.graphs),
                "zero": sum(item.delta_mean == 0 for item in summary.graphs),
                "positive": sum(item.delta_mean > 0 for item in summary.graphs),
            },
        },
    }


def _bootstrap_payload(bootstrap: BootstrapSummary) -> dict[str, Any]:
    payload = cast(dict[str, Any], bootstrap.as_dict())
    payload["observed_theta"] = _fraction_payload(bootstrap.observed_theta)
    payload["interval"] = [_fraction_payload(value) for value in bootstrap.interval]
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(dict(value)) + b"\n")
    temporary.replace(path)


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    decision = str(result.get("decision"))
    comparison = cast(Mapping[str, Any], result.get("canonical_comparison", {}))
    report_lines = [
        "# Stage 4E Retained Recovery Report",
        "",
        f"Decision: **{decision}**",
        "",
        "This report reduces the preserved Stage 4E primary and deterministic replay artifacts. "
        "It does not execute a graph, policy, model, App Server, oracle, or network provider.",
        "",
        "## Preservation and frozen provenance",
        "",
        f"- Preservation metadata: `{result.get('preservation', {}).get('metadata_path')}`.",
        f"- Preserved run: `{result.get('run')}`.",
        f"- Source run: `{result.get('source_run')}`.",
        f"- Preservation manifests byte-identical: `{result.get('preservation', {}).get('manifest_bytes_equal')}`.",
        f"- Source file count/bytes: `{cast(Mapping[str, Any], result.get('preservation', {}).get('source_root', {})).get('file_count')}` / `{cast(Mapping[str, Any], result.get('preservation', {}).get('source_root', {})).get('size_bytes')}`.",
        f"- Preserved-copy file count/bytes: `{cast(Mapping[str, Any], result.get('preservation', {}).get('copy_root', {})).get('file_count')}` / `{cast(Mapping[str, Any], result.get('preservation', {}).get('copy_root', {})).get('size_bytes')}`.",
        f"- Frozen manifest SHA-256: `{result.get('manifest_sha256')}`.",
        f"- Canonical reduction SHA-256: `{result.get('canonical_reduction_sha256')}`.",
        f"- Metrics-input SHA-256: `{result.get('metrics_input_sha256')}`.",
        f"- Provider/model/App Server/oracle/runtime-network calls: `{result.get('provider_calls')}`.",
        "",
        "## Canonical replay identity",
        "",
        "The explicit recursive projection removes only `timing_ns`; no arbitrary or "
        "scientific field is filtered.",
        f"- Primary rows: `{comparison.get('primary_episode_count')}`; replay rows: `{comparison.get('replay_episode_count')}`.",
        f"- Primary/replay shards: `{len(cast(Sequence[Any], cast(Mapping[str, Any], result.get('canonical_shards', {})).get('primary', [])))}` / `{len(cast(Sequence[Any], cast(Mapping[str, Any], result.get('canonical_shards', {})).get('replay', [])))}`.",
        f"- Timing-stripped row aggregate SHA-256: `{comparison.get('primary_canonical_rows_sha256')}` (both passes).",
        f"- Timing-stripped shard aggregate SHA-256: `{cast(Mapping[str, Any], result.get('canonical_shards', {})).get('primary_aggregate_sha256')}` (both passes).",
        f"- Timing-stripped rows exact: `{comparison.get('rows_exact')}`.",
        f"- Timing-stripped shards exact: `{cast(Mapping[str, Any], result.get('canonical_shards', {})).get('exact')}`.",
        f"- Non-timing differences: `{len(cast(Sequence[Any], comparison.get('non_timing_differences', [])))}`.",
        "",
    ]
    if decision == NON_TIMING_MISMATCH:
        report_lines.extend(
            [
                "Recovery stopped before metric computation because a non-timing replay difference remained.",
                "",
                "```json",
                json.dumps(comparison.get("non_timing_differences", []), indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    elif "paired_area" in result:
        paired = cast(Mapping[str, Any], result["paired_area"])
        bootstrap = cast(Mapping[str, Any], result["bootstrap"])
        gate = cast(Mapping[str, Any], result["terminal_gate"])
        interval = cast(list[Mapping[str, Any]], bootstrap["interval"])
        order_lines = [
            f"- order {item['order']}: delta = `{item['delta_mean']['fraction']}` ({item['delta_mean']['value']})"
            for item in cast(list[Mapping[str, Any]], paired["orders"])
        ]
        graph_lines = [
            f"- order {item['order']}, graph {item['graph_seed']}: delta = `{item['delta_mean']['fraction']}` ({item['delta_mean']['value']})"
            for item in cast(list[Mapping[str, Any]], paired["graphs"])
        ]
        report_lines.extend(
            [
                "## Frozen scientific result",
                "",
                f"- Paired-area theta: `{paired['theta']['fraction']}` ({paired['theta']['value']}).",
                f"- Comparator hierarchical mean AUC (mu_B): `{paired['mu_B']['fraction']}` ({paired['mu_B']['value']}).",
                f"- Relative improvement: `{paired['relative_improvement']['fraction']}` ({paired['relative_improvement']['value']}).",
                f"- Bootstrap: `{bootstrap['samples']}` draws, seed `{bootstrap['seed']}`, interval `[{interval[0]['fraction']}, {interval[1]['fraction']}]`.",
                f"- Bootstrap sign counts: `{bootstrap['sign_counts']}`.",
                f"- Terminal gate: `{gate}`.",
                "",
                "### Equal-weight order effects",
                "",
                *order_lines,
                "",
                "### Graph-cluster means",
                "",
                *graph_lines,
                "",
                "Historical Stage 4E artifacts remain distinct; no Stage 5 work was started and HEG was not modified.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report_lines), encoding="utf-8")


def _preservation_metadata(run: Path) -> dict[str, Any]:
    metadata_path = run.parent / "preservation-metadata.json"
    if not metadata_path.is_file():
        return {"metadata_path": None, "manifest_bytes_equal": None}
    metadata = _read_json(metadata_path)
    return {
        "metadata_path": str(metadata_path),
        "manifest_bytes_equal": metadata.get("manifest_bytes_equal"),
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "copy_manifest_sha256": metadata.get("copy_manifest_sha256"),
        "source": metadata.get("source"),
        "copy": metadata.get("copy"),
        "source_root": metadata.get("source_root"),
        "copy_root": metadata.get("copy_root"),
    }


def _write_artifacts(output: Path, result: Mapping[str, Any]) -> None:
    _write_json(output / "recovery-summary.json", result)
    _write_json(output / "canonical-comparison.json", cast(Mapping[str, Any], result["canonical_comparison"]))
    if "paired_area" in result:
        _write_json(output / "paired-area-summary.json", cast(Mapping[str, Any], result["paired_area"]))
        _write_json(output / "bootstrap-support.json", cast(Mapping[str, Any], result["bootstrap"]))
        _write_json(output / "terminal-gate.json", cast(Mapping[str, Any], result["terminal_gate"]))
    _write_json(output / "preservation-verification.json", cast(Mapping[str, Any], result["preservation"]))


def recover_retained(
    run: str | Path,
    *,
    output_dir: str | Path | None = None,
    report_path: str | Path = "docs/reports/STAGE4E_RETAINED_RECOVERY_REPORT.md",
) -> dict[str, Any]:
    """Recover Stage 4E from a preserved run without creating new episodes."""

    run_path = Path(run).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else run_path.parent / "stage4e-retained-recovery-result"
    result: dict[str, Any] = {
        "schema_version": RECOVERY_SCHEMA,
        "status": "inconclusive",
        "decision": RETAINED_EVIDENCE_FAILURE,
        "run": str(run_path),
        "source_run": None,
        "provider_calls": 0,
        "stage5_started": False,
        "heg_modified": False,
        "canonical_reduction_sha256": EXPECTED_CANONICAL_REDUCTION_SHA256,
        "metrics_input_sha256": EXPECTED_METRICS_INPUT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "preservation": _preservation_metadata(run_path),
        "canonical_comparison": {
            "projection": {
                "name": "recursive_timing_stripped",
                "excluded_fields": sorted(TIMING_ONLY_FIELDS),
            },
            "primary_episode_count": 0,
            "replay_episode_count": 0,
            "missing_primary_episode_ids": [],
            "missing_replay_episode_ids": [],
            "non_timing_differences": [],
            "rows_exact": False,
        },
        "canonical_shards": {"differences": [], "exact": False},
    }
    try:
        if not run_path.is_dir():
            raise RecoveryError(f"preserved run directory is missing: {run_path}")
        preservation = cast(Mapping[str, Any], result["preservation"])
        if preservation.get("manifest_bytes_equal") is False:
            raise RecoveryError("source and preserved-copy manifests do not match")
        project_root = Path(__file__).resolve().parents[2]
        manifest, freeze, expected = _load_frozen_context(project_root)
        result["source_run"] = preservation.get("source")
        primary = _read_pass(run_path, "primary", manifest, freeze, expected)
        replay = _read_pass(run_path, "replay", manifest, freeze, expected)
        comparison = compare_canonical_rows(primary.rows, replay.rows)
        shard_comparison = _compare_shards(primary, replay)
        comparison["primary_canonical_rows_sha256"] = comparison["canonical_row_hashes_sha256"]
        comparison["replay_canonical_rows_sha256"] = sha256(
            [
                {
                    "episode_id": episode_id,
                    "sha256": hashlib.sha256(
                        canonical_bytes(timing_stripped_projection(replay.rows[episode_id]))
                    ).hexdigest(),
                }
                for episode_id in sorted(replay.rows)
            ]
        )
        result["canonical_comparison"] = comparison
        result["canonical_shards"] = shard_comparison
        result["primary_summary"] = {
            key: primary.summary.get(key)
            for key in (
                "run_identity_sha256",
                "config_sha256",
                "shard_hashes_sha256",
                "canonical_reduction_sha256",
                "metrics_input_sha256",
                "counts",
            )
        }
        result["replay_summary"] = {
            key: replay.summary.get(key)
            for key in (
                "run_identity_sha256",
                "config_sha256",
                "shard_hashes_sha256",
                "canonical_reduction_sha256",
                "metrics_input_sha256",
                "counts",
            )
        }
        if not comparison["rows_exact"] or not shard_comparison["exact"]:
            result["decision"] = NON_TIMING_MISMATCH
            _write_artifacts(output, result)
            _write_report(Path(report_path), result)
            return result
        episodes = []
        for episode_id in sorted(primary.rows):
            row = primary.rows[episode_id]
            policies = cast(Mapping[str, Mapping[str, Any]], row["policies"])
            candidate_curve = policies[CHAMPION_ID].get("normalized_best_so_far_curve")
            comparator_curve = policies[COMPARATOR_ID].get("normalized_best_so_far_curve")
            if not isinstance(candidate_curve, list) or not isinstance(comparator_curve, list):
                raise RecoveryError(f"{episode_id}: normalized curves are missing")
            episodes.append(
                PairedAreaEpisode(
                    order=int(row["order"]),
                    graph_seed=int(row["graph_seed"]),
                    episode_id=episode_id,
                    candidate_curve=cast(list[float], candidate_curve),
                    comparator_curve=cast(list[float], comparator_curve),
                )
            )
        summary = summarize_paired_areas(episodes)
        bootstrap = bootstrap_paired_theta(
            summary,
            samples=BOOTSTRAP_SAMPLES,
            seed=BOOTSTRAP_SEED,
            confidence_level=CONFIDENCE_LEVEL,
        )
        paired_payload = _paired_payload(summary)
        bootstrap_payload = _bootstrap_payload(bootstrap)
        gate: dict[str, Any] = {
            "frozen_policy_identities_exact": primary.summary.get("policy_source_sha256") == POLICY_SOURCE_SHA256,
            "all_primary_episodes_complete": len(primary.rows) == 1536,
            "all_replay_episodes_complete": len(replay.rows) == 1536,
            "primary_replay_exact": bool(comparison["rows_exact"] and shard_comparison["exact"]),
            "graph_validity_100_percent": all(int(row.get("invalid_graphs", 0)) == 0 for row in primary.rows.values()),
            "worker_failures_crashes_timeouts_protocol_zero": all(int(row.get("policy_failures", 0)) == 0 for row in primary.rows.values()),
            "selected_plan_only_and_oracle_zero": all(int(row.get("oracle_score_calls", 0)) == 0 for row in primary.rows.values()),
            "model_app_server_calls_zero": all(
                int(row.get("model_calls", 0)) == 0 and int(row.get("app_server_calls", 0)) == 0
                for row in primary.rows.values()
            ),
        }
        gate.update(
            terminal_gate_checks(
                summary,
                bootstrap,
                minimum_relative_improvement=RELATIVE_THRESHOLD,
                minimum_bootstrap_lower_bound=0,
            )
        )
        gate["order_effects_nonnegative"] = all(item.delta_mean >= 0 for item in summary.orders)
        result["paired_area"] = paired_payload
        result["bootstrap"] = bootstrap_payload
        result["terminal_gate"] = {"checks": gate, "all_pass": all(gate.values())}
        result["decision"] = "GO_TO_STAGE_5" if all(gate.values()) else "NO_GO"
        result["status"] = "completed"
        _write_artifacts(output, result)
        _write_report(Path(report_path), result)
        return result
    except RecoveryError as error:
        result["error"] = f"{type(error).__name__}: {error}"
    except (OSError, TypeError, ValueError, KeyError, IndexError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    _write_artifacts(output, result)
    _write_report(Path(report_path), result)
    return result


__all__ = [
    "NON_TIMING_MISMATCH",
    "RECOVERY_SCHEMA",
    "RETAINED_EVIDENCE_FAILURE",
    "compare_canonical_rows",
    "recover_retained",
    "timing_stripped_projection",
]
