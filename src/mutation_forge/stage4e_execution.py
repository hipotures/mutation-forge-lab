"""Provider-free, replayable execution for the Stage 4E confirmation pass.

The executor deliberately has no CLI or generation dependency.  Its only
evaluation entry point is :func:`stage3.evaluation.run_development_episode`.
Callers provide the two already-frozen policy source strings and their public
IDs; Stage 3's internal baseline labels are used only at that boundary.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from mutation_forge.stage2d.manifest import read_cpu_topology
from mutation_forge.stage3.evaluation import (
    THREAD_ENVIRONMENT,
    canonical_projection,
    run_development_episode,
)
from mutation_forge.stage3.manifest import canonical_bytes, sha256

SCHEMA_VERSION = "stage4e.execution.v1"
SHARD_COUNT = 24
EPISODES_PER_SHARD = 64
EPISODE_COUNT = SHARD_COUNT * EPISODES_PER_SHARD
MAX_WORKERS = 8
_FORBIDDEN_COUNTERS = (
    "model_calls",
    "app_server_calls",
    "oracle_score_calls",
    "runtime_network_calls",
)


def _get(value: object, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _episodes(manifest: object) -> list[dict[str, Any]]:
    rows = _get(manifest, "episodes", manifest)
    if isinstance(rows, Mapping):
        rows = rows.values()
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes)):
        raise ValueError("Stage 4E manifest must contain an episodes iterable")
    result = [dict(cast(Mapping[str, Any], row)) for row in rows]
    ids = [str(row.get("episode_id", "")) for row in result]
    if len(result) != EPISODE_COUNT:
        raise ValueError(f"Stage 4E manifest must contain exactly {EPISODE_COUNT} episodes")
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Stage 4E manifest episode IDs must be unique and non-empty")
    return sorted(result, key=lambda row: str(row["episode_id"]))


def _manifest_hash(manifest: object, rows: list[dict[str, Any]]) -> str:
    declared = _get(manifest, "manifest_sha256")
    return declared if isinstance(declared, str) and len(declared) == 64 else sha256(rows)


def _config_hash(config: object) -> str:
    stable = _get(config, "stable_hash")
    if callable(stable):
        result = stable()
        if isinstance(result, str) and len(result) == 64:
            return result
    if isinstance(config, Mapping):
        return sha256(dict(config))
    resolved = _get(config, "resolved_dict")
    if callable(resolved):
        return sha256(resolved())
    values = getattr(config, "__dict__", None)
    if isinstance(values, dict):
        return sha256(values)
    raise TypeError("Stage 4E config must provide stable_hash or a serializable mapping")


def _policy_sources(policies: Mapping[str, str]) -> dict[str, str]:
    if len(policies) != 2:
        raise ValueError("Stage 4E requires exactly two frozen policy sources")
    result: dict[str, str] = {}
    for policy_id, source in policies.items():
        name = str(policy_id)
        if not name or name in {"__class__", "__proto__"} or not isinstance(source, str):
            raise ValueError("Stage 4E policy IDs must be safe strings with string sources")
        result[name] = source
    if len(result) != 2:
        raise ValueError("Stage 4E policy IDs must be unique")
    return {name: result[name] for name in sorted(result)}


def _identity(
    config: object, manifest: object, rows: list[dict[str, Any]], policies: Mapping[str, str]
) -> tuple[str, dict[str, str], str, str]:
    source_hashes = {
        name: hashlib.sha256(source.encode("utf-8")).hexdigest()
        for name, source in sorted(policies.items())
    }
    manifest_hash, config_hash = _manifest_hash(manifest, rows), _config_hash(config)
    run_identity = sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": manifest_hash,
            "config_sha256": config_hash,
            "policy_source_sha256": source_hashes,
        }
    )
    return run_identity, source_hashes, manifest_hash, config_hash


def _shards(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(SHARD_COUNT)]
    for index, row in enumerate(rows):
        shards[index % SHARD_COUNT].append(row)
    if any(len(shard) != EPISODES_PER_SHARD for shard in shards):
        raise AssertionError("Stage 4E shard allocation drifted")
    return shards


def _strip_timing(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_timing(item)
            for key, item in value.items()
            if key not in {"started_at", "finished_at", "elapsed_seconds", "timing", "path"}
            and not str(key).endswith("_ns")
        }
    if isinstance(value, (list, tuple)):
        return [_strip_timing(item) for item in value]
    return value


def _rename_policy_keys(value: Any, names: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            names.get(str(key), str(key)): _rename_policy_keys(item, names)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_policy_keys(item, names) for item in value]
    return value


def _compact_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in cast(Mapping[str, Any], _strip_timing(value)).items()
        if str(key) != "final_score"
    }


def _metrics(record: Mapping[str, Any], policy_ids: list[str]) -> dict[str, Any]:
    policies = cast(Mapping[str, Any], record["policies"])
    return {
        "episode_id": record["episode_id"],
        "terminal_status": record["terminal_status"],
        "horizon": record["horizon"],
        "policies": {
            name: {
                key: policies[name].get(key, 0)
                for key in ("auc", "best_total_witnesses", "accepted_count", "failure_count")
            }
            for name in policy_ids
        },
    }


def _compact_record(
    raw: Mapping[str, Any], policy_ids: list[str], source_hashes: Mapping[str, str]
) -> dict[str, Any]:
    internal_to_public = {"random": policy_ids[0], "structural": policy_ids[1]}
    remapped = cast(dict[str, Any], _rename_policy_keys(raw, internal_to_public))
    if str(remapped.get("episode_id", "")) == "":
        raise ValueError("Stage 4E evaluator omitted an episode ID")
    if remapped.get("terminal_status") != "completed":
        raise ValueError("Stage 4E evaluator returned a non-completed episode")
    policies = remapped.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != set(policy_ids):
        raise ValueError("Stage 4E evaluator policy roster drifted")
    if any(int(remapped.get(counter, 0)) != 0 for counter in _FORBIDDEN_COUNTERS):
        raise ValueError("Stage 4E crossed a forbidden provider, oracle, or network boundary")
    canonical = sha256(canonical_projection(remapped))
    result = {
        str(key): _strip_timing(value)
        for key, value in remapped.items()
        if key not in {"policies", "policy_identities", "steps", "canonical_episode_sha256"}
    }
    result["policies"] = {
        name: _compact_policy(cast(Mapping[str, Any], policies[name])) for name in policy_ids
    }
    result["policy_source_sha256"] = dict(source_hashes)
    result["metrics_input"] = _metrics(remapped, policy_ids)
    result["canonical_episode_sha256"] = canonical
    return result


def _set_threads(cpu_id: int | None = None) -> list[int]:
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    if cpu_id is None:
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Linux CPU affinity is required for a reserved Stage 4E run")
    os.sched_setaffinity(0, {cpu_id})
    observed = sorted(os.sched_getaffinity(0))
    if observed != [cpu_id]:
        raise RuntimeError("Stage 4E worker affinity mismatch")
    return observed


def _cpu_plan(config: object, workers: int) -> tuple[list[int | None], list[int]]:
    limits = _get(config, "limits", {})
    reserved = int(_get(limits, "reserved_physical_cores", 0))
    if reserved == 0:
        return [None] * workers, []
    topology = read_cpu_topology()
    if len(topology) < workers + reserved:
        raise RuntimeError("cannot preserve the frozen Stage 4 physical-core reserve")
    worker_ids: list[int | None] = [topology[index].cpu_id for index in range(workers)]
    reserved_ids = [topology[index].cpu_id for index in range(workers, workers + reserved)]
    if set(worker_ids) & set(reserved_ids):
        raise RuntimeError("Stage 4E worker and reserved CPU sets overlap")
    return worker_ids, reserved_ids


def _uses_process_workers(config: object) -> bool:
    return int(_get(_get(config, "limits", {}), "reserved_physical_cores", 0)) > 0


def _evaluate_shard(
    config: object,
    rows: list[dict[str, Any]],
    public_sources: Mapping[str, str],
    source_hashes: Mapping[str, str],
    cpu_id: int | None,
    reserved_cpu_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = _set_threads(cpu_id)
    policy_ids = list(public_sources)
    # Stage 3 validates these two labels, but they are an internal adapter only.
    internal_sources = {
        "random": public_sources[policy_ids[0]],
        "structural": public_sources[policy_ids[1]],
    }
    records: list[dict[str, Any]] = []
    for episode in rows:
        raw = run_development_episode(config, episode, internal_sources)
        record = _compact_record(cast(Mapping[str, Any], raw), policy_ids, source_hashes)
        if record["episode_id"] != episode["episode_id"]:
            raise ValueError(
                "Stage 4E evaluator output does not match authoritative episode roster"
            )
        records.append(record)
    return records, {
        "process_id": os.getpid(),
        "cpu_id": cpu_id,
        "observed_affinity": observed,
        "reserved_cpu_ids": reserved_cpu_ids,
        "thread_environment": dict(THREAD_ENVIRONMENT),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(dict(value)) + b"\n")
    temporary.replace(path)


def _write_shard(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = b"".join(canonical_bytes(record) + b"\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(
        fileobj=raw, mode="wb", filename="", mtime=0
    ) as out:
        out.write(payload)
    temporary.replace(path)
    return {
        "path": path.name,
        "record_count": len(records),
        "episode_ids": [str(record["episode_id"]) for record in records],
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _read_shard(
    path: Path, entry: Mapping[str, Any], expected_ids: list[str]
) -> list[dict[str, Any]]:
    if (
        path.name != str(entry.get("path"))
        or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("file_sha256")
    ):
        raise ValueError("Stage 4E shard hash mismatch")
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Stage 4E shard row must be an object")
                records.append(cast(dict[str, Any], row))
    ids = [str(record.get("episode_id", "")) for record in records]
    if ids != expected_ids or len(records) != int(entry.get("record_count", -1)):
        raise ValueError("Stage 4E shard roster mismatch")
    return records


def _paths(root: Path, identity: str, pass_name: str) -> tuple[Path, Path]:
    prefix = f"stage4e-{identity[:24]}-{pass_name}"
    return root / f"{prefix}-state.json", root / f"{prefix}-summary.json"


def _load_completed(summary_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("status") != "completed":
        return None
    result = verify_stage4e_pass(summary_path, {"episodes": rows})
    if not result["exact"]:
        raise ValueError("existing Stage 4E summary failed verification")
    return cast(dict[str, Any], dict(raw))


def execute_stage4e_pass(
    config: object,
    manifest: object,
    policies: Mapping[str, str],
    output_dir: str | Path,
    pass_name: str,
    *,
    workers: int = MAX_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one 24-by-64 Stage 4E pass using exactly two frozen source strings.

    A completed pass may be loaded again.  An interrupted pass is only
    resumable while no shard outcome has reached disk; after that point the
    frozen run is deliberately non-resumable.
    """
    if not pass_name or "/" in pass_name or "\\" in pass_name:
        raise ValueError("Stage 4E pass name must be a safe filename component")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("Stage 4E workers must be between 1 and 8")
    rows, public_sources = _episodes(manifest), _policy_sources(policies)
    identity, source_hashes, manifest_hash, config_hash = _identity(
        config, manifest, rows, public_sources
    )
    root = Path(output_dir)
    state_path, summary_path = _paths(root, identity, pass_name)
    completed = _load_completed(summary_path, rows) if resume else None
    if completed is not None:
        return {**completed, "records": _load_records(completed, summary_path.parent, rows)}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state.get("outcomes_persisted", 0)) > 0:
            raise RuntimeError("Stage 4E may resume only before the first persisted outcome")
    shards = _shards(rows)
    worker_cpu_ids, reserved_cpu_ids = _cpu_plan(config, workers)
    _write_json(
        state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "prepared",
            "run_identity_sha256": identity,
            "pass": pass_name,
            "outcomes_persisted": 0,
        },
    )
    started = time.perf_counter_ns()
    executor_type = ProcessPoolExecutor if _uses_process_workers(config) else ThreadPoolExecutor
    results: list[list[dict[str, Any]] | None] = [None] * SHARD_COUNT
    health: list[dict[str, Any] | None] = [None] * SHARD_COUNT
    with executor_type(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _evaluate_shard,
                config,
                shard,
                public_sources,
                source_hashes,
                worker_cpu_ids[index % workers],
                reserved_cpu_ids,
            ): index
            for index, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index], health[index] = future.result()
    entries: list[dict[str, Any]] = []
    for index, records in enumerate(results):
        assert records is not None and health[index] is not None
        shard_path = root / f"stage4e-{identity[:24]}-{pass_name}-shard-{index:02d}.jsonl.gz"
        entry = _write_shard(shard_path, records)
        entry["worker_health"] = health[index]
        entries.append(entry)
        _write_json(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "outcomes_persisted",
                "run_identity_sha256": identity,
                "pass": pass_name,
                "outcomes_persisted": index + 1,
            },
        )
    ordered = sorted(
        [record for shard in results if shard is not None for record in shard],
        key=lambda record: str(record["episode_id"]),
    )
    expected_ids = [str(row["episode_id"]) for row in rows]
    if [str(record["episode_id"]) for record in ordered] != expected_ids:
        raise ValueError("Stage 4E reduction roster drifted")
    shard_hashes = [entry["file_sha256"] for entry in entries]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "pass": pass_name,
        "artifact_dir": str(root),
        "run_identity_sha256": identity,
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "policy_source_sha256": source_hashes,
        "policy_ids": list(public_sources),
        "record_count": len(ordered),
        "shard_count": SHARD_COUNT,
        "episodes_per_shard": EPISODES_PER_SHARD,
        "manifest_episode_ids": expected_ids,
        "manifest_shards": [[str(row["episode_id"]) for row in shard] for shard in shards],
        "shard_hashes_sha256": sha256(shard_hashes),
        "canonical_reduction_sha256": sha256(
            [record["canonical_episode_sha256"] for record in ordered]
        ),
        "metrics_input_sha256": sha256([record["metrics_input"] for record in ordered]),
        "counts": {**{counter: 0 for counter in _FORBIDDEN_COUNTERS}, "episodes": len(ordered)},
        "worker_health": {
            "requested": workers,
            "effective": workers,
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "reserved_cpu_ids": reserved_cpu_ids,
            "assignments": [cast(dict[str, Any], item) for item in health],
        },
        "phase_timings_ns": {"evaluation": time.perf_counter_ns() - started},
        "shards": entries,
    }
    _write_json(summary_path, summary)
    _write_json(
        state_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "run_identity_sha256": identity,
            "pass": pass_name,
            "outcomes_persisted": SHARD_COUNT,
            "summary": summary_path.name,
        },
    )
    return {**summary, "records": ordered}


def _load_records(
    summary: Mapping[str, Any], root: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    shards = summary.get("shards")
    expected_shards = _shards(rows)
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise ValueError("Stage 4E summary shard count mismatch")
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(shards):
        if not isinstance(entry, Mapping):
            raise ValueError("Stage 4E summary has an invalid shard entry")
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or relative.name != str(relative):
            raise ValueError("Stage 4E shard path escapes its output directory")
        records.extend(
            _read_shard(
                root / relative,
                entry,
                [str(row["episode_id"]) for row in expected_shards[index]],
            )
        )
    return sorted(records, key=lambda record: str(record["episode_id"]))


def verify_stage4e_pass(
    value: Mapping[str, Any] | str | Path, manifest: object | None = None
) -> dict[str, Any]:
    """Validate persisted shard, reduction, metric, and source-identity hashes."""
    try:
        summary_path: Path | None = None
        if isinstance(value, Mapping):
            summary = dict(value)
            root = Path(str(summary.get("artifact_dir", ".")))
        else:
            summary_path = Path(value)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            root = summary_path.parent
        if not isinstance(summary, Mapping) or summary.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unexpected Stage 4E summary schema")
        if manifest is None:
            raw_rows = summary.get("manifest_episode_ids")
            if not isinstance(raw_rows, list):
                raise ValueError("Stage 4E summary has no manifest roster")
            # Only identities are needed to validate an already declared roster.
            rows = [{"episode_id": str(item)} for item in raw_rows]
            if len(rows) != EPISODE_COUNT:
                raise ValueError("Stage 4E summary manifest roster has wrong size")
        else:
            rows = _episodes(manifest)
        records = _load_records(summary, root, rows)
        expected_ids = [str(row["episode_id"]) for row in rows]
        if [str(record["episode_id"]) for record in records] != expected_ids:
            raise ValueError("Stage 4E record roster mismatch")
        reduction = sha256([record["canonical_episode_sha256"] for record in records])
        metrics = sha256([record["metrics_input"] for record in records])
        entries = cast(list[Mapping[str, Any]], summary["shards"])
        shard_hashes = sha256([entry["file_sha256"] for entry in entries])
        exact = (
            reduction == summary.get("canonical_reduction_sha256")
            and metrics == summary.get("metrics_input_sha256")
            and shard_hashes == summary.get("shard_hashes_sha256")
        )
        return {
            "status": "completed" if exact else "failed",
            "exact": exact,
            "record_count": len(records),
            "canonical_reduction_sha256": reduction,
            "metrics_input_sha256": metrics,
            "shard_hashes_sha256": shard_hashes,
        }
    except Exception as error:
        return {"status": "failed", "exact": False, "error": f"{type(error).__name__}: {error}"}


def canonical_replay_identity(value: Mapping[str, Any]) -> str:
    """Return the pass-name- and timing-independent Stage 4E replay identity."""
    stable = {
        key: value.get(key)
        for key in (
            "schema_version",
            "run_identity_sha256",
            "manifest_sha256",
            "config_sha256",
            "policy_source_sha256",
            "policy_ids",
            "record_count",
            "shard_count",
            "episodes_per_shard",
            "manifest_episode_ids",
            "manifest_shards",
            "shard_hashes_sha256",
            "canonical_reduction_sha256",
            "metrics_input_sha256",
            "counts",
        )
    }
    return sha256(stable)


def verify_stage4e_replay(
    primary: Mapping[str, Any] | str | Path, replay: Mapping[str, Any] | str | Path
) -> dict[str, Any]:
    """Compare primary and replay summaries without invoking an evaluator."""
    try:
        def load(item: Mapping[str, Any] | str | Path) -> dict[str, Any]:
            if isinstance(item, Mapping):
                return dict(item)
            return cast(dict[str, Any], json.loads(Path(item).read_text(encoding="utf-8")))

        left, right = load(primary), load(replay)
        left_hash, right_hash = canonical_replay_identity(left), canonical_replay_identity(right)
        exact = left_hash == right_hash
        return {
            "status": "completed" if exact else "failed",
            "decision": "exact" if exact else "mismatch",
            "exact": exact,
            "primary_sha256": left_hash,
            "replay_sha256": right_hash,
            "provider_calls": 0,
        }
    except Exception as error:
        return {
            "status": "failed",
            "decision": "mismatch",
            "exact": False,
            "provider_calls": 0,
            "error": f"{type(error).__name__}: {error}",
        }


def execute_stage4e_confirmation(
    config: object,
    manifest: object,
    policies: Mapping[str, str],
    output_dir: str | Path,
    *,
    workers: int = MAX_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute the primary and replay passes and return their verified summaries."""
    root = Path(output_dir)
    primary = execute_stage4e_pass(
        config, manifest, policies, root / "primary", "primary", workers=workers, resume=resume
    )
    replay = execute_stage4e_pass(
        config, manifest, policies, root / "replay", "replay", workers=workers, resume=resume
    )
    return {
        "primary": primary,
        "replay": replay,
        "replay_verification": verify_stage4e_replay(primary, replay),
    }


__all__ = [
    "EPISODE_COUNT",
    "EPISODES_PER_SHARD",
    "MAX_WORKERS",
    "SCHEMA_VERSION",
    "SHARD_COUNT",
    "canonical_replay_identity",
    "execute_stage4e_confirmation",
    "execute_stage4e_pass",
    "verify_stage4e_pass",
    "verify_stage4e_replay",
]
