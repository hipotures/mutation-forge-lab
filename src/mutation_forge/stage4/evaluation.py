"""Deterministic Stage 4 candidate evaluation and compact pass persistence.

Stage 4 deliberately delegates trajectory semantics to :mod:`stage3.evaluation`.
This module only supplies deterministic manifest sharding, candidate caching and
the small candidate-facing projection used by the evolutionary search.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage2d.manifest import read_cpu_topology
from mutation_forge.stage3.evaluation import (
    THREAD_ENVIRONMENT,
    canonical_projection,
    run_development_episode,
)
from mutation_forge.stage3.manifest import canonical_bytes, sha256

MAX_WORKERS = 8
VALIDATION_ALLOWLIST = frozenset({"champion", "stage3-candidate-slot-04", "random", "structural"})


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _episodes(manifest: object) -> list[dict[str, Any]]:
    rows = _get(manifest, "episodes", manifest)
    if isinstance(rows, Mapping):
        rows = rows.values()
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes)):
        raise ValueError("manifest must contain an episodes iterable")
    result = [dict(cast(Mapping[str, Any], row)) for row in rows]
    ids = [str(row.get("episode_id", "")) for row in result]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest episode IDs must be unique and non-empty")
    return sorted(result, key=lambda row: str(row["episode_id"]))


def _manifest_hash(manifest: object) -> str:
    declared = _get(manifest, "manifest_sha256")
    if isinstance(declared, str) and len(declared) == 64:
        return declared
    return sha256(_episodes(manifest))


def _config_hash(config: object) -> str:
    explicit = _get(config, "stable_hash")
    if callable(explicit):
        try:
            resolved = explicit()
            if isinstance(resolved, str):
                return resolved
        except Exception:
            pass
    if isinstance(config, Mapping):
        payload: Any = dict(config)
    else:
        payload = getattr(config, "resolved_dict", lambda: repr(config))()
    return sha256(payload)


def _source(value: Any) -> Any:
    """Return the policy object accepted by Stage 3 (source or ranker)."""
    if isinstance(value, Mapping) and "source" in value:
        return value["source"]
    source = getattr(value, "source", None)
    if isinstance(source, str):
        return source
    return value


def _identity(value: Any) -> str:
    source = _source(value)
    identity = getattr(value, "identity", None)
    if identity is not None:
        digest = getattr(identity, "source_sha256", None)
        if isinstance(digest, str):
            return digest
    if isinstance(source, str):
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    return hashlib.sha256(canonical_bytes(repr(source))).hexdigest()


def candidate_cache_key(
    candidate_id: str,
    candidate: Any,
    manifest: object,
    config: object,
    pass_name: str,
) -> str:
    """Stable key for a candidate/pass cache entry."""
    payload = [
        "stage4.candidate-cache.v1",
        candidate_id,
        _identity(candidate),
        _manifest_hash(manifest),
        _config_hash(config),
        pass_name,
    ]
    return sha256(payload)


def _strip(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip(item)
            for key, item in value.items()
            if key not in {"started_at", "finished_at", "elapsed_seconds", "timing", "path"}
            and not str(key).endswith("_ns")
        }
    if isinstance(value, (list, tuple)):
        return [_strip(item) for item in value]
    return value


def metrics_input_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Projection consumed by fitness; excludes full shared score/trace objects."""
    policies = record.get("policies", {})
    candidate_id = str(record.get("candidate_id", ""))
    policy: Any = (
        policies.get(candidate_id, {})
        if isinstance(policies, Mapping) and candidate_id
        else next(iter(policies.values()), {})
        if isinstance(policies, Mapping)
        else {}
    )
    if not isinstance(policy, Mapping):
        policy = {}
    return {
        "episode_id": record.get("episode_id"),
        "terminal_status": record.get("terminal_status"),
        "horizon": record.get("horizon"),
        "auc": policy.get("auc", 0.0),
        "best_total_witnesses": policy.get("best_total_witnesses"),
        "accepted_count": policy.get("accepted_count", 0),
        "failure_count": policy.get("failure_count", 0),
    }


def _compact_record(record: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    """Store candidate-only trajectory evidence and shared episode facts once."""
    compact = {
        key: _strip(value)
        for key, value in record.items()
        if key not in {"policies", "policy_identities", "steps", "canonical_episode_sha256"}
    }
    policies = record.get("policies")
    selected = policies.get(candidate_id) if isinstance(policies, Mapping) else None
    if selected is None and isinstance(policies, Mapping) and len(policies) == 1:
        selected = next(iter(policies.values()))
    # Keep the complete candidate+baseline roster as compact summaries. Full
    # traces and score objects remain shared Stage 3 evidence and are omitted.
    if isinstance(policies, Mapping):
        compact["policies"] = {
            str(name): {
                key: _strip(value) for key, value in summary.items() if key != "final_score"
            }
            for name, summary in policies.items()
            if isinstance(summary, Mapping)
        }
    compact["candidate_id"] = candidate_id
    compact["metrics_input"] = metrics_input_projection(
        {**record, "candidate_id": candidate_id, "policies": {candidate_id: selected or {}}}
    )
    compact["source_identity_sha256"] = (
        record.get("policy_identities", {}).get(candidate_id, {}).get("source_sha256")
        if isinstance(record.get("policy_identities"), Mapping)
        else None
    )
    compact["canonical_episode_sha256"] = sha256(canonical_projection(record))
    return compact


def _cpu_plan(config: object, workers: int) -> tuple[list[int | None], list[int]]:
    limits = _get(config, "limits", {})
    reserved = int(_get(limits, "reserved_physical_cores", 0))
    if reserved == 0:
        return [None] * workers, []
    topology = read_cpu_topology()
    if len(topology) < workers + reserved:
        raise RuntimeError("cannot preserve the frozen Stage 4 physical-core reserve")
    worker_ids: list[int | None] = [
        topology[index].cpu_id for index in range(workers)
    ]
    reserved_ids = [
        topology[index].cpu_id for index in range(workers, workers + reserved)
    ]
    if set(worker_ids) & set(reserved_ids):
        raise RuntimeError("Stage 4 worker and reserved CPU sets overlap")
    return worker_ids, reserved_ids


def _set_threads(cpu_id: int | None = None) -> list[int]:
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    if cpu_id is None:
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Linux CPU affinity is required for Stage 4 evaluation")
    os.sched_setaffinity(0, {cpu_id})
    observed = sorted(os.sched_getaffinity(0))
    if observed != [cpu_id]:
        raise RuntimeError("Stage 4 evaluation worker affinity mismatch")
    return observed


def _shard(rows: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for index, row in enumerate(rows):
        result[index % count].append(row)
    return result


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    payload = b"".join(canonical_bytes(dict(record)) + b"\n" for record in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as out:
        out.write(payload)
    tmp.replace(path)
    return {
        "path": path.name,
        "record_count": len(rows),
        "episode_ids": [str(record.get("episode_id")) for record in rows],
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("shard record is not an object")
                records.append(cast(dict[str, Any], value))
    return records


def _load_valid_shard(
    path: Path, expected_ids: set[str], entry: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    try:
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get(
            "file_sha256"
        ):
            return None
        records = _read_jsonl(path)
        ids = [str(item.get("episode_id")) for item in records]
        if (
            len(ids) != len(set(ids))
            or set(ids) != expected_ids
            or len(records) != int(entry.get("record_count", -1))
        ):
            return None
        return records
    except Exception:
        return None


def _evaluate_shard(
    config: object,
    rows: list[dict[str, Any]],
    policies: Mapping[str, Any],
    candidate_id: str,
    cpu_id: int | None,
    reserved_cpu_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = _set_threads(cpu_id)
    result: list[dict[str, Any]] = []
    for episode in rows:
        raw = run_development_episode(config, episode, policies)
        if (
            raw.get("model_calls", 0)
            or raw.get("app_server_calls", 0)
            or raw.get("oracle_score_calls", 0)
            or raw.get("runtime_network_calls", 0)
        ):
            raise ValueError(
                "Stage 4 evaluation crossed a forbidden provider/oracle/network boundary"
            )
        result.append(_compact_record(raw, candidate_id))
    return result, {
        "cpu_id": cpu_id,
        "observed_affinity": observed,
        "reserved_cpu_ids": reserved_cpu_ids,
        "thread_environment": dict(THREAD_ENVIRONMENT),
    }


def _pass_paths(root: Path, key: str, pass_name: str, shard_count: int) -> tuple[Path, Path]:
    prefix = f"stage4-{key}-{pass_name}"
    return root / f"{prefix}-shards.json", root / f"{prefix}-summary.json"


def evaluate_program_manifest(
    config: object,
    manifest: object,
    candidate_id: str,
    candidate: Any,
    *,
    baselines: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    workers: int | None = None,
    shard_count: int | None = None,
    pass_name: str = "primary",
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate one candidate over every manifest episode deterministically."""
    if not candidate_id or candidate_id in {"__proto__", "__class__"}:
        raise ValueError("invalid candidate ID")
    rows = _episodes(manifest)
    count = int(shard_count or _get(_get(config, "experiment", config), "shard_count", 8))
    count = max(1, min(MAX_WORKERS, count))
    requested = int(
        workers or _get(_get(config, "resources", config), "max_evaluation_workers", count)
    )
    if requested < 1 or requested > MAX_WORKERS:
        raise ValueError("workers must be between 1 and 8")
    base = dict(baselines or {})
    if candidate_id in base:
        base.pop(candidate_id)
    for name in ("random", "structural"):
        if name not in base and name != candidate_id:
            raise ValueError(f"missing required baseline {name}")
    policies = {**base, candidate_id: _source(candidate)}
    if candidate_id in {"random", "structural"}:
        policies[candidate_id] = _source(candidate)
    key = candidate_cache_key(candidate_id, candidate, manifest, config, pass_name)
    root = Path(output_dir or _get(config, "run_root", "."))
    shards_manifest_path, summary_path = _pass_paths(root, key, pass_name, count)
    shards = _shard(rows, count)
    worker_cpu_ids, reserved_cpu_ids = _cpu_plan(config, requested)
    entries: list[dict[str, Any] | None] = [None] * count
    records_by_shard: list[list[dict[str, Any]] | None] = [None] * count
    if resume and shards_manifest_path.is_file():
        try:
            raw = json.loads(shards_manifest_path.read_text(encoding="utf-8"))
            if raw.get("candidate_cache_key") == key and int(raw.get("shard_count")) == count:
                for index, shard in enumerate(raw.get("shards", [])):
                    if index >= count or not isinstance(shard, Mapping):
                        continue
                    path = root / str(shard.get("path", ""))
                    records_by_shard[index] = _load_valid_shard(
                        path, {str(row["episode_id"]) for row in shards[index]}, shard
                    )
                    if records_by_shard[index] is not None:
                        entries[index] = dict(shard)
        except Exception:
            pass
    started = time.perf_counter_ns()
    missing = [index for index, records in enumerate(records_by_shard) if records is None]
    health: list[dict[str, Any]] = []
    if missing:
        with ThreadPoolExecutor(max_workers=requested, thread_name_prefix="stage4-eval") as pool:
            futures = {
                pool.submit(
                    _evaluate_shard,
                    config,
                    shards[index],
                    policies,
                    candidate_id,
                    worker_cpu_ids[index % requested],
                    reserved_cpu_ids,
                ): index
                for index in missing
            }
            for future in as_completed(futures):
                index = futures[future]
                records_by_shard[index], worker_health = future.result()
                health.append({"shard": index, **worker_health})
    for index, records in enumerate(records_by_shard):
        assert records is not None
        path = root / f"stage4-{key}-{pass_name}-shard-{index:02d}.jsonl.gz"
        if entries[index] is None:
            entry = _write_jsonl(path, records)
            entry["worker_health"] = next(
                item for item in health if item["shard"] == index
            )
            entries[index] = entry
    final_entries = [cast(dict[str, Any], entry) for entry in entries]
    all_records = [record for records in records_by_shard if records for record in records]
    ordered = sorted(all_records, key=lambda record: str(record["episode_id"]))
    if len({str(record["episode_id"]) for record in ordered}) != len(rows):
        raise ValueError("missing or duplicate episode records")
    reduction_hash = sha256([record["canonical_episode_sha256"] for record in ordered])
    metrics_hash = sha256([record["metrics_input"] for record in ordered])
    manifest_shards = [[str(row["episode_id"]) for row in shard_rows] for shard_rows in shards]
    shard_manifest = {
        "schema_version": "stage4.evaluation_shards.v1",
        "candidate_cache_key": key,
        "candidate_id": candidate_id,
        "manifest_sha256": _manifest_hash(manifest),
        "config_sha256": _config_hash(config),
        "pass": pass_name,
        "shard_count": count,
        "shards": final_entries,
        "record_count": len(ordered),
        "manifest_episode_ids": [str(row["episode_id"]) for row in rows],
        "manifest_shards": manifest_shards,
        "canonical_reduction_sha256": reduction_hash,
        "metrics_input_sha256": metrics_hash,
    }
    shards_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shards_manifest_path.write_text(
        json.dumps(shard_manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "stage4.evaluation.v1",
        "candidate_id": candidate_id,
        "candidate_cache_key": key,
        "artifact_dir": str(root),
        "manifest_sha256": _manifest_hash(manifest),
        "config_sha256": _config_hash(config),
        "pass": pass_name,
        "record_count": len(ordered),
        "shard_count": count,
        "manifest_episode_ids": [str(row["episode_id"]) for row in rows],
        "manifest_shards": manifest_shards,
        "canonical_reduction_sha256": reduction_hash,
        "metrics_input_sha256": metrics_hash,
        "worker_health": {
            "requested": requested,
            "effective": requested,
            "completed_shards": count,
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "affinity_supported": hasattr(os, "sched_setaffinity"),
            "assignments": [
                cast(Mapping[str, Any], entry.get("worker_health", {}))
                for entry in final_entries
            ],
            "reserved_cpu_ids": reserved_cpu_ids,
        },
        "counts": {
            "model_calls": 0,
            "app_server_calls": 0,
            "oracle_score_calls": 0,
            "episodes": len(ordered),
        },
        "phase_timings_ns": {"evaluation": time.perf_counter_ns() - started},
        "fitness_immutable": pass_name in {"primary", "replay"},
        "shards": final_entries,
    }
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return {**summary, "records": ordered, "shard_manifest": shard_manifest}


def evaluate_policy_roster_manifest(
    config: object,
    manifest: object,
    policies: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    workers: int = 8,
    shard_count: int = 8,
    pass_name: str = "primary",
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate a complete policy roster once per episode."""
    if "random" not in policies or "structural" not in policies:
        raise ValueError("random and structural policies are required")
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policy IDs must be unique")
    identities = [_identity(value) for value in policies.values()]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate policy source identity")
    sandbox = _get(_get(config, "stage2b", config), "sandbox")
    if sandbox is not None:
        ast_hashes: list[str] = []
        for value in policies.values():
            source = _source(value)
            if isinstance(source, str):
                checked = validate_policy(source, sandbox)
                if not checked.valid:
                    raise ValueError("invalid policy source")
                ast_hash = checked.identity.normalized_ast_sha256
                if ast_hash is not None:
                    ast_hashes.append(ast_hash)
        if len(ast_hashes) != len(set(ast_hashes)):
            raise ValueError("duplicate normalized policy AST")
    rows = _episodes(manifest)
    if not rows:
        raise ValueError("manifest roster cannot be empty")
    if workers < 1 or workers > MAX_WORKERS or shard_count < 1 or shard_count > MAX_WORKERS:
        raise ValueError("workers and shard_count must be between 1 and 8")
    roster_key = sha256(
        [
            "stage4.roster-cache.v1",
            [[name, _identity(value)] for name, value in sorted(policies.items())],
            _manifest_hash(manifest),
            _config_hash(config),
            pass_name,
        ]
    )
    root = Path(output_dir or _get(config, "run_root", "."))
    manifest_path = root / f"stage4-{roster_key}-{pass_name}-shards.json"
    summary_path = root / f"stage4-{roster_key}-{pass_name}-summary.json"
    shards = _shard(rows, shard_count)
    worker_cpu_ids, reserved_cpu_ids = _cpu_plan(config, workers)
    records_by_shard: list[list[dict[str, Any]] | None] = [None] * shard_count
    entries: list[dict[str, Any] | None] = [None] * shard_count
    if resume and manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            for index, entry in enumerate(raw.get("shards", [])):
                if index >= shard_count or not isinstance(entry, Mapping):
                    continue
                path = root / str(entry.get("path", ""))
                valid = _load_valid_shard(
                    path, {str(row["episode_id"]) for row in shards[index]}, entry
                )
                if valid is not None:
                    records_by_shard[index], entries[index] = valid, dict(entry)
        except Exception:
            pass

    def run_shard(
        index: int,
        shard_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        observed = _set_threads(worker_cpu_ids[index % workers])
        compact: list[dict[str, Any]] = []
        for episode in shard_rows:
            raw = run_development_episode(
                config, episode, {name: _source(value) for name, value in policies.items()}
            )
            if any(
                raw.get(key, 0)
                for key in (
                    "model_calls",
                    "app_server_calls",
                    "oracle_score_calls",
                    "runtime_network_calls",
                )
            ):
                raise ValueError(
                    "Stage 4 evaluation crossed a forbidden provider/oracle/network boundary"
                )
            shared = {
                key: _strip(value)
                for key, value in raw.items()
                if key not in {"policies", "policy_identities", "steps", "canonical_episode_sha256"}
            }
            summaries = raw.get("policies", {})
            summaries_map = summaries if isinstance(summaries, Mapping) else {}
            shared["policies"] = {
                str(name): {
                    key: _strip(value) for key, value in summary.items() if key != "final_score"
                }
                for name, summary in summaries_map.items()
                if isinstance(summary, Mapping)
            }
            shared["candidate_id"] = "roster"
            shared["metrics_input"] = {
                "episode_id": episode["episode_id"],
                "policies": shared["policies"],
            }
            shared["canonical_episode_sha256"] = sha256(canonical_projection(raw))
            compact.append(shared)
        return compact, {
            "shard": index,
            "cpu_id": worker_cpu_ids[index % workers],
            "observed_affinity": observed,
            "reserved_cpu_ids": reserved_cpu_ids,
            "thread_environment": dict(THREAD_ENVIRONMENT),
        }

    missing = [index for index, value in enumerate(records_by_shard) if value is None]
    if missing:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stage4-roster") as pool:
            futures = {
                pool.submit(run_shard, index, shards[index]): index for index in missing
            }
            for future in as_completed(futures):
                index = futures[future]
                records_by_shard[index], health = future.result()
                entries[index] = {"worker_health": health}
    for index, records in enumerate(records_by_shard):
        assert records is not None
        path = root / f"stage4-{roster_key}-{pass_name}-shard-{index:02d}.jsonl.gz"
        existing_entry = entries[index]
        if existing_entry is None or "path" not in existing_entry:
            worker_health = (
                cast(Mapping[str, Any], existing_entry.get("worker_health", {}))
                if existing_entry is not None
                else {}
            )
            entry = _write_jsonl(path, records)
            entry["worker_health"] = dict(worker_health)
            entries[index] = entry
    final_entries = [cast(dict[str, Any], entry) for entry in entries]
    ordered = sorted(
        [record for shard in records_by_shard if shard for record in shard],
        key=lambda record: str(record["episode_id"]),
    )
    expected_episode_ids = [str(row["episode_id"]) for row in rows]
    observed_episode_ids = [str(record.get("episode_id", "")) for record in ordered]
    if (
        observed_episode_ids != expected_episode_ids
        or len(set(observed_episode_ids)) != len(expected_episode_ids)
    ):
        raise ValueError("evaluation output does not match authoritative episode roster")
    reduction = sha256([record["canonical_episode_sha256"] for record in ordered])
    metrics = sha256(
        [record.get("metrics_input", metrics_input_projection(record)) for record in ordered]
    )
    manifest_shards = [[str(row["episode_id"]) for row in shard] for shard in shards]
    summary = {
        "schema_version": "stage4.evaluation.v1",
        "candidate_id": "roster",
        "candidate_cache_key": roster_key,
        "artifact_dir": str(root),
        "manifest_sha256": _manifest_hash(manifest),
        "config_sha256": _config_hash(config),
        "pass": pass_name,
        "record_count": len(ordered),
        "shard_count": shard_count,
        "manifest_episode_ids": [str(row["episode_id"]) for row in rows],
        "manifest_shards": manifest_shards,
        "canonical_reduction_sha256": reduction,
        "metrics_input_sha256": metrics,
        "worker_health": {
            "requested": workers,
            "effective": workers,
            "completed_shards": shard_count,
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "affinity_supported": hasattr(os, "sched_setaffinity"),
            "assignments": [
                cast(Mapping[str, Any], entry.get("worker_health", {}))
                for entry in final_entries
            ],
            "reserved_cpu_ids": reserved_cpu_ids,
        },
        "counts": {
            "model_calls": 0,
            "app_server_calls": 0,
            "oracle_score_calls": 0,
            "episodes": len(ordered),
        },
        "fitness_immutable": pass_name in {"primary", "replay"},
        "shards": final_entries,
    }
    manifest_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return {**summary, "records": ordered}


def verify_candidate_pass(
    value: Mapping[str, Any] | str | Path,
    manifest: object | None = None,
) -> dict[str, Any]:
    """Validate shard hashes, roster, reduction and metrics hashes."""
    try:
        raw = (
            dict(value)
            if isinstance(value, Mapping)
            else json.loads(Path(value).read_text(encoding="utf-8"))
        )
        root = (
            Path(value).parent
            if not isinstance(value, Mapping)
            else Path(str(raw.get("artifact_dir", ".")))
        )
        if raw.get("schema_version") != "stage4.evaluation.v1":
            raise ValueError("unexpected Stage 4 pass schema")
        declared_ids = [str(item) for item in raw.get("manifest_episode_ids", [])]
        declared_shards: Any = raw.get("manifest_shards")
        if manifest is not None:
            authoritative = _episodes(manifest)
            expected_ids = [str(row["episode_id"]) for row in authoritative]
            shard_count = int(raw.get("shard_count", 0))
            expected_shards: Any = [
                [str(row["episode_id"]) for row in shard]
                for shard in _shard(authoritative, shard_count)
            ]
        else:
            expected_ids = declared_ids
            expected_shards = declared_shards
            shard_count = int(raw.get("shard_count", 0))
        if not expected_ids or not isinstance(expected_shards, list):
            raise ValueError("authoritative manifest roster is missing or empty")
        if expected_ids != sorted(expected_ids) or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("authoritative manifest roster is invalid")
        if len(expected_shards) != shard_count:
            raise ValueError("manifest shard roster mismatch")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        entries = raw.get("shards")
        if not isinstance(entries, list) or len(entries) != shard_count:
            raise ValueError("evaluation shard count mismatch")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError("invalid evaluation shard entry")
            path = root / str(entry["path"])
            shard = _load_valid_shard(path, set(expected_shards[index]), entry)
            if shard is None:
                raise ValueError(f"invalid shard {path.name}")
            if [str(item.get("episode_id")) for item in shard] != expected_shards[index]:
                raise ValueError("shard episode membership/order mismatch")
            for record in shard:
                if str(record["episode_id"]) in seen:
                    raise ValueError("duplicate episode record")
                seen.add(str(record["episode_id"]))
                records.append(record)
        ordered = sorted(records, key=lambda record: str(record["episode_id"]))
        if (
            len(ordered) != len(expected_ids)
            or [str(item["episode_id"]) for item in ordered] != expected_ids
        ):
            raise ValueError("missing or extra episode records")
        reduction = sha256([record["canonical_episode_sha256"] for record in ordered])
        metrics = sha256([record["metrics_input"] for record in ordered])
        if reduction != raw.get("canonical_reduction_sha256") or metrics != raw.get(
            "metrics_input_sha256"
        ):
            raise ValueError("canonical reduction or metrics hash mismatch")
        return {
            "status": "completed",
            "exact": True,
            "record_count": len(ordered),
            "canonical_reduction_sha256": reduction,
            "metrics_input_sha256": metrics,
        }
    except Exception as error:
        return {"status": "failed", "exact": False, "error": f"{type(error).__name__}: {error}"}


def evaluate_validation_manifest(
    config: object,
    manifest: object,
    candidates: Mapping[str, Any],
    *,
    search_manifest: object | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate only the four frozen validation policies on a disjoint split."""
    if search_manifest is not None:
        search_ids = {str(row["episode_id"]) for row in _episodes(search_manifest)}
        validation_ids = {str(row["episode_id"]) for row in _episodes(manifest)}
        if search_ids & validation_ids:
            raise ValueError("validation manifest overlaps search manifest")
    if set(candidates) != VALIDATION_ALLOWLIST:
        raise ValueError("validation candidates must be exactly the frozen allowlist")
    baselines = {name: candidates[name] for name in ("random", "structural")}
    return {
        name: evaluate_program_manifest(
            config,
            manifest,
            name,
            candidates[name],
            baselines=baselines,
            pass_name="validation",
            **kwargs,
        )
        for name in sorted(candidates)
    }


def verify_replay(primary: Any, replay: Any) -> dict[str, Any]:
    from .replay import verify_replay as _verify

    return _verify(primary, replay)


__all__ = [
    "candidate_cache_key",
    "evaluate_program_manifest",
    "evaluate_policy_roster_manifest",
    "evaluate_validation_manifest",
    "metrics_input_projection",
    "verify_candidate_pass",
    "verify_replay",
]
