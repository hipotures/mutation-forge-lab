"""Provider-free, resumable Stage 5 execution and replay verification."""
# ruff: noqa: E501

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

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import GraphState
from mutation_forge.stage2d.manifest import read_cpu_topology
from mutation_forge.stage3.evaluation import THREAD_ENVIRONMENT, run_development_episode
from mutation_forge.stage3.manifest import canonical_bytes, sha256

from .stage5_relabel import make_relabel_proof, relabel_graph

SCHEMA_VERSION = "stage5.generalization.execution.v1"
SHARD_COUNT = 24
EPISODES_PER_SHARD = 64
EPISODE_COUNT = SHARD_COUNT * EPISODES_PER_SHARD
MAX_WORKERS = 8
MAX_UNCOMPRESSED_SHARD_BYTES = 32 * 1024 * 1024
FORBIDDEN_COUNTERS = (
    "model_calls",
    "app_server_calls",
    "oracle_score_calls",
    "runtime_network_calls",
)
TIMING_ONLY_FIELDS = frozenset(
    {
        "timing_ns",
        "first_improvement_ns",
        "ranker_elapsed_ns",
        "selected_scoring_ns",
        "pool_legality_ns",
        "pool_feature_ns",
        "elapsed_ns",
        "started_at",
        "finished_at",
    }
)


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def timing_stripped_projection(value: Any) -> Any:
    """Recursively remove only the frozen timing-only fields."""
    if isinstance(value, Mapping):
        return {
            str(key): timing_stripped_projection(item)
            for key, item in value.items()
            if str(key) not in TIMING_ONLY_FIELDS
            and str(key) not in {"timing", "timing_profile", "elapsed_seconds"}
            and not str(key).endswith("_ns")
        }
    if isinstance(value, (list, tuple)):
        return [timing_stripped_projection(item) for item in value]
    return value


def _episodes(manifest: object) -> list[dict[str, Any]]:
    rows = _get(manifest, "episodes", manifest)
    if isinstance(rows, Mapping):
        rows = rows.values()
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes)):
        raise ValueError("Stage 5 manifest must contain an episodes iterable")
    result = [dict(cast(Mapping[str, Any], row)) for row in rows]
    if len(result) != EPISODE_COUNT:
        raise ValueError(f"Stage 5 manifest must contain exactly {EPISODE_COUNT} episodes")
    ids = [str(row.get("episode_id", "")) for row in result]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Stage 5 manifest episode IDs must be unique and non-empty")
    required = {"order", "graph_seed", "relabeling_seed", "policy_seed", "horizon", "shard_id"}
    if any(not required.issubset(row) for row in result):
        raise ValueError("Stage 5 manifest episode row is incomplete")
    return sorted(
        result,
        key=lambda row: (
            int(row["order"]),
            int(row["graph_seed"]),
            int(row["relabeling_seed"]),
            int(row["policy_seed"]),
        ),
    )


def _manifest_shards(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {f"shard-{i:02d}": [] for i in range(SHARD_COUNT)}
    for row in rows:
        shard_id = str(row["shard_id"])
        if shard_id not in grouped:
            raise ValueError(f"unknown Stage 5 shard ID: {shard_id}")
        grouped[shard_id].append(row)
    shards = [grouped[f"shard-{i:02d}"] for i in range(SHARD_COUNT)]
    if any(len(shard) != EPISODES_PER_SHARD for shard in shards):
        raise ValueError("Stage 5 shard layout is not 24 shards of 64 episodes")
    return shards


def _manifest_hash(manifest: object, rows: list[dict[str, Any]]) -> str:
    declared = _get(manifest, "manifest_sha256")
    if isinstance(declared, str) and len(declared) == 64:
        return declared
    return sha256(rows)


def _config_hash(config: object) -> str:
    stable = _get(config, "stable_hash")
    if callable(stable):
        value = stable()
        if isinstance(value, str) and len(value) == 64:
            return value
    resolved = _get(config, "resolved_dict")
    if callable(resolved):
        return sha256(resolved())
    if isinstance(config, Mapping):
        return sha256(config)
    raise TypeError("Stage 5 config must provide stable_hash or resolved_dict")


def _policy_sources(policies: Mapping[str, str]) -> dict[str, str]:
    if len(policies) != 4:
        raise ValueError("Stage 5 requires exactly four frozen policy sources")
    result = {str(key): value for key, value in policies.items()}
    if any(not key or not isinstance(value, str) for key, value in result.items()):
        raise ValueError("Stage 5 policy IDs and sources must be non-empty strings")
    expected = {"program-d5ad1c8203e0d9f25f03aabd", "candidate-slot-04", "random", "structural"}
    if set(result) != expected:
        raise ValueError("Stage 5 policy roster differs from the freeze")
    return {key: result[key] for key in ("program-d5ad1c8203e0d9f25f03aabd", "candidate-slot-04", "random", "structural")}


def _identity(config: object, manifest: object, rows: list[dict[str, Any]], policies: Mapping[str, str]) -> tuple[str, dict[str, str], str, str]:
    sources = _policy_sources(policies)
    source_hashes = {name: hashlib.sha256(source.encode("utf-8")).hexdigest() for name, source in sources.items()}
    manifest_hash, config_hash = _manifest_hash(manifest, rows), _config_hash(config)
    identity = sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": manifest_hash,
            "config_sha256": config_hash,
            "policy_source_sha256": source_hashes,
            "policy_ids": list(sources),
            "relabeling_algorithm": "fisher-yates-sha256-v1",
        }
    )
    return identity, source_hashes, manifest_hash, config_hash


class RelabeledToyBackend(ToyBackend):
    """Toy backend that relabels one deterministic base graph before execution."""

    def __init__(self, *, order: int, graph_seed: int, relabeling_seed: int) -> None:
        super().__init__()
        self.order = order
        self.graph_seed = graph_seed
        self.relabeling_seed = relabeling_seed
        self.base_graph = super().generate_seed(order=order, seed=graph_seed)
        self.graph, self.permutation = relabel_graph(
            self.base_graph, graph_seed=graph_seed, relabeling_seed=relabeling_seed
        )
        self.proof = make_relabel_proof(
            self.base_graph,
            self.graph,
            graph_seed=graph_seed,
            relabeling_seed=relabeling_seed,
            permutation=self.permutation,
        )

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        if (order, seed) != (self.order, self.graph_seed):
            raise ValueError("Stage 5 backend received a foreign graph identity")
        return self.graph


def _compact_trace(value: Mapping[str, Any]) -> dict[str, Any]:
    kept = {
        key: value.get(key)
        for key in (
            "step",
            "pool_hash",
            "rank_pool_hash",
            "pool_size",
            "pool_attempted",
            "pool_deduplicated",
            "pool_rejected",
            "selected_proposal_id",
            "selected_k",
            "selected_operator_family",
            "selected_selector_tags",
            "accepted",
            "selected_ordering_key",
            "previous_ordering_key",
            "selected_total_witnesses",
            "previous_total_witnesses",
            "current_total_witnesses",
            "selected_witness_delta",
            "selected_penalty_delta",
            "best_total_witnesses",
            "state_hash",
            "ranker_flags",
        )
    }
    return cast(dict[str, Any], timing_stripped_projection(kept))


def _compact_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "auc",
        "normalized_best_so_far_curve",
        "best_total_witnesses",
        "accepted_count",
        "rejected_count",
        "nonimproving_count",
        "divergence_count",
        "first_improvement_step",
        "evaluations_to_first_improvement",
        "failure_count",
    )
    return cast(dict[str, Any], timing_stripped_projection({key: value.get(key) for key in keys}))


def _metrics_input(record: Mapping[str, Any], policy_ids: list[str]) -> dict[str, Any]:
    policies = cast(Mapping[str, Mapping[str, Any]], record["policies"])
    return {
        "episode_id": record["episode_id"],
        "order": record["order"],
        "graph_seed": record["graph_seed"],
        "relabeling_seed": record["relabeling_seed"],
        "policy_seed": record["policy_seed"],
        "horizon": record["horizon"],
        "policies": {
            name: {
                key: policies[name].get(key)
                for key in (
                    "auc",
                    "normalized_best_so_far_curve",
                    "best_total_witnesses",
                    "accepted_count",
                    "failure_count",
                )
            }
            for name in policy_ids
        },
    }


def _compact_record(
    raw: Mapping[str, Any],
    episode: Mapping[str, Any],
    policy_ids: list[str],
    source_hashes: Mapping[str, str],
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    if str(raw.get("episode_id", "")) != str(episode["episode_id"]):
        raise ValueError("Stage 5 evaluator episode ID mismatch")
    if raw.get("terminal_status") != "completed":
        raise ValueError("Stage 5 evaluator returned a non-completed episode")
    policies = raw.get("policies")
    if not isinstance(policies, Mapping) or set(str(key) for key in policies) != set(policy_ids):
        raise ValueError("Stage 5 evaluator policy roster drifted")
    for counter in FORBIDDEN_COUNTERS:
        if int(raw.get(counter, 0)) != 0:
            raise ValueError(f"forbidden Stage 5 counter is nonzero: {counter}")
    if int(raw.get("invalid_graphs", 0)) != 0 or int(raw.get("policy_failures", 0)) != 0:
        raise ValueError("Stage 5 episode contains an invalid graph or policy failure")
    base = {
        "schema_version": SCHEMA_VERSION,
        "terminal_status": "completed",
        "episode_id": episode["episode_id"],
        "order": episode["order"],
        "graph_seed": episode["graph_seed"],
        "relabeling_seed": episode["relabeling_seed"],
        "policy_seed": episode["policy_seed"],
        "horizon": episode["horizon"],
        "initial_graph_hash": raw.get("initial_graph_hash"),
        "base_graph_hash": proof.get("base_graph_hash"),
        "relabeled_graph_hash": proof.get("relabeled_graph_hash"),
        "canonical_unlabeled_hash": proof.get("canonical_unlabeled_hash"),
        "relabel_proof": dict(proof),
        "divergence_step": raw.get("divergence_step"),
        "shared_pool_steps": raw.get("shared_pool_steps"),
        "independent_pool_steps": raw.get("independent_pool_steps"),
        "policies": {
            policy_id: _compact_policy(cast(Mapping[str, Any], policies[policy_id]))
            for policy_id in policy_ids
        },
        "steps": [
            {
                "step": item.get("step"),
                "trajectory_seed": item.get("trajectory_seed"),
                "states_identical_before_step": item.get("states_identical_before_step"),
                "shared_pool": item.get("shared_pool"),
                "policies": {
                    policy_id: _compact_trace(cast(Mapping[str, Any], cast(Mapping[str, Any], item["policies"])[policy_id]))
                    for policy_id in policy_ids
                },
            }
            for item in cast(list[Mapping[str, Any]], raw.get("steps", []))
        ],
        "initial_score_calls": int(raw.get("initial_score_calls", 0)),
        "selected_score_calls": int(raw.get("selected_score_calls", 0)),
        "evaluation_count": int(raw.get("evaluation_count", 0)),
        "invalid_graphs": int(raw.get("invalid_graphs", 0)),
        "policy_failures": int(raw.get("policy_failures", 0)),
        "model_calls": int(raw.get("model_calls", 0)),
        "app_server_calls": int(raw.get("app_server_calls", 0)),
        "oracle_score_calls": int(raw.get("oracle_score_calls", 0)),
        "runtime_network_calls": int(raw.get("runtime_network_calls", 0)),
        "policy_source_sha256": dict(source_hashes),
    }
    base["metrics_input"] = _metrics_input(base, policy_ids)
    base["canonical_episode_sha256"] = _canonical_episode_hash(base)
    return base


def _canonical_episode_hash(record: Mapping[str, Any]) -> str:
    without_hash = {key: value for key, value in record.items() if key != "canonical_episode_sha256"}
    return hashlib.sha256(canonical_bytes(timing_stripped_projection(without_hash))).hexdigest()


def _set_threads(cpu_id: int | None) -> list[int]:
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    if cpu_id is None:
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Linux CPU affinity is required for Stage 5")
    os.sched_setaffinity(0, {cpu_id})
    observed = sorted(os.sched_getaffinity(0))
    if observed != [cpu_id]:
        raise RuntimeError("Stage 5 worker affinity mismatch")
    return observed


def _cpu_plan(config: object, workers: int) -> tuple[list[int | None], list[int]]:
    reserved = int(_get(_get(config, "limits", {}), "reserved_physical_cores", 0))
    if reserved == 0:
        return cast(list[int | None], [None] * workers), []
    topology = read_cpu_topology()
    if len(topology) < workers + reserved:
        raise RuntimeError("cannot preserve the frozen Stage 5 physical-core reserve")
    workers_ids: list[int | None] = [topology[index].cpu_id for index in range(workers)]
    reserved_ids = [topology[index].cpu_id for index in range(workers, workers + reserved)]
    if set(workers_ids) & set(reserved_ids):
        raise RuntimeError("Stage 5 worker and reserve CPU sets overlap")
    return workers_ids, reserved_ids


def _evaluate_shard(
    config: object,
    rows: list[dict[str, Any]],
    policy_sources: Mapping[str, str],
    source_hashes: Mapping[str, str],
    cpu_id: int | None,
    reserved_cpu_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = _set_threads(cpu_id)
    policy_ids = list(policy_sources)
    records: list[dict[str, Any]] = []
    for episode in rows:
        backend = RelabeledToyBackend(
            order=int(episode["order"]),
            graph_seed=int(episode["graph_seed"]),
            relabeling_seed=int(episode["relabeling_seed"]),
        )
        raw = run_development_episode(
            config,
            episode,
            policy_sources,
            backend=backend,
            require_baselines=True,
        )
        record = _compact_record(raw, episode, policy_ids, source_hashes, backend.proof.as_dict())
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
    if len(payload) > MAX_UNCOMPRESSED_SHARD_BYTES:
        raise ValueError("Stage 5 shard exceeds the 32 MiB uncompressed artifact limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as out:
        out.write(payload)
    temporary.replace(path)
    return {
        "path": path.name,
        "record_count": len(records),
        "episode_ids": [str(record["episode_id"]) for record in records],
        "uncompressed_bytes": len(payload),
        "compressed_bytes": path.stat().st_size,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _read_shard(path: Path, entry: Mapping[str, Any], expected_ids: list[str]) -> list[dict[str, Any]]:
    if not path.is_file() or path.name != str(entry.get("path")):
        raise ValueError("Stage 5 shard path is missing")
    if hashlib.sha256(path.read_bytes()).hexdigest() != str(entry.get("file_sha256")):
        raise ValueError("Stage 5 shard hash mismatch")
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Stage 5 shard row must be an object")
                records.append(cast(dict[str, Any], row))
    ids = [str(record.get("episode_id", "")) for record in records]
    if ids != expected_ids or len(records) != int(entry.get("record_count", -1)):
        raise ValueError("Stage 5 shard roster mismatch")
    for record in records:
        if str(record.get("canonical_episode_sha256", "")) != _canonical_episode_hash(record):
            raise ValueError("Stage 5 canonical episode hash mismatch")
    return records


def _paths(root: Path, identity: str, pass_name: str) -> tuple[Path, Path]:
    prefix = f"stage5-{identity[:24]}-{pass_name}"
    return root / f"{prefix}-state.json", root / f"{prefix}-summary.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "status": "prepared", "completed_shards": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Stage 5 state schema mismatch")
    completed = value.get("completed_shards", {})
    if not isinstance(completed, dict):
        raise ValueError("Stage 5 state completed_shards is invalid")
    return cast(dict[str, Any], value)


def _write_state(path: Path, identity: str, pass_name: str, entries: Mapping[str, Any], status: str) -> None:
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "run_identity_sha256": identity,
            "pass": pass_name,
            "completed_shards": dict(entries),
        },
    )


def _load_completed_records(
    root: Path,
    entries: Mapping[str, Any],
    shards: list[list[dict[str, Any]]],
    indices: Iterable[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in indices:
        entry = entries.get(str(index))
        if not isinstance(entry, Mapping):
            raise ValueError(f"Stage 5 shard {index:02d} has no persisted entry")
        records.extend(_read_shard(root / str(entry["path"]), entry, [str(row["episode_id"]) for row in shards[index]]))
    return records


def _summary(
    root: Path,
    identity: str,
    manifest_hash: str,
    config_hash: str,
    source_hashes: Mapping[str, str],
    rows: list[dict[str, Any]],
    shards: list[list[dict[str, Any]]],
    entries: Mapping[str, Any],
    records: list[dict[str, Any]],
    worker_health: list[Mapping[str, Any]],
    elapsed_ns: int,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: str(row["episode_id"]))
    expected = [str(row["episode_id"]) for row in rows]
    if [str(row["episode_id"]) for row in ordered] != expected:
        raise ValueError("Stage 5 reduction roster drifted")
    canonical_rows = [timing_stripped_projection(row) for row in ordered]
    policy_rows = [
        {"episode_id": row["episode_id"], "policies": row["policies"]}
        for row in ordered
    ]
    selected = [
        {"episode_id": row["episode_id"], "steps": row["steps"]}
        for row in ordered
    ]
    assignments: list[Mapping[str, Any]] = []
    for index in range(SHARD_COUNT):
        entry = entries[str(index)]
        recorded = entry.get("worker_health") if isinstance(entry, Mapping) else None
        if isinstance(recorded, Mapping):
            assignments.append(
                {
                    "shard": index,
                    "requested": int(entry.get("requested_workers", recorded.get("requested", 0))),
                    **dict(recorded),
                }
            )
    if not assignments:
        assignments = list(worker_health)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "pass": "primary" if "primary" in str(root) else "replay",
        "artifact_dir": str(root),
        "run_identity_sha256": identity,
        "manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "policy_source_sha256": dict(source_hashes),
        "policy_ids": list(source_hashes),
        "record_count": len(ordered),
        "shard_count": SHARD_COUNT,
        "episodes_per_shard": EPISODES_PER_SHARD,
        "manifest_episode_ids": expected,
        "manifest_shards": [[str(row["episode_id"]) for row in shard] for shard in shards],
        "shard_hashes_sha256": sha256([entries[str(index)]["file_sha256"] for index in range(SHARD_COUNT)]),
        "canonical_reduction_sha256": sha256([row["canonical_episode_sha256"] for row in ordered]),
        "timing_stripped_reduction_sha256": sha256(canonical_rows),
        "metrics_input_sha256": sha256([row["metrics_input"] for row in ordered]),
        "policy_rows_sha256": sha256(policy_rows),
        "selected_plan_sha256": sha256(selected),
        "counts": {
            **{counter: sum(int(row.get(counter, 0)) for row in ordered) for counter in FORBIDDEN_COUNTERS},
            "invalid_graphs": sum(int(row.get("invalid_graphs", 0)) for row in ordered),
            "policy_failures": sum(int(row.get("policy_failures", 0)) for row in ordered),
            "episodes": len(ordered),
        },
        "worker_health": {
            "requested": max((int(item.get("requested", 0)) for item in assignments), default=0),
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "assignments": assignments,
        },
        "phase_timings_ns": {"evaluation": elapsed_ns},
        "shards": [entries[str(index)] for index in range(SHARD_COUNT)],
    }


def execute_stage5_pass(
    config: object,
    manifest: object,
    policies: Mapping[str, str],
    output_dir: str | Path,
    pass_name: str,
    *,
    workers: int = MAX_WORKERS,
    shard_indices: Iterable[int] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    if not pass_name or "/" in pass_name or "\\" in pass_name:
        raise ValueError("Stage 5 pass name must be a safe filename component")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("Stage 5 workers must be between 1 and 8")
    rows, source_map = _episodes(manifest), _policy_sources(policies)
    shards = _manifest_shards(rows)
    identity, source_hashes, manifest_hash, config_hash = _identity(config, manifest, rows, source_map)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state_path, summary_path = _paths(root, identity, pass_name)
    state = _load_state(state_path) if resume else {"completed_shards": {}}
    entries = cast(dict[str, Any], state.get("completed_shards", {}))
    if summary_path.is_file() and resume:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, dict) and summary.get("status") == "completed":
            verify = verify_stage5_pass(summary_path, manifest)
            if verify["exact"]:
                return {**summary, "records": _load_completed_records(root, entries, shards, range(SHARD_COUNT))}
            raise ValueError("existing Stage 5 summary failed verification")
    requested = list(range(SHARD_COUNT)) if shard_indices is None else sorted(set(int(i) for i in shard_indices))
    if any(index < 0 or index >= SHARD_COUNT for index in requested):
        raise ValueError("Stage 5 shard index is outside the frozen layout")
    # Validate all previously completed shard artifacts before deciding what to resume.
    for key, entry in list(entries.items()):
        index = int(key)
        if index not in range(SHARD_COUNT) or not isinstance(entry, Mapping):
            raise ValueError("Stage 5 state contains an invalid completed shard")
        _read_shard(root / str(entry["path"]), entry, [str(row["episode_id"]) for row in shards[index]])
    missing = [index for index in requested if str(index) not in entries]
    worker_ids, reserved_ids = _cpu_plan(config, workers)
    _write_state(state_path, identity, pass_name, entries, "prepared")
    started = time.perf_counter_ns()
    health: list[Mapping[str, Any]] = []
    if missing:
        executor_type = ProcessPoolExecutor if int(_get(_get(config, "limits", {}), "reserved_physical_cores", 0)) > 0 else ThreadPoolExecutor
        with executor_type(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_shard,
                    config,
                    shards[index],
                    source_map,
                    source_hashes,
                    worker_ids[position % workers],
                    reserved_ids,
                ): index
                for position, index in enumerate(missing)
            }
            for future in as_completed(futures):
                index = futures[future]
                records, worker = future.result()
                path = root / f"stage5-{identity[:24]}-{pass_name}-shard-{index:02d}.jsonl.gz"
                entry = _write_shard(path, records)
                # Read-back is part of persistence acceptance, before state advances.
                _read_shard(path, entry, [str(row["episode_id"]) for row in shards[index]])
                entry["worker_health"] = worker
                entry["requested_workers"] = workers
                entries[str(index)] = entry
                health.append({"shard": index, "requested": workers, **worker})
                _write_state(state_path, identity, pass_name, entries, "shards_persisted")
    if len(entries) < SHARD_COUNT:
        records = _load_completed_records(root, entries, shards, sorted(int(key) for key in entries))
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "partial",
            "pass": pass_name,
            "run_identity_sha256": identity,
            "manifest_sha256": manifest_hash,
            "config_sha256": config_hash,
            "completed_shards": sorted(int(key) for key in entries),
            "records": sorted(records, key=lambda row: str(row["episode_id"])),
        }
    records = _load_completed_records(root, entries, shards, range(SHARD_COUNT))
    summary = _summary(root, identity, manifest_hash, config_hash, source_hashes, rows, shards, entries, records, health, time.perf_counter_ns() - started)
    _write_json(summary_path, summary)
    _write_state(state_path, identity, pass_name, entries, "completed")
    return {**summary, "records": sorted(records, key=lambda row: str(row["episode_id"]))}


def verify_stage5_pass(value: Mapping[str, Any] | str | Path, manifest: object | None = None) -> dict[str, Any]:
    try:
        if isinstance(value, Mapping):
            summary = dict(value)
            summary_path = None
        else:
            summary_path = Path(value)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping) or summary.get("schema_version") != SCHEMA_VERSION or summary.get("status") != "completed":
            raise ValueError("unexpected Stage 5 summary schema")
        root = Path(str(summary.get("artifact_dir", summary_path.parent if summary_path else ".")))
        if manifest is None:
            expected_ids = [str(item) for item in cast(list[Any], summary["manifest_episode_ids"])]
            rows = [{"episode_id": item} for item in expected_ids]
            shards = [[{"episode_id": item} for item in cast(list[Any], group)] for group in cast(list[Any], summary["manifest_shards"])]
        else:
            rows = _episodes(manifest)
            shards = _manifest_shards(rows)
            expected_ids = [str(row["episode_id"]) for row in rows]
        entries = summary.get("shards")
        if not isinstance(entries, list) or len(entries) != SHARD_COUNT:
            raise ValueError("Stage 5 summary shard count mismatch")
        records: list[dict[str, Any]] = []
        for index, entry in enumerate(cast(list[Mapping[str, Any]], entries)):
            records.extend(_read_shard(root / str(entry["path"]), entry, [str(row["episode_id"]) for row in shards[index]]))
        records.sort(key=lambda row: str(row["episode_id"]))
        if [str(row["episode_id"]) for row in records] != expected_ids:
            raise ValueError("Stage 5 summary roster mismatch")
        canonical_hash = sha256([row["canonical_episode_sha256"] for row in records])
        timing_hash = sha256([timing_stripped_projection(row) for row in records])
        metric_hash = sha256([row["metrics_input"] for row in records])
        shard_hash = sha256([entry["file_sha256"] for entry in entries])
        exact = (
            canonical_hash == summary.get("canonical_reduction_sha256")
            and timing_hash == summary.get("timing_stripped_reduction_sha256")
            and metric_hash == summary.get("metrics_input_sha256")
            and shard_hash == summary.get("shard_hashes_sha256")
            and int(summary.get("record_count", -1)) == len(records)
        )
        return {
            "status": "completed" if exact else "failed",
            "exact": exact,
            "record_count": len(records),
            "canonical_reduction_sha256": canonical_hash,
            "timing_stripped_reduction_sha256": timing_hash,
            "metrics_input_sha256": metric_hash,
            "shard_hashes_sha256": shard_hash,
        }
    except Exception as error:
        return {"status": "failed", "exact": False, "error": f"{type(error).__name__}: {error}"}


def _load_summary(value: Mapping[str, Any] | str | Path) -> tuple[dict[str, Any], Path]:
    if isinstance(value, Mapping):
        summary = dict(value)
        return summary, Path(str(summary.get("artifact_dir", ".")))
    path = Path(value)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8"))), path.parent


def _summary_records(summary: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    entries = cast(list[Mapping[str, Any]], summary["shards"])
    rows: dict[str, dict[str, Any]] = {}
    for entry in entries:
        with gzip.open(root / str(entry["path"]), "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows[str(row["episode_id"])] = cast(dict[str, Any], row)
    return rows


def _diff_paths(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[dict[str, Any]] = []
        keys = sorted({str(key) for key in left} | {str(key) for key in right})
        left_map = {str(key): item for key, item in left.items()}
        right_map = {str(key): item for key, item in right.items()}
        for key in keys:
            child = f"{path}.{key}"
            if key not in left_map or key not in right_map:
                differences.append({"path": child, "primary": left_map.get(key), "replay": right_map.get(key)})
            else:
                differences.extend(_diff_paths(left_map[key], right_map[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        list_differences: list[dict[str, Any]] = []
        if len(left) != len(right):
            list_differences.append({"path": f"{path}.length", "primary": len(left), "replay": len(right)})
        for index, pair in enumerate(zip(left, right, strict=False)):
            list_differences.extend(_diff_paths(pair[0], pair[1], f"{path}[{index}]"))
        return list_differences
    return [] if left == right else [{"path": path, "primary": left, "replay": right}]


def compare_timing_stripped_rows(primary: Mapping[str, Mapping[str, Any]], replay: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    primary_ids, replay_ids = set(primary), set(replay)
    differences: list[dict[str, Any]] = []
    row_hashes: list[dict[str, str]] = []
    for episode_id in sorted(primary_ids & replay_ids):
        left = timing_stripped_projection(primary[episode_id])
        right = timing_stripped_projection(replay[episode_id])
        row_hashes.append({"episode_id": episode_id, "sha256": hashlib.sha256(canonical_bytes(left)).hexdigest()})
        fields = _diff_paths(left, right)
        if fields:
            differences.append({"episode_id": episode_id, "fields": fields})
    return {
        "projection": {"name": "recursive_timing_stripped", "excluded_fields": sorted(TIMING_ONLY_FIELDS)},
        "primary_episode_count": len(primary_ids),
        "replay_episode_count": len(replay_ids),
        "missing_primary_episode_ids": sorted(replay_ids - primary_ids),
        "missing_replay_episode_ids": sorted(primary_ids - replay_ids),
        "non_timing_differences": differences,
        "canonical_row_hashes_sha256": sha256(row_hashes),
        "rows_exact": not (primary_ids ^ replay_ids) and not differences,
    }


def verify_stage5_replay(primary: Mapping[str, Any] | str | Path, replay: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    try:
        left_summary, left_root = _load_summary(primary)
        right_summary, right_root = _load_summary(replay)
        left_rows, right_rows = _summary_records(left_summary, left_root), _summary_records(right_summary, right_root)
        comparison = compare_timing_stripped_rows(left_rows, right_rows)
        primary_hash = sha256({"summary": left_summary.get("run_identity_sha256"), "rows": sorted((key, timing_stripped_projection(value)) for key, value in left_rows.items())})
        replay_hash = sha256({"summary": right_summary.get("run_identity_sha256"), "rows": sorted((key, timing_stripped_projection(value)) for key, value in right_rows.items())})
        exact = bool(comparison["rows_exact"]) and left_summary.get("manifest_shards") == right_summary.get("manifest_shards") and left_summary.get("policy_source_sha256") == right_summary.get("policy_source_sha256")
        return {
            "status": "completed" if exact else "failed",
            "decision": "exact" if exact else "mismatch",
            "exact": exact,
            "primary_sha256": primary_hash,
            "replay_sha256": replay_hash,
            "comparison": comparison,
            "provider_calls": 0,
        }
    except Exception as error:
        return {"status": "failed", "decision": "mismatch", "exact": False, "provider_calls": 0, "error": f"{type(error).__name__}: {error}"}


__all__ = [
    "EPISODE_COUNT",
    "EPISODES_PER_SHARD",
    "MAX_WORKERS",
    "SCHEMA_VERSION",
    "SHARD_COUNT",
    "compare_timing_stripped_rows",
    "execute_stage5_pass",
    "timing_stripped_projection",
    "verify_stage5_pass",
    "verify_stage5_replay",
]
