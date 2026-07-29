from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mutation_forge.models import JsonValue
from mutation_forge.stage2d.config import Stage2DConfig

STAGE2D_MANIFEST_VERSION = "stage2d.manifest.v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CpuCore:
    cpu_id: int
    core_id: int
    socket_id: int
    node_id: int

    @property
    def physical_id(self) -> str:
        return f"{self.socket_id}:{self.core_id}"

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "cpu_id": self.cpu_id,
            "core_id": self.core_id,
            "socket_id": self.socket_id,
            "node_id": self.node_id,
            "physical_id": self.physical_id,
        }


def read_cpu_topology() -> tuple[CpuCore, ...]:
    result = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    cores: dict[tuple[int, int], CpuCore] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 5 or fields[4].upper() != "Y":
            continue
        cpu_id, core_id, socket_id, node_id = (int(item) for item in fields[:4])
        key = (socket_id, core_id)
        candidate = CpuCore(cpu_id, core_id, socket_id, node_id)
        existing = cores.get(key)
        if existing is None or candidate.cpu_id < existing.cpu_id:
            cores[key] = candidate
    if not cores:
        raise RuntimeError("lscpu did not report any online physical CPU cores")
    return tuple(
        sorted(
            cores.values(),
            key=lambda item: (item.socket_id, item.core_id, item.cpu_id),
        )
    )


def _logical_cpu_count() -> int:
    result = subprocess.run(
        ["lscpu", "-p=CPU"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return sum(
        1
        for raw_line in result.stdout.splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
    )


def _episode_id(order: int, graph_seed: int, policy_seed: int) -> str:
    return f"o{order:02d}-g{graph_seed:04d}-p{policy_seed:04d}"


def build_manifest(config: Stage2DConfig) -> dict[str, JsonValue]:
    topology = read_cpu_topology()
    physical_count = len(topology)
    concurrency = min(
        config.resources.max_concurrent_shards,
        max(1, physical_count - config.resources.minimum_reserved_physical_cores),
        max(1, physical_count // 2),
    )
    if physical_count < 2:
        raise RuntimeError("Stage 2D requires at least two physical CPU cores")
    assigned_cores = topology[:concurrency]
    episodes: list[dict[str, JsonValue]] = []
    shard_episodes: dict[str, list[str]] = {
        f"shard-{index:02d}": []
        for index in range(config.experiment.shard_count)
    }
    for index, (order, graph_seed, policy_seed) in enumerate(
        (order, graph_seed, policy_seed)
        for order in config.experiment.orders
        for graph_seed in config.experiment.graph_seeds
        for policy_seed in config.experiment.policy_seeds
    ):
        shard_id = f"shard-{index % config.experiment.shard_count:02d}"
        episode_id = _episode_id(order, graph_seed, policy_seed)
        episodes.append(
            {
                "episode_id": episode_id,
                "order": order,
                "graph_seed": graph_seed,
                "policy_seed": policy_seed,
                "horizon": config.experiment.horizon,
                "shard_id": shard_id,
            }
        )
        shard_episodes[shard_id].append(episode_id)
    shards: list[dict[str, JsonValue]] = []
    for index, (shard_id, episode_ids) in enumerate(sorted(shard_episodes.items())):
        core = assigned_cores[index % len(assigned_cores)]
        assignment: dict[str, JsonValue] = {
            "shard_id": shard_id,
            "episode_ids": cast(list[JsonValue], episode_ids),
            "episode_count": len(episode_ids),
            "affinity": core.as_dict(),
            "assignment_sha256": _hash(
                {"shard_id": shard_id, "episode_ids": episode_ids}
            ),
        }
        shards.append(assignment)
    base: dict[str, JsonValue] = {
        "schema_version": STAGE2D_MANIFEST_VERSION,
        "config_sha256": config.stable_hash(),
        "episode_count": len(episodes),
        "shard_count": len(shards),
        "episodes_per_shard": config.experiment.episodes_per_shard,
        "episodes": cast(list[JsonValue], episodes),
        "shards": cast(list[JsonValue], shards),
        "cpu_topology": {
            "physical_core_count": physical_count,
            "logical_cpu_count": _logical_cpu_count(),
            "recommended_concurrency": concurrency,
            "minimum_reserved_physical_cores": (
                config.resources.minimum_reserved_physical_cores
            ),
            "reserved_physical_cores": physical_count - concurrency,
            "selected_cores": [core.as_dict() for core in assigned_cores],
        },
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
    }
    return {**base, "manifest_sha256": _hash(base)}


def validate_manifest(
    config: Stage2DConfig,
    manifest: dict[str, JsonValue],
) -> None:
    manifest_hash = manifest.get("manifest_sha256")
    base = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != STAGE2D_MANIFEST_VERSION:
        raise ValueError("unexpected Stage 2D manifest schema")
    if manifest_hash != _hash(base):
        raise ValueError("Stage 2D manifest hash mismatch")
    if manifest.get("config_sha256") != config.stable_hash():
        raise ValueError("Stage 2D manifest config hash mismatch")
    episodes = cast(list[dict[str, JsonValue]], manifest.get("episodes"))
    shards = cast(list[dict[str, JsonValue]], manifest.get("shards"))
    expected_count = (
        len(config.experiment.orders)
        * len(config.experiment.graph_seeds)
        * len(config.experiment.policy_seeds)
    )
    if len(episodes) != expected_count or len(shards) != config.experiment.shard_count:
        raise ValueError("Stage 2D manifest cardinality mismatch")
    ids = [cast(str, episode["episode_id"]) for episode in episodes]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Stage 2D episode IDs must be canonical and unique")
    expected_episodes: list[dict[str, JsonValue]] = []
    for index, (order, graph_seed, policy_seed) in enumerate(
        (order, graph_seed, policy_seed)
        for order in config.experiment.orders
        for graph_seed in config.experiment.graph_seeds
        for policy_seed in config.experiment.policy_seeds
    ):
        expected_episodes.append(
            {
                "episode_id": _episode_id(order, graph_seed, policy_seed),
                "order": order,
                "graph_seed": graph_seed,
                "policy_seed": policy_seed,
                "horizon": config.experiment.horizon,
                "shard_id": f"shard-{index % config.experiment.shard_count:02d}",
            }
        )
    if episodes != expected_episodes:
        raise ValueError("Stage 2D manifest episode coverage mismatch")
    assigned: list[str] = []
    for index, shard in enumerate(shards):
        expected_shard_id = f"shard-{index:02d}"
        if shard.get("shard_id") != expected_shard_id:
            raise ValueError("Stage 2D shard IDs are not canonical")
        episode_ids = cast(list[str], shard.get("episode_ids"))
        if len(episode_ids) != config.experiment.episodes_per_shard:
            raise ValueError("Stage 2D shard size mismatch")
        if shard.get("episode_count") != len(episode_ids):
            raise ValueError("Stage 2D shard episode count mismatch")
        if shard.get("assignment_sha256") != _hash(
            {"shard_id": expected_shard_id, "episode_ids": episode_ids}
        ):
            raise ValueError("Stage 2D shard assignment hash mismatch")
        assigned.extend(episode_ids)
    if sorted(assigned) != ids or len(assigned) != len(set(assigned)):
        raise ValueError("Stage 2D manifest is not an exact partition")
    topology = cast(dict[str, JsonValue], manifest.get("cpu_topology"))
    if (
        cast(int, topology.get("recommended_concurrency")) < 1
        or cast(int, topology.get("recommended_concurrency"))
        > config.resources.max_concurrent_shards
        or cast(int, topology.get("reserved_physical_cores"))
        < config.resources.minimum_reserved_physical_cores
    ):
        raise ValueError("Stage 2D manifest CPU reservation mismatch")
    affinity_ids = {
        cast(
            int,
            cast(dict[str, JsonValue], shard["affinity"])["cpu_id"],
        )
        for shard in shards
    }
    if len(affinity_ids) < cast(int, topology["recommended_concurrency"]):
        raise ValueError("Stage 2D shard affinity assignments are not distinct")
    expected_threads: dict[str, JsonValue] = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    if manifest.get("thread_environment") != expected_threads:
        raise ValueError("Stage 2D thread environment mismatch")


def load_manifest(config: Stage2DConfig, path: Path | None = None) -> dict[str, JsonValue]:
    source = path or config.inputs.manifest
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Stage 2D manifest must be an object")
    manifest = cast(dict[str, JsonValue], payload)
    validate_manifest(config, manifest)
    return manifest


def write_manifest(config: Stage2DConfig, destination: Path | None = None) -> dict[str, JsonValue]:
    path = destination or config.inputs.manifest
    manifest = build_manifest(config)
    validate_manifest(config, manifest)
    encoded = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("immutable Stage 2D manifest already exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)
    return manifest
