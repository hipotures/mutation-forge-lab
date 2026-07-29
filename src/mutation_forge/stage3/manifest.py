"""Immutable development-evaluation episode manifest.

The manifest deliberately contains no policy identifier.  Policy identities are
an input to an evaluation and the same 128 paired episodes are used for every
roster member.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, cast

from mutation_forge.models import JsonValue

STAGE3_MANIFEST_VERSION = "stage3.manifest.v1"
ORDERS = (10, 12)
GRAPH_SEEDS = tuple(range(301, 305))
POLICY_SEEDS = tuple(range(3001, 3017))
HORIZON = 32
SHARD_COUNT = 8
EPISODES_PER_SHARD = 16
THREAD_ENVIRONMENT = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CpuCore:
    cpu_id: int
    core_id: int
    socket_id: int = 0
    node_id: int = 0

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
    """Return one logical CPU per physical core; fail closed if unavailable."""
    try:
        process = subprocess.run(
            ["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("lscpu topology is required for Stage 3") from error
    cores: dict[tuple[int, int], CpuCore] = {}
    for raw in process.stdout.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split(",")
        if len(fields) != 5 or fields[4].upper() != "Y":
            continue
        cpu, core, socket, node = (int(x) for x in fields[:4])
        candidate = CpuCore(cpu, core, socket, node)
        key = (socket, core)
        if key not in cores or cpu < cores[key].cpu_id:
            cores[key] = candidate
    topology = tuple(sorted(cores.values(), key=lambda c: (c.socket_id, c.core_id, c.cpu_id)))
    if len(topology) < 16:
        raise RuntimeError("Stage 3 requires at least 16 physical cores")
    return topology


def _cfg(config: object, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def episode_id(order: int, graph_seed: int, policy_seed: int) -> str:
    return f"o{order:02d}-g{graph_seed:04d}-p{policy_seed:04d}"


def build_manifest(config: object | None = None) -> dict[str, JsonValue]:
    experiment = _cfg(config, "experiment", config)
    orders = tuple(_cfg(experiment, "orders", ORDERS))
    graph_seeds = tuple(_cfg(experiment, "graph_seeds", GRAPH_SEEDS))
    policy_seeds = tuple(_cfg(experiment, "policy_seeds", POLICY_SEEDS))
    horizon = int(_cfg(experiment, "horizon", HORIZON))
    shard_count = int(_cfg(experiment, "shard_count", SHARD_COUNT))
    if shard_count != SHARD_COUNT:
        raise ValueError("Stage 3 requires exactly eight shards")
    rows: list[dict[str, JsonValue]] = []
    shard_ids: dict[str, list[str]] = {f"shard-{i:02d}": [] for i in range(shard_count)}
    index = 0
    for order in orders:
        for graph_seed in graph_seeds:
            for policy_seed in policy_seeds:
                sid = f"shard-{index % shard_count:02d}"
                eid = episode_id(order, graph_seed, policy_seed)
                rows.append(
                    {
                        "episode_id": eid,
                        "order": order,
                        "graph_seed": graph_seed,
                        "policy_seed": policy_seed,
                        "horizon": horizon,
                        "shard_id": sid,
                    }
                )
                shard_ids[sid].append(eid)
                index += 1
    topology = read_cpu_topology()
    physical = len(topology)
    if physical < 16:
        raise RuntimeError("Stage 3 requires eight workers and eight reserved physical cores")
    concurrency = min(8, physical - 8, physical // 2)
    if concurrency < 8:
        raise RuntimeError("Stage 3 requires eight distinct worker cores")
    shards: list[dict[str, JsonValue]] = []
    for i in range(shard_count):
        sid = f"shard-{i:02d}"
        ids = shard_ids[sid]
        shards.append(
            {"shard_id": sid, "episode_ids": cast(list[JsonValue], ids), "episode_count": len(ids)}
        )
    base: dict[str, JsonValue] = {
        "schema_version": STAGE3_MANIFEST_VERSION,
        "episode_count": len(rows),
        "shard_count": shard_count,
        "episodes_per_shard": EPISODES_PER_SHARD,
        "episodes": cast(list[JsonValue], rows),
        "shards": cast(list[JsonValue], shards),
    }
    return {**base, "manifest_sha256": sha256(base)}


def validate_manifest(manifest: dict[str, JsonValue], config: object | None = None) -> None:
    if manifest.get("schema_version") != STAGE3_MANIFEST_VERSION:
        raise ValueError("unexpected Stage 3 manifest schema")
    if set(manifest) != {
        "schema_version",
        "episode_count",
        "shard_count",
        "episodes_per_shard",
        "episodes",
        "shards",
        "manifest_sha256",
    }:
        raise ValueError("manifest keys mismatch")
    expected_hash = sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("manifest hash mismatch")
    expected = build_manifest(config)
    for key in ("episode_count", "shard_count", "episodes_per_shard", "episodes", "shards"):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"manifest {key} mismatch")
    episodes = cast(list[dict[str, JsonValue]], manifest["episodes"])
    if any(
        set(e) != {"episode_id", "order", "graph_seed", "policy_seed", "horizon", "shard_id"}
        for e in episodes
    ):
        raise ValueError("episode keys mismatch")
    shards = cast(list[dict[str, JsonValue]], manifest["shards"])
    if any(set(s) != {"shard_id", "episode_ids", "episode_count"} for s in shards):
        raise ValueError("shard keys mismatch")
    if [str(e["episode_id"]) for e in episodes] != sorted(str(e["episode_id"]) for e in episodes):
        raise ValueError("episodes must be sorted by canonical ID")
    if len({str(e["episode_id"]) for e in episodes}) != len(episodes):
        raise ValueError("duplicate episode ID")


def write_manifest(
    path: str | os.PathLike[str], config: object | None = None
) -> dict[str, JsonValue]:
    from pathlib import Path

    value = build_manifest(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    if target.exists() and target.read_bytes() != encoded:
        raise ValueError("refusing to overwrite immutable manifest")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return value


def load_manifest(
    path: str | os.PathLike[str], config: object | None = None
) -> dict[str, JsonValue]:
    from pathlib import Path

    value = cast(dict[str, JsonValue], json.loads(Path(path).read_text(encoding="utf-8")))
    validate_manifest(value, config)
    return value
