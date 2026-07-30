"""Deterministic development manifest and frozen evaluation gates.

The manifest is intentionally non-held-out: all candidates consume the same
retained Stage-2D-style toy trajectories and equal budgets.  No generated
result is stored here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue

STAGE3_MANIFEST_VERSION = "stage3.manifest.v1"
ORDERS, GRAPH_SEEDS, POLICY_SEEDS = (10, 12), (301, 302, 303, 304), tuple(range(3001, 3017))
HORIZON, SHARD_COUNT, EPISODES_PER_SHARD = 32, 8, 16
GATE_SPEC: dict[str, JsonValue] = {
    "min_unique_valid_candidates": 4,
    "require_no_baseline_ast_duplicate": True,
    "champion_pooled_median_auc_relative_improvement": 0.05,
    "champion_structural_fraction": 0.90,
    "require_exact_replay": True,
    "max_invalid_candidates": 0,
    "max_generation_failures": 0,
    "selection": "selected-only",
    "oracle_access": False,
    "equal_budgets": True,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode()


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def episode_id(order: int, graph_seed: int, policy_seed: int) -> str:
    return f"o{order:02d}-g{graph_seed:04d}-p{policy_seed:04d}"


def _cfg(config: object, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def build_manifest(config: object | None = None) -> dict[str, JsonValue]:
    experiment = _cfg(config, "experiment", config)
    orders = tuple(_cfg(experiment, "orders", ORDERS))
    graph_seeds = tuple(_cfg(experiment, "graph_seeds", GRAPH_SEEDS))
    policy_seeds = tuple(_cfg(experiment, "policy_seeds", POLICY_SEEDS))
    horizon = int(_cfg(experiment, "horizon", HORIZON))
    shards = int(_cfg(experiment, "shard_count", SHARD_COUNT))
    if (orders, graph_seeds, policy_seeds, horizon, shards) != (
        ORDERS,
        GRAPH_SEEDS,
        POLICY_SEEDS,
        HORIZON,
        SHARD_COUNT,
    ):
        raise ValueError("Stage 3 manifest matrix is frozen")
    rows: list[dict[str, JsonValue]] = []
    grouped: dict[str, list[str]] = {f"shard-{i:02d}": [] for i in range(shards)}
    index = 0
    for order in orders:
        for graph_seed in graph_seeds:
            for policy_seed in policy_seeds:
                sid = f"shard-{index % shards:02d}"
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
                grouped[sid].append(eid)
                index += 1
    shard_rows = [
        {"shard_id": sid, "episode_ids": cast(list[JsonValue], ids), "episode_count": len(ids)}
        for sid, ids in grouped.items()
    ]
    base: dict[str, JsonValue] = {
        "schema_version": STAGE3_MANIFEST_VERSION,
        "dataset": "retained-stage2d-toy-trajectories",
        "held_out": False,
        "source_stage2d_manifest": "configs/manifests/stage2d-episodes-v1.json",
        "source_stage2d_evidence": "docs/reports/STAGE2D_CONFIRMATORY_REPORT.md",
        "episode_count": len(rows),
        "shard_count": shards,
        "episodes_per_shard": EPISODES_PER_SHARD,
        "episodes": cast(list[JsonValue], rows),
        "shards": cast(list[JsonValue], shard_rows),
        "gate_spec": GATE_SPEC,
    }
    return {**base, "manifest_sha256": sha256(base)}


def validate_manifest(manifest: dict[str, JsonValue], config: object | None = None) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != STAGE3_MANIFEST_VERSION:
        raise ValueError("unexpected Stage 3 manifest schema")
    if manifest.get("manifest_sha256") != sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    ):
        raise ValueError("manifest hash mismatch")
    expected = build_manifest(config)
    for key in (
        "episode_count",
        "shard_count",
        "episodes_per_shard",
        "episodes",
        "shards",
        "dataset",
        "held_out",
        "source_stage2d_manifest",
        "source_stage2d_evidence",
        "gate_spec",
    ):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"manifest {key} mismatch")
    episodes = cast(list[dict[str, JsonValue]], manifest["episodes"])
    if len(episodes) != 128 or len({str(e.get("episode_id")) for e in episodes}) != len(episodes):
        raise ValueError("invalid episode roster")
    if [str(e["episode_id"]) for e in episodes] != sorted(str(e["episode_id"]) for e in episodes):
        raise ValueError("episodes must be sorted by canonical ID")
    if any(
        set(e) != {"episode_id", "order", "graph_seed", "policy_seed", "horizon", "shard_id"}
        for e in episodes
    ):
        raise ValueError("episode keys mismatch")


def load_manifest(path: str | Path, config: object | None = None) -> dict[str, JsonValue]:
    value = cast(
        dict[str, JsonValue],
        json.loads(Path(path).read_text(encoding="utf-8")),
    )
    validate_manifest(value, config)
    return value


__all__ = [
    "STAGE3_MANIFEST_VERSION",
    "GATE_SPEC",
    "canonical_bytes",
    "sha256",
    "episode_id",
    "build_manifest",
    "validate_manifest",
    "load_manifest",
]
