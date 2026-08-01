"""Frozen configuration and manifest construction for Stage 4E."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage2b.config import Stage2BConfig, load_stage2b_config
from mutation_forge.stage3.manifest import canonical_bytes, episode_id

STAGE4E_CONFIG_VERSION = "stage4e.confirmation.v1"
STAGE4E_MANIFEST_VERSION = "stage4e.confirmation.manifest.v1"
START_COMMIT = "584f8092ef15ca7c12ffdbb6d7fbb30ad80ada41"
HEG_COMMIT = "fd97451b0f3d87400d1d955a2c6b1b18303344ff"
CHAMPION_ID = "program-d5ad1c8203e0d9f25f03aabd"
COMPARATOR_ID = "candidate-slot-04"
CHAMPION_SOURCE_SHA256 = "e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b"
CHAMPION_AST_SHA256 = "2243214df58c805e9a9343dc31ed082279e1c2ac31b21243bf889dbc9a19e165"
COMPARATOR_SOURCE_SHA256 = "a5f540459695bbf7d454eeccbb8e48158d6130df6a769b67d1447de18276dc01"
COMPARATOR_AST_SHA256 = "cef05bb644e2e0a9acbc4972fbaa6d4ba3e033ee8a73ecd756da44100c767f5c"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_080_102
CONFIDENCE_LEVEL = 0.95
RELATIVE_THRESHOLD = 0.02


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


@dataclass(frozen=True, slots=True)
class Stage4EExperiment:
    orders: tuple[int, ...]
    graph_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    horizon: int
    shard_count: int
    episodes_per_shard: int

    @property
    def episode_count(self) -> int:
        return len(self.orders) * len(self.graph_seeds) * len(self.policy_seeds)


@dataclass(frozen=True, slots=True)
class Stage4EResources:
    workers: int
    reserved_physical_cores: int
    thread_count: int


@dataclass(frozen=True, slots=True)
class Stage4EConfig:
    source_path: Path
    run_root: Path
    project_repo: Path
    heg_repo: Path
    frozen_project_commit: str
    frozen_heg_commit: str
    stage2b_config_path: Path
    stage2b: Stage2BConfig
    manifest_path: Path
    prior_manifest_paths: tuple[Path, ...]
    champion_source_path: Path
    comparator_source_path: Path
    champion_id: str
    comparator_id: str
    champion_source_sha256: str
    champion_ast_sha256: str
    comparator_source_sha256: str
    comparator_ast_sha256: str
    experiment: Stage4EExperiment
    resources: Stage4EResources
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float
    relative_improvement_threshold: float

    def resolved_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)

        def normalize(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(key): normalize(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [normalize(item) for item in value]
            return value

        result = cast(dict[str, JsonValue], normalize(raw))
        result.pop("source_path", None)
        result.pop("stage2b", None)
        result["stage2b_config_sha256"] = sha256_bytes(self.stage2b_config_path.read_bytes())
        return result

    def stable_hash(self) -> str:
        return sha256_value(self.resolved_dict())

    @property
    def limits(self) -> Stage4EResources:
        """Compatibility view consumed by the shared worker/affinity helpers."""
        return self.resources


def _path(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value)
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _tuple_ints(value: object, expected: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or tuple(value) != expected:
        raise ValueError(f"{name} is not frozen as {expected}")
    return expected


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Stage 4E config must be a TOML table")
    return value


def load_stage4e_config(path: str | Path = "configs/stage4e-confirmation.toml") -> Stage4EConfig:
    source_path = Path(path).resolve()
    raw = _load_raw(source_path)
    if raw.get("schema_version") != STAGE4E_CONFIG_VERSION:
        raise ValueError("invalid Stage 4E schema_version")
    for table in ("run", "repositories", "inputs", "policies", "experiment", "resources", "bootstrap", "gate"):
        if not isinstance(raw.get(table), dict):
            raise ValueError(f"missing Stage 4E [{table}] table")
    base = source_path.parent
    run = cast(dict[str, Any], raw["run"])
    repositories = cast(dict[str, Any], raw["repositories"])
    inputs = cast(dict[str, Any], raw["inputs"])
    policies = cast(dict[str, Any], raw["policies"])
    experiment = cast(dict[str, Any], raw["experiment"])
    resources = cast(dict[str, Any], raw["resources"])
    bootstrap = cast(dict[str, Any], raw["bootstrap"])
    gate = cast(dict[str, Any], raw["gate"])
    expected_orders = (10, 12, 16)
    expected_graphs = tuple(range(501, 517))
    expected_policy_seeds = tuple(range(5001, 5033))
    parsed_experiment = Stage4EExperiment(
        _tuple_ints(experiment.get("orders"), expected_orders, "experiment.orders"),
        _tuple_ints(experiment.get("graph_seeds"), expected_graphs, "experiment.graph_seeds"),
        _tuple_ints(experiment.get("policy_seeds"), expected_policy_seeds, "experiment.policy_seeds"),
        int(experiment.get("horizon", 0)),
        int(experiment.get("shard_count", 0)),
        int(experiment.get("episodes_per_shard", 0)),
    )
    if (parsed_experiment.horizon, parsed_experiment.shard_count, parsed_experiment.episodes_per_shard, parsed_experiment.episode_count) != (32, 24, 64, 1536):
        raise ValueError("Stage 4E experiment matrix is not frozen")
    parsed_resources = Stage4EResources(int(resources.get("workers", 0)), int(resources.get("reserved_physical_cores", 0)), int(resources.get("thread_count", 0)))
    if parsed_resources != Stage4EResources(8, 8, 1):
        raise ValueError("Stage 4E resources are not frozen")
    if int(bootstrap.get("samples", 0)) != BOOTSTRAP_SAMPLES or int(bootstrap.get("seed", 0)) != BOOTSTRAP_SEED or float(bootstrap.get("confidence_level", 0)) != CONFIDENCE_LEVEL:
        raise ValueError("Stage 4E bootstrap is not frozen")
    if float(gate.get("relative_improvement_threshold", 0)) != RELATIVE_THRESHOLD:
        raise ValueError("Stage 4E gate threshold is not frozen")
    champion_id = policies.get("champion_id")
    comparator_id = policies.get("comparator_id")
    if not isinstance(champion_id, str) or not isinstance(comparator_id, str):
        raise ValueError("Stage 4E policy IDs must be strings")
    if (champion_id, comparator_id) != (CHAMPION_ID, COMPARATOR_ID):
        raise ValueError("Stage 4E policy IDs are not frozen")
    if repositories.get("frozen_project_commit") != START_COMMIT or repositories.get("frozen_heg_commit") != HEG_COMMIT:
        raise ValueError("Stage 4E repository pins are not frozen")
    stage2b_path = _path(base, inputs.get("stage2b_config"), "inputs.stage2b_config")
    stage2b = load_stage2b_config(stage2b_path)
    champion_path = _path(base, policies.get("champion_source"), "policies.champion_source")
    comparator_path = _path(base, policies.get("comparator_source"), "policies.comparator_source")
    expected = {
        "champion_source_sha256": _sha(policies.get("champion_source_sha256"), "champion source hash"),
        "champion_ast_sha256": _sha(policies.get("champion_ast_sha256"), "champion AST hash"),
        "comparator_source_sha256": _sha(policies.get("comparator_source_sha256"), "comparator source hash"),
        "comparator_ast_sha256": _sha(policies.get("comparator_ast_sha256"), "comparator AST hash"),
    }
    for name, source_path, source_hash, ast_hash in (
        ("champion", champion_path, expected["champion_source_sha256"], expected["champion_ast_sha256"]),
        ("comparator", comparator_path, expected["comparator_source_sha256"], expected["comparator_ast_sha256"]),
    ):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source = source_path.read_text(encoding="utf-8")
        if sha256_bytes(source.encode("utf-8")) != source_hash:
            raise ValueError(f"{name} source hash mismatch")
        validation = validate_policy(source, stage2b.sandbox)
        if not validation.valid or validation.identity.normalized_ast_sha256 != ast_hash:
            raise ValueError(f"{name} normalized AST identity mismatch")
    project_repo = _path(base, repositories.get("project_repo"), "repositories.project_repo")
    heg_repo = _path(base, repositories.get("heg_repo"), "repositories.heg_repo")
    current_project = _git(project_repo, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(project_repo), "merge-base", "--is-ancestor", START_COMMIT, current_project],
        capture_output=True,
        timeout=30,
    )
    if ancestor.returncode != 0:
        raise ValueError("project repository is not based on the frozen Stage 4E base")
    if _git(heg_repo, "rev-parse", "HEAD") != HEG_COMMIT or _git(heg_repo, "status", "--short"):
        raise ValueError("HEG is not clean at the frozen pin")
    prior = tuple(_path(base, value, "inputs.prior_manifest") for value in inputs.get("prior_manifests", []))
    manifest_path = _path(base, inputs.get("manifest"), "inputs.manifest")
    return Stage4EConfig(source_path, _path(base, run.get("run_root"), "run.run_root"), project_repo, heg_repo, START_COMMIT, HEG_COMMIT, stage2b_path, stage2b, manifest_path, prior, champion_path, comparator_path, champion_id, comparator_id, expected["champion_source_sha256"], expected["champion_ast_sha256"], expected["comparator_source_sha256"], expected["comparator_ast_sha256"], parsed_experiment, parsed_resources, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL, RELATIVE_THRESHOLD)


def build_manifest(config: Stage4EConfig) -> dict[str, JsonValue]:
    rows: list[dict[str, JsonValue]] = []
    grouped: dict[str, list[str]] = {f"shard-{index:02d}": [] for index in range(config.experiment.shard_count)}
    index = 0
    for order in config.experiment.orders:
        for graph_seed in config.experiment.graph_seeds:
            for policy_seed in config.experiment.policy_seeds:
                episode: dict[str, JsonValue] = {
                    "episode_id": episode_id(order, graph_seed, policy_seed),
                    "order": order,
                    "graph_seed": graph_seed,
                    "policy_seed": policy_seed,
                    "horizon": config.experiment.horizon,
                    "shard_id": f"shard-{index % config.experiment.shard_count:02d}",
                }
                rows.append(episode)
                grouped[str(episode["shard_id"])].append(str(episode["episode_id"]))
                index += 1
    shards = [
        {"shard_id": shard_id, "episode_ids": cast(list[JsonValue], ids), "episode_count": len(ids)}
        for shard_id, ids in grouped.items()
    ]
    base: dict[str, JsonValue] = {
        "schema_version": STAGE4E_MANIFEST_VERSION,
        "dataset": "stage4e-unseen-toy-graphs",
        "held_out": True,
        "orders": list(config.experiment.orders),
        "graph_seeds": list(config.experiment.graph_seeds),
        "policy_seeds": list(config.experiment.policy_seeds),
        "horizon": config.experiment.horizon,
        "episode_count": len(rows),
        "shard_count": config.experiment.shard_count,
        "episodes_per_shard": config.experiment.episodes_per_shard,
        "episodes": cast(list[JsonValue], rows),
        "shards": cast(list[JsonValue], shards),
        "frozen_policy_ids": [config.champion_id, config.comparator_id],
        "stage4e_base_commit": config.frozen_project_commit,
        "heg_commit": config.frozen_heg_commit,
    }
    return {**base, "manifest_sha256": sha256_value(base)}


def _collect_episode_keys(value: object, result: set[tuple[int, int, int]]) -> None:
    if isinstance(value, Mapping):
        keys = {"order", "graph_seed", "policy_seed"}
        if keys.issubset(value) and all(isinstance(value[key], int) for key in keys):
            result.add((int(value["order"]), int(value["graph_seed"]), int(value["policy_seed"])))
        for item in value.values():
            _collect_episode_keys(item, result)
    elif isinstance(value, list):
        for item in value:
            _collect_episode_keys(item, result)


def validate_manifest(manifest: Mapping[str, Any], config: Stage4EConfig) -> None:
    expected = build_manifest(config)
    if dict(manifest) != expected:
        raise ValueError("Stage 4E manifest differs from the frozen deterministic manifest")
    if manifest.get("manifest_sha256") != sha256_value({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValueError("Stage 4E manifest hash mismatch")
    prior_keys: set[tuple[int, int, int]] = set()
    for path in config.prior_manifest_paths:
        if not path.is_file():
            continue
        try:
            _collect_episode_keys(json.loads(path.read_text(encoding="utf-8")), prior_keys)
        except (OSError, json.JSONDecodeError):
            continue
    new_keys = {(int(row["order"]), int(row["graph_seed"]), int(row["policy_seed"])) for row in cast(list[Mapping[str, Any]], manifest["episodes"])}
    if prior_keys & new_keys:
        raise ValueError("Stage 4E seed range overlaps a prior manifest")
    if len(new_keys) != config.experiment.episode_count:
        raise ValueError("Stage 4E episode roster is not unique")


def write_manifest(config: Stage4EConfig) -> dict[str, JsonValue]:
    manifest = build_manifest(config)
    validate_manifest(manifest, config)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    return manifest


def load_manifest(config: Stage4EConfig) -> dict[str, JsonValue]:
    value = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Stage 4E manifest must be an object")
    manifest = cast(dict[str, JsonValue], value)
    validate_manifest(manifest, config)
    return manifest


__all__ = [
    "STAGE4E_CONFIG_VERSION",
    "STAGE4E_MANIFEST_VERSION",
    "Stage4EConfig",
    "Stage4EExperiment",
    "Stage4EResources",
    "build_manifest",
    "load_manifest",
    "load_stage4e_config",
    "sha256_bytes",
    "sha256_value",
    "validate_manifest",
    "write_manifest",
]
