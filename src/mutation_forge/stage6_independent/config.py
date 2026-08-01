"""Frozen, provider-independent Stage 6 verification configuration.

This module deliberately has no dependency on the Stage 5 analysis code.  It
only reads the preregistered TOML/JSON objects, verifies their immutable
identities, and constructs the complete episode manifest deterministically.
"""
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
from mutation_forge.stage2b.config import Stage2BConfig, load_stage2b_config
from mutation_forge.stage3.manifest import canonical_bytes

STAGE6_CONFIG_VERSION = "stage6.verification.v1"
STAGE6_MANIFEST_VERSION = "stage6.verification.manifest.v1"

# The Stage 5 freeze is the reviewed baseline from which this independent
# verification is derived.  Stage 6 does not import Stage 5 analysis modules.
PROJECT_COMMIT = "cc2f7b7254705d47fd4995a4b8a2bd45d545795c"
HEG_COMMIT = "fd97451b0f3d87400d1d955a2c6b1b18303344ff"
STAGE5_MANIFEST_SHA256 = "ded50562899fd3b5d6214757f2581a2aab6507444a216643ac11fba0bb748c9d"
STAGE5_EVIDENCE_MANIFEST_SHA256 = "e996563c145ac12bc7e7ae9bb284ae98d14a2990aaac9bce17e9992486780cce"
STAGE5_FREEZE_SHA256 = "53f2df2d71b723dbdcd5983d24dcff25f977e2709a0089011882c4c56f860645"

POLICY_IDS = (
    "program-d5ad1c8203e0d9f25f03aabd",
    "candidate-slot-04",
    "random",
    "structural",
)
POLICY_SOURCE_SHA256 = {
    POLICY_IDS[0]: "e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b",
    POLICY_IDS[1]: "a5f540459695bbf7d454eeccbb8e48158d6130df6a769b67d1447de18276dc01",
    POLICY_IDS[2]: "d4994fb96bdc3c23b8b24d9bca041f2822bc30329bcf8f9cdbd2e277e65b0612",
    POLICY_IDS[3]: "68aba299d7735198d38a8d30e221ef99cdbb7d846c502aca41691c49ceef87be",
}
POLICY_AST_SHA256 = {
    POLICY_IDS[0]: "2243214df58c805e9a9343dc31ed082279e1c2ac31b21243bf889dbc9a19e165",
    POLICY_IDS[1]: "cef05bb644e2e0a9acbc4972fbaa6d4ba3e033ee8a73ecd756da44100c767f5c",
    POLICY_IDS[2]: "f7f502b0319df5dc32ef0f8476024c4986dcb3422ef2e03b117a3d394bbfc7b7",
    POLICY_IDS[3]: "5b017c2ba79953e31b224df91e060d4af27c3b212695a03e8650ec91e8b0ad81",
}
POLICY_BEHAVIOR_SHA256 = {
    POLICY_IDS[0]: "8c2bdaa213f11b253d3ffcae1653bd01536879bb5c254a1586ded9ae522a868e",
    POLICY_IDS[1]: "3694bd0b8813621c2abe98186dbbe933f3f7758b7b660d4c9229616e77d76c3c",
    POLICY_IDS[2]: "2bc6f8a22a1b43431ee6cc1817716f7a55c688f736a8d5e04be020c7bf1821f2",
    POLICY_IDS[3]: "8ab4d5d6f5ed5d908fca2a637cbb1541c5bf7e6e09d5bbadeb3b703b2aa673aa",
}

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_080_104
CONFIDENCE_LEVEL = 0.95
CHAMPION_STAGE3_THRESHOLD = 0.02
CHAMPION_RANDOM_THRESHOLD = 0.05
STRUCTURAL_RETENTION_THRESHOLD = 0.99


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest excluding its self-referential ``manifest_sha256``."""
    return sha256_value({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def episode_id(order: int, graph_seed: int, relabeling_seed: int, policy_seed: int) -> str:
    return f"o{order:02d}-g{graph_seed:04d}-r{relabeling_seed:04d}-p{policy_seed:04d}"


@dataclass(frozen=True, slots=True)
class Stage6Experiment:
    orders: tuple[int, ...]
    graph_seeds: tuple[int, ...]
    relabeling_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    horizon: int
    identity_count: int
    shard_count: int
    episodes_per_shard: int

    @property
    def episode_count(self) -> int:
        return len(self.orders) * len(self.graph_seeds) * len(self.relabeling_seeds) * len(self.policy_seeds)


@dataclass(frozen=True, slots=True)
class Stage6Resources:
    workers: int
    reserved_physical_cores: int
    thread_count: int


@dataclass(frozen=True, slots=True)
class Stage6Policy:
    path: Path
    source_sha256: str
    normalized_ast_sha256: str
    behavior_signature_sha256: str


@dataclass(frozen=True, slots=True)
class Stage6Config:
    source_path: Path
    run_root: Path
    project_repo: Path
    heg_repo: Path
    frozen_project_commit: str
    frozen_heg_commit: str
    stage5_freeze_path: Path
    stage5_manifest_path: Path
    expected_stage5_manifest_sha256: str
    stage5_evidence_path: Path
    stage5_evidence_manifest_path: Path
    expected_stage5_evidence_manifest_sha256: str
    stage2b_config_path: Path
    stage2b: Stage2BConfig
    manifest_path: Path
    prior_manifest_paths: tuple[Path, ...]
    policy_paths: dict[str, Path]
    policy_source_hashes: dict[str, str]
    policy_ast_hashes: dict[str, str]
    policy_behavior_hashes: dict[str, str]
    experiment: Stage6Experiment
    resources: Stage6Resources
    relabel_algorithm: str
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float
    champion_stage3_threshold: float
    champion_random_threshold: float
    structural_retention_threshold: float

    @property
    def policies(self) -> dict[str, Stage6Policy]:
        return {
            policy_id: Stage6Policy(
                self.policy_paths[policy_id],
                self.policy_source_hashes[policy_id],
                self.policy_ast_hashes[policy_id],
                self.policy_behavior_hashes[policy_id],
            )
            for policy_id in POLICY_IDS
        }

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
        return result

    def stable_hash(self) -> str:
        return sha256_value(self.resolved_dict())


def _path(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value)
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _tuple_ints(value: object, expected: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or tuple(value) != expected:
        raise ValueError(f"{name} is not frozen as {expected}")
    return expected


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Stage 6 config must be a TOML table")
    return value


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def load_config(path: str | Path = "configs/stage6-verification.toml") -> Stage6Config:
    source_path = Path(path).resolve()
    raw = _load_raw(source_path)
    if raw.get("schema_version") != STAGE6_CONFIG_VERSION:
        raise ValueError("invalid Stage 6 schema_version")
    required = ("run", "repositories", "inputs", "policies", "experiment", "relabeling", "resources", "bootstrap", "gates")
    if any(not isinstance(raw.get(name), dict) for name in required):
        raise ValueError("Stage 6 config is missing a required table")
    base = source_path.parent
    run = cast(dict[str, Any], raw["run"])
    repositories = cast(dict[str, Any], raw["repositories"])
    inputs = cast(dict[str, Any], raw["inputs"])
    policies = cast(dict[str, Any], raw["policies"])
    experiment = cast(dict[str, Any], raw["experiment"])
    relabeling = cast(dict[str, Any], raw["relabeling"])
    resources = cast(dict[str, Any], raw["resources"])
    bootstrap = cast(dict[str, Any], raw["bootstrap"])
    gates = cast(dict[str, Any], raw["gates"])

    parsed_experiment = Stage6Experiment(
        _tuple_ints(experiment.get("orders"), (20, 24, 28), "experiment.orders"),
        _tuple_ints(experiment.get("graph_seeds"), tuple(range(701, 709)), "experiment.graph_seeds"),
        _tuple_ints(experiment.get("relabeling_seeds"), (7101, 7102), "experiment.relabeling_seeds"),
        _tuple_ints(experiment.get("policy_seeds"), tuple(range(7001, 7017)), "experiment.policy_seeds"),
        int(experiment.get("horizon", 0)),
        int(experiment.get("identity_count", 0)),
        int(experiment.get("shard_count", 0)),
        int(experiment.get("episodes_per_shard", 0)),
    )
    if (parsed_experiment.horizon, parsed_experiment.identity_count, parsed_experiment.shard_count, parsed_experiment.episodes_per_shard, parsed_experiment.episode_count) != (32, 768, 12, 64, 768):
        raise ValueError("Stage 6 experiment matrix is not frozen")
    parsed_resources = Stage6Resources(int(resources.get("workers", 0)), int(resources.get("reserved_physical_cores", 0)), int(resources.get("thread_count", 0)))
    if (
        parsed_resources.workers < 1
        or parsed_resources.workers > 8
        or parsed_resources.reserved_physical_cores < 8
        or parsed_resources.thread_count != 1
    ):
        raise ValueError("Stage 6 resources exceed the frozen bounds")
    if int(bootstrap.get("samples", 0)) != BOOTSTRAP_SAMPLES or int(bootstrap.get("seed", 0)) != BOOTSTRAP_SEED or float(bootstrap.get("confidence_level", 0)) != CONFIDENCE_LEVEL:
        raise ValueError("Stage 6 bootstrap is not frozen")
    if (float(gates.get("champion_stage3_threshold", 0)), float(gates.get("champion_random_threshold", 0)), float(gates.get("structural_retention_threshold", 0))) != (CHAMPION_STAGE3_THRESHOLD, CHAMPION_RANDOM_THRESHOLD, STRUCTURAL_RETENTION_THRESHOLD):
        raise ValueError("Stage 6 gates are not frozen")
    if relabeling.get("algorithm") != "fisher-yates-sha256-v1":
        raise ValueError("Stage 6 relabeling algorithm is not frozen")

    project_repo = _path(base, repositories.get("project_repo"), "repositories.project_repo")
    heg_repo = _path(base, repositories.get("heg_repo"), "repositories.heg_repo")
    if repositories.get("frozen_project_commit") != PROJECT_COMMIT or repositories.get("frozen_heg_commit") != HEG_COMMIT:
        raise ValueError("Stage 6 repository pins differ")
    current_project = _git(project_repo, "rev-parse", "HEAD")
    ancestor = subprocess.run(["git", "-C", str(project_repo), "merge-base", "--is-ancestor", PROJECT_COMMIT, current_project], capture_output=True, timeout=30)
    if ancestor.returncode != 0:
        raise ValueError("project repository is not based on the required frozen SHA")
    if _git(heg_repo, "rev-parse", "HEAD") != HEG_COMMIT or _git(heg_repo, "status", "--short"):
        raise ValueError("HEG is not clean at the frozen pin")

    stage5_freeze_path = _path(base, inputs.get("stage5_freeze"), "inputs.stage5_freeze")
    stage5_manifest_path = _path(base, inputs.get("stage5_manifest"), "inputs.stage5_manifest")
    stage5_evidence_path = _path(base, inputs.get("stage5_evidence"), "inputs.stage5_evidence")
    stage5_evidence_manifest_path = _path(
        base, inputs.get("stage5_evidence_manifest"), "inputs.stage5_evidence_manifest"
    )
    stage2b_config_path = _path(base, inputs.get("stage2b_config"), "inputs.stage2b_config")
    stage2b = load_stage2b_config(stage2b_config_path)
    expected_stage5_manifest = _sha(inputs.get("expected_stage5_manifest_sha256"), "inputs.expected_stage5_manifest_sha256")
    expected_stage5_evidence_manifest = _sha(
        inputs.get("expected_stage5_evidence_manifest_sha256"),
        "inputs.expected_stage5_evidence_manifest_sha256",
    )
    if (
        expected_stage5_manifest != STAGE5_MANIFEST_SHA256
        or expected_stage5_evidence_manifest != STAGE5_EVIDENCE_MANIFEST_SHA256
        or not stage5_freeze_path.is_file()
        or not stage5_manifest_path.is_file()
        or not stage5_evidence_path.is_dir()
        or not stage5_evidence_manifest_path.is_file()
    ):
        raise ValueError("Stage 5 evidence paths or hash are not frozen")
    stage5_manifest_raw = json.loads(stage5_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(stage5_manifest_raw, Mapping) or stage5_manifest_raw.get("manifest_sha256") != expected_stage5_manifest:
        raise ValueError("Stage 5 manifest evidence hash mismatch")
    freeze_raw = json.loads(stage5_freeze_path.read_text(encoding="utf-8"))
    if not isinstance(freeze_raw, Mapping) or freeze_raw.get("manifest_sha256") != expected_stage5_manifest:
        raise ValueError("Stage 5 freeze does not reference the expected manifest")
    if sha256_bytes(stage5_evidence_manifest_path.read_bytes()) != expected_stage5_evidence_manifest:
        raise ValueError("Stage 5 evidence manifest SHA-256 mismatch")

    manifest_path = _path(base, inputs.get("manifest"), "inputs.manifest")
    prior_raw = inputs.get("prior_manifests")
    if not isinstance(prior_raw, list) or not prior_raw:
        raise ValueError("Stage 6 prior_manifests must be a non-empty list")
    prior = tuple(_path(base, value, "inputs.prior_manifest") for value in prior_raw)
    if any(not item.is_file() for item in prior):
        raise FileNotFoundError(", ".join(str(item) for item in prior if not item.is_file()))

    configured_ids = tuple(policies.get("policy_ids", []))
    if configured_ids != POLICY_IDS:
        raise ValueError("Stage 6 policy IDs are not frozen")
    paths_raw = policies.get("paths")
    source_raw = policies.get("source_sha256")
    ast_raw = policies.get("normalized_ast_sha256")
    behavior_raw = policies.get("behavior_signature_sha256")
    if not all(isinstance(value, Mapping) for value in (paths_raw, source_raw, ast_raw, behavior_raw)):
        raise ValueError("Stage 6 policy identity tables are missing")
    policy_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    ast_hashes: dict[str, str] = {}
    behavior_hashes: dict[str, str] = {}
    for policy_id in POLICY_IDS:
        policy_paths[policy_id] = _path(base, cast(Mapping[str, Any], paths_raw).get(policy_id), f"policies.paths.{policy_id}")
        if not policy_paths[policy_id].is_file():
            raise FileNotFoundError(policy_paths[policy_id])
        source_hashes[policy_id] = _sha(cast(Mapping[str, Any], source_raw).get(policy_id), f"policies.source_sha256.{policy_id}")
        ast_hashes[policy_id] = _sha(cast(Mapping[str, Any], ast_raw).get(policy_id), f"policies.normalized_ast_sha256.{policy_id}")
        behavior_hashes[policy_id] = _sha(cast(Mapping[str, Any], behavior_raw).get(policy_id), f"policies.behavior_signature_sha256.{policy_id}")
        if source_hashes[policy_id] != POLICY_SOURCE_SHA256[policy_id] or ast_hashes[policy_id] != POLICY_AST_SHA256[policy_id] or behavior_hashes[policy_id] != POLICY_BEHAVIOR_SHA256[policy_id]:
            raise ValueError(f"{policy_id} policy identity differs from the Stage 5 freeze")
        if sha256_bytes(policy_paths[policy_id].read_bytes()) != source_hashes[policy_id]:
            raise ValueError(f"{policy_id} source hash mismatch")

    return Stage6Config(
        source_path, _path(base, run.get("run_root"), "run.run_root"), project_repo, heg_repo,
        PROJECT_COMMIT, HEG_COMMIT, stage5_freeze_path, stage5_manifest_path,
        expected_stage5_manifest, stage5_evidence_path, stage5_evidence_manifest_path,
        expected_stage5_evidence_manifest, stage2b_config_path, stage2b, manifest_path, prior,
        policy_paths, source_hashes,
        ast_hashes, behavior_hashes, parsed_experiment, parsed_resources,
        str(relabeling["algorithm"]), BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED,
        CONFIDENCE_LEVEL, CHAMPION_STAGE3_THRESHOLD, CHAMPION_RANDOM_THRESHOLD,
        STRUCTURAL_RETENTION_THRESHOLD,
    )


def _collect_episode_keys(value: object, result: set[tuple[int, int, int, int | None]]) -> None:
    if isinstance(value, Mapping):
        if {"order", "graph_seed", "policy_seed"}.issubset(value) and all(isinstance(value[key], int) and not isinstance(value[key], bool) for key in ("order", "graph_seed", "policy_seed")):
            relabel = value.get("relabeling_seed")
            if relabel is None or (isinstance(relabel, int) and not isinstance(relabel, bool)):
                result.add((int(value["order"]), int(value["graph_seed"]), int(value["policy_seed"]), None if relabel is None else int(relabel)))
        for item in value.values():
            _collect_episode_keys(item, result)
    elif isinstance(value, list):
        for item in value:
            _collect_episode_keys(item, result)


def _prior_keys(paths: tuple[Path, ...]) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int, int]], set[int], dict[str, str]]:
    base: set[tuple[int, int, int]] = set()
    complete: set[tuple[int, int, int, int]] = set()
    orders: set[int] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        keys: set[tuple[int, int, int, int | None]] = set()
        _collect_episode_keys(raw, keys)
        for order, graph_seed, policy_seed, relabel in keys:
            base.add((order, graph_seed, policy_seed))
            orders.add(order)
            if relabel is not None:
                complete.add((order, graph_seed, relabel, policy_seed))
        hashes[str(path)] = sha256_bytes(path.read_bytes())
    return base, complete, orders, hashes


def build_manifest(config: Stage6Config) -> dict[str, JsonValue]:
    rows: list[dict[str, JsonValue]] = []
    grouped: dict[str, list[str]] = {f"shard-{index:02d}": [] for index in range(config.experiment.shard_count)}
    index = 0
    for order in config.experiment.orders:
        for graph_seed in config.experiment.graph_seeds:
            for relabeling_seed in config.experiment.relabeling_seeds:
                for policy_seed in config.experiment.policy_seeds:
                    row: dict[str, JsonValue] = {
                        "episode_id": episode_id(order, graph_seed, relabeling_seed, policy_seed),
                        "order": order, "graph_seed": graph_seed, "relabeling_seed": relabeling_seed,
                        "policy_seed": policy_seed, "horizon": config.experiment.horizon,
                        "shard_id": f"shard-{index % config.experiment.shard_count:02d}",
                    }
                    rows.append(row)
                    grouped[str(row["shard_id"])].append(str(row["episode_id"]))
                    index += 1
    shards = [{"shard_id": shard_id, "episode_ids": ids, "episode_count": len(ids)} for shard_id, ids in grouped.items()]
    base: dict[str, JsonValue] = {
        "schema_version": STAGE6_MANIFEST_VERSION,
        "dataset": "stage6-independent-verification-fresh-graphs",
        "held_out": True,
        "independent": True,
        "orders": list(config.experiment.orders), "graph_seeds": list(config.experiment.graph_seeds),
        "relabeling_seeds": list(config.experiment.relabeling_seeds), "policy_seeds": list(config.experiment.policy_seeds),
        "horizon": config.experiment.horizon, "identity_count": config.experiment.identity_count,
        "episode_count": len(rows), "shard_count": config.experiment.shard_count,
        "episodes_per_shard": config.experiment.episodes_per_shard, "episodes": cast(list[JsonValue], rows),
        "shards": cast(list[JsonValue], shards), "frozen_policy_ids": list(POLICY_IDS),
        "stage5_manifest_sha256": config.expected_stage5_manifest_sha256,
        "project_commit": config.frozen_project_commit, "heg_commit": config.frozen_heg_commit,
        "relabeling_algorithm": config.relabel_algorithm,
    }
    return {**base, "manifest_sha256": manifest_hash(base)}


def validate_manifest(manifest: Mapping[str, Any], config: Stage6Config) -> dict[str, Any]:
    expected = build_manifest(config)
    if dict(manifest) != expected:
        raise ValueError("Stage 6 manifest differs from the frozen deterministic manifest")
    if manifest.get("manifest_sha256") != manifest_hash(manifest):
        raise ValueError("Stage 6 manifest hash mismatch")
    rows = cast(list[Mapping[str, Any]], manifest["episodes"])
    new_base = {(int(row["order"]), int(row["graph_seed"]), int(row["policy_seed"])) for row in rows}
    new_complete = {(int(row["order"]), int(row["graph_seed"]), int(row["relabeling_seed"]), int(row["policy_seed"])) for row in rows}
    if len(new_base) != len(rows) // len(config.experiment.relabeling_seeds) or len(new_complete) != len(rows):
        raise ValueError("Stage 6 episode identities are not unique")
    prior_base, prior_complete, prior_orders, prior_hashes = _prior_keys(config.prior_manifest_paths)
    if set(config.experiment.orders) & prior_orders:
        raise ValueError("Stage 6 orders overlap a prior scientific manifest")
    if new_base & prior_base:
        raise ValueError("Stage 6 base seed identities overlap a prior manifest")
    if new_complete & prior_complete:
        raise ValueError("Stage 6 relabeled seed identities overlap a prior manifest")
    return {
        "new_base_identity_count": len(new_base), "new_complete_identity_count": len(new_complete),
        "prior_base_identity_count": len(prior_base), "prior_complete_identity_count": len(prior_complete),
        "prior_orders": sorted(prior_orders), "prior_manifest_sha256": prior_hashes,
        "orders_disjoint": True, "base_identities_disjoint": True, "complete_identities_disjoint": True,
    }


def write_manifest(config: Stage6Config) -> dict[str, JsonValue]:
    manifest = build_manifest(config)
    validate_manifest(manifest, config)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    return manifest


__all__ = [
    "BOOTSTRAP_SAMPLES", "BOOTSTRAP_SEED", "CHAMPION_RANDOM_THRESHOLD", "CHAMPION_STAGE3_THRESHOLD",
    "CONFIDENCE_LEVEL", "HEG_COMMIT", "POLICY_IDS", "PROJECT_COMMIT", "STAGE5_EVIDENCE_MANIFEST_SHA256", "STAGE5_MANIFEST_SHA256",
    "STAGE6_CONFIG_VERSION", "STAGE6_MANIFEST_VERSION", "STRUCTURAL_RETENTION_THRESHOLD", "Stage6Config",
    "Stage6Experiment", "Stage6Policy", "Stage6Resources", "build_manifest", "episode_id", "load_config",
    "manifest_hash", "sha256_bytes", "sha256_value", "validate_manifest", "write_manifest",
]
