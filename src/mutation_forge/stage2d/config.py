from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.stage2b.config import Stage2BConfig, load_stage2b_config

STAGE2D_CONFIG_VERSION = "stage2d.1"
EXPECTED_ORDERS = (10, 12)
EXPECTED_GRAPH_SEEDS = tuple(range(201, 209))
EXPECTED_POLICY_SEEDS = tuple(range(1001, 1033))
EXPECTED_HORIZON = 32
EXPECTED_SHARD_COUNT = 8
EXPECTED_EPISODES_PER_SHARD = 64
EXPECTED_BOOTSTRAP_SAMPLES = 10_000
EXPECTED_BOOTSTRAP_SEED = 2026072902


@dataclass(frozen=True, slots=True)
class Stage2DRunConfig:
    run_root: Path
    max_episode_bytes: int
    max_shard_bytes: int


@dataclass(frozen=True, slots=True)
class Stage2DRepositoryConfig:
    project_repo: Path
    heg_repo: Path
    frozen_stage2c_commit: str
    frozen_heg_commit: str
    preregistration_tag: str


@dataclass(frozen=True, slots=True)
class Stage2DInputConfig:
    stage2b_config: Path
    random_policy: Path
    structural_policy: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class Stage2DExperimentConfig:
    backend: str
    orders: tuple[int, ...]
    graph_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    horizon: int
    shard_count: int
    episodes_per_shard: int


@dataclass(frozen=True, slots=True)
class Stage2DStatisticsConfig:
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float
    primary_order: int
    secondary_order: int
    relative_median_threshold: float
    minimum_nonnegative_graph_seeds: int


@dataclass(frozen=True, slots=True)
class Stage2DResourceConfig:
    max_concurrent_shards: int
    minimum_reserved_physical_cores: int
    thread_count: int


@dataclass(frozen=True, slots=True)
class Stage2DIdentityConfig:
    random_source_sha256: str
    random_ast_sha256: str
    structural_source_sha256: str
    structural_ast_sha256: str


@dataclass(frozen=True, slots=True)
class Stage2DConfig:
    schema_version: str
    source_path: Path
    run: Stage2DRunConfig
    repositories: Stage2DRepositoryConfig
    inputs: Stage2DInputConfig
    experiment: Stage2DExperimentConfig
    statistics: Stage2DStatisticsConfig
    resources: Stage2DResourceConfig
    identity: Stage2DIdentityConfig
    stage2b: Stage2BConfig

    def resolved_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)
        raw.pop("source_path")
        raw.pop("stage2b")
        base = self.source_path.parent
        raw["run"]["run_root"] = os.path.relpath(self.run.run_root, base)
        raw["repositories"]["project_repo"] = os.path.relpath(
            self.repositories.project_repo, base
        )
        raw["repositories"]["heg_repo"] = os.path.relpath(
            self.repositories.heg_repo, base
        )
        raw["inputs"]["stage2b_config"] = os.path.relpath(
            self.inputs.stage2b_config, base
        )
        raw["inputs"]["random_policy"] = os.path.relpath(
            self.inputs.random_policy, base
        )
        raw["inputs"]["structural_policy"] = os.path.relpath(
            self.inputs.structural_policy, base
        )
        raw["inputs"]["manifest"] = os.path.relpath(self.inputs.manifest, base)
        raw["stage2b_config_sha256"] = hashlib.sha256(
            self.inputs.stage2b_config.read_bytes()
        ).hexdigest()
        return cast(dict[str, JsonValue], raw)

    def stable_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.resolved_dict(),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid [{name}] table")
    return cast(dict[str, Any], value)


def _exact_keys(table: dict[str, Any], expected: set[str], name: str) -> None:
    if set(table) != expected:
        raise ValueError(
            f"[{name}] keys mismatch; "
            f"missing={sorted(expected.difference(table))}, "
            f"extra={sorted(set(table).difference(expected))}"
        )


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _commit(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _int_tuple(value: object, name: str, maximum: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer array")
    result = tuple(_positive_int(item, name, maximum) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _rate(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError(f"{name} must be numeric in (0, 1)")
    return float(value)


def _require_frozen_contract(config: Stage2DConfig) -> None:
    experiment = config.experiment
    statistics = config.statistics
    stage2b = config.stage2b
    required = {
        "backend": experiment.backend == "toy",
        "orders": experiment.orders == EXPECTED_ORDERS,
        "graph_seeds": experiment.graph_seeds == EXPECTED_GRAPH_SEEDS,
        "policy_seeds": experiment.policy_seeds == EXPECTED_POLICY_SEEDS,
        "horizon": experiment.horizon == EXPECTED_HORIZON,
        "shards": experiment.shard_count == EXPECTED_SHARD_COUNT,
        "episodes_per_shard": (
            experiment.episodes_per_shard == EXPECTED_EPISODES_PER_SHARD
        ),
        "bootstrap_samples": (
            statistics.bootstrap_samples == EXPECTED_BOOTSTRAP_SAMPLES
        ),
        "bootstrap_seed": statistics.bootstrap_seed == EXPECTED_BOOTSTRAP_SEED,
        "confidence_level": statistics.confidence_level == 0.95,
        "primary_order": statistics.primary_order == 10,
        "secondary_order": statistics.secondary_order == 12,
        "relative_threshold": statistics.relative_median_threshold == 0.10,
        "graph_seed_gate": statistics.minimum_nonnegative_graph_seeds == 6,
        "pool_size": stage2b.pool.pool_size == 12,
        "k_values": stage2b.pool.k_values == (2, 3, 4),
        "witness_cap": stage2b.search.witness_cap == 64,
        "concurrency": config.resources.max_concurrent_shards == 8,
        "reserved_cores": config.resources.minimum_reserved_physical_cores == 8,
        "thread_count": config.resources.thread_count == 1,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise ValueError(f"Stage 2D frozen contract mismatch: {failed}")


def load_stage2d_config(path: str | Path) -> Stage2DConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    expected_tables = {
        "schema_version",
        "run",
        "repositories",
        "inputs",
        "experiment",
        "statistics",
        "resources",
        "identity",
    }
    if set(raw) != expected_tables:
        raise ValueError("Stage 2D config has unexpected or missing tables")
    if raw["schema_version"] != STAGE2D_CONFIG_VERSION:
        raise ValueError(f"schema_version must be {STAGE2D_CONFIG_VERSION!r}")
    run = _table(raw, "run")
    repositories = _table(raw, "repositories")
    inputs = _table(raw, "inputs")
    experiment = _table(raw, "experiment")
    statistics = _table(raw, "statistics")
    resources = _table(raw, "resources")
    identity = _table(raw, "identity")
    _exact_keys(run, {"run_root", "max_episode_bytes", "max_shard_bytes"}, "run")
    _exact_keys(
        repositories,
        {
            "project_repo",
            "heg_repo",
            "frozen_stage2c_commit",
            "frozen_heg_commit",
            "preregistration_tag",
        },
        "repositories",
    )
    _exact_keys(
        inputs,
        {"stage2b_config", "random_policy", "structural_policy", "manifest"},
        "inputs",
    )
    _exact_keys(
        experiment,
        {
            "backend",
            "orders",
            "graph_seeds",
            "policy_seeds",
            "horizon",
            "shard_count",
            "episodes_per_shard",
        },
        "experiment",
    )
    _exact_keys(
        statistics,
        {
            "bootstrap_samples",
            "bootstrap_seed",
            "confidence_level",
            "primary_order",
            "secondary_order",
            "relative_median_threshold",
            "minimum_nonnegative_graph_seeds",
        },
        "statistics",
    )
    _exact_keys(
        resources,
        {"max_concurrent_shards", "minimum_reserved_physical_cores", "thread_count"},
        "resources",
    )
    _exact_keys(
        identity,
        {
            "random_source_sha256",
            "random_ast_sha256",
            "structural_source_sha256",
            "structural_ast_sha256",
        },
        "identity",
    )
    base = source_path.parent
    backend = experiment["backend"]
    tag = repositories["preregistration_tag"]
    if backend != "toy":
        raise ValueError("experiment.backend must be 'toy'")
    if not isinstance(tag, str) or not tag or any(character.isspace() for character in tag):
        raise ValueError("repositories.preregistration_tag must be a non-empty tag")
    config = Stage2DConfig(
        schema_version=STAGE2D_CONFIG_VERSION,
        source_path=source_path,
        run=Stage2DRunConfig(
            run_root=_path(run["run_root"], "run.run_root", base),
            max_episode_bytes=_positive_int(
                run["max_episode_bytes"], "run.max_episode_bytes", 1_048_576
            ),
            max_shard_bytes=_positive_int(
                run["max_shard_bytes"], "run.max_shard_bytes", 268_435_456
            ),
        ),
        repositories=Stage2DRepositoryConfig(
            project_repo=_path(
                repositories["project_repo"], "repositories.project_repo", base
            ),
            heg_repo=_path(repositories["heg_repo"], "repositories.heg_repo", base),
            frozen_stage2c_commit=_commit(
                repositories["frozen_stage2c_commit"],
                "repositories.frozen_stage2c_commit",
            ),
            frozen_heg_commit=_commit(
                repositories["frozen_heg_commit"],
                "repositories.frozen_heg_commit",
            ),
            preregistration_tag=tag,
        ),
        inputs=Stage2DInputConfig(
            stage2b_config=_path(
                inputs["stage2b_config"], "inputs.stage2b_config", base
            ),
            random_policy=_path(inputs["random_policy"], "inputs.random_policy", base),
            structural_policy=_path(
                inputs["structural_policy"], "inputs.structural_policy", base
            ),
            manifest=_path(inputs["manifest"], "inputs.manifest", base),
        ),
        experiment=Stage2DExperimentConfig(
            backend=cast(str, backend),
            orders=_int_tuple(experiment["orders"], "experiment.orders", 100),
            graph_seeds=_int_tuple(
                experiment["graph_seeds"], "experiment.graph_seeds", 2**31 - 1
            ),
            policy_seeds=_int_tuple(
                experiment["policy_seeds"], "experiment.policy_seeds", 2**31 - 1
            ),
            horizon=_positive_int(experiment["horizon"], "experiment.horizon", 1000),
            shard_count=_positive_int(
                experiment["shard_count"], "experiment.shard_count", 64
            ),
            episodes_per_shard=_positive_int(
                experiment["episodes_per_shard"],
                "experiment.episodes_per_shard",
                10_000,
            ),
        ),
        statistics=Stage2DStatisticsConfig(
            bootstrap_samples=_positive_int(
                statistics["bootstrap_samples"],
                "statistics.bootstrap_samples",
                100_000,
            ),
            bootstrap_seed=_positive_int(
                statistics["bootstrap_seed"],
                "statistics.bootstrap_seed",
                2**31 - 1,
            ),
            confidence_level=_rate(
                statistics["confidence_level"], "statistics.confidence_level"
            ),
            primary_order=_positive_int(
                statistics["primary_order"], "statistics.primary_order", 100
            ),
            secondary_order=_positive_int(
                statistics["secondary_order"], "statistics.secondary_order", 100
            ),
            relative_median_threshold=_rate(
                statistics["relative_median_threshold"],
                "statistics.relative_median_threshold",
            ),
            minimum_nonnegative_graph_seeds=_positive_int(
                statistics["minimum_nonnegative_graph_seeds"],
                "statistics.minimum_nonnegative_graph_seeds",
                100,
            ),
        ),
        resources=Stage2DResourceConfig(
            max_concurrent_shards=_positive_int(
                resources["max_concurrent_shards"],
                "resources.max_concurrent_shards",
                8,
            ),
            minimum_reserved_physical_cores=_positive_int(
                resources["minimum_reserved_physical_cores"],
                "resources.minimum_reserved_physical_cores",
                256,
            ),
            thread_count=_positive_int(
                resources["thread_count"], "resources.thread_count", 1
            ),
        ),
        identity=Stage2DIdentityConfig(
            random_source_sha256=_sha256(
                identity["random_source_sha256"], "identity.random_source_sha256"
            ),
            random_ast_sha256=_sha256(
                identity["random_ast_sha256"], "identity.random_ast_sha256"
            ),
            structural_source_sha256=_sha256(
                identity["structural_source_sha256"],
                "identity.structural_source_sha256",
            ),
            structural_ast_sha256=_sha256(
                identity["structural_ast_sha256"],
                "identity.structural_ast_sha256",
            ),
        ),
        stage2b=load_stage2b_config(
            _path(inputs["stage2b_config"], "inputs.stage2b_config", base)
        ),
    )
    _require_frozen_contract(config)
    return config
