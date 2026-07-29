from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.stage2b.config import Stage2BConfig, load_stage2b_config

STAGE2C_CONFIG_VERSION = "stage2c.1"


@dataclass(frozen=True, slots=True)
class Stage2CRunConfig:
    run_root: Path
    output: str
    max_artifact_bytes: int
    max_record_bytes: int
    record_shard_bytes: int
    max_record_count: int
    max_record_total_bytes: int


@dataclass(frozen=True, slots=True)
class Stage2CRepositoryConfig:
    project_repo: Path
    heg_repo: Path
    frozen_project_commit: str
    frozen_heg_commit: str


@dataclass(frozen=True, slots=True)
class Stage2CControlConfig:
    stage2b_config: Path
    durable_result: Path
    expected_config_hash: str
    expected_behavior_hash: str
    expected_random_median_auc: float
    expected_structural_median_auc: float
    expected_relative_improvement: float
    expected_ci: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Stage2CMatrixConfig:
    orders: tuple[int, ...]
    graph_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    horizons: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Stage2CDiagnosticLimits:
    top_k_values: tuple[int, ...]
    feature_sample_cap: int
    distinct_value_cap: int
    near_constant_epsilon: float


@dataclass(frozen=True, slots=True)
class Stage2CConfig:
    schema_version: str
    source_path: Path
    run: Stage2CRunConfig
    repositories: Stage2CRepositoryConfig
    control: Stage2CControlConfig
    matrix: Stage2CMatrixConfig
    diagnostics: Stage2CDiagnosticLimits
    stage2b: Stage2BConfig

    def resolved_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)
        raw.pop("source_path")
        raw.pop("stage2b")
        raw["run"]["run_root"] = str(self.run.run_root)
        raw["repositories"]["project_repo"] = str(self.repositories.project_repo)
        raw["repositories"]["heg_repo"] = str(self.repositories.heg_repo)
        raw["control"]["stage2b_config"] = str(self.control.stage2b_config)
        raw["control"]["durable_result"] = str(self.control.durable_result)
        raw["control"]["expected_ci"] = list(self.control.expected_ci)
        raw["stage2b_config_hash"] = self.stage2b.stable_hash()
        return cast(dict[str, JsonValue], raw)

    def stable_hash(self) -> str:
        encoded = json.dumps(
            self.resolved_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


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


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _commit(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return value


def _positive_int(value: object, name: str, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _int_tuple(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer array")
    result: list[int] = []
    for item in value:
        if (
            not isinstance(item, int)
            or isinstance(item, bool)
            or not minimum <= item <= maximum
        ):
            raise ValueError(f"{name} values must be in [{minimum}, {maximum}]")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(result)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not -1.0e12 <= result <= 1.0e12:
        raise ValueError(f"{name} must be finite and bounded")
    return result


def load_stage2c_config(path: str | Path) -> Stage2CConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    expected = {
        "schema_version",
        "run",
        "repositories",
        "control",
        "matrix",
        "diagnostics",
    }
    if set(raw) != expected:
        raise ValueError("Stage 2C config has unexpected or missing tables")
    if raw["schema_version"] != STAGE2C_CONFIG_VERSION:
        raise ValueError(f"schema_version must be {STAGE2C_CONFIG_VERSION!r}")
    run = _table(raw, "run")
    repositories = _table(raw, "repositories")
    control = _table(raw, "control")
    matrix = _table(raw, "matrix")
    diagnostics = _table(raw, "diagnostics")
    _exact_keys(
        run,
        {
            "run_root",
            "output",
            "max_artifact_bytes",
            "max_record_bytes",
            "record_shard_bytes",
            "max_record_count",
            "max_record_total_bytes",
        },
        "run",
    )
    _exact_keys(
        repositories,
        {
            "project_repo",
            "heg_repo",
            "frozen_project_commit",
            "frozen_heg_commit",
        },
        "repositories",
    )
    _exact_keys(
        control,
        {
            "stage2b_config",
            "durable_result",
            "expected_config_hash",
            "expected_behavior_hash",
            "expected_random_median_auc",
            "expected_structural_median_auc",
            "expected_relative_improvement",
            "expected_ci",
        },
        "control",
    )
    _exact_keys(
        matrix,
        {"orders", "graph_seeds", "policy_seeds", "horizons"},
        "matrix",
    )
    _exact_keys(
        diagnostics,
        {
            "top_k_values",
            "feature_sample_cap",
            "distinct_value_cap",
            "near_constant_epsilon",
        },
        "diagnostics",
    )
    output = run["output"]
    if output not in {"rich", "json"}:
        raise ValueError("run.output must be 'rich' or 'json'")
    base = source_path.parent
    stage2b_path = _path(control["stage2b_config"], "control.stage2b_config", base)
    expected_ci_raw = control["expected_ci"]
    if not isinstance(expected_ci_raw, list) or len(expected_ci_raw) != 2:
        raise ValueError("control.expected_ci must contain exactly two numbers")
    expected_ci = tuple(
        _finite_number(item, "control.expected_ci") for item in expected_ci_raw
    )
    near_constant = _finite_number(
        diagnostics["near_constant_epsilon"],
        "diagnostics.near_constant_epsilon",
    )
    if not 0.0 <= near_constant <= 1.0:
        raise ValueError("diagnostics.near_constant_epsilon must be in [0, 1]")
    result = Stage2CConfig(
        schema_version=STAGE2C_CONFIG_VERSION,
        source_path=source_path,
        run=Stage2CRunConfig(
            run_root=_path(run["run_root"], "run.run_root", base),
            output=cast(str, output),
            max_artifact_bytes=_positive_int(
                run["max_artifact_bytes"],
                "run.max_artifact_bytes",
                maximum=16 * 1024 * 1024,
            ),
            max_record_bytes=_positive_int(
                run["max_record_bytes"],
                "run.max_record_bytes",
                maximum=1024 * 1024,
            ),
            record_shard_bytes=_positive_int(
                run["record_shard_bytes"],
                "run.record_shard_bytes",
                maximum=16 * 1024 * 1024,
            ),
            max_record_count=_positive_int(
                run["max_record_count"],
                "run.max_record_count",
                maximum=1_000_000,
            ),
            max_record_total_bytes=_positive_int(
                run["max_record_total_bytes"],
                "run.max_record_total_bytes",
                maximum=1024 * 1024 * 1024,
            ),
        ),
        repositories=Stage2CRepositoryConfig(
            project_repo=_path(
                repositories["project_repo"],
                "repositories.project_repo",
                base,
            ),
            heg_repo=_path(
                repositories["heg_repo"],
                "repositories.heg_repo",
                base,
            ),
            frozen_project_commit=_commit(
                repositories["frozen_project_commit"],
                "repositories.frozen_project_commit",
            ),
            frozen_heg_commit=_commit(
                repositories["frozen_heg_commit"],
                "repositories.frozen_heg_commit",
            ),
        ),
        control=Stage2CControlConfig(
            stage2b_config=stage2b_path,
            durable_result=_path(
                control["durable_result"],
                "control.durable_result",
                base,
            ),
            expected_config_hash=_sha(
                control["expected_config_hash"],
                "control.expected_config_hash",
            ),
            expected_behavior_hash=_sha(
                control["expected_behavior_hash"],
                "control.expected_behavior_hash",
            ),
            expected_random_median_auc=_finite_number(
                control["expected_random_median_auc"],
                "control.expected_random_median_auc",
            ),
            expected_structural_median_auc=_finite_number(
                control["expected_structural_median_auc"],
                "control.expected_structural_median_auc",
            ),
            expected_relative_improvement=_finite_number(
                control["expected_relative_improvement"],
                "control.expected_relative_improvement",
            ),
            expected_ci=cast(tuple[float, float], expected_ci),
        ),
        matrix=Stage2CMatrixConfig(
            orders=_int_tuple(
                matrix["orders"],
                "matrix.orders",
                minimum=4,
                maximum=100,
            ),
            graph_seeds=_int_tuple(
                matrix["graph_seeds"],
                "matrix.graph_seeds",
                minimum=1,
                maximum=2**31 - 1,
            ),
            policy_seeds=_int_tuple(
                matrix["policy_seeds"],
                "matrix.policy_seeds",
                minimum=1,
                maximum=2**31 - 1,
            ),
            horizons=_int_tuple(
                matrix["horizons"],
                "matrix.horizons",
                minimum=1,
                maximum=1000,
            ),
        ),
        diagnostics=Stage2CDiagnosticLimits(
            top_k_values=_int_tuple(
                diagnostics["top_k_values"],
                "diagnostics.top_k_values",
                minimum=1,
                maximum=16,
            ),
            feature_sample_cap=_positive_int(
                diagnostics["feature_sample_cap"],
                "diagnostics.feature_sample_cap",
                maximum=100_000,
            ),
            distinct_value_cap=_positive_int(
                diagnostics["distinct_value_cap"],
                "diagnostics.distinct_value_cap",
                maximum=100_000,
            ),
            near_constant_epsilon=near_constant,
        ),
        stage2b=load_stage2b_config(stage2b_path),
    )
    if result.matrix.orders != (8, 10, 12):
        raise ValueError("matrix.orders must remain frozen at [8, 10, 12]")
    if result.matrix.graph_seeds != (101, 102, 103, 104):
        raise ValueError("matrix.graph_seeds must remain frozen at [101, 102, 103, 104]")
    if result.matrix.policy_seeds != tuple(range(1, 33)):
        raise ValueError("matrix.policy_seeds must remain frozen at 1..32")
    if result.matrix.horizons != (8, 16, 32):
        raise ValueError("matrix.horizons must remain frozen at [8, 16, 32]")
    if result.stage2b.stable_hash() != result.control.expected_config_hash:
        raise ValueError("Stage 2B control config hash does not match the frozen expectation")
    if result.run.record_shard_bytes > result.run.max_record_total_bytes:
        raise ValueError("record shard bound cannot exceed total record bound")
    return result
