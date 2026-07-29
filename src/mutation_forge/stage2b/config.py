from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.proposals.k_switch import (
    SUPPORTED_SELECTORS,
    FeatureLimits,
    PoolLimits,
)
from mutation_forge.sandbox.contracts import SandboxLimits

STAGE2B_CONFIG_VERSION = "stage2b.1"


@dataclass(frozen=True, slots=True)
class Stage2BRunConfig:
    run_root: Path
    output: str


@dataclass(frozen=True, slots=True)
class Stage2BRepositoryConfig:
    project_repo: Path
    heg_repo: Path
    frozen_project_commit: str
    frozen_heg_commit: str


@dataclass(frozen=True, slots=True)
class Stage2BSearchConfig:
    steps: int
    witness_cap: int


@dataclass(frozen=True, slots=True)
class ToyGateConfig:
    order: int
    graph_seed: int
    policy_seeds: tuple[int, ...]
    bootstrap_samples: int
    auc_relative_improvement_threshold: float
    confidence_level: float


@dataclass(frozen=True, slots=True)
class HegPilotConfig:
    enabled: bool
    order: int
    graph_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    steps: int


@dataclass(frozen=True, slots=True)
class Stage2BConfig:
    schema_version: str
    source_path: Path
    run: Stage2BRunConfig
    repositories: Stage2BRepositoryConfig
    pool: PoolLimits
    features: FeatureLimits
    search: Stage2BSearchConfig
    toy_gate: ToyGateConfig
    heg_pilot: HegPilotConfig
    sandbox: SandboxLimits

    def resolved_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)
        raw.pop("source_path")
        raw["run"]["run_root"] = str(self.run.run_root)
        raw["repositories"]["project_repo"] = str(self.repositories.project_repo)
        raw["repositories"]["heg_repo"] = str(self.repositories.heg_repo)
        return cast(dict[str, JsonValue], raw)

    def stable_hash(self) -> str:
        encoded = json.dumps(
            self.resolved_dict(),
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


def _positive_int(value: object, name: str, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _positive_tuple(
    value: object,
    name: str,
    *,
    maximum: int,
    unique: bool = True,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer array")
    result = tuple(_positive_int(item, name, maximum=maximum) for item in value)
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty string array")
    return tuple(cast(list[str], value))


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


def _rate(value: object, name: str, *, lower: float, upper: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not lower < float(value) < upper
    ):
        raise ValueError(f"{name} must be numeric in ({lower}, {upper})")
    return float(value)


def load_stage2b_config(path: str | Path) -> Stage2BConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    expected_tables = {
        "schema_version",
        "run",
        "repositories",
        "pool",
        "features",
        "search",
        "toy_gate",
        "heg_pilot",
        "sandbox",
    }
    if set(raw) != expected_tables:
        raise ValueError("Stage 2B config has unexpected or missing tables")
    if raw["schema_version"] != STAGE2B_CONFIG_VERSION:
        raise ValueError(f"schema_version must be {STAGE2B_CONFIG_VERSION!r}")
    run = _table(raw, "run")
    repositories = _table(raw, "repositories")
    pool = _table(raw, "pool")
    features_raw = _table(raw, "features")
    search = _table(raw, "search")
    toy = _table(raw, "toy_gate")
    pilot = _table(raw, "heg_pilot")
    sandbox_raw = _table(raw, "sandbox")
    _exact_keys(run, {"run_root", "output"}, "run")
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
        pool,
        {
            "pool_size",
            "k_values",
            "selectors",
            "selector_weights",
            "retry_limit",
            "matching_limit",
        },
        "pool",
    )
    _exact_keys(
        features_raw,
        {
            "forbidden_lengths",
            "witness_sample_cap",
            "cycle_node_budget",
            "distance_query_budget",
            "local_risk_budget",
        },
        "features",
    )
    _exact_keys(search, {"steps", "witness_cap"}, "search")
    _exact_keys(
        toy,
        {
            "order",
            "graph_seed",
            "policy_seeds",
            "bootstrap_samples",
            "auc_relative_improvement_threshold",
            "confidence_level",
        },
        "toy_gate",
    )
    _exact_keys(
        pilot,
        {"enabled", "order", "graph_seeds", "policy_seeds", "steps"},
        "heg_pilot",
    )
    expected_sandbox = {field.name for field in fields(SandboxLimits)}
    _exact_keys(sandbox_raw, expected_sandbox, "sandbox")
    output = run["output"]
    if output not in {"rich", "json"}:
        raise ValueError("run.output must be 'rich' or 'json'")
    selectors = _string_tuple(pool["selectors"], "pool.selectors")
    if any(selector not in SUPPORTED_SELECTORS for selector in selectors):
        raise ValueError("pool.selectors contains an unsupported selector")
    selector_weights = _positive_tuple(
        pool["selector_weights"],
        "pool.selector_weights",
        maximum=100,
        unique=False,
    )
    if len(selector_weights) != len(selectors):
        raise ValueError("pool.selector_weights must align with selectors")
    toy_policy_seeds = _positive_tuple(
        toy["policy_seeds"],
        "toy_gate.policy_seeds",
        maximum=2**31 - 1,
    )
    if len(toy_policy_seeds) < 32:
        raise ValueError("toy_gate.policy_seeds must contain at least 32 seeds")
    base = source_path.parent
    enabled = pilot["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("heg_pilot.enabled must be a boolean")
    sandbox = SandboxLimits(**sandbox_raw)
    for field in fields(SandboxLimits):
        value = getattr(sandbox, field.name)
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"sandbox.{field.name} must be positive")
    return Stage2BConfig(
        schema_version=STAGE2B_CONFIG_VERSION,
        source_path=source_path,
        run=Stage2BRunConfig(
            run_root=_path(run["run_root"], "run.run_root", base),
            output=cast(str, output),
        ),
        repositories=Stage2BRepositoryConfig(
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
        pool=PoolLimits(
            pool_size=_positive_int(
                pool["pool_size"],
                "pool.pool_size",
                maximum=64,
            ),
            k_values=_positive_tuple(
                pool["k_values"],
                "pool.k_values",
                maximum=4,
            ),
            selectors=selectors,
            selector_weights=selector_weights,
            retry_limit=_positive_int(
                pool["retry_limit"],
                "pool.retry_limit",
                maximum=1_024,
            ),
            matching_limit=_positive_int(
                pool["matching_limit"],
                "pool.matching_limit",
                maximum=105,
            ),
        ),
        features=FeatureLimits(
            forbidden_lengths=_positive_tuple(
                features_raw["forbidden_lengths"],
                "features.forbidden_lengths",
                maximum=16,
            ),
            witness_sample_cap=_positive_int(
                features_raw["witness_sample_cap"],
                "features.witness_sample_cap",
                maximum=256,
            ),
            cycle_node_budget=_positive_int(
                features_raw["cycle_node_budget"],
                "features.cycle_node_budget",
                maximum=1_000_000,
            ),
            distance_query_budget=_positive_int(
                features_raw["distance_query_budget"],
                "features.distance_query_budget",
                maximum=4_096,
            ),
            local_risk_budget=_positive_int(
                features_raw["local_risk_budget"],
                "features.local_risk_budget",
                maximum=100_000,
            ),
        ),
        search=Stage2BSearchConfig(
            steps=_positive_int(search["steps"], "search.steps", maximum=1_000),
            witness_cap=_positive_int(
                search["witness_cap"],
                "search.witness_cap",
                maximum=1_000_000,
            ),
        ),
        toy_gate=ToyGateConfig(
            order=_positive_int(toy["order"], "toy_gate.order", maximum=100),
            graph_seed=_positive_int(
                toy["graph_seed"],
                "toy_gate.graph_seed",
                maximum=2**31 - 1,
            ),
            policy_seeds=toy_policy_seeds,
            bootstrap_samples=_positive_int(
                toy["bootstrap_samples"],
                "toy_gate.bootstrap_samples",
                maximum=100_000,
            ),
            auc_relative_improvement_threshold=_rate(
                toy["auc_relative_improvement_threshold"],
                "toy_gate.auc_relative_improvement_threshold",
                lower=0.0,
                upper=10.0,
            ),
            confidence_level=_rate(
                toy["confidence_level"],
                "toy_gate.confidence_level",
                lower=0.5,
                upper=1.0,
            ),
        ),
        heg_pilot=HegPilotConfig(
            enabled=enabled,
            order=_positive_int(pilot["order"], "heg_pilot.order", maximum=100),
            graph_seeds=_positive_tuple(
                pilot["graph_seeds"],
                "heg_pilot.graph_seeds",
                maximum=2**31 - 1,
            ),
            policy_seeds=_positive_tuple(
                pilot["policy_seeds"],
                "heg_pilot.policy_seeds",
                maximum=2**31 - 1,
            ),
            steps=_positive_int(
                pilot["steps"],
                "heg_pilot.steps",
                maximum=100,
            ),
        ),
        sandbox=sandbox,
    )
