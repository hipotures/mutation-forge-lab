from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue

CONFIG_SCHEMA_VERSION = "1.0"
SUPPORTED_OPERATORS = (
    "heg_uniform_two_switch",
    "heg_forbidden_cycle_break",
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int
    wall_seconds: float
    output: str
    run_root: Path


@dataclass(frozen=True, slots=True)
class HegConfig:
    repo: Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    orders: tuple[int, ...]
    graph_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    split: str


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    witness_cap: int


@dataclass(frozen=True, slots=True)
class SearchConfig:
    controller: str
    evaluations_per_episode: int
    proposal_pool_size: int
    profiling_enabled: bool
    deep_profiling_enabled: bool
    score_cache_enabled: bool
    score_cutoff_enabled: bool
    prepared_graph_cache_enabled: bool
    prepared_proposal_handoff_enabled: bool
    score_longest_first_enabled: bool
    score_compact_dominated_enabled: bool


@dataclass(frozen=True, slots=True)
class ProposalConfig:
    operator_families: tuple[str, ...]
    k_values: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LabConfig:
    schema_version: str
    run: RunConfig
    heg: HegConfig
    dataset: DatasetConfig
    score: ScoreConfig
    search: SearchConfig
    proposals: ProposalConfig
    source_path: Path

    def resolved_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)
        raw.pop("source_path")
        raw["run"]["run_root"] = str(self.run.run_root)
        raw["heg"]["repo"] = str(self.heg.repo)
        return cast(dict[str, JsonValue], raw)

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.resolved_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid [{name}] table")
    return cast(dict[str, Any], value)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer array")
    return tuple(_positive_int(item, name) for item in value)


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty string array")
    return tuple(cast(list[str], value))


def load_config(path: str | Path) -> LabConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    schema_version = raw.get("schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    run = _table(raw, "run")
    heg = _table(raw, "heg")
    dataset = _table(raw, "dataset")
    score = _table(raw, "score")
    search = _table(raw, "search")
    proposals = _table(raw, "proposals")

    output = run.get("output")
    if output not in {"rich", "json"}:
        raise ValueError("run.output must be 'rich' or 'json'")
    controller = search.get("controller")
    if controller != "fixed_ils_tabu":
        raise ValueError("Stage 1 supports only search.controller='fixed_ils_tabu'")
    profiling_enabled = search.get("profiling_enabled", True)
    if not isinstance(profiling_enabled, bool):
        raise ValueError("search.profiling_enabled must be a boolean")
    deep_profiling_enabled = search.get("deep_profiling_enabled", False)
    if not isinstance(deep_profiling_enabled, bool):
        raise ValueError("search.deep_profiling_enabled must be a boolean")
    score_cache_enabled = search.get("score_cache_enabled", True)
    if not isinstance(score_cache_enabled, bool):
        raise ValueError("search.score_cache_enabled must be a boolean")
    score_cutoff_enabled = search.get("score_cutoff_enabled", True)
    if not isinstance(score_cutoff_enabled, bool):
        raise ValueError("search.score_cutoff_enabled must be a boolean")
    prepared_graph_cache_enabled = search.get("prepared_graph_cache_enabled", True)
    if not isinstance(prepared_graph_cache_enabled, bool):
        raise ValueError("search.prepared_graph_cache_enabled must be a boolean")
    prepared_proposal_handoff_enabled = search.get(
        "prepared_proposal_handoff_enabled",
        True,
    )
    if not isinstance(prepared_proposal_handoff_enabled, bool):
        raise ValueError(
            "search.prepared_proposal_handoff_enabled must be a boolean"
        )
    score_longest_first_enabled = search.get(
        "score_longest_first_enabled",
        True,
    )
    if not isinstance(score_longest_first_enabled, bool):
        raise ValueError(
            "search.score_longest_first_enabled must be a boolean"
        )
    score_compact_dominated_enabled = search.get(
        "score_compact_dominated_enabled",
        True,
    )
    if not isinstance(score_compact_dominated_enabled, bool):
        raise ValueError(
            "search.score_compact_dominated_enabled must be a boolean"
        )
    operator_families = _str_tuple(
        proposals.get("operator_families"), "proposals.operator_families"
    )
    unsupported = set(operator_families).difference(SUPPORTED_OPERATORS)
    if unsupported:
        raise ValueError(f"unsupported Stage 1 operator families: {sorted(unsupported)}")
    k_values = _int_tuple(proposals.get("k_values"), "proposals.k_values")
    if k_values != (2,):
        raise ValueError("Stage 1 supports only proposals.k_values=[2]")

    orders = _int_tuple(dataset.get("orders"), "dataset.orders")
    if any(order < 4 or order % 2 for order in orders):
        raise ValueError("connected cubic dataset orders must be even and at least 4")

    wall_seconds = run.get("wall_seconds")
    if not isinstance(wall_seconds, int | float) or isinstance(wall_seconds, bool):
        raise ValueError("run.wall_seconds must be numeric")
    if wall_seconds <= 0:
        raise ValueError("run.wall_seconds must be positive")

    repo_value = heg.get("repo")
    root_value = run.get("run_root")
    split = dataset.get("split")
    if not isinstance(repo_value, str) or not repo_value:
        raise ValueError("heg.repo must be a path string")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("run.run_root must be a path string")
    if not isinstance(split, str) or not split:
        raise ValueError("dataset.split must be a non-empty string")

    return LabConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        run=RunConfig(
            seed=_positive_int(run.get("seed"), "run.seed"),
            wall_seconds=float(wall_seconds),
            output=cast(str, output),
            run_root=Path(root_value).resolve(),
        ),
        heg=HegConfig(repo=Path(repo_value).resolve()),
        dataset=DatasetConfig(
            orders=orders,
            graph_seeds=_int_tuple(dataset.get("graph_seeds"), "dataset.graph_seeds"),
            policy_seeds=_int_tuple(dataset.get("policy_seeds"), "dataset.policy_seeds"),
            split=split,
        ),
        score=ScoreConfig(
            witness_cap=_positive_int(score.get("witness_cap"), "score.witness_cap")
        ),
        search=SearchConfig(
            controller=cast(str, controller),
            evaluations_per_episode=_positive_int(
                search.get("evaluations_per_episode"),
                "search.evaluations_per_episode",
            ),
            proposal_pool_size=_positive_int(
                search.get("proposal_pool_size"), "search.proposal_pool_size"
            ),
            profiling_enabled=profiling_enabled,
            deep_profiling_enabled=deep_profiling_enabled,
            score_cache_enabled=score_cache_enabled,
            score_cutoff_enabled=score_cutoff_enabled,
            prepared_graph_cache_enabled=prepared_graph_cache_enabled,
            prepared_proposal_handoff_enabled=(
                prepared_proposal_handoff_enabled
            ),
            score_longest_first_enabled=score_longest_first_enabled,
            score_compact_dominated_enabled=(
                score_compact_dominated_enabled
            ),
        ),
        proposals=ProposalConfig(
            operator_families=operator_families,
            k_values=k_values,
        ),
        source_path=source_path,
    )
