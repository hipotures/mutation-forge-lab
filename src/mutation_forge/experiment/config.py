"""Configuration loading for experiment workspaces.

Only invocation fields (``run.wall_seconds`` and ``run.output``) are mutable
after an experiment is created.  The raw TOML is retained so that every
session can preserve the exact bytes supplied by its caller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tomllib
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

EXPERIMENT_SCHEMA_VERSION = "mforge.experiment.v1"
MAX_EXPERIMENT_ID_BYTES = 128
_CREDENTIAL_KEY = re.compile(
    r"(?i)(?:token|password|secret|credential|auth[_-]?json|api[_-]?key|private[_-]?key)"
)


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _table(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a table")
    return cast(dict[str, Any], value)


def _strings(value: object, name: str, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        if default:
            return default
        raise ValueError(f"{name} must be a non-empty string array")
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty string array")
    return tuple(cast(list[str], value))


def _ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer array")
    return tuple(_positive_int(item, name) for item in value)


def validate_experiment_id(value: object) -> str:
    """Validate an experiment identifier without changing its spelling."""

    if not isinstance(value, str) or not value:
        raise ValueError("exp_id must be a non-empty string")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_EXPERIMENT_ID_BYTES:
        raise ValueError(f"exp_id must be at most {MAX_EXPERIMENT_ID_BYTES} UTF-8 bytes")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError("exp_id must be exactly one safe directory name")
    if Path(value).is_absolute() or os.path.splitdrive(value)[0]:
        raise ValueError("exp_id must not be an absolute path")
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise ValueError("exp_id must not contain control characters")
    # A colon is a drive/name separator on Windows and is not portable as a
    # directory identifier, even when the current host is POSIX.
    if ":" in value:
        raise ValueError("exp_id must not contain a drive separator")
    if Path(value).name != value:
        raise ValueError("exp_id must be exactly one directory name")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentRunConfig:
    wall_seconds: float
    output: str = "rich"


@dataclass(frozen=True, slots=True)
class ExperimentModelConfig:
    provider: str
    name: str
    effort: str
    concurrency: int
    max_repairs: int


@dataclass(frozen=True, slots=True)
class ExperimentSearchConfig:
    population_size: int
    max_generations: int
    max_model_turns: int
    selection: str


@dataclass(frozen=True, slots=True)
class ExperimentEvaluationConfig:
    orders: tuple[int, ...]
    graph_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    horizon: int
    proposal_pool_size: int
    baselines: tuple[str, ...]
    replay: bool


@dataclass(frozen=True, slots=True)
class ExperimentResourcesConfig:
    workers: int
    thread_count: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: str
    exp_id: str
    workspace: Path
    kind: str
    preset: str
    run: ExperimentRunConfig
    model: ExperimentModelConfig
    search: ExperimentSearchConfig
    evaluation: ExperimentEvaluationConfig
    resources: ExperimentResourcesConfig
    source_path: Path
    source_bytes: bytes = field(repr=False, compare=False)
    raw: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def source_dir(self) -> Path:
        return self.source_path.parent

    @property
    def config_path(self) -> Path:
        return self.source_path

    @property
    def workspace_path(self) -> Path:
        return self.workspace

    @property
    def resolved_workspace(self) -> Path:
        return self.workspace

    @property
    def experiment_root(self) -> Path:
        return self.workspace / self.exp_id

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_bytes).hexdigest()

    def immutable_projection(self) -> dict[str, Any]:
        """Return the canonical projection used by ``experiment.lock.json``."""

        value = copy.deepcopy(dict(self.raw))
        value.pop("run", None)
        return cast(dict[str, Any], _canonicalize_paths(value, self.source_dir))

    @property
    def immutable_config(self) -> dict[str, Any]:
        return self.immutable_projection()

    def immutable_config_sha256(self) -> str:
        payload = json.dumps(
            self.immutable_projection(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def resolved_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(dict(self.raw))
        result["workspace"] = str(self.workspace)
        result["exp_id"] = self.exp_id
        result["schema_version"] = self.schema_version
        return cast(dict[str, Any], _canonicalize_paths(result, self.source_dir))


def _canonicalize_paths(value: object, base: Path) -> object:
    """Normalize known path fields without changing scientific list order."""

    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            if isinstance(item, str) and (
                key.endswith("_path")
                or key.endswith("_file")
                or key in {"workspace", "run_root", "repo", "project_repo", "heg_repo"}
            ):
                path = Path(item)
                result[key] = str(
                    (base / path).resolve() if not path.is_absolute() else path.resolve()
                )
            else:
                result[key] = _canonicalize_paths(item, base)
        return result
    if isinstance(value, list):
        return [_canonicalize_paths(item, base) for item in value]
    return value


def _reject_credentials(value: object, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{path}.{key}" if path else str(key)
            if _CREDENTIAL_KEY.search(str(key)):
                raise ValueError(
                    f"credential field {name!r} is not allowed; use local Codex authentication"
                )
            _reject_credentials(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_credentials(item, f"{path}[{index}]")


def load_experiment_config(path: str | Path = "experiment.toml") -> ExperimentConfig:
    source_path = Path(path).resolve()
    source_bytes = source_path.read_bytes()
    raw_value = tomllib.loads(source_bytes.decode("utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("experiment configuration must be a TOML table")
    raw = raw_value
    _reject_credentials(raw)
    if raw.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {EXPERIMENT_SCHEMA_VERSION!r}, "
            f"got {raw.get('schema_version')!r}"
        )
    exp_id = validate_experiment_id(raw.get("exp_id"))
    workspace_value = raw.get("workspace")
    if not isinstance(workspace_value, str) or not workspace_value:
        raise ValueError("workspace must be a non-empty path string")
    workspace_path = Path(workspace_value)
    workspace = (
        (source_path.parent / workspace_path).resolve()
        if not workspace_path.is_absolute()
        else workspace_path.resolve()
    )
    kind = raw.get("kind")
    preset = raw.get("preset")
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    if not isinstance(preset, str) or not preset:
        raise ValueError("preset must be a non-empty string")

    run_raw = _table(raw, "run")
    unknown_run_keys = set(run_raw).difference({"wall_seconds", "output"})
    if unknown_run_keys:
        raise ValueError(f"unsupported [run] fields: {sorted(unknown_run_keys)}")
    output = run_raw.get("output", "rich")
    if output not in {"rich", "json"}:
        raise ValueError("run.output must be 'rich' or 'json'")
    run = ExperimentRunConfig(
        _positive_number(run_raw.get("wall_seconds"), "run.wall_seconds"), output
    )

    model_raw = _table(raw, "model")
    provider, name, effort = (model_raw.get(key) for key in ("provider", "name", "effort"))
    if not all(isinstance(item, str) and item for item in (provider, name, effort)):
        raise ValueError("model.provider, model.name, and model.effort must be non-empty strings")
    model = ExperimentModelConfig(
        cast(str, provider),
        cast(str, name),
        cast(str, effort),
        _positive_int(model_raw.get("concurrency"), "model.concurrency"),
        _positive_int(model_raw.get("max_repairs"), "model.max_repairs", allow_zero=True),
    )

    search_raw = _table(raw, "search")
    selection = search_raw.get("selection")
    if not isinstance(selection, str) or not selection:
        raise ValueError("search.selection must be a non-empty string")
    search = ExperimentSearchConfig(
        _positive_int(search_raw.get("population_size"), "search.population_size"),
        _positive_int(search_raw.get("max_generations"), "search.max_generations"),
        _positive_int(search_raw.get("max_model_turns"), "search.max_model_turns"),
        selection,
    )

    evaluation_raw = _table(raw, "evaluation")
    replay = evaluation_raw.get("replay")
    if not isinstance(replay, bool):
        raise ValueError("evaluation.replay must be a boolean")
    evaluation = ExperimentEvaluationConfig(
        _ints(evaluation_raw.get("orders"), "evaluation.orders"),
        _ints(evaluation_raw.get("graph_seeds"), "evaluation.graph_seeds"),
        _ints(evaluation_raw.get("policy_seeds"), "evaluation.policy_seeds"),
        _positive_int(evaluation_raw.get("horizon"), "evaluation.horizon"),
        _positive_int(evaluation_raw.get("proposal_pool_size"), "evaluation.proposal_pool_size"),
        _strings(evaluation_raw.get("baselines"), "evaluation.baselines"),
        replay,
    )

    resources_raw = _table(raw, "resources")
    resources = ExperimentResourcesConfig(
        _positive_int(resources_raw.get("workers"), "resources.workers"),
        _positive_int(resources_raw.get("thread_count"), "resources.thread_count"),
    )

    return ExperimentConfig(
        EXPERIMENT_SCHEMA_VERSION,
        exp_id,
        workspace,
        kind,
        preset,
        run,
        model,
        search,
        evaluation,
        resources,
        source_path,
        source_bytes,
        raw,
    )


# Short alias for callers that treat the experiment loader like the legacy
# stage loaders.
load_config = load_experiment_config

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "MAX_EXPERIMENT_ID_BYTES",
    "ExperimentConfig",
    "ExperimentEvaluationConfig",
    "ExperimentModelConfig",
    "ExperimentResourcesConfig",
    "ExperimentRunConfig",
    "ExperimentSearchConfig",
    "load_config",
    "load_experiment_config",
    "validate_experiment_id",
]
