from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from mutation_forge.sandbox.contracts import SandboxLimits

POLICY_CONFIG_SCHEMA_VERSION = "stage2a.1"


@dataclass(frozen=True, slots=True)
class PolicyEvaluationConfig:
    source_path: Path
    run_root: Path
    output: str
    project_repo: Path
    heg_repo: Path
    frozen_project_commit: str
    frozen_heg_commit: str
    limits: SandboxLimits


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid [{name}] table")
    return value


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


def load_policy_config(path: str | Path) -> PolicyEvaluationConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != POLICY_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {POLICY_CONFIG_SCHEMA_VERSION!r}"
        )
    if set(raw) != {"schema_version", "run", "repositories", "sandbox"}:
        raise ValueError(
            "Stage 2A config tables must be schema_version, run, repositories, sandbox"
        )
    run = _table(raw, "run")
    repositories = _table(raw, "repositories")
    sandbox = _table(raw, "sandbox")
    if set(run) != {"run_root", "output"}:
        raise ValueError("[run] must contain exactly run_root and output")
    if set(repositories) != {
        "project_repo",
        "heg_repo",
        "frozen_project_commit",
        "frozen_heg_commit",
    }:
        raise ValueError("[repositories] contains unexpected or missing keys")
    expected_limit_names = {field.name for field in fields(SandboxLimits)}
    if set(sandbox) != expected_limit_names:
        missing = sorted(expected_limit_names.difference(sandbox))
        extra = sorted(set(sandbox).difference(expected_limit_names))
        raise ValueError(f"[sandbox] keys mismatch; missing={missing}, extra={extra}")
    output = run["output"]
    if output not in {"rich", "json"}:
        raise ValueError("run.output must be 'rich' or 'json'")
    limits = SandboxLimits(**sandbox)
    for field in fields(SandboxLimits):
        value = getattr(limits, field.name)
        if isinstance(field.default, int):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"sandbox.{field.name} must be a positive integer")
        elif (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"sandbox.{field.name} must be positive numeric")
    base = source_path.parent
    return PolicyEvaluationConfig(
        source_path=source_path,
        run_root=_path(run["run_root"], "run.run_root", base),
        output=output,
        project_repo=_path(
            repositories["project_repo"],
            "repositories.project_repo",
            base,
        ),
        heg_repo=_path(repositories["heg_repo"], "repositories.heg_repo", base),
        frozen_project_commit=_commit(
            repositories["frozen_project_commit"],
            "repositories.frozen_project_commit",
        ),
        frozen_heg_commit=_commit(
            repositories["frozen_heg_commit"],
            "repositories.frozen_heg_commit",
        ),
        limits=limits,
    )
