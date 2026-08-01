# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.stage7_heg_bridge.contract import (
    BENCHMARK_SCHEMA_VERSION,
    CATALOG_ID,
    CONTRACT_SCHEMA_VERSION,
    HEG_COMMIT,
    PROJECT_ENTRY_COMMIT,
    REDTEAM_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
)

CONFIG_SCHEMA_VERSION = "stage7.heg.integration.config.v1"


@dataclass(frozen=True, slots=True)
class Stage7Config:
    config_path: Path
    project_repo: Path
    heg_repo: Path
    evidence_root: Path
    capability_matrix_path: Path
    contract_path: Path
    identity_path: Path
    fixture_path: Path
    replay_path: Path
    redteam_path: Path
    benchmark_path: Path
    replay_records: int
    benchmark_calls: int
    sandbox: SandboxLimits

    def resolved_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "project_repo": str(self.project_repo),
            "heg_repo": str(self.heg_repo),
            "evidence_root": str(self.evidence_root),
            "capability_matrix_path": str(self.capability_matrix_path),
            "contract_path": str(self.contract_path),
            "identity_path": str(self.identity_path),
            "fixture_path": str(self.fixture_path),
            "replay_path": str(self.replay_path),
            "redteam_path": str(self.redteam_path),
            "benchmark_path": str(self.benchmark_path),
            "replay_records": self.replay_records,
            "benchmark_calls": self.benchmark_calls,
            "sandbox": self.sandbox.as_dict(),
        }

    def stable_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.resolved_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return cast(dict[str, Any], value)


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase 40-character SHA-1")
    return value


def load_stage7_config(path: str | Path) -> Stage7Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    expected = {"schema_version", "entry", "paths", "replay", "benchmark", "sandbox"}
    if set(raw) != expected:
        raise ValueError(f"Stage 7 config keys mismatch: {sorted(set(raw) ^ expected)}")
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    entry = _table(raw, "entry")
    paths = _table(raw, "paths")
    replay = _table(raw, "replay")
    benchmark = _table(raw, "benchmark")
    sandbox_raw = _table(raw, "sandbox")
    if set(entry) != {"mutation_forge_commit", "heg_commit", "catalog_id", "contract_schema_version", "replay_schema_version", "redteam_schema_version", "benchmark_schema_version"}:
        raise ValueError("[entry] keys mismatch")
    if _sha(entry["mutation_forge_commit"], "entry.mutation_forge_commit") != PROJECT_ENTRY_COMMIT:
        raise ValueError("entry Mutation Forge commit is not the required issue state")
    if _sha(entry["heg_commit"], "entry.heg_commit") != HEG_COMMIT:
        raise ValueError("entry HEG commit is not the required issue state")
    if entry["catalog_id"] != CATALOG_ID or entry["contract_schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ValueError("frozen catalog/contract identity mismatch")
    if entry["replay_schema_version"] != REPLAY_SCHEMA_VERSION or entry["redteam_schema_version"] != REDTEAM_SCHEMA_VERSION or entry["benchmark_schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("frozen Stage 7 schema identity mismatch")
    expected_paths = {"capability_matrix", "contract", "identity", "fixtures", "replay", "redteam", "benchmark", "evidence_root"}
    if set(paths) != expected_paths:
        raise ValueError("[paths] keys mismatch")
    if set(replay) != {"record_count"} or not isinstance(replay["record_count"], int) or replay["record_count"] < 2048:
        raise ValueError("replay.record_count must be at least 2048")
    if set(benchmark) != {"policy_calls"} or not isinstance(benchmark["policy_calls"], int) or benchmark["policy_calls"] < 100_000:
        raise ValueError("benchmark.policy_calls must be at least 100000")
    sandbox_expected = set(SandboxLimits.__dataclass_fields__)
    if set(sandbox_raw) != sandbox_expected:
        raise ValueError("[sandbox] keys mismatch")
    sandbox = SandboxLimits(**sandbox_raw)
    base = config_path.parent.parent
    return Stage7Config(
        config_path=config_path,
        project_repo=base,
        heg_repo=_path("../heg", "heg_repo", base),
        evidence_root=_path(paths["evidence_root"], "paths.evidence_root", base),
        capability_matrix_path=_path(paths["capability_matrix"], "paths.capability_matrix", base),
        contract_path=_path(paths["contract"], "paths.contract", base),
        identity_path=_path(paths["identity"], "paths.identity", base),
        fixture_path=_path(paths["fixtures"], "paths.fixtures", base),
        replay_path=_path(paths["replay"], "paths.replay", base),
        redteam_path=_path(paths["redteam"], "paths.redteam", base),
        benchmark_path=_path(paths["benchmark"], "paths.benchmark", base),
        replay_records=int(replay["record_count"]),
        benchmark_calls=int(benchmark["policy_calls"]),
        sandbox=sandbox,
    )


load_config = load_stage7_config
