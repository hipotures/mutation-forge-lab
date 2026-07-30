"""Strict loader for frozen Stage 4 search configuration."""
# Frozen key lines intentionally remain readable.
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.stage2b.config import Stage2BConfig, load_stage2b_config

from .contracts import load_seed_manifest

STAGE4_CONFIG_VERSION = "stage4.search.v1"
EXPECTED_MODEL, EXPECTED_EFFORT = "gpt-5.6-luna", "high"
EXPECTED_PROJECT_COMMIT = "49b4a611a91f87bde0b7b7be2f97c5deec8f1e89"
EXPECTED_HEG_COMMIT = "fd97451b0f3d87400d1d955a2c6b1b18303344ff"
EXPECTED_SEARCH_TAG = "stage4-search-frozen-v1"
EXPECTED_VALIDATION_TAG = "stage4-validation-frozen-v1"


@dataclass(frozen=True, slots=True)
class Stage4ModelConfig:
    name: str
    effort: str
    generations: int
    slots: int
    concurrency: int
    max_repairs: int
    max_initial_turns: int
    max_accepted_turns: int


@dataclass(frozen=True, slots=True)
class Stage4Limits:
    max_generation_workers: int
    max_evaluation_workers: int
    reserved_physical_cores: int
    thread_count: int
    artifact_uncompressed_shard_bytes: int
    artifact_uncompressed_campaign_bytes: int
    artifact_compressed_campaign_bytes: int
    dry_run_max_fraction: float


@dataclass(frozen=True, slots=True)
class Stage4Evaluation:
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float


@dataclass(frozen=True, slots=True)
class Stage4Experiment:
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
class Stage4Identity:
    manifest_sha256: str
    validation_manifest_sha256: str
    seed_manifest_sha256: str
    system_prompt_sha256: str
    request_prompt_sha256: str
    repair_prompt_sha256: str
    output_schema_sha256: str
    freeze_payload_sha256: str


@dataclass(frozen=True, slots=True)
class Stage4SearchConfig:
    schema_version: str
    source_path: Path
    run_root: Path
    model: Stage4ModelConfig
    limits: Stage4Limits
    evaluation: Stage4Evaluation
    experiment: Stage4Experiment
    stage2b: Stage2BConfig
    sandbox: SandboxLimits
    stage2b_config_path: Path
    random_policy_path: Path
    structural_policy_path: Path
    context_schema_path: Path
    proposal_schema_path: Path
    semantic_glossary_path: Path
    stage3_source_run: Path
    seed_manifest_path: Path
    manifest_path: Path
    validation_manifest_path: Path
    briefs_dir: Path
    system_prompt_path: Path
    request_prompt_path: Path
    repair_prompt_path: Path
    output_schema_path: Path
    project_repo: Path
    heg_repo: Path
    frozen_project_commit: str
    frozen_heg_commit: str
    search_tag: str
    validation_tag: str
    identity: Stage4Identity

    def resolved_dict(self) -> dict[str, JsonValue]:
        def norm(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): norm(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [norm(v) for v in value]
            return value
        result = cast(dict[str, JsonValue], norm(asdict(self)))
        result.pop("source_path", None)
        return result

    def stable_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.resolved_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return cast(dict[str, Any], value)


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a path")
    p = Path(value)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ints(value: object, expected: tuple[int, ...], name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or tuple(value) != expected or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise ValueError(f"{name} is frozen as {expected}")
    return expected


def load_stage4_config(path: str | Path = "configs/stage4-search.toml") -> Stage4SearchConfig:
    source_path = Path(path).resolve()
    raw = tomllib.loads(source_path.read_text())
    if raw.get("schema_version") != STAGE4_CONFIG_VERSION:
        raise ValueError("invalid Stage 4 config schema_version")
    base = source_path.parent
    run, model, limits, resources, evaluation, experiment, inputs, repos, identity = (_table(raw, n) for n in ("run", "model", "limits", "resources", "evaluation", "experiment", "inputs", "repositories", "identity"))
    expected_tables = {"run", "model", "limits", "resources", "evaluation", "experiment", "inputs", "repositories", "identity"}
    if set(raw) != expected_tables | {"schema_version"}:
        raise ValueError("invalid Stage 4 config root keys")
    table_keys = {
        "run": {"run_root"},
        "limits": {"artifact_uncompressed_shard_bytes", "artifact_uncompressed_campaign_bytes", "artifact_compressed_campaign_bytes", "dry_run_max_fraction"},
        "resources": {"max_generation_workers", "max_evaluation_workers", "reserved_physical_cores", "thread_count"},
        "evaluation": {"bootstrap_samples", "bootstrap_seed", "confidence_level"},
        "experiment": {"orders", "graph_seeds", "policy_seeds", "horizon", "shard_count", "episodes_per_shard"},
        "inputs": {"stage2b_config", "random_policy", "structural_policy", "context_schema", "proposal_schema", "semantic_glossary", "stage3_source_run", "seed_manifest", "manifest", "validation_manifest", "briefs_dir", "system_prompt", "request_prompt", "repair_prompt", "output_schema"},
        "repositories": {"project_repo", "heg_repo", "frozen_project_commit", "frozen_heg_commit", "search_tag", "validation_tag"},
        "identity": set(Stage4Identity.__dataclass_fields__),
    }
    for table_name, keys in table_keys.items():
        if set(_table(raw, table_name)) != keys:
            raise ValueError(f"[{table_name}] keys mismatch")
    if set(model) != {"name", "effort", "generations", "slots", "concurrency", "max_repairs", "max_initial_turns", "max_accepted_turns"} or model.get("name") != EXPECTED_MODEL or model.get("effort") != EXPECTED_EFFORT:
        raise ValueError("model must be gpt-5.6-luna/high")
    parsed_model = Stage4ModelConfig(EXPECTED_MODEL, EXPECTED_EFFORT, int(model.get("generations", 0)), int(model.get("slots", 0)), int(model.get("concurrency", 0)), int(model.get("max_repairs", 0)), int(model.get("max_initial_turns", 0)), int(model.get("max_accepted_turns", 0)))
    if parsed_model != Stage4ModelConfig(EXPECTED_MODEL, EXPECTED_EFFORT, 4, 8, 8, 1, 32, 64):
        raise ValueError("model generation budget is not frozen")
    parsed_limits = Stage4Limits(int(resources.get("max_generation_workers", 0)), int(resources.get("max_evaluation_workers", 0)), int(resources.get("reserved_physical_cores", 0)), int(resources.get("thread_count", 0)), int(limits.get("artifact_uncompressed_shard_bytes", 0)), int(limits.get("artifact_uncompressed_campaign_bytes", 0)), int(limits.get("artifact_compressed_campaign_bytes", 0)), float(limits.get("dry_run_max_fraction", 0)))
    if parsed_limits != Stage4Limits(8, 8, 8, 1, 32 * 1024 * 1024, 512 * 1024 * 1024, 128 * 1024 * 1024, 0.50):
        raise ValueError("resource and artifact limits are not frozen")
    parsed_eval = Stage4Evaluation(int(evaluation.get("bootstrap_samples", 0)), int(evaluation.get("bootstrap_seed", 0)), float(evaluation.get("confidence_level", 0)))
    if parsed_eval != Stage4Evaluation(10_000, 2026073004, 0.95):
        raise ValueError("bootstrap evaluation is not frozen")
    parsed_experiment = Stage4Experiment(_ints(experiment.get("orders"), (10, 12), "orders"), _ints(experiment.get("graph_seeds"), (401, 402, 403, 404), "graph_seeds"), _ints(experiment.get("policy_seeds"), tuple(range(4001, 4017)), "policy_seeds"), int(experiment.get("horizon", 0)), int(experiment.get("shard_count", 0)), int(experiment.get("episodes_per_shard", 0)))
    if (parsed_experiment.horizon, parsed_experiment.shard_count, parsed_experiment.episodes_per_shard, parsed_experiment.episode_count) != (32, 8, 16, 128):
        raise ValueError("experiment matrix is not frozen")
    stage2b_path = _path(inputs["stage2b_config"], "inputs.stage2b_config", base)
    stage2b = load_stage2b_config(stage2b_path)
    paths = {k: _path(inputs[k], f"inputs.{k}", base) for k in ("random_policy", "structural_policy", "context_schema", "proposal_schema", "semantic_glossary", "stage3_source_run", "seed_manifest", "manifest", "validation_manifest", "briefs_dir", "system_prompt", "request_prompt", "repair_prompt", "output_schema")}
    if repos.get("frozen_project_commit") != EXPECTED_PROJECT_COMMIT or repos.get("frozen_heg_commit") != EXPECTED_HEG_COMMIT:
        raise ValueError("repository commits are not frozen")
    if repos.get("search_tag") != EXPECTED_SEARCH_TAG or repos.get("validation_tag") != EXPECTED_VALIDATION_TAG:
        raise ValueError("preregistration tags are not frozen")
    ident = Stage4Identity(**{k: _sha(identity[k], f"identity.{k}") for k in Stage4Identity.__dataclass_fields__})
    result = Stage4SearchConfig(STAGE4_CONFIG_VERSION, source_path, _path(run["run_root"], "run.run_root", base), parsed_model, parsed_limits, parsed_eval, parsed_experiment, stage2b, stage2b.sandbox, stage2b_path, paths["random_policy"], paths["structural_policy"], paths["context_schema"], paths["proposal_schema"], paths["semantic_glossary"], paths["stage3_source_run"], paths["seed_manifest"], paths["manifest"], paths["validation_manifest"], paths["briefs_dir"], paths["system_prompt"], paths["request_prompt"], paths["repair_prompt"], paths["output_schema"], _path(repos["project_repo"], "repositories.project_repo", base), _path(repos["heg_repo"], "repositories.heg_repo", base), EXPECTED_PROJECT_COMMIT, EXPECTED_HEG_COMMIT, EXPECTED_SEARCH_TAG, EXPECTED_VALIDATION_TAG, ident)
    for key, p in (("system_prompt_sha256", result.system_prompt_path), ("request_prompt_sha256", result.request_prompt_path), ("repair_prompt_sha256", result.repair_prompt_path), ("output_schema_sha256", result.output_schema_path), ("seed_manifest_sha256", result.seed_manifest_path), ("manifest_sha256", result.manifest_path), ("validation_manifest_sha256", result.validation_manifest_path)):
        if not p.is_file() or _file_hash(p) != getattr(ident, key):
            raise ValueError(f"identity.{key} does not match file bytes")
    load_seed_manifest(result.seed_manifest_path)
    raw_identity = dict(raw["identity"])
    raw_identity.pop("freeze_payload_sha256", None)
    payload = dict(raw)
    payload["identity"] = raw_identity
    freeze_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if freeze_hash != ident.freeze_payload_sha256:
        raise ValueError("identity.freeze_payload_sha256 mismatch")
    return result


__all__ = ["STAGE4_CONFIG_VERSION", "Stage4SearchConfig", "Stage4Config", "Stage4ModelConfig", "Stage4Limits", "Stage4Evaluation", "Stage4Experiment", "Stage4Identity", "load_stage4_config", "load_stage4_search_config"]

# Friendly aliases used by orchestration code.
Stage4Config = Stage4SearchConfig
load_stage4_search_config = load_stage4_config
