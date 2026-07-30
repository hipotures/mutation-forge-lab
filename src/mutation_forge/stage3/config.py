"""Strict loader for the frozen Stage 3.1 generation configuration."""
# The frozen key/error messages intentionally remain readable on one line.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.stage2b.config import Stage2BConfig, load_stage2b_config

STAGE3_CONFIG_VERSION = "stage3.1"
EXPECTED_MODEL, EXPECTED_EFFORT, EXPECTED_SLOTS, EXPECTED_REPAIRS = "gpt-5.6-luna", "high", 8, 1
EXPECTED_PROJECT_COMMIT = "1670f7b023dcf110259ea39b63ba1a55cb011521"
EXPECTED_HEG_COMMIT = "fd97451b0f3d87400d1d955a2c6b1b18303344ff"
EXPECTED_TAG = "stage3-generation-frozen-v13"


@dataclass(frozen=True, slots=True)
class Stage3ModelConfig:
    name: str
    effort: str
    slots: tuple[str, ...]
    max_repairs: int


@dataclass(frozen=True, slots=True)
class Stage3AppServerConfig:
    sandbox_mode: str
    approval_policy: str


@dataclass(frozen=True, slots=True)
class Stage3Limits:
    resource_address_space_bytes: int
    resource_cpu_seconds: float
    resource_file_bytes: int
    resource_open_files: int
    resource_processes: int
    request_bytes: int
    response_bytes: int
    event_bytes: int
    transcript_bytes: int
    stdout_bytes: int
    stderr_bytes: int
    turn_seconds: float
    campaign_seconds: float
    artifact_bytes: int


@dataclass(frozen=True, slots=True)
class Stage3Resources:
    max_generation_workers: int
    max_evaluation_workers: int
    reserved_physical_cores: int
    thread_count: int


@dataclass(frozen=True, slots=True)
class Stage3Evaluation:
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float
    random_relative_threshold: float
    structural_fraction_threshold: float


@dataclass(frozen=True, slots=True)
class Stage3ExperimentConfig:
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
class Stage3Identity:
    context_schema_sha256: str
    proposal_schema_sha256: str
    semantic_glossary_sha256: str
    freeze_payload_sha256: str
    manifest_sha256: str
    system_prompt_sha256: str
    request_prompt_sha256: str
    output_schema_sha256: str


@dataclass(frozen=True, slots=True)
class Stage3GenerationConfig:
    schema_version: str
    source_path: Path
    run_root: Path
    model: Stage3ModelConfig
    app_server: Stage3AppServerConfig
    limits: Stage3Limits
    resources: Stage3Resources
    evaluation: Stage3Evaluation
    experiment: Stage3ExperimentConfig
    stage2b: Stage2BConfig
    sandbox: SandboxLimits
    manifest_path: Path
    context_schema_path: Path
    proposal_schema_path: Path
    semantic_glossary_path: Path
    stage2b_config_path: Path
    random_policy_path: Path
    structural_policy_path: Path
    slot_briefs_dir: Path
    system_prompt_path: Path
    request_prompt_path: Path
    output_schema_path: Path
    project_repo: Path
    heg_repo: Path
    frozen_project_commit: str
    frozen_heg_commit: str
    preregistration_tag: str
    identity: Stage3Identity

    def resolved_dict(self) -> dict[str, JsonValue]:
        def normalize(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): normalize(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [normalize(v) for v in value]
            return value

        raw = cast(dict[str, JsonValue], normalize(asdict(self)))
        raw.pop("source_path", None)
        return raw

    def stable_hash(self) -> str:
        payload = self.resolved_dict()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def episode_shard(self, order: int, graph_seed: int, policy_seed: int) -> int:
        matrix = [
            (o, g, p)
            for o in self.experiment.orders
            for g in self.experiment.graph_seeds
            for p in self.experiment.policy_seeds
        ]
        try:
            return matrix.index((order, graph_seed, policy_seed)) % self.experiment.shard_count
        except ValueError as exc:
            raise ValueError("episode tuple is not in the frozen experiment matrix") from exc


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid [{name}] table")
    return cast(dict[str, Any], value)


def _exact(table: dict[str, Any], keys: set[str], name: str) -> None:
    if set(table) != keys:
        raise ValueError(
            f"[{name}] keys mismatch; missing={sorted(keys - set(table))}, extra={sorted(set(table) - keys)}"
        )


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    p = Path(value)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _ints(value: object, name: str, expected: tuple[int, ...]) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or tuple(value) != expected
        or any(isinstance(x, bool) or not isinstance(x, int) for x in value)
    ):
        raise ValueError(f"{name} is frozen as {expected}")
    return expected


def _positive(value: object, name: str, *, floating: bool = False) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        or (not floating and not isinstance(value, int))
    ):
        raise ValueError(f"{name} must be a positive finite {'number' if floating else 'integer'}")
    return float(value) if floating else int(value)


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _commit(value: object, name: str, expected: str) -> str:
    if value != expected:
        raise ValueError(f"{name} must be pinned to {expected}")
    return expected


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage3_config(path: str | Path) -> Stage3GenerationConfig:
    source_path = Path(path).resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = {
        "schema_version",
        "run",
        "model",
        "app_server",
        "limits",
        "resources",
        "evaluation",
        "experiment",
        "inputs",
        "repositories",
        "identity",
    }
    if set(raw) != root or raw.get("schema_version") != STAGE3_CONFIG_VERSION:
        raise ValueError("invalid Stage 3.1 config root")
    run, model, app_server, limits, resources = (
        _table(raw, n) for n in ("run", "model", "app_server", "limits", "resources")
    )
    evaluation, experiment, inputs = (
        _table(raw, n) for n in ("evaluation", "experiment", "inputs")
    )
    repositories, identity = _table(raw, "repositories"), _table(raw, "identity")
    _exact(run, {"run_root"}, "run")
    _exact(model, {"name", "effort", "slots", "max_repairs"}, "model")
    _exact(app_server, {"sandbox_mode", "approval_policy"}, "app_server")
    _exact(
        limits,
        {
            "resource_address_space_bytes",
            "resource_cpu_seconds",
            "resource_file_bytes",
            "resource_open_files",
            "resource_processes",
            "request_bytes",
            "response_bytes",
            "event_bytes",
            "transcript_bytes",
            "stdout_bytes",
            "stderr_bytes",
            "turn_seconds",
            "campaign_seconds",
            "artifact_bytes",
        },
        "limits",
    )
    _exact(
        resources,
        {
            "max_generation_workers",
            "max_evaluation_workers",
            "reserved_physical_cores",
            "thread_count",
        },
        "resources",
    )
    _exact(
        evaluation,
        {
            "bootstrap_samples",
            "bootstrap_seed",
            "confidence_level",
            "random_relative_threshold",
            "structural_fraction_threshold",
        },
        "evaluation",
    )
    _exact(
        experiment,
        {"orders", "graph_seeds", "policy_seeds", "horizon", "shard_count", "episodes_per_shard"},
        "experiment",
    )
    _exact(
        inputs,
        {
            "stage2b_config",
            "random_policy",
            "structural_policy",
            "manifest",
            "context_schema",
            "proposal_schema",
            "semantic_glossary",
            "slot_briefs_dir",
            "system_prompt",
            "request_prompt",
            "output_schema",
        },
        "inputs",
    )
    _exact(
        repositories,
        {
            "project_repo",
            "heg_repo",
            "frozen_project_commit",
            "frozen_heg_commit",
            "preregistration_tag",
        },
        "repositories",
    )
    _exact(
        identity,
        {
            "context_schema_sha256",
            "proposal_schema_sha256",
            "semantic_glossary_sha256",
            "freeze_payload_sha256",
            "manifest_sha256",
            "system_prompt_sha256",
            "request_prompt_sha256",
            "output_schema_sha256",
        },
        "identity",
    )
    if (
        model["name"] != EXPECTED_MODEL
        or model["effort"] != EXPECTED_EFFORT
        or tuple(model["slots"]) != tuple(f"slot-{i:02d}" for i in range(8))
        or model["max_repairs"] != 1
    ):
        raise ValueError("model is frozen to gpt-5.6-luna/high, eight slots and one repair")
    if (
        app_server["sandbox_mode"] not in {"read-only", "danger-full-access"}
        or app_server["approval_policy"] != "never"
    ):
        raise ValueError(
            "app_server requires sandbox_mode read-only/danger-full-access "
            "and approval_policy never"
        )
    parsed_limits = Stage3Limits(
        resource_address_space_bytes=int(
            _positive(
                limits["resource_address_space_bytes"],
                "limits.resource_address_space_bytes",
            )
        ),
        resource_cpu_seconds=float(
            _positive(
                limits["resource_cpu_seconds"],
                "limits.resource_cpu_seconds",
                floating=True,
            )
        ),
        resource_file_bytes=int(
            _positive(limits["resource_file_bytes"], "limits.resource_file_bytes")
        ),
        resource_open_files=int(
            _positive(limits["resource_open_files"], "limits.resource_open_files")
        ),
        resource_processes=int(
            _positive(limits["resource_processes"], "limits.resource_processes")
        ),
        request_bytes=int(_positive(limits["request_bytes"], "limits.request_bytes")),
        response_bytes=int(_positive(limits["response_bytes"], "limits.response_bytes")),
        event_bytes=int(_positive(limits["event_bytes"], "limits.event_bytes")),
        transcript_bytes=int(_positive(limits["transcript_bytes"], "limits.transcript_bytes")),
        stdout_bytes=int(_positive(limits["stdout_bytes"], "limits.stdout_bytes")),
        stderr_bytes=int(_positive(limits["stderr_bytes"], "limits.stderr_bytes")),
        turn_seconds=float(_positive(limits["turn_seconds"], "limits.turn_seconds", floating=True)),
        campaign_seconds=float(
            _positive(limits["campaign_seconds"], "limits.campaign_seconds", floating=True)
        ),
        artifact_bytes=int(_positive(limits["artifact_bytes"], "limits.artifact_bytes")),
    )
    if (
        parsed_limits.request_bytes,
        parsed_limits.response_bytes,
        parsed_limits.event_bytes,
        parsed_limits.transcript_bytes,
        parsed_limits.stdout_bytes,
        parsed_limits.stderr_bytes,
    ) != (65_536, 1_638_400, 655_360, 2_621_440, 20_971_520, 65_536):
        raise ValueError(
            "transport limits are frozen to 64 KiB requests, 1,600 KiB responses, "
            "640 KiB events, 2.5 MiB transcripts, 20 MiB aggregate stdout, "
            "and 64 KiB stderr"
        )
    parsed_resources = Stage3Resources(
        *(
            int(_positive(resources[k], f"resources.{k}"))
            for k in (
                "max_generation_workers",
                "max_evaluation_workers",
                "reserved_physical_cores",
                "thread_count",
            )
        )
    )
    if parsed_resources != Stage3Resources(8, 8, 8, 1):
        raise ValueError("resources are frozen to eight workers/evaluation workers and one thread")
    parsed_eval = Stage3Evaluation(
        int(_positive(evaluation["bootstrap_samples"], "evaluation.bootstrap_samples")),
        int(_positive(evaluation["bootstrap_seed"], "evaluation.bootstrap_seed")),
        float(evaluation["confidence_level"]),
        float(evaluation["random_relative_threshold"]),
        float(evaluation["structural_fraction_threshold"]),
    )
    if (
        not 0 < parsed_eval.confidence_level < 1
        or parsed_eval.random_relative_threshold != 0.05
        or parsed_eval.structural_fraction_threshold != 0.90
    ):
        raise ValueError("evaluation gates are not frozen")
    if (
        parsed_limits.resource_address_space_bytes != 2 * 1024 * 1024 * 1024
        or parsed_limits.resource_cpu_seconds != 120.0
        or parsed_limits.resource_file_bytes != 8 * 1024 * 1024
        or parsed_limits.resource_open_files != 256
        or parsed_limits.resource_processes != 102_400
        or parsed_limits.turn_seconds != 600.0
        or parsed_limits.campaign_seconds != 1800.0
        or parsed_limits.artifact_bytes != 32 * 1024 * 1024
    ):
        raise ValueError("App Server resource limits do not match the frozen evidence")
    if (
        parsed_limits.request_bytes != 65_536
        or parsed_limits.response_bytes != 1_638_400
        or parsed_limits.event_bytes != 655_360
        or parsed_limits.transcript_bytes != 2_621_440
    ):
        raise ValueError("transport bounds are not frozen")
    base = source_path.parent
    stage2b_path = _path(inputs["stage2b_config"], "inputs.stage2b_config", base)
    stage2b = load_stage2b_config(stage2b_path)
    parsed_experiment = Stage3ExperimentConfig(
        _ints(experiment["orders"], "experiment.orders", (10, 12)),
        _ints(experiment["graph_seeds"], "experiment.graph_seeds", (301, 302, 303, 304)),
        _ints(experiment["policy_seeds"], "experiment.policy_seeds", tuple(range(3001, 3017))),
        int(_positive(experiment["horizon"], "experiment.horizon")),
        int(_positive(experiment["shard_count"], "experiment.shard_count")),
        int(_positive(experiment["episodes_per_shard"], "experiment.episodes_per_shard")),
    )
    if (
        parsed_experiment.horizon,
        parsed_experiment.shard_count,
        parsed_experiment.episode_count,
        parsed_experiment.episodes_per_shard,
    ) != (32, 8, 128, 16):
        raise ValueError("experiment matrix is frozen to 128 episodes and eight shards")
    paths = {
        name: _path(inputs[name], f"inputs.{name}", base)
        for name in (
            "manifest",
            "context_schema",
            "proposal_schema",
            "semantic_glossary",
            "random_policy",
            "structural_policy",
            "slot_briefs_dir",
            "system_prompt",
            "request_prompt",
            "output_schema",
        )
    }
    if repositories["preregistration_tag"] != EXPECTED_TAG:
        raise ValueError(f"repositories.preregistration_tag must be {EXPECTED_TAG}")
    result = Stage3GenerationConfig(
        STAGE3_CONFIG_VERSION,
        source_path,
        _path(run["run_root"], "run.run_root", base),
        Stage3ModelConfig(model["name"], model["effort"], tuple(model["slots"]), 1),
        Stage3AppServerConfig(app_server["sandbox_mode"], app_server["approval_policy"]),
        parsed_limits,
        parsed_resources,
        parsed_eval,
        parsed_experiment,
        stage2b,
        stage2b.sandbox,
        paths["manifest"],
        paths["context_schema"],
        paths["proposal_schema"],
        paths["semantic_glossary"],
        stage2b_path,
        paths["random_policy"],
        paths["structural_policy"],
        paths["slot_briefs_dir"],
        paths["system_prompt"],
        paths["request_prompt"],
        paths["output_schema"],
        _path(repositories["project_repo"], "repositories.project_repo", base),
        _path(repositories["heg_repo"], "repositories.heg_repo", base),
        _commit(
            repositories["frozen_project_commit"],
            "repositories.frozen_project_commit",
            EXPECTED_PROJECT_COMMIT,
        ),
        _commit(
            repositories["frozen_heg_commit"], "repositories.frozen_heg_commit", EXPECTED_HEG_COMMIT
        ),
        EXPECTED_TAG,
        Stage3Identity(**{k: _sha(identity[k], f"identity.{k}") for k in identity}),
    )
    for name, file_path in (
        ("context_schema_sha256", result.context_schema_path),
        ("proposal_schema_sha256", result.proposal_schema_path),
        ("semantic_glossary_sha256", result.semantic_glossary_path),
        ("system_prompt_sha256", result.system_prompt_path),
        ("request_prompt_sha256", result.request_prompt_path),
        ("output_schema_sha256", result.output_schema_path),
    ):
        if not file_path.is_file() or _file_hash(file_path) != getattr(result.identity, name):
            raise ValueError(f"identity.{name} does not match file bytes")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or _file_hash(result.manifest_path) != result.identity.manifest_sha256
        or manifest.get("manifest_sha256")
        != _canonical_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    ):
        raise ValueError("manifest identity/hash mismatch")
    raw_identity = dict(raw["identity"])
    raw_identity.pop("freeze_payload_sha256", None)
    payload = dict(raw)
    payload["identity"] = raw_identity
    if _canonical_hash(payload) != result.identity.freeze_payload_sha256:
        raise ValueError("freeze_payload_sha256 does not match config payload")
    return result


__all__ = [
    "Stage3GenerationConfig",
    "Stage3Evaluation",
    "Stage3Resources",
    "Stage3Limits",
    "Stage3ModelConfig",
    "Stage3AppServerConfig",
    "Stage3ExperimentConfig",
    "Stage3Identity",
    "load_stage3_config",
]
