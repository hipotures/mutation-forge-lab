"""Strict, deterministic Stage 3 generation configuration loader."""

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

STAGE3_CONFIG_VERSION = "stage3.1"
EXPECTED_MODEL = "gpt-5.6-luna"
EXPECTED_EFFORT = "high"
EXPECTED_SLOTS = 8
EXPECTED_REPAIRS = 1


@dataclass(frozen=True, slots=True)
class Stage3ModelConfig:
    name: str
    effort: str
    slots: tuple[str, ...]
    max_repairs: int


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
    limits: Stage3Limits
    resources: Stage3Resources
    evaluation: Stage3Evaluation
    experiment: Stage3ExperimentConfig
    stage2b: Stage2BConfig
    sandbox: SandboxLimits
    manifest_path: Path
    context_schema_path: Path
    proposal_schema_path: Path
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
        raw = asdict(self)
        raw.pop("source_path", None)

        def normalize(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        raw = cast(dict[str, Any], normalize(raw))
        raw["run_root"] = str(self.run_root)
        raw["manifest_path"] = str(self.manifest_path)
        raw["context_schema_path"] = str(self.context_schema_path)
        raw["proposal_schema_path"] = str(self.proposal_schema_path)
        for key in (
            "stage2b_config_path",
            "random_policy_path",
            "structural_policy_path",
            "slot_briefs_dir",
            "system_prompt_path",
            "request_prompt_path",
            "output_schema_path",
            "project_repo",
            "heg_repo",
        ):
            raw[key] = str(getattr(self, key))
        return cast(dict[str, JsonValue], raw)

    def stable_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.resolved_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def episode_shard(self, order: int, graph_seed: int, policy_seed: int) -> int:
        """Return the deterministic shard assignment for an experiment episode."""
        try:
            index = list(
                (o, g, p)
                for o in self.experiment.orders
                for g in self.experiment.graph_seeds
                for p in self.experiment.policy_seeds
            ).index((order, graph_seed, policy_seed))
        except ValueError as error:
            raise ValueError("episode tuple is not in the frozen experiment matrix") from error
        return index % self.experiment.shard_count


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid [{name}] table")
    return cast(dict[str, Any], value)


def _exact(table: dict[str, Any], keys: set[str], name: str) -> None:
    if set(table) != keys:
        raise ValueError(
            f"[{name}] keys mismatch; missing={sorted(keys - set(table))}, "
            f"extra={sorted(set(table) - keys)}"
        )


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _int_array(
    value: object, name: str, *, expected: tuple[int, ...] | None = None
) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must be a non-empty integer array")
    result = tuple(cast(list[int], value))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    if expected is not None and result != expected:
        raise ValueError(f"{name} is frozen as {expected}")
    return result


def _positive(value: object, name: str, *, floating: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")
    if not floating and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return float(value) if floating else int(value)


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _commit(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return value


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
    _exact(
        raw,
        {
            "schema_version",
            "run",
            "model",
            "limits",
            "resources",
            "evaluation",
            "experiment",
            "inputs",
            "repositories",
            "identity",
        },
        "root",
    )
    if raw["schema_version"] != STAGE3_CONFIG_VERSION:
        raise ValueError(f"schema_version must be {STAGE3_CONFIG_VERSION!r}")
    run = _table(raw, "run")
    model = _table(raw, "model")
    limits = _table(raw, "limits")
    resources = _table(raw, "resources")
    evaluation = _table(raw, "evaluation")
    experiment = _table(raw, "experiment")
    inputs = _table(raw, "inputs")
    repositories = _table(raw, "repositories")
    identity = _table(raw, "identity")
    _exact(run, {"run_root"}, "run")
    _exact(model, {"name", "effort", "slots", "max_repairs"}, "model")
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
            "freeze_payload_sha256",
            "manifest_sha256",
            "system_prompt_sha256",
            "request_prompt_sha256",
            "output_schema_sha256",
        },
        "identity",
    )
    if model["name"] != EXPECTED_MODEL or model["effort"] != EXPECTED_EFFORT:
        raise ValueError("Stage 3 model is frozen to gpt-5.6-luna with high effort")
    slots = model["slots"]
    if not isinstance(slots, list) or tuple(slots) != tuple(
        f"slot-{i:02d}" for i in range(EXPECTED_SLOTS)
    ):
        raise ValueError("model.slots must be exactly slot-00 through slot-07")
    if model["max_repairs"] != EXPECTED_REPAIRS:
        raise ValueError("model.max_repairs must be one")
    parsed_limits = Stage3Limits(
        resource_address_space_bytes=int(
            _positive(limits["resource_address_space_bytes"], "limits.resource_address_space_bytes")
        ),
        resource_cpu_seconds=float(
            _positive(limits["resource_cpu_seconds"], "limits.resource_cpu_seconds", floating=True)
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
    )
    parsed_resources = Stage3Resources(
        max_generation_workers=int(
            _positive(resources["max_generation_workers"], "resources.max_generation_workers")
        ),
        max_evaluation_workers=int(
            _positive(resources["max_evaluation_workers"], "resources.max_evaluation_workers")
        ),
        reserved_physical_cores=int(
            _positive(resources["reserved_physical_cores"], "resources.reserved_physical_cores")
        ),
        thread_count=int(_positive(resources["thread_count"], "resources.thread_count")),
    )
    if parsed_resources != Stage3Resources(8, 8, 8, 1):
        raise ValueError("resources are frozen to eight workers, eight reserved cores, one thread")
    parsed_evaluation = Stage3Evaluation(
        bootstrap_samples=int(
            _positive(evaluation["bootstrap_samples"], "evaluation.bootstrap_samples")
        ),
        bootstrap_seed=int(_positive(evaluation["bootstrap_seed"], "evaluation.bootstrap_seed")),
        confidence_level=float(evaluation["confidence_level"]),
        random_relative_threshold=float(evaluation["random_relative_threshold"]),
        structural_fraction_threshold=float(evaluation["structural_fraction_threshold"]),
    )
    if (
        not 0 < parsed_evaluation.confidence_level < 1
        or parsed_evaluation.random_relative_threshold != 0.05
        or parsed_evaluation.structural_fraction_threshold != 0.90
    ):
        raise ValueError("Stage 3 evaluation thresholds do not match the frozen gate")
    base = source_path.parent
    stage2b_config_path = _path(inputs["stage2b_config"], "inputs.stage2b_config", base)
    stage2b = load_stage2b_config(stage2b_config_path)
    project_repo = _path(repositories["project_repo"], "repositories.project_repo", base)
    heg_repo = _path(repositories["heg_repo"], "repositories.heg_repo", base)
    parsed_experiment = Stage3ExperimentConfig(
        orders=_int_array(experiment["orders"], "experiment.orders", expected=(10, 12)),
        graph_seeds=_int_array(
            experiment["graph_seeds"], "experiment.graph_seeds", expected=(301, 302, 303, 304)
        ),
        policy_seeds=_int_array(
            experiment["policy_seeds"], "experiment.policy_seeds", expected=tuple(range(3001, 3017))
        ),
        horizon=int(_positive(experiment["horizon"], "experiment.horizon")),
        shard_count=int(_positive(experiment["shard_count"], "experiment.shard_count")),
        episodes_per_shard=int(
            _positive(experiment["episodes_per_shard"], "experiment.episodes_per_shard")
        ),
    )
    if (
        parsed_experiment.horizon != 32
        or parsed_experiment.shard_count != 8
        or parsed_experiment.episode_count != 128
        or parsed_experiment.episodes_per_shard != 16
    ):
        raise ValueError("experiment matrix is frozen to 128 episodes and 8 shards of 16")
    result = Stage3GenerationConfig(
        schema_version=STAGE3_CONFIG_VERSION,
        source_path=source_path,
        run_root=_path(run["run_root"], "run.run_root", base),
        model=Stage3ModelConfig(
            name=model["name"],
            effort=model["effort"],
            slots=tuple(slots),
            max_repairs=int(model["max_repairs"]),
        ),
        limits=parsed_limits,
        resources=parsed_resources,
        evaluation=parsed_evaluation,
        experiment=parsed_experiment,
        stage2b=stage2b,
        sandbox=stage2b.sandbox,
        manifest_path=_path(inputs["manifest"], "inputs.manifest", base),
        context_schema_path=_path(inputs["context_schema"], "inputs.context_schema", base),
        proposal_schema_path=_path(inputs["proposal_schema"], "inputs.proposal_schema", base),
        stage2b_config_path=stage2b_config_path,
        random_policy_path=_path(inputs["random_policy"], "inputs.random_policy", base),
        structural_policy_path=_path(inputs["structural_policy"], "inputs.structural_policy", base),
        slot_briefs_dir=_path(inputs["slot_briefs_dir"], "inputs.slot_briefs_dir", base),
        system_prompt_path=_path(inputs["system_prompt"], "inputs.system_prompt", base),
        request_prompt_path=_path(inputs["request_prompt"], "inputs.request_prompt", base),
        output_schema_path=_path(inputs["output_schema"], "inputs.output_schema", base),
        project_repo=project_repo,
        heg_repo=heg_repo,
        frozen_project_commit=_commit(
            repositories["frozen_project_commit"], "repositories.frozen_project_commit"
        ),
        frozen_heg_commit=_commit(
            repositories["frozen_heg_commit"], "repositories.frozen_heg_commit"
        ),
        preregistration_tag=str(repositories["preregistration_tag"]),
        identity=Stage3Identity(
            **{key: _sha(identity[key], f"identity.{key}") for key in identity}
        ),
    )
    expected_files = {
        "context_schema_sha256": result.context_schema_path,
        "proposal_schema_sha256": result.proposal_schema_path,
        "system_prompt_sha256": result.system_prompt_path,
        "request_prompt_sha256": result.request_prompt_path,
        "output_schema_sha256": result.output_schema_path,
    }
    for name, file_path in expected_files.items():
        if not file_path.is_file() or _file_hash(file_path) != getattr(result.identity, name):
            raise ValueError(f"identity.{name} does not match file bytes")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("manifest_sha256") != _canonical_hash(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    ):
        raise ValueError("manifest_sha256 does not match canonical manifest content")
    if _file_hash(result.manifest_path) != result.identity.manifest_sha256:
        raise ValueError("identity.manifest_sha256 does not match manifest bytes")
    raw_identity = dict(raw["identity"])
    raw_identity.pop("freeze_payload_sha256", None)
    freeze_payload = dict(raw)
    freeze_payload["identity"] = raw_identity
    if _canonical_hash(freeze_payload) != result.identity.freeze_payload_sha256:
        raise ValueError("identity.freeze_payload_sha256 does not match frozen config payload")
    return result


__all__ = [
    "Stage3GenerationConfig",
    "Stage3Evaluation",
    "Stage3Resources",
    "Stage3Limits",
    "Stage3ModelConfig",
    "Stage3ExperimentConfig",
    "Stage3Identity",
    "load_stage3_config",
]
