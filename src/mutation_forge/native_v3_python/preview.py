"""Explicit guarded experiment route for the ordinary-Python preview."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.config import validate_experiment_id
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue

from .contracts import (
    PYTHON_EXPERIMENT_PROTOCOL_ID,
    PYTHON_WORKSPACE_SCHEMA_VERSION,
    PythonWorkspaceProtocolError,
    require_python_workspace_schema_version,
)
from .provenance import ensure_m5_acceptance_provenance
from .runtime_contracts import PolicyRuntimeLimitsV1
from .scientific_evaluation import (
    ScientificEvaluationOptionsV1,
    workload_projection,
)
from .scientific_search import (
    M10_BASELINE_FILENAME,
    M10_BASELINE_RESULT_FILENAME,
    M10_REPORT_FILENAME,
    M10_REPORT_PROTOCOL_ID,
    M10_RUNTIME_FILENAME,
    M10_SEARCH_PROTOCOL_ID,
    M10_STOP_FILENAME,
    M10ScientificEvaluator,
    ScientificResumeBudgetV1,
    ScientificSearchOptionsV2,
    hourly_token_usage,
    resolve_resume_generation,
    run_sustained_search,
)
from .search import (
    M5_REPORT_PROTOCOL_ID,
    M5_SEARCH_PROTOCOL_ID,
    DevelopmentCaseV1,
    M5OperatorStop,
    M5ScientificEvaluator,
    M5SearchProvider,
    M10SearchProvider,
    run_m5_search,
)
from .search_provider import (
    CodexM5SearchProvider,
    CodexM10SearchProvider,
    PythonPanelScientificEvaluator,
    specification_ack_schema,
)

PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION = "mforge.experiment.native_python_preview_config.v1"
PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION = (
    "mforge.experiment.native_python_scientific_search_config.v2"
)
PYTHON_PREVIEW_STATE_SCHEMA_VERSION = "mforge.experiment.status.native_python_preview.v1"
PYTHON_PREVIEW_PROTOCOL_VERSION = "mforge.native.python_preview.v1"
PYTHON_PREVIEW_MODE = "ordinary-python"
V2_PROTOCOL = "native-v2"
_SUPERSEDED_JSON_DSL_SELECTOR = "v3"
_STATE_NAME = "python-preview-state.json.gz"
_CONFIG_NAME = "python-preview.toml"
_STOP_REQUEST_NAME = "python-preview-stop-request.json.gz"
_STOP_REQUEST_PROTOCOL_ID = "mforge.native.python_preview.stop_request.v1"
_M10_PAUSE_RECORD_SCHEMA_VERSION = "mforge.native.python_m10_emergency_stop_evidence.v1"
_PAUSED_FOR_BUDGET = "PAUSED_FOR_BUDGET"
_LEGACY_GENERATION_LIMIT_CAPSULE_MISSING = (
    "legacy_generation_limit_provider_runtime_missing"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TERMINAL_CANDIDATE_STATUSES = frozenset(
    {
        "evaluated",
        "contract_invalid",
        "duplicate",
        "provider_failed",
        "missing",
        "evaluation_infrastructure_failure",
    }
)
_V2_TOP_LEVEL_FIELDS = frozenset(
    {"kind", "preset", "run", "model", "search", "evaluation", "resources"}
)
_PUBLIC_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "workspace_schema_version",
        "protocol",
        "protocol_version",
        "preview_mode",
        "preview_active",
        "native_v2_default",
        "dsl_runtime_used",
        "safe_api_expanded",
        "state",
        "resumable",
        "run_terminal",
        "terminal_reason",
        "scientific_result_kind",
        "scientific_success",
        "last_error",
        "last_boundary",
        "resume_attempts",
    }
)

type ProviderFactory = Callable[
    ["PythonPreviewConfig", str],
    M5SearchProvider | M10SearchProvider,
]
type BackendFactory = Callable[["PythonPreviewConfig"], GraphBackend]
type EvaluatorFactory = Callable[
    ["PythonPreviewConfig", GraphBackend],
    M5ScientificEvaluator,
]
type ProvenanceGuard = Callable[..., Mapping[str, JsonValue]]


@dataclass(frozen=True, slots=True)
class PythonPreviewConfig:
    """Strict opt-in configuration for the bounded Python preview."""

    schema_version: str
    protocol: str
    exp_id: str
    workspace: Path
    model: str
    effort: str
    timeout_seconds: float
    heg_repo: Path
    scientific_search: ScientificSearchOptionsV2 | None
    source_path: Path
    source_bytes: bytes = field(repr=False, compare=False)

    @property
    def experiment_root(self) -> Path:
        return self.workspace / self.exp_id

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_bytes).hexdigest()

    @property
    def scientific_config_sha256(self) -> str:
        return self._scientific_config_sha256(include_generation_limit=False)

    @property
    def legacy_scientific_config_sha256(self) -> str:
        return self._scientific_config_sha256(include_generation_limit=True)

    def _scientific_config_sha256(
        self,
        *,
        include_generation_limit: bool,
    ) -> str:
        scientific_identity = (
            self.scientific_search.scientific_identity_dict()
            if self.scientific_search is not None
            else None
        )
        if (
            include_generation_limit
            and scientific_identity is not None
            and self.scientific_search is not None
        ):
            scientific_identity = {
                "generation_limit": self.scientific_search.generation_limit,
                **scientific_identity,
            }
        payload = {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "model": self.model,
            "effort": self.effort,
            "scientific_search": scientific_identity,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


class PythonPreviewWorkspaceError(RuntimeError):
    """A workspace cannot be interpreted by the Python preview protocol."""


def _raw_config(path: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    source_path = Path(path).resolve()
    source_bytes = source_path.read_bytes()
    value = tomllib.loads(source_bytes.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Python preview configuration must be a TOML table")
    return source_path, source_bytes, value


def experiment_protocol(path: str | Path = "experiment.toml") -> str:
    """Select explicit Python preview or the unchanged Native v2 default."""

    _, _, raw = _raw_config(path)
    protocol = raw.get("protocol")
    if protocol is None:
        return V2_PROTOCOL
    if protocol == PYTHON_EXPERIMENT_PROTOCOL_ID:
        return PYTHON_EXPERIMENT_PROTOCOL_ID
    if protocol == _SUPERSEDED_JSON_DSL_SELECTOR:
        raise ValueError("the superseded JSON-DSL experiment protocol was removed")
    raise ValueError(
        f"unsupported experiment protocol selector: {protocol!r}; "
        f"expected {PYTHON_EXPERIMENT_PROTOCOL_ID!r} or omit it for Native v2"
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty integer array")
    return tuple(value)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{name} must be a nonempty string array")
    return tuple(value)


def _resolved_path(value: object, name: str, source_dir: Path) -> Path:
    raw = Path(_string(value, name))
    return (source_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()


def load_python_preview_config(
    path: str | Path,
) -> PythonPreviewConfig:
    """Parse the separate fail-closed Python preview configuration."""

    source_path, source_bytes, raw = _raw_config(path)
    mixed = sorted(_V2_TOP_LEVEL_FIELDS.intersection(raw))
    if mixed:
        raise ValueError(f"Python preview configuration cannot contain Native v2 fields: {mixed}")
    allowed = {
        "schema_version",
        "protocol",
        "exp_id",
        "workspace",
        "python_preview",
    }
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported top-level fields: {unknown}")
    schema_version = raw.get("schema_version")
    if schema_version not in {
        PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION,
        PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION,
    }:
        raise ValueError("Python preview requires a supported schema_version")
    if raw.get("protocol") != PYTHON_EXPERIMENT_PROTOCOL_ID:
        raise ValueError(f"Python preview requires protocol {PYTHON_EXPERIMENT_PROTOCOL_ID!r}")
    exp_id = validate_experiment_id(raw.get("exp_id"))
    workspace = _resolved_path(
        raw.get("workspace"),
        "workspace",
        source_path.parent,
    )
    preview_value = raw.get("python_preview")
    if not isinstance(preview_value, dict):
        raise ValueError("[python_preview] must be a table")
    preview = cast(dict[str, Any], preview_value)
    allowed_preview = {
        "model",
        "effort",
        "timeout_seconds",
        "heg_repo",
        "scientific_search",
    }
    unknown_preview = sorted(set(preview).difference(allowed_preview))
    if unknown_preview:
        raise ValueError(f"unsupported [python_preview] fields: {unknown_preview}")
    model = _string(preview.get("model"), "python_preview.model")
    effort = _string(preview.get("effort"), "python_preview.effort")
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("python_preview.effort is unsupported")
    timeout_seconds = _positive_number(
        preview.get("timeout_seconds"),
        "python_preview.timeout_seconds",
    )
    heg_repo = _resolved_path(
        preview.get("heg_repo"),
        "python_preview.heg_repo",
        source_path.parent,
    )
    scientific_value = preview.get("scientific_search")
    scientific_search: ScientificSearchOptionsV2 | None = None
    if scientific_value is not None:
        if schema_version != PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "python_preview.scientific_search requires the scientific search config schema"
            )
        if not isinstance(scientific_value, dict):
            raise ValueError("[python_preview.scientific_search] must be a table")
        scientific = cast(dict[str, Any], scientific_value)
        scientific_fields = {
            "generation_limit",
            "evaluator_workers",
            "provider_concurrency",
            "wall_seconds",
            "max_total_tokens_per_hour",
            "primary_program_slots",
            "repair_turn_limit",
            "provider_total_turn_limit",
            "validated_queue_target",
            "validated_queue_capacity",
            "stop_on_verified",
            "resume_enabled",
            "replace_terminal_slots",
            "evaluation",
        }
        unknown_scientific = sorted(set(scientific).difference(scientific_fields))
        if unknown_scientific:
            raise ValueError(
                f"unsupported [python_preview.scientific_search] fields: {unknown_scientific}"
            )
        optional_budget_fields = {
            "generation_limit",
            "wall_seconds",
            "max_total_tokens_per_hour",
            "primary_program_slots",
            "repair_turn_limit",
            "provider_total_turn_limit",
        }
        required_scientific_fields = scientific_fields - optional_budget_fields
        if not required_scientific_fields.issubset(scientific):
            missing = sorted(required_scientific_fields.difference(scientific))
            raise ValueError(f"missing [python_preview.scientific_search] fields: {missing}")
        evaluation_value = scientific["evaluation"]
        if not isinstance(evaluation_value, dict):
            raise ValueError("[python_preview.scientific_search.evaluation] must be a table")
        evaluation = cast(dict[str, Any], evaluation_value)
        evaluation_fields = {
            "graph_mode",
            "order_schedule",
            "min_order",
            "max_order",
            "orders_per_generation",
            "graph_seeds",
            "policy_seeds",
            "horizon",
            "witness_cap",
            "baselines",
            "replay",
        }
        unknown_evaluation = sorted(set(evaluation).difference(evaluation_fields))
        if unknown_evaluation:
            raise ValueError(
                "unsupported "
                "[python_preview.scientific_search.evaluation] fields: "
                f"{unknown_evaluation}"
            )
        if set(evaluation) != evaluation_fields:
            missing = sorted(evaluation_fields.difference(evaluation))
            raise ValueError(
                f"missing [python_preview.scientific_search.evaluation] fields: {missing}"
            )
        scientific_search = ScientificSearchOptionsV2(
            generation_limit=(
                _positive_integer(
                    scientific["generation_limit"],
                    "python_preview.scientific_search.generation_limit",
                )
                if "generation_limit" in scientific
                else None
            ),
            evaluator_workers=_positive_integer(
                scientific["evaluator_workers"],
                "python_preview.scientific_search.evaluator_workers",
            ),
            provider_concurrency=_positive_integer(
                scientific["provider_concurrency"],
                "python_preview.scientific_search.provider_concurrency",
            ),
            wall_seconds=(
                _positive_number(
                    scientific["wall_seconds"],
                    "python_preview.scientific_search.wall_seconds",
                )
                if "wall_seconds" in scientific
                else None
            ),
            max_total_tokens_per_hour=(
                _positive_integer(
                    scientific["max_total_tokens_per_hour"],
                    ("python_preview.scientific_search.max_total_tokens_per_hour"),
                )
                if "max_total_tokens_per_hour" in scientific
                else None
            ),
            primary_program_slots=(
                _positive_integer(
                    scientific["primary_program_slots"],
                    "python_preview.scientific_search.primary_program_slots",
                )
                if "primary_program_slots" in scientific
                else None
            ),
            repair_turn_limit=(
                _nonnegative_integer(
                    scientific["repair_turn_limit"],
                    "python_preview.scientific_search.repair_turn_limit",
                )
                if "repair_turn_limit" in scientific
                else None
            ),
            provider_total_turn_limit=(
                _positive_integer(
                    scientific["provider_total_turn_limit"],
                    "python_preview.scientific_search.provider_total_turn_limit",
                )
                if "provider_total_turn_limit" in scientific
                else None
            ),
            validated_queue_target=_positive_integer(
                scientific["validated_queue_target"],
                "python_preview.scientific_search.validated_queue_target",
            ),
            validated_queue_capacity=_positive_integer(
                scientific["validated_queue_capacity"],
                "python_preview.scientific_search.validated_queue_capacity",
            ),
            stop_on_verified=_boolean(
                scientific["stop_on_verified"],
                "python_preview.scientific_search.stop_on_verified",
            ),
            resume_enabled=_boolean(
                scientific["resume_enabled"],
                "python_preview.scientific_search.resume_enabled",
            ),
            replace_terminal_slots=_boolean(
                scientific["replace_terminal_slots"],
                "python_preview.scientific_search.replace_terminal_slots",
            ),
            evaluation=ScientificEvaluationOptionsV1(
                graph_mode=_string(
                    evaluation["graph_mode"],
                    ("python_preview.scientific_search.evaluation.graph_mode"),
                ),
                order_schedule=_string(
                    evaluation["order_schedule"],
                    ("python_preview.scientific_search.evaluation.order_schedule"),
                ),
                min_order=_positive_integer(
                    evaluation["min_order"],
                    ("python_preview.scientific_search.evaluation.min_order"),
                ),
                max_order=_positive_integer(
                    evaluation["max_order"],
                    ("python_preview.scientific_search.evaluation.max_order"),
                ),
                orders_per_generation=_positive_integer(
                    evaluation["orders_per_generation"],
                    ("python_preview.scientific_search.evaluation.orders_per_generation"),
                ),
                graph_seeds=_integer_tuple(
                    evaluation["graph_seeds"],
                    ("python_preview.scientific_search.evaluation.graph_seeds"),
                ),
                policy_seeds=_integer_tuple(
                    evaluation["policy_seeds"],
                    ("python_preview.scientific_search.evaluation.policy_seeds"),
                ),
                horizon=_positive_integer(
                    evaluation["horizon"],
                    ("python_preview.scientific_search.evaluation.horizon"),
                ),
                witness_cap=_positive_integer(
                    evaluation["witness_cap"],
                    ("python_preview.scientific_search.evaluation.witness_cap"),
                ),
                baselines=_string_tuple(
                    evaluation["baselines"],
                    ("python_preview.scientific_search.evaluation.baselines"),
                ),
                replay=_boolean(
                    evaluation["replay"],
                    ("python_preview.scientific_search.evaluation.replay"),
                ),
            ),
        )
    elif schema_version == PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION:
        raise ValueError("scientific search config requires [python_preview.scientific_search]")
    config = PythonPreviewConfig(
        schema_version=str(schema_version),
        protocol=PYTHON_EXPERIMENT_PROTOCOL_ID,
        exp_id=exp_id,
        workspace=workspace,
        model=model,
        effort=effort,
        timeout_seconds=timeout_seconds,
        heg_repo=heg_repo,
        scientific_search=scientific_search,
        source_path=source_path,
        source_bytes=source_bytes,
    )
    if (
        not config.experiment_root.exists()
        and scientific_search is not None
        and scientific_search.primary_program_slots is not None
        and scientific_search.primary_program_slots
        != scientific_search.current_primary_program_slots
    ):
        raise ValueError(
            "primary_program_slots must provide exactly eight slots per generation"
        )
    return config


def _state_path(config: PythonPreviewConfig) -> Path:
    return config.experiment_root / _STATE_NAME


def _provider_runtime_missing(root: Path) -> bool:
    state_paths = (
        root / "provider-runtime" / "coordinator" / "provider-state.json.gz",
        root / "provider-runtime" / "provider-state.json.gz",
    )
    for state_path in state_paths:
        state = _load_mapping(state_path)
        if state is None:
            continue
        capsule_root = state.get("capsule_root")
        if isinstance(capsule_root, str) and capsule_root:
            return not Path(capsule_root).is_dir()
    return False


def _generation_limit_lifecycle(
    config: PythonPreviewConfig,
    state: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any] | None, bool]:
    if config.scientific_search is None:
        return dict(state), None, False
    report = _load_mapping(config.experiment_root / M10_REPORT_FILENAME)
    if (
        report is None
        or report.get("protocol_id") != M10_REPORT_PROTOCOL_ID
        or report.get("stop_reason") != "generation_budget"
    ):
        return dict(state), report, False
    generation_count = _nonnegative_int(report.get("generation_count"))
    generation_limit = config.scientific_search.generation_limit
    at_limit = generation_limit is not None and generation_count >= generation_limit
    terminal_reason = state.get("terminal_reason")
    if (
        terminal_reason == _LEGACY_GENERATION_LIMIT_CAPSULE_MISSING
        and not at_limit
    ):
        return dict(state), report, False
    legacy_generation_limit_state = (
        terminal_reason == "generation_budget"
        or (
            state.get("state") == "completed"
            and terminal_reason in {None, "generation_budget"}
        )
        or (
            state.get("state") == "failed"
            and terminal_reason == "provider_runtime_missing"
        )
    )
    if not legacy_generation_limit_state:
        return dict(state), report, at_limit
    exact = report.get("exact_verified") is True
    return (
        {
            **state,
            "state": "blocked",
            "resumable": True,
            "run_terminal": False,
            "terminal_reason": "generation_budget",
            "scientific_result_kind": (
                "VERIFIED_COUNTEREXAMPLE"
                if exact
                else "DEVELOPMENT_SEARCH_EVIDENCE"
            ),
            "scientific_success": exact,
            "last_error": None,
        },
        report,
        at_limit,
    )


def _generation_limit_noop_status(
    state: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    generation_count = _nonnegative_int(report.get("generation_count"))
    return {
        **_public(state),
        "search_protocol": M10_SEARCH_PROTOCOL_ID,
        "generation_index": generation_count - 1 if generation_count else None,
    }


def _legacy_workspace_config_sha256(
    config: PythonPreviewConfig,
) -> str | None:
    try:
        frozen_config = load_python_preview_config(
            config.experiment_root / _CONFIG_NAME
        )
    except (OSError, ValueError):
        return None
    if frozen_config.scientific_config_sha256 != config.scientific_config_sha256:
        return None
    return frozen_config.legacy_scientific_config_sha256


def _base_state(config: PythonPreviewConfig) -> dict[str, Any]:
    return {
        "schema_version": PYTHON_PREVIEW_STATE_SCHEMA_VERSION,
        "workspace_schema_version": PYTHON_WORKSPACE_SCHEMA_VERSION,
        "protocol": PYTHON_EXPERIMENT_PROTOCOL_ID,
        "protocol_version": PYTHON_PREVIEW_PROTOCOL_VERSION,
        "preview_mode": PYTHON_PREVIEW_MODE,
        "preview_active": True,
        "native_v2_default": True,
        "dsl_runtime_used": False,
        "safe_api_expanded": False,
        "state": "not_created",
        "resumable": False,
        "run_terminal": False,
        "terminal_reason": None,
        "scientific_result_kind": "NONE",
        "scientific_success": False,
        "last_error": None,
        "last_boundary": None,
        "resume_attempts": 0,
        "config_sha256": config.scientific_config_sha256,
    }


def _public(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key in _PUBLIC_STATE_FIELDS}


def _safe_error(error: Exception, config: PythonPreviewConfig) -> str:
    del config
    return type(error).__name__


def _write_state(
    config: PythonPreviewConfig,
    state: Mapping[str, Any],
) -> None:
    write_json(_state_path(config), dict(state))


def _load_state(config: PythonPreviewConfig) -> dict[str, Any]:
    path = _state_path(config)
    if not path.is_file():
        raise PythonPreviewWorkspaceError(
            "existing workspace is not an ordinary-Python preview workspace; "
            "JSON-DSL and Native v2 workspaces cannot be migrated or reinterpreted"
        )
    raw = read_json(path)
    if not isinstance(raw, Mapping):
        raise PythonPreviewWorkspaceError("Python preview state is not an object")
    unknown = sorted(set(raw).difference(_base_state(config)))
    if unknown:
        raise PythonPreviewWorkspaceError(
            f"Python preview state contains unsupported fields: {unknown}"
        )
    try:
        require_python_workspace_schema_version(raw.get("workspace_schema_version"))
    except PythonWorkspaceProtocolError as error:
        raise PythonPreviewWorkspaceError(str(error)) from error
    if (
        raw.get("schema_version") != PYTHON_PREVIEW_STATE_SCHEMA_VERSION
        or raw.get("protocol") != PYTHON_EXPERIMENT_PROTOCOL_ID
        or raw.get("protocol_version") != PYTHON_PREVIEW_PROTOCOL_VERSION
        or raw.get("preview_mode") != PYTHON_PREVIEW_MODE
    ):
        raise PythonPreviewWorkspaceError(
            "Python preview workspace protocol does not match this runtime"
        )
    if raw.get("config_sha256") != config.scientific_config_sha256:
        try:
            frozen_config = load_python_preview_config(
                config.experiment_root / _CONFIG_NAME
            )
        except (OSError, ValueError):
            frozen_config = None
        legacy_generation_limit_identity = (
            frozen_config is not None
            and raw.get("config_sha256")
            == frozen_config.legacy_scientific_config_sha256
            and frozen_config.scientific_config_sha256
            == config.scientific_config_sha256
        )
        if not legacy_generation_limit_identity:
            raise PythonPreviewWorkspaceError(
                "existing experiment scientific configuration differs from the "
                "frozen workspace identity; use a new exp_id"
            )
    return dict(raw)


def _initialize_workspace(config: PythonPreviewConfig) -> dict[str, Any]:
    config.workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{config.exp_id}.native-python.",
            dir=config.workspace,
        )
    )
    initial = {**_base_state(config), "state": "ready", "resumable": True}
    try:
        (temporary / _CONFIG_NAME).write_bytes(config.source_bytes)
        write_json(temporary / _STATE_NAME, initial)
        os.replace(temporary, config.experiment_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return initial


def _load_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except (OSError, ValueError, RecursionError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _candidates(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(root.glob("generations/generation-*/slot-*/candidate.json.gz")):
        value = _load_mapping(path)
        if value is not None:
            values.append(value)
    return values


def _manifests(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(root.glob("generations/generation-*/manifest.json.gz")):
        value = _load_mapping(path)
        if value is not None:
            values.append(value)
    return values


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _fitness_interval_fractions(
    profile: Mapping[str, Any],
) -> tuple[Fraction, Fraction]:
    interval = profile.get("fitness_interval")
    if not isinstance(interval, Mapping):
        raise PythonPreviewWorkspaceError("fitness interval is malformed")
    values: list[Fraction] = []
    for endpoint in ("lower", "upper"):
        raw = interval.get(endpoint)
        numerator = raw.get("numerator") if isinstance(raw, Mapping) else None
        denominator = raw.get("denominator") if isinstance(raw, Mapping) else None
        if (
            not isinstance(numerator, int)
            or isinstance(numerator, bool)
            or not isinstance(denominator, int)
            or isinstance(denominator, bool)
            or denominator == 0
        ):
            raise PythonPreviewWorkspaceError("fitness interval is malformed")
        values.append(Fraction(numerator, denominator))
    if values[0] > values[1]:
        raise PythonPreviewWorkspaceError("fitness interval is inverted")
    return values[0], values[1]


def _fitness_value(profile: Mapping[str, Any]) -> float:
    lower, _ = _fitness_interval_fractions(profile)
    return float(lower)


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _fitness_delta_interval(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    candidate_lower, candidate_upper = _fitness_interval_fractions(candidate)
    reference_lower, reference_upper = _fitness_interval_fractions(reference)
    return {
        "lower": _fraction_dict(candidate_lower - reference_upper),
        "upper": _fraction_dict(candidate_upper - reference_lower),
    }


def _usage_total(
    candidates: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any] | None,
) -> dict[str, int]:
    keys = (
        "inputTokens",
        "cachedInputTokens",
        "cacheWriteInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    totals = {key: 0 for key in keys}
    attempts: list[Mapping[str, Any]] = []
    if anchor is not None:
        attempts.append(anchor)
    for candidate in candidates:
        raw_attempts = candidate.get("provider_attempts")
        if isinstance(raw_attempts, Sequence) and not isinstance(raw_attempts, str | bytes):
            attempts.extend(item for item in raw_attempts if isinstance(item, Mapping))
    for attempt in attempts:
        usage = attempt.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in keys:
            totals[key] += _nonnegative_int(usage.get(key))
    return totals


_EVALUATION_TELEMETRY_FIELDS = (
    "starts",
    "rotations",
    "failures",
    "timeouts",
    "maximum_rss_kib",
    "policy_invocations",
    "graph_score_attempts",
    "unique_graph_scores",
    "sandbox_wall_seconds",
    "selector_wall_seconds",
    "action_wall_seconds",
    "scoring_wall_seconds",
)
_EVALUATION_TELEMETRY_MAX_FIELDS = frozenset({"maximum_rss_kib"})


def _compact_evaluation_telemetry(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int | float], bool]:
    """Sum bounded panel aggregates without opening per-case evidence."""

    totals: dict[str, int | float] = {
        key: 0.0 if key.endswith("_seconds") else 0
        for key in _EVALUATION_TELEMETRY_FIELDS
    }
    complete = True
    for record in records:
        if _nonnegative_int(record.get("evaluation_case_count")) == 0:
            continue
        summary = record.get("evaluation_telemetry")
        if not isinstance(summary, Mapping):
            complete = False
            continue
        for key in _EVALUATION_TELEMETRY_FIELDS:
            raw = summary.get(key)
            if (
                not isinstance(raw, int | float)
                or isinstance(raw, bool)
                or raw < 0
            ):
                complete = False
                break
            value: int | float = (
                float(raw) if key.endswith("_seconds") else int(raw)
            )
            if key in _EVALUATION_TELEMETRY_MAX_FIELDS:
                totals[key] = max(totals[key], value)
            else:
                totals[key] += value
    return totals, complete


def _provider_telemetry(
    root: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor = _load_mapping(root / "anchor.json.gz")
    attempts = sum(
        len(raw)
        for candidate in candidates
        if isinstance((raw := candidate.get("provider_attempts")), Sequence)
        and not isinstance(raw, str | bytes)
    )
    repairs = sum(_nonnegative_int(item.get("repairs")) for item in candidates)
    warnings = sum(_nonnegative_int(item.get("warnings")) for item in candidates)
    provider_state = _load_mapping(root / "provider-runtime" / "provider-state.json.gz")
    counters = (
        provider_state.get("telemetry")
        if isinstance(provider_state, Mapping)
        and isinstance(provider_state.get("telemetry"), Mapping)
        else {}
    )
    return {
        "turns": attempts + (1 if anchor is not None else 0),
        "candidate_turns": attempts,
        "contract_repairs": repairs,
        "transport_retries": _nonnegative_int(
            cast(Mapping[str, Any], counters).get("transport_retries")
        ),
        "warnings": warnings
        + _nonnegative_int(cast(Mapping[str, Any], anchor or {}).get("warnings")),
        "process_restarts": _nonnegative_int(
            cast(Mapping[str, Any], counters).get("process_restarts")
        ),
        "thread_resume_attempts": _nonnegative_int(
            cast(Mapping[str, Any], counters).get("thread_resume_attempts")
        ),
        "forks": len(
            list(
                root.glob("generations/generation-*/slot-*/provider-*/fork/m5-fork-result.json.gz")
            )
        ),
        "usage": _usage_total(candidates, anchor),
    }


def _program_projection(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates[:64]:
        profile = candidate.get("behavior_profile")
        result.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "generation": candidate.get("generation"),
                "slot": candidate.get("slot"),
                "kind": candidate.get("kind"),
                "status": candidate.get("status"),
                "parent_candidate_id": candidate.get("parent_candidate_id"),
                "program_hash": candidate.get("program_hash"),
                "behavior_signature": candidate.get("behavior_signature"),
                "fitness_interval": (
                    profile.get("fitness_interval") if isinstance(profile, Mapping) else None
                ),
                "selector_frequencies": (
                    profile.get("selector_frequencies") if isinstance(profile, Mapping) else {}
                ),
                "action_frequencies": (
                    profile.get("action_frequencies") if isinstance(profile, Mapping) else {}
                ),
                "accepted_rewrites": (
                    _nonnegative_int(profile.get("accepted_rewrite_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "rejected_rewrites": (
                    _nonnegative_int(profile.get("rejected_rewrite_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "no_plan": (
                    _nonnegative_int(profile.get("no_plan_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "program_failures": (
                    _nonnegative_int(profile.get("program_failure_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "illegal_final_states": (
                    _nonnegative_int(profile.get("illegal_final_state_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
            }
        )
    return result


def _progress(
    config: PythonPreviewConfig,
    retained_state: Mapping[str, Any],
) -> dict[str, Any]:
    root = config.experiment_root
    manifests = _manifests(root)
    candidates = _candidates(root)
    statuses = Counter(str(item.get("status", "unknown")) for item in candidates)
    planned = sum(
        len(cast(Sequence[object], manifest.get("slots", ())))
        for manifest in manifests
        if isinstance(manifest.get("slots"), Sequence)
        and not isinstance(manifest.get("slots"), str | bytes)
    )
    terminal = sum(
        count for status, count in statuses.items() if status in _TERMINAL_CANDIDATE_STATUSES
    )
    pending = max(0, planned - terminal)
    allocations = Counter(
        str(slot.get("kind"))
        for manifest in manifests
        for slot in cast(Sequence[Mapping[str, Any]], manifest.get("slots", ()))
        if isinstance(slot, Mapping)
    )
    report = _load_mapping(root / M10_REPORT_FILENAME) or _load_mapping(root / "m5-report.json.gz")
    stop = _load_mapping(root / M10_STOP_FILENAME) or _load_mapping(root / "m5-stop.json.gz")
    runtime = _load_mapping(root / M10_RUNTIME_FILENAME) or {}
    state = str(retained_state.get("state", "ready"))
    resumable = bool(retained_state.get("resumable", True))
    run_terminal = retained_state.get("run_terminal") is True
    terminal_reason = (
        str(retained_state["terminal_reason"])
        if retained_state.get("terminal_reason") is not None
        else None
    )
    if terminal_reason is None and retained_state.get("last_error") == "FileNotFoundError":
        terminal_reason = "provider_runtime_missing"
    elif terminal_reason is None and retained_state.get("last_error"):
        terminal_reason = "infrastructure_failure"
    result_kind = str(retained_state.get("scientific_result_kind", "NONE"))
    exact = retained_state.get("scientific_success") is True
    report_protocol = report.get("protocol_id") if report is not None else None
    active_boundary = (
        state == "running"
        and retained_state.get("last_boundary") not in {None, "report_persisted"}
    )
    if report is not None and (
        report_protocol == M10_REPORT_PROTOCOL_ID
        or (report_protocol == M5_REPORT_PROTOCOL_ID and retained_state.get("state") == "completed")
    ) and not active_boundary and not retained_state.get("last_error"):
        terminal_reason = str(report.get("stop_reason", "generation_budget"))
        generation_limit_stop = (
            report_protocol == M10_REPORT_PROTOCOL_ID
            and terminal_reason == "generation_budget"
        )
        report_resumable = generation_limit_stop or terminal_reason in {
            "hourly_token_limit",
            "wall_clock_budget",
        }
        state = "blocked" if report_resumable else "completed"
        resumable = report_resumable
        run_terminal = not report_resumable
        exact = report.get("exact_verified") is True
        result_kind = "VERIFIED_COUNTEREXAMPLE" if exact else "DEVELOPMENT_SEARCH_EVIDENCE"
    elif stop is not None and state != "running":
        state = "blocked"
        resumable = stop.get("resumable") is True
        result_kind = "NO_SCIENTIFIC_RESULT"
    generations = [_nonnegative_int(manifest.get("generation")) for manifest in manifests]
    evaluation_workload: Mapping[str, JsonValue] | None = None
    baseline_projection: dict[str, JsonValue] = {}
    baseline_details: dict[str, JsonValue] = {}
    baseline_profiles: list[Mapping[str, Any]] = []
    baseline_results: list[Mapping[str, Any]] = []
    generation_baseline_profiles: dict[int, dict[str, Mapping[str, Any]]] = {}
    if config.scientific_search is not None:
        baseline_projection = {
            baseline: None for baseline in config.scientific_search.evaluation.baselines
        }
        baseline_details = {
            baseline: None for baseline in config.scientific_search.evaluation.baselines
        }
        for generation in generations:
            generation_baseline_profiles[generation] = {}
            summary = _load_mapping(
                root / "generations" / f"generation-{generation:04d}" / M10_BASELINE_FILENAME
            )
            raw_baselines = (
                summary.get("baselines")
                if summary is not None and isinstance(summary.get("baselines"), Mapping)
                else {}
            )
            for baseline in config.scientific_search.evaluation.baselines:
                individual = _load_mapping(
                    root
                    / "generations"
                    / f"generation-{generation:04d}"
                    / "baselines"
                    / baseline
                    / M10_BASELINE_RESULT_FILENAME
                )
                if individual is not None:
                    baseline_results.append(individual)
                profile = (
                    individual.get("profile")
                    if individual is not None
                    else cast(Mapping[str, Any], raw_baselines).get(baseline)
                )
                if isinstance(profile, Mapping):
                    baseline_profiles.append(profile)
                    generation_baseline_profiles[generation][baseline] = profile
                    if generation == max(generations, default=-1):
                        baseline_projection[baseline] = _fitness_value(profile)
                        baseline_details[baseline] = cast(
                            JsonValue,
                            {
                                "status": "complete",
                                "fitness_interval": profile.get("fitness_interval"),
                                "evaluation_case_count": len(
                                    cast(
                                        Sequence[object],
                                        profile.get(
                                            "episode_statuses",
                                            (),
                                        ),
                                    )
                                )
                                if isinstance(
                                    profile.get("episode_statuses"),
                                    Sequence,
                                )
                                and not isinstance(
                                    profile.get("episode_statuses"),
                                    str | bytes,
                                )
                                else 0,
                            },
                        )
        if generations:
            current_generation = max(generations)
            current_manifest = next(
                (
                    manifest
                    for manifest in manifests
                    if _nonnegative_int(manifest.get("generation")) == current_generation
                ),
                None,
            )
            raw_panel = current_manifest.get("panel") if current_manifest is not None else None
            if isinstance(raw_panel, Sequence) and not isinstance(raw_panel, str | bytes):
                panel = tuple(
                    DevelopmentCaseV1.from_dict(case)
                    for case in raw_panel
                    if isinstance(case, Mapping)
                )
                evaluation_workload = workload_projection(
                    generation=current_generation,
                    panel=panel,
                    options=config.scientific_search.evaluation,
                )
    generation_objectives: list[dict[str, JsonValue]] = []
    for generation in generations:
        generation_candidates = [
            item
            for item in candidates
            if _nonnegative_int(item.get("generation")) == generation
        ]
        generation_statuses = Counter(
            str(item.get("status", "unknown")) for item in generation_candidates
        )
        ranked: list[
            tuple[
                Fraction,
                Fraction,
                str,
                Mapping[str, Any],
                Mapping[str, Any],
            ]
        ] = []
        for generation_candidate in generation_candidates:
            profile = generation_candidate.get("behavior_profile")
            if (
                generation_candidate.get("status") not in {"evaluated", "duplicate"}
                or not isinstance(profile, Mapping)
            ):
                continue
            lower, upper = _fitness_interval_fractions(profile)
            ranked.append(
                (
                    lower,
                    upper,
                    str(generation_candidate.get("program_hash", "")),
                    generation_candidate,
                    profile,
                )
            )
        best_entry = min(
            ranked,
            key=lambda item: (-item[0], -item[1], item[2]),
            default=None,
        )
        generation_best_candidate = best_entry[3] if best_entry is not None else None
        generation_best_profile = best_entry[4] if best_entry is not None else None
        baseline_values: dict[str, JsonValue] = {}
        for baseline in (
            config.scientific_search.evaluation.baselines
            if config.scientific_search is not None
            else ()
        ):
            profile = generation_baseline_profiles.get(generation, {}).get(baseline)
            baseline_values[baseline] = cast(
                JsonValue,
                (
                    {
                        "fitness_interval": profile.get("fitness_interval"),
                        "value": _fitness_value(profile),
                        "best_minus_reference": (
                            _fitness_delta_interval(generation_best_profile, profile)
                            if generation_best_profile is not None
                            else None
                        ),
                    }
                    if profile is not None
                    else None
                ),
            )
        generation_objectives.append(
            {
                "generation": generation,
                "panel_hash": (
                    (
                        _load_mapping(
                            root
                            / "generations"
                            / f"generation-{generation:04d}"
                            / M10_BASELINE_FILENAME
                        )
                        or {}
                    ).get("panel_hash")
                ),
                "best": cast(
                    JsonValue,
                    (
                        {
                            "candidate_id": generation_best_candidate.get(
                                "candidate_id"
                            ),
                            "program_hash": generation_best_candidate.get(
                                "program_hash"
                            ),
                            "fitness_interval": generation_best_profile.get(
                                "fitness_interval"
                            ),
                            "value": _fitness_value(generation_best_profile),
                        }
                        if generation_best_candidate is not None
                        and generation_best_profile is not None
                        else None
                    ),
                ),
                "baselines": baseline_values,
                "archive": {
                    "valid": generation_statuses["evaluated"]
                    + generation_statuses["duplicate"],
                    "invalid": generation_statuses["contract_invalid"],
                    "failed": generation_statuses["provider_failed"]
                    + generation_statuses["evaluation_infrastructure_failure"],
                    "duplicate": generation_statuses["duplicate"],
                },
            }
        )
    telemetry, profiling_available = _compact_evaluation_telemetry(
        [*candidates, *baseline_results]
    )
    candidate_case_count = sum(
        _nonnegative_int(candidate.get("evaluation_case_count"))
        for candidate in candidates
    )
    baseline_case_count = sum(
        _nonnegative_int(result.get("evaluation_case_count"))
        for result in baseline_results
    )
    profiles = _program_projection(candidates)
    behavior_profiles = [
        cast(Mapping[str, Any], item["behavior_profile"])
        for item in candidates
        if isinstance(item.get("behavior_profile"), Mapping)
    ]
    program_failure_episodes = sum(
        _nonnegative_int(profile.get("program_failure_count")) for profile in behavior_profiles
    )
    program_failed_candidates = sum(
        _nonnegative_int(profile.get("program_failure_count")) > 0 for profile in behavior_profiles
    )
    accepted_rewrites = sum(
        _nonnegative_int(profile.get("accepted_rewrite_count")) for profile in behavior_profiles
    )
    no_plans = sum(_nonnegative_int(profile.get("no_plan_count")) for profile in behavior_profiles)
    illegal_final_states = sum(
        _nonnegative_int(profile.get("illegal_final_state_count")) for profile in behavior_profiles
    )
    selector_frequencies: Counter[str] = Counter()
    action_frequencies: Counter[str] = Counter()
    for profile in behavior_profiles:
        selectors = profile.get("selector_frequencies")
        if isinstance(selectors, Mapping):
            selector_frequencies.update(
                {str(key): _nonnegative_int(value) for key, value in selectors.items()}
            )
        actions = profile.get("action_frequencies")
        if isinstance(actions, Mapping):
            action_frequencies.update(
                {str(key): _nonnegative_int(value) for key, value in actions.items()}
            )
    elapsed = max(
        0.0,
        float(runtime.get("active_elapsed_seconds", 0.0))
        if isinstance(runtime.get("active_elapsed_seconds"), int | float)
        and not isinstance(runtime.get("active_elapsed_seconds"), bool)
        else 0.0,
    )
    provider_wait = max(
        0.0,
        float(runtime.get("provider_wait_seconds", 0.0))
        if isinstance(runtime.get("provider_wait_seconds"), int | float)
        and not isinstance(runtime.get("provider_wait_seconds"), bool)
        else 0.0,
    )
    provider_active_wall = max(
        0.0,
        float(runtime.get("provider_active_wall_seconds", 0.0))
        if isinstance(runtime.get("provider_active_wall_seconds"), int | float)
        and not isinstance(runtime.get("provider_active_wall_seconds"), bool)
        else 0.0,
    )
    evaluator_busy = max(
        0.0,
        float(runtime.get("evaluator_busy_seconds", 0.0))
        if isinstance(runtime.get("evaluator_busy_seconds"), int | float)
        and not isinstance(runtime.get("evaluator_busy_seconds"), bool)
        else 0.0,
    )
    persistence = max(
        0.0,
        float(runtime.get("persistence_seconds", 0.0))
        if isinstance(runtime.get("persistence_seconds"), int | float)
        and not isinstance(runtime.get("persistence_seconds"), bool)
        else 0.0,
    )
    phase_times = {
        "provider": provider_wait,
        "evaluator/scorer": evaluator_busy,
        "persistence": persistence,
    }
    dominant_key, dominant_value = max(
        phase_times.items(),
        key=lambda item: (item[1], item[0]),
    )
    sorted_phase_values = sorted(phase_times.values(), reverse=True)
    bottleneck = (
        "balanced"
        if len(sorted_phase_values) > 1
        and dominant_value > 0
        and sorted_phase_values[1] / dominant_value >= 0.8
        else dominant_key
    )
    configured_workers = (
        config.scientific_search.evaluator_workers if config.scientific_search is not None else 1
    )
    active_workers = _nonnegative_int(runtime.get("active_evaluators"))
    evaluator_capacity_seconds = elapsed * configured_workers
    evaluator_idle_seconds = max(
        0.0,
        evaluator_capacity_seconds - evaluator_busy,
    )
    evaluator_queue_wait = max(
        0.0,
        float(runtime.get("evaluator_queue_wait_seconds", 0.0))
        if isinstance(
            runtime.get("evaluator_queue_wait_seconds"),
            int | float,
        )
        and not isinstance(
            runtime.get("evaluator_queue_wait_seconds"),
            bool,
        )
        else 0.0,
    )
    current_run_elapsed = max(
        0.0,
        float(runtime.get("current_run_elapsed_seconds", 0.0))
        if isinstance(runtime.get("current_run_elapsed_seconds"), int | float)
        and not isinstance(runtime.get("current_run_elapsed_seconds"), bool)
        else 0.0,
    )
    raw_evaluation_progress = runtime.get("evaluation_progress", {})
    evaluation_progress: dict[str, dict[str, int | float | str]] = {}
    if isinstance(raw_evaluation_progress, Mapping):
        for raw_key, raw_value in raw_evaluation_progress.items():
            if not isinstance(raw_key, str) or not isinstance(raw_value, Mapping):
                continue
            completed = _nonnegative_int(raw_value.get("completed"))
            total = _nonnegative_int(raw_value.get("total"))
            evaluation_progress[raw_key] = {
                "completed": min(completed, total),
                "total": total,
                "queued": _nonnegative_int(raw_value.get("queued")),
                "running": _nonnegative_int(raw_value.get("running")),
                "state": (
                    str(raw_value["state"])
                    if raw_value.get("state") in {"queued", "running", "terminal"}
                    else "queued"
                ),
            }
    active_evaluation_cases = sum(
        int(item["completed"]) for item in evaluation_progress.values()
    )
    active_evaluation_total = sum(int(item["total"]) for item in evaluation_progress.values())
    profiling_available = profiling_available and active_evaluation_cases == 0

    def active_evaluation_case_projection(prefix: str) -> dict[str, int]:
        active = [
            item
            for key, item in evaluation_progress.items()
            if key.startswith(f"{prefix}:")
        ]
        return {
            "active_completed": sum(int(item["completed"]) for item in active),
            "active_total": sum(int(item["total"]) for item in active),
        }

    baseline_evaluation_total = 0
    candidate_evaluation_total = 0
    configured_baseline_count = (
        len(config.scientific_search.evaluation.baselines)
        if config.scientific_search is not None
        else 0
    )
    for manifest in manifests:
        raw_manifest_panel = manifest.get("panel")
        panel_case_count = (
            len(raw_manifest_panel)
            if isinstance(raw_manifest_panel, Sequence)
            and not isinstance(raw_manifest_panel, str | bytes)
            else 0
        )
        baseline_evaluation_total += panel_case_count * configured_baseline_count
        generation = _nonnegative_int(manifest.get("generation"))
        slots = manifest.get("slots")
        if not isinstance(slots, Sequence) or isinstance(slots, str | bytes):
            continue
        for raw_slot in slots:
            slot = raw_slot.get("slot") if isinstance(raw_slot, Mapping) else None
            if (
                isinstance(slot, str)
                and (
                    root
                    / "generations"
                    / f"generation-{generation:04d}"
                    / slot
                    / "prepared-candidate.json.gz"
                ).is_file()
            ):
                candidate_evaluation_total += panel_case_count
    if config.scientific_search is None:
        baseline_completed = _nonnegative_int(
            runtime.get("baseline_evaluation_cases_completed")
        )
        baseline_total = _nonnegative_int(
            runtime.get("baseline_evaluation_cases_total")
        )
        candidate_completed = _nonnegative_int(
            runtime.get("candidate_evaluation_cases_completed")
        )
        candidate_total = _nonnegative_int(
            runtime.get("candidate_evaluation_cases_total")
        )
    else:
        baseline_completed = min(
            baseline_evaluation_total,
            baseline_case_count
            + active_evaluation_case_projection("baseline")["active_completed"],
        )
        baseline_total = baseline_evaluation_total
        candidate_completed = min(
            candidate_evaluation_total,
            candidate_case_count
            + active_evaluation_case_projection("candidate")["active_completed"],
        )
        candidate_total = candidate_evaluation_total
    baseline_evaluation_cases = {
        **active_evaluation_case_projection("baseline"),
        "completed": baseline_completed,
        "total": baseline_total,
    }
    candidate_evaluation_cases = {
        **active_evaluation_case_projection("candidate"),
        "completed": candidate_completed,
        "total": candidate_total,
    }
    raw_active_evaluation_work = runtime.get("active_evaluation_work", {})
    active_evaluation_work = (
        cast(Mapping[str, Mapping[str, JsonValue]], raw_active_evaluation_work)
        if isinstance(raw_active_evaluation_work, Mapping)
        else {}
    )
    policy_invocations = telemetry["policy_invocations"]
    graph_score_attempts = telemetry["graph_score_attempts"]
    last_improvement = runtime.get("last_scientific_improvement_epoch_seconds")
    time_since_improvement = (
        max(0.0, time.time() - float(last_improvement))
        if isinstance(last_improvement, int | float) and not isinstance(last_improvement, bool)
        else None
    )
    best_candidate_id = runtime.get("best_candidate_id")
    best_candidate = next(
        (item for item in candidates if item.get("candidate_id") == best_candidate_id),
        None,
    )
    best_program = _program_projection([best_candidate])[0] if best_candidate is not None else None
    verifier_submissions = sum(
        _nonnegative_int(
            cast(Mapping[str, Any], item.get("behavior_profile", {})).get(
                "exact_verifier_submissions"
            )
        )
        for item in candidates
        if isinstance(item.get("behavior_profile"), Mapping)
    )
    verifier_submissions += sum(
        _nonnegative_int(profile.get("exact_verifier_submissions")) for profile in baseline_profiles
    )
    verifier_records = sum(
        _nonnegative_int(
            cast(Mapping[str, Any], item.get("behavior_profile", {})).get("exact_verifier_records")
        )
        for item in candidates
        if isinstance(item.get("behavior_profile"), Mapping)
    )
    verifier_records += sum(
        _nonnegative_int(profile.get("exact_verifier_records")) for profile in baseline_profiles
    )
    exact |= any(profile.get("exact_verified") is True for profile in baseline_profiles)
    provider_projection = _provider_telemetry(root, candidates)
    if config.scientific_search is not None:
        reserved_turns = runtime.get("provider_turns_submitted")
        if (
            isinstance(reserved_turns, int)
            and not isinstance(reserved_turns, bool)
            and reserved_turns >= 0
        ):
            provider_projection["candidate_turns"] = reserved_turns
            provider_projection["turns"] = reserved_turns + int((root / "anchor.json.gz").is_file())
    candidates_by_id = {
        str(item["candidate_id"]): item
        for item in candidates
        if isinstance(item.get("candidate_id"), str)
    }
    provider_turns_by_key: dict[str, Mapping[str, Any]] = {}
    raw_timeline = runtime.get("provider_concurrency_timeline", ())
    resume_started_epoch = runtime.get("resume_started_epoch_seconds")
    resume_started_epoch = (
        float(resume_started_epoch)
        if isinstance(resume_started_epoch, int | float)
        and not isinstance(resume_started_epoch, bool)
        else None
    )
    if isinstance(raw_timeline, Sequence) and not isinstance(raw_timeline, str | bytes):
        for raw_turn in raw_timeline:
            if not isinstance(raw_turn, Mapping):
                continue
            key = raw_turn.get("key")
            if isinstance(key, str):
                provider_turns_by_key[key] = raw_turn

    def provider_turn_started(turn: Mapping[str, Any]) -> float:
        value = turn.get("started_epoch_seconds")
        return (
            float(value)
            if isinstance(value, int | float) and not isinstance(value, bool)
            else 0.0
        )

    active_provider_keys = {
        key
        for key, raw_turn in provider_turns_by_key.items()
        if isinstance(raw_turn, Mapping)
        and "finished_epoch_seconds" not in raw_turn
        and (
            resume_started_epoch is None
            or (
                isinstance(raw_turn.get("started_epoch_seconds"), int | float)
                and not isinstance(raw_turn.get("started_epoch_seconds"), bool)
                and float(raw_turn["started_epoch_seconds"]) >= resume_started_epoch
            )
        )
    }
    if _nonnegative_int(runtime.get("active_provider_turns")) == 0:
        active_provider_keys.clear()
    active_provider_elapsed = sum(
        max(0.0, time.time() - provider_turn_started(raw_turn))
        for key, raw_turn in provider_turns_by_key.items()
        if key in active_provider_keys
    )
    timing_profile = {
        "phase_seconds": {
            "provider/transport": provider_wait + active_provider_elapsed,
            "evaluator/scorer": evaluator_busy,
            "sandbox": telemetry["sandbox_wall_seconds"],
            "HEG scoring": telemetry["scoring_wall_seconds"],
            "persistence": persistence,
        },
        "phase_calls": {
            "provider/transport": _nonnegative_int(runtime.get("provider_turns_submitted")),
            "evaluator/scorer": _nonnegative_int(runtime.get("completed_evaluations")),
        },
        "unattributed_fraction": 0.0,
    }
    legacy_runtime_stale = (
        resume_started_epoch is None
        and not isinstance(runtime.get("current_run_elapsed_seconds"), int | float)
        and _nonnegative_int(runtime.get("active_provider_turns")) == 0
    )

    active_provider_candidate_ids: list[str] = []
    slot_projection: list[dict[str, JsonValue]] = []
    for manifest in manifests:
        generation = _nonnegative_int(manifest.get("generation"))
        raw_slots = manifest.get("slots", ())
        if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, str | bytes):
            continue
        for raw_slot in raw_slots:
            if not isinstance(raw_slot, Mapping):
                continue
            slot = raw_slot.get("slot")
            if not isinstance(slot, str):
                continue
            candidate_id = f"g{generation:04d}-{slot}"
            candidate = candidates_by_id.get(candidate_id)
            request_key = raw_slot.get("request_key")
            provider_turn_key: str | None = None
            provider_turn: Mapping[str, Any] | None = None
            if isinstance(request_key, str):
                request_prefix = f"{request_key}-"
                related_turns = [
                    (key, turn)
                    for key, turn in provider_turns_by_key.items()
                    if key == f"{request_key}-initial"
                    or key.startswith(f"{request_key}-initial-resume-")
                    or key.startswith(f"{request_prefix}repair-")
                ]
                if related_turns:
                    # An active repair belongs to the same slot as its
                    # completed initial turn.  Prefer it over the retained
                    # initial record, then prefer the newest turn.
                    provider_turn_key, provider_turn = min(
                        related_turns,
                        key=lambda item: (
                            item[0] not in active_provider_keys,
                            -provider_turn_started(item[1]),
                        ),
                    )
            provider_started = (
                provider_turn.get("started_epoch_seconds") if provider_turn is not None else None
            )
            provider_finished = (
                provider_turn.get("finished_epoch_seconds") if provider_turn is not None else None
            )
            started_epoch = (
                float(provider_started)
                if isinstance(provider_started, int | float)
                and not isinstance(provider_started, bool)
                else None
            )
            current_provider_turn = (
                provider_turn
                if provider_turn is not None
                and not legacy_runtime_stale
                and provider_turn_key in active_provider_keys
                else None
            )
            finished_epoch = (
                float(provider_finished)
                if isinstance(provider_finished, int | float)
                and not isinstance(provider_finished, bool)
                else None
            )
            elapsed_seconds = (
                max(0.0, (finished_epoch or time.time()) - started_epoch)
                if started_epoch is not None
                else None
            )
            prepared = (
                root
                / "generations"
                / f"generation-{generation:04d}"
                / slot
                / "prepared-candidate.json.gz"
            ).is_file()
            progress = evaluation_progress.get(f"candidate:{candidate_id}")
            progress_state = (
                str(progress.get("state"))
                if isinstance(progress, Mapping)
                else "queued"
            )
            provider_artifact_dir = (
                root
                / "generations"
                / f"generation-{generation:04d}"
                / slot
                / (
                    f"provider-repair-{provider_turn_key.rsplit('-repair-', 1)[-1]}"
                    if provider_turn_key is not None and "-repair-" in provider_turn_key
                    else f"provider-resume-{provider_turn_key.rsplit('-resume-', 1)[-1]}"
                    if provider_turn_key is not None and "-resume-" in provider_turn_key
                    else "provider-initial"
                )
            )
            provider_turn_terminal = (
                current_provider_turn is not None
                and any(provider_artifact_dir.glob("*.turn-terminal.json*"))
            )
            candidate_status = (
                str(candidate.get("status"))
                if candidate is not None
                else f"evaluation_{progress_state}"
                if prepared
                else "persisting"
                if provider_turn_terminal
                else "model"
                if current_provider_turn is not None
                else "failed"
                if (
                    provider_turn is not None
                    and (
                        provider_turn.get("failed") is True
                        or (
                            state == "blocked"
                            and terminal_reason
                            in {"infrastructure_failure", "provider_runtime_missing"}
                            and retained_state.get("last_error")
                        )
                    )
                )
                else "queued"
            )
            if (
                current_provider_turn is not None
            ):
                active_provider_candidate_ids.append(candidate_id)
            slot_projection.append(
                {
                    "candidate_id": candidate_id,
                    "generation": generation,
                    "slot": slot,
                    "kind": str(raw_slot.get("kind", "root")),
                    "parent_candidate_id": (
                        str(raw_slot["parent_candidate_id"])
                        if raw_slot.get("parent_candidate_id") is not None
                        else None
                    ),
                    "state": candidate_status,
                    "phase": (
                        "archived"
                        if candidate_status in _TERMINAL_CANDIDATE_STATUSES
                        else "evaluation"
                        if prepared
                        else "response"
                        if provider_turn_terminal
                        else "provider"
                    ),
                    "started_epoch_seconds": (
                        started_epoch if current_provider_turn is not None else None
                    ),
                    "elapsed_seconds": (
                        elapsed_seconds if current_provider_turn is not None else None
                    ),
                    "repairs": (
                        _nonnegative_int(candidate.get("repairs")) if candidate is not None else 0
                    ),
                    "usage": cast(
                        JsonValue,
                        (
                            dict(
                                cast(
                                    Mapping[str, JsonValue],
                                    candidate.get("usage", {}),
                                )
                            )
                            if candidate is not None and isinstance(candidate.get("usage"), Mapping)
                            else {}
                        ),
                    ),
                    "evaluation_completed": (
                        int(progress["completed"]) if progress is not None else 0
                    ),
                    "evaluation_total": (
                        int(progress["total"]) if progress is not None else None
                    ),
                    "error": (
                        str(provider_turn.get("error"))
                        if provider_turn is not None and provider_turn.get("error")
                        else str(retained_state.get("last_error"))
                        if candidate_status == "failed" and retained_state.get("last_error")
                        else None
                    ),
                }
            )
    hourly_usage = hourly_token_usage(
        root,
        (
            config.scientific_search.max_total_tokens_per_hour
            if config.scientific_search is not None
            else None
        ),
    )
    return {
        **_public(retained_state),
        **hourly_usage,
        "state": state,
        "resumable": resumable,
        "run_terminal": run_terminal,
        "terminal_reason": terminal_reason,
        "scientific_result_kind": result_kind,
        "scientific_success": exact,
        "search_protocol": (
            M10_SEARCH_PROTOCOL_ID
            if config.scientific_search is not None
            else M5_SEARCH_PROTOCOL_ID
        ),
        "safe_api_expanded": config.scientific_search is not None,
        "generation_index": max(generations, default=None),
        "generation_manifest_hashes": [manifest.get("sha256") for manifest in manifests],
        "counts": {
            "planned": planned,
            "terminal": terminal,
            "pending": pending,
            "valid": statuses["evaluated"] + statuses["duplicate"],
            "contract_invalid": statuses["contract_invalid"],
            "duplicate": statuses["duplicate"],
            "provider_failed": statuses["provider_failed"],
            "evaluation_infrastructure_failure": statuses["evaluation_infrastructure_failure"],
            "evaluated": statuses["evaluated"],
            "missing": statuses["missing"],
            "roots": allocations["root"],
            "children": allocations["child"],
            **(
                {
                    "program_failed": program_failed_candidates,
                    "repaired_valid": sum(
                        _nonnegative_int(item.get("repairs")) > 0
                        and item.get("status") in {"evaluated", "duplicate"}
                        for item in candidates
                    ),
                }
                if config.scientific_search is not None
                else {}
            ),
        },
        "candidate_status_counts": dict(sorted(statuses.items())),
        "provider": {
            **provider_projection,
            "program_turns_reserved": _nonnegative_int(runtime.get("provider_turns_submitted")),
            "primary_turns": _nonnegative_int(runtime.get("primary_turns_submitted")),
            "repair_turns": _nonnegative_int(runtime.get("repair_turns_submitted")),
            "active": _nonnegative_int(runtime.get("active_provider_turns")),
            "peak_active": _nonnegative_int(runtime.get("peak_active_provider_turns")),
            "configured_concurrency": (
                config.scientific_search.provider_concurrency
                if config.scientific_search is not None
                else 1
            ),
            "active_candidate_ids": active_provider_candidate_ids,
            "active_wall_seconds": max(0.0, provider_active_wall),
            "concurrency_timeline": cast(
                JsonValue,
                runtime.get("provider_concurrency_timeline", []),
            ),
            "wait_seconds": provider_wait,
        },
        "evaluation_progress": cast(JsonValue, evaluation_progress),
        "evaluation_cases": {
            "active_completed": active_evaluation_cases,
            "active_total": active_evaluation_total,
            "completed": baseline_completed + candidate_completed,
            "total": baseline_total + candidate_total,
            "baseline": baseline_evaluation_cases,
            "candidate": candidate_evaluation_cases,
        },
        "sandbox": {
            key: telemetry[key] if profiling_available else None
            for key in (
                "starts",
                "rotations",
                "failures",
                "timeouts",
                "maximum_rss_kib",
            )
        },
        "policy_invocations": (
            policy_invocations if profiling_available else None
        ),
        "graph_scores": {
            "attempts": (
                graph_score_attempts if profiling_available else None
            ),
            "unique_graphs": (
                telemetry["unique_graph_scores"]
                if profiling_available
                else None
            ),
        },
        "evaluators": {
            "configured": configured_workers,
            "active": active_workers,
            "idle": max(0, configured_workers - active_workers),
            "peak_active": _nonnegative_int(runtime.get("peak_active_evaluators")),
            "queued": _nonnegative_int(runtime.get("queued_evaluations")),
            "peak_queued": _nonnegative_int(runtime.get("peak_queued_evaluations")),
            "completed": _nonnegative_int(runtime.get("completed_evaluations")),
            "failed": _nonnegative_int(runtime.get("failed_evaluations")),
            "busy_seconds": evaluator_busy,
            "idle_capacity_seconds": evaluator_idle_seconds,
            "queue_wait_seconds": evaluator_queue_wait,
            "utilization": (
                evaluator_busy / evaluator_capacity_seconds
                if evaluator_capacity_seconds > 0
                else 0.0
            ),
            "active_work": cast(
                JsonValue,
                [
                    {
                        "worker_id": worker_id,
                        **dict(item),
                    }
                    for worker_id, item in sorted(
                        active_evaluation_work.items()
                    )
                    if isinstance(item, Mapping)
                ],
            ),
        },
        "throughput": {
            "policy_invocations_per_second": (policy_invocations / elapsed if elapsed > 0 else 0.0),
            "graph_score_attempts_per_second": (
                graph_score_attempts / elapsed if elapsed > 0 else 0.0
            ),
            "accepted_rewrites_per_second": (accepted_rewrites / elapsed if elapsed > 0 else 0.0),
            "valid_unique_programs_per_provider_minute": (
                statuses["evaluated"] / (provider_active_wall / 60.0)
                if provider_active_wall > 0
                else 0.0
            ),
            "provider_wait_share": (provider_active_wall / elapsed if elapsed > 0 else 0.0),
            "elapsed_seconds": elapsed,
            "current_run_elapsed_seconds": current_run_elapsed,
        },
        "scientific_activity": {
            "accepted_rewrites": accepted_rewrites,
            "program_failure_episodes": program_failure_episodes,
            "no_plan_count": no_plans,
            "no_plan_rate": (no_plans / policy_invocations if policy_invocations > 0 else 0.0),
            "illegal_final_state_count": illegal_final_states,
            "illegal_final_state_rate": (
                illegal_final_states / policy_invocations if policy_invocations > 0 else 0.0
            ),
            "selector_frequencies": dict(sorted(selector_frequencies.items())),
            "action_frequencies": dict(sorted(action_frequencies.items())),
        },
        "phase_timings": {
            "provider_wait_seconds": provider_wait,
            "evaluator_busy_seconds": evaluator_busy,
            "sandbox_seconds": telemetry["sandbox_wall_seconds"],
            "selector_seconds": telemetry["selector_wall_seconds"],
            "action_seconds": telemetry["action_wall_seconds"],
            "heg_scoring_seconds": telemetry["scoring_wall_seconds"],
            "persistence_seconds": persistence,
            "dominant_bottleneck": bottleneck,
        },
        "profiling_enabled": profiling_available,
        "timing_profile": timing_profile if profiling_available else None,
        "best": {
            "candidate_id": best_candidate_id,
            "fitness_interval": runtime.get("best_fitness_interval"),
            "program": best_program,
            "seconds_since_scientific_improvement": time_since_improvement,
        },
        "programs": profiles,
        "slots": slot_projection,
        "recovery": {
            "state": (
                "terminal" if run_terminal else "resumable" if resumable and candidates else "fresh"
            ),
            "resume_attempts": _nonnegative_int(retained_state.get("resume_attempts")),
            "last_boundary": retained_state.get("last_boundary"),
            "completed_slots_will_not_repeat": True,
        },
        "exact_verification": {
            "authority": "exact_verifier_only",
            "submissions": verifier_submissions,
            "records": verifier_records,
            "verified": exact,
            "queue": 0,
        },
        "evaluation_workload": evaluation_workload,
        "baselines": baseline_projection,
        "baseline_details": baseline_details,
        "generation_objectives": generation_objectives,
        "equal_development_panel": (
            evaluation_workload.get("panel_hash") if evaluation_workload is not None else None
        ),
    }


def python_preview_status(
    config_path: str | Path,
    *,
    pause_record_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a bounded read-only status projection without provider contact."""

    config = load_python_preview_config(config_path)
    if not config.experiment_root.exists():
        return _progress(config, _base_state(config))
    try:
        state = _load_state(config)
    except (OSError, ValueError, PythonPreviewWorkspaceError) as error:
        return {
            **_public(_base_state(config)),
            "state": "failed",
            "run_terminal": True,
            "terminal_reason": "workspace_mismatch",
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "last_error": _safe_error(error, config),
        }
    state, _, _ = _generation_limit_lifecycle(config, state)
    status = _progress(config, state)
    if pause_record_path is None:
        return status
    return _apply_pause_record(config, status, Path(pause_record_path))


def python_preview_bootstrap_status(
    config_path: str | Path,
) -> dict[str, Any]:
    """Return the minimal campaign state needed for the first dashboard frame."""

    config = load_python_preview_config(config_path)
    if not config.experiment_root.exists():
        return {
            **_public(_base_state(config)),
            "profiling_enabled": False,
        }
    try:
        state = _load_state(config)
    except (OSError, ValueError, PythonPreviewWorkspaceError) as error:
        return {
            **_public(_base_state(config)),
            "state": "failed",
            "run_terminal": True,
            "terminal_reason": "workspace_mismatch",
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "last_error": _safe_error(error, config),
            "profiling_enabled": False,
        }
    state, report, _ = _generation_limit_lifecycle(config, state)
    status = {
        **_public(state),
        "profiling_enabled": False,
    }
    if report is None:
        return status
    return {
        **status,
        "search_protocol": M10_SEARCH_PROTOCOL_ID,
        "generation_index": (
            _nonnegative_int(report.get("generation_count")) - 1
            if _nonnegative_int(report.get("generation_count"))
            else None
        ),
    }


def _apply_pause_record(
    config: PythonPreviewConfig,
    status: Mapping[str, Any],
    pause_record_path: Path,
) -> dict[str, Any]:
    resolved_pause_record = pause_record_path.resolve()
    try:
        raw_record = json.loads(resolved_pause_record.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raw_record = None
    record = dict(raw_record) if isinstance(raw_record, Mapping) else None
    if (
        record is None
        or record.get("schema_version") != _M10_PAUSE_RECORD_SCHEMA_VERSION
        or record.get("state") != _PAUSED_FOR_BUDGET
        or record.get("experiment") != config.exp_id
    ):
        raise PythonPreviewWorkspaceError(
            "Budget pause record does not match this ordinary-Python workspace"
        )

    slots_record = record.get("slots")
    counts = status.get("counts")
    if not isinstance(slots_record, Mapping) or not isinstance(counts, Mapping):
        raise PythonPreviewWorkspaceError("Budget pause slot accounting is malformed")
    terminal = _nonnegative_int(slots_record.get("terminal_total"))
    pending = _nonnegative_int(slots_record.get("pending_total"))
    if terminal != _nonnegative_int(counts.get("terminal")) or pending != _nonnegative_int(
        counts.get("pending")
    ):
        raise PythonPreviewWorkspaceError(
            "Budget pause slot accounting does not match durable workspace artifacts"
        )

    interrupted = _string_sequence(
        slots_record.get("in_flight_slots"),
        "slots.in_flight_slots",
    )
    unstarted = _string_sequence(
        slots_record.get("pending_unstarted_slots"),
        "slots.pending_unstarted_slots",
    )
    if (
        len(interrupted) != _nonnegative_int(slots_record.get("in_flight_cancelled_at_stop"))
        or len(interrupted) + len(unstarted) != pending
        or set(interrupted).intersection(unstarted)
    ):
        raise PythonPreviewWorkspaceError("Budget pause pending identities are malformed")

    provider_record = record.get("provider_turns")
    best_record = record.get("best")
    exact_record = record.get("exact_verifier")
    if (
        not isinstance(provider_record, Mapping)
        or not isinstance(best_record, Mapping)
        or not isinstance(exact_record, Mapping)
    ):
        raise PythonPreviewWorkspaceError("Budget pause scientific accounting is malformed")
    raw_usage = provider_record.get("persisted_usage_including_specification_anchor")
    fitness = best_record.get("fitness_interval")
    if not isinstance(raw_usage, Mapping) or not isinstance(fitness, Mapping):
        raise PythonPreviewWorkspaceError("Budget pause scientific accounting is malformed")

    usage_fields = {
        "inputTokens": "input_tokens",
        "cachedInputTokens": "cached_input_tokens",
        "outputTokens": "output_tokens",
        "reasoningOutputTokens": "reasoning_output_tokens",
        "totalTokens": "total_tokens",
    }
    usage = {
        target: _nonnegative_int(raw_usage.get(source)) for target, source in usage_fields.items()
    }
    usage["cacheWriteInputTokens"] = 0

    provider = status.get("provider")
    evaluators = status.get("evaluators")
    best = status.get("best")
    exact = status.get("exact_verification")
    recovery = status.get("recovery")
    provider_projection = dict(provider) if isinstance(provider, Mapping) else {}
    evaluator_projection = dict(evaluators) if isinstance(evaluators, Mapping) else {}
    best_projection = dict(best) if isinstance(best, Mapping) else {}
    exact_projection = dict(exact) if isinstance(exact, Mapping) else {}
    recovery_projection = dict(recovery) if isinstance(recovery, Mapping) else {}
    provider_projection.update(
        {
            "program_turns_reserved": _nonnegative_int(provider_record.get("started_reservations")),
            "primary_turns": _nonnegative_int(provider_record.get("primary_turns_submitted")),
            "repair_turns": _nonnegative_int(provider_record.get("repair_turns_submitted")),
            "completed_turns": _nonnegative_int(provider_record.get("completed_turns")),
            "interrupted_turns": _nonnegative_int(
                provider_record.get("in_flight_started_without_finished")
            ),
            "active": 0,
            "usage": usage,
        }
    )
    evaluator_projection["active"] = 0
    best_projection.update(
        {
            "candidate_id": best_record.get("candidate_id"),
            "program_hash": best_record.get("program_hash"),
            "program": {
                **(
                    dict(best_projection["program"])
                    if isinstance(best_projection.get("program"), Mapping)
                    else {}
                ),
                "program_hash": best_record.get("program_hash"),
            },
            "fitness_interval": dict(fitness),
        }
    )
    exact_projection.update(
        {
            "submissions": _nonnegative_int(exact_record.get("candidate_submissions")),
            "records": _nonnegative_int(exact_record.get("candidate_results")),
            "verified": exact_record.get("all_candidate_exact_verified") is True,
        }
    )
    recovery_projection.update(
        {
            "state": "resumable",
            "completed_slots_will_not_repeat": True,
        }
    )

    paused_slot_states = {
        **{candidate_id: "interrupted" for candidate_id in interrupted},
        **{candidate_id: "queued" for candidate_id in unstarted},
    }
    slot_projection: list[dict[str, Any]] = []
    raw_slots = status.get("slots")
    if isinstance(raw_slots, Sequence) and not isinstance(raw_slots, str | bytes):
        for raw_slot in raw_slots:
            if not isinstance(raw_slot, Mapping):
                continue
            candidate_id = raw_slot.get("candidate_id")
            projected = dict(raw_slot)
            if isinstance(candidate_id, str) and candidate_id in paused_slot_states:
                projected["state"] = paused_slot_states[candidate_id]
                projected["phase"] = "paused"
            slot_projection.append(projected)
    missing_pending = set(paused_slot_states).difference(
        str(item.get("candidate_id")) for item in slot_projection
    )
    if missing_pending:
        raise PythonPreviewWorkspaceError(
            "Budget pause identities do not match durable generation manifests"
        )

    return {
        **status,
        "state": _PAUSED_FOR_BUDGET,
        "resumable": True,
        "run_terminal": False,
        "terminal_reason": "provider_budget",
        "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
        "scientific_success": False,
        "counts": {**counts, "terminal": terminal, "pending": pending},
        "provider": provider_projection,
        "evaluators": evaluator_projection,
        "best": best_projection,
        "slots": slot_projection,
        "recovery": recovery_projection,
        "exact_verification": exact_projection,
        "pause": {
            "state": _PAUSED_FOR_BUDGET,
            "record_path": str(resolved_pause_record),
            "interrupted_slots": list(interrupted),
            "unstarted_slots": list(unstarted),
            "resumable_pending": pending,
        },
    }


def _string_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise PythonPreviewWorkspaceError(f"Budget pause {name} must be a list")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise PythonPreviewWorkspaceError(f"Budget pause {name} must contain non-empty strings")
    return cast(tuple[str, ...], items)


def _stop_request_path(config: PythonPreviewConfig) -> Path:
    return config.experiment_root / _STOP_REQUEST_NAME


def _stop_requested(config: PythonPreviewConfig) -> bool:
    path = _stop_request_path(config)
    if not path.is_file():
        return False
    value = _load_mapping(path)
    if (
        value is None
        or value.get("protocol_id") != _STOP_REQUEST_PROTOCOL_ID
        or not isinstance(value.get("active"), bool)
    ):
        raise PythonPreviewWorkspaceError("Python preview stop request is malformed")
    return value["active"] is True


def request_python_preview_stop(
    config_path: str | Path,
) -> dict[str, Any]:
    """Request a resumable stop at the next durable candidate boundary."""

    config = load_python_preview_config(config_path)
    state = _load_state(config)
    if state.get("state") not in {"ready", "running"} or state.get("run_terminal") is True:
        raise PythonPreviewWorkspaceError(
            "Python preview is not running and cannot accept a stop request"
        )
    write_json(
        _stop_request_path(config),
        {
            "protocol_id": _STOP_REQUEST_PROTOCOL_ID,
            "active": True,
        },
    )
    return {
        **python_preview_status(config_path),
        "stop_requested": True,
    }


def _consume_stop_request(config: PythonPreviewConfig) -> None:
    if not _stop_requested(config):
        return
    write_json(
        _stop_request_path(config),
        {
            "protocol_id": _STOP_REQUEST_PROTOCOL_ID,
            "active": False,
        },
    )


def _prompt_inputs() -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    system_prompt = (
        (_PROJECT_ROOT / "prompts/native-v3-python/m5-system.md")
        .read_text(encoding="utf-8")
        .strip()
    )
    request_template = (_PROJECT_ROOT / "prompts/native-v3-python/m4-request.md").read_text(
        encoding="utf-8"
    )
    specification_prompt = (
        "Retain the complete policy specification below for later root and "
        "mutation turns. Do not generate a policy on this turn. Return only "
        "the required specification acknowledgement.\n\n" + request_template
    )
    policy_schema = json.loads(
        (_PROJECT_ROOT / "configs/native/native-v3-python-policy-response.schema.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(policy_schema, dict):
        raise ValueError("Python policy output schema must be an object")
    _assert_current_policy_contract(request_template, policy_schema)
    return (
        system_prompt,
        request_template,
        specification_prompt,
        policy_schema,
        specification_ack_schema(),
    )


def _assert_current_policy_contract(
    request_template: str,
    policy_schema: Mapping[str, Any],
) -> None:
    if (
        "def propose(ctx, graph, api, seed):" not in request_template
        or "priority(ctx, proposal)" in request_template
    ):
        raise ValueError("ordinary-Python request does not use the canonical propose contract")
    properties = policy_schema.get("properties")
    required = policy_schema.get("required")
    if (
        not isinstance(properties, Mapping)
        or set(properties) != {"schema_version", "source"}
        or not isinstance(required, Sequence)
        or isinstance(required, str | bytes)
        or set(required) != {"schema_version", "source"}
        or policy_schema.get("additionalProperties") is not False
        or not isinstance(properties.get("schema_version"), Mapping)
        or cast(Mapping[str, Any], properties["schema_version"]).get("const")
        != "mforge.native.python_policy_response.v1"
    ):
        raise ValueError("ordinary-Python output schema is not the canonical two-field schema")


def _panel(backend: GraphBackend) -> tuple[DevelopmentCaseV1, ...]:
    forbidden_lengths = backend.target_forbidden_lengths(30)
    return (
        DevelopmentCaseV1(
            case_id="order-30-seed-101",
            order=30,
            graph_seed=101,
            policy_seed=17,
            horizon=1,
            witness_cap=64,
            forbidden_lengths=forbidden_lengths,
            graph_mode="unrestricted_min_degree_3",
        ),
        DevelopmentCaseV1(
            case_id="order-30-seed-103",
            order=30,
            graph_seed=103,
            policy_seed=19,
            horizon=1,
            witness_cap=64,
            forbidden_lengths=forbidden_lengths,
            graph_mode="unrestricted_min_degree_3",
        ),
    )


def _default_provider(
    config: PythonPreviewConfig,
    system_prompt: str,
) -> M5SearchProvider | M10SearchProvider:
    if config.scientific_search is not None:
        return CodexM10SearchProvider(
            workspace=config.experiment_root / "provider-runtime",
            model=config.model,
            effort=config.effort,
            base_instructions=system_prompt,
            auth_json=Path.home() / ".codex" / "auth.json",
            turn_timeout_seconds=config.timeout_seconds,
            provider_concurrency=(config.scientific_search.provider_concurrency),
            provider_total_turn_limit=(
                config.scientific_search.current_provider_total_turn_limit
            ),
        )
    return CodexM5SearchProvider(
        workspace=config.experiment_root / "provider-runtime",
        model=config.model,
        effort=config.effort,
        base_instructions=system_prompt,
        auth_json=Path.home() / ".codex" / "auth.json",
        turn_timeout_seconds=config.timeout_seconds,
        program_turn_limit=None,
    )


def _default_backend(config: PythonPreviewConfig) -> GraphBackend:
    graph_mode = (
        config.scientific_search.evaluation.graph_mode
        if config.scientific_search is not None
        else "unrestricted_min_degree_3"
    )
    return HegBackend(config.heg_repo, graph_mode=graph_mode)


def _default_evaluator(
    config: PythonPreviewConfig,
    backend: GraphBackend,
) -> M10ScientificEvaluator:
    return PythonPanelScientificEvaluator(
        backend=backend,
        artifact_root=config.experiment_root / "scientific-artifacts",
        runtime_limits=PolicyRuntimeLimitsV1(),
    )


class _BackendOwnedEvaluator:
    """Close one evaluator worker's private backend with the worker."""

    def __init__(
        self,
        *,
        evaluator: M10ScientificEvaluator,
        backend: GraphBackend,
    ) -> None:
        self._evaluator = evaluator
        self._backend = backend

    def set_worker_name(self, name: str) -> None:
        setter = getattr(self._backend, "set_score_worker_name", None)
        if callable(setter):
            setter(name)

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        return self._evaluator.evaluate(
            source=source,
            case=case,
            candidate_id=candidate_id,
        )

    def evaluate_baseline(
        self,
        *,
        baseline: str,
        case: DevelopmentCaseV1,
        generation: int,
    ) -> Mapping[str, JsonValue]:
        return self._evaluator.evaluate_baseline(
            baseline=baseline,
            case=case,
            generation=generation,
        )

    def close(self) -> None:
        self._backend.close()


class _RemoteEvaluationError(RuntimeError):
    def __init__(self, error_type: str, message: str) -> None:
        self.remote_type = error_type
        super().__init__(f"{error_type}: {message}" if message else error_type)


def _set_evaluator_process_name(name: str) -> None:
    if not sys.platform.startswith("linux"):
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(15, name[:15].encode("ascii", "replace"), 0, 0, 0)


def _evaluation_process_main(
    connection: Any,
    config: PythonPreviewConfig,
    backend_factory: BackendFactory,
    evaluator_factory: EvaluatorFactory,
) -> None:
    backend: GraphBackend | None = None
    try:
        process_group = None
        if hasattr(os, "setsid"):
            os.setsid()
            process_group = os.getpgrp()
        backend = backend_factory(config)
        evaluator = _BackendOwnedEvaluator(
            evaluator=cast(
                M10ScientificEvaluator,
                evaluator_factory(config, backend),
            ),
            backend=backend,
        )
        connection.send(("ready", os.getpid(), process_group))
        while True:
            request = connection.recv()
            if not isinstance(request, tuple) or not request:
                raise RuntimeError("evaluation worker request is malformed")
            operation = request[0]
            if operation == "close":
                connection.send(("closed",))
                return
            if operation == "name":
                name = str(request[1])[:15]
                _set_evaluator_process_name(name)
                evaluator.set_worker_name(name)
                connection.send(("named",))
                continue
            try:
                if operation == "candidate":
                    result = evaluator.evaluate(
                        source=str(request[1]),
                        case=cast(DevelopmentCaseV1, request[2]),
                        candidate_id=str(request[3]),
                    )
                elif operation == "baseline":
                    result = evaluator.evaluate_baseline(
                        baseline=str(request[1]),
                        case=cast(DevelopmentCaseV1, request[2]),
                        generation=int(request[3]),
                    )
                else:
                    raise RuntimeError("evaluation worker operation is unknown")
                connection.send(("result", dict(result)))
            except Exception as error:
                connection.send(("error", type(error).__name__, str(error)[:2048]))
    except BaseException as error:
        with suppress(Exception):
            connection.send(("init_error", type(error).__name__, str(error)[:2048]))
    finally:
        if backend is not None:
            with suppress(Exception):
                backend.close()
        connection.close()


class _ProcessOwnedEvaluator:
    """One persistent evaluator process behind a synchronous case interface."""

    def __init__(
        self,
        config: PythonPreviewConfig,
        *,
        backend_factory: BackendFactory = _default_backend,
        evaluator_factory: EvaluatorFactory = _default_evaluator,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._closed = False
        self._process_group: int | None = None
        self._process = context.Process(
            target=_evaluation_process_main,
            args=(child, config, backend_factory, evaluator_factory),
        )
        self._process.start()
        child.close()
        if not parent.poll(30):
            self.close(force=True)
            raise RuntimeError("evaluation worker did not initialize")
        response = parent.recv()
        if not isinstance(response, tuple) or not response or response[0] != "ready":
            self.close(force=True)
            error_type = str(response[1]) if len(response) > 1 else "RuntimeError"
            message = str(response[2]) if len(response) > 2 else "evaluation worker failed"
            raise _RemoteEvaluationError(error_type, message)
        self.worker_pid = int(response[1])
        self._process_group = (
            int(response[2]) if isinstance(response[2], int) else None
        )

    def _request(self, request: tuple[Any, ...]) -> Mapping[str, JsonValue]:
        if self._closed:
            raise RuntimeError("evaluation worker is closed")
        try:
            self._connection.send(request)
            response = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError) as error:
            raise RuntimeError("evaluation worker exited unexpectedly") from error
        if not isinstance(response, tuple) or not response:
            raise RuntimeError("evaluation worker response is malformed")
        if response[0] == "error":
            raise _RemoteEvaluationError(str(response[1]), str(response[2]))
        if response[0] != "result" or not isinstance(response[1], Mapping):
            raise RuntimeError("evaluation worker response is malformed")
        return cast(Mapping[str, JsonValue], response[1])

    def set_worker_name(self, name: str) -> None:
        if self._closed:
            return
        self._connection.send(("name", name[:15]))
        response = self._connection.recv()
        if response != ("named",):
            raise RuntimeError("evaluation worker did not accept its process name")

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        return self._request(("candidate", source, case, candidate_id))

    def evaluate_baseline(
        self,
        *,
        baseline: str,
        case: DevelopmentCaseV1,
        generation: int,
    ) -> Mapping[str, JsonValue]:
        return self._request(("baseline", baseline, case, generation))

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if force:
            self._terminate_process_group()
        else:
            try:
                self._connection.send(("close",))
                if self._connection.poll(5):
                    self._connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._terminate_process_group()
        self._connection.close()

    def _terminate_process_group(self) -> None:
        process_group = self._process_group
        if (
            self._process.is_alive()
            and process_group is not None
            and process_group == self._process.pid
            and hasattr(os, "killpg")
        ):
            with suppress(ProcessLookupError):
                os.killpg(process_group, signal.SIGTERM)
        elif self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=0.5)
        if self._process.is_alive():
            if (
                process_group is not None
                and process_group == self._process.pid
                and hasattr(os, "killpg")
            ):
                with suppress(ProcessLookupError):
                    os.killpg(process_group, signal.SIGKILL)
            else:
                self._process.kill()
            self._process.join(timeout=0.5)


def run_python_preview(
    config_path: str | Path,
    *,
    provider_factory: ProviderFactory = _default_provider,
    backend_factory: BackendFactory = _default_backend,
    evaluator_factory: EvaluatorFactory = _default_evaluator,
    provenance_guard: ProvenanceGuard = ensure_m5_acceptance_provenance,
    auth_available: Callable[[Path], bool] = Path.is_file,
    resume_budget: ScientificResumeBudgetV1 | None = None,
    force_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run or resume the explicit ordinary-Python preview."""

    config = load_python_preview_config(config_path)
    existed = config.experiment_root.exists()
    state = _load_state(config) if existed else _initialize_workspace(config)
    generation_limit_state, generation_limit_report, at_generation_limit = (
        _generation_limit_lifecycle(config, state)
    )
    if existed and generation_limit_state != state:
        state = generation_limit_state
        _write_state(config, state)
    if existed and at_generation_limit and generation_limit_report is not None:
        return _generation_limit_noop_status(state, generation_limit_report)
    if existed and _provider_runtime_missing(config.experiment_root):
        if generation_limit_report is not None:
            blocked = {
                **state,
                "state": "blocked",
                "resumable": False,
                "run_terminal": False,
                "terminal_reason": _LEGACY_GENERATION_LIMIT_CAPSULE_MISSING,
                "scientific_result_kind": "DEVELOPMENT_SEARCH_EVIDENCE",
                "scientific_success": False,
                "last_error": "FileNotFoundError",
            }
            _write_state(config, blocked)
            return _generation_limit_noop_status(blocked, generation_limit_report)
        failed = {
            **state,
            "state": "failed",
            "resumable": False,
            "run_terminal": True,
            "terminal_reason": "provider_runtime_missing",
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": "FileNotFoundError",
        }
        _write_state(config, failed)
        return _progress(config, failed)
    if resume_budget is not None and (
        not existed or config.scientific_search is None or state.get("resumable") is not True
    ):
        raise PythonPreviewWorkspaceError(
            "current-generation budget requires a resumable scientific workspace"
        )
    if resume_budget is not None:
        assert config.scientific_search is not None
        resolve_resume_generation(
            root=config.experiment_root,
            options=config.scientific_search,
            budget=resume_budget,
        )
    resumable_terminal_reasons = {
        "resume_generation_complete",
        "hourly_token_limit",
        "wall_clock_budget",
    }
    if (
        state.get("state") == "completed"
        and state.get("terminal_reason") not in resumable_terminal_reasons
        and not (
            config.scientific_search is not None
            and state.get("terminal_reason") == "generation_budget"
        )
    ):
        return _progress(config, state)
    if not (config.heg_repo / "src" / "sglab").is_dir():
        blocked = {
            **state,
            "state": "blocked",
            "resumable": True,
            "last_error": "HEG repository is unavailable",
        }
        _write_state(config, blocked)
        return _progress(config, blocked)
    (
        system_prompt,
        request_template,
        specification_prompt,
        policy_schema,
        ack_schema,
    ) = _prompt_inputs()
    _assert_current_policy_contract(request_template, policy_schema)
    try:
        provenance_guard(
            workspace=config.experiment_root,
            resume=existed,
            repository_root=_PROJECT_ROOT,
            heg_root=config.heg_repo,
            experiment_config=config.source_path,
            experiment_config_sha256=config.scientific_config_sha256,
            model=config.model,
            effort=config.effort,
            system_prompt=system_prompt,
            request_template=request_template,
            specification_prompt=specification_prompt,
            output_schema=policy_schema,
            specification_ack_schema=ack_schema,
            runtime_limits=PolicyRuntimeLimitsV1(),
            legacy_experiment_config_sha256=(
                _legacy_workspace_config_sha256(config)
                if existed
                else None
            ),
        )
    except Exception as error:
        failed = {
            **state,
            "state": "failed",
            "resumable": False,
            "run_terminal": True,
            "terminal_reason": "provenance_mismatch",
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": _safe_error(error, config),
        }
        _write_state(config, failed)
        return _progress(config, failed)
    if config.scientific_search is not None:
        hourly_usage = hourly_token_usage(
            config.experiment_root,
            config.scientific_search.max_total_tokens_per_hour,
        )
        if hourly_usage["hourly_limit_reached"] is True:
            blocked = {
                **state,
                "state": "blocked",
                "resumable": True,
                "run_terminal": False,
                "terminal_reason": "hourly_token_limit",
                "scientific_result_kind": "DEVELOPMENT_SEARCH_EVIDENCE",
                "scientific_success": False,
                "last_error": None,
            }
            _write_state(config, blocked)
            return _progress(config, blocked)
    auth_json = Path.home() / ".codex" / "auth.json"
    if not auth_available(auth_json):
        blocked = {
            **state,
            "state": "blocked",
            "resumable": True,
            "last_error": "local Codex authentication is unavailable",
        }
        _write_state(config, blocked)
        return _progress(config, blocked)
    running = {
        **state,
        "state": "running",
        "resumable": True,
        "run_terminal": False,
        "terminal_reason": None,
        "last_error": None,
        "resume_attempts": _nonnegative_int(state.get("resume_attempts")) + int(existed),
    }
    _write_state(config, running)

    def boundary_hook(boundary: str) -> None:
        running["last_boundary"] = boundary
        _write_state(config, running)

    provider: M5SearchProvider | M10SearchProvider | None = None
    backend: GraphBackend | None = None
    final_state: dict[str, Any]
    primary_error: Exception | None = None
    try:
        backend = backend_factory(config)
        provider = provider_factory(config, system_prompt)
        if config.scientific_search is None:
            panel = _panel(backend)
            evaluator = evaluator_factory(config, backend)
            report = run_m5_search(
                provider=provider,
                evaluator=evaluator,
                workspace=config.experiment_root,
                panel=panel,
                system_prompt=system_prompt,
                specification_prompt=specification_prompt,
                specification_ack_schema=ack_schema,
                policy_schema=policy_schema,
                preview_active=True,
                close_provider=False,
                boundary_hook=boundary_hook,
                operator_stop=lambda: _stop_requested(config),
            )
        else:
            scientific_search = config.scientific_search
            scientific_backend = backend
            process_isolation = (
                backend_factory is _default_backend
                and evaluator_factory is _default_evaluator
            )

            def make_evaluator() -> M10ScientificEvaluator:
                if process_isolation:
                    return _ProcessOwnedEvaluator(config)
                worker_backend = backend_factory(config)
                try:
                    evaluator = cast(
                        M10ScientificEvaluator,
                        evaluator_factory(config, worker_backend),
                    )
                except BaseException:
                    with suppress(Exception):
                        worker_backend.close()
                    raise
                return _BackendOwnedEvaluator(
                    evaluator=evaluator,
                    backend=worker_backend,
                )

            report = run_sustained_search(
                provider=cast(M10SearchProvider, provider),
                evaluator_factory=make_evaluator,
                workspace=config.experiment_root,
                panel_factory=lambda generation: scientific_search.evaluation.panel_for_generation(
                    generation=generation,
                    backend=scientific_backend,
                ),
                system_prompt=system_prompt,
                specification_prompt=specification_prompt,
                specification_ack_schema=ack_schema,
                policy_schema=policy_schema,
                options=config.scientific_search,
                provider_turn_timeout_seconds=config.timeout_seconds,
                resume_budget=resume_budget,
                boundary_hook=boundary_hook,
                operator_stop=lambda: _stop_requested(config),
                force_stop=force_stop,
            )
        resumable_stop = report.get("stop_reason") in {
            "resume_generation_complete",
            "hourly_token_limit",
            "wall_clock_budget",
        } or (
            config.scientific_search is not None
            and report.get("stop_reason") == "generation_budget"
        )
        final_state = {
            **running,
            "state": ("blocked" if resumable_stop else "completed"),
            "resumable": resumable_stop,
            "run_terminal": not resumable_stop,
            "terminal_reason": report.get("stop_reason"),
            "scientific_result_kind": (
                "VERIFIED_COUNTEREXAMPLE"
                if report.get("exact_verified") is True
                else "DEVELOPMENT_SEARCH_EVIDENCE"
            ),
            "scientific_success": report.get("exact_verified") is True,
            "last_error": None,
        }
    except M5OperatorStop as error:
        primary_error = error
        _consume_stop_request(config)
        final_state = {
            **running,
            "state": "blocked",
            "resumable": True,
            "run_terminal": False,
            "terminal_reason": "operator_stop",
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": None,
        }
    except Exception as error:
        primary_error = error
        final_state = {
            **running,
            "state": "blocked",
            "resumable": True,
            "run_terminal": False,
            "terminal_reason": (
                "provider_runtime_missing"
                if isinstance(error, FileNotFoundError)
                else "infrastructure_failure"
            ),
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": _safe_error(error, config),
        }
    cleanup_errors: list[Exception] = []
    if provider is not None:
        try:
            if isinstance(provider, CodexM5SearchProvider | CodexM10SearchProvider):
                provider.close(
                    cleanup_capsule=(
                        primary_error is None and final_state.get("state") == "completed"
                    )
                )
            else:
                provider.close()
        except Exception as error:
            cleanup_errors.append(error)
    if backend is not None:
        try:
            backend.close()
        except Exception as error:
            cleanup_errors.append(error)
    if cleanup_errors and primary_error is None:
        final_state = {
            **running,
            "state": "blocked",
            "resumable": True,
            "run_terminal": False,
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": f"{type(cleanup_errors[0]).__name__}_during_cleanup",
        }
    _write_state(config, final_state)
    return _progress(config, final_state)


__all__ = [
    "PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION",
    "PYTHON_PREVIEW_MODE",
    "PYTHON_PREVIEW_PROTOCOL_VERSION",
    "PYTHON_PREVIEW_STATE_SCHEMA_VERSION",
    "PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION",
    "V2_PROTOCOL",
    "PythonPreviewConfig",
    "PythonPreviewWorkspaceError",
    "experiment_protocol",
    "load_python_preview_config",
    "python_preview_bootstrap_status",
    "python_preview_status",
    "request_python_preview_stop",
    "run_python_preview",
]
