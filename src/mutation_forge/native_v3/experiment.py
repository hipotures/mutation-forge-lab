"""Explicit Native v3 routing for the public experiment CLI."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.config import validate_experiment_id
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider

from .cohort import run_sequential_cohort
from .preview import (
    FORBIDDEN_LENGTHS,
    PERSISTENT_SINGLE_AST,
    run_persistent_single_ast_cohort,
)
from .single_program_ir import (
    SLOT_SPECIFIC_OUTPUT_CONTRACT,
    slot_specific_contract_sha256,
)

V2_PROTOCOL = "native-v2"
V3_SELECTOR = "v3"
V3_CONFIG_SCHEMA_VERSION = "mforge.experiment.v3"
V3_PROTOCOL_VERSION = "v3"
V3_STATUS_SCHEMA_VERSION = "mforge.experiment.status.v3"
MULTI_PROGRAM_BATCH = "multi_program_batch"
V3_COMMUNICATION_MODES = frozenset({PERSISTENT_SINGLE_AST, MULTI_PROGRAM_BATCH})
V3_OUTPUT_CONTRACTS = frozenset({SLOT_SPECIFIC_OUTPUT_CONTRACT})
V3_DEFAULT_COMMUNICATION_MODE = MULTI_PROGRAM_BATCH
_STATE_NAME = "v3-state.json.gz"
_V2_TOP_LEVEL_FIELDS = frozenset(
    {"kind", "preset", "run", "model", "search", "evaluation", "resources"}
)

type ProviderFactory = Callable[["V3Config"], LocalCodexAppServerProvider]
type BackendFactory = Callable[["V3Config"], GraphBackend]
type AuthAvailable = Callable[[Path], bool]
type PreviewRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class V3Config:
    schema_version: str
    protocol: str
    exp_id: str
    workspace: Path
    model: str
    effort: str
    timeout_seconds: float
    heg_repo: Path
    communication_mode: str
    output_contract: str | None
    source_path: Path
    source_bytes: bytes = field(repr=False, compare=False)

    @property
    def experiment_root(self) -> Path:
        return self.workspace / self.exp_id

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_bytes).hexdigest()


class V3WorkspaceError(RuntimeError):
    """The selected workspace is not a compatible v3 experiment root."""


def _raw_config(path: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    source_path = Path(path).resolve()
    source_bytes = source_path.read_bytes()
    value = tomllib.loads(source_bytes.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must be a TOML table")
    return source_path, source_bytes, value


def experiment_protocol(path: str | Path = "experiment.toml") -> str:
    """Return the explicit v3 selector or the unchanged Native v2 default."""

    _, _, raw = _raw_config(path)
    protocol = raw.get("protocol")
    if protocol is None:
        return V2_PROTOCOL
    if protocol != V3_SELECTOR:
        raise ValueError(
            f"unsupported experiment protocol selector: {protocol!r}; "
            f"expected {V3_SELECTOR!r} or omit it for Native v2"
        )
    return V3_SELECTOR


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _resolved_path(value: object, name: str, source_dir: Path) -> Path:
    raw = Path(_string(value, name))
    return (source_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()


def load_v3_config(
    path: str | Path = "experiment.toml",
) -> V3Config:
    source_path, source_bytes, raw = _raw_config(path)
    mixed = sorted(_V2_TOP_LEVEL_FIELDS.intersection(raw))
    if mixed:
        raise ValueError(f"v3 configuration cannot contain Native v2 fields: {mixed}")
    allowed = {
        "schema_version",
        "protocol",
        "exp_id",
        "workspace",
        "v3",
    }
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported top-level fields: {unknown}")
    if raw.get("schema_version") != V3_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"v3 requires schema_version {V3_CONFIG_SCHEMA_VERSION!r}")
    if raw.get("protocol") != V3_SELECTOR:
        raise ValueError(f"v3 requires protocol {V3_SELECTOR!r}")
    exp_id = validate_experiment_id(raw.get("exp_id"))
    workspace = _resolved_path(raw.get("workspace"), "workspace", source_path.parent)
    v3_value = raw.get("v3")
    if not isinstance(v3_value, dict):
        raise ValueError("[v3] must be a table")
    v3 = cast(dict[str, Any], v3_value)
    allowed_v3 = {
        "model",
        "effort",
        "timeout_seconds",
        "heg_repo",
        "communication_mode",
        "output_contract",
    }
    unknown_v3 = sorted(set(v3).difference(allowed_v3))
    if unknown_v3:
        raise ValueError(f"unsupported [v3] fields: {unknown_v3}")
    model = _string(v3.get("model"), "v3.model")
    effort = _string(v3.get("effort"), "v3.effort")
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("v3.effort is unsupported")
    timeout_seconds = _positive_number(
        v3.get("timeout_seconds"),
        "v3.timeout_seconds",
    )
    heg_repo = _resolved_path(
        v3.get("heg_repo"),
        "v3.heg_repo",
        source_path.parent,
    )
    communication_mode = _string(
        v3.get("communication_mode", V3_DEFAULT_COMMUNICATION_MODE),
        "v3.communication_mode",
    )
    if communication_mode not in V3_COMMUNICATION_MODES:
        raise ValueError(f"v3.communication_mode must be one of {sorted(V3_COMMUNICATION_MODES)}")
    raw_output_contract = v3.get("output_contract")
    output_contract = (
        None
        if raw_output_contract is None
        else _string(raw_output_contract, "v3.output_contract")
    )
    if communication_mode == PERSISTENT_SINGLE_AST:
        if output_contract not in V3_OUTPUT_CONTRACTS:
            raise ValueError(
                "v3 persistent_single_ast requires explicit "
                f"output_contract {SLOT_SPECIFIC_OUTPUT_CONTRACT!r}"
            )
    elif output_contract is not None:
        raise ValueError("v3.output_contract is only valid for persistent_single_ast")
    return V3Config(
        V3_CONFIG_SCHEMA_VERSION,
        V3_SELECTOR,
        exp_id,
        workspace,
        model,
        effort,
        timeout_seconds,
        heg_repo,
        communication_mode,
        output_contract,
        source_path,
        source_bytes,
    )


def _output_schema_sha256(config: V3Config) -> str | None:
    if config.output_contract != SLOT_SPECIFIC_OUTPUT_CONTRACT:
        return None
    return slot_specific_contract_sha256(FORBIDDEN_LENGTHS)


def _base_status(config: V3Config) -> dict[str, Any]:
    return {
        "schema_version": V3_STATUS_SCHEMA_VERSION,
        "protocol": V3_SELECTOR,
        "protocol_version": V3_PROTOCOL_VERSION,
        "exp_id": config.exp_id,
        "workspace": str(config.experiment_root),
        "communication_mode": config.communication_mode,
        "provider_mode": config.communication_mode,
        "output_contract": config.output_contract,
        "output_schema_sha256": _output_schema_sha256(config),
        "compaction_mode": (
            "disabled"
            if config.communication_mode == PERSISTENT_SINGLE_AST
            else None
        ),
        "rollback_mode": MULTI_PROGRAM_BATCH,
        "diagnostic_mode": "fresh_single_ast",
        "state": "not_created",
        "resumable": True,
        "terminal": False,
        "latest_infrastructure_stop_reason": None,
        "latest_scientific_stop_reason": None,
        "last_stop_reason": None,
        "last_error": None,
        "provider_turns": 0,
        "program_turns": 0,
        "time_to_first_valid_ast_ms": None,
        "first_valid_ast_published_before_cohort_complete": None,
        "provider_attempts": 0,
        "failed_provider_attempts": 0,
        "provider_retries": 0,
        "provider_warnings": 0,
        "provider_process_restarts": 0,
        "thread_resume_attempts": 0,
        "failed_thread_resume_attempts": 0,
        "active_provider_attempt": None,
        "last_provider_attempt": None,
        "evaluation_count": 0,
        "valid_ast": False,
        "cohort_outcome": None,
        "valid_slots": 0,
        "unique_valid_programs": 0,
        "duplicate_aliases": 0,
        "selected_program_hash": None,
        "scientific_terminal_result": False,
        "usage": None,
        "artifacts": {},
    }


def _preview_progress(config: V3Config) -> dict[str, Any]:
    if config.communication_mode != PERSISTENT_SINGLE_AST:
        return {}
    path = (
        config.experiment_root / "native-v3-output" / "epoch-0000" / "communication-state.json.gz"
    )
    if not path.is_file():
        return {}
    try:
        raw = read_json(path)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, Mapping):
        return {}
    model_turns = int(raw.get("model_turns", 0))
    return {
        "provider_turns": model_turns,
        "program_turns": int(raw.get("program_turns", max(0, model_turns - 1))),
        "provider_attempts": int(raw.get("provider_attempts", model_turns)),
        "failed_provider_attempts": int(raw.get("failed_provider_attempts", 0)),
        "provider_retries": int(raw.get("provider_retries", 0)),
        "provider_warnings": int(raw.get("provider_warnings", 0)),
        "provider_process_restarts": int(raw.get("provider_process_restarts", 0)),
        "thread_resume_attempts": int(raw.get("thread_resume_attempts", 0)),
        "failed_thread_resume_attempts": int(raw.get("failed_thread_resume_attempts", 0)),
        "active_provider_attempt": raw.get("active_provider_attempt"),
        "last_provider_attempt": raw.get("last_provider_attempt"),
        "time_to_first_valid_ast_ms": raw.get("time_to_first_valid_ast_ms"),
    }


def _state_path(config: V3Config) -> Path:
    return config.experiment_root / _STATE_NAME


def _state_payload(
    config: V3Config,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(status),
        "config_sha256": config.source_sha256,
    }


def _public_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "config_sha256"}


def _load_workspace_state(config: V3Config) -> dict[str, Any]:
    state_path = _state_path(config)
    if not state_path.is_file():
        raise V3WorkspaceError(
            "existing workspace is not a v3 workspace; "
            "use a fresh exp_id and never reinterpret a Native v2 workspace"
        )
    value = read_json(state_path)
    if not isinstance(value, Mapping):
        raise V3WorkspaceError("v3 state is not an object")
    if (
        value.get("schema_version") != V3_STATUS_SCHEMA_VERSION
        or value.get("protocol") != V3_SELECTOR
        or value.get("protocol_version") != V3_PROTOCOL_VERSION
    ):
        raise V3WorkspaceError("v3 workspace protocol does not match this runtime")
    if value.get("config_sha256") != config.source_sha256:
        raise V3WorkspaceError("v3 configuration changed; create a fresh workspace")
    stored_config = config.experiment_root / "experiment.toml"
    if (
        not stored_config.is_file()
        or hashlib.sha256(stored_config.read_bytes()).hexdigest() != config.source_sha256
    ):
        raise V3WorkspaceError("v3 workspace configuration identity mismatch")
    return _public_state(value)


def _persist_state(
    config: V3Config,
    status: Mapping[str, Any],
) -> None:
    write_json(_state_path(config), _state_payload(config, status))


def _initialize_workspace(config: V3Config) -> dict[str, Any]:
    config.workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{config.exp_id}.v3.",
            dir=config.workspace,
        )
    )
    initial = {**_base_status(config), "state": "ready"}
    try:
        (temporary / "experiment.toml").write_bytes(config.source_bytes)
        write_json(temporary / _STATE_NAME, _state_payload(config, initial))
        os.replace(temporary, config.experiment_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return initial


def _failed_status(
    config: V3Config,
    error: Exception,
) -> dict[str, Any]:
    return {
        **_base_status(config),
        "state": "failed",
        "resumable": False,
        "last_stop_reason": "workspace_mismatch",
        "last_error": str(error),
    }


def v3_status(
    config_path: str | Path = "experiment.toml",
) -> dict[str, Any]:
    """Read v3 status without provider or scorer contact."""

    config = load_v3_config(config_path)
    if not config.experiment_root.exists():
        return _base_status(config)
    try:
        status = _load_workspace_state(config)
        return {**status, **_preview_progress(config)}
    except (OSError, ValueError, V3WorkspaceError) as error:
        return _failed_status(config, error)


def _default_provider(
    config: V3Config,
) -> LocalCodexAppServerProvider:
    return LocalCodexAppServerProvider(
        model=config.model,
        effort=config.effort,
        concurrency=1,
        max_repairs=1,
        turn_timeout_base_seconds=config.timeout_seconds / 2,
        auth_json=Path.home() / ".codex" / "auth.json",
        persist_artifacts=False,
    )


def _default_backend(config: V3Config) -> GraphBackend:
    return HegBackend(config.heg_repo)


def _blocked_preflight(
    config: V3Config,
    message: str,
) -> dict[str, Any]:
    return {
        **_base_status(config),
        "state": "blocked",
        "resumable": True,
        "latest_infrastructure_stop_reason": "preflight_failed",
        "last_stop_reason": "preflight_failed",
        "last_error": message,
    }


def _status_from_report(
    config: V3Config,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(report.get("status"))
    common = {
        **_base_status(config),
        **_preview_progress(config),
        "provider_turns": int(report.get("model_turns", 0)),
        "program_turns": int(report.get("program_turns", 0)),
        "time_to_first_valid_ast_ms": report.get("time_to_first_valid_ast_ms"),
        "first_valid_ast_published_before_cohort_complete": report.get(
            "first_valid_ast_published_before_cohort_complete"
        ),
        "evaluation_count": int(report.get("graph_evaluations", 0)),
        "valid_ast": report.get("valid_ast") is True,
        "cohort_outcome": report.get("cohort_outcome"),
        "valid_slots": int(report.get("valid_slots", 0)),
        "unique_valid_programs": int(report.get("unique_valid_programs", 0)),
        "duplicate_aliases": int(report.get("duplicate_aliases", 0)),
        "selected_program_hash": report.get("selected_program_hash"),
        "scientific_terminal_result": (report.get("scientific_terminal_result") is True),
        "usage": report.get("usage"),
        "artifacts": {
            key: report[key]
            for key in (
                "epoch_manifest",
                "cohort_report",
            )
            if isinstance(report.get(key), str)
        },
    }
    if status == "completed":
        return {
            **common,
            "state": "completed",
            "resumable": False,
            "terminal": True,
            "latest_scientific_stop_reason": "cohort_complete",
            "last_stop_reason": "cohort_complete",
        }
    if status == "inconclusive":
        return {
            **common,
            "state": "inconclusive",
            "resumable": False,
            "terminal": True,
            "latest_scientific_stop_reason": "cohort_inconclusive",
            "last_stop_reason": "cohort_inconclusive",
        }
    if status == "provider_error":
        authentication = report.get("error_classification") == "authentication"
        stop_reason = "preflight_failed" if authentication else "provider_failed"
        resumable = report.get("resumable") is True
        return {
            **common,
            "state": "blocked" if resumable else "failed",
            "resumable": resumable,
            "terminal": not resumable,
            "latest_infrastructure_stop_reason": stop_reason,
            "last_stop_reason": stop_reason,
            "last_error": report.get("error"),
        }
    if status in {"evaluation_error", "evaluation_failed"}:
        return {
            **common,
            "state": "blocked",
            "resumable": False,
            "latest_scientific_stop_reason": "program_failure",
            "last_stop_reason": "program_failure",
            "last_error": report.get("error"),
        }
    return {
        **common,
        "state": "blocked",
        "resumable": False,
        "latest_infrastructure_stop_reason": "evaluation_failed",
        "last_stop_reason": "evaluation_failed",
        "last_error": report.get("error"),
    }


def run_v3(
    config_path: str | Path = "experiment.toml",
    *,
    provider_factory: ProviderFactory = _default_provider,
    backend_factory: BackendFactory = _default_backend,
    auth_available: AuthAvailable = Path.is_file,
    preview_runner: PreviewRunner = run_persistent_single_ast_cohort,
) -> dict[str, Any]:
    """Run or resume the bounded sequential v3 cohort."""

    config = load_v3_config(config_path)
    if config.experiment_root.exists():
        status = _load_workspace_state(config)
        if status.get("state") == "completed":
            return status
        if status.get("resumable") is not True:
            raise V3WorkspaceError("v3 workspace is not resumable")
    else:
        status = _initialize_workspace(config)

    auth_json = Path.home() / ".codex" / "auth.json"
    if not auth_available(auth_json):
        status = _blocked_preflight(
            config,
            "local Codex authentication is unavailable",
        )
        _persist_state(config, status)
        return status
    if not (config.heg_repo / "src" / "sglab").is_dir():
        status = _blocked_preflight(
            config,
            f"HEG repository is unavailable: {config.heg_repo}",
        )
        _persist_state(config, status)
        return status

    running = {
        **status,
        "state": "running",
        "resumable": True,
        "last_stop_reason": None,
        "last_error": None,
    }
    _persist_state(config, running)
    provider: LocalCodexAppServerProvider | None = None
    try:
        if config.communication_mode == MULTI_PROGRAM_BATCH:
            provider = provider_factory(config)
            report = run_sequential_cohort(
                provider,
                config.experiment_root,
                backend_factory=lambda: backend_factory(config),
                episode_id=f"{config.exp_id}/epoch-0000",
            )
        else:
            report = preview_runner(
                config.experiment_root,
                model=config.model,
                effort=config.effort,
                timeout_seconds=config.timeout_seconds,
                auth_json=Path.home() / ".codex" / "auth.json",
                backend_factory=lambda: backend_factory(config),
                episode_id=f"{config.exp_id}/epoch-0000",
                output_contract=cast(str, config.output_contract),
            )
    except Exception as error:
        status = {
            **_base_status(config),
            **_preview_progress(config),
            "state": "blocked",
            "resumable": True,
            "latest_infrastructure_stop_reason": "provider_failed",
            "last_stop_reason": "provider_failed",
            "last_error": f"{type(error).__name__}: {error}",
        }
    else:
        status = _status_from_report(config, report)
    finally:
        if provider is not None:
            try:
                provider.close()
            except Exception as error:
                status = {
                    **_base_status(config),
                    "state": "failed",
                    "resumable": False,
                    "terminal": True,
                    "latest_infrastructure_stop_reason": "provider_close_failed",
                    "last_stop_reason": "provider_close_failed",
                    "last_error": f"{type(error).__name__}: {error}",
                }
    _persist_state(config, status)
    return status


__all__ = [
    "V2_PROTOCOL",
    "V3_CONFIG_SCHEMA_VERSION",
    "V3_DEFAULT_COMMUNICATION_MODE",
    "V3_COMMUNICATION_MODES",
    "V3_OUTPUT_CONTRACTS",
    "V3_PROTOCOL_VERSION",
    "V3_SELECTOR",
    "V3_STATUS_SCHEMA_VERSION",
    "V3Config",
    "V3WorkspaceError",
    "MULTI_PROGRAM_BATCH",
    "PERSISTENT_SINGLE_AST",
    "SLOT_SPECIFIC_OUTPUT_CONTRACT",
    "experiment_protocol",
    "load_v3_config",
    "run_v3",
    "v3_status",
]
