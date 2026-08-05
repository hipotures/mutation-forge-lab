"""Explicit one-slot Native v3 preview routing for the public experiment CLI."""

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

from .provider_evaluation import run_provider_evaluation_smoke
from .serial_evaluator import SerialEpisodeConfig

V2_PROTOCOL = "native-v2"
V3_PREVIEW_SELECTOR = "native-v3-preview"
V3_PREVIEW_CONFIG_SCHEMA_VERSION = "mforge.experiment.v3-preview.v1"
V3_PREVIEW_PROTOCOL_VERSION = "native-v3-preview.v1"
V3_PREVIEW_STATUS_SCHEMA_VERSION = "mforge.experiment.status.v3-preview.v1"
_STATE_NAME = "native-v3-preview-state.json.gz"
_V2_TOP_LEVEL_FIELDS = frozenset(
    {"kind", "preset", "run", "model", "search", "evaluation", "resources"}
)

type ProviderFactory = Callable[
    ["NativeV3PreviewConfig"], LocalCodexAppServerProvider
]
type BackendFactory = Callable[["NativeV3PreviewConfig"], GraphBackend]
type AuthAvailable = Callable[[Path], bool]


@dataclass(frozen=True, slots=True)
class NativeV3PreviewConfig:
    schema_version: str
    protocol: str
    exp_id: str
    workspace: Path
    model: str
    effort: str
    timeout_seconds: float
    heg_repo: Path
    source_path: Path
    source_bytes: bytes = field(repr=False, compare=False)

    @property
    def experiment_root(self) -> Path:
        return self.workspace / self.exp_id

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_bytes).hexdigest()


class NativeV3PreviewWorkspaceError(RuntimeError):
    """The selected workspace is not a compatible Native v3 preview root."""


def _raw_config(path: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    source_path = Path(path).resolve()
    source_bytes = source_path.read_bytes()
    value = tomllib.loads(source_bytes.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment configuration must be a TOML table")
    return source_path, source_bytes, value


def experiment_protocol(path: str | Path = "experiment.toml") -> str:
    """Return the explicit preview selector or the unchanged Native v2 default."""

    _, _, raw = _raw_config(path)
    protocol = raw.get("protocol")
    if protocol is None:
        return V2_PROTOCOL
    if protocol != V3_PREVIEW_SELECTOR:
        raise ValueError(
            f"unsupported experiment protocol selector: {protocol!r}; "
            f"expected {V3_PREVIEW_SELECTOR!r} or omit it for Native v2"
        )
    return V3_PREVIEW_SELECTOR


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


def load_v3_preview_config(
    path: str | Path = "experiment.toml",
) -> NativeV3PreviewConfig:
    source_path, source_bytes, raw = _raw_config(path)
    mixed = sorted(_V2_TOP_LEVEL_FIELDS.intersection(raw))
    if mixed:
        raise ValueError(
            "native-v3-preview configuration cannot contain Native v2 fields: "
            f"{mixed}"
        )
    allowed = {
        "schema_version",
        "protocol",
        "exp_id",
        "workspace",
        "native_v3_preview",
    }
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported top-level fields: {unknown}")
    if raw.get("schema_version") != V3_PREVIEW_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "native-v3-preview requires schema_version "
            f"{V3_PREVIEW_CONFIG_SCHEMA_VERSION!r}"
        )
    if raw.get("protocol") != V3_PREVIEW_SELECTOR:
        raise ValueError(
            f"native-v3-preview requires protocol {V3_PREVIEW_SELECTOR!r}"
        )
    exp_id = validate_experiment_id(raw.get("exp_id"))
    workspace = _resolved_path(raw.get("workspace"), "workspace", source_path.parent)
    preview_value = raw.get("native_v3_preview")
    if not isinstance(preview_value, dict):
        raise ValueError("[native_v3_preview] must be a table")
    preview = cast(dict[str, Any], preview_value)
    allowed_preview = {"model", "effort", "timeout_seconds", "heg_repo"}
    unknown_preview = sorted(set(preview).difference(allowed_preview))
    if unknown_preview:
        raise ValueError(
            f"unsupported [native_v3_preview] fields: {unknown_preview}"
        )
    model = _string(preview.get("model"), "native_v3_preview.model")
    effort = _string(preview.get("effort"), "native_v3_preview.effort")
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("native_v3_preview.effort is unsupported")
    timeout_seconds = _positive_number(
        preview.get("timeout_seconds"),
        "native_v3_preview.timeout_seconds",
    )
    heg_repo = _resolved_path(
        preview.get("heg_repo"),
        "native_v3_preview.heg_repo",
        source_path.parent,
    )
    return NativeV3PreviewConfig(
        V3_PREVIEW_CONFIG_SCHEMA_VERSION,
        V3_PREVIEW_SELECTOR,
        exp_id,
        workspace,
        model,
        effort,
        timeout_seconds,
        heg_repo,
        source_path,
        source_bytes,
    )


def _base_status(config: NativeV3PreviewConfig) -> dict[str, Any]:
    return {
        "schema_version": V3_PREVIEW_STATUS_SCHEMA_VERSION,
        "protocol": V3_PREVIEW_SELECTOR,
        "protocol_version": V3_PREVIEW_PROTOCOL_VERSION,
        "exp_id": config.exp_id,
        "workspace": str(config.experiment_root),
        "state": "not_created",
        "resumable": True,
        "terminal": False,
        "latest_infrastructure_stop_reason": None,
        "latest_scientific_stop_reason": None,
        "last_stop_reason": None,
        "last_error": None,
        "provider_turns": 0,
        "evaluation_count": 0,
        "valid_ast": False,
        "scientific_terminal_result": False,
        "usage": None,
        "artifacts": {},
    }


def _state_path(config: NativeV3PreviewConfig) -> Path:
    return config.experiment_root / _STATE_NAME


def _state_payload(
    config: NativeV3PreviewConfig,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(status),
        "config_sha256": config.source_sha256,
    }


def _public_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "config_sha256"}


def _load_workspace_state(config: NativeV3PreviewConfig) -> dict[str, Any]:
    state_path = _state_path(config)
    if not state_path.is_file():
        raise NativeV3PreviewWorkspaceError(
            "existing workspace is not a Native v3 preview workspace; "
            "use a fresh exp_id and never reinterpret a Native v2 workspace"
        )
    value = read_json(state_path)
    if not isinstance(value, Mapping):
        raise NativeV3PreviewWorkspaceError("Native v3 preview state is not an object")
    if (
        value.get("schema_version") != V3_PREVIEW_STATUS_SCHEMA_VERSION
        or value.get("protocol") != V3_PREVIEW_SELECTOR
        or value.get("protocol_version") != V3_PREVIEW_PROTOCOL_VERSION
    ):
        raise NativeV3PreviewWorkspaceError(
            "Native v3 preview workspace protocol does not match this runtime"
        )
    if value.get("config_sha256") != config.source_sha256:
        raise NativeV3PreviewWorkspaceError(
            "Native v3 preview configuration changed; create a fresh workspace"
        )
    stored_config = config.experiment_root / "experiment.toml"
    if (
        not stored_config.is_file()
        or hashlib.sha256(stored_config.read_bytes()).hexdigest()
        != config.source_sha256
    ):
        raise NativeV3PreviewWorkspaceError(
            "Native v3 preview workspace configuration identity mismatch"
        )
    return _public_state(value)


def _persist_state(
    config: NativeV3PreviewConfig,
    status: Mapping[str, Any],
) -> None:
    write_json(_state_path(config), _state_payload(config, status))


def _initialize_workspace(config: NativeV3PreviewConfig) -> dict[str, Any]:
    config.workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{config.exp_id}.native-v3-preview.",
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
    config: NativeV3PreviewConfig,
    error: Exception,
) -> dict[str, Any]:
    return {
        **_base_status(config),
        "state": "failed",
        "resumable": False,
        "last_stop_reason": "workspace_mismatch",
        "last_error": str(error),
    }


def v3_preview_status(
    config_path: str | Path = "experiment.toml",
) -> dict[str, Any]:
    """Read Native v3 preview status without provider or scorer contact."""

    config = load_v3_preview_config(config_path)
    if not config.experiment_root.exists():
        return _base_status(config)
    try:
        return _load_workspace_state(config)
    except (OSError, ValueError, NativeV3PreviewWorkspaceError) as error:
        return _failed_status(config, error)


def _default_provider(
    config: NativeV3PreviewConfig,
) -> LocalCodexAppServerProvider:
    return LocalCodexAppServerProvider(
        model=config.model,
        effort=config.effort,
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=config.timeout_seconds / 2,
        auth_json=Path.home() / ".codex" / "auth.json",
        persist_artifacts=False,
    )


def _default_backend(config: NativeV3PreviewConfig) -> GraphBackend:
    return HegBackend(config.heg_repo)


def _blocked_preflight(
    config: NativeV3PreviewConfig,
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
    config: NativeV3PreviewConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(report.get("status"))
    common = {
        **_base_status(config),
        "provider_turns": int(report.get("model_turns", 0)),
        "evaluation_count": int(report.get("graph_evaluations", 0)),
        "valid_ast": report.get("valid_ast") is True,
        "scientific_terminal_result": (
            report.get("scientific_terminal_result") is True
        ),
        "usage": report.get("usage"),
        "artifacts": {
            key: report[key]
            for key in (
                "provider_turn_directory",
                "evaluation_result",
                "provider_report",
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
            "latest_scientific_stop_reason": "smoke_panel_complete",
            "last_stop_reason": "smoke_panel_complete",
        }
    if status == "provider_error":
        authentication = report.get("error_classification") == "authentication"
        stop_reason = "preflight_failed" if authentication else "provider_failed"
        return {
            **common,
            "state": "blocked",
            "resumable": True,
            "latest_infrastructure_stop_reason": stop_reason,
            "last_stop_reason": stop_reason,
            "last_error": report.get("error"),
        }
    if status == "evaluation_failed":
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


def run_v3_preview(
    config_path: str | Path = "experiment.toml",
    *,
    provider_factory: ProviderFactory = _default_provider,
    backend_factory: BackendFactory = _default_backend,
    auth_available: AuthAvailable = Path.is_file,
) -> dict[str, Any]:
    """Run or resume the bounded one-slot Native v3 preview."""

    config = load_v3_preview_config(config_path)
    if config.experiment_root.exists():
        status = _load_workspace_state(config)
        if status.get("state") == "completed":
            return status
        if status.get("resumable") is not True:
            raise NativeV3PreviewWorkspaceError(
                "Native v3 preview workspace is not resumable"
            )
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
        provider = provider_factory(config)
        report = run_provider_evaluation_smoke(
            provider,
            config.experiment_root,
            backend_factory=lambda: backend_factory(config),
            config=SerialEpisodeConfig(
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                episode_id=f"{config.exp_id}/slot-00",
            ),
        )
    except Exception as error:
        status = {
            **_base_status(config),
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
                    "state": "blocked",
                    "resumable": True,
                    "latest_infrastructure_stop_reason": "provider_close_failed",
                    "last_stop_reason": "provider_close_failed",
                    "last_error": f"{type(error).__name__}: {error}",
                }
    _persist_state(config, status)
    return status


__all__ = [
    "V2_PROTOCOL",
    "V3_PREVIEW_CONFIG_SCHEMA_VERSION",
    "V3_PREVIEW_PROTOCOL_VERSION",
    "V3_PREVIEW_SELECTOR",
    "V3_PREVIEW_STATUS_SCHEMA_VERSION",
    "NativeV3PreviewConfig",
    "NativeV3PreviewWorkspaceError",
    "experiment_protocol",
    "load_v3_preview_config",
    "run_v3_preview",
    "v3_preview_status",
]
