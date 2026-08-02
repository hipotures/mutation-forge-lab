"""Immutable experiment identity locks and continuation comparison."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .config import ExperimentConfig
from .layout import ExperimentLayout

LOCK_SCHEMA_VERSION = "mforge.experiment.lock.v1"
ARTIFACT_FORMAT_VERSION = "mforge.experiment.artifacts.v1"


class LockError(ValueError):
    """A lock is missing, malformed, or not compatible with the invocation."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _git_state(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    try:
        return {
            "repo": str(repo.resolve()),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--short")),
            "branch": git("branch", "--show-current"),
        }
    except (OSError, subprocess.SubprocessError):
        return {"repo": str(repo.resolve()), "commit": None, "dirty": None, "branch": None}


def _codex_version() -> str | None:
    try:
        result = subprocess.run(
            ["codex", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _codex_profile_identity() -> dict[str, Any]:
    """Describe the local non-secret Codex profile used by the adapter."""

    path = Path.home() / ".codex" / "config.toml"
    digest = sha256_file(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raw = {}
    values: dict[str, Any] = {}
    for key in (
        "model",
        "model_reasoning_effort",
        "sandbox_mode",
        "approval_policy",
        "service_tier",
    ):
        value = raw.get(key)
        if isinstance(value, str):
            values[key] = value
    return {
        "path": str(path) if digest is not None else None,
        "sha256": digest,
        "values": values,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _optional_path(raw: Mapping[str, Any], base: Path, *names: str) -> Path | None:
    for name in names:
        value = raw.get(name)
        if isinstance(value, str) and value:
            path = Path(value)
            return (base / path).resolve() if not path.is_absolute() else path.resolve()
    return None


def _path_identities(value: object, base: Path, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, str) and (
                str(key).endswith(("_path", "_file", "_schema"))
                or str(key) in {"system_prompt", "request_prompt", "repair_prompt", "output_schema"}
            ):
                path = Path(item)
                resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
                digest = sha256_file(resolved)
                result[name] = {"path": str(resolved), "sha256": digest}
            else:
                result.update(_path_identities(item, base, name))
    return result


def _sandbox_limits(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = raw.get("sandbox")
    if isinstance(value, Mapping):
        return {str(key): _redact(value_item, str(key)) for key, value_item in value.items()}
    value = raw.get("limits")
    if isinstance(value, Mapping):
        return {str(key): _redact(value_item, str(key)) for key, value_item in value.items()}
    return {}


def _preset_metadata(config: ExperimentConfig, project: Path) -> dict[str, Any]:
    """Resolve the scientific assets named by the experiment preset."""

    if config.preset != "heg-ranker-evolution-v1":
        return {"name": config.preset, "resolved": False, "assets": {}}
    configured = config.raw.get("legacy_stage4_config")
    stage4_path = (
        (config.source_dir / configured).resolve()
        if isinstance(configured, str) and configured and not Path(configured).is_absolute()
        else Path(configured).resolve()
        if isinstance(configured, str) and configured
        else project / "configs" / "stage4-search.toml"
    )
    try:
        from mutation_forge.sandbox.validation import validate_policy
        from mutation_forge.stage4.config import load_stage4_config

        stage4 = load_stage4_config(stage4_path)
        asset_paths = {
            name: getattr(stage4, name)
            for name in (
                "system_prompt_path",
                "request_prompt_path",
                "repair_prompt_path",
                "output_schema_path",
                "context_schema_path",
                "proposal_schema_path",
                "semantic_glossary_path",
                "seed_manifest_path",
                "manifest_path",
                "validation_manifest_path",
                "random_policy_path",
                "structural_policy_path",
            )
        }
        identities = {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
            }
            for name, path in asset_paths.items()
        }
        baseline_identities: dict[str, Any] = {}
        for name, path in (
            ("random", stage4.random_policy_path),
            ("structural", stage4.structural_policy_path),
        ):
            source = Path(path).read_text(encoding="utf-8")
            identity = validate_policy(source, stage4.sandbox).identity
            baseline_identities[name] = {
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "normalized_ast_sha256": identity.normalized_ast_sha256,
            }
        freeze_identity: dict[str, Any] = {}
        try:
            from mutation_forge.stage4.commands import campaign_root

            freeze_path = campaign_root(stage4) / "search-freeze.json"
            if freeze_path.is_file():
                freeze_value = json.loads(freeze_path.read_text(encoding="utf-8"))
                if isinstance(freeze_value, Mapping):
                    freeze_identity = {
                        "path": str(freeze_path.resolve()),
                        "sha256": sha256_file(freeze_path),
                        "doctor_sha256": freeze_value.get("doctor_sha256"),
                    }
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            freeze_identity = {}
        return {
            "name": config.preset,
            "resolved": True,
            "stage4_config": str(stage4_path.resolve()),
            "stage4_config_sha256": sha256_file(stage4_path),
            "resolved_config": stage4.resolved_dict(),
            "identity": asdict(stage4.identity),
            "assets": identities,
            "baseline_identities": baseline_identities,
            "search_freeze": freeze_identity,
            "selection": stage4.model.max_accepted_turns,
        }
    except Exception as exc:
        raise LockError(f"cannot resolve experiment preset {config.preset!r}: {exc}") from exc


def _redact(value: object, key: str = "") -> object:
    lowered = key.lower().replace("-", "_")
    if any(
        token in lowered
        for token in ("token", "password", "secret", "credential", "auth_json", "api_key")
    ):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(name): _redact(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    return value


def build_lock(config: ExperimentConfig, layout: ExperimentLayout) -> dict[str, Any]:
    """Build the authoritative lock without reading credentials."""

    project = _project_root()
    raw = config.raw
    repositories = raw.get("repositories")
    repositories_map = repositories if isinstance(repositories, Mapping) else {}
    heg_path = _optional_path(
        repositories_map,
        config.source_dir,
        "heg_repo",
        "backend_repo",
    )
    if heg_path is None:
        # Stage 1 keeps the sibling checkout read-only, but its exact identity
        # still belongs in experiment metadata even when the minimal config
        # omits an explicit repositories table.
        heg_path = project.parent / "heg"
    uv_lock = project / "uv.lock"
    preset_metadata = _preset_metadata(config, project)
    doctor_sha = raw.get("app_server_doctor_sha256")
    if doctor_sha is not None and (
        not isinstance(doctor_sha, str)
        or len(doctor_sha) != 64
        or any(char not in "0123456789abcdef" for char in doctor_sha)
    ):
        raise LockError("app_server_doctor_sha256 must be a lowercase SHA-256")
    freeze_doctor = preset_metadata.get("search_freeze", {}).get("doctor_sha256")
    if doctor_sha is None and isinstance(freeze_doctor, str):
        doctor_sha = freeze_doctor
    immutable = config.immutable_projection()
    source_hash = config.source_sha256
    prompt_identities = _path_identities(raw, config.source_dir)
    preset_assets = preset_metadata.get("assets", {})
    resolved_prompt_identities = {
        **prompt_identities,
        **{
            f"preset.{name}": value
            for name, value in preset_assets.items()
            if isinstance(value, Mapping)
        },
    }
    resolved_manifest_identities = {
        name: value
        for name, value in resolved_prompt_identities.items()
        if "manifest" in name or "seed" in name
    }
    profile_identity = _codex_profile_identity()
    binary_version = _codex_version()
    heg_state = _git_state(heg_path)
    if heg_state.get("commit") is None or heg_state.get("dirty") is None:
        raise LockError(f"required HEG/backend repository is unavailable: {heg_path}")
    project_state = _git_state(project)
    if project_state.get("commit") is None or project_state.get("dirty") is None:
        raise LockError(f"mutation-forge repository is unavailable: {project}")
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "lock_schema_version": LOCK_SCHEMA_VERSION,
        "exp_id": config.exp_id,
        "workspace": str(layout.workspace.resolve()),
        "experiment_root": str(layout.root.resolve()),
        "kind": config.kind,
        "preset": config.preset,
        "normalized_immutable_config": immutable,
        "immutable_config_sha256": config.immutable_config_sha256(),
        "source_config_sha256": source_hash,
        "provenance": {
            "mutation_forge": project_state,
            "heg": heg_state,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "uv_lock_sha256": sha256_file(uv_lock),
        },
        "python_version": platform.python_version(),
        "uv_lock_sha256": sha256_file(uv_lock),
        "model": {
            "provider": config.model.provider,
            "name": config.model.name,
            "effort": config.model.effort,
        },
        "prompt_identities": resolved_prompt_identities,
        "response_schema_identities": {
            key: value for key, value in resolved_prompt_identities.items() if "schema" in key
        },
        "sandbox_limits": _sandbox_limits(raw),
        "evaluation_manifest_identities": resolved_manifest_identities,
        "seed_lists": {
            "orders": list(config.evaluation.orders),
            "graph_seeds": list(config.evaluation.graph_seeds),
            "policy_seeds": list(config.evaluation.policy_seeds),
        },
        "generation": {
            "population_size": config.search.population_size,
            "max_generations": config.search.max_generations,
            "max_model_turns": config.search.max_model_turns,
        },
        "selection": config.search.selection,
        "preset_identity": preset_metadata,
        "proposal_schema_identities": {
            key: value
            for key, value in preset_metadata.get("assets", {}).items()
            if "proposal" in key
        },
        "context_schema_identities": {
            **{
                key: value
                for key, value in prompt_identities.items()
                if "context" in key
            },
            **{
                key: value
                for key, value in preset_metadata.get("assets", {}).items()
                if "context" in key
            },
        },
        "baseline_identities": preset_metadata.get("baseline_identities", {}),
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "app_server": {
            "protocol": str(raw.get("app_server_protocol", "codex-app-server")),
            "profile": str(raw.get("app_server_profile", "default")),
            "model": config.model.name,
            "effort": config.model.effort,
            "binary_version": binary_version,
            "auth_mode": "local-profile",
            "doctor_sha256": str(doctor_sha) if isinstance(doctor_sha, str) else None,
            "profile_identity": profile_identity,
            "strict_config": True,
            "resolved": (
                isinstance(doctor_sha, str)
                and binary_version is not None
                and profile_identity.get("sha256") is not None
            ),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_lock(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LockError(f"experiment lock is immutable: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def load_lock(path: str | Path) -> dict[str, Any]:
    lock_path = Path(path)
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockError(f"cannot read experiment lock: {lock_path}") from exc
    if not isinstance(value, dict):
        raise LockError("experiment lock must be a JSON object")
    if value.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise LockError("unsupported experiment lock schema")
    return cast(dict[str, Any], value)


def _diff_values(left: object, right: object, prefix: str = "") -> list[tuple[str, object, object]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right), key=str)
        result: list[tuple[str, object, object]] = []
        for key in keys:
            name = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_diff_values(left.get(key), right.get(key), name))
        return result
    if left != right:
        return [(prefix, left, right)]
    return []


def immutable_differences(
    lock: Mapping[str, Any], config: ExperimentConfig
) -> list[tuple[str, object, object]]:
    expected = lock.get("normalized_immutable_config")
    if not isinstance(expected, Mapping):
        raise LockError("experiment lock has no immutable configuration projection")
    return _diff_values(expected, config.immutable_projection())


def format_differences(differences: list[tuple[str, object, object]]) -> str:
    lines = ["Experiment configuration differs from the locked specification:"]
    for name, locked, current in differences:
        lines.extend(("", f"  {name}:", f"    locked:  {locked!r}", f"    current: {current!r}"))
    lines.extend(("", "Use a new exp_id to create a distinct experiment."))
    return "\n".join(lines)


def verify_lock(
    lock: Mapping[str, Any], config: ExperimentConfig, layout: ExperimentLayout
) -> None:
    if lock.get("exp_id") != config.exp_id:
        raise LockError("experiment lock exp_id does not match configuration")
    if Path(str(lock.get("experiment_root", ""))).resolve() != layout.root.resolve():
        raise LockError("experiment lock experiment_root does not match configuration")
    expected_hash = lock.get("immutable_config_sha256")
    differences = immutable_differences(lock, config)
    if expected_hash != config.immutable_config_sha256() or differences:
        raise LockError(format_differences(differences))
    project = _project_root()
    locked_preset = lock.get("preset_identity")
    if locked_preset is not None and locked_preset != _preset_metadata(config, project):
        raise LockError("resolved preset assets differ from the locked experiment identity")
    locked_app_server = lock.get("app_server")
    if isinstance(locked_app_server, Mapping):
        current_app_server = {
            "protocol": str(config.raw.get("app_server_protocol", "codex-app-server")),
            "profile": str(config.raw.get("app_server_profile", "default")),
            "model": config.model.name,
            "effort": config.model.effort,
            "binary_version": _codex_version(),
            "auth_mode": "local-profile",
        }
        for field in ("protocol", "profile", "model", "effort", "binary_version", "auth_mode"):
            if locked_app_server.get(field) != current_app_server[field]:
                raise LockError(f"App Server identity drifted for {field}")
        locked_profile = locked_app_server.get("profile_identity")
        if isinstance(locked_profile, Mapping) and locked_profile != _codex_profile_identity():
            raise LockError("local Codex profile identity drifted")
    locked_provenance = lock.get("provenance")
    if isinstance(locked_provenance, Mapping):
        current_project = _git_state(project)
        current_heg = locked_provenance.get("heg")
        heg_repo = current_heg.get("repo") if isinstance(current_heg, Mapping) else None
        current_heg_state = _git_state(Path(str(heg_repo))) if heg_repo else None
        for name, current in (("mutation_forge", current_project), ("heg", current_heg_state)):
            locked = locked_provenance.get(name)
            if (
                isinstance(locked, Mapping)
                and isinstance(current, Mapping)
                and (
                    locked.get("commit") != current.get("commit")
                    or locked.get("dirty") != current.get("dirty")
                )
            ):
                raise LockError(f"repository provenance drifted for {name}")


compare_lock = verify_lock
create_lock = build_lock
check_immutable_config = immutable_differences


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "LOCK_SCHEMA_VERSION",
    "LockError",
    "build_lock",
    "canonical_bytes",
    "check_immutable_config",
    "compare_lock",
    "create_lock",
    "format_differences",
    "immutable_differences",
    "load_lock",
    "sha256_file",
    "verify_lock",
    "write_lock",
]
