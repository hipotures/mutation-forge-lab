"""Immutable experiment identity locks and continuation comparison."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping
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
    immutable = config.immutable_projection()
    source_hash = config.source_sha256
    prompt_identities = _path_identities(raw, config.source_dir)
    manifest_identities = {
        name: value
        for name, value in prompt_identities.items()
        if "manifest" in name or "seed" in name
    }
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
            "mutation_forge": _git_state(project),
            "heg": _git_state(heg_path),
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
        "prompt_identities": prompt_identities,
        "response_schema_identities": {
            key: value for key, value in prompt_identities.items() if "schema" in key
        },
        "context_schema_identities": {
            key: value for key, value in prompt_identities.items() if "context" in key
        },
        "sandbox_limits": _sandbox_limits(raw),
        "evaluation_manifest_identities": manifest_identities,
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
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "app_server": {
            "protocol": str(raw.get("app_server_protocol", "codex-app-server")),
            "profile": str(raw.get("app_server_profile", "default")),
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
    if expected_hash != config.immutable_config_sha256() or immutable_differences(lock, config):
        raise LockError(format_differences(immutable_differences(lock, config)))


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
