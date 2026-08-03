"""Immutable experiment identity locks and continuation comparison."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tomllib
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


_NATIVE_PRESET_ASSETS: dict[str, dict[str, str]] = {
    "heg-ranker-evolution-v1": {
        "system_prompt": "prompts/native/system.md",
        "request_prompt": "prompts/native/request.md",
        "repair_prompt": "prompts/native/repair.md",
        "output_schema": "configs/native/generated-policy.schema.json",
        "context_schema": "configs/schemas/stage2b-context.schema.json",
        "proposal_schema": "configs/schemas/stage2b-proposal.schema.json",
        "semantic_glossary": "configs/stage3-field-semantics.v1.json",
        "baseline_rankers": "configs/native/baseline-rankers.json",
        **{
            f"mutation_brief_{index:02d}": f"configs/stage3-slots/slot-{index:02d}.json"
            for index in range(8)
        },
    }
}


def _preset_metadata(config: ExperimentConfig, project: Path) -> dict[str, Any]:
    """Resolve a preset from version-controlled, stage-independent assets."""

    asset_names = _NATIVE_PRESET_ASSETS.get(config.preset)
    if asset_names is None:
        return {"name": config.preset, "resolved": False, "assets": {}}
    assets: dict[str, Any] = {}
    for name, relative_path in asset_names.items():
        path = (project / relative_path).resolve()
        digest = sha256_file(path)
        if digest is None:
            raise LockError(f"native preset asset is missing: {path}")
        assets[name] = {"path": str(path), "sha256": digest}

    baseline_identities: dict[str, Any] = {}
    try:
        baseline_path = Path(assets["baseline_rankers"]["path"])
        baseline_value = json.loads(baseline_path.read_text(encoding="utf-8"))
        rankers = baseline_value.get("rankers") if isinstance(baseline_value, Mapping) else None
        if not isinstance(rankers, list):
            raise ValueError("rankers must be an array")
        source_sha256 = sha256_file(baseline_path)
        for ranker in rankers:
            if not isinstance(ranker, Mapping) or not isinstance(ranker.get("policy_id"), str):
                raise ValueError("each baseline ranker must have a policy_id")
            policy_id = str(ranker["policy_id"])
            identity = {
                "source_sha256": source_sha256,
                "definition_sha256": sha256_bytes(canonical_bytes(ranker)),
            }
            baseline_identities[policy_id.removeprefix("native_")] = identity
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LockError(f"cannot resolve native preset {config.preset!r}") from exc
    return {
        "name": config.preset,
        "resolved": True,
        "assets": assets,
        "baseline_identities": baseline_identities,
    }


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


def build_lock(
    config: ExperimentConfig,
    layout: ExperimentLayout,
    *,
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the authoritative lock without reading credentials."""

    project = _project_root()
    raw = config.raw
    # The sibling HEG checkout is a required, read-only dependency.  Its
    # current commit and dirty state are part of the lock; callers cannot
    # redirect it through experiment.toml.
    heg_path = project.parent / "heg"
    uv_lock = project / "uv.lock"
    preflight_doctor = preflight.get("doctor") if isinstance(preflight, Mapping) else None
    preset_metadata = _preset_metadata(config, project)
    doctor_sha: object = None
    if isinstance(preflight_doctor, Mapping):
        doctor_sha = sha256_bytes(canonical_bytes(preflight_doctor))
    if doctor_sha is not None and (
        not isinstance(doctor_sha, str)
        or len(doctor_sha) != 64
        or any(char not in "0123456789abcdef" for char in doctor_sha)
    ):
        raise LockError("app_server_doctor_sha256 must be a lowercase SHA-256")
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
            "concurrency": config.model.concurrency,
            "max_repairs": config.model.max_repairs,
        },
        "search": {
            "population_size": config.search.population_size,
            "max_generations": config.search.max_generations,
            "max_model_turns": config.search.max_model_turns,
            "selection": config.search.selection,
        },
        "evaluation": {
            "orders": list(config.evaluation.orders),
            "graph_seeds": list(config.evaluation.graph_seeds),
            "policy_seeds": list(config.evaluation.policy_seeds),
            "horizon": config.evaluation.horizon,
            "proposal_pool_size": config.evaluation.proposal_pool_size,
            "baselines": list(config.evaluation.baselines),
            "replay": config.evaluation.replay,
        },
        "resources": {
            "workers": config.resources.workers,
            "thread_count": config.resources.thread_count,
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
            "protocol": "codex-app-server",
            "profile": "default",
            "model": config.model.name,
            "effort": config.model.effort,
            "binary_version": binary_version,
            "auth_mode": "local-profile",
            "doctor_sha256": str(doctor_sha) if isinstance(doctor_sha, str) else None,
            "preflight": _redact(dict(preflight)) if isinstance(preflight, Mapping) else None,
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


def model_turn_limit_difference(
    lock: Mapping[str, Any], config: ExperimentConfig
) -> tuple[int, int] | None:
    """Return ``(locked, requested)`` for the one allowed budget increase.

    A larger model-turn cap is a continuation budget, not a new scientific
    identity, but only when it is the sole immutable difference.  The caller
    still has to prove that the existing workspace stopped at that cap.
    """

    differences = immutable_differences(lock, config)
    if len(differences) != 1 or differences[0][0] != "search.max_model_turns":
        return None
    locked, requested = differences[0][1:]
    if (
        isinstance(locked, bool)
        or not isinstance(locked, int)
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested <= locked
    ):
        return None
    return locked, requested


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
    "model_turn_limit_difference",
    "sha256_file",
    "verify_lock",
    "write_lock",
]
