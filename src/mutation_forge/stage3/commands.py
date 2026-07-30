"""Command handlers for the deterministic Stage 3 runner.

The handlers return plain JSON-compatible mappings.  The CLI is deliberately a
thin renderer around this boundary so Rich and ``--json`` expose the same
result, while the JSON mode remains one compact, event-safe line.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage2d.manifest import read_cpu_topology

from .app_server import AppServerGenerationProvider, AppServerLimits, CodexAppServerAdapter
from .artifacts import GenerationArtifacts, replay_generation, safe_value
from .config import Stage3GenerationConfig, load_stage3_config
from .evaluation import (
    THREAD_ENVIRONMENT,
    canonical_projection,
    reduce_records,
    run_development_episode,
    write_records,
)
from .generation import GenerationConfig, OneShotGenerator
from .isolation import IsolatedCapsule, secure_capsule_parent
from .manifest import load_manifest, sha256
from .prompts import PromptBundle, load_prompt_bundle
from .replay import verify_replay as verify_replay_artifacts
from .revalidation import replay_saved_revalidation, revalidate_saved_generation
from .statistics import evaluate_gate, gate_report, summarize_development

SLOTS = tuple(f"slot-{index:02d}" for index in range(8))
FREEZE_SCHEMA = "stage3.freeze.v1"
SLOT_SCHEMA_VERSION = "stage3.slot.v1"
MAX_SLOT_BRIEF_BYTES = 4096


def canonical_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a command result into a JSON-safe deterministic mapping."""
    return cast(dict[str, Any], json.loads(json.dumps(value, sort_keys=True, default=str)))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, max_bytes: int) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if len(payload) > max_bytes:
        raise RuntimeError(f"artifact exceeds bound: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _prompt_bundle(config: Stage3GenerationConfig) -> PromptBundle:
    return load_prompt_bundle(
        context_schema=config.context_schema_path,
        proposal_schema=config.proposal_schema_path,
        semantics_glossary=config.semantic_glossary_path,
        output_schema=config.output_schema_path,
    )


def _load_slot_briefs(config: Stage3GenerationConfig) -> dict[str, dict[str, str]]:
    slots = sorted(config.slot_briefs_dir.glob("slot-*.json"))
    if [path.stem for path in slots] != list(SLOTS):
        raise RuntimeError("slot briefs must contain exactly slot-00 through slot-07")
    result: dict[str, dict[str, str]] = {}
    for path in slots:
        raw = path.read_bytes()
        if not raw or len(raw) > MAX_SLOT_BRIEF_BYTES:
            raise RuntimeError(f"slot brief size is invalid: {path.name}")
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or set(value)
            != {"schema_version", "slot_id", "brief", "generation_mode", "focus"}
            or value.get("schema_version") != SLOT_SCHEMA_VERSION
            or value.get("slot_id") != path.stem
            or value.get("generation_mode") != "new_strategy"
        ):
            raise RuntimeError(f"invalid frozen slot brief {path.stem}")
        for key, limit in (("brief", 2048), ("focus", 128)):
            field = value.get(key)
            if (
                not isinstance(field, str)
                or not field.strip()
                or len(field.encode("utf-8")) > limit
            ):
                raise RuntimeError(f"invalid {key} in frozen slot brief {path.stem}")
        result[path.stem] = cast(dict[str, str], value)
    return result


def _validate_output_schema(config: Stage3GenerationConfig) -> dict[str, Any]:
    raw = config.output_schema_path.read_bytes()
    if not raw or len(raw) > 64 * 1024:
        raise RuntimeError("generated-policy output schema size is invalid")
    value = json.loads(raw)
    required = {"schema_version", "source", "design_summary", "used_fields", "assumptions"}
    root_keys = {
        "$schema",
        "$id",
        "title",
        "description",
        "type",
        "additionalProperties",
        "required",
        "properties",
    }
    properties = value.get("properties") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or set(value) != root_keys
        or value.get("type") != "object"
        or value.get("additionalProperties") is not False
        or set(value.get("required", [])) != required
        or not isinstance(properties, dict)
        or set(properties) != required
        or properties["schema_version"].get("type") != "string"
        or properties["schema_version"].get("const")
        != "stage3.generated_policy.v1"
    ):
        raise RuntimeError("generated-policy output schema is not the frozen strict schema")
    for name, property_schema in properties.items():
        if (
            not isinstance(property_schema, dict)
            or property_schema.get("type") not in {"string", "array"}
        ):
            raise RuntimeError(f"generated-policy property {name} requires an explicit type")
        allowed_keys = (
            {"type", "const"} if name == "schema_version" else {"type", "items"}
            if property_schema["type"] == "array"
            else {"type"}
        )
        if set(property_schema) != allowed_keys:
            raise RuntimeError(
                f"generated-policy property {name} uses unsupported schema keywords"
            )
        if property_schema["type"] == "array":
            items = property_schema.get("items")
            if (
                not isinstance(items, dict)
                or items.get("type") != "string"
                or set(items) != {"type"}
            ):
                raise RuntimeError(
                    f"generated-policy array property {name} requires typed string items"
                )
    return cast(dict[str, Any], value)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, timeout=10
    )
    return completed.stdout.strip()


def _git_state(repo: Path) -> dict[str, Any]:
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--short")),
    }


def _freeze_path(config: Stage3GenerationConfig) -> Path:
    return config.run_root / "freeze.json"


def _load_freeze(config: Stage3GenerationConfig) -> dict[str, Any]:
    path = _freeze_path(config)
    if not path.is_file():
        raise RuntimeError(f"verified freeze artifact is required: {path}")
    value = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if value.get("schema_version") != FREEZE_SCHEMA or value.get("verified") is not True:
        raise RuntimeError("freeze artifact is not verified")
    if value.get("project_commit") != _git(config.project_repo, "rev-parse", "HEAD"):
        raise RuntimeError("freeze artifact project commit no longer matches")
    if value.get("heg_commit") != _git(config.heg_repo, "rev-parse", "HEAD"):
        raise RuntimeError("freeze artifact HEG commit no longer matches")
    if value.get("preregistration_tag") != config.preregistration_tag:
        raise RuntimeError("freeze artifact preregistration tag mismatch")
    expected_freeze_sha256 = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "freeze_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if value.get("freeze_sha256") != expected_freeze_sha256:
        raise RuntimeError("freeze artifact hash mismatch")
    if value.get("config_hash") != config.stable_hash():
        raise RuntimeError("freeze artifact config hash mismatch")
    if _git(config.project_repo, "cat-file", "-t", config.preregistration_tag) != "tag":
        raise RuntimeError("preregistration tag must remain an annotated tag")
    current = _verify_freeze_inputs(config)
    for key in (
        "hashes",
        "manifest_sha256",
        "prompt_bundle_sha256",
        "slot_brief_hashes",
        "slot_briefs",
        "baseline_identities",
    ):
        if value.get(key) != current.get(key):
            raise RuntimeError(f"freeze evidence drifted: {key}")
    return value


def _generation_run_id(config: Stage3GenerationConfig) -> str:
    """Choose one of two retained attempts without overwriting scientific evidence."""
    base = f"stage3-generation-{config.stable_hash()[:12]}"
    for attempt in (1, 2):
        run_id = f"{base}-attempt-{attempt:02d}"
        root = config.run_root / run_id
        if not root.exists():
            return run_id
        try:
            summary = GenerationArtifacts.read_summary(root)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"existing generation attempt is unreadable: {root}") from error
        usage = summary.get("usage_totals")
        usage_values = list(usage.values()) if isinstance(usage, Mapping) else []
        accepted = summary.get("accepted_model_turns")
        no_transport_evidence = (
            not any((root / "slots").rglob("*")) if (root / "slots").is_dir() else True
        )
        preflight_only = (
            summary.get("provider_calls") == 0
            and summary.get("initial_turn_count") == 0
            and no_transport_evidence
        )
        model_content = False
        for path in (root / "slots").glob("*/events.json"):
            try:
                events = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                model_content = True
                break
            if not isinstance(events, list):
                model_content = True
                break
            for event in events:
                if not isinstance(event, Mapping):
                    model_content = True
                    break
                for turn_name in ("initial", "repair"):
                    turn = event.get(turn_name)
                    if isinstance(turn, Mapping) and turn.get("content") is True:
                        model_content = True
                        break
        if any(path.stat().st_size for path in (root / "slots").glob("*/*.response.md")):
            model_content = True
        for path in (root / "slots").glob("*/*.events.jsonl"):
            if "agentMessage" in path.read_text(encoding="utf-8", errors="replace"):
                model_content = True
                break
        reconciled_zero = (
            accepted == 0
            and bool(usage_values)
            and all(value == 0 for value in usage_values)
            and not model_content
        )
        if summary.get("status") not in {"failed", "infrastructure_failure"} or not (
            preflight_only or reconciled_zero
        ):
            raise RuntimeError(
                "generation evidence already contains an accepted/charged turn; "
                "replacement attempts are forbidden"
            )
    raise RuntimeError("the two bounded infrastructure-only generation attempts are exhausted")


def _auth_status(*, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read authentication status without starting app-server or a turn."""
    executable = shutil.which("codex")
    if executable:
        try:
            result = subprocess.run(
                [executable, "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                env=dict(environment) if environment is not None else None,
            )
            text = (result.stdout + result.stderr).lower()
            authenticated = (
                result.returncode == 0
                and "not authenticated" not in text
                and "logged out" not in text
            )
            return {
                "authenticated": authenticated,
                "source": "codex login status",
                "returncode": result.returncode,
            }
        except (OSError, subprocess.SubprocessError) as error:
            return {
                "authenticated": False,
                "source": "codex login status",
                "error": type(error).__name__,
            }
    source_env = environment if environment is not None else os.environ
    authenticated = bool(source_env.get("CODEX_API_KEY") or source_env.get("OPENAI_API_KEY"))
    return {"authenticated": authenticated, "source": "environment"}


def appserver_doctor(
    config_path: str | Path,
    *,
    auth_json: str | Path | None = None,
) -> dict[str, Any]:
    """Audit the installed protocol and private capsule without starting a thread."""
    config = load_stage3_config(config_path)
    config.run_root.mkdir(parents=True, exist_ok=True)
    transport_dir = config.run_root / "appserver-doctor"
    checks: dict[str, Any] = {}
    executable = shutil.which("codex")
    checks["executable"] = {"ok": executable is not None, "path": executable}
    if executable:
        try:
            version = subprocess.run(
                [executable, "--version"], check=True, capture_output=True, text=True, timeout=10
            )
            checks["version"] = {"ok": True, "value": version.stdout.strip()}
        except (OSError, subprocess.SubprocessError) as error:
            checks["version"] = {"ok": False, "error": str(error)}
    else:
        checks["version"] = {"ok": False, "error": "codex executable not found"}
    checks["model"] = {"ok": False, "error": "model profile audit unavailable"}
    checks["profiles"] = {"ok": False, "error": "model profile audit unavailable"}
    capsule: IsolatedCapsule | None = None
    adapter: CodexAppServerAdapter | None = None
    try:
        capsule = IsolatedCapsule.create(
            secure_capsule_parent(),
            auth_json=auth_json,
            sandbox_mode=config.app_server.sandbox_mode,
            approval_policy=config.app_server.approval_policy,
        )
        checks["auth"] = _auth_status(environment=capsule.env)
        adapter = CodexAppServerAdapter(
            capsule=capsule,
            auth_checker=lambda _: False,
            artifact_dir=transport_dir,
            artifact_prefix="doctor",
            artifact_root=transport_dir,
            artifact_max_bytes=config.limits.artifact_bytes,
            sandbox_mode=config.app_server.sandbox_mode,
            approval_policy=config.app_server.approval_policy,
        )
        catalog = adapter.model_catalog()
        selected = next(
            (item for item in catalog if item.get("model") == config.model.name),
            None,
        )
        effort_values = (
            [
                item.get("reasoningEffort")
                for item in cast(
                    list[Mapping[str, Any]], selected.get("supportedReasoningEfforts", [])
                )
                if isinstance(item, Mapping)
            ]
            if selected is not None
            else []
        )
        profile_ok = selected is not None and config.model.effort in effort_values
        checks["model"] = {
            "ok": profile_ok,
            "name": config.model.name,
            "effort": config.model.effort,
            "available_models": sorted(
                str(item["model"]) for item in catalog if isinstance(item.get("model"), str)
            ),
        }
        checks["profiles"] = {
            "ok": profile_ok,
            "selected_supported_efforts": effort_values,
        }
        checks["protocol"] = {
            "ok": True,
            "value": "installed stdio JSON-RPC initialize/skills/list/model/list",
        }
    except Exception as error:
        safe_error = str(safe_value(str(error)))[:512]
        checks.setdefault("auth", {"authenticated": False, "source": "private capsule"})
        checks["model"] = {
            "ok": False,
            "error": type(error).__name__,
            "message": safe_error,
        }
        checks["profiles"] = {
            "ok": False,
            "error": type(error).__name__,
            "message": safe_error,
        }
        checks["protocol"] = {
            "ok": False,
            "error": type(error).__name__,
            "message": safe_error,
        }
    finally:
        if adapter is not None:
            adapter.close()
        if capsule is not None:
            capsule.cleanup()
    try:
        topology = read_cpu_topology()
        checks["cpu_topology"] = {"ok": len(topology) >= 16, "physical_cores": len(topology)}
    except Exception as error:
        checks["cpu_topology"] = {"ok": False, "error": str(error)}
    offline_required = ("executable", "version", "protocol", "model", "profiles", "cpu_topology")
    offline_ok = all(bool(checks[name].get("ok", False)) for name in offline_required)
    auth_ok = bool(checks["auth"].get("authenticated", False))
    result = canonical_result(
        {
            "schema_version": "stage3.appserver_doctor.v1",
            "status": "completed" if offline_ok and auth_ok else "inconclusive",
            "decision": "READY"
            if offline_ok and auth_ok
            else "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
            "offline_validation": "passed" if offline_ok else "failed",
            "live_generation_ready": offline_ok and auth_ok,
            "inference": False,
            "checks": checks,
            "transport_artifacts": {
                "directory": str(transport_dir),
                "rpc": str(transport_dir / "doctor.codex-rpc.jsonl"),
                "events": str(transport_dir / "doctor.events.jsonl"),
                "stdout": str(transport_dir / "doctor.stdout.jsonl"),
                "stderr": str(transport_dir / "doctor.stderr.txt"),
                "transcript_sha256": str(transport_dir / "doctor.transcript.sha256"),
            },
        }
    )
    # Keep a durable, credential-redacted audit even when auth/profile/
    # protocol checks fail.  The auth path itself is never persisted.
    try:
        artifact = config.run_root / "appserver-doctor.json"
        result["artifact"] = str(artifact)
        _atomic_json(artifact, result, max_bytes=config.limits.artifact_bytes)
    except OSError:
        # Reporting remains useful when the configured run root is read-only.
        pass
    return canonical_result(result)


def _verify_freeze_inputs(config: Stage3GenerationConfig) -> dict[str, Any]:
    project = _git_state(config.project_repo)
    heg = _git_state(config.heg_repo)
    tag_type: str | None = None
    try:
        tag_commit = _git(config.project_repo, "rev-list", "-n", "1", config.preregistration_tag)
        tag_ok = tag_commit == project["commit"]
        tag_type = _git(config.project_repo, "cat-file", "-t", config.preregistration_tag)
    except Exception:
        tag_commit, tag_ok = None, False
    if heg["commit"] != config.frozen_heg_commit or heg["dirty"]:
        raise RuntimeError("HEG must be at the frozen clean commit")
    if project["dirty"]:
        raise RuntimeError("project repository must be clean at freeze")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(config.project_repo),
            "merge-base",
            "--is-ancestor",
            config.frozen_project_commit,
            project["commit"],
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("project repository is not based on frozen commit")
    if tag_commit is None:
        raise RuntimeError(
            f"annotated preregistration tag is required: {config.preregistration_tag}"
        )
    if tag_type != "tag":
        raise RuntimeError(f"preregistration tag must be annotated (found {tag_type or 'missing'})")
    if not tag_ok:
        raise RuntimeError("preregistration tag does not point at the current commit")
    manifest = load_manifest(config.manifest_path, config)
    prompt_bundle = _prompt_bundle(config)
    if prompt_bundle.system != config.system_prompt_path.read_text(encoding="utf-8").rstrip("\n"):
        raise RuntimeError("checked-in system prompt differs from schema-derived prompt")
    if prompt_bundle.request != config.request_prompt_path.read_text(encoding="utf-8").rstrip("\n"):
        raise RuntimeError("checked-in request prompt differs from schema-derived prompt")
    if prompt_bundle.output_schema != config.output_schema_path.read_text(encoding="utf-8"):
        raise RuntimeError("checked-in output schema differs from prompt bundle")
    files = {
        "context_schema": config.context_schema_path,
        "proposal_schema": config.proposal_schema_path,
        "semantic_glossary": config.semantic_glossary_path,
        "system_prompt": config.system_prompt_path,
        "request_prompt": config.request_prompt_path,
        "output_schema": config.output_schema_path,
        "manifest": config.manifest_path,
        "config": config.source_path,
    }
    hashes = {name: _sha_file(path) for name, path in files.items()}
    for identity_name, file_name in (
        ("context_schema_sha256", "context_schema"),
        ("proposal_schema_sha256", "proposal_schema"),
        ("semantic_glossary_sha256", "semantic_glossary"),
        ("system_prompt_sha256", "system_prompt"),
        ("request_prompt_sha256", "request_prompt"),
        ("output_schema_sha256", "output_schema"),
        ("manifest_sha256", "manifest"),
    ):
        if hashes[file_name] != getattr(config.identity, identity_name):
            raise RuntimeError(f"{identity_name} hash drift")
    _validate_output_schema(config)
    slot_briefs = _load_slot_briefs(config)
    slots = [config.slot_briefs_dir / f"{slot}.json" for slot in SLOTS]
    slot_hashes = {path.stem: _sha_file(path) for path in slots}
    baseline_identities: dict[str, Any] = {}
    for name, path in (
        ("random", config.random_policy_path),
        ("structural", config.structural_policy_path),
    ):
        source = path.read_text(encoding="utf-8")
        validation = validate_policy(source, config.sandbox)
        if not validation.valid:
            raise RuntimeError(f"baseline {name} is invalid")
        baseline_identities[name] = validation.identity.as_dict()
    return {
        "project": project,
        "heg": heg,
        "tag": {
            "name": config.preregistration_tag,
            "commit": tag_commit,
            "type": tag_type,
        },
        "hashes": hashes,
        "manifest_sha256": manifest["manifest_sha256"],
        "prompt_bundle_sha256": prompt_bundle.stable_hash(),
        "slot_brief_hashes": slot_hashes,
        "slot_briefs": slot_briefs,
        "baseline_identities": baseline_identities,
    }


def freeze(config_path: str | Path) -> dict[str, Any]:
    config = load_stage3_config(config_path)
    evidence = _verify_freeze_inputs(config)
    # Freeze is intentionally model/auth independent.  The app-server doctor
    # is a separate preflight command; invoking it here would consult private
    # credentials and violate the no-live-model freeze boundary.
    audit = {
        "schema_version": "stage3.appserver_doctor.v1",
        "status": "not_run",
        "offline_validation": "not_run",
        "live_generation_ready": False,
        "inference": False,
        "reason": "freeze does not perform authentication or model inference",
    }
    config.run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "verified": True,
        "live_model_results_observed": True,
        "official_live_model_results_observed": True,
        "diagnostic_live_turns_attempted": 3,
        "diagnostic_live_turns_completed": 2,
        "diagnostic_evidence_excluded": True,
        "user_reviewed_diagnostic_restart": True,
        "prior_official_infrastructure_attempts": 5,
        "prior_official_turn_start_attempts": 36,
        "prior_official_model_content_observed": True,
        "prior_official_partial_model_delta_slots": 4,
        "prior_official_usage_observed": True,
        "prior_official_failure_codes": [
            "invalid_json_schema:missing_type",
            "invalid_json_schema:uniqueItems_not_permitted",
            "transport:aggregate_stdout_limit",
            "runtime:user_wide_nproc_limit",
            "runtime:turn_timeout",
            "transport:log_byte_limit",
            "transport:incoming_message_limit",
            "orchestration:static_ast_repair_classification",
        ],
        "user_authorized_schema_repair_restart": True,
        "user_authorized_infrastructure_repair_restart": True,
        "user_authorized_limit_increase_restart": True,
        "user_authorized_repair_classifier_restart": True,
        "inference": False,
        "auth_ready": False,
        "doctor": audit,
        "config_hash": config.stable_hash(),
        "project_commit": evidence["project"]["commit"],
        "heg_commit": evidence["heg"]["commit"],
        "preregistration_tag": config.preregistration_tag,
        **evidence,
    }
    payload["freeze_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in payload.items() if k != "freeze_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path = _freeze_path(config)
    _atomic_json(path, payload, max_bytes=config.limits.artifact_bytes)
    return canonical_result({"status": "completed", "freeze_artifact": str(path), **payload})


def generate(
    config_path: str | Path,
    *,
    provider: Any | None = None,
    concurrency: int = 8,
    auth_checker: Any | None = None,
    auth_json: str | Path | None = None,
) -> dict[str, Any]:
    if concurrency != 8:
        raise ValueError("Stage 3 generation concurrency is frozen at 8")
    config = load_stage3_config(config_path)
    freeze_value = _load_freeze(config)
    project_state = _git_state(config.project_repo)
    heg_state = _git_state(config.heg_repo)
    if project_state["dirty"]:
        raise RuntimeError("project repository is dirty; generation requires a clean freeze")
    if heg_state["dirty"]:
        raise RuntimeError("HEG repository is dirty; generation requires a clean freeze")
    tag_commit = _git(config.project_repo, "rev-list", "-n", "1", config.preregistration_tag)
    project_commit = _git(config.project_repo, "rev-parse", "HEAD")
    if tag_commit != project_commit:
        raise RuntimeError("preregistration tag must point at the frozen project commit")
    run_id = _generation_run_id(config)
    artifacts = GenerationArtifacts(
        config.run_root,
        run_id,
        max_total_bytes=config.limits.artifact_bytes,
    )
    artifacts.start(
        {
            "schema_version": "stage3.generation.v1",
            "run_id": run_id,
            "freeze_sha256": freeze_value.get("freeze_sha256"),
            "project_commit": project_commit,
            "heg_commit": config.frozen_heg_commit,
            "provider_calls": 0,
            "initial_turn_count": 0,
            "repair_turn_count": 0,
        }
    )
    prompt_bundle = _prompt_bundle(config)
    manifest = load_manifest(config.manifest_path, config)
    topology = read_cpu_topology()
    artifacts.write("freeze.json", freeze_value)
    artifacts.write(
        "environment.json",
        {
            "physical_cores": len(topology),
            "evaluation_worker_cores": [
                core.as_dict() for core in topology[: config.resources.max_evaluation_workers]
            ],
            "reserved_physical_cores": config.resources.reserved_physical_cores,
            "thread_environment": THREAD_ENVIRONMENT,
            "project_commit": project_commit,
            "heg_commit": config.frozen_heg_commit,
        },
    )
    artifacts.write(
        "generation_config.json",
        {
            "config": config.resolved_dict(),
            "config_sha256": config.stable_hash(),
            "model": config.model.name,
            "effort": config.model.effort,
            "protocol_version": "stage3.generation.v1",
            "smoke_calls": 10_000,
            "sandbox_limits": asdict(config.sandbox),
        },
    )
    artifacts.write(
        "prompt_bundle.json",
        {
            "version": prompt_bundle.version,
            "stable_sha256": prompt_bundle.stable_hash(),
            "system_sha256": config.identity.system_prompt_sha256,
            "request_sha256": config.identity.request_prompt_sha256,
            "output_schema_sha256": config.identity.output_schema_sha256,
            "context_schema_sha256": config.identity.context_schema_sha256,
            "proposal_schema_sha256": config.identity.proposal_schema_sha256,
            "semantic_glossary_sha256": config.identity.semantic_glossary_sha256,
        },
    )
    artifacts.write("development_manifest.json", manifest)
    doctor: Mapping[str, Any] | None = None
    if auth_checker is not None:
        auth = auth_checker()
    elif provider is None:
        # Production generation must prove the exact installed profile before
        # opening any generation turn.  Injected offline providers retain the
        # explicit auth_checker seam and never invoke the live doctor.
        doctor = appserver_doctor(config_path, auth_json=auth_json)
        doctor_checks = doctor.get("checks", {})
        auth_value = doctor_checks.get("auth", {}) if isinstance(doctor_checks, Mapping) else {}
        auth = dict(auth_value) if isinstance(auth_value, Mapping) else {}
        if doctor.get("status") != "completed":
            auth = {
                **auth,
                "authenticated": False,
                "source": "appserver-doctor",
                "failure": doctor.get("decision", "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"),
            }
    else:
        raise ValueError("an injected provider requires an explicit auth_checker")
    if not auth.get("authenticated", False):
        artifacts.write(
            "appserver.json",
            {
                "status": "authentication_unavailable",
                "authenticated": False,
                "source": auth.get("source"),
                "returncode": auth.get("returncode"),
                "initial_turns": 0,
                "repair_turns": 0,
                "model_calls": 0,
            },
        )
        failure = {
            "schema_version": "stage3.generation.v1",
            "run_id": run_id,
            "terminal_status": "infrastructure_failure",
            "failure": {
                "code": (
                    (
                        "appserver_protocol_failure"
                        if isinstance(doctor, Mapping)
                        and isinstance(doctor.get("checks"), Mapping)
                        and not bool(
                            cast(Mapping[str, Any], doctor["checks"])
                            .get("protocol", {})
                            .get("ok", False)
                        )
                        else "appserver_profile_unavailable"
                    )
                    if doctor is not None and doctor.get("decision") != "READY"
                    else "private_capsule_auth_unavailable"
                ),
                "message": (
                    "isolated Codex home is unauthenticated; no supported credential "
                    "reuse mechanism was used"
                ),
            },
            "provider_calls": 0,
            "initial_turn_count": 0,
            "repair_turn_count": 0,
            "model_calls": 0,
            "live_model_results_observed": False,
            "slots": [],
        }
        artifacts.write(
            "gate_decision.json",
            {
                "decision": "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
                "infrastructure": {"auth_failure": True},
                "provider_calls": 0,
            },
        )
        artifacts.finish("infrastructure_failure", failure)
        return canonical_result(
            {
                "status": "infrastructure_failure",
                "decision": "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
                "run_id": run_id,
                "run_path": str(artifacts.root),
                "provider_calls": 0,
                "summary": failure,
            }
        )
    if doctor is not None:
        artifacts.write("appserver-doctor.json", doctor)
    doctor_sha256 = sha256(doctor) if doctor is not None else None
    if provider is None:
        limits = AppServerLimits(
            max_request_bytes=config.limits.request_bytes,
            max_response_bytes=config.limits.response_bytes,
            max_message_bytes=config.limits.event_bytes,
            max_stdout_bytes=config.limits.stdout_bytes,
            max_stderr_bytes=config.limits.stderr_bytes,
            max_transcript_bytes=config.limits.transcript_bytes,
            turn_timeout=config.limits.turn_seconds,
            resource_cpu_seconds=int(config.limits.resource_cpu_seconds),
            resource_address_space_bytes=config.limits.resource_address_space_bytes,
            resource_file_bytes=config.limits.resource_file_bytes,
            resource_open_files=config.limits.resource_open_files,
            resource_processes=config.limits.resource_processes,
        )
        provider = AppServerGenerationProvider(
            auth_json=auth_json,
            limits=limits,
            artifact_max_bytes=config.limits.artifact_bytes,
            sandbox_mode=config.app_server.sandbox_mode,
            approval_policy=config.app_server.approval_policy,
        )
    output_schema = cast(Mapping[str, Any], _validate_output_schema(config))
    slot_briefs = _load_slot_briefs(config)
    slot_requests: dict[str, Mapping[str, Any]] = {}
    for slot in SLOTS:
        brief_value = slot_briefs[slot]
        slot_requests[slot] = {
            "slot": slot,
            "model": config.model.name,
            "effort": config.model.effort,
            "protocol_version": "stage3.generation.v1",
            "system_prompt": prompt_bundle.system,
            "prompt": prompt_bundle.render_slot_request(
                slot,
                brief_value["brief"],
                generation_mode=brief_value["generation_mode"],
                focus=brief_value["focus"],
            ),
            "output_schema": dict(output_schema),
            "appserver_doctor_sha256": doctor_sha256,
        }
    result = OneShotGenerator(
        cast(Any, provider),
        config=GenerationConfig(
            model=config.model.name,
            effort=config.model.effort,
            smoke_calls=10_000,
            max_repair_diagnostics=8,
            allow_infrastructure_retry=False,
        ),
        limits=config.sandbox,
        artifacts=artifacts,
        existing_sources=(
            config.random_policy_path.read_text(encoding="utf-8"),
            config.structural_policy_path.read_text(encoding="utf-8"),
        ),
        slot_requests=slot_requests,
    ).run(run_id=run_id)
    provider_calls = sum(1 + int(slot.repairs) for slot in result.slots)
    generation_decision = (
        "READY_FOR_EVALUATION"
        if result.status == "completed"
        else "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"
    )
    return canonical_result(
        {
            "status": result.status,
            "decision": generation_decision,
            "run_id": run_id,
            "run_path": str(artifacts.root),
            "provider_calls": provider_calls,
            "freeze_sha256": freeze_value.get("freeze_sha256"),
            "summary": result.summary,
        }
    )


def validate(run: str | Path) -> dict[str, Any]:
    root = Path(run)
    summary = replay_generation(root)
    generation_status = summary.get("status")
    valid = generation_status == "completed" and summary.get("replay_validated") is True
    return canonical_result(
        {
            "status": "completed" if valid else "inconclusive",
            "decision": "VALID" if valid else "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
            "run": str(root),
            "provider_calls": 0,
            "replay_validated": valid,
            "summary": summary,
        }
    )


def revalidate(config_path: str | Path, run: str | Path) -> dict[str, Any]:
    """Persist a provider-free revalidation of retained final responses."""
    config = load_stage3_config(config_path)
    summary = revalidate_saved_generation(config, run, persist=True)
    return canonical_result(summary)


def _sources_for_run(root: Path, config: Stage3GenerationConfig) -> dict[str, str]:
    policies = {
        "random": config.random_policy_path.read_text(encoding="utf-8"),
        "structural": config.structural_policy_path.read_text(encoding="utf-8"),
    }
    revalidation_path = root / "revalidation_summary.json"
    if revalidation_path.is_file():
        summary = cast(
            dict[str, Any], json.loads(revalidation_path.read_text(encoding="utf-8"))
        )
        source_prefix = root / "revalidation" / "slots"
    else:
        summary = GenerationArtifacts.read_summary(root)
        source_prefix = root / "slots"
    for slot in cast(list[Mapping[str, Any]], summary.get("slots", [])):
        if slot.get("status") != "accepted":
            continue
        name = str(slot.get("slot"))
        if name not in SLOTS:
            raise RuntimeError("generation summary contains an invalid slot identifier")
        path = source_prefix / name / "source.py"
        if path.is_file():
            policies[f"candidate-{name}"] = path.read_text(encoding="utf-8")
    return policies


def _run_evaluation_shard(
    payload: tuple[str, list[Mapping[str, Any]], Mapping[str, str], int, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one immutable shard on one physical CPU."""
    config_path, episodes, policies, cpu_id, reserved_cpu_ids = payload
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("Linux CPU affinity is required for Stage 3 evaluation")
    os.sched_setaffinity(0, {cpu_id})
    observed = sorted(os.sched_getaffinity(0))
    if observed != [cpu_id]:
        raise RuntimeError("evaluation worker affinity mismatch")
    config = load_stage3_config(config_path)
    records = [
        cast(dict[str, Any], run_development_episode(config, episode, policies))
        for episode in episodes
    ]
    return records, {
        "cpu_id": cpu_id,
        "observed_affinity": observed,
        "reserved_cpu_ids": list(reserved_cpu_ids),
        "thread_environment": dict(THREAD_ENVIRONMENT),
    }


def _run_evaluation_pass(
    config: Stage3GenerationConfig,
    episodes: Sequence[Mapping[str, Any]],
    policies: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topology = read_cpu_topology()
    required = config.resources.max_evaluation_workers + config.resources.reserved_physical_cores
    if len(topology) < required:
        raise RuntimeError("cannot preserve the frozen eight-core reserve")
    worker_cores = topology[: config.resources.max_evaluation_workers]
    reserved_cores = topology[
        config.resources.max_evaluation_workers : config.resources.max_evaluation_workers
        + config.resources.reserved_physical_cores
    ]
    reserved_cpu_ids = tuple(core.cpu_id for core in reserved_cores)
    worker_cpu_ids = {core.cpu_id for core in worker_cores}
    if len(reserved_cpu_ids) != config.resources.reserved_physical_cores or worker_cpu_ids & set(
        reserved_cpu_ids
    ):
        raise RuntimeError("worker and reserved physical cores must be disjoint")
    shards = [
        list(episodes[index :: config.resources.max_evaluation_workers])
        for index in range(config.resources.max_evaluation_workers)
    ]
    payloads = [
        (
            str(config.source_path),
            shard,
            policies,
            worker_cores[index].cpu_id,
            reserved_cpu_ids,
        )
        for index, shard in enumerate(shards)
    ]
    with ProcessPoolExecutor(max_workers=config.resources.max_evaluation_workers) as pool:
        batches = list(pool.map(_run_evaluation_shard, payloads))
    return (
        [record for batch, _ in batches for record in batch],
        [metadata for _, metadata in batches],
    )


def _write_evaluation_record_shards(
    config: Stage3GenerationConfig,
    root: Path,
    label: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: list[list[Mapping[str, Any]]] = [
        [] for _ in range(config.experiment.shard_count)
    ]
    for record in records:
        shard_index = config.episode_shard(
            int(record["order"]),
            int(record["graph_seed"]),
            int(record["policy_seed"]),
        )
        grouped[shard_index].append(record)
    expected = config.experiment.episodes_per_shard
    if any(len(shard) != expected for shard in grouped):
        raise ValueError("evaluation record shard cardinality mismatch")
    artifacts: list[dict[str, Any]] = []
    for index, shard_records in enumerate(grouped):
        ordered = sorted(shard_records, key=lambda record: str(record["episode_id"]))
        artifacts.append(
            cast(
                dict[str, Any],
                write_records(
                    root / f"evaluation-{label}-shard-{index:02d}.jsonl.gz",
                    ordered,
                    maximum_bytes=config.limits.artifact_bytes,
                ),
            )
        )
    manifest: dict[str, Any] = {
        "schema_version": "stage3.evaluation_shards.v1",
        "label": label,
        "shard_count": len(artifacts),
        "episodes_per_shard": expected,
        "record_count": sum(int(artifact["record_count"]) for artifact in artifacts),
        "uncompressed_bytes": sum(
            int(artifact["uncompressed_bytes"]) for artifact in artifacts
        ),
        "canonical_records_sha256": sha256(
            [canonical_projection(record) for record in records]
        ),
        "shards": artifacts,
    }
    _atomic_json(
        root / f"evaluation-{label}-shards.json",
        manifest,
        max_bytes=config.limits.artifact_bytes,
    )
    return manifest


def _evaluate(
    config_path: str | Path,
    run: str | Path,
    *,
    workers: int = 8,
    episodes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if workers != 8:
        raise ValueError("Stage 3 evaluation workers are frozen at 8")
    config = load_stage3_config(config_path)
    root = Path(run)
    _load_freeze(config)
    project_state = _git_state(config.project_repo)
    heg_state = _git_state(config.heg_repo)
    if project_state["dirty"] or heg_state["dirty"]:
        raise RuntimeError("evaluation requires clean frozen project and HEG repositories")
    if heg_state["commit"] != config.frozen_heg_commit:
        raise RuntimeError("HEG commit drifted after the generation freeze")
    generation_summary = (
        replay_saved_revalidation(config, root)
        if (root / "revalidation_summary.json").is_file()
        else replay_generation(root)
    )
    if (
        generation_summary.get("status") != "completed"
        or generation_summary.get("replay_validated") is not True
    ):
        result = canonical_result(
            {
                "status": "inconclusive",
                "decision": "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
                "run": str(root),
                "provider_calls": 0,
                "failure": "generation campaign is not complete",
            }
        )
        with suppress(OSError):
            (root / "evaluation_summary.json").write_text(
                json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        return result
    policies = _sources_for_run(root, config)
    manifest = load_manifest(config.manifest_path, config)
    frozen_episodes = cast(list[Mapping[str, Any]], manifest["episodes"])
    if episodes is not None and canonical_result({"episodes": list(episodes)}) != canonical_result(
        {"episodes": frozen_episodes}
    ):
        raise RuntimeError("evaluation episode override differs from the frozen manifest")
    selected = frozen_episodes
    records, primary_affinity = _run_evaluation_pass(config, selected, policies)
    replay_records, replay_affinity = _run_evaluation_pass(config, selected, policies)
    expected_ids = {str(e["episode_id"]) for e in selected}
    identities = {
        name: validate_policy(source, config.sandbox).identity.as_dict()
        for name, source in policies.items()
    }
    source_hashes = {
        name: cast(str, identity["source_sha256"]) for name, identity in identities.items()
    }
    ast_hashes = {
        name: cast(str, identity["normalized_ast_sha256"]) for name, identity in identities.items()
    }
    reduced = reduce_records(
        cast(list[Mapping[str, Any]], records),
        expected_ids,
        set(policies),
        expected_source_hashes=source_hashes,
        expected_ast_hashes=ast_hashes,
    )
    replay_reduced = reduce_records(
        cast(list[Mapping[str, Any]], replay_records),
        expected_ids,
        set(policies),
        expected_source_hashes=source_hashes,
        expected_ast_hashes=ast_hashes,
    )
    primary_hash = sha256([canonical_projection(record) for record in reduced])
    replay_hash = sha256([canonical_projection(record) for record in replay_reduced])
    replay_exact = primary_hash == replay_hash
    summary = summarize_development(
        cast(list[Mapping[str, Any]], reduced),
        policies,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
    )

    def integer(record: Mapping[str, Any], key: str) -> int:
        value = record.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"episode {key} must be an integer")
        return value

    policy_failures = sum(integer(record, "policy_failures") for record in reduced)
    invalid_records = sum(integer(record, "invalid_graphs") for record in reduced)
    accounting_parity = all(
        integer(record, "initial_score_calls") == 1
        and integer(record, "oracle_score_calls") == 0
        and integer(record, "selected_score_calls") == integer(record, "horizon") * len(policies)
        and integer(record, "evaluation_count") == integer(record, "selected_score_calls")
        for record in reduced
    )
    distinct_ast = len(set(ast_hashes.values())) == len(ast_hashes)
    affinities_valid = (
        len({metadata["cpu_id"] for metadata in primary_affinity}) == 8
        and len({metadata["cpu_id"] for metadata in replay_affinity}) == 8
        and all(len(set(metadata["reserved_cpu_ids"])) == 8 for metadata in primary_affinity)
        and all(len(set(metadata["reserved_cpu_ids"])) == 8 for metadata in replay_affinity)
        and all(
            metadata["cpu_id"] not in metadata["reserved_cpu_ids"]
            for metadata in (*primary_affinity, *replay_affinity)
        )
        and len(
            {
                tuple(metadata["reserved_cpu_ids"])
                for metadata in (*primary_affinity, *replay_affinity)
            }
        )
        == 1
        and all(
            metadata["thread_environment"] == THREAD_ENVIRONMENT
            for metadata in (*primary_affinity, *replay_affinity)
        )
    )
    frozen = cast(
        Mapping[str, Any],
        json.loads((root / "freeze.json").read_text(encoding="utf-8")),
    )
    doctor_path = root / "appserver-doctor.json"
    frozen_doctor: Mapping[str, Any] | None = None
    if doctor_path.is_file():
        doctor_value = json.loads(doctor_path.read_text(encoding="utf-8"))
        if isinstance(doctor_value, Mapping):
            frozen_doctor = doctor_value
    protocol_safety = (
        isinstance(frozen_doctor, Mapping)
        and frozen_doctor.get("offline_validation") == "passed"
        and frozen_doctor.get("inference") is False
        and frozen_doctor.get("live_generation_ready") is True
    )
    campaign_authority = (
        generation_summary.get("initial_turn_count") == 8
        and int(generation_summary.get("total_live_turns", 17)) <= 16
        and int(generation_summary.get("initial_max_active", 9)) <= 8
        and generation_summary.get("status") == "completed"
    )
    tag_commit = _git(config.project_repo, "rev-list", "-n", "1", config.preregistration_tag)
    source_generation_tag = str(
        generation_summary.get("source_generation_tag", config.preregistration_tag)
    )
    source_generation_tag_commit = _git(
        config.project_repo, "rev-list", "-n", "1", source_generation_tag
    )
    repository_and_heg = (
        source_generation_tag_commit == frozen.get("project_commit")
        and tag_commit == _git(config.project_repo, "rev-parse", "HEAD")
        and _git(config.heg_repo, "rev-parse", "HEAD") == config.frozen_heg_commit
        and not _git(config.heg_repo, "status", "--short")
    )
    summary.update(
        {
            "scientific_valid": len(reduced) == len(selected),
            "dependency_provenance": (
                frozen.get("heg_commit") == config.frozen_heg_commit
                and frozen.get("project_commit") == source_generation_tag_commit
                and generation_summary.get("replay_validated") is True
            ),
            "protocol_safety": protocol_safety,
            "campaign_authority": campaign_authority,
            "exact_usage": bool(generation_summary.get("exact_usage_complete", False)),
            "baseline_ast_distinct": distinct_ast,
            "primary_replay_exact": replay_exact,
            "accounting_parity": accounting_parity,
            "provenance_valid": True,
            "reduction_valid": len(reduced) == len(selected),
            "resource_valid": affinities_valid,
            "invalid_records": invalid_records,
            "worker_failures": policy_failures,
            "selected_only_equal_bounded_parity": (
                accounting_parity and affinities_valid and len(reduced) == len(selected)
            ),
            "repository_and_heg_validation": repository_and_heg,
            "primary_records_sha256": primary_hash,
            "replay_records_sha256": replay_hash,
            "policy_identities": identities,
            "primary_affinity": primary_affinity,
            "replay_affinity": replay_affinity,
        }
    )
    gate = gate_report(summary)
    output = {
        "status": "completed",
        "run": str(root),
        "workers": 8,
        "concurrency": 8,
        "provider_calls": 0,
        "episode_count": len(reduced),
        "replay_episode_count": len(replay_reduced),
        "primary_records_sha256": primary_hash,
        "replay_records_sha256": replay_hash,
        "summary": summary,
        "gate": gate,
        "decision": evaluate_gate(summary),
    }
    primary_shards = _write_evaluation_record_shards(
        config,
        root,
        "primary",
        cast(list[Mapping[str, Any]], reduced),
    )
    replay_shards = _write_evaluation_record_shards(
        config,
        root,
        "replay",
        cast(list[Mapping[str, Any]], replay_reduced),
    )
    output["primary_shards"] = primary_shards
    output["replay_shards"] = replay_shards
    _atomic_json(root / "gate.json", gate, max_bytes=config.limits.artifact_bytes)
    _atomic_json(
        root / "evaluation_summary.json",
        output,
        max_bytes=config.limits.artifact_bytes,
    )
    return canonical_result(output)


def evaluate(
    config_path: str | Path,
    run: str | Path,
    *,
    workers: int = 8,
    episodes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run both deterministic passes and terminally classify infrastructure failures."""
    if workers != 8:
        raise ValueError("Stage 3 evaluation workers are frozen at 8")
    root = Path(run)
    try:
        return _evaluate(config_path, run, workers=workers, episodes=episodes)
    except Exception as error:
        result = canonical_result(
            {
                "status": "inconclusive",
                "decision": "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
                "run": str(root),
                "provider_calls": 0,
                "failure": {
                    "code": type(error).__name__,
                    "message": str(error)[:512],
                },
            }
        )
        if root.is_dir():
            payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
            temporary = root / ".evaluation_summary.json.tmp"
            with suppress(OSError):
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, root / "evaluation_summary.json")
                gate = {
                    "decision": "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
                    "checks": {},
                    "failure": result["failure"],
                }
                gate_temporary = root / ".gate.json.tmp"
                gate_temporary.write_text(
                    json.dumps(gate, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                os.replace(gate_temporary, root / "gate.json")
        return result


def verify_replay(primary: str | Path, replay: str | Path) -> dict[str, Any]:
    return canonical_result(verify_replay_artifacts(primary, replay))


run_appserver_doctor = appserver_doctor
freeze_stage3 = freeze
generate_stage3 = generate
validate_stage3 = validate
revalidate_stage3 = revalidate
evaluate_stage3 = evaluate
verify_stage3_replay = verify_replay

__all__ = [
    "appserver_doctor",
    "freeze",
    "generate",
    "validate",
    "revalidate",
    "evaluate",
    "verify_replay",
    "canonical_result",
    "run_appserver_doctor",
    "freeze_stage3",
    "generate_stage3",
    "validate_stage3",
    "revalidate_stage3",
    "evaluate_stage3",
    "verify_stage3_replay",
]
