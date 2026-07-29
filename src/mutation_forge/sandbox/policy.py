from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from mutation_forge.artifacts import canonical_json_hash, git_state
from mutation_forge.models import JsonValue
from mutation_forge.sandbox.config import PolicyEvaluationConfig
from mutation_forge.sandbox.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    BEHAVIOR_SCHEMA_VERSION,
    ProbeContext,
    ProbeProposal,
    SandboxLimits,
)
from mutation_forge.sandbox.errors import (
    ProtocolError,
    WorkerCrashError,
    WorkerTimeoutError,
)
from mutation_forge.sandbox.validation import ValidationResult, validate_policy
from mutation_forge.sandbox.worker import PolicyWorker

FIXED_PROBE_BUNDLES: tuple[
    tuple[ProbeContext, tuple[ProbeProposal, ...]], ...
] = (
    (
        {
            "probe_id": "weighted",
            "step": 0,
            "budget_remaining": 8,
            "features": {"temperature": 1.0, "offset": 0},
        },
        (
            {
                "proposal_id": "p-alpha",
                "kind": "probe",
                "features": {"weight": 2.0, "penalty": 1, "values": [1, 2]},
            },
            {
                "proposal_id": "p-beta",
                "kind": "probe",
                "features": {"weight": 4.0, "penalty": 3, "values": [3, 1]},
            },
            {
                "proposal_id": "p-gamma",
                "kind": "probe",
                "features": {"weight": 2.0, "penalty": 1, "values": [2, 2]},
            },
        ),
    ),
    (
        {
            "probe_id": "conditional",
            "step": 3,
            "budget_remaining": 2,
            "features": {"temperature": 0.5, "offset": -1},
        },
        (
            {
                "proposal_id": "p-delta",
                "kind": "probe",
                "features": {"weight": -1.0, "penalty": 0, "values": [5, -2]},
            },
            {
                "proposal_id": "p-epsilon",
                "kind": "probe",
                "features": {"weight": 0.0, "penalty": 0, "values": []},
            },
        ),
    ),
)


def _failure_flags(kind: str | None = None) -> dict[str, bool]:
    return {
        "exception": kind == "exception",
        "timeout": kind == "timeout",
        "crash": kind == "crash",
        "protocol": kind == "protocol",
    }


def _failure_kind(error: BaseException) -> str:
    if isinstance(error, WorkerTimeoutError):
        return "timeout"
    if isinstance(error, WorkerCrashError):
        return "crash"
    if isinstance(error, ProtocolError):
        return "protocol"
    return "exception"


def _rank_key(item: dict[str, JsonValue]) -> tuple[int | float, str]:
    priority = item["priority"]
    proposal_id = item["proposal_id"]
    if (
        isinstance(priority, bool)
        or not isinstance(priority, int | float)
        or not isinstance(proposal_id, str)
    ):
        raise TypeError("finite probe result has an invalid priority or proposal ID")
    return -priority, proposal_id


def _signature_payload(
    source: str,
    limits: SandboxLimits,
    validation: ValidationResult,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    bundles: list[JsonValue] = []
    all_flags = _failure_flags()
    terminal_status = "completed"
    worker = PolicyWorker(source, limits)
    try:
        stop = False
        for ctx, proposals in FIXED_PROBE_BUNDLES:
            results: list[dict[str, JsonValue]] = []
            for proposal in proposals:
                try:
                    result = worker.call(ctx, proposal)
                    flags = _failure_flags(
                        None if result.status == "ok" else "exception"
                    )
                    priority = result.priority
                    finite = result.status == "ok"
                    error = result.error
                except BaseException as exc:
                    kind = _failure_kind(exc)
                    flags = _failure_flags(kind)
                    priority = None
                    finite = False
                    error = {
                        "code": f"worker_{kind}",
                        "message": str(exc)[:1024],
                    }
                for key, value in flags.items():
                    all_flags[key] = all_flags[key] or value
                results.append(
                    {
                        "proposal_id": proposal["proposal_id"],
                        "priority": priority,
                        "finite": finite,
                        **flags,
                        "error": error,
                    }
                )
                if any(flags.values()):
                    terminal_status = "failed"
                    stop = True
                    break
            ranked = sorted(
                (result for result in results if result["finite"] is True),
                key=_rank_key,
            )
            bundles.append(
                {
                    "probe_id": ctx["probe_id"],
                    "priorities": cast(list[JsonValue], results),
                    "rank_order": [
                        cast(str, item["proposal_id"]) for item in ranked
                    ],
                    "selected_proposal_id": (
                        cast(str, ranked[0]["proposal_id"]) if ranked else None
                    ),
                }
            )
            if stop:
                break
        signature_base: dict[str, JsonValue] = {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "identity": validation.identity.as_dict(),
            "probes": bundles,
            "flags": cast(dict[str, JsonValue], all_flags),
            "terminal_status": terminal_status,
        }
        signature: dict[str, JsonValue] = {
            **signature_base,
            "signature_sha256": canonical_json_hash(signature_base),
        }
        telemetry = worker.telemetry()
        return signature, telemetry
    finally:
        worker.close()


def behavior_signature(
    source: str,
    limits: SandboxLimits | None = None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    applied_limits = limits or SandboxLimits()
    validation = validate_policy(source, applied_limits)
    if not validation.valid:
        raise ValueError(json.dumps(validation.as_dict(), sort_keys=True))
    return _signature_payload(source, applied_limits, validation)


def probe_policy(
    source: str,
    limits: SandboxLimits | None = None,
) -> dict[str, JsonValue]:
    applied_limits = limits or SandboxLimits()
    validation = validate_policy(source, applied_limits)
    result: dict[str, JsonValue] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "invalid",
        "validation": validation.as_dict(),
        "identity": validation.identity.as_dict(),
        "limits": applied_limits.as_dict(),
        "behavior_signature": None,
        "worker_telemetry": None,
    }
    if not validation.valid:
        return result
    try:
        signature, telemetry = _signature_payload(
            source,
            applied_limits,
            validation,
        )
        result["behavior_signature"] = signature
        result["worker_telemetry"] = telemetry
        result["status"] = cast(str, signature["terminal_status"])
    except BaseException as error:
        result["status"] = "failed"
        result["worker_telemetry"] = {
            "calls": 0,
            "failures": 1,
            "startup_error": {
                "error_type": type(error).__name__,
                "message": str(error)[:1024],
            },
        }
    return result


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _new_run_path(config: PolicyEvaluationConfig, source_hash: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = config.run_root / f"stage2a-{timestamp}-{source_hash[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "artifacts" / "programs").mkdir(parents=True)
    return path


def _git_is_ancestor(repo: Path, ancestor: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, "HEAD"],
        check=False,
        capture_output=True,
        timeout=10,
    )
    return completed.returncode == 0


def evaluate_policy(
    policy_path: str | Path,
    config: PolicyEvaluationConfig,
) -> dict[str, JsonValue]:
    source_path = Path(policy_path).resolve()
    source = source_path.read_text()
    validation = validate_policy(source, config.limits)
    run_path = _new_run_path(config, validation.identity.source_sha256)
    program_path = run_path / "artifacts" / "programs" / "policy.py"
    program_path.write_text(source)
    shutil.copy2(config.source_path, run_path / "policy_config.toml")
    started = time.monotonic()
    project_state = git_state(config.project_repo)
    heg_state = git_state(config.heg_repo)
    heg_pin_verified = (
        heg_state["commit"] == config.frozen_heg_commit
        and heg_state["dirty"] is False
    )
    project_base_verified = _git_is_ancestor(
        config.project_repo,
        config.frozen_project_commit,
    )
    execution_gate_verified = heg_pin_verified and project_base_verified
    probe: dict[str, JsonValue]
    if execution_gate_verified:
        probe = probe_policy(source, config.limits)
    else:
        probe = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "failed",
            "validation": validation.as_dict(),
            "identity": validation.identity.as_dict(),
            "limits": config.limits.as_dict(),
            "behavior_signature": None,
            "worker_telemetry": None,
            "gate_error": {
                "code": "frozen_entry_point_mismatch",
                "message": "project base and clean pinned HEG must match Stage 2A",
            },
        }
    provenance: dict[str, JsonValue] = {
        "frozen_entry_point": {
            "mutation_forge": config.frozen_project_commit,
            "heg": config.frozen_heg_commit,
        },
        "observed": {
            "mutation_forge": project_state,
            "heg": heg_state,
        },
        "heg_read_only": True,
        "heg_pin_verified": heg_pin_verified,
        "project_base_verified": project_base_verified,
        "execution_gate_verified": execution_gate_verified,
        "model_calls": 0,
        "network_calls": 0,
    }
    result: dict[str, JsonValue] = {
        **probe,
        "run_path": str(run_path),
        "source_path": str(source_path),
        "program_path": str(program_path),
        "provenance": provenance,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(run_path / "validation.json", probe["validation"])
    _write_json(run_path / "identity.json", probe["identity"])
    _write_json(run_path / "limits.json", probe["limits"])
    _write_json(
        run_path / "behavior_signature.json",
        probe["behavior_signature"],
    )
    _write_json(run_path / "worker_telemetry.json", probe["worker_telemetry"])
    _write_json(run_path / "provenance.json", provenance)
    _write_json(run_path / "result.json", result)
    _write_json(
        run_path / "terminal_status.json",
        {"status": result["status"], "schema_version": ARTIFACT_SCHEMA_VERSION},
    )
    return result


def replay_policy(
    run_path: str | Path,
    limits: SandboxLimits | None = None,
) -> dict[str, JsonValue]:
    path = Path(run_path).resolve()
    source = (path / "artifacts" / "programs" / "policy.py").read_text()
    expected_identity = json.loads((path / "identity.json").read_text())
    expected_signature = json.loads((path / "behavior_signature.json").read_text())
    applied_limits = limits
    if applied_limits is None:
        persisted_limits = json.loads((path / "limits.json").read_text())
        if not isinstance(persisted_limits, dict):
            raise ValueError("persisted limits must be a JSON object")
        applied_limits = SandboxLimits(**persisted_limits)
    replayed = probe_policy(source, applied_limits)
    identity_match = replayed["identity"] == expected_identity
    signature_match = replayed["behavior_signature"] == expected_signature
    result: dict[str, JsonValue] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "completed" if identity_match and signature_match else "failed",
        "source_sha256_match": (
            cast(dict[str, JsonValue], replayed["identity"])["source_sha256"]
            == cast(dict[str, JsonValue], expected_identity)["source_sha256"]
        ),
        "normalized_ast_sha256_match": (
            cast(dict[str, JsonValue], replayed["identity"])[
                "normalized_ast_sha256"
            ]
            == cast(dict[str, JsonValue], expected_identity)[
                "normalized_ast_sha256"
            ]
        ),
        "identity_match": identity_match,
        "behavior_signature_match": signature_match,
        "replayed": replayed,
    }
    _write_json(path / "replay.json", result)
    return result
