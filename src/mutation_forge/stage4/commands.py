"""Command orchestration for the frozen Stage 4 campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage2d.manifest import read_cpu_topology
from mutation_forge.stage3.isolation import IsolatedCapsule, secure_capsule_parent

from .app_server import (
    Stage4AppServerAdapter,
    Stage4AppServerProvider,
    _available_artifact_prefix,
)
from .archive import ProgramArchive, ProgramRecord, deterministic_program_id
from .artifacts import (
    build_evidence_manifest,
    canonical_bytes,
    project_real_shape,
    verify_evidence_manifest,
    write_raw_slot_record,
)
from .checkpoint import CheckpointStore
from .config import Stage4SearchConfig, load_stage4_config
from .contracts import PROVENANCE, SeedRecord, load_seed_manifest
from .evaluation import (
    evaluate_policy_roster_manifest,
    evaluate_program_manifest,
    verify_candidate_pass,
)
from .generation import (
    Candidate,
    GenerationConfig,
    GenerationCoordinator,
    GenerationResult,
    SlotResult,
    cached_pre_turn_auth_retry_allowed,
)
from .replay import verify_replay as verify_replay_pair
from .selection import select_parents
from .statistics import (
    gate_report,
    hierarchical_bootstrap,
    select_champion,
    summarize_development,
)

SEARCH_FREEZE_SCHEMA = "stage4.search.freeze.v1"
VALIDATION_FREEZE_SCHEMA = "stage4.validation.freeze.v1"
RUN_SCHEMA = "stage4.campaign.v1"
AUTH_RECOVERY_SCHEMA = "stage4.auth_recovery.v1"
AUTH_RECOVERY_ROOT = "recovery/pre-auth-v1"
POST_LIVE_AMENDMENT_SCHEMA = "stage4.search.technical_amendment.v1"
POST_LIVE_AMENDMENT_PATH = "search-technical-amendment-v3.json"
SEARCH_AMENDMENT_TAG = "stage4-search-amendment-v3"
SEARCH_AMENDMENT_CATEGORY = "authenticated_same_request_resume"
SEARCH_AMENDMENT_CATEGORIES = {
    "stage4-search-amendment-v1": "evaluation_worker_process_isolation",
    "stage4-search-amendment-v2": "replay_metrics_timing_projection",
    SEARCH_AMENDMENT_TAG: SEARCH_AMENDMENT_CATEGORY,
}
STAGE3_CANONICAL_SHA256 = PROVENANCE["stage3_canonical_sha256"]
STAGE3_EVIDENCE_MANIFEST_SHA256 = PROVENANCE["evidence_manifest_sha256"]
STAGE3_ARCHIVE_SHA256 = PROVENANCE["archive_sha256"]

Observer = Callable[[Mapping[str, Any]], None]


def canonical_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic JSON-compatible command result."""

    return cast(
        dict[str, Any],
        json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False)),
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_value(value: object) -> str:
    return _sha_bytes(canonical_bytes(value))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_bytes(value) + b"\n")


def _atomic_source(path: Path, source: str) -> None:
    _atomic_write(path, source.encode("utf-8"))


def _available_evidence_path(path: Path) -> Path:
    """Select an additive evidence path without replacing an earlier report."""
    if not path.exists():
        return path
    for attempt in range(1, 65):
        candidate = path.with_name(
            f"{path.stem}.retry-{attempt:02d}{path.suffix}"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"evidence retry namespace is exhausted: {path.name}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _git_state(repo: Path) -> dict[str, Any]:
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--short")),
        "branch": _git(repo, "branch", "--show-current"),
    }


def _tag_state(repo: Path, name: str) -> dict[str, Any]:
    try:
        return {
            "name": name,
            "object": _git(repo, "rev-parse", name),
            "type": _git(repo, "cat-file", "-t", name),
            "commit": _git(repo, "rev-list", "-n", "1", name),
        }
    except (OSError, subprocess.SubprocessError):
        return {"name": name, "object": None, "type": None, "commit": None}


def campaign_root(config: Stage4SearchConfig) -> Path:
    return config.run_root / f"campaign-{config.stable_hash()[:12]}"


def _archive_root(run: Path) -> Path:
    return run / "archive"


def _source_path(run: Path, program_id: str) -> Path:
    return _archive_root(run) / "sources" / f"{program_id}.py"


def _relative_to_run(run: Path, path: Path) -> str:
    return path.resolve().relative_to(run.resolve()).as_posix()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"manifest has no episodes: {path}")
    if len(episodes) != int(value.get("episode_count", -1)):
        raise ValueError(f"manifest episode count mismatch: {path}")
    return value


def _thread_environment() -> dict[str, str]:
    return {
        name: "1"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    }


def _auth_status(*, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    executable = shutil.which("codex")
    if executable is None:
        return {"ok": False, "authenticated": False, "executable": None}
    try:
        version = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        login = subprocess.run(
            [executable, "login", "status"],
            capture_output=True,
            text=True,
            timeout=15,
            env=dict(environment) if environment is not None else None,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "ok": False,
            "authenticated": False,
            "executable": executable,
            "error": type(error).__name__,
        }
    status_text = f"{login.stdout}\n{login.stderr}".lower()
    authenticated = (
        login.returncode == 0
        and "logged in" in status_text
        and "not authenticated" not in status_text
        and "logged out" not in status_text
    )
    return {
        "ok": authenticated,
        "authenticated": authenticated,
        "executable": executable,
        "version": version,
        "source": (
            "private capsule codex login status"
            if environment is not None
            else "host codex login status"
        ),
    }


def _appserver_profile_status(
    run: Path,
    *,
    auth_json: str | Path | None,
) -> dict[str, Any]:
    """Audit private-capsule auth and the frozen profile without starting a turn."""

    capsule: IsolatedCapsule | None = None
    adapter: Stage4AppServerAdapter | None = None
    try:
        capsule = IsolatedCapsule.create(
            secure_capsule_parent(),
            auth_json=auth_json,
            sandbox_mode="danger-full-access",
            approval_policy="never",
        )
        auth = _auth_status(environment=capsule.env)
        adapter = Stage4AppServerAdapter(
            capsule=capsule,
            auth_checker=lambda _: False,
            artifact_dir=run / "offline-appserver-doctor",
            artifact_prefix=_available_artifact_prefix(
                run / "offline-appserver-doctor",
                "authenticated-catalog" if auth_json is not None else "catalog",
            ),
            artifact_root=run,
        )
        catalog = adapter.model_catalog()
        selected = next(
            (item for item in catalog if item.get("model") == "gpt-5.6-luna"),
            None,
        )
        supported = (
            [
                item.get("reasoningEffort")
                for item in cast(
                    Sequence[Mapping[str, Any]],
                    selected.get("supportedReasoningEfforts", ()),
                )
                if isinstance(item, Mapping)
            ]
            if selected is not None
            else []
        )
        ok = selected is not None and "high" in supported
        return {
            "ok": ok,
            "auth": auth,
            "model": "gpt-5.6-luna",
            "effort": "high",
            "supported_efforts": supported,
            "protocol": "initialize/model/list",
            "inference": False,
        }
    except Exception as error:
        return {
            "ok": False,
            "auth": {
                "ok": False,
                "authenticated": False,
                "source": "private capsule codex login status",
            },
            "model": "gpt-5.6-luna",
            "effort": "high",
            "error_type": type(error).__name__,
            "error": str(error)[:512],
            "inference": False,
        }
    finally:
        if adapter is not None:
            adapter.close()
        if capsule is not None:
            capsule.cleanup()


def _stage3_checks(config: Stage4SearchConfig) -> dict[str, Any]:
    run = config.stage3_source_run.parent
    evidence_manifest = run / "EVIDENCE_MANIFEST.json"
    evaluation_summary = run / "evaluation_summary.json"
    archive = (
        config.project_repo.parent
        / "mutation-forge-evidence"
        / "mutation-forge-stage3-v13-evidence.tar.zst"
    )
    summary = _read_json(evaluation_summary) if evaluation_summary.is_file() else {}
    primary = summary.get("primary_records_sha256")
    replay = summary.get("replay_records_sha256")
    checks = {
        "expanded_run": config.stage3_source_run.is_dir(),
        "evidence_manifest": (
            evidence_manifest.is_file()
            and _sha_file(evidence_manifest) == STAGE3_EVIDENCE_MANIFEST_SHA256
        ),
        "canonical_primary_replay": (
            primary == STAGE3_CANONICAL_SHA256 and replay == STAGE3_CANONICAL_SHA256
        ),
        "decision": summary.get("decision") == "GO_TO_STAGE_4",
        "verified_archive": (
            archive.is_file() and _sha_file(archive) == STAGE3_ARCHIVE_SHA256
        ),
    }
    return {"ok": all(checks.values()), "checks": checks, "archive": str(archive)}


def doctor(
    config_path: str | Path,
    *,
    auth_json: str | Path | None = None,
    check_auth: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run all non-inference Stage 4 prerequisite and dry-run checks."""

    config = load_stage4_config(config_path)
    project = _git_state(config.project_repo)
    heg = _git_state(config.heg_repo)
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
        timeout=30,
    ).returncode == 0
    topology = read_cpu_topology()
    search_manifest = _load_manifest(config.manifest_path)
    validation_manifest = _load_manifest(config.validation_manifest_path)
    search_ids = {str(item["episode_id"]) for item in search_manifest["episodes"]}
    validation_ids = {str(item["episode_id"]) for item in validation_manifest["episodes"]}
    seeds = load_seed_manifest(config.seed_manifest_path)
    run = campaign_root(config)
    projection = project_real_shape(run / "offline-dry-run")
    projection_result = cast(Mapping[str, Any], projection["projection"])
    stage3 = _stage3_checks(config)
    profile = (
        _appserver_profile_status(run, auth_json=auth_json)
        if check_auth
        else {
            "ok": True,
            "auth": {"ok": True, "authenticated": None, "skipped": True},
            "skipped": True,
            "inference": False,
        }
    )
    profile_auth = profile.get("auth")
    auth = (
        dict(profile_auth)
        if isinstance(profile_auth, Mapping)
        else {"ok": False, "authenticated": False}
    )
    profile = {key: value for key, value in profile.items() if key != "auth"}
    checks: dict[str, Any] = {
        "config": True,
        "project_base": ancestor,
        "project_clean": not project["dirty"],
        "heg_pin": heg["commit"] == config.frozen_heg_commit and not heg["dirty"],
        "stage3": bool(stage3["ok"]),
        "seed_capsule": len(seeds) == 8,
        "manifest_matrix": (
            len(search_ids) == 128
            and len(validation_ids) == 128
            and search_ids.isdisjoint(validation_ids)
        ),
        "physical_cores": len(topology) >= 16,
        "worker_budget": (
            config.limits.max_evaluation_workers <= 8
            and config.limits.reserved_physical_cores >= 8
        ),
        "artifact_headroom": min(
            float(projection_result["shard_headroom"]),
            float(projection_result["campaign_uncompressed_headroom"]),
            float(projection_result["campaign_compressed_headroom"]),
        )
        >= 2.0,
        "authentication": bool(auth["ok"]),
        "model_profile": bool(profile["ok"]),
    }
    doctor_path = (
        _available_evidence_path(
            run
            / (
                "authenticated-appserver-doctor.json"
                if auth_json is not None
                else "offline-doctor.json"
            )
        )
        if write
        else None
    )
    result = canonical_result(
        {
            "schema_version": "stage4.doctor.v1",
            "status": "completed" if all(checks.values()) else "inconclusive",
            "decision": "READY" if all(checks.values()) else "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
            "inference": False,
            "live_model_results_observed": False,
            "checks": checks,
            "project": project,
            "heg": heg,
            "stage3": stage3,
            "auth": auth,
            "model_profile": profile,
            "physical_cores": len(topology),
            "evaluation_cores": [asdict(core) for core in topology[:8]],
            "reserved_cores": [asdict(core) for core in topology[8:16]],
            "thread_environment": _thread_environment(),
            "projection": projection,
            "run": str(run),
            "artifact_path": (
                _relative_to_run(run, doctor_path)
                if doctor_path is not None
                else None
            ),
        }
    )
    if doctor_path is not None:
        _atomic_json(doctor_path, result)
    return result


def _freeze_files(config: Stage4SearchConfig) -> dict[str, str]:
    stage3_run = config.stage3_source_run.parent
    stage3_archive = (
        config.project_repo.parent
        / "mutation-forge-evidence"
        / "mutation-forge-stage3-v13-evidence.tar.zst"
    )
    paths = {
        "config": config.source_path,
        "stage2b_config": config.stage2b_config_path,
        "random_policy": config.random_policy_path,
        "structural_policy": config.structural_policy_path,
        "search_manifest": config.manifest_path,
        "validation_manifest": config.validation_manifest_path,
        "seed_manifest": config.seed_manifest_path,
        "system_prompt": config.system_prompt_path,
        "request_prompt": config.request_prompt_path,
        "repair_prompt": config.repair_prompt_path,
        "output_schema": config.output_schema_path,
        "context_schema": config.context_schema_path,
        "proposal_schema": config.proposal_schema_path,
        "semantic_glossary": config.semantic_glossary_path,
        "stage3_evidence_manifest": stage3_run / "EVIDENCE_MANIFEST.json",
        "stage3_evaluation_summary": stage3_run / "evaluation_summary.json",
        "stage3_archive": stage3_archive,
    }
    for brief in sorted(config.briefs_dir.glob("slot-*.md")):
        paths[f"brief_{brief.stem}"] = brief
    for seed in sorted((config.project_repo / "fixtures" / "stage4-seeds").glob("slot-*.py")):
        paths[f"seed_{seed.stem}"] = seed
    return {name: _sha_file(path) for name, path in paths.items()}


def _freeze_digest(value: Mapping[str, Any]) -> str:
    return _sha_value({key: item for key, item in value.items() if key != "freeze_sha256"})


def _recovery_digest(value: Mapping[str, Any]) -> str:
    return _sha_value(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )


def _technical_amendment_digest(value: Mapping[str, Any]) -> str:
    return _sha_value(
        {key: item for key, item in value.items() if key != "amendment_sha256"}
    )


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).returncode
        == 0
    )


def _live_model_result_evidence(run: Path) -> dict[str, Any]:
    archive = ProgramArchive(_archive_root(run))
    generated = [
        record.program_id
        for record in archive.records()
        if record.generation > 0
        or record.effective_request_id is not None
        or record.turn_id is not None
        or record.app_server_turn_id is not None
    ]
    paths = {
        "search_summary": (run / "search-summary.json").is_file(),
        "generation_checkpoint": (run / "generation-checkpoint.json").is_file(),
        "checkpoint_records": any((run / "checkpoints").glob("checkpoint-*.json")),
        "app_server_records": any(path.is_file() for path in (run / "appserver").glob("**/*")),
        "raw_generation_records": any(path.is_file() for path in (run / "raw").glob("**/*")),
    }
    observed = bool(generated) or any(paths.values())
    return {
        "observed": observed,
        "generated_program_ids": generated,
        "artifact_presence": paths,
    }


def _authentication_recovery_evidence(run: Path) -> dict[str, Any]:
    checkpoint_path = run / "generation-checkpoint.json"
    summary_path = run / "search-summary.json"
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise RuntimeError("authentication recovery evidence is incomplete")
    checkpoint = _read_json(checkpoint_path)
    summary = _read_json(summary_path)
    slots = checkpoint.get("slots")
    if not isinstance(slots, Mapping) or len(slots) != 32:
        raise RuntimeError("authentication recovery requires exactly 32 checkpoint slots")
    slot_values = [
        dict(value) for value in slots.values() if isinstance(value, Mapping)
    ]
    generation_slot_counts = [
        sum(int(value.get("generation", -1)) == generation for value in slot_values)
        for generation in range(4)
    ]
    if (
        len(slot_values) != 32
        or not all(cached_pre_turn_auth_retry_allowed(value) for value in slot_values)
        or generation_slot_counts != [8, 8, 8, 8]
        or any(value.get("repair") is not None for value in slot_values)
    ):
        raise RuntimeError("checkpoint contains non-recoverable Stage 4 turn evidence")
    request_keys = {
        str(cast(Mapping[str, Any], value.get("request", {})).get("idempotency_key", ""))
        for value in slot_values
    }
    if "" in request_keys or request_keys != {str(key) for key in slots}:
        raise RuntimeError("authentication recovery request identities drifted")
    archive = ProgramArchive(_archive_root(run)).reindex()
    generated = [record for record in archive.records if record.generation > 0]
    if (
        not archive.ok
        or len(generated) != 32
        or not all(record.tombstone for record in generated)
        or {record.effective_request_id for record in generated} != request_keys
        or any(record.usage for record in generated)
        or any(record.metadata.get("accepted_turn_count") != 0 for record in generated)
    ):
        raise RuntimeError("canonical archive contains non-recoverable Stage 4 output")
    if (
        summary.get("decision") != "NO_GO"
        or summary.get("decision_reason") != "minimum_unique_offspring_not_met"
        or summary.get("initial_turns") != 32
        or summary.get("repair_turns") != 0
        or summary.get("accepted_live_turns") != 0
        or summary.get("new_unique_valid_offspring") != 0
        or summary.get("exact_usage") is not False
        or summary.get("unauthorized_tool_approval") is not False
    ):
        raise RuntimeError("search summary is not the retained authentication failure")
    return {
        "schema_version": AUTH_RECOVERY_SCHEMA,
        "verified": True,
        "inference": False,
        "replacement_requests_authorized": True,
        "reason": "private_capsule_authentication_missing",
        "slot_count": 32,
        "generation_slot_counts": generation_slot_counts,
        "accepted_turns": 0,
        "repair_turns": 0,
        "contentful_turns": 0,
        "charged_turns": 0,
        "usage_tokens": 0,
        "live_stage4_model_output_observed": False,
        "retained_summary_live_results_claim": summary.get(
            "live_stage4_model_results_observed"
        ),
        "request_identities_sha256": _sha_value(sorted(request_keys)),
        "checkpoint_sha256": _sha_file(checkpoint_path),
        "search_summary_sha256": _sha_file(summary_path),
        "archive_sha256": archive.archive_hash,
    }


def _preserve_recovery_file(
    run: Path,
    source: Path,
    recovery: Path,
) -> dict[str, Any]:
    relative = source.resolve().relative_to(run.resolve())
    destination = recovery / relative
    payload = source.read_bytes()
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"recovery artifact collision: {relative.as_posix()}")
    else:
        _atomic_write(destination, payload)
    return {
        "source": relative.as_posix(),
        "retained": destination.relative_to(run).as_posix(),
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _retained_recovery_entries_valid(
    run: Path,
    entries: object,
) -> bool:
    if not isinstance(entries, list) or not entries:
        return False
    for item in entries:
        if not isinstance(item, Mapping):
            return False
        retained = item.get("retained")
        sha256 = item.get("sha256")
        if not isinstance(retained, str) or not isinstance(sha256, str):
            return False
        path = run / retained
        try:
            path.resolve().relative_to(run.resolve())
        except ValueError:
            return False
        if not path.is_file() or _sha_file(path) != sha256:
            return False
    return True


def _prepare_authentication_recovery(run: Path) -> dict[str, Any]:
    recovery = run / AUTH_RECOVERY_ROOT
    manifest_path = recovery / "RECOVERY_MANIFEST.json"
    if manifest_path.is_file():
        completed_manifest = _read_json(manifest_path)
        if (
            completed_manifest.get("schema_version") != AUTH_RECOVERY_SCHEMA
            or completed_manifest.get("verified") is not True
            or completed_manifest.get("phase") != "completed"
            or completed_manifest.get("manifest_sha256")
            != _recovery_digest(completed_manifest)
            or not _retained_recovery_entries_valid(
                run,
                completed_manifest.get("retained_files"),
            )
            or not _retained_recovery_entries_valid(
                run,
                completed_manifest.get("moved_tombstones"),
            )
        ):
            raise RuntimeError("retained authentication recovery manifest is invalid")
        return completed_manifest

    intent_path = recovery / "RECOVERY_INTENT.json"
    if intent_path.is_file():
        intent = _read_json(intent_path)
        if (
            intent.get("schema_version") != AUTH_RECOVERY_SCHEMA
            or intent.get("verified") is not True
            or intent.get("phase") != "prepared"
            or intent.get("manifest_sha256") != _recovery_digest(intent)
        ):
            raise RuntimeError("authentication recovery intent is invalid")
        evidence = {
            key: value
            for key, value in intent.items()
            if key
            not in {
                "phase",
                "retained_files",
                "moved_tombstones",
                "manifest_sha256",
            }
        }
        retained = cast(list[dict[str, Any]], intent.get("retained_files", []))
        moved = cast(list[dict[str, Any]], intent.get("moved_tombstones", []))
    else:
        evidence = _authentication_recovery_evidence(run)
        retained = []
        exact_files = (
            run / "generation-checkpoint.json",
            run / "search-summary.json",
            run / "EVIDENCE_MANIFEST.json",
        )
        for path in exact_files:
            if not path.is_file():
                raise RuntimeError(
                    f"required authentication recovery artifact is missing: {path.name}"
                )
            retained.append(_preserve_recovery_file(run, path, recovery))
        for directory in ("appserver", "raw", "selection"):
            root = run / directory
            if root.is_dir():
                for path in sorted(
                    candidate for candidate in root.rglob("*") if candidate.is_file()
                ):
                    retained.append(_preserve_recovery_file(run, path, recovery))

        archive = ProgramArchive(_archive_root(run)).reindex()
        moved = []
        for record in archive.records:
            if record.generation == 0:
                continue
            source = _archive_root(run) / "programs" / f"{record.program_id}.json"
            retained_tombstone = _preserve_recovery_file(run, source, recovery)
            moved.append({"program_id": record.program_id, **retained_tombstone})
        intent = {
            **evidence,
            "phase": "prepared",
            "retained_files": retained,
            "moved_tombstones": moved,
        }
        intent["manifest_sha256"] = _recovery_digest(intent)
        _atomic_json(intent_path, intent)

    if len(moved) != 32:
        raise RuntimeError("authentication recovery intent must retain 32 tombstones")
    if not _retained_recovery_entries_valid(run, retained):
        raise RuntimeError("retained authentication recovery evidence is invalid")
    if not _retained_recovery_entries_valid(run, moved):
        raise RuntimeError("retained authentication tombstones are invalid")
    for item in moved:
        source = run / str(item["source"])
        destination = run / str(item["retained"])
        if source.is_file():
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError("authentication tombstone recovery collision")
            source.unlink()
    canonical = ProgramArchive(_archive_root(run)).reindex()
    if (
        not canonical.ok
        or len(canonical.records) != 8
        or any(record.generation != 0 for record in canonical.records)
    ):
        raise RuntimeError("authentication recovery did not restore the seed-only archive")
    active_checkpoint = _read_json(run / "generation-checkpoint.json")
    recovery_marker = active_checkpoint.get("authentication_recovery")
    if recovery_marker is None:
        if _sha_file(run / "generation-checkpoint.json") != evidence["checkpoint_sha256"]:
            raise RuntimeError("active authentication checkpoint drifted before recovery")
        active_checkpoint["slots"] = {}
        active_checkpoint["callbacks"] = {}
        active_checkpoint.pop("summary", None)
        active_checkpoint["authentication_recovery"] = {
            "schema_version": AUTH_RECOVERY_SCHEMA,
            "retained_checkpoint_sha256": evidence["checkpoint_sha256"],
            "replacement_requests_authorized": True,
        }
        _atomic_json(run / "generation-checkpoint.json", active_checkpoint)
    elif not (
        isinstance(recovery_marker, Mapping)
        and recovery_marker.get("schema_version") == AUTH_RECOVERY_SCHEMA
        and recovery_marker.get("retained_checkpoint_sha256")
        == evidence["checkpoint_sha256"]
        and recovery_marker.get("replacement_requests_authorized") is True
        and active_checkpoint.get("slots") == {}
        and active_checkpoint.get("callbacks") == {}
    ):
        raise RuntimeError("active authentication recovery checkpoint is invalid")
    manifest: dict[str, Any] = {
        **evidence,
        "phase": "completed",
        "canonical_seed_archive_sha256": canonical.archive_hash,
        "retained_files": retained,
        "moved_tombstones": moved,
    }
    manifest["manifest_sha256"] = _recovery_digest(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def _amendment_states(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = value.get("amendment_tags")
    if isinstance(history, list) and all(isinstance(item, Mapping) for item in history):
        return [dict(item) for item in history]
    amendment = value.get("amendment_tag")
    return [dict(amendment)] if isinstance(amendment, Mapping) else []


def _amendment_backup_name(tag_name: str) -> str:
    if tag_name == "stage4-search-amendment-v1":
        return "search-freeze-pre-amendment.json"
    version = tag_name.rsplit("-", 1)[-1]
    return f"search-freeze-pre-amendment-{version}.json"


def _amendment_tag_chain_valid(
    repo: Path,
    value: Mapping[str, Any],
    descendant: str,
) -> bool:
    states = _amendment_states(value)
    amendment = value.get("amendment_tag")
    if not states:
        return amendment is None
    if not isinstance(amendment, Mapping) or states[-1] != amendment:
        return False
    names = [str(state.get("name", "")) for state in states]
    if names != list(SEARCH_AMENDMENT_CATEGORIES)[: len(names)]:
        return False
    return all(
        state.get("type") == "tag"
        and _tag_state(repo, str(state.get("name", ""))) == state
        and _is_ancestor(repo, str(state.get("commit", "")), descendant)
        for state in states
    )


def _previous_search_freeze(
    config: Stage4SearchConfig,
    run: Path,
    search_tag: Mapping[str, Any],
) -> dict[str, Any]:
    source = run / "search-freeze.json"
    if not source.is_file():
        raise RuntimeError("technical amendment requires the retained original search freeze")
    value = _read_json(source)
    if (
        value.get("schema_version") != SEARCH_FREEZE_SCHEMA
        or value.get("verified") is not True
        or value.get("freeze_sha256") != _freeze_digest(value)
        or value.get("config_sha256") != config.stable_hash()
        or value.get("live_stage4_model_results_observed") is not False
        or value.get("search_tag") != search_tag
        or value.get("frozen_hashes") != _freeze_files(config)
        or not _amendment_freeze_valid(run, value)
        or not _amendment_tag_chain_valid(
            config.project_repo,
            value,
            _git_state(config.project_repo)["commit"],
        )
    ):
        raise RuntimeError("retained prior Stage 4 search freeze is invalid")
    states = _amendment_states(value)
    if not states and value.get("project_commit") != search_tag.get("commit"):
        raise RuntimeError("original Stage 4 search freeze commit is invalid")
    retained = run / _amendment_backup_name(SEARCH_AMENDMENT_TAG)
    if retained.is_file() and _read_json(retained) != value:
        raise RuntimeError("retained pre-amendment freeze conflicts with the prior freeze")
    if not retained.is_file():
        _atomic_json(retained, value)
    return value


def _amendment_freeze_valid(
    run: Path,
    value: Mapping[str, Any],
    visited: tuple[str, ...] = (),
) -> bool:
    amendment = value.get("amendment_tag")
    if not isinstance(amendment, Mapping):
        return all(
            value.get(key) is None
            for key in (
                "amendment_category",
                "previous_freeze_sha256",
                "scientific_identity_unchanged",
                "pre_amendment_live_model_evidence",
                "previous_freeze_path",
                "amendment_tags",
            )
        )
    tag_name = str(amendment.get("name", ""))
    category = SEARCH_AMENDMENT_CATEGORIES.get(tag_name)
    if category is None:
        return False
    evidence = value.get("pre_amendment_live_model_evidence")
    retained_name = str(
        value.get("previous_freeze_path", _amendment_backup_name(tag_name))
    )
    allowed_names = {
        _amendment_backup_name(name) for name in SEARCH_AMENDMENT_CATEGORIES
    }
    if retained_name not in allowed_names or retained_name in visited:
        return False
    retained_path = run / retained_name
    try:
        retained = _read_json(retained_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    states = _amendment_states(value)
    retained_states = _amendment_states(retained)
    return (
        value.get("amendment_category") == category
        and value.get("scientific_identity_unchanged") is True
        and isinstance(evidence, Mapping)
        and evidence.get("observed") is False
        and bool(states)
        and states[-1] == amendment
        and states[:-1] == retained_states
        and retained.get("freeze_sha256") == value.get("previous_freeze_sha256")
        and retained.get("freeze_sha256") == _freeze_digest(retained)
        and retained.get("frozen_hashes") == value.get("frozen_hashes")
        and retained.get("config_sha256") == value.get("config_sha256")
        and _amendment_freeze_valid(run, retained, (*visited, retained_name))
    )


def _record_post_live_amendment(
    config: Stage4SearchConfig,
    run: Path,
    project: Mapping[str, Any],
    search_tag: Mapping[str, Any],
    amendment_tag: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    freeze_value = _read_json(run / "search-freeze.json")
    if (
        freeze_value.get("schema_version") != SEARCH_FREEZE_SCHEMA
        or freeze_value.get("verified") is not True
        or freeze_value.get("freeze_sha256") != _freeze_digest(freeze_value)
        or freeze_value.get("config_sha256") != config.stable_hash()
        or freeze_value.get("search_tag") != search_tag
        or not _amendment_freeze_valid(run, freeze_value)
        or not _amendment_tag_chain_valid(
            config.project_repo,
            freeze_value,
            str(project["commit"]),
        )
        or not _is_ancestor(
            config.project_repo,
            str(freeze_value.get("project_commit", "")),
            str(project["commit"]),
        )
    ):
        raise RuntimeError("pre-recovery Stage 4 search freeze is invalid")
    recovery = _authentication_recovery_evidence(run)
    payload: dict[str, Any] = {
        "schema_version": POST_LIVE_AMENDMENT_SCHEMA,
        "verified": True,
        "inference": False,
        "live_stage4_model_output_observed": False,
        "project_commit": project["commit"],
        "project_branch": project["branch"],
        "technical_tag": amendment_tag,
        "amendment_category": SEARCH_AMENDMENT_CATEGORY,
        "search_freeze_sha256": freeze_value["freeze_sha256"],
        "config_sha256": config.stable_hash(),
        "frozen_hashes": _freeze_files(config),
        "scientific_identity_unchanged": (
            freeze_value.get("frozen_hashes") == _freeze_files(config)
            and freeze_value.get("config_sha256") == config.stable_hash()
        ),
        "authenticated_doctor_sha256": _sha_value(audit),
        "authenticated_doctor_artifact": audit.get("artifact_path"),
        "recovery_evidence": recovery,
    }
    audit_auth = audit.get("auth")
    if (
        payload["scientific_identity_unchanged"] is not True
        or audit.get("status") != "completed"
        or audit.get("inference") is not False
        or not isinstance(audit_auth, Mapping)
        or audit_auth.get("authenticated") is not True
        or not isinstance(payload["authenticated_doctor_artifact"], str)
    ):
        raise RuntimeError("authenticated Stage 4 amendment evidence is invalid")
    payload["amendment_sha256"] = _technical_amendment_digest(payload)
    _atomic_json(run / POST_LIVE_AMENDMENT_PATH, payload)
    return canonical_result(
        {
            "status": "completed",
            "run": str(run),
            "technical_amendment": True,
            **payload,
        }
    )


def _post_live_amendment_valid(
    config: Stage4SearchConfig,
    run: Path,
    freeze_value: Mapping[str, Any],
    descendant: str,
) -> bool:
    path = run / POST_LIVE_AMENDMENT_PATH
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    tag = value.get("technical_tag")
    recovery = value.get("recovery_evidence")
    if not isinstance(tag, Mapping) or not isinstance(recovery, Mapping):
        return False
    doctor_artifact = value.get("authenticated_doctor_artifact")
    doctor_artifact_valid = False
    if isinstance(doctor_artifact, str):
        doctor_path = run / doctor_artifact
        try:
            doctor_path.resolve().relative_to(run.resolve())
            doctor_value = _read_json(doctor_path)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        else:
            doctor_auth = doctor_value.get("auth")
            doctor_artifact_valid = (
                doctor_value.get("status") == "completed"
                and doctor_value.get("inference") is False
                and isinstance(doctor_auth, Mapping)
                and doctor_auth.get("authenticated") is True
                and _sha_value(doctor_value)
                == value.get("authenticated_doctor_sha256")
            )
    if (
        value.get("schema_version") != POST_LIVE_AMENDMENT_SCHEMA
        or value.get("verified") is not True
        or value.get("inference") is not False
        or value.get("live_stage4_model_output_observed") is not False
        or value.get("amendment_sha256") != _technical_amendment_digest(value)
        or value.get("amendment_category") != SEARCH_AMENDMENT_CATEGORY
        or value.get("search_freeze_sha256") != freeze_value.get("freeze_sha256")
        or value.get("config_sha256") != config.stable_hash()
        or value.get("frozen_hashes") != _freeze_files(config)
        or value.get("scientific_identity_unchanged") is not True
        or not isinstance(value.get("authenticated_doctor_sha256"), str)
        or len(str(value.get("authenticated_doctor_sha256"))) != 64
        or not isinstance(value.get("authenticated_doctor_artifact"), str)
        or not doctor_artifact_valid
        or tag.get("name") != SEARCH_AMENDMENT_TAG
        or tag.get("type") != "tag"
        or _tag_state(config.project_repo, SEARCH_AMENDMENT_TAG) != tag
        or not _is_ancestor(
            config.project_repo,
            str(tag.get("commit", "")),
            descendant,
        )
        or recovery.get("schema_version") != AUTH_RECOVERY_SCHEMA
        or recovery.get("verified") is not True
        or recovery.get("replacement_requests_authorized") is not True
        or recovery.get("slot_count") != 32
        or recovery.get("accepted_turns") != 0
        or recovery.get("contentful_turns") != 0
        or recovery.get("charged_turns") != 0
        or recovery.get("usage_tokens") != 0
    ):
        return False
    recovery_manifest = run / AUTH_RECOVERY_ROOT / "RECOVERY_MANIFEST.json"
    if recovery_manifest.is_file():
        try:
            retained = _read_json(recovery_manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            retained.get("phase") == "completed"
            and retained.get("manifest_sha256") == _recovery_digest(retained)
            and retained.get("checkpoint_sha256") == recovery.get("checkpoint_sha256")
            and retained.get("search_summary_sha256")
            == recovery.get("search_summary_sha256")
            and retained.get("archive_sha256") == recovery.get("archive_sha256")
        )
    try:
        current = _authentication_recovery_evidence(run)
    except RuntimeError:
        return False
    return all(
        current.get(key) == recovery.get(key)
        for key in (
            "checkpoint_sha256",
            "search_summary_sha256",
            "archive_sha256",
            "request_identities_sha256",
        )
    )


def freeze(
    config_path: str | Path,
    *,
    auth_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write the immutable search freeze after its annotated tag exists."""

    config = load_stage4_config(config_path)
    project = _git_state(config.project_repo)
    heg = _git_state(config.heg_repo)
    tag = _tag_state(config.project_repo, config.search_tag)
    amendment_tag = _tag_state(config.project_repo, SEARCH_AMENDMENT_TAG)
    if project["dirty"]:
        raise RuntimeError("project worktree must be clean before Stage 4 search freeze")
    if heg["dirty"] or heg["commit"] != config.frozen_heg_commit:
        raise RuntimeError("HEG must remain clean at the frozen commit")
    if tag["type"] != "tag":
        raise RuntimeError("Stage 4 search tag must remain annotated")
    amended = tag["commit"] != project["commit"]
    if amended and (
        amendment_tag["type"] != "tag"
        or amendment_tag["commit"] != project["commit"]
        or not _is_ancestor(
            config.project_repo,
            str(tag["commit"]),
            str(project["commit"]),
        )
    ):
        raise RuntimeError("Stage 4 technical amendment tag must be annotated at current HEAD")
    run = campaign_root(config)
    previous: dict[str, Any] | None = None
    live_evidence = _live_model_result_evidence(run)
    if amended:
        if live_evidence["observed"] is not False:
            if auth_json is None:
                raise RuntimeError(
                    "post-live technical amendment requires the proven --auth-json preflight"
                )
            live_audit = doctor(
                config_path,
                auth_json=auth_json,
                check_auth=True,
                write=True,
            )
            if live_audit.get("status") != "completed":
                raise RuntimeError("private-capsule App Server doctor is not READY")
            return _record_post_live_amendment(
                config,
                run,
                project,
                tag,
                amendment_tag,
                live_audit,
            )
        previous = _previous_search_freeze(config, run, tag)
    audit = doctor(
        config_path,
        auth_json=auth_json,
        check_auth=True,
        write=True,
    )
    if audit["status"] != "completed":
        raise RuntimeError("Stage 4 doctor is not READY")
    projection = cast(Mapping[str, Any], audit["projection"])
    archive_projection = cast(Mapping[str, Any], projection["manifest"])
    payload: dict[str, Any] = {
        "schema_version": SEARCH_FREEZE_SCHEMA,
        "verified": True,
        "inference": False,
        "live_stage4_model_results_observed": False,
        "project_commit": project["commit"],
        "project_branch": project["branch"],
        "heg_commit": heg["commit"],
        "search_tag": tag,
        "config_sha256": config.stable_hash(),
        "frozen_hashes": _freeze_files(config),
        "seed_capsule_count": 8,
        "seed_archive_sha256": config.identity.seed_manifest_sha256,
        "model": config.model.name,
        "reasoning_effort": config.model.effort,
        "generation_count": config.model.generations,
        "slots_per_generation": config.model.slots,
        "initial_turn_budget": config.model.max_initial_turns,
        "accepted_turn_budget": config.model.max_accepted_turns,
        "max_repairs_per_offspring": config.model.max_repairs,
        "generation_concurrency": config.model.concurrency,
        "evaluation_workers": config.limits.max_evaluation_workers,
        "reserved_physical_cores": config.limits.reserved_physical_cores,
        "artifact_projection": projection["projection"],
        "offline_evidence_manifest_sha256": archive_projection["manifest_sha256"],
        "doctor_sha256": _sha_value(audit),
    }
    if amended:
        assert previous is not None
        amendment_tags = [*_amendment_states(previous), amendment_tag]
        payload.update(
            {
                "amendment_tag": amendment_tag,
                "amendment_tags": amendment_tags,
                "amendment_category": SEARCH_AMENDMENT_CATEGORY,
                "previous_freeze_sha256": previous["freeze_sha256"],
                "previous_freeze_path": _amendment_backup_name(SEARCH_AMENDMENT_TAG),
                "scientific_identity_unchanged": (
                    previous.get("frozen_hashes") == payload["frozen_hashes"]
                    and previous.get("config_sha256") == payload["config_sha256"]
                ),
                "pre_amendment_live_model_evidence": live_evidence,
            }
        )
    payload["freeze_sha256"] = _freeze_digest(payload)
    _atomic_json(run / "search-freeze.json", payload)
    return canonical_result({"status": "completed", "run": str(run), **payload})


def _load_search_freeze(config: Stage4SearchConfig) -> dict[str, Any]:
    run = campaign_root(config)
    value = _read_json(run / "search-freeze.json")
    if (
        value.get("schema_version") != SEARCH_FREEZE_SCHEMA
        or value.get("verified") is not True
        or value.get("freeze_sha256") != _freeze_digest(value)
        or value.get("config_sha256") != config.stable_hash()
        or value.get("live_stage4_model_results_observed") is not False
        or not _amendment_freeze_valid(run, value)
    ):
        raise RuntimeError("Stage 4 search freeze is invalid")
    project = _git_state(config.project_repo)
    heg = _git_state(config.heg_repo)
    tag = _tag_state(config.project_repo, config.search_tag)
    amendment_states = _amendment_states(value)
    exact_freeze_head = (
        project["commit"] == value.get("project_commit")
        and (
            (
                bool(amendment_states)
                and amendment_states[-1].get("commit") == project["commit"]
            )
            or (
                not amendment_states
                and tag.get("commit") == project["commit"]
            )
        )
    )
    post_live_amendment = _post_live_amendment_valid(
        config,
        run,
        value,
        project["commit"],
    )
    if (
        not (exact_freeze_head or post_live_amendment)
        or project["dirty"]
        or heg["commit"] != config.frozen_heg_commit
        or heg["dirty"]
        or tag != value.get("search_tag")
        or not _amendment_tag_chain_valid(
            config.project_repo,
            value,
            project["commit"],
        )
    ):
        raise RuntimeError("repository state drifted after Stage 4 search freeze")
    if _freeze_files(config) != value.get("frozen_hashes"):
        raise RuntimeError("scientific Stage 4 inputs drifted after freeze")
    if not _stage3_checks(config)["ok"]:
        raise RuntimeError("authoritative Stage 3 evidence drifted after freeze")
    return value


def _load_retained_search_freeze(config: Stage4SearchConfig) -> dict[str, Any]:
    """Verify the search freeze after the repository advances to validation freeze."""

    run = campaign_root(config)
    value = _read_json(run / "search-freeze.json")
    tag = _tag_state(config.project_repo, config.search_tag)
    project = _git_state(config.project_repo)
    ancestor = _is_ancestor(
        config.project_repo,
        str(value.get("project_commit", "")),
        project["commit"],
    )
    technical_path = run / POST_LIVE_AMENDMENT_PATH
    technical_ok = (
        not technical_path.is_file()
        or _post_live_amendment_valid(
            config,
            run,
            value,
            project["commit"],
        )
    )
    if (
        value.get("schema_version") != SEARCH_FREEZE_SCHEMA
        or value.get("verified") is not True
        or value.get("freeze_sha256") != _freeze_digest(value)
        or value.get("config_sha256") != config.stable_hash()
        or value.get("live_stage4_model_results_observed") is not False
        or not _amendment_freeze_valid(run, value)
        or tag != value.get("search_tag")
        or not _amendment_tag_chain_valid(
            config.project_repo,
            value,
            project["commit"],
        )
        or not ancestor
        or not technical_ok
        or _freeze_files(config) != value.get("frozen_hashes")
        or not _stage3_checks(config)["ok"]
    ):
        raise RuntimeError("retained Stage 4 search freeze is invalid")
    return value


def archive_inspect(run: str | Path) -> dict[str, Any]:
    root = Path(run).resolve()
    archive = ProgramArchive(_archive_root(root))
    return canonical_result(
        {
            "schema_version": "stage4.archive.inspect.v1",
            "status": "completed",
            "run": str(root),
            **archive.inspect(),
        }
    )


def archive_reindex(run: str | Path) -> dict[str, Any]:
    root = Path(run).resolve()
    report = ProgramArchive(_archive_root(root)).reindex()
    return canonical_result(
        {
            "schema_version": "stage4.archive.reindex.v1",
            "status": "completed" if report.ok else "failed",
            "run": str(root),
            **report.as_dict(),
        }
    )


def _baseline_sources(config: Stage4SearchConfig) -> dict[str, str]:
    return {
        "random": config.random_policy_path.read_text(encoding="utf-8"),
        "structural": config.structural_policy_path.read_text(encoding="utf-8"),
    }


def _seed_sources(config: Stage4SearchConfig) -> tuple[SeedRecord, ...]:
    return load_seed_manifest(config.seed_manifest_path)


def _summary_metrics(summary: Mapping[str, Any], policy_id: str) -> Mapping[str, JsonValue]:
    policies = summary.get("policies")
    if not isinstance(policies, Mapping) or not isinstance(policies.get(policy_id), Mapping):
        raise RuntimeError(f"evaluation summary is missing policy {policy_id}")
    return cast(Mapping[str, JsonValue], policies[policy_id])


def _policy_evaluation_valid(
    records: Sequence[Mapping[str, Any]],
    policy_id: str,
) -> bool:
    for record in records:
        policies = record.get("policies")
        policy = policies.get(policy_id) if isinstance(policies, Mapping) else None
        if (
            record.get("terminal_status") != "completed"
            or int(record.get("invalid_graphs", 0)) != 0
            or not isinstance(policy, Mapping)
            or int(policy.get("failure_count", 0)) != 0
        ):
            return False
    return bool(records)


def _append_seed_records(
    run: Path,
    archive: ProgramArchive,
    seeds: Sequence[SeedRecord],
    summary: Mapping[str, Any],
) -> None:
    existing = {record.program_id for record in archive.records()}
    for seed in seeds:
        if seed.candidate_id in existing:
            continue
        source_path = _source_path(run, seed.candidate_id)
        _atomic_source(source_path, seed.source)
        archive.append(
            ProgramRecord(
                program_id=seed.candidate_id,
                source_path=_relative_to_run(run, source_path),
                source_sha256=seed.source_sha256,
                normalized_ast_sha256=seed.ast_sha256,
                behavior_signature=seed.behavior_signature,
                generation=0,
                slot=seed.slot_id,
                validation_status="valid",
                probe_status="passed",
                smoke_10k_status="passed",
                replay_status="verified",
                search_metrics=_summary_metrics(summary, seed.candidate_id),
                fitness_status="verified",
                seed_id=seed.candidate_id,
                generation_mode="imported-stage3",
                metadata={
                    "design_summary": seed.design_summary,
                    "used_fields": list(seed.used_fields),
                    "assumptions": list(seed.assumptions),
                    "provenance": cast(JsonValue, dict(seed.provenance)),
                },
            )
        )


def _program_source(run: Path, record: ProgramRecord) -> str:
    path = Path(record.source_path)
    resolved = path if path.is_absolute() else run / path
    return resolved.read_text(encoding="utf-8")


def _compact_parent_feedback(record: ProgramRecord) -> str:
    metrics = record.search_metrics
    by_order = metrics.get("by_order", {}) if isinstance(metrics, Mapping) else {}
    value = {
        "program_id": record.program_id,
        "pooled_median_auc": metrics.get("pooled_median_auc"),
        "order_10_median_auc": (
            cast(Mapping[str, Any], by_order.get("10", {})).get("median_auc")
            if isinstance(by_order, Mapping)
            else None
        ),
        "median_best_total_witness": metrics.get(
            "pooled_median_best_total_witness",
            metrics.get("pooled_median_best_witnesses"),
        ),
        "behavior_signature": record.behavior_signature_sha256,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bounded_archive_context(records: Sequence[ProgramRecord]) -> str:
    ranked = sorted(
        (record for record in records if record.unique and record.fitness_status == "verified"),
        key=lambda item: item.program_id,
    )
    value = [
        {
            "program_id": item.program_id,
            "generation": item.generation,
            "ast_sha256": item.normalized_ast_sha256,
            "behavior_signature": item.behavior_signature_sha256,
        }
        for item in ranked[:8]
    ]
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _usage_complete(value: Mapping[str, Any]) -> bool:
    required = (
        "inputTokens",
        "cachedInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    return (
        value.get("final") is True
        and value.get("partial") is False
        and all(
            isinstance(value.get(name), int)
            and not isinstance(value.get(name), bool)
            and int(value[name]) >= 0
            for name in required
        )
    )


def _slot_usage(slot: SlotResult) -> Mapping[str, JsonValue]:
    values: dict[str, int | bool] = {}
    for envelope in (slot.initial, slot.repair or {}):
        usage = envelope.get("usage") if isinstance(envelope, Mapping) else None
        if not isinstance(usage, Mapping):
            continue
        for key, item in usage.items():
            if isinstance(item, int) and not isinstance(item, bool):
                values[str(key)] = int(values.get(str(key), 0)) + item
            elif isinstance(item, bool):
                values[str(key)] = bool(values.get(str(key), False)) or item
    return cast(Mapping[str, JsonValue], values)


def _write_generation_raw(run: Path, generation: int, slot: SlotResult) -> Path:
    return write_raw_slot_record(
        run,
        generation,
        int(slot.slot[-2:]),
        {
            "source": slot.candidate.source if slot.candidate else "",
            "request": dict(slot.request),
            "response": dict(slot.raw_result),
            "transcript": {
                "initial": dict(slot.initial),
                "repair": dict(slot.repair) if slot.repair else None,
            },
            "usage": dict(_slot_usage(slot)),
            "reference": {
                "generation": generation,
                "slot": slot.slot,
                "parent_id": slot.parent_id,
                "status": slot.status,
            },
        },
    )


def _candidate_id(candidate: Candidate) -> str:
    return deterministic_program_id(
        generation=candidate.generation + 1,
        slot=candidate.slot,
        source_sha256=candidate.source_sha256,
        parent_id=candidate.parent_id,
    )


def _program_metrics_summary(
    records: Sequence[ProgramRecord],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    policies: dict[str, Any] = {}
    identities: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record.unique or record.fitness_status != "verified":
            continue
        policies[record.program_id] = dict(record.search_metrics)
        identities[record.program_id] = {
            "normalized_ast_sha256": record.normalized_ast_sha256,
            "origin": "stage3" if record.generation == 0 else "stage4",
            "generation": record.generation,
            "is_stage4": record.generation > 0,
        }
    return {"policies": policies, "policy_identities": identities}, identities


def evolve(
    config_path: str | Path,
    *,
    provider: Any | None = None,
    concurrency: int = 8,
    resume: bool = True,
    auth_json: str | Path | None = None,
    observer: Observer | None = None,
) -> dict[str, Any]:
    """Run or resume the exact four-generation Stage 4 search campaign."""

    if concurrency != 8:
        raise ValueError("Stage 4 generation concurrency is frozen at 8")
    config = load_stage4_config(config_path)
    authenticated_preflight_sha256: str | None = None
    if provider is None:
        if auth_json is None:
            raise RuntimeError(
                "Stage 4 live generation requires the proven Stage 3 --auth-json flow"
            )
        live_doctor = doctor(
            config_path,
            auth_json=auth_json,
            check_auth=True,
            write=True,
        )
        if (
            live_doctor.get("status") != "completed"
            or cast(Mapping[str, Any], live_doctor.get("auth", {})).get(
                "authenticated"
            )
            is not True
        ):
            raise RuntimeError(
                "Stage 4 private-capsule App Server doctor is not READY"
            )
        authenticated_preflight_sha256 = _sha_value(live_doctor)
    freeze_value = _load_search_freeze(config)
    run = campaign_root(config)
    auth_recovery: Mapping[str, Any] | None = None
    generation_doctor_sha256 = str(freeze_value["doctor_sha256"])
    if (run / POST_LIVE_AMENDMENT_PATH).is_file():
        technical_amendment = _read_json(run / POST_LIVE_AMENDMENT_PATH)
        generation_doctor_sha256 = str(
            technical_amendment["authenticated_doctor_sha256"]
        )
        auth_recovery = _prepare_authentication_recovery(run)
    archive = ProgramArchive(_archive_root(run))
    manifest = _load_manifest(config.manifest_path)
    seeds = _seed_sources(config)
    baselines = _baseline_sources(config)
    baseline_ast_representatives = {
        cast(str, validate_policy(source, config.sandbox).identity.normalized_ast_sha256): (
            f"baseline-{name}"
        )
        for name, source in baselines.items()
    }
    emit = observer or (lambda _: None)
    started_ns = time.monotonic_ns()
    emit(
        {
            "event": "stage4_started",
            "run": str(run),
            "generation": 0,
            "generation_slots": 8,
            "evaluation_workers": 8,
            "reserved_physical_cores": 8,
            "checkpoint": str(run / "generation-checkpoint.json"),
            "remaining_initial_turns": 32,
            "remaining_accepted_turn_budget": 64,
        }
    )

    seed_ids = {seed.candidate_id for seed in seeds}
    existing_by_id = {record.program_id: record for record in archive.records()}
    for seed in seeds:
        existing = existing_by_id.get(seed.candidate_id)
        if existing is not None and (
            existing.generation != 0
            or existing.source_sha256 != seed.source_sha256
            or existing.normalized_ast_sha256 != seed.ast_sha256
            or existing.behavior_signature_sha256 != seed.behavior_signature
        ):
            raise RuntimeError(f"partial seed archive identity mismatch: {seed.candidate_id}")
    seed_summary_path = run / "evaluations" / "seed-summary.json"
    if not seed_ids.issubset(existing_by_id) or not seed_summary_path.is_file():
        seed_policies = {seed.candidate_id: seed.source for seed in seeds}
        roster = {**baselines, **seed_policies}
        amendment = freeze_value.get("amendment_tag")
        amendment_version = (
            str(amendment.get("name", "")).rsplit("-", 1)[-1]
            if isinstance(amendment, Mapping)
            else ""
        )
        seed_attempt = (
            f"search-seeds-amendment-{amendment_version}"
            if amendment_version
            else "search-seeds"
        )
        seed_root = run / "evaluations" / seed_attempt
        primary = evaluate_policy_roster_manifest(
            config,
            manifest,
            roster,
            output_dir=seed_root,
            workers=8,
            shard_count=8,
            pass_name="primary",
            resume=resume,
        )
        replay = evaluate_policy_roster_manifest(
            config,
            manifest,
            roster,
            output_dir=seed_root,
            workers=8,
            shard_count=8,
            pass_name="replay",
            resume=resume,
        )
        replay_check = verify_replay_pair(primary, replay)
        if replay_check.get("exact") is not True:
            raise RuntimeError("seed primary/replay evaluation is not exact")
        seed_summary = summarize_development(
            cast(Sequence[Mapping[str, Any]], primary["records"]),
            roster,
            bootstrap_samples=config.evaluation.bootstrap_samples,
            bootstrap_seed=config.evaluation.bootstrap_seed,
        )
        _append_seed_records(run, archive, seeds, seed_summary)
        _atomic_json(
            seed_summary_path,
            {
                "summary": seed_summary,
                "primary": {k: v for k, v in primary.items() if k != "records"},
                "replay": {k: v for k, v in replay.items() if k != "records"},
                "replay_check": replay_check,
            },
        )
    if not seed_ids.issubset({record.program_id for record in archive.records()}):
        raise RuntimeError("seed archive initialization is incomplete")

    app_provider = provider or Stage4AppServerProvider(
        auth_json=auth_json,
        artifact_dir=run / "appserver",
        artifact_root=run,
        artifact_max_bytes=config.limits.artifact_compressed_campaign_bytes,
    )
    coordinator: GenerationCoordinator

    def parent_selector(generation: int) -> Mapping[str, str]:
        records = archive.records()
        selection = select_parents(records)
        assignments = dict(selection.slots)
        by_id = {record.program_id: record for record in records}
        for program_id in assignments.values():
            record = by_id[program_id]
            coordinator.parent_sources[program_id] = _program_source(run, record)
            coordinator.parent_records[program_id] = record
        coordinator.search_feedback = {
            generation: {
                slot: _compact_parent_feedback(by_id[program_id])
                for slot, program_id in assignments.items()
            }
        }
        coordinator.archive_context = {generation: _bounded_archive_context(records)}
        _atomic_json(
            run / "selection" / f"generation-{generation + 1:02d}.json",
            {
                "schema_version": "stage4.parent_selection.v1",
                "generation": generation + 1,
                **selection.as_dict(),
            },
        )
        return assignments

    def generation_completed(
        generation: int,
        slots: tuple[SlotResult, ...],
        _cumulative: tuple[Candidate, ...],
    ) -> None:
        for slot in slots:
            _write_generation_raw(run, generation + 1, slot)
        existing_records = archive.records()
        known_slots = {(record.generation, record.slot) for record in existing_records}
        by_ast = dict(baseline_ast_representatives)
        by_ast.update(
            {
            record.normalized_ast_sha256: record.program_id
            for record in existing_records
            if record.unique
            }
        )
        representative_by_ast = dict(by_ast)
        accepted: dict[str, Candidate] = {}
        slot_ids: dict[str, str] = {}
        for slot in slots:
            if slot.candidate is None:
                continue
            program_id = _candidate_id(slot.candidate)
            slot_ids[slot.slot] = program_id
            ast_sha256 = slot.candidate.normalized_ast_sha256
            if ast_sha256 not in representative_by_ast:
                accepted[program_id] = slot.candidate
                representative_by_ast[ast_sha256] = program_id
        evaluation_summary: dict[str, Any] = {"policies": {}}
        replay_check: Mapping[str, Any] = {"exact": True, "empty": True}
        primary_records: Sequence[Mapping[str, Any]] = ()
        if accepted:
            roster = {
                **baselines,
                **{program_id: candidate.source for program_id, candidate in accepted.items()},
            }
            root = run / "evaluations" / f"generation-{generation + 1:02d}"
            primary = evaluate_policy_roster_manifest(
                config,
                manifest,
                roster,
                output_dir=root,
                workers=8,
                shard_count=8,
                pass_name="primary",
                resume=resume,
            )
            replay = evaluate_policy_roster_manifest(
                config,
                manifest,
                roster,
                output_dir=root,
                workers=8,
                shard_count=8,
                pass_name="replay",
                resume=resume,
            )
            replay_check = verify_replay_pair(primary, replay)
            if replay_check.get("exact") is not True:
                raise RuntimeError(f"generation {generation + 1} replay mismatch")
            primary_records = cast(
                Sequence[Mapping[str, Any]],
                primary["records"],
            )
            evaluation_summary = summarize_development(
                primary_records,
                roster,
                bootstrap_samples=config.evaluation.bootstrap_samples,
                bootstrap_seed=config.evaluation.bootstrap_seed,
            )
            _atomic_json(
                root / "generation-summary.json",
                {
                    "summary": evaluation_summary,
                    "primary": {k: v for k, v in primary.items() if k != "records"},
                    "replay": {k: v for k, v in replay.items() if k != "records"},
                    "replay_check": replay_check,
                },
            )
        evaluated_valid = {
            program_id
            for program_id in accepted
            if _policy_evaluation_valid(primary_records, program_id)
        }
        current = {record.program_id for record in archive.records()}
        for slot in slots:
            if (generation + 1, slot.slot) in known_slots:
                continue
            if (
                slot.candidate is None
                and cached_pre_turn_auth_retry_allowed(slot.as_dict())
            ):
                continue
            usage = _slot_usage(slot)
            request_id = None
            thread_id = None
            turn_id = None
            for envelope in (slot.repair or {}, slot.initial):
                if not isinstance(envelope, Mapping):
                    continue
                request_id = request_id or envelope.get("request_id")
                thread_id = thread_id or envelope.get("thread_id")
                turn_id = turn_id or envelope.get("turn_id")
            request_id = request_id or slot.request.get("idempotency_key")
            unauthorized = any(
                bool(envelope.get("unauthorized_tool_approval", False))
                for envelope in (slot.initial, slot.repair or {})
                if isinstance(envelope, Mapping)
            )
            accepted_turn_count = sum(
                envelope.get("accepted") is True
                for envelope in (slot.initial, slot.repair or {})
                if isinstance(envelope, Mapping)
            )
            if slot.candidate is None:
                identity_payload = {
                    "generation": generation + 1,
                    "slot": slot.slot,
                    "parent": slot.parent_id,
                    "request": request_id,
                }
                pseudo = _sha_value(identity_payload)
                program_id = f"tombstone-{pseudo[:24]}"
                record = ProgramRecord(
                    program_id=program_id,
                    source_path="",
                    source_sha256=pseudo,
                    normalized_ast_sha256=pseudo,
                    behavior_signature=pseudo,
                    generation=generation + 1,
                    slot=slot.slot,
                    parent_id=slot.parent_id,
                    mutation_brief_id=cast(str | None, slot.request.get("brief_id")),
                    request_id=str(request_id) if request_id is not None else None,
                    thread_id=str(thread_id) if thread_id is not None else None,
                    turn_id=str(turn_id) if turn_id is not None else None,
                    usage=usage,
                    validation_status="failed",
                    probe_status="failed",
                    smoke_10k_status="failed",
                    replay_status="not_evaluated",
                    fitness_status="failed",
                    tombstone=True,
                    error="; ".join(str(item.get("code", "failed")) for item in slot.errors),
                    seed_id=archive.lineage(slot.parent_id)[-1],
                    generation_mode="mutation",
                    metadata={
                        "repairs": slot.repairs,
                        "status": slot.status,
                        "usage_complete": _usage_complete(usage),
                        "unauthorized_tool_approval": unauthorized,
                        "accepted_turn_count": accepted_turn_count,
                    },
                )
            else:
                candidate = slot.candidate
                program_id = slot_ids[slot.slot]
                representative = representative_by_ast.get(candidate.normalized_ast_sha256)
                duplicate_of = representative if representative != program_id else None
                source_path = _source_path(run, program_id)
                _atomic_source(source_path, candidate.source)
                record = ProgramRecord(
                    program_id=program_id,
                    source_path=_relative_to_run(run, source_path),
                    source_sha256=candidate.source_sha256,
                    normalized_ast_sha256=candidate.normalized_ast_sha256,
                behavior_signature=cast(
                    Mapping[str, JsonValue],
                    candidate.behavior_signature,
                ),
                    generation=generation + 1,
                    slot=slot.slot,
                    parent_id=slot.parent_id,
                    mutation_brief_id=cast(str | None, slot.request.get("brief_id")),
                    request_id=str(request_id) if request_id is not None else None,
                    thread_id=str(thread_id) if thread_id is not None else None,
                    turn_id=str(turn_id) if turn_id is not None else None,
                    usage=usage,
                    validation_status="valid",
                    probe_status="passed",
                    smoke_10k_status="passed",
                    replay_status="verified" if duplicate_of is None else "duplicate",
                    duplicate_of=duplicate_of,
                    search_metrics=(
                        _summary_metrics(evaluation_summary, program_id)
                        if duplicate_of is None
                        else {}
                    ),
                    fitness_status=(
                        "verified"
                        if duplicate_of is None and program_id in evaluated_valid
                        else "failed"
                        if duplicate_of is None
                        else "duplicate"
                    ),
                    seed_id=archive.lineage(slot.parent_id)[-1],
                    generation_mode="mutation",
                    metadata={
                        "repairs": slot.repairs,
                        "status": slot.status,
                        "replay_exact": replay_check.get("exact"),
                        "usage_complete": _usage_complete(usage),
                        "unauthorized_tool_approval": unauthorized,
                        "accepted_turn_count": accepted_turn_count,
                    },
                )
            if record.program_id not in current:
                archive.append(record)
                current.add(record.program_id)
        reindex = archive.reindex()
        if not reindex.ok:
            raise RuntimeError(f"archive reindex failed: {reindex.errors}")
        archive_inspection = archive.inspect()
        accepted_turns = sum(
            value
            for record in archive.records()
            if isinstance(
                (value := record.metadata.get("accepted_turn_count", 0)),
                int,
            )
            and not isinstance(value, bool)
        )
        elapsed_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
        completed_initial = (generation + 1) * 8
        emit(
            {
                "event": "generation_completed",
                "generation": generation + 1,
                "slots": len(slots),
                "slot_results": [
                    {
                        "slot": slot.slot,
                        "status": slot.status,
                        "yield": slot.candidate is not None,
                        "repairs": slot.repairs,
                        "usage": dict(_slot_usage(slot)),
                    }
                    for slot in slots
                ],
                "accepted_unique": len(accepted),
                "archive_hash": reindex.archive_hash,
                "leaders": archive_inspection["leaders"],
                "usage": archive_inspection["usage"],
                "primary_shards": 8 if accepted else 0,
                "replay_shards": 8 if accepted else 0,
                "failures": sum(slot.candidate is None for slot in slots),
                "checkpoint": str(run / "generation-checkpoint.json"),
                "elapsed_seconds": elapsed_seconds,
                "remaining_initial_turns": 32 - completed_initial,
                "remaining_accepted_turn_budget": 64 - accepted_turns,
            }
        )

    briefs = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(config.briefs_dir.glob("slot-*.md"))
    }
    coordinator = GenerationCoordinator(
        app_provider,
        config=GenerationConfig(
            campaign_id=f"stage4-{config.stable_hash()[:12]}",
            sandbox_limits=config.sandbox,
            checkpoint_path=run / "generation-checkpoint.json",
            model=config.model.name,
            effort=config.model.effort,
            appserver_doctor_sha256=generation_doctor_sha256,
        ),
        briefs=briefs,
        parent_selector=parent_selector,
        checkpoint_path=run / "generation-checkpoint.json",
        existing_sources=[seed.source for seed in seeds] + list(baselines.values()),
        retry_infrastructure=True,
        checkpoint_store=CheckpointStore(run / "checkpoints"),
        generation_completed=generation_completed,
    )
    generation_result: GenerationResult = coordinator.run(resume=resume)
    records = archive.records()
    search_summary, identities = _program_metrics_summary(records)
    new_unique_valid = sum(
        record.generation > 0
        and record.unique
        and record.validation_status == "valid"
        and record.probe_status == "passed"
        and record.smoke_10k_status == "passed"
        and record.replay_status == "verified"
        and record.fitness_status == "verified"
        for record in records
    )
    champion = (
        select_champion(
            search_summary,
            identities,
            generation=4,
            seed_ast_hashes=[seed.ast_sha256 for seed in seeds],
            baseline_ast_hashes=[
                validate_policy(source, config.sandbox).identity.normalized_ast_sha256 or ""
                for source in baselines.values()
            ],
        )
        if new_unique_valid >= 16
        else None
    )
    archive_report = archive.reindex()
    generated_records = [record for record in records if record.generation > 0]
    initial_turns = len(generation_result.slots)
    repair_turns = sum(
        value
        for record in generated_records
        if isinstance((value := record.metadata.get("repairs", 0)), int)
        and not isinstance(value, bool)
    )
    exact_usage = bool(generated_records) and all(
        bool(record.metadata.get("usage_complete", False)) for record in generated_records
    )
    accepted_live_turns = sum(
        value
        for record in generated_records
        if isinstance((value := record.metadata.get("accepted_turn_count", 0)), int)
        and not isinstance(value, bool)
    )
    if accepted_live_turns > config.model.max_accepted_turns:
        raise RuntimeError("Stage 4 accepted live-turn budget was exceeded")
    unauthorized = any(
        bool(record.metadata.get("unauthorized_tool_approval", False))
        for record in generated_records
    )
    infrastructure_failures = [
        slot
        for slot in generation_result.slots
        if slot.candidate is None
        and cached_pre_turn_auth_retry_allowed(slot.as_dict())
    ]
    result: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "status": generation_result.status,
        "run": str(run),
        "search_freeze_sha256": freeze_value["freeze_sha256"],
        "authentication_recovery": (
            {
                "manifest_sha256": auth_recovery.get("manifest_sha256"),
                "replacement_requests_authorized": auth_recovery.get(
                    "replacement_requests_authorized"
                ),
                "slot_count": auth_recovery.get("slot_count"),
            }
            if auth_recovery is not None
            else None
        ),
        "authenticated_preflight_sha256": authenticated_preflight_sha256,
        "generation_doctor_sha256": generation_doctor_sha256,
        "generation": generation_result.summary,
        "archive": archive.inspect(),
        "archive_reindex": archive_report.as_dict(),
        "search_summary": search_summary,
        "initial_turns": initial_turns,
        "repair_turns": repair_turns,
        "accepted_live_turns": accepted_live_turns,
        "new_unique_valid_offspring": new_unique_valid,
        "exact_usage": exact_usage,
        "unauthorized_tool_approval": unauthorized,
        "live_stage4_model_results_observed": accepted_live_turns > 0,
    }
    if infrastructure_failures or initial_turns != 32 or not exact_usage:
        result["champion"] = None
        result["decision"] = "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"
        result["decision_reason"] = (
            "pre_turn_infrastructure_failures"
            if infrastructure_failures
            else "incomplete_turn_or_usage_accounting"
        )
        result["infrastructure_failure_count"] = len(infrastructure_failures)
    elif champion is None:
        result["champion"] = None
        result["decision"] = "NO_GO"
        result["decision_reason"] = (
            "minimum_unique_offspring_not_met"
            if new_unique_valid < 16
            else "no_distinct_stage4_champion"
        )
    else:
        champion_record = next(record for record in records if record.program_id == champion)
        result["champion"] = {
            "program_id": champion,
            "source_sha256": champion_record.source_sha256,
            "normalized_ast_sha256": champion_record.normalized_ast_sha256,
            "generation": champion_record.generation,
            "source_path": champion_record.source_path,
        }
        result["decision"] = "PENDING_VALIDATION"
    _atomic_json(run / "search-summary.json", result)
    build_evidence_manifest(run)
    emit({"event": "search_completed", "champion": champion, "run": str(run)})
    return canonical_result(result)


def resume(
    run: str | Path,
    *,
    config_path: str | Path = "configs/stage4-search.toml",
    provider: Any | None = None,
    auth_json: str | Path | None = None,
    observer: Observer | None = None,
) -> dict[str, Any]:
    config = load_stage4_config(config_path)
    expected = campaign_root(config).resolve()
    if Path(run).resolve() != expected:
        raise ValueError(f"run must be the frozen campaign path {expected}")
    return evolve(
        config_path,
        provider=provider,
        concurrency=8,
        resume=True,
        auth_json=auth_json,
        observer=observer,
    )


def evaluate_candidate(
    run: str | Path,
    program_id: str,
    *,
    pass_name: str,
    workers: int = 8,
    config_path: str | Path = "configs/stage4-search.toml",
) -> dict[str, Any]:
    if pass_name not in {"primary", "replay"}:
        raise ValueError("evaluation pass must be primary or replay")
    config = load_stage4_config(config_path)
    root = Path(run).resolve()
    archive = ProgramArchive(_archive_root(root))
    record = next(
        (item for item in archive.records() if item.program_id == program_id),
        None,
    )
    if record is None or not record.unique or record.tombstone:
        raise ValueError("program is not an eligible unique archive program")
    manifest = _load_manifest(config.manifest_path)
    result = evaluate_program_manifest(
        config,
        manifest,
        program_id,
        _program_source(root, record),
        baselines=_baseline_sources(config),
        output_dir=root / "evaluations" / "individual" / program_id,
        workers=workers,
        shard_count=8,
        pass_name=pass_name,
        resume=True,
    )
    return canonical_result({key: value for key, value in result.items() if key != "records"})


def freeze_validation(
    run: str | Path,
    *,
    config_path: str | Path = "configs/stage4-search.toml",
) -> dict[str, Any]:
    config = load_stage4_config(config_path)
    root = Path(run).resolve()
    if root != campaign_root(config).resolve():
        raise ValueError("validation freeze requires the frozen official campaign")
    search = _read_json(root / "search-summary.json")
    champion = search.get("champion")
    if not isinstance(champion, Mapping):
        raise RuntimeError("search champion identity is missing")
    search_freeze = _load_retained_search_freeze(config)
    project = _git_state(config.project_repo)
    heg = _git_state(config.heg_repo)
    tag = _tag_state(config.project_repo, config.validation_tag)
    if project["dirty"]:
        raise RuntimeError("project worktree must be clean before validation freeze")
    if heg["dirty"] or heg["commit"] != config.frozen_heg_commit:
        raise RuntimeError("HEG drift before validation freeze")
    if tag["type"] != "tag" or tag["commit"] != project["commit"]:
        raise RuntimeError("Stage 4 validation tag must be annotated at current HEAD")
    archive = ProgramArchive(_archive_root(root)).reindex()
    if not archive.ok:
        raise RuntimeError("archive must reindex exactly before validation freeze")
    champion_id = champion.get("program_id")
    champion_record = next(
        (record for record in archive.records if record.program_id == champion_id),
        None,
    )
    if (
        champion_record is None
        or not champion_record.unique
        or champion_record.generation < 1
        or champion_record.fitness_status != "verified"
        or champion.get("source_sha256") != champion_record.source_sha256
        or champion.get("normalized_ast_sha256") != champion_record.normalized_ast_sha256
        or champion.get("generation") != champion_record.generation
        or search.get("decision") != "PENDING_VALIDATION"
        or int(search.get("new_unique_valid_offspring", 0)) < 16
    ):
        raise RuntimeError("search champion/archive binding is invalid")
    statistics_path = config.project_repo / "src" / "mutation_forge" / "stage4" / "statistics.py"
    payload: dict[str, Any] = {
        "schema_version": VALIDATION_FREEZE_SCHEMA,
        "verified": True,
        "inference": False,
        "final_stage4_validation_results_observed": False,
        "project_commit": project["commit"],
        "heg_commit": heg["commit"],
        "validation_tag": tag,
        "champion": dict(champion),
        "archive_sha256": archive.archive_hash,
        "search_summary_sha256": _sha_file(root / "search-summary.json"),
        "search_freeze_sha256": search_freeze["freeze_sha256"],
        "config_sha256": config.stable_hash(),
        "validation_manifest_sha256": _sha_file(config.validation_manifest_path),
        "statistics_sha256": _sha_file(statistics_path),
        "gate_names": list(gate_report({}, champion=None)["checks"]),
        "bootstrap_samples": config.evaluation.bootstrap_samples,
        "bootstrap_seed": config.evaluation.bootstrap_seed,
        "confidence_level": config.evaluation.confidence_level,
    }
    payload["freeze_sha256"] = _freeze_digest(payload)
    _atomic_json(root / "validation-freeze.json", payload)
    build_evidence_manifest(root)
    return canonical_result({"status": "completed", "run": str(root), **payload})


def _load_validation_freeze(
    config: Stage4SearchConfig,
    run: Path,
) -> dict[str, Any]:
    if run.resolve() != campaign_root(config).resolve():
        raise RuntimeError("validation requires the frozen official campaign")
    value = _read_json(run / "validation-freeze.json")
    if (
        value.get("schema_version") != VALIDATION_FREEZE_SCHEMA
        or value.get("verified") is not True
        or value.get("freeze_sha256") != _freeze_digest(value)
        or value.get("final_stage4_validation_results_observed") is not False
    ):
        raise RuntimeError("validation freeze is invalid")
    project = _git_state(config.project_repo)
    tag = _tag_state(config.project_repo, config.validation_tag)
    archive = ProgramArchive(_archive_root(run)).reindex()
    search = _read_json(run / "search-summary.json")
    champion = search.get("champion")
    champion_id = champion.get("program_id") if isinstance(champion, Mapping) else None
    champion_record = next(
        (record for record in archive.records if record.program_id == champion_id),
        None,
    )
    search_freeze = _load_retained_search_freeze(config)
    if (
        project["dirty"]
        or project["commit"] != value.get("project_commit")
        or tag != value.get("validation_tag")
        or _git_state(config.heg_repo)["commit"] != config.frozen_heg_commit
        or _git_state(config.heg_repo)["dirty"]
        or _sha_file(config.validation_manifest_path)
        != value.get("validation_manifest_sha256")
        or _sha_file(
            config.project_repo / "src" / "mutation_forge" / "stage4" / "statistics.py"
        )
        != value.get("statistics_sha256")
        or value.get("config_sha256") != config.stable_hash()
        or value.get("search_freeze_sha256") != search_freeze.get("freeze_sha256")
        or value.get("search_summary_sha256")
        != _sha_file(run / "search-summary.json")
        or not archive.ok
        or value.get("archive_sha256") != archive.archive_hash
        or not isinstance(champion, Mapping)
        or value.get("champion") != dict(champion)
        or champion_record is None
        or champion_record.source_sha256 != champion.get("source_sha256")
        or champion_record.normalized_ast_sha256
        != champion.get("normalized_ast_sha256")
    ):
        raise RuntimeError("repository drift after validation freeze")
    return value


def _rename_stage3_policy(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        policies = value.get("policies")
        if isinstance(policies, Mapping):
            renamed = dict(policies)
            renamed["stage3_champion"] = renamed.pop("stage3-candidate-slot-04")
            value["policies"] = renamed
        result.append(value)
    return result


def _validation_gate_evidence(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    champion: str,
) -> dict[str, Any]:
    policies = cast(Mapping[str, Mapping[str, Any]], summary["policies"])
    champion_metrics = policies[champion]
    stage3_metrics = policies["stage3_champion"]
    pooled = float(champion_metrics["pooled_median_auc"])
    stage3 = float(stage3_metrics["pooled_median_auc"])
    order_deltas = {
        order: float(cast(Mapping[str, Any], champion_metrics["by_order"])[order]["median_auc"])
        - float(cast(Mapping[str, Any], stage3_metrics["by_order"])[order]["median_auc"])
        for order in ("10", "12")
    }
    graph_counts: dict[str, int] = {}
    for order in (10, 12):
        count = 0
        graph_seeds = sorted(
            {int(record["graph_seed"]) for record in records if int(record["order"]) == order}
        )
        for graph_seed in graph_seeds:
            deltas: list[float] = []
            for record in records:
                if int(record["order"]) != order or int(record["graph_seed"]) != graph_seed:
                    continue
                policies_record = cast(Mapping[str, Mapping[str, Any]], record["policies"])
                champion_curve = cast(
                    Sequence[float],
                    policies_record[champion]["normalized_best_so_far_curve"],
                )
                stage3_curve = cast(
                    Sequence[float],
                    policies_record["stage3_champion"]["normalized_best_so_far_curve"],
                )
                deltas.append(
                    sum(float(item) for item in champion_curve) / len(champion_curve)
                    - sum(float(item) for item in stage3_curve) / len(stage3_curve)
                )
            ordered = sorted(deltas)
            median = (
                ordered[len(ordered) // 2]
                if len(ordered) % 2
                else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
            )
            count += int(median >= 0)
        graph_counts[str(order)] = count
    bootstrap = hierarchical_bootstrap(
        records,
        policy=champion,
        baseline="stage3_champion",
        samples=10_000,
        seed=2026073004,
        confidence=0.95,
    )
    return {
        "stage3_champion_median_auc": stage3,
        "pooled_relative_improvement": (pooled - stage3) / max(abs(stage3), 1e-12),
        "bootstrap": bootstrap,
        "pooled_bootstrap_lower_bound": cast(
            Sequence[float],
            bootstrap["pooled"]["interval"],
        )[0],
        "order_deltas": order_deltas,
        "graph_seed_nonnegative_counts": graph_counts,
        "structural_retention": pooled
        / max(abs(float(policies["structural"]["pooled_median_auc"])), 1e-12),
    }


def _validation_health(
    records: Sequence[Mapping[str, Any]],
    *passes: Mapping[str, Any],
) -> dict[str, Any]:
    expected_policies = {"champion", "stage3_champion", "random", "structural"}
    invalid_graphs = 0
    policy_failures = 0
    terminal_failures = 0
    oracle_score_calls = 0
    equal_budgets = True
    for record in records:
        invalid_graphs += int(record.get("invalid_graphs", 0))
        policy_failures += int(record.get("policy_failures", 0))
        oracle_score_calls += int(record.get("oracle_score_calls", 0))
        terminal_failures += record.get("terminal_status") != "completed"
        policies = record.get("policies")
        horizon = int(record.get("horizon", -1))
        expected_evaluations = horizon * len(expected_policies)
        equal_budgets = equal_budgets and (
            isinstance(policies, Mapping)
            and set(policies) == expected_policies
            and int(record.get("selected_score_calls", -1)) == expected_evaluations
            and int(record.get("evaluation_count", -2)) == expected_evaluations
        )
    affinity_ok = True
    for pass_result in passes:
        worker_health = pass_result.get("worker_health")
        if not isinstance(worker_health, Mapping):
            affinity_ok = False
            continue
        assignments = worker_health.get("assignments")
        reserved = worker_health.get("reserved_cpu_ids")
        affinity_ok = affinity_ok and (
            worker_health.get("completed_shards") == 8
            and isinstance(assignments, Sequence)
            and len(assignments) == 8
            and isinstance(reserved, Sequence)
            and len(reserved) == 8
        )
        if isinstance(assignments, Sequence):
            for assignment in assignments:
                affinity_ok = affinity_ok and (
                    isinstance(assignment, Mapping)
                    and assignment.get("observed_affinity") == [assignment.get("cpu_id")]
                    and assignment.get("reserved_cpu_ids") == reserved
                )
    worker_failures = policy_failures + terminal_failures + (0 if affinity_ok else 1)
    return {
        "invalid_graphs": invalid_graphs,
        "worker_failures": worker_failures,
        "selected_plan_only": equal_budgets,
        "oracle_score_calls": oracle_score_calls,
        "equal_budgets": equal_budgets,
        "cpu_affinity_and_reserve_verified": affinity_ok,
    }


def _reduce_validation_records(
    records: Sequence[Mapping[str, Any]],
    health: Mapping[str, Any],
    config: Stage4SearchConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policies = ("champion", "stage3_champion", "random", "structural")
    try:
        summary = summarize_development(
            records,
            policies,
            bootstrap_samples=config.evaluation.bootstrap_samples,
            bootstrap_seed=config.evaluation.bootstrap_seed,
        )
        evidence = _validation_gate_evidence(records, summary, champion="champion")
        return summary, evidence
    except (KeyError, TypeError, ValueError) as error:
        if health["invalid_graphs"] == 0 and health["worker_failures"] == 0:
            raise
        return (
            {
                "policies": {},
                "reduction_error": type(error).__name__,
            },
            {
                "pooled_relative_improvement": -1.0,
                "pooled_bootstrap_lower_bound": -1.0,
                "order_deltas": {"10": -1.0, "12": -1.0},
                "graph_seed_nonnegative_counts": {"10": 0, "12": 0},
                "structural_retention": 0.0,
            },
        )


def validate(
    run: str | Path,
    *,
    workers: int = 8,
    config_path: str | Path = "configs/stage4-search.toml",
    observer: Observer | None = None,
) -> dict[str, Any]:
    """Run the frozen disjoint development-validation pass and terminal gate."""

    if workers != 8:
        raise ValueError("Stage 4 validation workers are frozen at 8")
    config = load_stage4_config(config_path)
    root = Path(run).resolve()
    freeze_value = _load_validation_freeze(config, root)
    if (root / "validation-summary.json").exists():
        raise RuntimeError("Stage 4 validation already has one terminal decision")
    search = _read_json(root / "search-summary.json")
    champion_value = cast(Mapping[str, Any], search["champion"])
    champion = str(champion_value["program_id"])
    archive = ProgramArchive(_archive_root(root))
    champion_record = next(record for record in archive.records() if record.program_id == champion)
    seeds = {seed.slot_id: seed for seed in _seed_sources(config)}
    roster = {
        "champion": _program_source(root, champion_record),
        "stage3-candidate-slot-04": seeds["slot-04"].source,
        **_baseline_sources(config),
    }
    manifest = _load_manifest(config.validation_manifest_path)
    output = root / "evaluations" / "validation"
    primary = evaluate_policy_roster_manifest(
        config,
        manifest,
        roster,
        output_dir=output,
        workers=8,
        shard_count=8,
        pass_name="primary",
        resume=True,
    )
    replay = evaluate_policy_roster_manifest(
        config,
        manifest,
        roster,
        output_dir=output,
        workers=8,
        shard_count=8,
        pass_name="replay",
        resume=True,
    )
    replay_check = verify_replay_pair(primary, replay)
    if replay_check.get("exact") is not True:
        raise RuntimeError("final validation primary/replay mismatch")
    records = _rename_stage3_policy(
        cast(Sequence[Mapping[str, Any]], primary["records"])
    )
    replay_records = _rename_stage3_policy(
        cast(Sequence[Mapping[str, Any]], replay["records"])
    )
    health = _validation_health(records, primary, replay)
    replay_health = _validation_health(replay_records, replay, primary)
    summary, gate_evidence = _reduce_validation_records(records, health, config)
    replay_summary, replay_gate_evidence = _reduce_validation_records(
        replay_records,
        replay_health,
        config,
    )
    statistics_exact = _sha_value(summary) == _sha_value(replay_summary)
    bootstrap_exact = _sha_value(summary.get("policies", {})) == _sha_value(
        replay_summary.get("policies", {})
    )
    gate_inputs_exact = _sha_value(gate_evidence) == _sha_value(replay_gate_evidence)
    health_exact = _sha_value(health) == _sha_value(replay_health)
    archive_report = archive.reindex()
    build_evidence_manifest(root)
    full_summary: dict[str, Any] = {
        **summary,
        **gate_evidence,
        "dependency_import_provenance_heg": (
            freeze_value.get("verified") is True
            and _stage3_checks(config)["ok"]
            and _git_state(config.heg_repo)["commit"] == config.frozen_heg_commit
            and not _git_state(config.heg_repo)["dirty"]
        ),
        "generation_count": 4,
        "initial_turns": search.get("initial_turns"),
        "exact_usage": search.get("exact_usage", False),
        "unauthorized_tool_approval": search.get("unauthorized_tool_approval", False),
        "new_unique_valid_offspring": sum(
            record.generation > 0
            and record.unique
            and record.validation_status == "valid"
            and record.probe_status == "passed"
            and record.smoke_10k_status == "passed"
            and record.replay_status == "verified"
            for record in archive_report.records
        ),
        "champion_distinct": (
            champion_record.normalized_ast_sha256
            not in {
                seed.ast_sha256 for seed in seeds.values()
            }
            | {
                validate_policy(source, config.sandbox).identity.normalized_ast_sha256
                for source in _baseline_sources(config).values()
            }
        ),
        "primary_replay_exact": replay_check.get("exact") is True,
        "primary_replay_records_exact": (
            replay_check.get("canonical_reduction_match") is True
        ),
        "primary_replay_hashes_exact": (
            replay_check.get("primary_sha256") == replay_check.get("replay_sha256")
        ),
        "primary_replay_metrics_exact": (
            replay_check.get("metrics_input_match") is True and statistics_exact
        ),
        "primary_replay_bootstrap_exact": bootstrap_exact,
        "primary_replay_gate_exact": gate_inputs_exact,
        "primary_replay_aggregate_exact": statistics_exact and health_exact,
        **health,
        "archive_lineage_checkpoint_reindex_bounds_rich_json_repository_heg": (
            archive_report.ok
            and verify_evidence_manifest(root)
            and _git_state(config.heg_repo)["commit"] == config.frozen_heg_commit
            and not _git_state(config.heg_repo)["dirty"]
        ),
    }
    gate = gate_report(full_summary, champion="champion")
    result: dict[str, Any] = {
        "schema_version": "stage4.validation.result.v1",
        "status": "completed",
        "run": str(root),
        "validation_freeze_sha256": freeze_value["freeze_sha256"],
        "champion": dict(champion_value),
        "summary": full_summary,
        "replay": replay_check,
        "primary_pass": {key: value for key, value in primary.items() if key != "records"},
        "replay_pass": {key: value for key, value in replay.items() if key != "records"},
        "gate": gate,
        "decision": gate["decision"],
        "final_stage4_validation_results_observed": True,
        "stage5_execution_authorized": False,
    }
    _atomic_json(root / "validation-summary.json", result)
    evidence = build_evidence_manifest(root)
    result["evidence_manifest_sha256"] = evidence["manifest_sha256"]
    if observer:
        observer({"event": "validation_completed", "decision": result["decision"]})
    return canonical_result(result)


def verify_replay(
    run: str | Path,
) -> dict[str, Any]:
    root = Path(run).resolve()
    config = load_stage4_config("configs/stage4-search.toml")
    if root != campaign_root(config).resolve():
        raise ValueError("replay verification requires the frozen official campaign")
    documents = sorted((root / "evaluations").rglob("*-summary.json"))
    validation_summary = root / "validation-summary.json"
    if validation_summary.is_file():
        documents.append(validation_summary)
    results: list[dict[str, Any]] = []
    seen_documents: set[Path] = set()
    for document in documents:
        if document in seen_documents:
            continue
        seen_documents.add(document)
        value = _read_json(document)
        primary = value.get("primary", value.get("primary_pass"))
        replay = value.get("replay_pass")
        if replay is None and isinstance(value.get("replay"), Mapping):
            candidate = value["replay"]
            replay = (
                candidate
                if candidate.get("schema_version") == "stage4.evaluation.v1"
                else None
            )
        if not isinstance(primary, Mapping) or not isinstance(replay, Mapping):
            continue
        manifest = (
            _load_manifest(config.validation_manifest_path)
            if document == validation_summary
            else _load_manifest(config.manifest_path)
        )
        primary_check = verify_candidate_pass(primary, manifest)
        replay_check = verify_candidate_pass(replay, manifest)
        pair = verify_replay_pair(primary, replay)
        results.append(
            {
                "document": str(document.relative_to(root)),
                "primary_pass": primary_check,
                "replay_pass": replay_check,
                **pair,
                "exact": (
                    pair.get("exact") is True
                    and primary_check.get("exact") is True
                    and replay_check.get("exact") is True
                ),
            }
        )
    archive_report = ProgramArchive(_archive_root(root)).reindex()
    evidence_before = verify_evidence_manifest(root)
    exact = (
        bool(results)
        and all(item.get("exact") is True for item in results)
        and archive_report.ok
        and evidence_before
    )
    value = {
        "schema_version": "stage4.replay.audit.v1",
        "status": "completed" if exact else "failed",
        "exact": exact,
        "pairs": results,
        "archive": archive_report.as_dict(),
        "evidence_manifest_verified": evidence_before,
    }
    _atomic_json(root / "replay-audit.json", value)
    build_evidence_manifest(root)
    verify_evidence_manifest(root)
    return canonical_result(value)


__all__ = [
    "archive_inspect",
    "archive_reindex",
    "campaign_root",
    "canonical_result",
    "doctor",
    "evaluate_candidate",
    "evolve",
    "freeze",
    "freeze_validation",
    "resume",
    "validate",
    "verify_replay",
]
