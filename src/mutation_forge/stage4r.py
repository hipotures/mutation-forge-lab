"""Bounded Stage 4R recovery experiment for issue #11."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage4 import commands as stage4
from mutation_forge.stage4.app_server import Stage4AppServerProvider, _codex_transport_schema
from mutation_forge.stage4.archive import (
    ProgramArchive,
    ProgramRecord,
    canonical_bytes,
    deterministic_program_id,
)
from mutation_forge.stage4.artifacts import read_raw_slot_record, write_raw_slot_record
from mutation_forge.stage4.config import Stage4SearchConfig, load_stage4_config
from mutation_forge.stage4.evaluation import evaluate_policy_roster_manifest
from mutation_forge.stage4.generation import (
    SLOTS,
    GenerationConfig,
    GenerationCoordinator,
    GenerationRequest,
    SlotResult,
)
from mutation_forge.stage4.replay import verify_replay as verify_replay_pair
from mutation_forge.stage4.selection import select_parents
from mutation_forge.stage4.statistics import (
    hierarchical_bootstrap,
    select_champion,
    summarize_development,
)

RETAINED_ARCHIVE_SHA256 = "01d9e73e598d2cad952e507654688bb71e2715671c2f63a4e812b7708b3754c6"
VALIDATION_MANIFEST_SHA256 = (
    "87f5b6298e4c312feac2d9c4f6bafea63b70a3b29c0104a0aef33d4b91dcc91e"
)
SEARCH_TAG = "stage4r-search-frozen-v1"
VALIDATION_TAG = "stage4r-validation-frozen-v1"
RECOVERY_GENERATION = 5
AUTH_MODE = "--auth-json ~/.codex/auth.json"
CANARY_BRIEF_SLOT = "slot-00"
SEARCH_FREEZE_SCHEMA = "stage4r.search.freeze.v1"
VALIDATION_FREEZE_SCHEMA = "stage4r.validation.freeze.v1"

CHAMPION_RULE = (
    "pooled search-training median AUC descending",
    "order-10 median AUC descending",
    "median best-total-witness count ascending",
    "normalized AST SHA-256 ascending",
)
SCIENTIFIC_GATES = (
    "champion_is_distinct_stage4_offspring",
    "pooled_relative_improvement_at_least_0_02",
    "pooled_hierarchical_bootstrap_lower_bound_positive",
    "order_10_and_12_median_deltas_nonnegative",
    "three_of_four_graph_seeds_nonnegative_at_each_order",
    "structural_baseline_retention_at_least_0_99",
    "primary_replay_exact_after_timing_removal",
    "evaluation_health_and_no_provider_calls",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_value(value: object) -> str:
    return _sha_bytes(canonical_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_bytes(value) + b"\n")


def _atomic_source(path: Path, source: str) -> None:
    _atomic_bytes(path, source.encode("utf-8"))


def _usage_complete(usage: Mapping[str, Any]) -> bool:
    required = (
        "inputTokens",
        "cachedInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    return (
        usage.get("final") is True
        and usage.get("partial") is False
        and all(
            isinstance(usage.get(name), int)
            and not isinstance(usage.get(name), bool)
            and int(usage[name]) >= 0
            for name in required
        )
    )


def _slot_usage(slot: SlotResult) -> dict[str, int | bool]:
    result: dict[str, int | bool] = {}
    complete = True
    observed = False
    for envelope in (slot.initial, slot.repair or {}):
        usage = envelope.get("usage") if isinstance(envelope, Mapping) else None
        if not isinstance(usage, Mapping) or not usage:
            continue
        observed = True
        complete = complete and _usage_complete(usage)
        for key, value in usage.items():
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and key.lower().endswith("tokens")
            ):
                result[key] = int(result.get(key, 0)) + value
    result["final"] = observed and complete
    result["partial"] = not (observed and complete)
    return result


def _retained_records(retained_run: Path) -> tuple[ProgramRecord, ...]:
    report = ProgramArchive(retained_run / "archive").reindex()
    if not report.ok or report.archive_hash != RETAINED_ARCHIVE_SHA256:
        raise RuntimeError("retained Stage 4 archive does not match issue #11")
    return report.records


def _eligible_retained(records: Sequence[ProgramRecord]) -> tuple[ProgramRecord, ...]:
    result = tuple(
        record
        for record in records
        if record.generation > 0
        and record.unique
        and record.validation_status == "valid"
        and record.probe_status == "passed"
        and record.smoke_10k_status == "passed"
        and record.replay_status == "verified"
        and record.fitness_status == "verified"
    )
    if len(result) != 19:
        raise RuntimeError(f"expected 19 retained eligible offspring, found {len(result)}")
    return result


def _parent_context(
    config: Stage4SearchConfig,
    retained_run: Path,
    records: Sequence[ProgramRecord],
    assignments: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, ProgramRecord], dict[int, dict[str, str]], dict[int, str]]:
    by_id = {record.program_id: record for record in records}
    parent_sources = {
        program_id: stage4._program_source(retained_run, by_id[program_id])
        for program_id in assignments.values()
    }
    parent_records = {program_id: by_id[program_id] for program_id in assignments.values()}
    feedback = {
        0: {
            slot: stage4._compact_parent_feedback(by_id[program_id])
            for slot, program_id in assignments.items()
        }
    }
    archive_context = {0: stage4._bounded_archive_context(records)}
    return parent_sources, parent_records, feedback, archive_context


def _briefs(config: Stage4SearchConfig) -> dict[str, str]:
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(config.briefs_dir.glob("slot-*.md"))
    }


def _coordinator(
    provider: Any,
    config: Stage4SearchConfig,
    retained_run: Path,
    records: Sequence[ProgramRecord],
    assignments: Mapping[str, str],
    *,
    campaign_id: str,
    doctor_sha256: str,
) -> GenerationCoordinator:
    parent_sources, parent_records, feedback, archive_context = _parent_context(
        config,
        retained_run,
        records,
        assignments,
    )
    return GenerationCoordinator(
        provider,
        config=GenerationConfig(
            campaign_id=campaign_id,
            sandbox_limits=config.sandbox,
            model=config.model.name,
            effort=config.model.effort,
            appserver_doctor_sha256=doctor_sha256,
        ),
        briefs=_briefs(config),
        parent_sources=parent_sources,
        parent_records=parent_records,
        search_feedback=feedback,
        archive_context=archive_context,
    )


def _canary_parent(records: Sequence[ProgramRecord]) -> ProgramRecord:
    eligible = _eligible_retained(records)
    selected = select_parents(eligible, slot_count=1)
    if len(selected.parents) != 1:
        raise RuntimeError("could not select one retained Stage 4 canary parent")
    by_id = {record.program_id: record for record in eligible}
    return by_id[selected.parents[0]]


def _write_canary_candidate(
    attempt_root: Path,
    retained_run: Path,
    parent: ProgramRecord,
    slot: SlotResult,
) -> tuple[dict[str, Any], bool, bool]:
    raw_path = write_raw_slot_record(
        attempt_root,
        0,
        0,
        {
            "source": slot.candidate.source if slot.candidate else "",
            "request": dict(slot.request),
            "response": dict(slot.raw_result),
            "transcript": {
                "initial": dict(slot.initial),
                "repair": None,
            },
            "usage": _slot_usage(slot),
            "reference": {
                "diagnostic_only": True,
                "excluded_from_scientific_archive": True,
            },
        },
    )
    raw_readable = read_raw_slot_record(raw_path).get("reference", {}).get(
        "diagnostic_only"
    ) is True
    candidate = slot.candidate
    if candidate is None:
        return {"raw_path": str(raw_path)}, raw_readable, False
    archive = ProgramArchive(attempt_root / "archive")
    retained_source = Path(parent.source_path)
    if not retained_source.is_absolute():
        retained_source = retained_run / retained_source
    archive.append(
        replace(
            parent,
            source_path=str(retained_source.resolve()),
            generation=0,
            slot="parent-reference",
            parent_id=None,
            parent_program_id=None,
            request_id=None,
            app_server_request_id=None,
            thread_id=None,
            app_server_thread_id=None,
            turn_id=None,
            app_server_turn_id=None,
            mutation_brief_id=None,
            seed_id=parent.program_id,
            generation_mode="stage4r-canary-parent-reference",
        )
    )
    program_id = deterministic_program_id(
        generation=1,
        slot=slot.slot,
        source_sha256=candidate.source_sha256,
        parent_id=parent.program_id,
    )
    source_path = attempt_root / "sources" / f"{program_id}.py"
    _atomic_source(source_path, candidate.source)
    request_id = slot.request.get("idempotency_key")
    archive.append(
        ProgramRecord(
            program_id=program_id,
            source_path=source_path.relative_to(attempt_root).as_posix(),
            source_sha256=candidate.source_sha256,
            normalized_ast_sha256=candidate.normalized_ast_sha256,
            behavior_signature=cast(Mapping[str, JsonValue], candidate.behavior_signature),
            generation=1,
            slot=slot.slot,
            parent_id=parent.program_id,
            mutation_brief_id=cast(str | None, slot.request.get("brief_id")),
            request_id=str(request_id) if request_id is not None else None,
            thread_id=candidate.thread_id,
            turn_id=candidate.turn_id,
            usage=cast(Mapping[str, JsonValue], dict(candidate.usage)),
            validation_status="valid",
            probe_status="passed",
            smoke_10k_status="passed",
            replay_status="diagnostic_only",
            fitness_status="excluded",
            seed_id=parent.seed_id,
            generation_mode="stage4r-canary-diagnostic",
            metadata={
                "diagnostic_only": True,
                "excluded_from_scientific_archive": True,
            },
        )
    )
    reindex = archive.reindex()
    artifact = {
        "raw_path": str(raw_path),
        "archive_root": str(archive.root),
        "archive_sha256": reindex.archive_hash,
        "program_id": program_id,
        "source_path": str(source_path),
        "source_sha256": candidate.source_sha256,
        "normalized_ast_sha256": candidate.normalized_ast_sha256,
    }
    return artifact, raw_readable and source_path.read_text(encoding="utf-8") == candidate.source, (
        reindex.ok and len(reindex.records) == 2
    )


def canary(
    *,
    config_path: str | Path,
    retained_run: str | Path,
    run: str | Path,
    auth_json: str | Path,
    attempt: int,
    provider: Any | None = None,
    doctor_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run exactly one diagnostic Stage 4R turn with no repair or automatic retry."""

    if attempt not in {1, 2, 3}:
        raise ValueError("canary attempt must be 1, 2, or 3")
    root = Path(run).resolve()
    attempt_root = root / "canary" / f"attempt-{attempt:02d}"
    summary_path = attempt_root / "canary-summary.json"
    if summary_path.exists() or (root / "canary-success.json").exists():
        raise RuntimeError("canary attempt is already terminal")
    prior = sorted((root / "canary").glob("attempt-*/canary-summary.json"))
    if len(prior) != attempt - 1:
        raise RuntimeError("canary attempts must be sequential")
    accepted_prior = sum(
        _read_json(path).get("checks", {}).get("terminal_completed") is True
        for path in prior
    )
    if accepted_prior >= 3:
        raise RuntimeError("three accepted canary turns have already been used")
    config = load_stage4_config(config_path)
    retained = Path(retained_run).resolve()
    records = _retained_records(retained)
    parent = _canary_parent(records)
    doctor = dict(
        doctor_result
        if doctor_result is not None
        else stage4.doctor(
            config_path,
            auth_json=auth_json,
            check_auth=True,
            write=False,
        )
    )
    doctor_ready = (
        doctor.get("status") == "completed"
        and doctor.get("decision") == "READY"
        and isinstance(doctor.get("auth"), Mapping)
        and cast(Mapping[str, Any], doctor["auth"]).get("authenticated") is True
    )
    doctor_sha256 = _sha_value(doctor)
    result: SlotResult | None = None
    if doctor_ready:
        live_provider = provider or Stage4AppServerProvider(
            auth_json=auth_json,
            artifact_dir=attempt_root / "appserver",
            artifact_root=attempt_root,
            artifact_max_bytes=config.limits.artifact_compressed_campaign_bytes,
        )
        coordinator = _coordinator(
            live_provider,
            config,
            retained,
            records,
            {CANARY_BRIEF_SLOT: parent.program_id},
            campaign_id="stage4r-issue-11-canary",
            doctor_sha256=doctor_sha256,
        )
        result = coordinator.run_slot(
            0,
            CANARY_BRIEF_SLOT,
            parent.program_id,
            allow_repair=False,
            allow_infrastructure_retry=False,
        )
    initial = result.initial if result is not None else {}
    usage = initial.get("usage") if isinstance(initial, Mapping) else None
    checks: dict[str, bool] = {
        "authenticated_and_initialized": doctor_ready,
        "terminal_completed": (
            result is not None
            and initial.get("status") == "completed"
            and initial.get("accepted") is True
            and bool(initial.get("thread_id"))
            and bool(initial.get("turn_id"))
        ),
        "structured_content_nonempty": (
            result is not None
            and initial.get("content") is True
            and initial.get("response") not in (None, "", {}, [])
        ),
        "exact_server_usage": isinstance(usage, Mapping) and _usage_complete(usage),
        "parsed": result is not None and result.candidate is not None,
        "validator_and_probe_passed": result is not None and result.candidate is not None,
        "artifact_written_and_readable": False,
        "artifact_reindexed": False,
    }
    artifact: dict[str, Any] = {}
    if result is not None:
        artifact, readable, reindexed = _write_canary_candidate(
            attempt_root,
            retained,
            parent,
            result,
        )
        checks["artifact_written_and_readable"] = readable and result.candidate is not None
        checks["artifact_reindexed"] = reindexed and result.candidate is not None
    passed = all(checks.values())
    summary: dict[str, Any] = {
        "schema_version": "stage4r.canary.v1",
        "status": "completed" if passed else "failed",
        "passed": passed,
        "attempt": attempt,
        "concurrency": 1,
        "accepted_canary_turns_through_attempt": accepted_prior
        + int(checks["terminal_completed"]),
        "model": config.model.name,
        "effort": config.model.effort,
        "auth_mode": AUTH_MODE,
        "doctor_sha256": doctor_sha256,
        "parent": {
            "program_id": parent.program_id,
            "source_sha256": parent.source_sha256,
            "normalized_ast_sha256": parent.normalized_ast_sha256,
        },
        "brief": {
            "slot": CANARY_BRIEF_SLOT,
            "path": str(config.briefs_dir / f"{CANARY_BRIEF_SLOT}.md"),
            "sha256": _sha_file(config.briefs_dir / f"{CANARY_BRIEF_SLOT}.md"),
        },
        "checks": checks,
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
        "request": (
            {
                "idempotency_key": result.request.get("idempotency_key"),
                "prompt_hash": result.request.get("prompt_hash"),
            }
            if result is not None
            else {}
        ),
        "turn": (
            {
                "status": initial.get("status"),
                "accepted": initial.get("accepted"),
                "content": initial.get("content"),
                "thread_id": initial.get("thread_id"),
                "turn_id": initial.get("turn_id"),
                "error": initial.get("error"),
            }
            if result is not None
            else {}
        ),
        "candidate": artifact,
        "excluded_from_scientific_archive": True,
    }
    _atomic_json(summary_path, summary)
    if passed:
        _atomic_json(root / "canary-success.json", summary)
    return summary


class _NoCallProvider:
    def generate(self, request: Mapping[str, Any]) -> Any:
        raise AssertionError(f"provider call is forbidden while freezing {request.get('slot')}")


def _freeze_projection(
    config: Stage4SearchConfig,
    retained_run: Path,
    canary_value: Mapping[str, Any],
    records: Sequence[ProgramRecord],
    requests: Sequence[GenerationRequest],
) -> dict[str, Any]:
    raw_schema = json.loads(config.output_schema_path.read_text(encoding="utf-8"))
    if not isinstance(raw_schema, Mapping):
        raise ValueError("Stage 4 output schema must be an object")
    transport_schema = _codex_transport_schema(raw_schema)
    request_projections = []
    for request in requests:
        request_value = request.as_dict()
        request_projections.append(
            {
                "slot": request.slot,
                "parent_id": request.parent_id,
                "parent_source_sha256": _sha_bytes(request.parent_source.encode("utf-8")),
                "brief_id": request.brief_id,
                "prompt_sha256": request.prompt_hash,
                "request_sha256": _sha_value(request_value),
                "idempotency_key": request.idempotency_key,
            }
        )
    projection: dict[str, Any] = {
        "schema_version": SEARCH_FREEZE_SCHEMA,
        "issue": 11,
        "results_observed": False,
        "canary": {
            "passed": canary_value.get("passed"),
            "attempt": canary_value.get("attempt"),
            "summary_sha256": _sha_value(canary_value),
            "candidate_excluded": True,
            "doctor_sha256": canary_value.get("doctor_sha256"),
        },
        "retained_campaign": {
            "path": str(retained_run),
            "archive_sha256": RETAINED_ARCHIVE_SHA256,
            "eligible_offspring": len(_eligible_retained(records)),
        },
        "recovery_generation": {
            "generation_count": 1,
            "initial_turns": 8,
            "slots": list(SLOTS),
            "concurrency": 8,
            "max_contract_repairs_per_slot": 1,
            "replacement_turns": 0,
            "model_calls_after_batch": 0,
        },
        "requests": request_projections,
        "briefs": {
            path.stem: _sha_file(path)
            for path in sorted(config.briefs_dir.glob("slot-*.md"))
        },
        "prompt_files": {
            "system": _sha_file(config.system_prompt_path),
            "request": _sha_file(config.request_prompt_path),
            "repair": _sha_file(config.repair_prompt_path),
        },
        "model": {
            "name": config.model.name,
            "effort": config.model.effort,
            "auth_mode": AUTH_MODE,
        },
        "transport_schema": {
            "frozen_sha256": _sha_value(raw_schema),
            "projected_sha256": _sha_value(transport_schema),
            "projection": "schema_version_const_adds_type_string",
        },
        "manifests": {
            "search_training_path": str(config.manifest_path),
            "search_training_sha256": _sha_file(config.manifest_path),
            "final_validation_path": str(config.validation_manifest_path),
            "final_validation_sha256": _sha_file(config.validation_manifest_path),
        },
        "champion_rule": list(CHAMPION_RULE),
        "scientific_gates": list(SCIENTIFIC_GATES),
        "statistics": {
            "bootstrap_samples": config.evaluation.bootstrap_samples,
            "bootstrap_seed": config.evaluation.bootstrap_seed,
            "confidence_level": config.evaluation.confidence_level,
            "statistics_sha256": _sha_file(Path(stage4.__file__).with_name("statistics.py")),
        },
    }
    projection["freeze_sha256"] = _sha_value(projection)
    return projection


def freeze_search(
    *,
    config_path: str | Path,
    retained_run: str | Path,
    run: str | Path,
    tracked_freeze: str | Path,
) -> dict[str, Any]:
    """Freeze the exact eight requests after, and only after, a successful canary."""

    root = Path(run).resolve()
    artifact_path = root / "search-freeze.json"
    tracked_path = Path(tracked_freeze).resolve()
    if artifact_path.exists() or tracked_path.exists():
        raise RuntimeError("Stage 4R search is already frozen")
    canary_value = _read_json(root / "canary-success.json")
    if canary_value.get("passed") is not True or not all(
        bool(value) for value in cast(Mapping[str, Any], canary_value.get("checks", {})).values()
    ):
        raise RuntimeError("a successful eight-condition canary is required")
    config = load_stage4_config(config_path)
    if _sha_file(config.validation_manifest_path) != VALIDATION_MANIFEST_SHA256:
        raise RuntimeError("final-validation manifest hash drifted before search freeze")
    retained = Path(retained_run).resolve()
    records = _retained_records(retained)
    selection = select_parents(records)
    assignments = dict(selection.slots)
    if tuple(assignments) != SLOTS:
        raise RuntimeError("Stage 4R requires eight ordered parent assignments")
    coordinator = _coordinator(
        _NoCallProvider(),
        config,
        retained,
        records,
        assignments,
        campaign_id="stage4r-issue-11-recovery-v1",
        doctor_sha256=str(canary_value["doctor_sha256"]),
    )
    requests = tuple(
        coordinator.build_request(0, slot, assignments[slot])
        for slot in SLOTS
    )
    projection = _freeze_projection(config, retained, canary_value, records, requests)
    artifact = {
        "schema_version": "stage4r.search.freeze.artifact.v1",
        "freeze": projection,
        "private_requests": [request.as_dict() for request in requests],
    }
    _atomic_json(artifact_path, artifact)
    _atomic_json(tracked_path, projection)
    return {
        "schema_version": "stage4r.search.freeze.result.v1",
        "status": "completed",
        "run": str(root),
        "tracked_freeze": str(tracked_path),
        **projection,
    }


def _load_search_freeze(
    run: Path,
    tracked_freeze: Path,
) -> tuple[dict[str, Any], tuple[GenerationRequest, ...]]:
    projection = _read_json(tracked_freeze)
    artifact = _read_json(run / "search-freeze.json")
    if (
        projection.get("schema_version") != SEARCH_FREEZE_SCHEMA
        or projection.get("freeze_sha256")
        != _sha_value({key: value for key, value in projection.items() if key != "freeze_sha256"})
        or artifact.get("freeze") != projection
        or projection.get("results_observed") is not False
    ):
        raise RuntimeError("Stage 4R search freeze is invalid")
    raw_requests = artifact.get("private_requests")
    if not isinstance(raw_requests, list) or len(raw_requests) != 8:
        raise RuntimeError("Stage 4R frozen requests are missing")
    requests = tuple(
        GenerationRequest.from_value(cast(Mapping[str, Any], value))
        for value in raw_requests
        if isinstance(value, Mapping)
    )
    if len(requests) != 8 or tuple(request.slot for request in requests) != SLOTS:
        raise RuntimeError("Stage 4R frozen request order drifted")
    projected = projection.get("requests")
    if not isinstance(projected, list):
        raise RuntimeError("Stage 4R request projection is missing")
    for request, item in zip(requests, projected, strict=True):
        if (
            not isinstance(item, Mapping)
            or item.get("request_sha256") != _sha_value(request.as_dict())
            or item.get("idempotency_key") != request.idempotency_key
        ):
            raise RuntimeError("Stage 4R frozen request identity drifted")
    return projection, requests


def _require_tagged_clean(config: Stage4SearchConfig, tag_name: str) -> dict[str, Any]:
    project = stage4._git_state(config.project_repo)
    heg = stage4._git_state(config.heg_repo)
    tag = stage4._tag_state(config.project_repo, tag_name)
    if (
        project["dirty"]
        or project["branch"] != "agent/stage4-salvage"
        or tag["type"] != "tag"
        or tag["commit"] != project["commit"]
        or heg["dirty"]
        or heg["commit"] != config.frozen_heg_commit
    ):
        raise RuntimeError(f"{tag_name} must annotate the current clean recovery branch")
    return {"project": project, "heg": heg, "tag": tag}


def _slot_result_path(root: Path, slot: str) -> Path:
    return root / "recovery" / "slots" / f"{slot}.json"


def _write_recovery_raw(root: Path, slot: SlotResult) -> None:
    write_raw_slot_record(
        root,
        1,
        int(slot.slot[-2:]),
        {
            "source": slot.candidate.source if slot.candidate else "",
            "request": dict(slot.request),
            "response": dict(slot.raw_result),
            "transcript": {
                "initial": dict(slot.initial),
                "repair": dict(slot.repair) if slot.repair else None,
            },
            "usage": _slot_usage(slot),
            "reference": {
                "stage4r_generation": 1,
                "slot": slot.slot,
                "parent_id": slot.parent_id,
                "status": slot.status,
            },
        },
    )


def _run_recovery_batch(
    config: Stage4SearchConfig,
    retained_run: Path,
    root: Path,
    freeze_value: Mapping[str, Any],
    requests: Sequence[GenerationRequest],
    auth_json: str | Path,
    records: Sequence[ProgramRecord],
    provider: Any | None,
) -> tuple[SlotResult, ...]:
    summary_path = root / "recovery" / "batch-summary.json"
    assignments = {request.slot: request.parent_id for request in requests}
    live_provider = provider or Stage4AppServerProvider(
        auth_json=auth_json,
        artifact_dir=root / "appserver",
        artifact_root=root,
        artifact_max_bytes=config.limits.artifact_compressed_campaign_bytes,
    )
    coordinator = _coordinator(
        live_provider,
        config,
        retained_run,
        records,
        assignments,
        campaign_id="stage4r-issue-11-recovery-v1",
        doctor_sha256=str(
            cast(Mapping[str, Any], freeze_value["canary"])["doctor_sha256"]
        ),
    )
    if summary_path.exists():
        return tuple(
            coordinator.load_slot_result(_read_json(_slot_result_path(root, slot)))
            for slot in SLOTS
        )
    existing = {
        slot: coordinator.load_slot_result(_read_json(_slot_result_path(root, slot)))
        for slot in SLOTS
        if _slot_result_path(root, slot).is_file()
    }
    pending = [request for request in requests if request.slot not in existing]
    futures: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="stage4r-recovery") as pool:
        for request in pending:
            retained_result = None
            loader = getattr(live_provider, "load_retained_result", None)
            if callable(loader):
                retained_result = loader(request.as_dict())
            futures[
                pool.submit(
                    coordinator.run_request,
                    request,
                    allow_repair=True,
                    allow_infrastructure_retry=True,
                    retained_result=retained_result,
                )
            ] = request.slot
        for future in as_completed(futures):
            slot = futures[future]
            result = future.result()
            if result.slot != slot:
                raise RuntimeError("Stage 4R provider returned a mismatched slot")
            _atomic_json(_slot_result_path(root, slot), result.as_dict())
            _write_recovery_raw(root, result)
            existing[slot] = result
    if set(existing) != set(SLOTS):
        raise RuntimeError("Stage 4R recovery batch did not retain all eight slots")
    ordered = tuple(existing[slot] for slot in SLOTS)
    summary = {
        "schema_version": "stage4r.recovery.batch.v1",
        "status": "completed",
        "generation_count": 1,
        "ordered_initial_slots": list(SLOTS),
        "initial_turns": 8,
        "concurrency": 8,
        "repair_turns": sum(slot.repairs for slot in ordered),
        "accepted_turns": sum(
            envelope.get("accepted") is True
            for slot in ordered
            for envelope in (slot.initial, slot.repair or {})
            if isinstance(envelope, Mapping)
        ),
        "candidate_count": sum(slot.candidate is not None for slot in ordered),
        "failures": sum(slot.candidate is None for slot in ordered),
        "model_calls_closed": True,
        "no_model_calls_after_batch": True,
        "slots": [
            {
                "slot": slot.slot,
                "status": slot.status,
                "parent_id": slot.parent_id,
                "candidate": slot.candidate is not None,
                "repairs": slot.repairs,
                "usage": _slot_usage(slot),
            }
            for slot in ordered
        ],
    }
    _atomic_json(summary_path, summary)
    return ordered


def _record_source_path(root: Path, program_id: str) -> Path:
    return root / "recovery" / "sources" / f"{program_id}.py"


def _recovery_records(
    config: Stage4SearchConfig,
    retained_run: Path,
    root: Path,
    retained: Sequence[ProgramRecord],
    slots: Sequence[SlotResult],
) -> tuple[ProgramRecord, ...]:
    baselines = stage4._baseline_sources(config)
    seen = {
        record.normalized_ast_sha256: record.program_id
        for record in retained
        if record.unique
    }
    for name, source in baselines.items():
        ast = validate_policy(source, config.sandbox).identity.normalized_ast_sha256
        if ast:
            seen[ast] = name
    records: list[ProgramRecord] = []
    for slot in slots:
        candidate = slot.candidate
        if candidate is None:
            continue
        program_id = deterministic_program_id(
            generation=RECOVERY_GENERATION,
            slot=slot.slot,
            source_sha256=candidate.source_sha256,
            parent_id=slot.parent_id,
        )
        source_path = _record_source_path(root, program_id)
        _atomic_source(source_path, candidate.source)
        duplicate_of = seen.get(candidate.normalized_ast_sha256)
        if duplicate_of is None:
            seen[candidate.normalized_ast_sha256] = program_id
        request = next(
            request
            for request in cast(
                Sequence[Mapping[str, Any]],
                _read_json(root / "search-freeze.json")["private_requests"],
            )
            if request.get("slot") == slot.slot
        )
        record = ProgramRecord(
            program_id=program_id,
            source_path=source_path.relative_to(root).as_posix(),
            source_sha256=candidate.source_sha256,
            normalized_ast_sha256=candidate.normalized_ast_sha256,
            behavior_signature=cast(Mapping[str, JsonValue], candidate.behavior_signature),
            generation=RECOVERY_GENERATION,
            slot=slot.slot,
            parent_id=slot.parent_id,
            mutation_brief_id=cast(str | None, request.get("brief_id")),
            request_id=cast(str, request["idempotency_key"]),
            thread_id=candidate.thread_id,
            turn_id=candidate.turn_id,
            usage=cast(Mapping[str, JsonValue], _slot_usage(slot)),
            validation_status="valid",
            probe_status="passed",
            smoke_10k_status="passed",
            replay_status="duplicate" if duplicate_of else "pending",
            duplicate_of=duplicate_of,
            fitness_status="duplicate" if duplicate_of else "pending",
            seed_id=next(
                (
                    item.seed_id
                    for item in retained
                    if item.program_id == slot.parent_id
                ),
                None,
            ),
            generation_mode="stage4r-recovery",
            metadata={
                "repairs": slot.repairs,
                "initial_turn_accepted": slot.initial.get("accepted") is True,
                "usage_complete": _usage_complete(candidate.usage),
                "scientific_archive": True,
            },
        )
        _atomic_json(root / "recovery" / "records" / f"{slot.slot}.json", record.as_dict())
        records.append(record)
    return tuple(records)


def _evaluate_new_records(
    config: Stage4SearchConfig,
    root: Path,
    records: Sequence[ProgramRecord],
) -> tuple[ProgramRecord, ...]:
    eligible = [record for record in records if record.unique]
    if not eligible:
        return tuple(records)
    baselines = stage4._baseline_sources(config)
    roster = {
        **baselines,
        **{
            record.program_id: (root / record.source_path).read_text(encoding="utf-8")
            for record in eligible
        },
    }
    output = root / "evaluations" / "search-recovery"
    manifest = stage4._load_manifest(config.manifest_path)
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
        raise RuntimeError("Stage 4R search-training primary/replay mismatch")
    primary_records = cast(Sequence[Mapping[str, Any]], primary["records"])
    summary = summarize_development(
        primary_records,
        roster,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
    )
    updated: list[ProgramRecord] = []
    for record in records:
        if not record.unique:
            updated.append(record)
            continue
        valid = stage4._policy_evaluation_valid(primary_records, record.program_id)
        value = replace(
            record,
            search_metrics=stage4._summary_metrics(summary, record.program_id),
            replay_status="verified" if valid else "failed",
            fitness_status="verified" if valid else "failed",
        )
        _atomic_json(root / "recovery" / "records" / f"{record.slot}.json", value.as_dict())
        updated.append(value)
    _atomic_json(
        output / "generation-summary.json",
        {
            "schema_version": "stage4r.search_evaluation.v1",
            "summary": summary,
            "primary": {key: value for key, value in primary.items() if key != "records"},
            "replay": {key: value for key, value in replay.items() if key != "records"},
            "replay_check": replay_check,
            "new_candidate_ids": [record.program_id for record in eligible],
        },
    )
    return tuple(updated)


def _selection_key(record: ProgramRecord) -> dict[str, Any]:
    metrics = record.search_metrics
    by_order = metrics.get("by_order", {}) if isinstance(metrics, Mapping) else {}
    order_10 = by_order.get("10", {}) if isinstance(by_order, Mapping) else {}
    return {
        "program_id": record.program_id,
        "pooled_median_auc": metrics.get("pooled_median_auc"),
        "order_10_median_auc": (
            order_10.get("median_auc") if isinstance(order_10, Mapping) else None
        ),
        "median_best_total_witness": metrics.get(
            "pooled_median_best_total_witness",
            metrics.get("pooled_median_best_witnesses"),
        ),
        "normalized_ast_sha256": record.normalized_ast_sha256,
        "source_sha256": record.source_sha256,
        "generation": record.generation,
    }


def _select_recovery_champion(
    config: Stage4SearchConfig,
    retained_run: Path,
    root: Path,
    retained: Sequence[ProgramRecord],
    recovery: Sequence[ProgramRecord],
) -> tuple[ProgramRecord, dict[str, Any]]:
    candidates = [
        *list(_eligible_retained(retained)),
        *[
            record
            for record in recovery
            if record.unique and record.fitness_status == "verified"
        ],
    ]
    policies = {record.program_id: dict(record.search_metrics) for record in candidates}
    identities = {
        record.program_id: {
            "normalized_ast_sha256": record.normalized_ast_sha256,
            "origin": "stage4",
            "generation": record.generation,
            "is_stage4": True,
        }
        for record in candidates
    }
    seeds = stage4._seed_sources(config)
    baselines = stage4._baseline_sources(config)
    champion_id = select_champion(
        {"policies": policies},
        identities,
        generation=RECOVERY_GENERATION,
        seed_ast_hashes=[seed.ast_sha256 for seed in seeds],
        baseline_ast_hashes=[
            validate_policy(source, config.sandbox).identity.normalized_ast_sha256 or ""
            for source in baselines.values()
        ],
    )
    champion = next(
        (record for record in candidates if record.program_id == champion_id),
        None,
    )
    if champion is None:
        raise RuntimeError("Stage 4R champion selection produced no eligible program")
    source_root = root if champion.generation == RECOVERY_GENERATION else retained_run
    source_path = Path(champion.source_path)
    resolved_source = source_path if source_path.is_absolute() else source_root / source_path
    value = {
        "program_id": champion.program_id,
        "source_sha256": champion.source_sha256,
        "normalized_ast_sha256": champion.normalized_ast_sha256,
        "generation": champion.generation,
        "slot": champion.slot,
        "origin": "stage4r-recovery"
        if champion.generation == RECOVERY_GENERATION
        else "retained-stage4",
        "source_reference": str(resolved_source.resolve()),
        "selection_key": _selection_key(champion),
    }
    return champion, value


def recover(
    *,
    config_path: str | Path,
    retained_run: str | Path,
    run: str | Path,
    tracked_freeze: str | Path,
    auth_json: str | Path,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Run one eight-slot recovery generation, then only local deterministic evaluation."""

    root = Path(run).resolve()
    summary_path = root / "search-summary.json"
    if summary_path.exists():
        return _read_json(summary_path)
    config = load_stage4_config(config_path)
    _require_tagged_clean(config, SEARCH_TAG)
    freeze_value, requests = _load_search_freeze(root, Path(tracked_freeze).resolve())
    retained_root = Path(retained_run).resolve()
    retained = _retained_records(retained_root)
    slots = _run_recovery_batch(
        config,
        retained_root,
        root,
        freeze_value,
        requests,
        auth_json,
        retained,
        provider,
    )
    recovery_records = _recovery_records(config, retained_root, root, retained, slots)
    recovery_records = _evaluate_new_records(config, root, recovery_records)
    champion, champion_value = _select_recovery_champion(
        config,
        retained_root,
        root,
        retained,
        recovery_records,
    )
    candidates = [
        *list(_eligible_retained(retained)),
        *[
            record
            for record in recovery_records
            if record.unique and record.fitness_status == "verified"
        ],
    ]
    result: dict[str, Any] = {
        "schema_version": "stage4r.search.result.v1",
        "status": "completed",
        "decision": "PENDING_FINAL_VALIDATION",
        "run": str(root),
        "search_freeze_sha256": freeze_value["freeze_sha256"],
        "retained_eligible_offspring": 19,
        "recovery_yield": {
            "initial_slots": 8,
            "candidate_slots": sum(slot.candidate is not None for slot in slots),
            "failed_slots": sum(slot.candidate is None for slot in slots),
            "duplicate_candidates": sum(
                record.duplicate_of is not None for record in recovery_records
            ),
            "new_unique_valid": sum(
                record.unique and record.fitness_status == "verified"
                for record in recovery_records
            ),
            "repair_turns": sum(slot.repairs for slot in slots),
        },
        "champion": champion_value,
        "champion_source_sha256_verified": (
            _sha_file(Path(cast(str, champion_value["source_reference"])))
            == champion.source_sha256
        ),
        "selection_rule": list(CHAMPION_RULE),
        "selection_inputs": [
            _selection_key(record)
            for record in sorted(candidates, key=lambda item: item.program_id)
        ],
        "canary_excluded": True,
        "stage3_seeds_excluded": True,
        "baselines_excluded": True,
        "failed_and_duplicates_excluded": True,
        "model_calls_closed": True,
    }
    _atomic_json(summary_path, result)
    return result


def freeze_validation(
    *,
    config_path: str | Path,
    run: str | Path,
    tracked_search_freeze: str | Path,
    tracked_validation_freeze: str | Path,
) -> dict[str, Any]:
    """Persist the champion and unseen final-validation identity before evaluation."""

    root = Path(run).resolve()
    artifact_path = root / "validation-freeze.json"
    tracked_path = Path(tracked_validation_freeze).resolve()
    if artifact_path.exists() or tracked_path.exists():
        raise RuntimeError("Stage 4R final validation is already frozen")
    config = load_stage4_config(config_path)
    search_freeze, _ = _load_search_freeze(
        root,
        Path(tracked_search_freeze).resolve(),
    )
    search = _read_json(root / "search-summary.json")
    champion = search.get("champion")
    if (
        search.get("decision") != "PENDING_FINAL_VALIDATION"
        or not isinstance(champion, Mapping)
        or search.get("model_calls_closed") is not True
    ):
        raise RuntimeError("Stage 4R champion is not frozen for final validation")
    manifest_sha256 = _sha_file(config.validation_manifest_path)
    if manifest_sha256 != VALIDATION_MANIFEST_SHA256:
        raise RuntimeError("final-validation manifest SHA-256 mismatch")
    payload: dict[str, Any] = {
        "schema_version": VALIDATION_FREEZE_SCHEMA,
        "issue": 11,
        "final_validation_results_observed": False,
        "champion": dict(champion),
        "search_summary_sha256": _sha_file(root / "search-summary.json"),
        "search_freeze_sha256": search_freeze["freeze_sha256"],
        "final_validation_manifest": {
            "path": str(config.validation_manifest_path),
            "sha256": manifest_sha256,
        },
        "roster": [
            "stage4r_champion",
            "stage3-candidate-slot-04",
            "random",
            "structural",
        ],
        "passes": ["primary", "replay"],
        "workers": 8,
        "shards": 8,
        "scientific_gates": list(SCIENTIFIC_GATES),
        "statistics_sha256": _sha_file(Path(stage4.__file__).with_name("statistics.py")),
    }
    payload["freeze_sha256"] = _sha_value(payload)
    _atomic_json(artifact_path, payload)
    _atomic_json(tracked_path, payload)
    return {
        "schema_version": "stage4r.validation.freeze.result.v1",
        "status": "completed",
        "run": str(root),
        "tracked_freeze": str(tracked_path),
        **payload,
    }


def _load_validation_freeze(
    config: Stage4SearchConfig,
    root: Path,
    tracked_path: Path,
) -> dict[str, Any]:
    tracked = _read_json(tracked_path)
    artifact = _read_json(root / "validation-freeze.json")
    if (
        tracked != artifact
        or tracked.get("schema_version") != VALIDATION_FREEZE_SCHEMA
        or tracked.get("freeze_sha256")
        != _sha_value({key: value for key, value in tracked.items() if key != "freeze_sha256"})
        or tracked.get("final_validation_results_observed") is not False
        or _sha_file(config.validation_manifest_path) != VALIDATION_MANIFEST_SHA256
        or tracked.get("search_summary_sha256")
        != _sha_file(root / "search-summary.json")
    ):
        raise RuntimeError("Stage 4R validation freeze is invalid")
    return tracked


def _rename_stage3(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return stage4._rename_stage3_policy(records)


def _validation_evidence(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    policies = cast(Mapping[str, Mapping[str, Any]], summary["policies"])
    champion_metrics = policies["champion"]
    stage3_metrics = policies["stage3_champion"]
    pooled = float(champion_metrics["pooled_median_auc"])
    stage3_auc = float(stage3_metrics["pooled_median_auc"])
    by_order_champion = cast(Mapping[str, Mapping[str, Any]], champion_metrics["by_order"])
    by_order_stage3 = cast(Mapping[str, Mapping[str, Any]], stage3_metrics["by_order"])
    order_deltas = {
        order: float(by_order_champion[order]["median_auc"])
        - float(by_order_stage3[order]["median_auc"])
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
                policy_rows = cast(Mapping[str, Mapping[str, Any]], record["policies"])
                champion_curve = cast(
                    Sequence[float],
                    policy_rows["champion"]["normalized_best_so_far_curve"],
                )
                stage3_curve = cast(
                    Sequence[float],
                    policy_rows["stage3_champion"]["normalized_best_so_far_curve"],
                )
                deltas.append(
                    sum(float(value) for value in champion_curve) / len(champion_curve)
                    - sum(float(value) for value in stage3_curve) / len(stage3_curve)
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
        policy="champion",
        baseline="stage3_champion",
        samples=10_000,
        seed=2026073004,
        confidence=0.95,
    )
    structural_auc = float(policies["structural"]["pooled_median_auc"])
    return {
        "stage3_champion_median_auc": stage3_auc,
        "pooled_relative_improvement": (pooled - stage3_auc)
        / max(abs(stage3_auc), 1e-12),
        "bootstrap": bootstrap,
        "pooled_bootstrap_lower_bound": cast(
            Sequence[float],
            cast(Mapping[str, Any], bootstrap["pooled"])["interval"],
        )[0],
        "order_deltas": order_deltas,
        "graph_seed_nonnegative_counts": graph_counts,
        "structural_retention": pooled / max(abs(structural_auc), 1e-12),
    }


def _scientific_gate(
    *,
    champion_distinct: bool,
    evidence: Mapping[str, Any],
    replay: Mapping[str, Any],
    primary_summary: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    primary_evidence: Mapping[str, Any],
    replay_evidence: Mapping[str, Any],
    health: Mapping[str, Any],
    replay_health: Mapping[str, Any],
) -> dict[str, Any]:
    order_deltas = cast(Mapping[str, Any], evidence["order_deltas"])
    graph_counts = cast(Mapping[str, Any], evidence["graph_seed_nonnegative_counts"])
    replay_exact = (
        replay.get("exact") is True
        and replay.get("canonical_reduction_match") is True
        and replay.get("metrics_input_match") is True
        and _sha_value(primary_summary) == _sha_value(replay_summary)
        and _sha_value(primary_evidence) == _sha_value(replay_evidence)
        and _sha_value(health) == _sha_value(replay_health)
    )
    health_ok = (
        health.get("invalid_graphs") == 0
        and health.get("worker_failures") == 0
        and health.get("selected_plan_only") is True
        and health.get("oracle_score_calls") == 0
        and health.get("provider_calls") == 0
        and health.get("equal_budgets") is True
        and replay_health == health
    )
    checks = {
        SCIENTIFIC_GATES[0]: champion_distinct,
        SCIENTIFIC_GATES[1]: float(evidence["pooled_relative_improvement"]) >= 0.02,
        SCIENTIFIC_GATES[2]: float(evidence["pooled_bootstrap_lower_bound"]) > 0.0,
        SCIENTIFIC_GATES[3]: all(
            float(order_deltas[order]) >= 0.0 for order in ("10", "12")
        ),
        SCIENTIFIC_GATES[4]: all(
            int(graph_counts[order]) >= 3 for order in ("10", "12")
        ),
        SCIENTIFIC_GATES[5]: float(evidence["structural_retention"]) >= 0.99,
        SCIENTIFIC_GATES[6]: replay_exact,
        SCIENTIFIC_GATES[7]: health_ok,
    }
    decision = "GO_TO_STAGE_5" if all(checks.values()) else "NO_GO"
    value: dict[str, Any] = {"checks": checks, "decision": decision}
    value["canonical_sha256"] = _sha_value(value)
    return value


def validate(
    *,
    config_path: str | Path,
    retained_run: str | Path,
    run: str | Path,
    tracked_validation_freeze: str | Path,
) -> dict[str, Any]:
    """Run one primary and one replay final comparison with no model provider."""

    root = Path(run).resolve()
    summary_path = root / "validation-summary.json"
    if summary_path.exists():
        return _read_json(summary_path)
    config = load_stage4_config(config_path)
    provenance = _require_tagged_clean(config, VALIDATION_TAG)
    freeze_value = _load_validation_freeze(
        config,
        root,
        Path(tracked_validation_freeze).resolve(),
    )
    search = _read_json(root / "search-summary.json")
    champion = cast(Mapping[str, Any], search["champion"])
    champion_source = Path(cast(str, champion["source_reference"]))
    if _sha_file(champion_source) != champion.get("source_sha256"):
        raise RuntimeError("Stage 4R champion source identity drifted")
    seeds = {seed.slot_id: seed for seed in stage4._seed_sources(config)}
    roster = {
        "champion": champion_source.read_text(encoding="utf-8"),
        "stage3-candidate-slot-04": seeds["slot-04"].source,
        **stage4._baseline_sources(config),
    }
    manifest = stage4._load_manifest(config.validation_manifest_path)
    output = root / "evaluations" / "final-validation"
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
    primary_records = _rename_stage3(
        cast(Sequence[Mapping[str, Any]], primary["records"])
    )
    replay_records = _rename_stage3(
        cast(Sequence[Mapping[str, Any]], replay["records"])
    )
    primary_summary = summarize_development(
        primary_records,
        ("champion", "stage3_champion", "random", "structural"),
        bootstrap_samples=config.evaluation.bootstrap_samples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
    )
    replay_summary = summarize_development(
        replay_records,
        ("champion", "stage3_champion", "random", "structural"),
        bootstrap_samples=config.evaluation.bootstrap_samples,
        bootstrap_seed=config.evaluation.bootstrap_seed,
    )
    primary_evidence = _validation_evidence(primary_records, primary_summary)
    replay_evidence = _validation_evidence(replay_records, replay_summary)
    health = {
        **stage4._validation_health(primary_records, primary, replay),
        "provider_calls": 0,
    }
    replay_health = {
        **stage4._validation_health(replay_records, replay, primary),
        "provider_calls": 0,
    }
    seed_hashes = {seed.ast_sha256 for seed in seeds.values()}
    baseline_hashes = {
        validate_policy(source, config.sandbox).identity.normalized_ast_sha256
        for source in stage4._baseline_sources(config).values()
    }
    champion_distinct = (
        champion.get("origin") in {"retained-stage4", "stage4r-recovery"}
        and int(champion.get("generation", 0)) > 0
        and champion.get("normalized_ast_sha256") not in seed_hashes | baseline_hashes
    )
    gate = _scientific_gate(
        champion_distinct=champion_distinct,
        evidence=primary_evidence,
        replay=replay_check,
        primary_summary=primary_summary,
        replay_summary=replay_summary,
        primary_evidence=primary_evidence,
        replay_evidence=replay_evidence,
        health=health,
        replay_health=replay_health,
    )
    result: dict[str, Any] = {
        "schema_version": "stage4r.validation.result.v1",
        "status": "completed",
        "run": str(root),
        "validation_freeze_sha256": freeze_value["freeze_sha256"],
        "champion": dict(champion),
        "summary": {**primary_summary, **primary_evidence, **health},
        "replay_summary": replay_summary,
        "replay": replay_check,
        "primary_pass": {key: value for key, value in primary.items() if key != "records"},
        "replay_pass": {key: value for key, value in replay.items() if key != "records"},
        "gate": gate,
        "decision": gate["decision"],
        "primary_replay_identity": gate["checks"][SCIENTIFIC_GATES[6]],
        "evaluation_provider_calls": 0,
        "final_validation_results_observed": True,
        "provenance": provenance,
        "stage5_execution_authorized": False,
        "retained_run": str(Path(retained_run).resolve()),
    }
    _atomic_json(summary_path, result)
    return result


__all__ = [
    "AUTH_MODE",
    "RETAINED_ARCHIVE_SHA256",
    "SEARCH_TAG",
    "VALIDATION_MANIFEST_SHA256",
    "VALIDATION_TAG",
    "canary",
    "freeze_search",
    "freeze_validation",
    "recover",
    "validate",
]
