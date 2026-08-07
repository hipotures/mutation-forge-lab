"""Guarded Native v3 preview communication modes."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    ModelProfile,
)
from mutation_forge.stage3.isolation import IsolatedCapsule, secure_capsule_parent

from .canonical import canonical_json_bytes
from .cohort import (
    SLOT_IDS,
    CohortEntry,
    build_epoch_manifest,
    finalize_cohort,
)
from .contracts import validated_program_artifact
from .lineage_experiment import _fork
from .persistent_experiment import (
    BRIEF_IDS,
    TurnObservation,
    _behavior_signature,
    _run_adapter_turn,
    bootstrap_prompt,
    bootstrap_schema,
    protocol_hash,
)
from .search_memory import (
    ActiveParentReference,
    DuplicateCandidateError,
    LineageSummary,
    PatternSummary,
    SearchMemoryV1,
    program_families,
    reject_duplicate,
)
from .single_program_contract import (
    SINGLE_PROGRAM_BRIEFS,
    build_single_program_contract,
)
from .single_program_ir import (
    SLOT_SPECIFIC_OUTPUT_CONTRACT,
    CandidateContractError,
    CompiledCandidateResponse,
    build_candidate_request,
    compile_slot_specific_response,
    slot_specific_contract_sha256,
    slot_specific_schema_hashes,
)

PERSISTENT_SINGLE_AST = "persistent_single_ast"
FRESH_SINGLE_AST = "fresh_single_ast"
ROLLBACK_MODE = "multi_program_batch"
PREVIEW_STATE_SCHEMA_VERSION = "mforge.native-v3.preview-state.v1"
PREVIEW_PROGRAM_RECORD_SCHEMA_VERSION = "mforge.native-v3.program-record.v1"
FORBIDDEN_LENGTHS = (4, 8, 16)
PREVIEW_SLOT_IDS = SLOT_IDS[:4]
WORKER_COUNT = 2
MAX_REPAIRS = 1
MAX_PROVIDER_TURNS = 1 + len(PREVIEW_SLOT_IDS) * (MAX_REPAIRS + 1)

AdapterFactory = Callable[
    [str, IsolatedCapsule, Path, str, float],
    CodexAppServerAdapter,
]
CapsuleFactory = Callable[[], IsolatedCapsule]
CapsuleReopener = Callable[[str | Path], IsolatedCapsule]

_ARTIFACT_SUFFIXES = frozenset(
    {
        "codex-profile.json.gz",
        "codex-rpc.jsonl",
        "events.jsonl",
        "output-schema.json.gz",
        "provider-raw.json.gz",
        "request.json.gz",
        "request.md",
        "response.json.gz",
        "response.md",
        "response.raw.txt",
        "stderr.txt",
        "stdout.jsonl",
        "system-prompt.md",
        "transcript.sha256",
        "usage.json.gz",
        "wire.jsonl",
    }
)


def _default_adapter_factory(
    base_instructions: str,
    capsule: IsolatedCapsule,
    turns_dir: Path,
    prefix: str,
    timeout_seconds: float,
) -> CodexAppServerAdapter:
    return CodexAppServerAdapter(
        capsule=capsule,
        limits=AppServerLimits(
            max_turns=MAX_PROVIDER_TURNS,
            max_campaigns=3,
            turn_timeout=timeout_seconds,
        ),
        base_instructions=base_instructions,
        artifact_dir=turns_dir,
        artifact_prefix=prefix,
        compress_json_artifacts=True,
        sandbox_mode="read-only",
        approval_policy="never",
        copy_rollout_artifact=False,
    )


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _summary(value: str) -> str:
    cleaned = " ".join(value.split())
    return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."


def _empty_memory() -> SearchMemoryV1:
    return SearchMemoryV1(
        protocol_hash=protocol_hash(FORBIDDEN_LENGTHS),
        seen_program_hashes=(),
        seen_behavior_signatures=(),
        successful_patterns=(),
        failed_patterns=(),
        active_lineages=(),
        validated_archive_ids=(),
    )


def _extend_memory(
    memory: SearchMemoryV1,
    response: CompiledCandidateResponse,
    *,
    slot_index: int,
) -> SearchMemoryV1:
    program = response.program
    behavior_signature = _behavior_signature(program.ast)
    selectors, actions = program_families(program.ast)
    candidate_id = f"g0000-s{slot_index:02d}"
    patterns = memory.successful_patterns
    if selectors or actions:
        patterns += (
            PatternSummary(
                pattern_id=f"validated-{slot_index:02d}",
                selector_families=selectors,
                action_families=actions,
                description=_summary(response.design_summary),
                description_source="model",
                evaluation_outcome="VALIDATED",
                evidence_kind="strength",
                main_evidence=_summary(response.hypothesis),
            ),
        )
    lineage = LineageSummary(
        candidate_id=candidate_id,
        parent_id=None,
        program_hash=program.program_hash,
        behavior_signature=behavior_signature,
        generation=0,
        slot=slot_index,
        evaluation_outcome="VALIDATED",
        summary=_summary(response.design_summary),
    )
    return SearchMemoryV1(
        protocol_hash=memory.protocol_hash,
        seen_program_hashes=memory.seen_program_hashes + (program.program_hash,),
        seen_behavior_signatures=(memory.seen_behavior_signatures + (behavior_signature,)),
        successful_patterns=patterns,
        failed_patterns=memory.failed_patterns,
        active_lineages=memory.active_lineages + (lineage,),
        validated_archive_ids=memory.validated_archive_ids + (candidate_id,),
        active_parent=ActiveParentReference(
            candidate_id=candidate_id,
            program_hash=program.program_hash,
        ),
    )


def _program_prompt(
    slot_id: str,
    brief_id: str,
    memory: SearchMemoryV1,
) -> str:
    request = build_candidate_request(
        candidate=SLOT_SPECIFIC_OUTPUT_CONTRACT,
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=FORBIDDEN_LENGTHS,
        accepted_behavior_signatures=memory.seen_behavior_signatures,
    )
    return request.prompt + "\n\nSearch Memory:\n" + _compact_json(
        {"search_memory": memory.as_dict()}
    )


def _repair_prompt(
    slot_id: str,
    brief_id: str,
    memory: SearchMemoryV1,
    error: str,
) -> str:
    return _compact_json(
        {
            "instruction": (
                "The host rejected the preceding response. Generate exactly one "
                "corrected slot-specific response for the same brief. Return only the "
                "structured response and do not repeat the rejected program."
            ),
            "slot_id": slot_id,
            "brief_id": brief_id,
            "host_validation_error": error[:512],
            "search_memory": memory.as_dict(),
        }
    )


def _artifact_complete(turns_dir: Path, prefix: str) -> bool:
    suffixes = {
        path.name.removeprefix(f"{prefix}.")
        for path in turns_dir.iterdir()
        if path.name.startswith(f"{prefix}.")
    }
    return suffixes == _ARTIFACT_SUFFIXES


def _available_prefix(turns_dir: Path, base: str) -> str:
    if not any(
        path.name == base or path.name.startswith(f"{base}.") for path in turns_dir.iterdir()
    ):
        return base
    attempt = 1
    while any(
        path.name == f"{base}.resume-{attempt:02d}"
        or path.name.startswith(f"{base}.resume-{attempt:02d}.")
        for path in turns_dir.iterdir()
    ):
        attempt += 1
    return f"{base}.resume-{attempt:02d}"


def _response(
    turns_dir: Path,
    prefix: str,
    *,
    slot_id: str,
    brief_id: str,
) -> CompiledCandidateResponse:
    return compile_slot_specific_response(
        (turns_dir / f"{prefix}.response.raw.txt").read_text(encoding="utf-8"),
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )


def _stored_output_schema_sha256(turns_dir: Path, prefix: str) -> str:
    value = read_json(turns_dir / f"{prefix}.output-schema.json.gz")
    if not isinstance(value, Mapping):
        raise ValueError("provider output schema artifact is not an object")
    return _hash(value)


def _memory_from_reports(
    reports: Sequence[Mapping[str, Any]],
    turns_dir: Path,
) -> tuple[SearchMemoryV1, list[CohortEntry]]:
    memory = _empty_memory()
    entries: list[CohortEntry] = []
    schema_hashes = slot_specific_schema_hashes(FORBIDDEN_LENGTHS)
    for slot_index, raw in enumerate(reports):
        report = dict(raw)
        slot_id = str(report["slot_id"])
        brief_id = str(report["brief_id"])
        accepted_prefix = report.get("accepted_prefix")
        if not isinstance(accepted_prefix, str):
            entries.append(
                CohortEntry(
                    slot_id,
                    None,
                    None,
                    str(report.get("error") or "invalid program"),
                )
            )
            continue
        if (
            _stored_output_schema_sha256(turns_dir, accepted_prefix)
            != schema_hashes[brief_id]
        ):
            raise ValueError("stored provider output-schema identity mismatch")
        response = _response(
            turns_dir,
            accepted_prefix,
            slot_id=slot_id,
            brief_id=brief_id,
        )
        memory = _extend_memory(memory, response, slot_index=slot_index)
        entries.append(
            CohortEntry(
                slot_id,
                response.program,
                response.design_summary,
                None,
            )
        )
    return memory, entries


def _manifest(model: str, effort: str, output_contract: str) -> dict[str, Any]:
    if output_contract != SLOT_SPECIFIC_OUTPUT_CONTRACT:
        raise ValueError("persistent preview requires slot_specific output contract")
    manifest = build_epoch_manifest(model=model, effort=effort)
    system_prompt = build_candidate_request(
        candidate=SLOT_SPECIFIC_OUTPUT_CONTRACT,
        slot_id=PREVIEW_SLOT_IDS[0],
        brief_id=BRIEF_IDS[0],
        forbidden_lengths=FORBIDDEN_LENGTHS,
    ).system_prompt
    schema_hashes = slot_specific_schema_hashes(FORBIDDEN_LENGTHS)
    contract_sha256 = slot_specific_contract_sha256(FORBIDDEN_LENGTHS)
    protocols = {
        **cast(dict[str, str], manifest["protocols"]),
        "provider_input": "native_v3_persistent_single_ast_slot_specific_input_v1",
        "provider_output": "native_v3_persistent_single_ast_slot_specific_output_v1",
    }
    protocol_bundle_hash = _hash(protocols)
    manifest.update(
        {
            "planned_slot_ids": list(PREVIEW_SLOT_IDS),
            "communication_mode": PERSISTENT_SINGLE_AST,
            "provider_mode": PERSISTENT_SINGLE_AST,
            "output_contract": output_contract,
            "output_schema_sha256": contract_sha256,
            "output_schema_sha256_by_brief": schema_hashes,
            "worker_count": WORKER_COUNT,
            "programs_per_turn": 1,
            "compaction_used": False,
            "compaction_mode": "disabled",
            "rotation_policy": "fresh_spec_fork_plus_search_memory",
            "rollback_mode": ROLLBACK_MODE,
            "diagnostic_mode": FRESH_SINGLE_AST,
            "single_program_system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "repair_prompt_sha256": None,
            "search_memory_schema_version": "mforge.native.search_memory.v1",
            "protocols": protocols,
            "protocol_bundle_hash": protocol_bundle_hash,
        }
    )
    manifest["provider_calls"] = [
        {
            "call_id": f"epoch-0000:single:{index:04d}",
            "slot_ids": [slot_id],
            "brief_id": BRIEF_IDS[index % len(BRIEF_IDS)],
            "worker_index": index % WORKER_COUNT,
        }
        for index, slot_id in enumerate(PREVIEW_SLOT_IDS)
    ]
    manifest["slots"] = [
        {
            "slot_id": slot_id,
            "parent_program_hashes": [],
            "brief_id": brief_id,
            "brief": SINGLE_PROGRAM_BRIEFS[brief_id],
            "brief_sha256": hashlib.sha256(
                SINGLE_PROGRAM_BRIEFS[brief_id].encode("utf-8")
            ).hexdigest(),
        }
        for index, slot_id in enumerate(PREVIEW_SLOT_IDS)
        for brief_id in (BRIEF_IDS[index % len(BRIEF_IDS)],)
    ]
    manifest["epoch_id"] = _hash(
        {
            "epoch_number": manifest["epoch_number"],
            "slots": manifest["slots"],
            "provider_calls": manifest["provider_calls"],
            "protocol_bundle_hash": protocol_bundle_hash,
            "communication_mode": PERSISTENT_SINGLE_AST,
            "output_contract": output_contract,
            "output_schema_sha256": contract_sha256,
            "model": model,
            "effort": effort,
        }
    )
    return manifest


def _new_state(
    capsule: IsolatedCapsule,
    anchor: TurnObservation,
    source_identity: Mapping[str, Any],
    forks: Sequence[Mapping[str, Any]],
    output_contract: str,
) -> dict[str, Any]:
    bootstrap_retries = int(source_identity.get("serverRetries", 0))
    bootstrap_warnings = int(source_identity.get("serverWarnings", 0))
    return {
        "schema_version": PREVIEW_STATE_SCHEMA_VERSION,
        "communication_mode": PERSISTENT_SINGLE_AST,
        "provider_mode": PERSISTENT_SINGLE_AST,
        "output_contract": output_contract,
        "output_schema_sha256": slot_specific_contract_sha256(FORBIDDEN_LENGTHS),
        "output_schema_sha256_by_brief": slot_specific_schema_hashes(FORBIDDEN_LENGTHS),
        "compaction_mode": "disabled",
        "rollback_mode": ROLLBACK_MODE,
        "diagnostic_mode": FRESH_SINGLE_AST,
        "status": "generating",
        "started_at_ms": time.time_ns() // 1_000_000,
        "first_valid_ast_at_ms": None,
        "cohort_completed_at_ms": None,
        "capsule_root": str(capsule.root),
        "specification_thread": {
            "thread_id": source_identity["threadId"],
            "session_id": source_identity["sessionId"],
            "thread_path": source_identity["threadPath"],
            "anchor_turn_id": anchor.turn_id,
        },
        "anchor": anchor.as_dict(),
        "workers": [dict(item) for item in forks],
        "next_slot": 0,
        "slot_reports": [],
        "model_turns": 1,
        "program_turns": 0,
        "provider_attempts": 1,
        "failed_provider_attempts": 0,
        "provider_retries": bootstrap_retries,
        "provider_warnings": bootstrap_warnings,
        "provider_process_restarts": 0,
        "thread_resume_attempts": 0,
        "failed_thread_resume_attempts": 0,
        "active_provider_attempt": None,
        "last_provider_attempt": {
            "phase": "bootstrap",
            "status": "completed",
            "prefix": anchor.prefix,
            "slot_id": None,
            "brief_id": None,
            "worker_index": None,
            "repair_index": None,
            "thread_id": anchor.thread_id,
            "turn_id": anchor.turn_id,
            "provider_retries": bootstrap_retries,
            "provider_warnings": bootstrap_warnings,
            "error": None,
        },
        "usages": [anchor.usage],
    }


def _worker_record(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_thread_id": value["source_thread_id"],
        "thread_id": value["child_thread_id"],
        "session_id": value["session_id"],
        "thread_path": value["thread_path"],
        "fork_parent_turn_id": value["last_turn_id"],
        "included_turn_ids": list(cast(Sequence[str], value["included_turn_ids"])),
    }


def _ensure_progress_state(state: dict[str, Any]) -> None:
    model_turns = int(state.get("model_turns", 0))
    state.setdefault("program_turns", max(0, model_turns - 1))
    state.setdefault("provider_attempts", model_turns)
    state.setdefault("failed_provider_attempts", 0)
    state.setdefault("provider_retries", 0)
    state.setdefault("provider_warnings", 0)
    state.setdefault("provider_process_restarts", 0)
    state.setdefault("thread_resume_attempts", 0)
    state.setdefault("failed_thread_resume_attempts", 0)
    state.setdefault("active_provider_attempt", None)
    state.setdefault("last_provider_attempt", None)


def _adapter_event_counts(
    adapter: CodexAppServerAdapter,
) -> tuple[int, int]:
    metadata = adapter.inspect_metadata()
    return (
        int(metadata.get("serverRetries", 0)),
        int(metadata.get("serverWarnings", 0)),
    )


def run_persistent_single_ast_cohort(
    experiment_root: str | Path,
    *,
    model: str,
    effort: str,
    timeout_seconds: float,
    auth_json: str | Path,
    backend_factory: Callable[[], GraphBackend],
    episode_id: str,
    output_contract: str,
    adapter_factory: AdapterFactory = _default_adapter_factory,
    capsule_factory: CapsuleFactory | None = None,
    capsule_reopener: CapsuleReopener = IsolatedCapsule.reopen,
) -> dict[str, Any]:
    """Generate one AST per durable worker turn, then run the frozen evaluator."""

    if output_contract != SLOT_SPECIFIC_OUTPUT_CONTRACT:
        raise ValueError("persistent preview requires slot_specific output contract")
    root = Path(experiment_root)
    output_root = root / "native-v3-output" / "epoch-0000"
    turns_dir = root / "provider-turns"
    state_path = output_root / "communication-state.json.gz"
    manifest_path = output_root / "epoch-manifest.json.gz"
    turns_dir.mkdir(parents=True, exist_ok=True)
    profile = ModelProfile("codex", model, effort)
    request = build_candidate_request(
        candidate=SLOT_SPECIFIC_OUTPUT_CONTRACT,
        slot_id=PREVIEW_SLOT_IDS[0],
        brief_id=BRIEF_IDS[0],
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    system_prompt = request.system_prompt
    contract_sha256 = slot_specific_contract_sha256(FORBIDDEN_LENGTHS)
    schema_hashes = slot_specific_schema_hashes(FORBIDDEN_LENGTHS)

    adapter: CodexAppServerAdapter | None = None
    if state_path.is_file():
        raw_state = read_json(state_path)
        if not isinstance(raw_state, Mapping):
            raise ValueError("preview communication state is not an object")
        state = {str(key): value for key, value in raw_state.items()}
        if (
            state.get("schema_version") != PREVIEW_STATE_SCHEMA_VERSION
            or state.get("communication_mode") != PERSISTENT_SINGLE_AST
            or state.get("output_contract") != output_contract
            or state.get("output_schema_sha256") != contract_sha256
            or state.get("output_schema_sha256_by_brief") != schema_hashes
        ):
            raise ValueError("preview communication state is incompatible")
        raw_reports = state.get("slot_reports", [])
        if not isinstance(raw_reports, list):
            raise ValueError("preview slot reports are invalid")
        reports = [cast(Mapping[str, Any], item) for item in raw_reports]
        memory, entries = _memory_from_reports(reports, turns_dir)
        capsule = capsule_reopener(str(state["capsule_root"]))
        _ensure_progress_state(state)
        state.update(
            {
                "status": "generating",
                "last_error": None,
                "active_provider_attempt": None,
                "provider_process_restarts": int(state["provider_process_restarts"]) + 1,
            }
        )
        workers = cast(list[dict[str, Any]], state["workers"])
        resume_prefix = _available_prefix(turns_dir, "00-worker-00-resume")
        adapter = adapter_factory(
            system_prompt,
            capsule,
            turns_dir,
            resume_prefix,
            timeout_seconds,
        )
        for worker_index, worker in enumerate(workers):
            prefix = _available_prefix(
                turns_dir,
                f"00-worker-{worker_index:02d}-resume",
            )
            resume_attempt = {
                "phase": "thread_resume",
                "status": "running",
                "prefix": prefix,
                "slot_id": None,
                "brief_id": None,
                "worker_index": worker_index,
                "repair_index": None,
                "thread_id": worker["thread_id"],
                "turn_id": None,
                "provider_retries": 0,
                "provider_warnings": 0,
                "error": None,
            }
            state["thread_resume_attempts"] = int(state["thread_resume_attempts"]) + 1
            state["active_provider_attempt"] = resume_attempt
            write_json(state_path, state)
            if worker_index:
                adapter.rotate_logger(turns_dir, prefix)
            try:
                if worker_index == 0:
                    adapter.resume_thread(
                        profile,
                        thread_id=str(worker["thread_id"]),
                        thread_path=str(worker["thread_path"]),
                    )
                else:
                    adapter.resume_forked_thread(
                        profile,
                        thread_id=str(worker["thread_id"]),
                        thread_path=str(worker["thread_path"]),
                    )
            except Exception as resume_error:
                resume_attempt.update(
                    {
                        "status": "failed",
                        "error": (f"{type(resume_error).__name__}: {resume_error}"),
                    }
                )
                state.update(
                    {
                        "status": "provider_failed",
                        "last_error": resume_attempt["error"],
                        "failed_thread_resume_attempts": int(state["failed_thread_resume_attempts"])
                        + 1,
                        "active_provider_attempt": None,
                        "last_provider_attempt": resume_attempt,
                    }
                )
                write_json(state_path, state)
                adapter.close(force=True)
                raise
            resume_attempt["status"] = "completed"
            state.update(
                {
                    "active_provider_attempt": None,
                    "last_provider_attempt": resume_attempt,
                }
            )
            write_json(state_path, state)
    else:
        expected_manifest = _manifest(model, effort, output_contract)
        if manifest_path.is_file():
            if read_json(manifest_path) != expected_manifest:
                raise ValueError("preview epoch manifest identity mismatch")
        else:
            write_json(manifest_path, expected_manifest, exclusive=True)
        capsule = (
            capsule_factory()
            if capsule_factory is not None
            else IsolatedCapsule.create(
                secure_capsule_parent(),
                auth_json=auth_json,
                sandbox_mode="read-only",
                approval_policy="never",
            )
        )
        anchor_prefix = _available_prefix(turns_dir, "00-spec-anchor")
        adapter = adapter_factory(
            system_prompt,
            capsule,
            turns_dir,
            anchor_prefix,
            timeout_seconds,
        )
        try:
            anchor = _run_adapter_turn(
                adapter,
                artifact_dir=turns_dir,
                prefix=anchor_prefix,
                prompt=bootstrap_prompt(FORBIDDEN_LENGTHS),
                system_prompt=system_prompt,
                schema=bootstrap_schema(protocol_hash(FORBIDDEN_LENGTHS)),
                profile=profile,
                persistent=True,
                forbidden_lengths=FORBIDDEN_LENGTHS,
                program_response=False,
            )
            forks = [
                _fork(
                    adapter,
                    turns_dir=turns_dir,
                    prefix=_available_prefix(
                        turns_dir,
                        f"00-worker-{index:02d}-fork",
                    ),
                    system_prompt=system_prompt,
                    profile=profile,
                    last_turn_id=anchor.turn_id,
                )
                for index in range(WORKER_COUNT)
            ]
            source_identity = dict(adapter.inspect_metadata())
        except Exception:
            adapter.close(force=True)
            capsule.cleanup()
            raise
        if not _artifact_complete(turns_dir, anchor.prefix):
            adapter.close(force=True)
            capsule.cleanup()
            raise ValueError("bootstrap turn artifact contract is incomplete")
        fork_records = [
            {
                "source_thread_id": fork.source_thread_id,
                "child_thread_id": fork.child_thread_id,
                "session_id": fork.session_id,
                "thread_path": fork.thread_path,
                "last_turn_id": fork.last_turn_id,
                "included_turn_ids": list(fork.included_turn_ids),
            }
            for fork in forks
        ]
        if any(
            tuple(cast(Sequence[str], item["included_turn_ids"])) != (anchor.turn_id,)
            for item in fork_records
        ):
            adapter.close(force=True)
            capsule.cleanup()
            raise ValueError("worker fork crossed the specification boundary")
        state = _new_state(
            capsule,
            anchor,
            source_identity,
            [_worker_record(item) for item in fork_records],
            output_contract,
        )
        write_json(state_path, state)
        reports = []
        memory = _empty_memory()
        entries = []
    assert adapter is not None
    _ensure_progress_state(state)
    workers = cast(list[dict[str, Any]], state["workers"])
    model_turns = int(state["model_turns"])
    usages = [cast(Mapping[str, Any], item) for item in cast(list[Any], state["usages"])]

    for slot_index in range(int(state["next_slot"]), len(PREVIEW_SLOT_IDS)):
        slot_id = PREVIEW_SLOT_IDS[slot_index]
        brief_id = BRIEF_IDS[slot_index % len(BRIEF_IDS)]
        worker_index = slot_index % WORKER_COUNT
        worker = workers[worker_index]
        attempts: list[dict[str, Any]] = []
        accepted: CompiledCandidateResponse | None = None
        accepted_prefix: str | None = None
        error: str | None = None

        for repair_index in range(MAX_REPAIRS + 1):
            prefix_base = (
                f"{slot_id}.initial"
                if repair_index == 0
                else f"{slot_id}.repair-{repair_index:02d}"
            )
            prefix = _available_prefix(turns_dir, prefix_base)
            candidate_request = build_candidate_request(
                candidate=SLOT_SPECIFIC_OUTPUT_CONTRACT,
                slot_id=slot_id,
                brief_id=brief_id,
                forbidden_lengths=FORBIDDEN_LENGTHS,
                accepted_behavior_signatures=memory.seen_behavior_signatures,
            )
            schema = candidate_request.output_schema
            expected_schema_sha256 = schema_hashes[brief_id]
            if _hash(schema) != expected_schema_sha256:
                raise ValueError("slot-specific output schema identity mismatch")
            prompt = (
                _program_prompt(slot_id, brief_id, memory)
                if repair_index == 0
                else _repair_prompt(
                    slot_id,
                    brief_id,
                    memory,
                    error or "invalid program",
                )
            )
            adapter.activate_forked_thread(str(worker["thread_id"]))
            retries_before, warnings_before = _adapter_event_counts(adapter)
            provider_attempt = {
                "phase": "program",
                "status": "running",
                "prefix": prefix,
                "slot_id": slot_id,
                "brief_id": brief_id,
                "worker_index": worker_index,
                "repair_index": repair_index,
                "thread_id": worker["thread_id"],
                "turn_id": None,
                "provider_retries": 0,
                "provider_warnings": 0,
                "output_contract": output_contract,
                "output_schema_sha256": expected_schema_sha256,
                "error": None,
            }
            state.update(
                {
                    "status": "generating",
                    "last_error": None,
                    "provider_attempts": int(state["provider_attempts"]) + 1,
                    "active_provider_attempt": provider_attempt,
                }
            )
            write_json(state_path, state)
            try:
                observation = _run_adapter_turn(
                    adapter,
                    artifact_dir=turns_dir,
                    prefix=prefix,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=schema,
                    profile=profile,
                    persistent=True,
                    forbidden_lengths=FORBIDDEN_LENGTHS,
                    program_response=False,
                )
                stored_schema_sha256 = _stored_output_schema_sha256(turns_dir, prefix)
                if stored_schema_sha256 != expected_schema_sha256:
                    raise ValueError(
                        "provider output-schema artifact identity mismatch"
                    )
            except Exception as provider_error:
                retries_after, warnings_after = _adapter_event_counts(adapter)
                retry_count = retries_after - retries_before
                warning_count = warnings_after - warnings_before
                thread_id, turn_id = adapter.experimental_turn_identity()
                provider_attempt.update(
                    {
                        "status": "failed",
                        "thread_id": thread_id or worker["thread_id"],
                        "turn_id": turn_id,
                        "provider_retries": retry_count,
                        "provider_warnings": warning_count,
                        "error": (f"{type(provider_error).__name__}: {provider_error}"),
                        "usage": dict(adapter.inspect_usage()),
                    }
                )
                state.update(
                    {
                        "status": "provider_failed",
                        "last_error": provider_attempt["error"],
                        "failed_provider_attempts": int(state["failed_provider_attempts"]) + 1,
                        "provider_retries": int(state["provider_retries"]) + retry_count,
                        "provider_warnings": int(state["provider_warnings"]) + warning_count,
                        "active_provider_attempt": None,
                        "last_provider_attempt": provider_attempt,
                    }
                )
                write_json(state_path, state)
                adapter.close(force=True)
                raise
            retries_after, warnings_after = _adapter_event_counts(adapter)
            retry_count = retries_after - retries_before
            warning_count = warnings_after - warnings_before
            provider_attempt.update(
                {
                    "status": "completed",
                    "thread_id": observation.thread_id,
                    "turn_id": observation.turn_id,
                    "provider_retries": retry_count,
                    "provider_warnings": warning_count,
                }
            )
            state.update(
                {
                    "provider_retries": int(state["provider_retries"]) + retry_count,
                    "provider_warnings": int(state["provider_warnings"]) + warning_count,
                    "active_provider_attempt": None,
                    "last_provider_attempt": provider_attempt,
                }
            )
            write_json(state_path, state)
            model_turns += 1
            state["program_turns"] = int(state["program_turns"]) + 1
            usages.append(observation.usage)
            worker["last_turn_id"] = observation.turn_id
            candidate: CompiledCandidateResponse | None = None
            duplicate_reason = None
            try:
                candidate = _response(
                    turns_dir,
                    prefix,
                    slot_id=slot_id,
                    brief_id=brief_id,
                )
            except CandidateContractError as contract_error:
                error = str(contract_error)
                observation = replace(observation, error=error)
            else:
                observation = replace(
                    observation,
                    program_hash=candidate.program.program_hash,
                    behavior_signature=_behavior_signature(candidate.program.ast),
                )
                try:
                    reject_duplicate(
                        memory,
                        program_hash=candidate.program.program_hash,
                        behavior_signature=_behavior_signature(candidate.program.ast),
                    )
                except DuplicateCandidateError as exc:
                    duplicate_reason = str(exc)
                    error = duplicate_reason
                    candidate = None
            attempt = {
                **observation.as_dict(),
                "artifact_complete": _artifact_complete(turns_dir, prefix),
                "duplicate_reason": duplicate_reason,
            }
            attempts.append(attempt)
            write_json(
                output_root / "program-attempts" / f"{prefix}.json.gz",
                {
                    "schema_version": PREVIEW_PROGRAM_RECORD_SCHEMA_VERSION,
                    "slot_id": slot_id,
                    "brief_id": brief_id,
                    "worker_index": worker_index,
                    "fork_parent_turn_id": worker["fork_parent_turn_id"],
                    "provider_mode": PERSISTENT_SINGLE_AST,
                    "output_contract": output_contract,
                    "output_schema_sha256": expected_schema_sha256,
                    "output_schema_contract_sha256": contract_sha256,
                    "prompt_contract_sha256": hashlib.sha256(
                        _compact_json(
                            {
                                "system_prompt": system_prompt,
                                "output_schema": schema,
                            }
                        ).encode("utf-8")
                    ).hexdigest(),
                    "attempt": attempt,
                    "valid_ast": candidate is not None,
                    "model_representation_sha256": (
                        None
                        if candidate is None
                        else candidate.representation_sha256
                    ),
                    "program": (
                        None if candidate is None else validated_program_artifact(candidate.program)
                    ),
                },
                exclusive=True,
            )
            if not attempt["artifact_complete"]:
                adapter.close(force=True)
                raise ValueError(f"{prefix} artifact contract is incomplete")
            if candidate is not None:
                accepted = candidate
                accepted_prefix = prefix
                break

        if accepted is not None:
            memory = _extend_memory(memory, accepted, slot_index=slot_index)
            published_at_ms = time.time_ns() // 1_000_000
            if state.get("first_valid_ast_at_ms") is None:
                state["first_valid_ast_at_ms"] = published_at_ms
            entry = CohortEntry(
                slot_id,
                accepted.program,
                accepted.design_summary,
                None,
            )
        else:
            published_at_ms = None
            entry = CohortEntry(slot_id, None, None, error or "invalid program")
        entries.append(entry)
        lineage = {
            "slot_id": slot_id,
            "brief_id": f"native-v3-brief-{slot_index:02d}",
            "parent_program_hashes": [],
            "worker_index": worker_index,
            "worker_thread_id": worker["thread_id"],
            "fork_parent_turn_id": worker["fork_parent_turn_id"],
            "output_contract": output_contract,
            "output_schema_sha256": schema_hashes[brief_id],
            "provider_attempts": attempts,
        }
        slot_report = {
            "slot_id": slot_id,
            "brief_id": brief_id,
            "worker_index": worker_index,
            "accepted_prefix": accepted_prefix,
            "published_at_ms": published_at_ms,
            "entry": entry.as_dict(),
            "lineage": lineage,
            "error": entry.error,
            "attempts": attempts,
        }
        reports.append(slot_report)
        write_json(
            output_root / "program-records" / f"{slot_id}.json.gz",
            {
                "schema_version": PREVIEW_PROGRAM_RECORD_SCHEMA_VERSION,
                **slot_report,
                "search_memory_sha256": memory.sha256,
                "provider_mode": PERSISTENT_SINGLE_AST,
                "output_contract": output_contract,
                "output_schema_sha256": schema_hashes[brief_id],
                "output_schema_contract_sha256": contract_sha256,
                "program": (
                    None if accepted is None else validated_program_artifact(accepted.program)
                ),
            },
            exclusive=True,
        )
        state.update(
            {
                "next_slot": slot_index + 1,
                "slot_reports": reports,
                "workers": workers,
                "model_turns": model_turns,
                "usages": usages,
                "search_memory_sha256": memory.sha256,
            }
        )
        write_json(state_path, state)

    adapter.close()
    adapter = None
    slot_lineage = {
        str(report["slot_id"]): cast(Mapping[str, Any], report["lineage"]) for report in reports
    }
    report = finalize_cohort(
        root,
        entries=entries,
        generation_reports=reports,
        slot_lineage=slot_lineage,
        usages=usages,
        model_turns=model_turns,
        backend_factory=backend_factory,
        episode_id=episode_id,
        communication_mode=PERSISTENT_SINGLE_AST,
        program_contract=build_single_program_contract(FORBIDDEN_LENGTHS),
    )
    completed_at_ms = time.time_ns() // 1_000_000
    first_valid_ast_at_ms = state.get("first_valid_ast_at_ms")
    time_to_first_valid_ast_ms = (
        None
        if not isinstance(first_valid_ast_at_ms, int)
        else first_valid_ast_at_ms - int(state["started_at_ms"])
    )
    report.update(
        {
            "provider_mode": PERSISTENT_SINGLE_AST,
            "output_contract": output_contract,
            "output_schema_sha256": contract_sha256,
            "output_schema_sha256_by_brief": schema_hashes,
            "compaction_mode": "disabled",
            "rollback_mode": ROLLBACK_MODE,
            "diagnostic_mode": FRESH_SINGLE_AST,
            "program_turns": int(state["program_turns"]),
            "time_to_first_valid_ast_ms": time_to_first_valid_ast_ms,
            "first_valid_ast_published_before_cohort_complete": (
                isinstance(first_valid_ast_at_ms, int)
                and first_valid_ast_at_ms <= completed_at_ms
            ),
        }
    )
    write_json(Path(str(report["cohort_report"])), report)
    state.update(
        {
            "status": "completed",
            "cohort_report": report["cohort_report"],
            "cohort_completed_at_ms": completed_at_ms,
            "time_to_first_valid_ast_ms": time_to_first_valid_ast_ms,
        }
    )
    write_json(state_path, state)
    capsule.cleanup()
    return report


__all__ = [
    "FRESH_SINGLE_AST",
    "FORBIDDEN_LENGTHS",
    "PERSISTENT_SINGLE_AST",
    "PREVIEW_PROGRAM_RECORD_SCHEMA_VERSION",
    "PREVIEW_SLOT_IDS",
    "PREVIEW_STATE_SCHEMA_VERSION",
    "ROLLBACK_MODE",
    "run_persistent_single_ast_cohort",
]
