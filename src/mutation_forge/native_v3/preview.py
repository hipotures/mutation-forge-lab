"""Guarded Native v3 preview communication modes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
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
    SingleProgramResponse,
    build_single_program_output_schema,
    build_single_program_request,
    validate_single_program_response,
)

PERSISTENT_SINGLE_AST = "persistent_single_ast"
PREVIEW_STATE_SCHEMA_VERSION = "mforge.native-v3.preview-state.v1"
PREVIEW_PROGRAM_RECORD_SCHEMA_VERSION = "mforge.native-v3.program-record.v1"
FORBIDDEN_LENGTHS = (4, 8, 16)
WORKER_COUNT = 2
MAX_REPAIRS = 1

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
            max_turns=1,
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
    response: SingleProgramResponse,
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
        seen_behavior_signatures=(
            memory.seen_behavior_signatures + (behavior_signature,)
        ),
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
    request = build_single_program_request(
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    payload = {
        "instruction": (
            "Generate exactly one program for this slot using the specification "
            "retained at the fork boundary. Return only the structured one-program "
            "response. Do not repeat a program or behavior signature in Search Memory."
        ),
        "slot_id": slot_id,
        "brief_id": brief_id,
        "search_memory": memory.as_dict(),
    }
    prompt = request.prompt.split("\n\nRequest:\n", maxsplit=1)[0]
    return prompt + "\n\nRequest:\n" + _compact_json(payload)


def _repair_prompt(
    slot_id: str,
    brief_id: str,
    memory: SearchMemoryV1,
    error: str,
) -> str:
    return _compact_json(
        {
            "instruction": (
                "The host rejected the preceding AST. Generate exactly one corrected "
                "complete program for the same brief. Return only the structured "
                "one-program response and do not repeat the rejected AST."
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
        path.name == base or path.name.startswith(f"{base}.")
        for path in turns_dir.iterdir()
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


def _response(turns_dir: Path, prefix: str) -> SingleProgramResponse:
    return validate_single_program_response(
        (turns_dir / f"{prefix}.response.raw.txt").read_text(encoding="utf-8"),
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )


def _memory_from_reports(
    reports: Sequence[Mapping[str, Any]],
    turns_dir: Path,
) -> tuple[SearchMemoryV1, list[CohortEntry]]:
    memory = _empty_memory()
    entries: list[CohortEntry] = []
    for slot_index, raw in enumerate(reports):
        report = dict(raw)
        slot_id = str(report["slot_id"])
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
        response = _response(turns_dir, accepted_prefix)
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


def _manifest(model: str, effort: str) -> dict[str, Any]:
    manifest = build_epoch_manifest(model=model, effort=effort)
    system_prompt = build_single_program_request(
        slot_id=SLOT_IDS[0],
        brief_id=BRIEF_IDS[0],
        forbidden_lengths=FORBIDDEN_LENGTHS,
    ).system_prompt
    schema = build_single_program_output_schema(FORBIDDEN_LENGTHS)
    manifest.update(
        {
            "communication_mode": PERSISTENT_SINGLE_AST,
            "worker_count": WORKER_COUNT,
            "programs_per_turn": 1,
            "compaction_used": False,
            "rotation_policy": "fresh_spec_fork_plus_search_memory",
            "single_program_system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "single_program_output_schema_sha256": _hash(schema),
            "search_memory_schema_version": "mforge.native.search_memory.v1",
        }
    )
    manifest["provider_calls"] = [
        {
            "call_id": f"epoch-0000:single:{index:04d}",
            "slot_ids": [slot_id],
            "brief_id": BRIEF_IDS[index % len(BRIEF_IDS)],
            "worker_index": index % WORKER_COUNT,
        }
        for index, slot_id in enumerate(SLOT_IDS)
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
        for index, slot_id in enumerate(SLOT_IDS)
        for brief_id in (BRIEF_IDS[index % len(BRIEF_IDS)],)
    ]
    manifest["epoch_id"] = _hash(
        {
            "epoch_number": manifest["epoch_number"],
            "slots": manifest["slots"],
            "provider_calls": manifest["provider_calls"],
            "protocol_bundle_hash": manifest["protocol_bundle_hash"],
            "communication_mode": PERSISTENT_SINGLE_AST,
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
) -> dict[str, Any]:
    return {
        "schema_version": PREVIEW_STATE_SCHEMA_VERSION,
        "communication_mode": PERSISTENT_SINGLE_AST,
        "status": "generating",
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


def run_persistent_single_ast_cohort(
    experiment_root: str | Path,
    *,
    model: str,
    effort: str,
    timeout_seconds: float,
    auth_json: str | Path,
    backend_factory: Callable[[], GraphBackend],
    episode_id: str,
    adapter_factory: AdapterFactory = _default_adapter_factory,
    capsule_factory: CapsuleFactory | None = None,
    capsule_reopener: CapsuleReopener = IsolatedCapsule.reopen,
) -> dict[str, Any]:
    """Generate one AST per durable worker turn, then run the frozen evaluator."""

    root = Path(experiment_root)
    output_root = root / "native-v3-output" / "epoch-0000"
    turns_dir = root / "provider-turns"
    state_path = output_root / "communication-state.json.gz"
    manifest_path = output_root / "epoch-manifest.json.gz"
    turns_dir.mkdir(parents=True, exist_ok=True)
    profile = ModelProfile("codex", model, effort)
    request = build_single_program_request(
        slot_id=SLOT_IDS[0],
        brief_id=BRIEF_IDS[0],
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    system_prompt = request.system_prompt
    schema = request.output_schema

    if state_path.is_file():
        raw_state = read_json(state_path)
        if not isinstance(raw_state, Mapping):
            raise ValueError("preview communication state is not an object")
        state = {str(key): value for key, value in raw_state.items()}
        if (
            state.get("schema_version") != PREVIEW_STATE_SCHEMA_VERSION
            or state.get("communication_mode") != PERSISTENT_SINGLE_AST
        ):
            raise ValueError("preview communication state is incompatible")
        capsule = capsule_reopener(str(state["capsule_root"]))
    else:
        expected_manifest = _manifest(model, effort)
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
        setup = adapter_factory(
            system_prompt,
            capsule,
            turns_dir,
            anchor_prefix,
            timeout_seconds,
        )
        try:
            anchor = _run_adapter_turn(
                setup,
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
                    setup,
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
            source_identity = dict(setup.inspect_metadata())
        except Exception:
            setup.close(force=True)
            capsule.cleanup()
            raise
        else:
            setup.close()
        if not _artifact_complete(turns_dir, anchor.prefix):
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
            tuple(cast(Sequence[str], item["included_turn_ids"]))
            != (anchor.turn_id,)
            for item in fork_records
        ):
            capsule.cleanup()
            raise ValueError("worker fork crossed the specification boundary")
        state = _new_state(
            capsule,
            anchor,
            source_identity,
            [_worker_record(item) for item in fork_records],
        )
        write_json(state_path, state)

    raw_reports = state.get("slot_reports", [])
    if not isinstance(raw_reports, list):
        raise ValueError("preview slot reports are invalid")
    reports = [cast(Mapping[str, Any], item) for item in raw_reports]
    memory, entries = _memory_from_reports(reports, turns_dir)
    workers = cast(list[dict[str, Any]], state["workers"])
    model_turns = int(state["model_turns"])
    usages = [
        cast(Mapping[str, Any], item)
        for item in cast(list[Any], state["usages"])
    ]

    for slot_index in range(int(state["next_slot"]), len(SLOT_IDS)):
        slot_id = SLOT_IDS[slot_index]
        brief_id = BRIEF_IDS[slot_index % len(BRIEF_IDS)]
        worker_index = slot_index % WORKER_COUNT
        worker = workers[worker_index]
        attempts: list[dict[str, Any]] = []
        accepted: SingleProgramResponse | None = None
        accepted_prefix: str | None = None
        error: str | None = None

        for repair_index in range(MAX_REPAIRS + 1):
            prefix_base = (
                f"{slot_id}.initial"
                if repair_index == 0
                else f"{slot_id}.repair-{repair_index:02d}"
            )
            prefix = _available_prefix(turns_dir, prefix_base)
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
            adapter = adapter_factory(
                system_prompt,
                capsule,
                turns_dir,
                prefix,
                timeout_seconds,
            )
            try:
                adapter.resume_thread(
                    profile,
                    thread_id=str(worker["thread_id"]),
                    thread_path=str(worker["thread_path"]),
                )
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
                    program_response=True,
                )
            finally:
                adapter.close()
            model_turns += 1
            usages.append(observation.usage)
            worker["last_turn_id"] = observation.turn_id
            error = observation.error
            candidate: SingleProgramResponse | None = None
            duplicate_reason = None
            if observation.program_hash is not None:
                candidate = _response(turns_dir, prefix)
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
                    "program": (
                        None
                        if candidate is None
                        else validated_program_artifact(candidate.program)
                    ),
                },
                exclusive=True,
            )
            if not attempt["artifact_complete"]:
                raise ValueError(f"{prefix} artifact contract is incomplete")
            if candidate is not None:
                accepted = candidate
                accepted_prefix = prefix
                break

        if accepted is not None:
            memory = _extend_memory(memory, accepted, slot_index=slot_index)
            entry = CohortEntry(
                slot_id,
                accepted.program,
                accepted.design_summary,
                None,
            )
        else:
            entry = CohortEntry(slot_id, None, None, error or "invalid program")
        entries.append(entry)
        lineage = {
            "slot_id": slot_id,
            "brief_id": f"native-v3-brief-{slot_index:02d}",
            "parent_program_hashes": [],
            "worker_index": worker_index,
            "worker_thread_id": worker["thread_id"],
            "fork_parent_turn_id": worker["fork_parent_turn_id"],
            "provider_attempts": attempts,
        }
        slot_report = {
            "slot_id": slot_id,
            "brief_id": brief_id,
            "worker_index": worker_index,
            "accepted_prefix": accepted_prefix,
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
                "program": (
                    None
                    if accepted is None
                    else validated_program_artifact(accepted.program)
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

    slot_lineage = {
        str(report["slot_id"]): cast(Mapping[str, Any], report["lineage"])
        for report in reports
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
    )
    state.update({"status": "completed", "cohort_report": report["cohort_report"]})
    write_json(state_path, state)
    capsule.cleanup()
    return report


__all__ = [
    "FORBIDDEN_LENGTHS",
    "PERSISTENT_SINGLE_AST",
    "PREVIEW_PROGRAM_RECORD_SCHEMA_VERSION",
    "PREVIEW_STATE_SCHEMA_VERSION",
    "run_persistent_single_ast_cohort",
]
