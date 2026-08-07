"""Native v3 Step 12D lineage-fork and bounded Search Memory experiment."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import write_json
from mutation_forge.stage3.app_server import (
    CodexAppServerAdapter,
    ForkResult,
    ModelProfile,
)

from .compaction_experiment import build_reference_manifest
from .persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    BOOTSTRAP_ACK_VALUE,
    TurnObservation,
    _run_adapter_turn,
    _usage_keys,
    bootstrap_prompt,
    bootstrap_schema,
)
from .search_memory import (
    ActiveParentReference,
    DuplicateCandidateError,
    LineageSummary,
    PatternSummary,
    SearchMemoryV1,
    program_control_flow,
    program_families,
    reject_duplicate,
)
from .single_program_contract import (
    build_single_program_output_schema,
    build_single_program_request,
    validate_single_program_response,
)

LINEAGE_EXPERIMENT_SCHEMA_VERSION = "mforge.native.lineage_experiment.v1"
ACK_SCHEMA_VERSION = "mforge.native.lineage_ack.v1"
AdapterFactory = Callable[[str], CodexAppServerAdapter]


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ack_schema(ack: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "ack"],
        "properties": {
            "schema_version": {"type": "string", "const": ACK_SCHEMA_VERSION},
            "ack": {"type": "string", "const": ack},
        },
    }


def _ack(ack: str) -> dict[str, str]:
    return {"schema_version": ACK_SCHEMA_VERSION, "ack": ack}


def build_search_memory(
    candidate_responses: Mapping[str, Mapping[str, Any]],
    *,
    forbidden_lengths: tuple[int, ...],
) -> SearchMemoryV1:
    """Project validated host evidence into the exact bounded 12D memory."""

    reference = build_reference_manifest(
        candidate_responses,
        forbidden_lengths=forbidden_lengths,
    )
    successful: list[PatternSummary] = []
    tested: list[PatternSummary] = []
    lineages: list[LineageSummary] = []
    for index, raw in enumerate(reference["candidates"]):
        candidate = dict(raw)
        selectors, actions = program_families(candidate["canonical_ast"])
        accepted = candidate["evaluation_outcome"] in {"ACCEPTED", "ACTIVE_PARENT"}
        scientific_outcome = "ACCEPTED_IMPROVEMENT" if accepted else "REJECTED_NOT_PROVED"
        pattern = PatternSummary(
            pattern_id=f"pattern-{index:02d}",
            selector_families=selectors,
            action_families=actions,
            control_flow=program_control_flow(candidate["canonical_ast"]),
            summary=candidate["strategy_summary"],
            contract_status="VALID",
            scientific_outcome=scientific_outcome,
            model_hypothesis=candidate["strategy_summary"],
            observed_effect=(
                candidate["main_strength"] if accepted else candidate["main_weakness"]
            ),
            primary_failure_code=None,
            terminal_fallback_reason=None,
        )
        (successful if accepted else tested).append(pattern)
        lineages.append(
            LineageSummary(
                candidate_id=candidate["candidate_id"],
                parent_id=candidate["parent_id"] or None,
                program_hash=candidate["canonical_hash"],
                behavior_signature=candidate["behavior_signature"],
                generation=index,
                slot=index + 1,
                contract_status="VALID",
                scientific_outcome=scientific_outcome,
                summary=candidate["strategy_summary"],
            )
        )
    active = dict(reference["candidates"][-1])
    return SearchMemoryV1(
        protocol_hash=reference["protocol_hash"],
        seen_program_hashes=tuple(
            str(candidate["canonical_hash"]) for candidate in reference["candidates"]
        ),
        seen_behavior_signatures=tuple(
            str(candidate["behavior_signature"]) for candidate in reference["candidates"]
        ),
        successful_patterns=tuple(successful),
        tested_patterns=tuple(tested),
        pending_patterns=(),
        active_lineages=tuple(lineages),
        validated_archive_ids=tuple(
            str(candidate["candidate_id"]) for candidate in reference["candidates"]
        ),
        active_parent=ActiveParentReference(
            candidate_id=active["candidate_id"],
            program_hash=active["canonical_hash"],
        ),
    )


def _fixture_prompt(candidate: Mapping[str, Any], *, role: str) -> str:
    return _compact_json(
        {
            "instruction": (
                "Retain this validated parent program in thread history. "
                "Do not generate a program; return only the required acknowledgement."
            ),
            "role": role,
            "candidate_id": candidate["candidate_id"],
            "canonical_program": candidate["canonical_ast"],
        }
    )


def _feedback_prompt(parent: Mapping[str, Any]) -> str:
    return _compact_json(
        {
            "instruction": (
                "Retain this compact evaluation feedback for the exact parent in "
                "history. Do not generate a program."
            ),
            "parent_id": parent["candidate_id"],
            "evaluation_outcome": parent["evaluation_outcome"],
            "main_strength": parent["main_strength"],
            "main_weakness": parent["main_weakness"],
        }
    )


def _child_prompt(parent: Mapping[str, Any]) -> str:
    return _compact_json(
        {
            "instruction": (
                "Generate exactly one valid mutated child of the exact parent program "
                "retained in this fork's history. Change its selector/action mechanism, "
                "preserve evidence-backed strengths, and return only the structured "
                "one-program response. Before returning, verify that every reachable "
                "path terminates exactly once in emit or no_plan; use an explicit "
                "no_plan fallback for any branch that can fail to bind or apply."
            ),
            "parent_id": parent["candidate_id"],
            "forbid_exact_parent_duplicate": True,
        }
    )


def _child_repair_prompt(
    parent: Mapping[str, Any],
    validation_error: str,
) -> str:
    return _compact_json(
        {
            "instruction": (
                "The host rejected the preceding child. Generate one corrected mutated "
                "child of the exact parent retained earlier in this fork. Return a new "
                "complete direct one-program response. Verify every reachable path "
                "terminates exactly once in emit or no_plan and do not repeat the "
                "rejected AST."
            ),
            "parent_id": parent["candidate_id"],
            "host_validation_error": validation_error[:512],
        }
    )


def _memory_prompt(memory: SearchMemoryV1) -> str:
    return _compact_json(
        {
            "instruction": (
                "Retain this host Search Memory. It is advisory context only; the host "
                "remains authoritative for duplicate rejection. Do not generate a program."
            ),
            "search_memory": memory.model_facing_dict(),
        }
    )


def _fresh_prompt() -> str:
    return _compact_json(
        {
            "instruction": (
                "Generate exactly one valid fresh root from the retained specification "
                "and Search Memory. Use a structurally different selector/action family "
                "from recorded patterns. No prior program AST is present in this fork. "
                "Return only the structured one-program response."
            ),
            "parent_id": None,
            "avoid_recorded_strategies": True,
        }
    )


def _read_response(turns_dir: Path, prefix: str) -> object:
    return json.loads((turns_dir / f"{prefix}.response.raw.txt").read_text(encoding="utf-8"))


def _run_ack_turn(
    adapter: CodexAppServerAdapter,
    *,
    turns_dir: Path,
    prefix: str,
    prompt: str,
    system_prompt: str,
    profile: ModelProfile,
    forbidden_lengths: tuple[int, ...],
    ack: str,
) -> TurnObservation:
    observation = _run_adapter_turn(
        adapter,
        artifact_dir=turns_dir,
        prefix=prefix,
        prompt=prompt,
        system_prompt=system_prompt,
        schema=_ack_schema(ack),
        profile=profile,
        persistent=True,
        forbidden_lengths=forbidden_lengths,
        program_response=False,
    )
    if _read_response(turns_dir, prefix) != _ack(ack):
        raise ValueError(f"{prefix} acknowledgement does not match")
    return observation


def _prepare_fork_artifacts(
    adapter: CodexAppServerAdapter,
    *,
    turns_dir: Path,
    prefix: str,
    system_prompt: str,
    profile: ModelProfile,
    last_turn_id: str,
) -> None:
    adapter.rotate_logger(turns_dir, prefix)
    assert adapter.logger is not None
    adapter.logger.profile(
        {
            "model": profile.model,
            "effort": profile.effort,
            "ephemeral": False,
            "artifactPrefix": prefix,
        }
    )
    request = {
        "method": "thread/fork",
        "params": {
            "threadId": adapter.inspect_metadata()["threadId"],
            "lastTurnId": last_turn_id,
        },
    }
    adapter.logger.raw_text("request.md", _compact_json(request))
    adapter.logger.document("request.json", request)
    adapter.logger.raw_text("system-prompt.md", system_prompt)
    adapter.logger.document("output-schema.json", {})


def _finish_fork_artifacts(
    adapter: CodexAppServerAdapter,
    result: ForkResult,
) -> None:
    assert adapter.logger is not None
    response = {
        "source_thread_id": result.source_thread_id,
        "child_thread_id": result.child_thread_id,
        "session_id": result.session_id,
        "thread_path": result.thread_path,
        "last_turn_id": result.last_turn_id,
        "included_turn_ids": list(result.included_turn_ids),
    }
    text = _compact_json(response)
    adapter.logger.raw_text("response.raw.txt", text)
    adapter.logger.raw_text("response.md", text)
    adapter.logger.document("response.json", response)
    adapter.logger.document("provider-raw.json", response)
    adapter.logger.document(
        "usage.json",
        {key: 0 for key in _usage_keys()},
    )


def _fork(
    adapter: CodexAppServerAdapter,
    *,
    turns_dir: Path,
    prefix: str,
    system_prompt: str,
    profile: ModelProfile,
    last_turn_id: str,
) -> ForkResult:
    _prepare_fork_artifacts(
        adapter,
        turns_dir=turns_dir,
        prefix=prefix,
        system_prompt=system_prompt,
        profile=profile,
        last_turn_id=last_turn_id,
    )
    result = adapter.fork_persistent_thread(
        profile,
        last_turn_id=last_turn_id,
        activate=False,
    )
    _finish_fork_artifacts(adapter, result)
    return result


def _program_record(
    observation: TurnObservation,
    *,
    candidate_id: str,
    parent_id: str | None,
    source_thread_id: str,
    fork: ForkResult,
    generation: int,
    slot: int,
    feedback: Mapping[str, Any] | None,
    memory: SearchMemoryV1,
) -> dict[str, Any]:
    duplicate_reason = None
    if observation.program_hash is not None and observation.behavior_signature is not None:
        try:
            reject_duplicate(
                memory,
                program_hash=observation.program_hash,
                behavior_signature=observation.behavior_signature,
            )
        except DuplicateCandidateError as exc:
            duplicate_reason = str(exc)
    return {
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "source_thread_id": source_thread_id,
        "fork_thread_id": fork.child_thread_id,
        "fork_parent_turn_id": fork.last_turn_id,
        "generation": generation,
        "slot": slot,
        "feedback": None if feedback is None else dict(feedback),
        "program_hash": observation.program_hash,
        "behavior_signature": observation.behavior_signature,
        "valid_ast": observation.program_hash is not None,
        "duplicate_rejected": duplicate_reason is not None,
        "duplicate_reason": duplicate_reason,
        "turn_id": observation.turn_id,
    }


def _fork_record(fork: ForkResult) -> dict[str, Any]:
    return {
        "source_thread_id": fork.source_thread_id,
        "child_thread_id": fork.child_thread_id,
        "session_id": fork.session_id,
        "thread_path": fork.thread_path,
        "last_turn_id": fork.last_turn_id,
        "included_turn_ids": list(fork.included_turn_ids),
    }


def _usage_total(observations: list[TurnObservation]) -> dict[str, int]:
    return {key: sum(item.usage[key] for item in observations) for key in _usage_keys()}


def _markdown_report(report: Mapping[str, Any]) -> str:
    child = report["lineage_child"]
    fresh = report["fresh_root"]
    return "\n".join(
        [
            "# Native v3 Step 12D lineage and Search Memory result",
            "",
            f"Status: **{report['status']}**",
            "",
            f"- Child AST valid: {child['valid_ast']}",
            f"- Fresh-root AST valid: {fresh['valid_ast']}",
            f"- Fresh-root behavior diverse: {report['fresh_root_diverse']}",
            f"- Child fork boundary exact: {report['boundary_proofs']['child_exact']}",
            f"- Fresh fork boundary exact: {report['boundary_proofs']['fresh_exact']}",
            f"- Search Memory canonical bytes: {report['search_memory']['canonical_bytes']}",
            f"- Duplicate rejection rate: {report['duplicate_rejection_rate']}",
            "",
            "Context compaction was not used because Step 12C was BEST_EFFORT_ONLY.",
            "",
        ]
    )


def run_lineage_experiment(
    workspace: str | Path,
    *,
    model: str,
    effort: str,
    forbidden_lengths: tuple[int, ...],
    candidate_responses: Mapping[str, Mapping[str, Any]],
    adapter_factory: AdapterFactory,
) -> dict[str, Any]:
    """Run one exact-parent child fork and one fresh specification-anchor fork."""

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    turns_dir = root / "provider-turns"
    turns_dir.mkdir()
    profile = ModelProfile("codex", model, effort)
    request = build_single_program_request(
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=forbidden_lengths,
    )
    system_prompt = request.system_prompt
    schema = build_single_program_output_schema(forbidden_lengths)
    reference = build_reference_manifest(
        candidate_responses,
        forbidden_lengths=forbidden_lengths,
    )
    memory = build_search_memory(
        candidate_responses,
        forbidden_lengths=forbidden_lengths,
    )
    parent = dict(reference["candidates"][-1])
    sibling = dict(reference["candidates"][2])
    observations: list[TurnObservation] = []
    adapter = adapter_factory(system_prompt)
    started = time.monotonic()
    try:
        anchor = _run_adapter_turn(
            adapter,
            artifact_dir=turns_dir,
            prefix="00-spec-anchor",
            prompt=bootstrap_prompt(forbidden_lengths),
            system_prompt=system_prompt,
            schema=bootstrap_schema(),
            profile=profile,
            persistent=True,
            forbidden_lengths=forbidden_lengths,
            program_response=False,
        )
        observations.append(anchor)
        expected_bootstrap = {
            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
            "ack": BOOTSTRAP_ACK_VALUE,
        }
        if _read_response(turns_dir, "00-spec-anchor") != expected_bootstrap:
            raise ValueError("specification anchor acknowledgement does not match")

        parent_turn = _run_ack_turn(
            adapter,
            turns_dir=turns_dir,
            prefix="01-root-parent",
            prompt=_fixture_prompt(parent, role="root-parent"),
            system_prompt=system_prompt,
            profile=profile,
            forbidden_lengths=forbidden_lengths,
            ack="root-parent",
        )
        observations.append(parent_turn)
        sibling_turn = _run_ack_turn(
            adapter,
            turns_dir=turns_dir,
            prefix="02-later-sibling",
            prompt=_fixture_prompt(sibling, role="later-sibling"),
            system_prompt=system_prompt,
            profile=profile,
            forbidden_lengths=forbidden_lengths,
            ack="later-sibling",
        )
        observations.append(sibling_turn)
        source_identity = dict(adapter.inspect_metadata())

        child_fork = _fork(
            adapter,
            turns_dir=turns_dir,
            prefix="03-child-fork",
            system_prompt=system_prompt,
            profile=profile,
            last_turn_id=parent_turn.turn_id,
        )
        fresh_fork = _fork(
            adapter,
            turns_dir=turns_dir,
            prefix="04-fresh-fork",
            system_prompt=system_prompt,
            profile=profile,
            last_turn_id=anchor.turn_id,
        )
        child_expected = (anchor.turn_id, parent_turn.turn_id)
        fresh_expected = (anchor.turn_id,)
        child_exact = child_fork.included_turn_ids == child_expected
        fresh_exact = fresh_fork.included_turn_ids == fresh_expected
        if not child_exact or sibling_turn.turn_id in child_fork.included_turn_ids:
            raise ValueError("child fork crossed the exact-parent boundary")
        if (
            not fresh_exact
            or parent_turn.turn_id in fresh_fork.included_turn_ids
            or sibling_turn.turn_id in fresh_fork.included_turn_ids
        ):
            raise ValueError("fresh fork crossed the specification-anchor boundary")

        adapter.activate_forked_thread(child_fork.child_thread_id)
        feedback = {
            "evaluation_outcome": parent["evaluation_outcome"],
            "main_strength": parent["main_strength"],
            "main_weakness": parent["main_weakness"],
        }
        feedback_turn = _run_ack_turn(
            adapter,
            turns_dir=turns_dir,
            prefix="05-child-feedback",
            prompt=_feedback_prompt(parent),
            system_prompt=system_prompt,
            profile=profile,
            forbidden_lengths=forbidden_lengths,
            ack="child-feedback",
        )
        observations.append(feedback_turn)
        child_prompt = _child_prompt(parent)
        child_observation = _run_adapter_turn(
            adapter,
            artifact_dir=turns_dir,
            prefix="06-child-mutation",
            prompt=child_prompt,
            system_prompt=system_prompt,
            schema=schema,
            profile=profile,
            persistent=True,
            forbidden_lengths=forbidden_lengths,
            program_response=True,
        )
        observations.append(child_observation)
        child_attempts = [child_observation.as_dict()]
        child_repair_prompt = None
        if child_observation.program_hash is None:
            if not child_observation.error:
                raise ValueError("invalid child has no host validation error")
            child_repair_prompt = _child_repair_prompt(
                parent,
                child_observation.error,
            )
            child_observation = _run_adapter_turn(
                adapter,
                artifact_dir=turns_dir,
                prefix="06-child-repair",
                prompt=child_repair_prompt,
                system_prompt=system_prompt,
                schema=schema,
                profile=profile,
                persistent=True,
                forbidden_lengths=forbidden_lengths,
                program_response=True,
            )
            observations.append(child_observation)
            child_attempts.append(child_observation.as_dict())

        adapter.activate_forked_thread(fresh_fork.child_thread_id)
        memory_prompt = _memory_prompt(memory)
        if "canonical_program" in memory_prompt or '"entry"' in memory_prompt:
            raise ValueError("Search Memory prompt contains a full program AST")
        memory_turn = _run_ack_turn(
            adapter,
            turns_dir=turns_dir,
            prefix="07-search-memory",
            prompt=memory_prompt,
            system_prompt=system_prompt,
            profile=profile,
            forbidden_lengths=forbidden_lengths,
            ack="search-memory",
        )
        observations.append(memory_turn)
        fresh_prompt = _fresh_prompt()
        fresh_observation = _run_adapter_turn(
            adapter,
            artifact_dir=turns_dir,
            prefix="08-fresh-root",
            prompt=fresh_prompt,
            system_prompt=system_prompt,
            schema=schema,
            profile=profile,
            persistent=True,
            forbidden_lengths=forbidden_lengths,
            program_response=True,
        )
        observations.append(fresh_observation)
    finally:
        adapter.close()

    source_thread_id = anchor.thread_id
    child_record = _program_record(
        child_observation,
        candidate_id="g0004-s00",
        parent_id=parent["candidate_id"],
        source_thread_id=source_thread_id,
        fork=child_fork,
        generation=4,
        slot=0,
        feedback=feedback,
        memory=memory,
    )
    fresh_record = _program_record(
        fresh_observation,
        candidate_id="g0004-s01",
        parent_id=None,
        source_thread_id=source_thread_id,
        fork=fresh_fork,
        generation=4,
        slot=1,
        feedback=None,
        memory=memory,
    )
    fresh_diverse = (
        fresh_record["valid_ast"]
        and not fresh_record["duplicate_rejected"]
        and fresh_record["behavior_signature"] not in memory.seen_behavior_signatures
    )
    duplicate_rejections = sum(item["duplicate_rejected"] for item in (child_record, fresh_record))
    status = (
        "completed"
        if (
            child_record["valid_ast"]
            and fresh_record["valid_ast"]
            and not child_record["duplicate_rejected"]
            and fresh_diverse
        )
        else "scientific_failure"
    )
    report = {
        "schema_version": LINEAGE_EXPERIMENT_SCHEMA_VERSION,
        "status": status,
        "model": model,
        "effort": effort,
        "protocol_hash": memory.protocol_hash,
        "compaction_used": False,
        "compaction_reason": "Step 12C classification was BEST_EFFORT_ONLY",
        "source_identity": {
            "thread_id": source_thread_id,
            "session_id": source_identity["sessionId"],
            "thread_path": source_identity["threadPath"],
        },
        "forks": {
            "lineage_child": _fork_record(child_fork),
            "fresh_root": _fork_record(fresh_fork),
        },
        "boundary_proofs": {
            "child_exact": child_exact,
            "child_included_turn_ids": list(child_fork.included_turn_ids),
            "child_excluded_later_sibling_turn_id": sibling_turn.turn_id,
            "fresh_exact": fresh_exact,
            "fresh_included_turn_ids": list(fresh_fork.included_turn_ids),
            "fresh_excluded_program_turn_ids": [
                parent_turn.turn_id,
                sibling_turn.turn_id,
            ],
        },
        "search_memory": {
            "sha256": memory.sha256,
            "canonical_bytes": len(memory.canonical_bytes()),
            "successful_pattern_count": len(memory.successful_patterns),
            "tested_pattern_count": len(memory.tested_patterns),
            "pending_pattern_count": len(memory.pending_patterns),
            "full_ast_injected": False,
            "value": memory.as_dict(),
        },
        "lineage_child": child_record,
        "child_attempts": child_attempts,
        "child_repair_used": len(child_attempts) == 2,
        "fresh_root": fresh_record,
        "fresh_root_diverse": fresh_diverse,
        "prompt_bytes": {
            "child_mutation": len(child_prompt.encode("utf-8")),
            "child_repair": (
                None if child_repair_prompt is None else len(child_repair_prompt.encode("utf-8"))
            ),
            "search_memory": len(memory_prompt.encode("utf-8")),
            "fresh_root": len(fresh_prompt.encode("utf-8")),
        },
        "usage": _usage_total(observations),
        "provider_turns": len(observations),
        "fork_rpc_count": 2,
        "duplicate_rejections": duplicate_rejections,
        "duplicate_rejection_rate": duplicate_rejections / 2,
        "wall_time_ms": round((time.monotonic() - started) * 1000),
    }
    write_json(root / "lineage-report.json.gz", report)
    (root / "lineage-report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    if child_observation.program_hash is not None:
        child_validated = validate_single_program_response(
            (turns_dir / f"{child_observation.prefix}.response.raw.txt").read_text(
                encoding="utf-8"
            ),
            forbidden_lengths=forbidden_lengths,
        )
        write_json(root / "child-program.json.gz", child_validated.program.ast)
    if fresh_observation.program_hash is not None:
        fresh_validated = validate_single_program_response(
            (turns_dir / "08-fresh-root.response.raw.txt").read_text(encoding="utf-8"),
            forbidden_lengths=forbidden_lengths,
        )
        write_json(root / "fresh-root-program.json.gz", fresh_validated.program.ast)
    return report
