"""Deterministic sequential eight-slot Native v3 cohort."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from mutation_forge.artifacts import git_state
from mutation_forge.backends.base import GraphBackend
from mutation_forge.counterexamples import CounterexamplePipeline
from mutation_forge.experiment.artifacts import (
    NATIVE_V3_PROGRAM_BATCH_PROJECTION,
    TurnArtifactStore,
    generated_policy_diagnostics,
)
from mutation_forge.experiment.generation import GenerationRequest
from mutation_forge.experiment.json_io import write_json
from mutation_forge.experiment.provider import (
    AuthenticationError,
    LocalCodexAppServerProvider,
)

from .canonical import (
    CANONICAL_PROTOCOL_ID,
    CanonicalJsonError,
    canonical_json_bytes,
    parse_strict_json,
)
from .contracts import (
    ACTION_ARGUMENT_TYPES,
    CTX_TYPES,
    FEATURE_TYPES,
    SELECTOR_ARGUMENT_TYPES,
    SELECTOR_TYPES,
    VALIDATOR_PROTOCOL_ID,
    ValidatedProgram,
    validate_program,
    validated_program_artifact,
)
from .graph_runtime import GRAPH_RUNTIME_PROTOCOL_ID
from .heg_scoring import scorer_for_backend
from .interpreter import INTERPRETER_PROTOCOL_ID
from .scoring import (
    FITNESS_PROTOCOL_ID,
    SCORE_PROTOCOL_ID,
    RationalInterval,
    conservative_fitness_key,
)
from .serial_evaluator import (
    SERIAL_EVALUATOR_PROTOCOL_ID,
    SerialEpisodeConfig,
    SerialEpisodeResult,
    SerialEvaluationStatus,
    evaluate_serial_program,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COHORT_PROTOCOL_ID = "native_v3_sequential_cohort_v1"
COHORT_SCHEMA_VERSION = "mforge.native-v3.cohort.v1"
EPOCH_MANIFEST_SCHEMA_VERSION = "mforge.native-v3.epoch-manifest.v1"
PROGRAM_BATCH_SCHEMA_VERSION = "mforge.native.program_batch.v3"
PROVIDER_INPUT_PROFILE_ID = "native_v3_input_4ast_v1"
PROVIDER_OUTPUT_PROFILE_ID = "native_v3_output_4ast_v1"
MAXIMUM_RESPONSE_BYTES = 320 * 1024

_BRIEFS = (
    "Explore a degree-changing add-edge strategy with low local cycle risk.",
    "Explore a safe remove-edge strategy that avoids structural bridges.",
    "Explore endpoint relocation while preserving a useful dense core.",
    "Explore fanout or fold operations that change the degree vector.",
    "Explore 2-, 3-, or 4-switches as one part of a broader strategy.",
    "Adapt operator choice to stagnation and the exploration-window context.",
    "Combine witness-load selectors with spatial or articulation-risk selectors.",
    "Favor a compact, diverse mechanism unlike the supplied parents.",
)
SLOT_IDS = tuple(f"slot-{index:02d}" for index in range(8))
PROVIDER_PARTITION = (SLOT_IDS[:4], SLOT_IDS[4:])


class NativeV3CohortError(RuntimeError):
    """The frozen cohort could not complete its bounded protocol."""


@dataclass(frozen=True, slots=True)
class CohortEntry:
    slot_id: str
    program: ValidatedProgram | None
    design_summary: str | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "program_hash": (
                self.program.program_hash if self.program is not None else None
            ),
            "design_summary": self.design_summary,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ParsedBatch:
    envelope: dict[str, Any] | None
    entries: tuple[CohortEntry, ...]


def _load_text(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _load_json(relative: str) -> dict[str, Any]:
    value = json.loads(_load_text(relative))
    if not isinstance(value, dict):
        raise NativeV3CohortError(f"{relative} is not a JSON object")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _semantic_registry() -> dict[str, Any]:
    return {
        "selectors": {
            selector_id: {
                "result": str(SELECTOR_TYPES[selector_id]),
                "arguments": {
                    name: str(value_type)
                    for name, value_type in SELECTOR_ARGUMENT_TYPES[
                        selector_id
                    ].items()
                },
            }
            for selector_id in sorted(SELECTOR_TYPES)
        },
        "actions": {
            action_id: {
                name: str(value_type)
                for name, value_type in ACTION_ARGUMENT_TYPES[action_id].items()
            }
            for action_id in sorted(ACTION_ARGUMENT_TYPES)
        },
        "context_fields": {
            name: str(value_type) for name, value_type in sorted(CTX_TYPES.items())
        },
        "graph_features": {
            name: str(value_type)
            for name, value_type in sorted(FEATURE_TYPES.items())
        },
    }


def render_batch_prompt(slot_ids: Sequence[str]) -> str:
    """Render only model-useful semantics for one frozen provider batch."""

    request = _load_text("prompts/native-v3/cohort-request.md").rstrip()
    payload = {
        "slots": [
            {
                "slot_id": slot_id,
                "brief": _BRIEFS[SLOT_IDS.index(slot_id)],
                "parent_programs": [],
            }
            for slot_id in slot_ids
        ],
        "program_schema": _load_json(
            "configs/native/native-v3-program.schema.json"
        ),
        "semantic_registry": _semantic_registry(),
    }
    return (
        request
        + "\n\nRequested batch and executable contract:\n\n"
        + json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
    )


def _protocols() -> dict[str, str]:
    return {
        "cohort": COHORT_PROTOCOL_ID,
        "canonical_json": CANONICAL_PROTOCOL_ID,
        "validator": VALIDATOR_PROTOCOL_ID,
        "interpreter": INTERPRETER_PROTOCOL_ID,
        "graph_runtime": GRAPH_RUNTIME_PROTOCOL_ID,
        "serial_evaluator": SERIAL_EVALUATOR_PROTOCOL_ID,
        "score_evidence": SCORE_PROTOCOL_ID,
        "fitness": FITNESS_PROTOCOL_ID,
        "provider_input": PROVIDER_INPUT_PROFILE_ID,
        "provider_output": PROVIDER_OUTPUT_PROFILE_ID,
    }


def build_epoch_manifest(*, model: str, effort: str) -> dict[str, Any]:
    """Freeze the complete epoch-zero cohort before provider contact."""

    calls = []
    for index, slot_ids in enumerate(PROVIDER_PARTITION):
        prompt = render_batch_prompt(slot_ids)
        calls.append(
            {
                "call_id": f"epoch-0000:provider:{index:04d}",
                "slot_ids": list(slot_ids),
                "prompt_sha256": _sha256_text(prompt),
            }
        )
    protocols = _protocols()
    protocol_bundle_hash = hashlib.sha256(
        canonical_json_bytes(protocols)
    ).hexdigest()
    slots = [
        {
            "slot_id": slot_id,
            "parent_program_hashes": [],
            "brief_id": f"native-v3-brief-{index:02d}",
            "brief": _BRIEFS[index],
            "brief_sha256": _sha256_text(_BRIEFS[index]),
        }
        for index, slot_id in enumerate(SLOT_IDS)
    ]
    identity = {
        "epoch_number": 0,
        "slots": slots,
        "provider_calls": calls,
        "protocol_bundle_hash": protocol_bundle_hash,
        "model": model,
        "effort": effort,
    }
    epoch_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return {
        "schema_version": EPOCH_MANIFEST_SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "epoch_number": 0,
        "planned_slot_ids": list(SLOT_IDS),
        "slots": slots,
        "provider_calls": calls,
        "model": model,
        "effort": effort,
        "system_prompt_sha256": _sha256_text(
            _load_text("prompts/native-v3/cohort-system.md")
        ),
        "repair_prompt_sha256": _sha256_text(
            _load_text("prompts/native-v3/cohort-repair.md")
        ),
        "output_schema_sha256": hashlib.sha256(
            canonical_json_bytes(
                _load_json(
                    "configs/native/native-v3-cohort-envelope.schema.json"
                )
            )
        ).hexdigest(),
        "protocols": protocols,
        "protocol_bundle_hash": protocol_bundle_hash,
    }


def build_batch_request(
    experiment_root: str | Path,
    *,
    call_index: int,
    model: str,
    effort: str,
) -> dict[str, Any]:
    """Build one of the two immutable Native v2 transport requests."""

    try:
        slot_ids = PROVIDER_PARTITION[call_index]
    except IndexError as exc:
        raise ValueError("call_index must be 0 or 1") from exc
    call_id = f"epoch-0000:provider:{call_index:04d}"
    prompt = render_batch_prompt(slot_ids)
    prompt_hash = _sha256_text(prompt)
    idempotency_key = hashlib.sha256(
        canonical_json_bytes(
            {
                "call_id": call_id,
                "prompt_sha256": prompt_hash,
                "model": model,
                "effort": effort,
            }
        )
    ).hexdigest()
    request = GenerationRequest(
        campaign_id="v3-cohort",
        generation=0,
        slot=slot_ids[0],
        parent_id="native-v3-empty-parent-set",
        brief_id=call_id,
        prompt=prompt,
        prompt_hash=prompt_hash,
        idempotency_key=idempotency_key,
        model=model,
        effort=effort,
        system_prompt=_load_text("prompts/native-v3/cohort-system.md"),
        output_schema=_load_json(
            "configs/native/native-v3-cohort-envelope.schema.json"
        ),
        repair_prompt=_load_text("prompts/native-v3/cohort-repair.md"),
        max_repairs=1,
        remaining_repairs=1,
    ).as_dict()
    artifacts = Path(experiment_root) / "artifacts"
    turn = TurnArtifactStore(artifacts).turn_directory(0, slot_ids[0])
    request.update(
        {
            "artifact_dir": str(turn),
            "artifact_root": str(artifacts),
            "artifact_prefix": slot_ids[0],
            "call_id": call_id,
            "slot_ids": list(slot_ids),
            "reasoning_effort": effort,
            "response_projection": NATIVE_V3_PROGRAM_BATCH_PROJECTION,
        }
    )
    return request


def _invalid_entries(
    slot_ids: Sequence[str],
    message: str,
) -> tuple[CohortEntry, ...]:
    return tuple(CohortEntry(slot_id, None, None, message) for slot_id in slot_ids)


def parse_batch_response(
    response_text: str,
    slot_ids: Sequence[str],
) -> ParsedBatch:
    """Validate siblings independently and restore frozen slot order."""

    try:
        outer_value = parse_strict_json(
            response_text,
            maximum_bytes=MAXIMUM_RESPONSE_BYTES,
        )
    except CanonicalJsonError as exc:
        return ParsedBatch(
            None,
            _invalid_entries(slot_ids, f"invalid provider JSON: {exc}"),
        )
    if not isinstance(outer_value, Mapping):
        return ParsedBatch(
            None,
            _invalid_entries(slot_ids, "provider response is not an object"),
        )
    envelope = {str(key): value for key, value in outer_value.items()}
    diagnostics = generated_policy_diagnostics(envelope)
    if diagnostics:
        summary = "; ".join(str(item["message"]) for item in diagnostics)
        return ParsedBatch(
            envelope,
            _invalid_entries(slot_ids, f"invalid v2 provider envelope: {summary}"),
        )
    source = envelope.get("source")
    if not isinstance(source, str):
        return ParsedBatch(
            envelope,
            _invalid_entries(slot_ids, "provider envelope source is not a string"),
        )
    try:
        batch_value = parse_strict_json(
            source,
            maximum_bytes=MAXIMUM_RESPONSE_BYTES,
        )
    except CanonicalJsonError as exc:
        return ParsedBatch(
            envelope,
            _invalid_entries(slot_ids, f"invalid batch JSON: {exc}"),
        )
    if (
        not isinstance(batch_value, Mapping)
        or set(batch_value) != {"schema_version", "programs"}
        or batch_value.get("schema_version") != PROGRAM_BATCH_SCHEMA_VERSION
        or not isinstance(batch_value.get("programs"), list)
    ):
        return ParsedBatch(
            envelope,
            _invalid_entries(slot_ids, "invalid provider batch envelope"),
        )
    raw_by_slot: dict[str, Mapping[str, Any]] = {}
    duplicate_slots: set[str] = set()
    for item in cast(list[object], batch_value["programs"]):
        if not isinstance(item, Mapping) or set(item) != {
            "slot_id",
            "program_json_raw",
            "design_summary",
        }:
            continue
        slot_id = item.get("slot_id")
        if not isinstance(slot_id, str) or slot_id not in slot_ids:
            continue
        if slot_id in raw_by_slot:
            duplicate_slots.add(slot_id)
        else:
            raw_by_slot[slot_id] = item

    entries: list[CohortEntry] = []
    for slot_id in slot_ids:
        if slot_id in duplicate_slots:
            entries.append(
                CohortEntry(slot_id, None, None, "duplicate slot in provider batch")
            )
            continue
        item = raw_by_slot.get(slot_id)
        if item is None:
            entries.append(
                CohortEntry(slot_id, None, None, "provider omitted planned slot")
            )
            continue
        design_summary = item.get("design_summary")
        if (
            not isinstance(design_summary, str)
            or not design_summary
            or len(design_summary) > 2048
        ):
            entries.append(
                CohortEntry(
                    slot_id,
                    None,
                    None,
                    "design_summary violates the provider contract",
                )
            )
            continue
        raw = item.get("program_json_raw")
        if not isinstance(raw, str):
            entries.append(
                CohortEntry(
                    slot_id,
                    None,
                    design_summary,
                    "program_json_raw must be a string",
                )
            )
            continue
        validation = validate_program(raw)
        if validation.program is None:
            summary = "; ".join(
                f"{item.code}@{item.path}: {item.message}"
                for item in validation.diagnostics
            )
            entries.append(
                CohortEntry(slot_id, None, design_summary, summary)
            )
        else:
            entries.append(
                CohortEntry(slot_id, validation.program, design_summary, None)
            )
    return ParsedBatch(envelope, tuple(entries))


def cohort_outcome(unique_valid_programs: int) -> str:
    if unique_valid_programs == 8:
        return "COMPLETE"
    if 4 <= unique_valid_programs <= 7:
        return "DEGRADED"
    if 0 <= unique_valid_programs <= 3:
        return "INCONCLUSIVE"
    raise ValueError("unique_valid_programs must be between 0 and 8")


def deduplicate_entries(
    entries: Sequence[CohortEntry],
) -> tuple[tuple[ValidatedProgram, ...], dict[str, tuple[str, ...]]]:
    """Return unique programs in canonical slot/program order with all aliases."""

    by_hash: dict[str, ValidatedProgram] = {}
    aliases: dict[str, list[str]] = {}
    for entry in sorted(entries, key=lambda value: value.slot_id):
        if entry.program is None:
            continue
        program_hash = entry.program.program_hash
        by_hash.setdefault(program_hash, entry.program)
        aliases.setdefault(program_hash, []).append(entry.slot_id)
    ordered_hashes = sorted(
        by_hash,
        key=lambda program_hash: (min(aliases[program_hash]), program_hash),
    )
    return (
        tuple(by_hash[program_hash] for program_hash in ordered_hashes),
        {
            program_hash: tuple(sorted(aliases[program_hash]))
            for program_hash in ordered_hashes
        },
    )


def _usage_total(usages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "inputTokens",
        "cachedInputTokens",
        "cacheWriteInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    return {
        **{
            field: sum(
                int(usage.get(field, 0))
                for usage in usages
                if isinstance(usage.get(field, 0), int)
            )
            for field in fields
        },
        "final": all(usage.get("final") is True for usage in usages),
        "partial": any(usage.get("partial") is True for usage in usages),
    }


def _failure_classification(error: Exception) -> str:
    message = str(error).lower()
    if (
        isinstance(error, AuthenticationError)
        or "auth" in type(error).__name__.lower()
        or "auth" in message
        or "login" in message
    ):
        return "authentication"
    return "provider"


def _turn_result(
    raw_result: Mapping[str, Any],
    parsed: ParsedBatch,
) -> dict[str, Any]:
    valid = [entry for entry in parsed.entries if entry.program is not None]
    errors = [
        f"{entry.slot_id}: {entry.error}"
        for entry in parsed.entries
        if entry.error is not None
    ]
    result = {
        **dict(raw_result),
        "validation": {
            "valid": False,
            "errors": errors
            or ["Native v2 Python validation is not applicable to a Native v3 batch"],
        },
        "validation_completed": True,
        "uncharged": not bool(raw_result.get("charged")),
        "identity": {
            "valid_entry_count": len(valid),
            "program_hashes": sorted(
                entry.program.program_hash
                for entry in valid
                if entry.program is not None
            ),
            "validator_version": VALIDATOR_PROTOCOL_ID,
        },
        "provenance": {
            "effort": raw_result.get("effort"),
            "model": raw_result.get("model"),
            "provider_request_id": raw_result.get("provider_request_id"),
            "provider_thread_id": raw_result.get("provider_thread_id"),
            "provider_turn_id": raw_result.get("provider_turn_id"),
            "transport_sha256": raw_result.get("transport_sha256"),
        },
    }
    if parsed.envelope is not None:
        result.update(
            {
                "response": parsed.envelope,
                "canonical_response": parsed.envelope,
            }
        )
    else:
        result["status"] = "invalid_output"
    return result


def _record_turn(
    store: TurnArtifactStore,
    request: Mapping[str, Any],
    raw_result: Mapping[str, Any],
    parsed: ParsedBatch,
    *,
    phase: str,
) -> tuple[Path, Mapping[str, Any]]:
    turn = Path(str(request["artifact_dir"]))
    result = _turn_result(raw_result, parsed)
    usage = result.get("usage")
    prefix = store.artifact_prefix(
        turn,
        result,
        str(request["artifact_prefix"]),
    )
    usage_path = turn / f"{prefix}.usage.json.gz"
    if isinstance(usage, Mapping) and not usage_path.exists():
        write_json(usage_path, usage, exclusive=True)
    manifest = store.record_existing_turn(
        turn,
        generation=0,
        slot=str(request["slot"]),
        phase=phase,
        request=request,
        result=result,
    )
    store.verify_turn(turn)
    return turn, manifest


def record_batch_turn(
    store: TurnArtifactStore,
    request: Mapping[str, Any],
    raw_result: Mapping[str, Any],
    parsed: ParsedBatch,
    *,
    phase: str,
) -> tuple[Path, Mapping[str, Any]]:
    """Finalize one provider batch through the production turn-artifact writer."""

    return _record_turn(
        store,
        request,
        raw_result,
        parsed,
        phase=phase,
    )


def _graph_evaluations(result: SerialEpisodeResult) -> int:
    return result.score_attempts


def _attempt_reference(
    raw_result: Mapping[str, Any],
    *,
    phase: str,
    turn: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "turn_directory": str(turn),
        "artifact_complete": manifest.get("artifact_complete") is True,
        "provider_request_id": raw_result.get("provider_request_id"),
        "provider_thread_id": raw_result.get("provider_thread_id"),
        "provider_turn_id": raw_result.get("provider_turn_id"),
        "transport_sha256": raw_result.get("transport_sha256"),
    }


def finalize_cohort(
    experiment_root: str | Path,
    *,
    entries: Sequence[CohortEntry],
    generation_reports: Sequence[Mapping[str, Any]],
    slot_lineage: Mapping[str, Mapping[str, Any]],
    usages: Sequence[Mapping[str, Any]],
    model_turns: int,
    backend_factory: Callable[[], GraphBackend],
    episode_id: str,
    communication_mode: str,
) -> dict[str, Any]:
    """Evaluate an already generated cohort without changing scientific semantics."""

    root = Path(experiment_root)
    output_root = root / "native-v3-output" / "epoch-0000"
    report_path = output_root / "cohort-report.json.gz"
    programs, aliases = deduplicate_entries(entries)
    outcome = cohort_outcome(len(programs))
    backend: GraphBackend | None = None
    evaluation_records: list[dict[str, Any]] = []
    graph_evaluations = 0
    try:
        if programs:
            backend = backend_factory()
            scorer = scorer_for_backend(backend)
            for evaluation_index, program in enumerate(programs):
                program_root = output_root / "programs" / program.program_hash
                program_payload: dict[str, Any] = {
                    **validated_program_artifact(program),
                    "slot_aliases": list(aliases[program.program_hash]),
                    "lineage": [
                        slot_lineage[slot_id]
                        for slot_id in aliases[program.program_hash]
                    ],
                }
                write_json(
                    program_root / "program.json.gz",
                    program_payload,
                    exclusive=True,
                )
                evaluation = evaluate_serial_program(
                    backend=backend,
                    scorer=scorer,
                    program=program,
                    config=SerialEpisodeConfig(
                        order=30,
                        graph_seed=101,
                        policy_seed=17,
                        horizon=1,
                        witness_cap=64,
                        episode_id=(
                            f"{episode_id}/program-{evaluation_index:02d}"
                        ),
                    ),
                    counterexample_pipeline=CounterexamplePipeline(
                        backend=backend,
                        artifact_root=program_root,
                    ),
                    provenance_source_kind="native_v3_provider_cohort",
                )
                graph_evaluations += _graph_evaluations(evaluation)
                record = {
                    "program_hash": program.program_hash,
                    "slot_aliases": list(aliases[program.program_hash]),
                    "evaluation_index": evaluation_index,
                    "evaluation": evaluation.as_dict(),
                }
                evaluation_records.append(record)
                write_json(
                    program_root / "evaluation.json.gz",
                    {
                        "schema_version": COHORT_SCHEMA_VERSION,
                        "protocols": _protocols(),
                        "program": validated_program_artifact(program),
                        "slot_aliases": list(aliases[program.program_hash]),
                        "lineage": [
                            slot_lineage[slot_id]
                            for slot_id in aliases[program.program_hash]
                        ],
                        "backend": {
                            "graph_backend_id": backend.backend_id,
                            "score_implementation": getattr(
                                backend,
                                "score_implementation",
                                None,
                            ),
                            "repo": str(getattr(backend, "repo", ""))
                            or None,
                            "commit": getattr(backend, "commit", None),
                            "dirty": getattr(backend, "dirty", None),
                        },
                        "mutation_forge": git_state(PROJECT_ROOT),
                        "evaluation": evaluation.as_dict(),
                    },
                    exclusive=True,
                )
    except Exception as error:
        report = {
            "schema_version": COHORT_SCHEMA_VERSION,
            "status": "evaluation_error",
            "communication_mode": communication_mode,
            "cohort_outcome": outcome,
            "error_type": type(error).__name__,
            "error": str(error),
            "valid_ast": bool(programs),
            "valid_slots": sum(entry.program is not None for entry in entries),
            "unique_valid_programs": len(programs),
            "duplicate_aliases": sum(
                max(0, len(slot_aliases) - 1)
                for slot_aliases in aliases.values()
            ),
            "model_turns": model_turns,
            "graph_evaluations": graph_evaluations,
            "scientific_terminal_result": False,
            "usage": _usage_total(usages),
            "epoch_manifest": str(output_root / "epoch-manifest.json.gz"),
            "cohort_report": str(report_path),
            "resumable": False,
        }
        write_json(report_path, report)
        return report
    finally:
        if backend is not None:
            backend.close()

    def selection_key(
        record: Mapping[str, Any],
    ) -> tuple[Fraction, bool, Fraction, Fraction, str]:
        evaluation = cast(Mapping[str, Any], record["evaluation"])
        fitness = cast(Mapping[str, Any], evaluation["fitness_interval"])
        lower = cast(Mapping[str, int], fitness["lower"])
        upper = cast(Mapping[str, int], fitness["upper"])

        return conservative_fitness_key(
            fitness=RationalInterval(
                Fraction(lower["numerator"], lower["denominator"]),
                Fraction(upper["numerator"], upper["denominator"]),
            ),
            program_hash=(
                f"{int(record['evaluation_index']):08d}:"
                f"{record['program_hash']}"
            ),
        )

    ranked = sorted(evaluation_records, key=selection_key)
    all_scientifically_comparable = all(
        cast(Mapping[str, Any], record["evaluation"]).get("status")
        in {
            SerialEvaluationStatus.COMPLETE.value,
            SerialEvaluationStatus.PROGRAM_FAILURE.value,
        }
        for record in ranked
    )
    selected_program_hash = (
        str(ranked[0]["program_hash"])
        if (
            ranked
            and outcome != "INCONCLUSIVE"
            and all_scientifically_comparable
        )
        else None
    )
    report = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "status": (
            "completed"
            if outcome != "INCONCLUSIVE" and all_scientifically_comparable
            else "inconclusive"
        ),
        "communication_mode": communication_mode,
        "cohort_outcome": outcome,
        "valid_ast": bool(programs),
        "valid_slots": sum(entry.program is not None for entry in entries),
        "unique_valid_programs": len(programs),
        "duplicate_aliases": sum(
            max(0, len(slot_aliases) - 1)
            for slot_aliases in aliases.values()
        ),
        "model_turns": model_turns,
        "graph_evaluations": graph_evaluations,
        "scientific_terminal_result": (
            outcome != "INCONCLUSIVE" and all_scientifically_comparable
        ),
        "selected_program_hash": selected_program_hash,
        "canonical_program_order": [
            program.program_hash for program in programs
        ],
        "program_aliases": {
            program_hash: list(slot_aliases)
            for program_hash, slot_aliases in aliases.items()
        },
        "slot_lineage": [
            slot_lineage[slot_id] for slot_id in SLOT_IDS
        ],
        "usage": _usage_total(usages),
        "epoch_manifest": str(output_root / "epoch-manifest.json.gz"),
        "cohort_report": str(report_path),
        "resumable": False,
    }
    report[
        "batch_reports"
        if communication_mode == "multi_program_batch"
        else "turn_reports"
    ] = [dict(item) for item in generation_reports]
    write_json(report_path, report)
    return report


def run_sequential_cohort(
    provider: LocalCodexAppServerProvider,
    experiment_root: str | Path,
    *,
    backend_factory: Callable[[], GraphBackend],
    episode_id: str,
) -> dict[str, Any]:
    """Generate two batches, deduplicate, then evaluate programs one at a time."""

    root = Path(experiment_root)
    output_root = root / "native-v3-output" / "epoch-0000"
    report_path = output_root / "cohort-report.json.gz"
    manifest = build_epoch_manifest(model=provider.model, effort=provider.effort)
    write_json(
        output_root / "epoch-manifest.json.gz",
        manifest,
        exclusive=True,
    )
    store = TurnArtifactStore(root / "artifacts")
    entries: list[CohortEntry] = []
    batch_reports: list[dict[str, Any]] = []
    slot_lineage: dict[str, dict[str, Any]] = {}
    usages: list[Mapping[str, Any]] = []
    model_turns = 0

    try:
        for call_index, slot_ids in enumerate(PROVIDER_PARTITION):
            request = build_batch_request(
                root,
                call_index=call_index,
                model=provider.model,
                effort=provider.effort,
            )
            raw_result = provider.generate(request)
            model_turns += 1
            response_text = raw_result.get("response_text")
            parsed = (
                parse_batch_response(response_text, slot_ids)
                if isinstance(response_text, str)
                else ParsedBatch(
                    None,
                    _invalid_entries(
                        slot_ids,
                        "provider result has no response_text",
                    ),
                )
            )
            turn, turn_manifest = _record_turn(
                store,
                request,
                raw_result,
                parsed,
                phase="initial",
            )
            attempts = [
                _attempt_reference(
                    raw_result,
                    phase="initial",
                    turn=turn,
                    manifest=turn_manifest,
                )
            ]
            usage = raw_result.get("usage")
            if isinstance(usage, Mapping):
                usages.append(usage)
            repaired = False
            if not any(entry.program is not None for entry in parsed.entries):
                diagnostics = [
                    {
                        "slot_id": entry.slot_id,
                        "code": "invalid_program",
                        "message": entry.error or "invalid program",
                    }
                    for entry in parsed.entries
                ]
                repair_idempotency_key = hashlib.sha256(
                    f"{request['idempotency_key']}:repair:01".encode()
                ).hexdigest()
                repair_request = {
                    **request,
                    "phase": "repair",
                    "repair_attempt": 1,
                    "max_repairs": 1,
                    "remaining_repairs": 0,
                    "repair_of_call_id": request["call_id"],
                    "idempotency_key": repair_idempotency_key,
                    "request_idempotency_key": repair_idempotency_key,
                    "artifact_dir": str(
                        store.turn_directory(
                            0,
                            str(request["slot"]),
                            "repair-01",
                        )
                    ),
                }
                repair_result = provider.repair(repair_request, diagnostics)
                model_turns += 1
                repair_text = repair_result.get("response_text")
                parsed = (
                    parse_batch_response(repair_text, slot_ids)
                    if isinstance(repair_text, str)
                    else ParsedBatch(
                        None,
                        _invalid_entries(
                            slot_ids,
                            "repair result has no response_text",
                        ),
                    )
                )
                turn, turn_manifest = _record_turn(
                    store,
                    repair_request,
                    repair_result,
                    parsed,
                    phase="repair-01",
                )
                usage = repair_result.get("usage")
                if isinstance(usage, Mapping):
                    usages.append(usage)
                repaired = True
                attempts.append(
                    _attempt_reference(
                        repair_result,
                        phase="repair-01",
                        turn=turn,
                        manifest=turn_manifest,
                    )
                )
            entries.extend(parsed.entries)
            for slot_id in slot_ids:
                slot_index = SLOT_IDS.index(slot_id)
                slot_lineage[slot_id] = {
                    "slot_id": slot_id,
                    "call_id": request["call_id"],
                    "brief_id": f"native-v3-brief-{slot_index:02d}",
                    "parent_program_hashes": [],
                    "provider_attempts": attempts,
                }
            batch_report = {
                "call_id": request["call_id"],
                "slot_ids": list(slot_ids),
                "repaired": repaired,
                "entries": [entry.as_dict() for entry in parsed.entries],
                "attempts": attempts,
                "turn_directory": str(turn),
                "turn_artifact_complete": (
                    turn_manifest.get("artifact_complete") is True
                ),
            }
            batch_reports.append(batch_report)
            write_json(
                output_root
                / "provider-batches"
                / f"call-{call_index:02d}.json.gz",
                batch_report,
                exclusive=True,
            )
    except Exception as error:
        report = {
            "schema_version": COHORT_SCHEMA_VERSION,
            "status": "provider_error",
            "cohort_outcome": "INCONCLUSIVE",
            "error_classification": _failure_classification(error),
            "error_type": type(error).__name__,
            "error": str(error),
            "valid_ast": False,
            "unique_valid_programs": 0,
            "model_turns": model_turns,
            "graph_evaluations": 0,
            "scientific_terminal_result": False,
            "usage": _usage_total(usages),
            "epoch_manifest": str(output_root / "epoch-manifest.json.gz"),
            "resumable": False,
        }
        write_json(report_path, report)
        return report

    return finalize_cohort(
        root,
        entries=entries,
        generation_reports=batch_reports,
        slot_lineage=slot_lineage,
        usages=usages,
        model_turns=model_turns,
        backend_factory=backend_factory,
        episode_id=episode_id,
        communication_mode="multi_program_batch",
    )


__all__ = [
    "COHORT_PROTOCOL_ID",
    "COHORT_SCHEMA_VERSION",
    "EPOCH_MANIFEST_SCHEMA_VERSION",
    "PROGRAM_BATCH_SCHEMA_VERSION",
    "PROVIDER_PARTITION",
    "SLOT_IDS",
    "CohortEntry",
    "NativeV3CohortError",
    "ParsedBatch",
    "build_batch_request",
    "build_epoch_manifest",
    "cohort_outcome",
    "deduplicate_entries",
    "finalize_cohort",
    "parse_batch_response",
    "render_batch_prompt",
    "record_batch_turn",
    "run_sequential_cohort",
]
