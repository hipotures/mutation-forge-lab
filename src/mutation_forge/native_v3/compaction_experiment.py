"""Standalone Native v3 Step 12C context-compaction experiment."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import write_json
from mutation_forge.stage3.app_server import (
    CodexAppServerAdapter,
    CompactionResult,
    ModelProfile,
)

from .canonical import CANONICAL_PROTOCOL_ID
from .persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    _behavior_signature,
    _run_adapter_turn,
    bootstrap_prompt,
    bootstrap_schema,
    protocol_hash,
)
from .single_program_contract import (
    build_single_program_output_schema,
    build_single_program_request,
    validate_single_program_response,
)

COMPACTION_EXPERIMENT_SCHEMA_VERSION = (
    "mforge.native.compaction_retention_experiment.v1"
)
ACK_SCHEMA_VERSION = "mforge.native.compaction_ack.v1"
CLASSIFICATIONS = (
    "RELIABLE_FOR_OPTIMIZATION",
    "BEST_EFFORT_ONLY",
    "UNUSABLE",
)
ARMS = ("directive", "control")
MINIMUM_REPETITIONS = 3
USAGE_KEYS = (
    "inputTokens",
    "cachedInputTokens",
    "cacheWriteInputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
)

AdapterFactory = Callable[[str, str, int], CodexAppServerAdapter]

_CANDIDATE_IDS = (
    "g0000-s01",
    "g0001-s02",
    "g0002-s03",
    "g0003-s04",
)
_SUMMARIES = (
    "Adds one low-cycle-risk edge to improve connectivity.",
    "Removes one low-bridge-risk edge to disrupt a forbidden witness.",
    "Relocates one endpoint while preserving edge count.",
    "Fans out one edge to redistribute degree around a forbidden witness.",
)
_OUTCOMES = ("ACCEPTED", "REJECTED", "ACCEPTED", "ACTIVE_PARENT")
_SCORES_MICROS = (73125, 41250, 80312, 91750)
_STRENGTHS = (
    "Simple selector-certified edit.",
    "Targets a weak bridge candidate.",
    "Changes cycle structure without changing edge count.",
    "Highest observed fixture score.",
)
_WEAKNESSES = (
    "May create a new forbidden cycle.",
    "Can reduce connectivity.",
    "Depends on a legal relocation being available.",
    "May over-concentrate degree.",
)
_PARENT_IDS = ("", "g0000-s01", "g0001-s02", "g0002-s03")
_RELATIONS = ("root", "mutation", "mutation", "mutation")
_PENDING_NEXT_ACTION = (
    "Generate one remove-edge child from g0003-s04 while avoiding rejected signatures."
)


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
            "schema_version": {
                "type": "string",
                "const": ACK_SCHEMA_VERSION,
            },
            "ack": {"type": "string", "const": ack},
        },
    }


def _ack(ack: str) -> dict[str, str]:
    return {"schema_version": ACK_SCHEMA_VERSION, "ack": ack}


def build_reference_manifest(
    candidate_responses: Mapping[str, Mapping[str, Any]],
    *,
    forbidden_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Build the deterministic host-held reference used by both arms."""

    brief_ids = ("add-edge", "remove-edge", "relocation", "fanout")
    if set(candidate_responses) != set(brief_ids):
        raise ValueError("candidate fixtures must contain exactly four known briefs")
    candidates: list[dict[str, Any]] = []
    for index, brief_id in enumerate(brief_ids):
        validated = validate_single_program_response(
            _compact_json(candidate_responses[brief_id]),
            forbidden_lengths=forbidden_lengths,
        )
        candidates.append(
            {
                "candidate_id": _CANDIDATE_IDS[index],
                "canonical_ast": validated.program.ast,
                "canonical_hash": validated.program.program_hash,
                "strategy_summary": _SUMMARIES[index],
                "evaluation_outcome": _OUTCOMES[index],
                "score_micros": _SCORES_MICROS[index],
                "main_strength": _STRENGTHS[index],
                "main_weakness": _WEAKNESSES[index],
                "parent_id": _PARENT_IDS[index],
                "relationship": _RELATIONS[index],
                "behavior_signature": _behavior_signature(validated.program.ast),
            }
        )
    active = candidates[-1]
    return {
        "protocol_hash": protocol_hash(forbidden_lengths),
        "specification": {
            "active_forbidden_lengths": list(forbidden_lengths),
            "program_schema_version": "mforge.native.program.v3",
            "canonical_protocol_id": CANONICAL_PROTOCOL_ID,
        },
        "active_generation": 3,
        "active_parent_candidate_id": active["candidate_id"],
        "active_parent_canonical_ast": active["canonical_ast"],
        "active_parent_canonical_hash": active["canonical_hash"],
        "earlier_candidate_ids": list(_CANDIDATE_IDS[:-1]),
        "candidates": candidates,
        "rejected_behavior_signatures": [
            candidates[1]["behavior_signature"],
            candidates[2]["behavior_signature"],
        ],
        "pending_next_action": _PENDING_NEXT_ACTION,
    }


def retention_manifest_projection(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact compact manifest expected from the retention probe."""

    active_id = reference["active_parent_candidate_id"]
    candidates = []
    for raw in reference["candidates"]:
        candidate = dict(raw)
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "canonical_hash": (
                    candidate["canonical_hash"]
                    if candidate["candidate_id"] == active_id
                    else ""
                ),
                "strategy_summary": candidate["strategy_summary"],
                "evaluation_outcome": candidate["evaluation_outcome"],
                "score_micros": candidate["score_micros"],
                "main_strength": candidate["main_strength"],
                "main_weakness": candidate["main_weakness"],
                "parent_id": candidate["parent_id"],
                "relationship": candidate["relationship"],
            }
        )
    return {
        "protocol_hash": reference["protocol_hash"],
        "active_forbidden_lengths": reference["specification"][
            "active_forbidden_lengths"
        ],
        "program_schema_version": reference["specification"][
            "program_schema_version"
        ],
        "canonical_protocol_id": reference["specification"][
            "canonical_protocol_id"
        ],
        "active_generation": reference["active_generation"],
        "active_parent_candidate_id": active_id,
        "active_parent_canonical_hash": reference[
            "active_parent_canonical_hash"
        ],
        "earlier_candidate_ids": reference["earlier_candidate_ids"],
        "candidates": candidates,
        "rejected_behavior_signatures": reference[
            "rejected_behavior_signatures"
        ],
        "pending_next_action": reference["pending_next_action"],
    }


def manifest_probe_schema() -> dict[str, Any]:
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "candidate_id",
            "canonical_hash",
            "strategy_summary",
            "evaluation_outcome",
            "score_micros",
            "main_strength",
            "main_weakness",
            "parent_id",
            "relationship",
        ],
        "properties": {
            "candidate_id": {"type": "string"},
            "canonical_hash": {"type": "string"},
            "strategy_summary": {"type": "string"},
            "evaluation_outcome": {"type": "string"},
            "score_micros": {"type": "integer"},
            "main_strength": {"type": "string"},
            "main_weakness": {"type": "string"},
            "parent_id": {"type": "string"},
            "relationship": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "protocol_hash",
            "active_forbidden_lengths",
            "program_schema_version",
            "canonical_protocol_id",
            "active_generation",
            "active_parent_candidate_id",
            "active_parent_canonical_hash",
            "earlier_candidate_ids",
            "candidates",
            "rejected_behavior_signatures",
            "pending_next_action",
        ],
        "properties": {
            "protocol_hash": {"type": "string"},
            "active_forbidden_lengths": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "program_schema_version": {"type": "string"},
            "canonical_protocol_id": {"type": "string"},
            "active_generation": {"type": "integer"},
            "active_parent_candidate_id": {"type": "string"},
            "active_parent_canonical_hash": {"type": "string"},
            "earlier_candidate_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "candidates": {
                "type": "array",
                "items": candidate,
                "minItems": 4,
                "maxItems": 4,
            },
            "rejected_behavior_signatures": {
                "type": "array",
                "items": {"type": "string"},
            },
            "pending_next_action": {"type": "string"},
        },
    }


def _fixture_ast_prompt(
    reference: Mapping[str, Any],
    start: int,
) -> str:
    candidates = []
    for raw in reference["candidates"][start : start + 2]:
        candidate = dict(raw)
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "canonical_hash": candidate["canonical_hash"],
                "canonical_ast": candidate["canonical_ast"],
            }
        )
    return (
        "Store these fixture candidates as experiment context. Do not modify or "
        "regenerate them. Return only the required acknowledgement.\n\n"
        + _compact_json({"fixture_candidates": candidates})
    )


def _evaluation_prompt(reference: Mapping[str, Any], start: int) -> str:
    summaries = []
    for raw in reference["candidates"][start : start + 2]:
        candidate = dict(raw)
        summaries.append(
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "strategy_summary",
                    "evaluation_outcome",
                    "score_micros",
                    "main_strength",
                    "main_weakness",
                    "parent_id",
                    "relationship",
                    "behavior_signature",
                )
            }
        )
    return (
        "Store these host evaluation summaries as experiment context. Return only "
        "the required acknowledgement.\n\n"
        + _compact_json({"evaluation_summaries": summaries})
    )


def _checkpoint_prompt(arm: str) -> tuple[str, str]:
    if arm == "directive":
        return (
            "retention-directive",
            "[CONTEXT COMPACTION RETENTION DIRECTIVE]\n"
            "Before the host triggers context compaction, preserve the full "
            "mathematical specification and protocol identity; preserve the exact "
            "active-parent AST and canonical hash; preserve compact strategy "
            "summaries, evaluation outcomes, strengths, weaknesses, scores, "
            "relationships, and rejected behavior signatures for other candidates; "
            "and preserve the pending next action. Discard full non-parent ASTs, raw "
            "responses, repeated schemas, detailed traces, and transport chatter. "
            "Return only the required acknowledgement.",
        )
    return (
        "control-checkpoint",
        "The fixture population phase is complete. Return only the required "
        "acknowledgement.",
    )


def _manifest_probe_prompt() -> str:
    return (
        "After compaction, reproduce the retained host manifest from context. "
        "Use an empty canonical_hash for every non-active candidate. Do not invent "
        "candidate IDs, hashes, scores, or relationships."
    )


def _parent_probe_prompt() -> str:
    return (
        "After compaction, reproduce the exact active-parent program AST retained "
        "from context. Return the direct one-program structured response. Do not "
        "design a replacement."
    )


def _read_response(turns_dir: Path, prefix: str) -> object:
    return json.loads(
        (turns_dir / f"{prefix}.response.raw.txt").read_text(encoding="utf-8")
    )


def _run_ack_turn(
    adapter: CodexAppServerAdapter,
    *,
    turns_dir: Path,
    prefix: str,
    prompt: str,
    system_prompt: str,
    profile: ModelProfile,
    forbidden_lengths: tuple[int, ...],
    acknowledgement: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> Any:
    observation = _run_adapter_turn(
        adapter,
        artifact_dir=turns_dir,
        prefix=prefix,
        prompt=prompt,
        system_prompt=system_prompt,
        schema=schema,
        profile=profile,
        persistent=True,
        forbidden_lengths=forbidden_lengths,
        program_response=False,
    )
    if _read_response(turns_dir, prefix) != acknowledgement:
        raise ValueError(f"{prefix} acknowledgement does not match")
    return observation


def _zero_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def _prepare_compaction_artifacts(
    adapter: CodexAppServerAdapter,
    *,
    turns_dir: Path,
    profile: ModelProfile,
    system_prompt: str,
) -> None:
    adapter.rotate_logger(turns_dir, "06-compaction")
    assert adapter.logger is not None
    adapter.logger.profile(
        {
            "model": profile.model,
            "effort": profile.effort,
            "ephemeral": False,
            "artifactPrefix": "06-compaction",
        }
    )
    request = {
        "method": "thread/compact/start",
        "params": {
            "threadId": adapter.inspect_metadata()["threadId"],
        },
    }
    adapter.logger.raw_text("request.md", "thread/compact/start")
    adapter.logger.document("request.json", request)
    adapter.logger.raw_text("system-prompt.md", system_prompt)
    adapter.logger.document("output-schema.json", {})


def _finish_compaction_artifacts(
    adapter: CodexAppServerAdapter,
    result: CompactionResult,
) -> None:
    assert adapter.logger is not None
    usage = _usage_from_compaction(result)
    adapter.logger.raw_text("response.raw.txt", "{}")
    adapter.logger.raw_text("response.md", "{}")
    adapter.logger.document("response.json", {})
    adapter.logger.document(
        "provider-raw.json",
        {
            "response": {},
            "thread_id": result.thread_id,
            "turn_id": result.turn_id,
            "item_id": result.item_id,
            "request_id": result.request_id,
            "usage": usage,
            "usage_observed": result.usage is not None,
        },
    )
    adapter.logger.document("usage.json", usage)


def _usage_from_compaction(result: CompactionResult) -> dict[str, int]:
    if result.usage is None:
        return _zero_usage()
    return {
        "inputTokens": result.usage.input_tokens,
        "cachedInputTokens": result.usage.cached_input_tokens,
        "cacheWriteInputTokens": result.usage.cache_write_input_tokens,
        "outputTokens": result.usage.output_tokens,
        "reasoningOutputTokens": result.usage.reasoning_output_tokens,
        "totalTokens": result.usage.total_tokens,
    }


def compare_manifest(
    actual: object,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare exact retained fields and classify omissions and hallucinations."""

    if not isinstance(actual, Mapping):
        return {
            "exact": False,
            "retained": [],
            "omitted": ["manifest"],
            "altered": [],
            "hallucinated": [],
            "summaries_retained": False,
            "signatures_retained": False,
        }
    scalar_fields = (
        "protocol_hash",
        "active_forbidden_lengths",
        "program_schema_version",
        "canonical_protocol_id",
        "active_generation",
        "active_parent_candidate_id",
        "active_parent_canonical_hash",
        "earlier_candidate_ids",
        "rejected_behavior_signatures",
        "pending_next_action",
    )
    retained: list[str] = []
    omitted: list[str] = []
    altered: list[str] = []
    for field in scalar_fields:
        value = actual.get(field)
        if value == expected[field]:
            retained.append(field)
        elif value in (None, "", [], -1):
            omitted.append(field)
        else:
            altered.append(field)

    expected_candidates = {
        item["candidate_id"]: item for item in expected["candidates"]
    }
    raw_candidates = actual.get("candidates")
    candidates = raw_candidates if isinstance(raw_candidates, list) else []
    returned: dict[str, Mapping[str, Any]] = {}
    hallucinated: list[str] = []
    for item in candidates:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("candidate_id"), str
        ):
            hallucinated.append("malformed_candidate")
            continue
        candidate_id = item["candidate_id"]
        if candidate_id in returned:
            hallucinated.append(f"duplicate_candidate:{candidate_id}")
        returned[candidate_id] = item
        if candidate_id not in expected_candidates:
            hallucinated.append(f"candidate_id:{candidate_id}")
    candidate_fields = (
        "canonical_hash",
        "strategy_summary",
        "evaluation_outcome",
        "score_micros",
        "main_strength",
        "main_weakness",
        "parent_id",
        "relationship",
    )
    for candidate_id, expected_candidate in expected_candidates.items():
        actual_candidate = returned.get(candidate_id)
        if actual_candidate is None:
            omitted.append(f"candidate:{candidate_id}")
            continue
        for field in candidate_fields:
            marker = f"candidate:{candidate_id}:{field}"
            value = actual_candidate.get(field)
            if value == expected_candidate[field]:
                retained.append(marker)
            elif value in (None, "", -1) and expected_candidate[field] not in (
                "",
                -1,
            ):
                omitted.append(marker)
            else:
                altered.append(marker)
                if field in {
                    "canonical_hash",
                    "score_micros",
                    "parent_id",
                    "relationship",
                }:
                    hallucinated.append(f"{field}:{candidate_id}:{value}")
    summaries_retained = all(
        f"candidate:{candidate_id}:{field}" in retained
        for candidate_id in expected_candidates
        for field in (
            "strategy_summary",
            "evaluation_outcome",
            "score_micros",
            "main_strength",
            "main_weakness",
            "parent_id",
            "relationship",
        )
    )
    signatures_retained = "rejected_behavior_signatures" in retained
    return {
        "exact": not omitted and not altered and not hallucinated,
        "retained": sorted(retained),
        "omitted": sorted(omitted),
        "altered": sorted(altered),
        "hallucinated": sorted(set(hallucinated)),
        "summaries_retained": summaries_retained,
        "signatures_retained": signatures_retained,
    }


def _run_repetition(
    root: Path,
    *,
    arm: str,
    repetition: int,
    model: str,
    effort: str,
    forbidden_lengths: tuple[int, ...],
    reference: Mapping[str, Any],
    adapter_factory: AdapterFactory,
) -> dict[str, Any]:
    repetition_root = root / arm / f"rep-{repetition:02d}"
    repetition_root.mkdir(parents=True)
    turns_dir = repetition_root / "provider-turns"
    turns_dir.mkdir()
    profile = ModelProfile("codex", model, effort)
    system_prompt = build_single_program_request(
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=forbidden_lengths,
    ).system_prompt
    adapter = adapter_factory(system_prompt, arm, repetition)
    stage = "bootstrap"
    observations = []
    checkpoint = None
    compaction = None
    manifest_observation = None
    comparison = None
    started = time.monotonic()
    try:
        identity = reference["protocol_hash"]
        bootstrap_ack = {
            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
            "protocol_hash": identity,
        }
        observations.append(
            _run_ack_turn(
                adapter,
                turns_dir=turns_dir,
                prefix="00-bootstrap",
                prompt=bootstrap_prompt(forbidden_lengths),
                system_prompt=system_prompt,
                profile=profile,
                forbidden_lengths=forbidden_lengths,
                acknowledgement=bootstrap_ack,
                schema=bootstrap_schema(identity),
            )
        )
        for index, start in enumerate((0, 2)):
            stage = f"fixture-ast-{index:02d}"
            ack = stage
            observations.append(
                _run_ack_turn(
                    adapter,
                    turns_dir=turns_dir,
                    prefix=f"0{index + 1}-fixture-ast-{index:02d}",
                    prompt=_fixture_ast_prompt(reference, start),
                    system_prompt=system_prompt,
                    profile=profile,
                    forbidden_lengths=forbidden_lengths,
                    acknowledgement=_ack(ack),
                    schema=_ack_schema(ack),
                )
            )
        for index, start in enumerate((0, 2)):
            stage = f"evaluation-{index:02d}"
            ack = stage
            observations.append(
                _run_ack_turn(
                    adapter,
                    turns_dir=turns_dir,
                    prefix=f"0{index + 3}-evaluation-{index:02d}",
                    prompt=_evaluation_prompt(reference, start),
                    system_prompt=system_prompt,
                    profile=profile,
                    forbidden_lengths=forbidden_lengths,
                    acknowledgement=_ack(ack),
                    schema=_ack_schema(ack),
                )
            )
        stage = "checkpoint"
        checkpoint_ack, checkpoint_prompt = _checkpoint_prompt(arm)
        checkpoint = _run_ack_turn(
            adapter,
            turns_dir=turns_dir,
            prefix="05-checkpoint",
            prompt=checkpoint_prompt,
            system_prompt=system_prompt,
            profile=profile,
            forbidden_lengths=forbidden_lengths,
            acknowledgement=_ack(checkpoint_ack),
            schema=_ack_schema(checkpoint_ack),
        )
        observations.append(checkpoint)

        stage = "compaction"
        _prepare_compaction_artifacts(
            adapter,
            turns_dir=turns_dir,
            profile=profile,
            system_prompt=system_prompt,
        )
        try:
            compaction = adapter.compact_persistent_thread()
        except Exception as exc:
            assert adapter.logger is not None
            adapter.logger.document(
                "provider-raw.json",
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:512]}",
                },
            )
            raise
        _finish_compaction_artifacts(adapter, compaction)

        stage = "manifest-probe"
        manifest_observation = _run_adapter_turn(
            adapter,
            artifact_dir=turns_dir,
            prefix="07-manifest-probe",
            prompt=_manifest_probe_prompt(),
            system_prompt=system_prompt,
            schema=manifest_probe_schema(),
            profile=profile,
            persistent=True,
            forbidden_lengths=forbidden_lengths,
            program_response=False,
        )
        observations.append(manifest_observation)
        actual_manifest = _read_response(turns_dir, "07-manifest-probe")
        expected_manifest = retention_manifest_projection(reference)
        comparison = compare_manifest(actual_manifest, expected_manifest)

        stage = "parent-probe"
        parent_observation = _run_adapter_turn(
            adapter,
            artifact_dir=turns_dir,
            prefix="08-parent-probe",
            prompt=_parent_probe_prompt(),
            system_prompt=system_prompt,
            schema=build_single_program_output_schema(forbidden_lengths),
            profile=profile,
            persistent=True,
            forbidden_lengths=forbidden_lengths,
            program_response=True,
        )
        observations.append(parent_observation)
        parent_hash_match = (
            parent_observation.program_hash
            == reference["active_parent_canonical_hash"]
        )
        return {
            "arm": arm,
            "repetition": repetition,
            "thread_id": observations[0].thread_id,
            "turn_ids": [item.turn_id for item in observations],
            "turns": [item.as_dict() for item in observations],
            "compaction_status": "completed",
            "compaction_turn_id": compaction.turn_id,
            "compaction_item_id": compaction.item_id,
            "compaction_latency_ms": compaction.duration_ms,
            "compaction_usage": _usage_from_compaction(compaction),
            "compaction_usage_observed": compaction.usage is not None,
            "usage_before_compaction": checkpoint.usage,
            "usage_after_compaction": {
                "manifest_probe": manifest_observation.usage,
                "parent_probe": parent_observation.usage,
            },
            "post_compaction_turn_latency_ms": {
                "manifest_probe": manifest_observation.duration_ms,
                "parent_probe": parent_observation.duration_ms,
            },
            "manifest_comparison": comparison,
            "active_parent_observed_hash": parent_observation.program_hash,
            "active_parent_exact_hash_match": parent_hash_match,
            "provider_wall_time_ms": round((time.monotonic() - started) * 1000),
            "error": parent_observation.error,
        }
    except Exception as exc:
        thread_id, turn_id = adapter.experimental_turn_identity()
        text = f"{type(exc).__name__}: {str(exc)[:512]}"
        compaction_completed = compaction is not None
        usage_after: dict[str, Any] = {}
        if manifest_observation is not None:
            usage_after["manifest_probe"] = manifest_observation.usage
        return {
            "arm": arm,
            "repetition": repetition,
            "thread_id": observations[0].thread_id if observations else thread_id,
            "turn_ids": [item.turn_id for item in observations],
            "compaction_status": (
                "completed"
                if compaction_completed
                else "timeout"
                if stage == "compaction" and "timed out" in str(exc).lower()
                else "failed"
                if stage == "compaction"
                else "not_run"
            ),
            "compaction_turn_id": (
                compaction.turn_id
                if compaction is not None
                else turn_id
                if stage == "compaction"
                else None
            ),
            "compaction_item_id": (
                compaction.item_id if compaction is not None else None
            ),
            "compaction_latency_ms": (
                compaction.duration_ms if compaction is not None else None
            ),
            "compaction_usage": (
                _usage_from_compaction(compaction)
                if compaction is not None
                else _zero_usage()
            ),
            "compaction_usage_observed": (
                compaction is not None and compaction.usage is not None
            ),
            "usage_before_compaction": (
                checkpoint.usage if checkpoint is not None else _zero_usage()
            ),
            "usage_after_compaction": usage_after,
            "post_compaction_turn_latency_ms": (
                {
                    "manifest_probe": manifest_observation.duration_ms,
                }
                if manifest_observation is not None
                else {}
            ),
            "manifest_comparison": comparison,
            "active_parent_observed_hash": None,
            "active_parent_exact_hash_match": False,
            "provider_wall_time_ms": round((time.monotonic() - started) * 1000),
            "failure_stage": stage,
            "error": text,
        }
    finally:
        adapter.close()


def _classify(repetitions: Sequence[Mapping[str, Any]]) -> str:
    directive = [item for item in repetitions if item["arm"] == "directive"]
    if len(directive) < MINIMUM_REPETITIONS:
        return "UNUSABLE"
    if all(
        item["compaction_status"] == "completed"
        and item["active_parent_exact_hash_match"] is True
        and isinstance(item.get("manifest_comparison"), Mapping)
        and item["manifest_comparison"].get("exact") is True
        for item in directive
    ):
        return "RELIABLE_FOR_OPTIMIZATION"
    if any(
        item["compaction_status"] == "completed"
        and (
            item["active_parent_exact_hash_match"] is True
            or (
                isinstance(item.get("manifest_comparison"), Mapping)
                and (
                    item["manifest_comparison"].get("summaries_retained") is True
                    or item["manifest_comparison"].get("signatures_retained") is True
                )
            )
        )
        for item in directive
    ):
        return "BEST_EFFORT_ONLY"
    return "UNUSABLE"


def _sum_usage(values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        key: sum(
            int(value.get(key, 0))
            for value in values
            if isinstance(value.get(key, 0), int)
        )
        for key in USAGE_KEYS
    }


def _arm_summary(
    repetitions: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    selected = [item for item in repetitions if item["arm"] == arm]
    completed = [
        item for item in selected if item["compaction_status"] == "completed"
    ]
    manifest_comparisons = [
        item["manifest_comparison"]
        for item in completed
        if isinstance(item.get("manifest_comparison"), Mapping)
    ]
    before = [
        item["usage_before_compaction"]
        for item in completed
        if isinstance(item.get("usage_before_compaction"), Mapping)
    ]
    after: list[Mapping[str, Any]] = []
    for item in completed:
        value = item.get("usage_after_compaction")
        if not isinstance(value, Mapping):
            continue
        after.extend(
            usage for usage in value.values() if isinstance(usage, Mapping)
        )
    latencies = [
        int(item["compaction_latency_ms"])
        for item in completed
        if isinstance(item.get("compaction_latency_ms"), int)
    ]
    return {
        "repetitions": len(selected),
        "compaction_successes": len(completed),
        "active_parent_exact_hash_matches": sum(
            item.get("active_parent_exact_hash_match") is True
            for item in selected
        ),
        "exact_manifest_matches": sum(
            comparison.get("exact") is True
            for comparison in manifest_comparisons
        ),
        "summary_retention_matches": sum(
            comparison.get("summaries_retained") is True
            for comparison in manifest_comparisons
        ),
        "signature_retention_matches": sum(
            comparison.get("signatures_retained") is True
            for comparison in manifest_comparisons
        ),
        "hallucination_count": sum(
            len(comparison.get("hallucinated", []))
            for comparison in manifest_comparisons
        ),
        "usage_before_compaction": _sum_usage(before),
        "usage_after_compaction": _sum_usage(after),
        "mean_compaction_latency_ms": (
            round(sum(latencies) / len(latencies)) if latencies else None
        ),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Native v3 Step 12C compaction-retention result",
        "",
        f"Classification: **{report['classification']}**",
        "",
        "| arm | repetition | compaction | exact parent | exact manifest | hallucinations |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ]
    for item in report["repetitions"]:
        comparison = item.get("manifest_comparison")
        hallucinations = (
            len(comparison.get("hallucinated", []))
            if isinstance(comparison, Mapping)
            else 0
        )
        lines.append(
            "| {arm} | {rep} | {status} | {parent} | {manifest} | {hallucinations} |".format(
                arm=item["arm"],
                rep=item["repetition"],
                status=item["compaction_status"],
                parent=item["active_parent_exact_hash_match"],
                manifest=(
                    comparison.get("exact")
                    if isinstance(comparison, Mapping)
                    else False
                ),
                hallucinations=hallucinations,
            )
        )
    lines.extend(
        [
            "",
            "A pre-compaction retention directive is not a protocol-level guarantee.",
            "This standalone experiment does not enable production compaction.",
            "",
        ]
    )
    return "\n".join(lines)


def run_compaction_experiment(
    workspace: str | Path,
    *,
    model: str,
    effort: str,
    forbidden_lengths: tuple[int, ...],
    candidate_responses: Mapping[str, Mapping[str, Any]],
    adapter_factory: AdapterFactory,
    repetitions_per_arm: int = MINIMUM_REPETITIONS,
) -> dict[str, Any]:
    """Run the bounded three-directive/three-control compaction experiment."""

    if repetitions_per_arm < MINIMUM_REPETITIONS:
        raise ValueError("at least three repetitions per arm are required")
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    reference = build_reference_manifest(
        candidate_responses,
        forbidden_lengths=forbidden_lengths,
    )
    repetitions = [
        _run_repetition(
            root,
            arm=arm,
            repetition=repetition,
            model=model,
            effort=effort,
            forbidden_lengths=forbidden_lengths,
            reference=reference,
            adapter_factory=adapter_factory,
        )
        for arm in ARMS
        for repetition in range(repetitions_per_arm)
    ]
    classification = _classify(repetitions)
    assert classification in CLASSIFICATIONS
    report = {
        "schema_version": COMPACTION_EXPERIMENT_SCHEMA_VERSION,
        "model": model,
        "effort": effort,
        "repetitions_per_arm": repetitions_per_arm,
        "retention_directive_is_protocol_guarantee": False,
        "production_compaction_enabled": False,
        "host_reference_manifest": reference,
        "classification": classification,
        "arm_summary": {
            arm: _arm_summary(repetitions, arm)
            for arm in ARMS
        },
        "repetitions": repetitions,
    }
    write_json(root / "compaction-report.json.gz", report)
    (root / "compaction-report.md").write_text(
        _markdown_report(report),
        encoding="utf-8",
    )
    return report
