"""Opt-in Step 12B persistent-thread communication experiment."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mutation_forge.experiment.artifacts import TurnArtifactStore
from mutation_forge.experiment.json_io import write_json
from mutation_forge.stage3.app_server import (
    AppServerError,
    CodexAppServerAdapter,
    GenerationResult,
    ModelProfile,
)
from mutation_forge.stage3.isolation import IsolationError

from .canonical import canonical_json_bytes
from .cohort import (
    PROVIDER_PARTITION,
    build_batch_request,
    parse_batch_response,
    record_batch_turn,
)
from .single_program_contract import (
    SINGLE_PROGRAM_BRIEFS,
    SingleProgramContractError,
    build_single_program_contract,
    build_single_program_output_schema,
    build_single_program_request,
    model_facing_contract,
    validate_single_program_response,
)

PERSISTENT_EXPERIMENT_SCHEMA_VERSION = "mforge.native.persistent_thread_experiment.v1"
BOOTSTRAP_ACK_SCHEMA_VERSION = "mforge.native.persistent_bootstrap_ack.v1"
BRIEF_IDS = ("add-edge", "remove-edge", "relocation", "fanout")
INFRASTRUCTURE_RETRY_LIMIT = 3

AdapterFactory = Callable[[str, str], CodexAppServerAdapter]


class BatchProvider(Protocol):
    model: str
    effort: str

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TurnObservation:
    prefix: str
    thread_id: str
    turn_id: str
    prompt_sha256: str
    duration_ms: int
    usage: dict[str, int]
    program_hash: str | None
    behavior_signature: str | None
    error: str | None
    terminal_status: str
    usage_final: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "prompt_sha256": self.prompt_sha256,
            "duration_ms": self.duration_ms,
            "usage": self.usage,
            "program_hash": self.program_hash,
            "behavior_signature": self.behavior_signature,
            "error": self.error,
            "terminal_status": self.terminal_status,
            "usage_final": self.usage_final,
        }


def _usage(result: GenerationResult) -> dict[str, int]:
    return {
        "inputTokens": result.usage.input_tokens,
        "cachedInputTokens": result.usage.cached_input_tokens,
        "cacheWriteInputTokens": result.usage.cache_write_input_tokens,
        "outputTokens": result.usage.output_tokens,
        "reasoningOutputTokens": result.usage.reasoning_output_tokens,
        "totalTokens": result.usage.total_tokens,
    }


def _behavior_signature(program: Mapping[str, Any]) -> str:
    tokens: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key in ("op", "selector_id", "action_id", "mode", "reason"):
                item = value.get(key)
                if isinstance(item, str):
                    tokens.append(f"{key}:{item}")
            for key in sorted(value):
                visit(value[key])
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(program)
    return hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest()


def protocol_hash(forbidden_lengths: tuple[int, ...]) -> str:
    contract = model_facing_contract(build_single_program_contract(forbidden_lengths))
    return hashlib.sha256(canonical_json_bytes(contract)).hexdigest()


def bootstrap_prompt(forbidden_lengths: tuple[int, ...]) -> str:
    contract = model_facing_contract(build_single_program_contract(forbidden_lengths))
    identity = protocol_hash(forbidden_lengths)
    return json.dumps(
        {
            "instruction": (
                "Retain this mathematical program-synthesis contract for later turns. "
                "Do not generate a program. Return only the required acknowledgement."
            ),
            "protocol_hash": identity,
            "active_forbidden_lengths": list(forbidden_lengths),
            "contract": contract,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def bootstrap_schema(identity: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "protocol_hash"],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": BOOTSTRAP_ACK_SCHEMA_VERSION,
            },
            "protocol_hash": {"type": "string", "const": identity},
        },
    }


def followup_prompt(brief_id: str, accepted_signatures: tuple[str, ...]) -> str:
    return (
        "Generate exactly one program using the contract retained from the bootstrap turn.\n"
        "Return the direct structured response. Prefer valid no_plan over an illegal rewrite.\n\n"
        + json.dumps(
            {
                "brief_id": brief_id,
                "slot_objective": SINGLE_PROGRAM_BRIEFS[brief_id],
                "accepted_behavior_signatures": list(accepted_signatures),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _record_turn_payloads(
    adapter: CodexAppServerAdapter,
    *,
    prompt: str,
    system_prompt: str,
    schema: Mapping[str, Any],
    result: GenerationResult,
) -> None:
    logger = adapter.logger
    if logger is None:
        return
    logger.raw_text("request.md", prompt)
    logger.document(
        "request.json",
        {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "output_schema": dict(schema),
        },
    )
    logger.raw_text("system-prompt.md", system_prompt)
    logger.document("output-schema.json", dict(schema))
    logger.raw_text("response.raw.txt", result.text)
    logger.raw_text("response.md", result.text)
    try:
        decoded = json.loads(result.text)
    except ValueError:
        decoded = result.text
    logger.document("response.json", decoded)
    logger.document(
        "provider-raw.json",
        {
            "response_text": result.text,
            "usage": _usage(result),
            "thread_id": result.thread_id,
            "turn_id": result.turn_id,
            "request_id": result.request_id,
        },
    )
    logger.document("usage.json", _usage(result))


def _run_adapter_turn(
    adapter: CodexAppServerAdapter,
    *,
    artifact_dir: Path,
    prefix: str,
    prompt: str,
    system_prompt: str,
    schema: Mapping[str, Any],
    profile: ModelProfile,
    persistent: bool,
    forbidden_lengths: tuple[int, ...],
    program_response: bool,
) -> TurnObservation:
    adapter.rotate_logger(artifact_dir, prefix)
    assert adapter.logger is not None
    adapter.logger.profile(
        {
            "model": profile.model,
            "effort": profile.effort,
            "ephemeral": not persistent,
            "artifactPrefix": prefix,
        }
    )
    adapter.logger.raw_text("request.md", prompt)
    adapter.logger.document(
        "request.json",
        {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "output_schema": dict(schema),
        },
    )
    adapter.logger.raw_text("system-prompt.md", system_prompt)
    adapter.logger.document("output-schema.json", dict(schema))
    started = time.monotonic()
    try:
        result = (
            adapter.generate_persistent(prompt, profile, output_schema=schema)
            if persistent
            else adapter.generate_ephemeral_experiment(
                prompt,
                profile,
                output_schema=schema,
            )
        )
    except Exception as exc:
        adapter.logger.document(
            "provider-raw.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:512]}",
                "metadata": dict(adapter.inspect_metadata()),
                "usage": dict(adapter.inspect_usage()),
                "diagnostics": list(adapter.diagnostics),
            },
        )
        raise
    duration_ms = round((time.monotonic() - started) * 1000)
    _record_turn_payloads(
        adapter,
        prompt=prompt,
        system_prompt=system_prompt,
        schema=schema,
        result=result,
    )
    program_hash = None
    signature = None
    error = None
    if program_response:
        try:
            validated = validate_single_program_response(
                result.text,
                forbidden_lengths=forbidden_lengths,
            )
            program_hash = validated.program.program_hash
            signature = _behavior_signature(validated.program.ast)
        except SingleProgramContractError as exc:
            error = str(exc)
    return TurnObservation(
        prefix=prefix,
        thread_id=result.thread_id,
        turn_id=result.turn_id,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        duration_ms=result.duration_ms or duration_ms,
        usage=_usage(result),
        program_hash=program_hash,
        behavior_signature=signature,
        error=error,
        terminal_status="completed",
        usage_final=True,
    )


def _aggregate(
    observations: list[TurnObservation],
    provider_wall_time_ms: int,
    *,
    setup: TurnObservation | None = None,
    time_to_first_valid_ast_ms: int | None = None,
    time_to_four_valid_unique_ast_ms: int | None = None,
) -> dict[str, Any]:
    programs = [item for item in observations if item.program_hash is not None]
    program_hashes = [item.program_hash for item in programs]
    signatures = [item.behavior_signature for item in programs]
    setup_duration_ms = setup.duration_ms if setup is not None else 0
    elapsed_to_first_ms: int | None = None
    elapsed_to_four_ms: int | None = None
    elapsed_ms = setup_duration_ms
    unique_hashes: set[str] = set()
    for item in observations:
        elapsed_ms += item.duration_ms
        if item.program_hash is None:
            continue
        if elapsed_to_first_ms is None:
            elapsed_to_first_ms = elapsed_ms
        unique_hashes.add(item.program_hash)
        if len(unique_hashes) >= 4 and elapsed_to_four_ms is None:
            elapsed_to_four_ms = elapsed_ms
    usage_keys = tuple(_usage_keys())
    return {
        "turns": [item.as_dict() for item in observations],
        "provider_wall_time_ms": provider_wall_time_ms,
        "time_to_first_valid_ast_ms": (
            elapsed_to_first_ms
            if time_to_first_valid_ast_ms is None
            else time_to_first_valid_ast_ms
        ),
        "time_to_four_valid_unique_ast_ms": (
            elapsed_to_four_ms
            if time_to_four_valid_unique_ast_ms is None
            else time_to_four_valid_unique_ast_ms
        ),
        "valid_program_rate": len(programs) / max(1, len(observations)),
        "canonical_duplicate_rate": _duplicate_rate(program_hashes),
        "behavior_signature_duplicate_rate": _duplicate_rate(signatures),
        "usage": {
            key: (setup.usage[key] if setup is not None else 0)
            + sum(item.usage[key] for item in observations)
            for key in usage_keys
        },
    }


def _usage_keys() -> tuple[str, ...]:
    return (
        "inputTokens",
        "cachedInputTokens",
        "cacheWriteInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )


def _duplicate_rate(values: Sequence[str | None]) -> float:
    retained = [value for value in values if value is not None]
    return 0.0 if not retained else 1 - len(set(retained)) / len(retained)


def run_ab_experiment(
    workspace: str | Path,
    *,
    model: str,
    effort: str,
    forbidden_lengths: tuple[int, ...],
    adapter_factory: AdapterFactory,
) -> dict[str, Any]:
    """Run fresh-thread A and persistent-thread B; C may be attached separately."""

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    turns_dir = root / "provider-turns"
    turns_dir.mkdir()
    profile = ModelProfile("codex", model, effort)
    system = build_single_program_request(
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=forbidden_lengths,
    ).system_prompt
    schema = build_single_program_output_schema(forbidden_lengths)

    a_started = time.monotonic()
    a_observations: list[TurnObservation] = []
    a_failed_attempts: list[str] = []
    a_time_to_first_valid_ms: int | None = None
    for index, brief_id in enumerate(BRIEF_IDS):
        request = build_single_program_request(
            slot_id=f"slot-{index:02d}",
            brief_id=brief_id,
            forbidden_lengths=forbidden_lengths,
        )
        base_prefix = f"a-slot-{index:02d}"
        for attempt in range(INFRASTRUCTURE_RETRY_LIMIT + 1):
            prefix = base_prefix if attempt == 0 else f"{base_prefix}.retry-{attempt:02d}"
            adapter = adapter_factory(request.system_prompt, prefix)
            try:
                observation = _run_adapter_turn(
                    adapter,
                    artifact_dir=turns_dir,
                    prefix=prefix,
                    prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    schema=request.output_schema,
                    profile=profile,
                    persistent=False,
                    forbidden_lengths=forbidden_lengths,
                    program_response=True,
                )
            except IsolationError as exc:
                if str(exc) != "server retry is forbidden" or attempt >= INFRASTRUCTURE_RETRY_LIMIT:
                    raise
                a_failed_attempts.append(prefix)
            else:
                a_observations.append(observation)
                if a_time_to_first_valid_ms is None and observation.program_hash is not None:
                    a_time_to_first_valid_ms = round((time.monotonic() - a_started) * 1000)
                break
            finally:
                adapter.close()
    a_wall_time_ms = round((time.monotonic() - a_started) * 1000)
    a_report = _aggregate(
        a_observations,
        a_wall_time_ms,
        time_to_first_valid_ast_ms=a_time_to_first_valid_ms,
        time_to_four_valid_unique_ast_ms=(
            a_wall_time_ms
            if len({item.program_hash for item in a_observations if item.program_hash is not None})
            >= 4
            else None
        ),
    )
    a_report["infrastructure_attempts"] = len(a_observations) + len(a_failed_attempts)
    a_report["failed_attempt_prefixes"] = a_failed_attempts

    b_started = time.monotonic()
    b_adapter = adapter_factory(system, "b-bootstrap")
    b_observations: list[TurnObservation] = []
    identity = protocol_hash(forbidden_lengths)
    try:
        bootstrap = _run_adapter_turn(
            b_adapter,
            artifact_dir=turns_dir,
            prefix="b-bootstrap",
            prompt=bootstrap_prompt(forbidden_lengths),
            system_prompt=system,
            schema=bootstrap_schema(identity),
            profile=profile,
            persistent=True,
            forbidden_lengths=forbidden_lengths,
            program_response=False,
        )
        acknowledgement = json.loads(
            next(Path(turns_dir).glob("b-bootstrap.response.raw.txt")).read_text(encoding="utf-8")
        )
        if acknowledgement != {
            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
            "protocol_hash": identity,
        }:
            raise ValueError("bootstrap acknowledgement does not match protocol hash")
        signatures: list[str] = []
        for index, brief_id in enumerate(BRIEF_IDS):
            prefix = f"b-slot-{index:02d}"
            prompt = followup_prompt(brief_id, tuple(signatures))
            turn_started = time.monotonic()
            try:
                observation = _run_adapter_turn(
                    b_adapter,
                    artifact_dir=turns_dir,
                    prefix=prefix,
                    prompt=prompt,
                    system_prompt=system,
                    schema=schema,
                    profile=profile,
                    persistent=True,
                    forbidden_lengths=forbidden_lengths,
                    program_response=True,
                )
            except AppServerError as exc:
                thread_id, turn_id = b_adapter.experimental_turn_identity()
                if not isinstance(thread_id, str) or not isinstance(turn_id, str):
                    raise
                raw_usage = b_adapter.inspect_usage().get("raw")
                usage = {
                    key: (
                        int(raw_usage[key])
                        if isinstance(raw_usage, Mapping) and isinstance(raw_usage.get(key), int)
                        else 0
                    )
                    for key in _usage_keys()
                }
                observation = TurnObservation(
                    prefix=prefix,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    duration_ms=round((time.monotonic() - turn_started) * 1000),
                    usage=usage,
                    program_hash=None,
                    behavior_signature=None,
                    error=f"{type(exc).__name__}: {exc}",
                    terminal_status="failed",
                    usage_final=False,
                )
            b_observations.append(observation)
            if observation.behavior_signature is not None:
                signatures.append(observation.behavior_signature)
            if observation.terminal_status != "completed":
                break
    finally:
        b_adapter.close()
    b_report = _aggregate(
        b_observations,
        round((time.monotonic() - b_started) * 1000),
        setup=bootstrap,
    )
    report = {
        "schema_version": PERSISTENT_EXPERIMENT_SCHEMA_VERSION,
        "model": model,
        "effort": effort,
        "protocol_hash": identity,
        "bootstrap": bootstrap.as_dict(),
        "A_fresh_threads": a_report,
        "B_persistent_thread": b_report,
    }
    write_json(root / "ab-report.json.gz", report)
    return report


def run_live_batch_reference(
    workspace: str | Path,
    *,
    provider: BatchProvider,
) -> dict[str, Any]:
    """Run one production four-program batch without evaluation or repair."""

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    request = build_batch_request(
        root,
        call_index=0,
        model=provider.model,
        effort=provider.effort,
    )
    started = time.monotonic()
    failed_attempts = 0
    for attempt in range(INFRASTRUCTURE_RETRY_LIMIT + 1):
        try:
            raw_result = provider.generate(request)
        except IsolationError as exc:
            if str(exc) != "server retry is forbidden" or attempt >= INFRASTRUCTURE_RETRY_LIMIT:
                raise
            failed_attempts += 1
        else:
            break
    else:  # pragma: no cover - the bounded range always exits or raises
        raise AssertionError("unreachable provider retry state")
    response_text = raw_result.get("response_text")
    if not isinstance(response_text, str):
        raise ValueError("batch provider returned no response text")
    slot_ids = PROVIDER_PARTITION[0]
    parsed = parse_batch_response(response_text, slot_ids)
    store = TurnArtifactStore(root / "artifacts")
    turn, manifest = record_batch_turn(
        store,
        request,
        raw_result,
        parsed,
        phase="initial",
    )
    valid = [entry.program for entry in parsed.entries if entry.program is not None]
    hashes = [program.program_hash for program in valid]
    signatures = [_behavior_signature(program.ast) for program in valid]
    elapsed_ms = round((time.monotonic() - started) * 1000)
    usage = raw_result.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("batch provider returned no usage")
    report = {
        "source": str(turn),
        "model": provider.model,
        "effort": provider.effort,
        "model_turns": 1,
        "infrastructure_attempts": failed_attempts + 1,
        "total_slots": len(slot_ids),
        "time_to_first_valid_ast_ms": elapsed_ms if valid else None,
        "time_to_four_valid_unique_ast_ms": (
            elapsed_ms if len(set(hashes)) >= len(slot_ids) else None
        ),
        "provider_wall_time_ms": elapsed_ms,
        "valid_program_rate": len(valid) / len(slot_ids),
        "canonical_duplicate_rate": _duplicate_rate(hashes),
        "behavior_signature_duplicate_rate": _duplicate_rate(signatures),
        "usage": {key: int(usage[key]) for key in _usage_keys()},
        "turn_artifact_complete": manifest.get("artifact_complete") is True,
    }
    write_json(root / "batch-reference-report.json.gz", report)
    return report
