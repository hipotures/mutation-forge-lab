#!/usr/bin/env python3
"""Run the bounded Native v3 Step 12E0 transport-schema experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.native_v3.canonical import canonical_json_bytes
from mutation_forge.native_v3.persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    BRIEF_IDS,
    _behavior_signature,
    bootstrap_schema,
)
from mutation_forge.native_v3.single_program_ir import (
    CandidateContractError,
    CandidateKind,
    build_candidate_request,
    build_schema_complexity_inventory,
    compile_candidate_response,
    schema_experiment_anchor_prompt,
)
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    GenerationResult,
    ModelProfile,
)

REPORT_SCHEMA_VERSION = "mforge.native.transport_schema_experiment.v1"
FORBIDDEN_LENGTHS = (4, 8, 16)
MEDIUM_EFFORT = "medium"
MAX_EFFORT = "max"
MAX_BRIEFS = ("add-edge", "relocation")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REQUEST_ID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_EXPECTED_SUCCESS_SUFFIXES = {
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

AdapterFactory = Callable[[str, str, str], CodexAppServerAdapter]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--auth-json", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--turn-timeout", type=float, default=600.0)
    parser.add_argument("--skip-max", action="store_true")
    return parser


def _usage(result: GenerationResult) -> dict[str, Any]:
    return {
        "inputTokens": result.usage.input_tokens,
        "cachedInputTokens": result.usage.cached_input_tokens,
        "cacheWriteInputTokens": result.usage.cache_write_input_tokens,
        "outputTokens": result.usage.output_tokens,
        "reasoningOutputTokens": result.usage.reasoning_output_tokens,
        "totalTokens": result.usage.total_tokens,
        "final": result.usage.final,
        "partial": result.usage.partial,
    }


def _raw_usage(adapter: CodexAppServerAdapter) -> dict[str, Any] | None:
    raw = adapter.inspect_usage().get("raw")
    return dict(raw) if isinstance(raw, Mapping) else None


def _document_response(
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
        decoded: object = json.loads(result.text)
    except ValueError:
        decoded = result.text
    logger.document("response.json", decoded)
    logger.document(
        "provider-raw.json",
        {
            "response_text": result.text,
            "usage": _usage(result),
            "thread_id": result.thread_id,
            "session_id": result.session_id,
            "thread_path": result.thread_path,
            "turn_id": result.turn_id,
            "request_id": result.request_id,
            "diagnostics": list(result.diagnostics),
        },
    )
    logger.document("usage.json", _usage(result))


def _document_failure(
    adapter: CodexAppServerAdapter,
    *,
    prompt: str,
    system_prompt: str,
    schema: Mapping[str, Any],
    error: Exception,
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
    logger.document(
        "provider-raw.json",
        {
            "status": "failed",
            "error": f"{type(error).__name__}: {str(error)[:512]}",
            "metadata": dict(adapter.inspect_metadata()),
            "usage": dict(adapter.inspect_usage()),
            "diagnostics": list(adapter.diagnostics),
        },
    )


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(cast(dict[str, Any], value))
    return values


def _event_timestamp(
    event: Mapping[str, Any],
    params: Mapping[str, Any],
) -> int | None:
    value = event.get("emittedAtMs", params.get("emittedAtMs"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _event_metrics(path: Path) -> dict[str, Any]:
    events = _read_json_lines(path)
    start_ms: int | None = None
    first_usage_ms: int | None = None
    first_reasoning_ms: int | None = None
    first_agent_delta_ms: int | None = None
    retries: list[int | None] = []
    warnings: list[int | None] = []
    request_ids: set[str] = set()
    for event in events:
        method = event.get("method")
        params = event.get("params")
        if not isinstance(method, str) or not isinstance(params, Mapping):
            continue
        emitted = _event_timestamp(event, params)
        if method == "turn/started" and start_ms is None:
            start_ms = emitted
        if method == "thread/tokenUsage/updated" and first_usage_ms is None:
            first_usage_ms = emitted
        if method.startswith("item/reasoning/") and first_reasoning_ms is None:
            first_reasoning_ms = emitted
        if method == "item/started":
            item = params.get("item")
            if (
                first_reasoning_ms is None
                and isinstance(item, Mapping)
                and item.get("type") == "reasoning"
            ):
                first_reasoning_ms = emitted
        if method == "item/agentMessage/delta" and first_agent_delta_ms is None:
            first_agent_delta_ms = emitted
        if method == "error" and params.get("willRetry") is True:
            retries.append(emitted)
        if method == "warning":
            warnings.append(emitted)
        if method in {"error", "warning"}:
            request_ids.update(_REQUEST_ID.findall(json.dumps(params, sort_keys=True)))

    def elapsed(value: int | None) -> int | None:
        return value - start_ms if value is not None and start_ms is not None else None

    return {
        "turn_start_emitted_at_ms": start_ms,
        "time_to_first_token_usage_ms": elapsed(first_usage_ms),
        "time_to_first_reasoning_item_ms": elapsed(first_reasoning_ms),
        "time_to_first_agent_delta_ms": elapsed(first_agent_delta_ms),
        "retry_count": len(retries),
        "retry_timestamps_ms": retries,
        "warning_count": len(warnings),
        "warning_timestamps_ms": warnings,
        "upstream_request_ids": sorted(request_ids),
    }


def _artifact_suffixes(directory: Path, prefix: str) -> list[str]:
    return sorted(
        path.name.removeprefix(f"{prefix}.")
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(f"{prefix}.")
    )


def _transport_counters(adapter: CodexAppServerAdapter) -> tuple[int, int]:
    metadata = adapter.inspect_metadata()
    retries = metadata.get("serverRetries")
    warnings = metadata.get("serverWarnings")
    return (
        retries if isinstance(retries, int) else 0,
        warnings if isinstance(warnings, int) else 0,
    )


def _run_turn(
    adapter: CodexAppServerAdapter,
    *,
    artifact_directory: Path,
    prefix: str,
    prompt: str,
    system_prompt: str,
    schema: dict[str, Any],
    profile: ModelProfile,
    candidate: CandidateKind | None,
    slot_id: str | None,
    brief_id: str | None,
    forbidden_lengths: tuple[int, ...],
    schema_first_use: bool,
) -> dict[str, Any]:
    adapter.rotate_logger(artifact_directory, prefix)
    assert adapter.logger is not None
    adapter.logger.profile(
        {
            "model": profile.model,
            "effort": profile.effort,
            "ephemeral": False,
            "artifactPrefix": prefix,
            "candidate": candidate,
        }
    )
    adapter.logger.raw_text("request.md", prompt)
    adapter.logger.document(
        "request.json",
        {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "output_schema": schema,
        },
    )
    adapter.logger.raw_text("system-prompt.md", system_prompt)
    adapter.logger.document("output-schema.json", schema)
    before_retries, before_warnings = _transport_counters(adapter)
    started = time.monotonic()
    result: GenerationResult | None = None
    transport_error: str | None = None
    try:
        result = adapter.generate_persistent(
            prompt,
            profile,
            output_schema=schema,
        )
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {str(exc)[:512]}"
        _document_failure(
            adapter,
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            error=exc,
        )
    wall_time_ms = round((time.monotonic() - started) * 1000)
    if result is not None:
        _document_response(
            adapter,
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            result=result,
        )
    after_retries, after_warnings = _transport_counters(adapter)
    event_metrics = _event_metrics(artifact_directory / f"{prefix}.events.jsonl")
    event_metrics["retry_count"] = max(
        event_metrics["retry_count"],
        after_retries - before_retries,
    )
    event_metrics["warning_count"] = max(
        event_metrics["warning_count"],
        after_warnings - before_warnings,
    )
    suffixes = _artifact_suffixes(artifact_directory, prefix)
    json_decoded = False
    schema_conformant = False
    compiler_success = False
    semantic_valid = False
    compiler_error: str | None = None
    program_hash: str | None = None
    representation_hash: str | None = None
    signature: str | None = None
    if result is not None:
        try:
            decoded = json.loads(result.text)
            json_decoded = True
        except json.JSONDecodeError:
            decoded = None
        if decoded is not None:
            schema_conformant = Draft202012Validator(schema).is_valid(decoded)
        if (
            candidate is not None
            and slot_id is not None
            and brief_id is not None
        ):
            try:
                compiled = compile_candidate_response(
                    candidate,
                    result.text,
                    slot_id=slot_id,
                    brief_id=brief_id,
                    forbidden_lengths=forbidden_lengths,
                )
            except CandidateContractError as exc:
                compiler_error = str(exc)
            else:
                compiler_success = True
                semantic_valid = True
                program_hash = compiled.program.program_hash
                representation_hash = compiled.representation_sha256
                signature = _behavior_signature(compiled.program.ast)
    schema_bytes = canonical_json_bytes(schema)
    prompt_bytes = prompt.encode("utf-8")
    system_bytes = system_prompt.encode("utf-8")
    thread_id, turn_id = adapter.experimental_turn_identity()
    if result is not None:
        thread_id = result.thread_id
        turn_id = result.turn_id
    return {
        "prefix": prefix,
        "candidate": candidate,
        "slot_id": slot_id,
        "brief_id": brief_id,
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "schema_bytes": len(schema_bytes),
        "schema_first_use": schema_first_use,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_bytes": len(prompt_bytes),
        "system_prompt_sha256": hashlib.sha256(system_bytes).hexdigest(),
        "system_prompt_bytes": len(system_bytes),
        "model": profile.model,
        "effort": profile.effort,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "app_server_request_id": result.request_id if result is not None else None,
        **event_metrics,
        "total_wall_time_ms": wall_time_ms,
        "terminal_transport_status": "completed" if result is not None else "failed",
        "transport_error": transport_error,
        "usage": _usage(result) if result is not None else _raw_usage(adapter),
        "json_decode_result": json_decoded,
        "schema_conformance_result": schema_conformant,
        "compiler_result": compiler_success,
        "host_semantic_validation_result": semantic_valid,
        "compiler_error": compiler_error,
        "canonical_program_hash": program_hash,
        "representation_sha256": representation_hash,
        "behavior_signature": signature,
        "canonical_duplicate": False,
        "behavior_signature_duplicate": False,
        "artifact_suffixes": suffixes,
        "artifact_parity": set(suffixes) == _EXPECTED_SUCCESS_SUFFIXES,
    }


def _mark_duplicates(turns: list[dict[str, Any]]) -> None:
    program_counts = Counter(
        turn["canonical_program_hash"]
        for turn in turns
        if turn["canonical_program_hash"] is not None
    )
    signature_counts = Counter(
        turn["behavior_signature"]
        for turn in turns
        if turn["behavior_signature"] is not None
    )
    for turn in turns:
        program_hash = turn["canonical_program_hash"]
        signature = turn["behavior_signature"]
        turn["canonical_duplicate"] = (
            program_hash is not None and program_counts[program_hash] > 1
        )
        turn["behavior_signature_duplicate"] = (
            signature is not None and signature_counts[signature] > 1
        )


def _aggregate(candidate: CandidateKind, turns: list[dict[str, Any]]) -> dict[str, Any]:
    _mark_duplicates(turns)
    usage_fields = (
        "inputTokens",
        "cachedInputTokens",
        "cacheWriteInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    repeated: list[dict[str, Any]] = []
    by_schema: dict[str, list[dict[str, Any]]] = {}
    for turn in turns:
        by_schema.setdefault(cast(str, turn["schema_sha256"]), []).append(turn)
    for schema_hash, matching in sorted(by_schema.items()):
        if len(matching) < 2:
            continue
        repeated.append(
            {
                "schema_sha256": schema_hash,
                "first_use": {
                    key: matching[0][key]
                    for key in (
                        "total_wall_time_ms",
                        "retry_count",
                        "time_to_first_agent_delta_ms",
                    )
                },
                "repeated_uses": [
                    {
                        key: turn[key]
                        for key in (
                            "total_wall_time_ms",
                            "retry_count",
                            "time_to_first_agent_delta_ms",
                        )
                    }
                    | {
                        "inputTokens": (
                            turn["usage"].get("inputTokens")
                            if isinstance(turn["usage"], Mapping)
                            else None
                        ),
                        "cachedInputTokens": (
                            turn["usage"].get("cachedInputTokens")
                            if isinstance(turn["usage"], Mapping)
                            else None
                        ),
                    }
                    for turn in matching[1:]
                ],
                "cache_interpretation": (
                    "Observed token caching is not proof of schema-grammar caching."
                ),
            }
        )
    return {
        "candidate": candidate,
        "turns": turns,
        "turn_count": len(turns),
        "transport_completed": sum(
            turn["terminal_transport_status"] == "completed" for turn in turns
        ),
        "reconnects": sum(cast(int, turn["retry_count"]) for turn in turns),
        "schema_conformant": sum(
            turn["schema_conformance_result"] is True for turn in turns
        ),
        "compiler_successes": sum(turn["compiler_result"] is True for turn in turns),
        "semantically_valid": sum(
            turn["host_semantic_validation_result"] is True for turn in turns
        ),
        "canonical_duplicates": sum(
            turn["canonical_duplicate"] is True for turn in turns
        ),
        "behavior_signature_duplicates": sum(
            turn["behavior_signature_duplicate"] is True for turn in turns
        ),
        "artifact_parity_turns": sum(turn["artifact_parity"] is True for turn in turns),
        "usage": {
            field: sum(
                int(turn["usage"].get(field, 0))
                for turn in turns
                if isinstance(turn["usage"], Mapping)
            )
            for field in usage_fields
        },
        "repeated_schema_observations": repeated,
    }


def _medium_eligible(cohort: Mapping[str, Any]) -> bool:
    return (
        cohort.get("turn_count") == 4
        and cohort.get("transport_completed") == 4
        and cohort.get("reconnects") == 0
        and cohort.get("schema_conformant") == 4
        and cohort.get("compiler_successes") == 4
        and cohort.get("semantically_valid") == 4
        and cohort.get("canonical_duplicates") == 0
        and cohort.get("behavior_signature_duplicates") == 0
        and cohort.get("artifact_parity_turns") == 4
    )


def _max_passed(cohort: Mapping[str, Any]) -> bool:
    attempted = cohort.get("turn_count")
    return (
        attempted == len(MAX_BRIEFS)
        and cohort.get("transport_completed") == attempted
        and cohort.get("reconnects") == 0
        and cohort.get("schema_conformant") == attempted
        and cohort.get("compiler_successes") == attempted
        and cohort.get("semantically_valid") == attempted
        and cohort.get("artifact_parity_turns") == attempted
    )


def _anchor_schema(identity: str) -> dict[str, Any]:
    return bootstrap_schema(identity)


def _run_candidate_cohort(
    *,
    workspace: Path,
    candidate: CandidateKind,
    model: str,
    effort: str,
    brief_ids: Sequence[str],
    forbidden_lengths: tuple[int, ...],
    adapter_factory: AdapterFactory,
) -> dict[str, Any]:
    turns_directory = workspace / "provider-turns"
    turns_directory.mkdir(parents=True, exist_ok=True)
    system_prompt = build_candidate_request(
        candidate=candidate,
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=forbidden_lengths,
    ).system_prompt
    profile = ModelProfile("codex", model, effort)
    adapter = adapter_factory(system_prompt, candidate, effort)
    anchor = schema_experiment_anchor_prompt(forbidden_lengths)
    identity = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
    expected_acknowledgement = {
        "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
        "protocol_hash": identity,
    }
    bootstrap_prefix = f"{candidate}-{effort}-bootstrap"
    seen_schema_hashes: set[str] = set()
    bootstrap: dict[str, Any]
    turns: list[dict[str, Any]] = []
    try:
        bootstrap = _run_turn(
            adapter,
            artifact_directory=turns_directory,
            prefix=bootstrap_prefix,
            prompt=anchor,
            system_prompt=system_prompt,
            schema=_anchor_schema(identity),
            profile=profile,
            candidate=None,
            slot_id=None,
            brief_id=None,
            forbidden_lengths=forbidden_lengths,
            schema_first_use=True,
        )
        response_path = turns_directory / f"{bootstrap_prefix}.response.raw.txt"
        try:
            acknowledgement = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            acknowledgement = None
        bootstrap["acknowledgement_valid"] = (
            bootstrap["terminal_transport_status"] == "completed"
            and bootstrap["artifact_parity"] is True
            and acknowledgement == expected_acknowledgement
        )
        if not bootstrap["acknowledgement_valid"]:
            return {
                **_aggregate(candidate, turns),
                "bootstrap": bootstrap,
                "medium_eligible": False,
            }
        accepted_signatures: list[str] = []
        for index, brief_id in enumerate(brief_ids):
            slot_id = f"slot-{index:02d}"
            request = build_candidate_request(
                candidate=candidate,
                slot_id=slot_id,
                brief_id=brief_id,
                forbidden_lengths=forbidden_lengths,
                accepted_behavior_signatures=tuple(accepted_signatures),
            )
            schema_hash = hashlib.sha256(
                canonical_json_bytes(request.output_schema)
            ).hexdigest()
            prefix = f"{candidate}-{effort}-{slot_id}"
            turn = _run_turn(
                adapter,
                artifact_directory=turns_directory,
                prefix=prefix,
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                schema=request.output_schema,
                profile=profile,
                candidate=candidate,
                slot_id=slot_id,
                brief_id=brief_id,
                forbidden_lengths=forbidden_lengths,
                schema_first_use=schema_hash not in seen_schema_hashes,
            )
            seen_schema_hashes.add(schema_hash)
            turns.append(turn)
            signature = turn["behavior_signature"]
            if isinstance(signature, str):
                accepted_signatures.append(signature)
            if turn["terminal_transport_status"] != "completed":
                break
    finally:
        adapter.close()
    aggregate = _aggregate(candidate, turns)
    aggregate["bootstrap"] = bootstrap
    aggregate["medium_eligible"] = (
        _medium_eligible(aggregate) if effort == MEDIUM_EFFORT else None
    )
    return aggregate


def _status_lines() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _recommendation(
    *,
    medium: Mapping[str, Mapping[str, Any]],
    maximum: Mapping[str, Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in ("slot_specific", "flat_ir")
        if _medium_eligible(medium[candidate])
        and candidate in maximum
        and _max_passed(maximum[candidate])
    ]
    if not eligible:
        return {
            "decision": "NO-GO",
            "selected_candidate": None,
            "reason": (
                "Neither experimental candidate passed both the medium eligibility "
                "gate and every attempted max turn."
            ),
        }
    average_schema_bytes = {
        candidate: sum(
            int(item["schema_bytes"])
            for item in inventory
            if item["candidate"] == candidate
        )
        / len(BRIEF_IDS)
        for candidate in eligible
    }
    selected = min(
        eligible,
        key=lambda candidate: (
            average_schema_bytes[candidate],
            int(medium[candidate]["usage"]["totalTokens"]),
            candidate,
        ),
    )
    return {
        "decision": "GO",
        "selected_candidate": selected,
        "eligible_candidates": eligible,
        "reason": (
            "Selected among fully gated candidates by smaller mean canonical schema, "
            "then lower medium total-token cost."
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Native v3 Step 12E0 transport-schema benchmark",
        "",
        f"- Decision: `{report['recommendation']['decision']}`",
        f"- Selected candidate: `{report['recommendation']['selected_candidate']}`",
        f"- Model: `{report['model']}`",
        "",
        "## Live turns",
        "",
        "| Gate | Candidate | Brief | Status | Retry | Schema | Compiler | Semantic | Wall ms |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for gate in ("medium", "max"):
        cohorts = report[gate]
        for candidate, cohort in cohorts.items():
            for turn in cohort["turns"]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            gate,
                            candidate,
                            str(turn["brief_id"]),
                            str(turn["terminal_transport_status"]),
                            str(turn["retry_count"]),
                            str(turn["schema_conformance_result"]),
                            str(turn["compiler_result"]),
                            str(turn["host_semantic_validation_result"]),
                            str(turn["total_wall_time_ms"]),
                        ]
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            str(report["recommendation"]["reason"]),
            "",
            "Cached input tokens are reported as token-cache evidence only; they do not "
            "prove schema-grammar caching.",
            "",
            "STOP — waiting for operator acceptance",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_report_event_metrics(workspace: str | Path) -> dict[str, Any]:
    """Rebuild event-derived metrics from an existing benchmark without inference."""

    root = Path(workspace)
    report_path = root / "benchmark-report.json.gz"
    value = read_json(report_path)
    if not isinstance(value, dict) or value.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("workspace has no compatible benchmark report")
    report = cast(dict[str, Any], value)
    turns_directory = root / "provider-turns"
    for gate in ("medium", "max"):
        cohorts = report.get(gate)
        if not isinstance(cohorts, dict):
            raise ValueError(f"benchmark report has invalid {gate} cohorts")
        for cohort in cohorts.values():
            if not isinstance(cohort, dict):
                raise ValueError(f"benchmark report has invalid {gate} cohort")
            records: list[dict[str, Any]] = []
            bootstrap = cohort.get("bootstrap")
            if isinstance(bootstrap, dict):
                records.append(cast(dict[str, Any], bootstrap))
            turns = cohort.get("turns")
            if isinstance(turns, list):
                records.extend(
                    cast(dict[str, Any], turn)
                    for turn in turns
                    if isinstance(turn, dict)
                )
            for record in records:
                prefix = record.get("prefix")
                if not isinstance(prefix, str) or not prefix:
                    raise ValueError("benchmark turn has no artifact prefix")
                metrics = _event_metrics(turns_directory / f"{prefix}.events.jsonl")
                record.update(metrics)
    write_json(report_path, report)
    (root / "benchmark-report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def run_transport_schema_benchmark(
    workspace: str | Path,
    *,
    model: str,
    forbidden_lengths: tuple[int, ...],
    adapter_factory: AdapterFactory,
    run_max: bool = True,
) -> dict[str, Any]:
    """Run two medium cohorts and only gate-eligible bounded max turns."""

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    status_before = _status_lines()
    inventory = build_schema_complexity_inventory(forbidden_lengths)
    medium: dict[str, dict[str, Any]] = {}
    maximum: dict[str, dict[str, Any]] = {}
    candidates: tuple[CandidateKind, ...] = ("slot_specific", "flat_ir")
    for candidate in candidates:
        medium[candidate] = _run_candidate_cohort(
            workspace=root,
            candidate=candidate,
            model=model,
            effort=MEDIUM_EFFORT,
            brief_ids=BRIEF_IDS,
            forbidden_lengths=forbidden_lengths,
            adapter_factory=adapter_factory,
        )
    if run_max:
        for candidate in candidates:
            if _medium_eligible(medium[candidate]):
                maximum[candidate] = _run_candidate_cohort(
                    workspace=root,
                    candidate=candidate,
                    model=model,
                    effort=MAX_EFFORT,
                    brief_ids=MAX_BRIEFS,
                    forbidden_lengths=forbidden_lengths,
                    adapter_factory=adapter_factory,
                )
    status_after = _status_lines()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "repository_head": _git_head(),
        "repository_status_before": status_before,
        "repository_status_after": status_after,
        "repository_status_unchanged": status_before == status_after,
        "model": model,
        "forbidden_lengths": list(forbidden_lengths),
        "schema_complexity": inventory,
        "medium": medium,
        "max": maximum,
        "max_turn_count": sum(
            int(cohort["turn_count"]) for cohort in maximum.values()
        ),
        "production_default_changed": False,
        "recommendation": _recommendation(
            medium=medium,
            maximum=maximum,
            inventory=inventory,
        ),
        "remaining_uncertainty": [
            "Each medium candidate has one bounded four-brief cohort.",
            "Each eligible max candidate has at most two turns.",
            "Cached input tokens do not identify schema-grammar cache behavior.",
            "App Server retains a small platform-owned runtime wrapper.",
        ],
    }
    write_json(root / "benchmark-report.json.gz", report)
    (root / "benchmark-report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    args = _parser().parse_args()
    if args.workspace.exists():
        raise SystemExit("workspace must not exist")
    if not args.auth_json.is_file():
        raise SystemExit("auth JSON does not exist")

    def factory(
        base_instructions: str,
        _candidate: str,
        effort: str,
    ) -> CodexAppServerAdapter:
        return CodexAppServerAdapter(
            auth_json=args.auth_json,
            limits=AppServerLimits(
                max_turns=5,
                max_campaigns=1,
                turn_timeout=args.turn_timeout,
            ),
            base_instructions=base_instructions,
            compress_json_artifacts=True,
            sandbox_mode="read-only",
            approval_policy="never",
        )

    report = run_transport_schema_benchmark(
        args.workspace,
        model=args.model,
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
        run_max=not args.skip_max,
    )
    print(
        json.dumps(
            {
                "report": str(args.workspace / "benchmark-report.json.gz"),
                "decision": report["recommendation"]["decision"],
                "selected_candidate": report["recommendation"]["selected_candidate"],
                "medium_eligible": {
                    candidate: cohort["medium_eligible"]
                    for candidate, cohort in report["medium"].items()
                },
                "max_turn_count": report["max_turn_count"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
