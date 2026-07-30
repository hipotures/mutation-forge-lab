"""Stage 3 one-shot generation orchestration.

The coordinator owns the deterministic eight-slot campaign.  Providers are
intentionally injected so the complete pipeline can be exercised without a
model; the production provider is :class:`AppServerGenerationProvider`.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol, cast

from mutation_forge.sandbox.contracts import (
    SandboxLimits,
    ScientificContext,
    ScientificProposal,
)
from mutation_forge.sandbox.validation import ValidationResult, validate_policy
from mutation_forge.stage2b.rankers import SourceRanker

from .artifacts import GenerationArtifacts, canonical_hash, replay_generation
from .contracts import GeneratedPolicy, parse_generated_policy

SLOTS: tuple[str, ...] = tuple(f"slot-{i:02d}" for i in range(8))
_REPAIRABLE = frozenset(
    {
        "structured_output",
        "invalid_json",
        "invalid_output",
        "invalid_keys",
        "invalid_string",
        "string_too_large",
        "invalid_list",
        "duplicate_value",
        "invalid_mapping",
        "mapping_too_large",
        "invalid_input",
        "invalid_float",
        "top_level_contract",
        "syntax_error",
        "forbidden_syntax",
        "wrong_signature",
        "wrong_function_name",
        "return_contract",
        "static_loop_bound",
        "loop_bound",
        "non_finite_literal",
    }
)
_RUNTIME_FAILURES = frozenset(
    {"finite_probe", "runtime_exception", "worker_timeout", "worker_crash", "worker_protocol"}
)


class GenerationProvider(Protocol):
    def generate(self, request: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model: str = "gpt-5.6-luna"
    effort: str = "high"
    protocol_version: str = "stage3.generation.v1"
    smoke_calls: int = 10_000
    max_repair_diagnostics: int = 8
    allow_infrastructure_retry: bool = False
    system_prompt: str = "Return exactly one Stage 3 generated-policy JSON object."

    def __post_init__(self) -> None:
        if self.model != "gpt-5.6-luna":
            raise ValueError("Stage 3 generation requires model gpt-5.6-luna")
        if self.effort != "high":
            raise ValueError("Stage 3 generation requires high reasoning effort")
        if self.smoke_calls != 10_000:
            raise ValueError("Stage 3 generation requires exactly 10,000 smoke calls")
        if self.allow_infrastructure_retry:
            raise ValueError("infrastructure retries are forbidden in Stage 3")


@dataclass(frozen=True, slots=True)
class Turn:
    response: Any
    accepted: bool
    charged: bool
    content: bool
    usage: Mapping[str, Any]
    status: str
    request_id: str | int | None = None
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    model: str | None = None
    effort: str | None = None
    transport_sha256: str | None = None
    appserver_doctor_sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    slot: str
    source: str
    source_sha256: str
    normalized_ast_sha256: str
    behavior_signature: Mapping[str, Any]
    worker_telemetry: Mapping[str, Any]
    provenance: Mapping[str, Any]
    duplicate: bool = False
    duplicate_of: str | None = None


@dataclass(frozen=True, slots=True)
class SlotResult:
    slot: str
    status: str
    candidate: Candidate | None = None
    errors: tuple[Mapping[str, Any], ...] = ()
    repairs: int = 0
    usage: Mapping[str, Any] = field(default_factory=dict)
    denied: tuple[str, ...] = ()
    initial: Mapping[str, Any] = field(default_factory=dict)
    repair: Mapping[str, Any] | None = None
    response: Any = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    status: str
    slots: tuple[SlotResult, ...]
    unique_candidates: tuple[Candidate, ...]
    summary: Mapping[str, Any]


def _sha_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _parse_envelope(response: Any) -> GeneratedPolicy:
    value = response
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("response envelope must be a JSON object")
    return parse_generated_policy(value)


def parse_envelope(response: Any) -> str:
    return _parse_envelope(response).source


def _turn(value: Any) -> Turn:
    if isinstance(value, Turn):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("provider must return an explicit Turn envelope")
    required = {"response", "accepted", "charged", "content", "usage", "status"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"provider turn is missing {sorted(missing)}")
    if any(not isinstance(value[k], bool) for k in ("accepted", "charged", "content")):
        raise ValueError("provider billing flags must be booleans")
    if not isinstance(value["usage"], Mapping):
        raise ValueError("provider usage must be an object")
    return Turn(
        response=value["response"],
        accepted=cast(bool, value["accepted"]),
        charged=cast(bool, value["charged"]),
        content=cast(bool, value["content"]),
        usage=cast(Mapping[str, Any], value["usage"]),
        status=str(value["status"]),
        request_id=value.get("request_id"),
        thread_id=value.get("thread_id"),
        session_id=value.get("session_id"),
        turn_id=value.get("turn_id"),
        model=value.get("model"),
        effort=value.get("effort"),
        transport_sha256=value.get("transport_sha256"),
        appserver_doctor_sha256=value.get("appserver_doctor_sha256"),
        error=value.get("error"),
    )


_USAGE_FIELDS = (
    "inputTokens",
    "cachedInputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
)


def _usage_complete(usage: Mapping[str, Any]) -> bool:
    fields = set(_USAGE_FIELDS) | (
        {"cacheWriteInputTokens"} if "cacheWriteInputTokens" in usage else set()
    )
    return (
        usage.get("final") is True
        and usage.get("partial") is False
        and all(
            isinstance(usage.get(k), int)
            and not isinstance(usage.get(k), bool)
            and cast(int, usage[k]) >= 0
            for k in fields
        )
    )


def _request_hash(request: Mapping[str, Any]) -> str:
    # Artifact paths and per-attempt control flags are not prompt identity.
    identity = {
        key: request[key]
        for key in (
            "slot",
            "model",
            "effort",
            "protocol_version",
            "prompt",
            "system_prompt",
            "output_schema",
            "appserver_doctor_sha256",
        )
        if key in request
    }
    return _value_hash(identity)


def _prompt_identity(request: Mapping[str, Any]) -> dict[str, str]:
    return {
        "prompt_sha256": _value_hash(request.get("prompt", "")),
        "system_prompt_sha256": _value_hash(request.get("system_prompt", "")),
        "output_schema_sha256": _value_hash(request.get("output_schema", {})),
        "request_sha256": _request_hash(request),
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value")
    return value


def _value_hash(value: Any) -> str:
    """Hash arbitrary provider values without allowing telemetry serialization to fail."""
    try:
        return canonical_hash(_canonical(value))
    except Exception:
        return hashlib.sha256(repr(value).encode("utf-8", "replace")).hexdigest()


def _artifact_value(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except Exception:
        return repr(value)


def _strip_timing(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _strip_timing(v)
            for k, v in value.items()
            if k not in {"started_at", "finished_at", "elapsed_seconds", "timing"}
            and not str(k).endswith("_ns")
        }
    if isinstance(value, (list, tuple)):
        return [_strip_timing(v) for v in value]
    return value


def _probe_inputs() -> tuple[ScientificContext, tuple[ScientificProposal, ...]]:
    context: ScientificContext = {
        "schema_version": "stage2b.context.v1",
        "order": 10,
        "forbidden_lengths": [4, 5],
        "capped_cycle_counts": [2, 1],
        "weighted_penalty": 13,
        "step": 0,
        "remaining_steps": 32,
        "stagnation": 0,
        "recent_best_improvement": 0.0,
        "recent_acceptance_rate": 0.0,
        "recent_duplicate_rate": 0.0,
    }

    def p(index: int, risk: int, broken: int, k: int) -> ScientificProposal:
        return {
            "schema_version": "stage2b.proposal.v1",
            "proposal_id": f"{index:064x}",
            "k": k,
            "operator_family": f"legal_{k}_switch",
            "selector_tags": ["uniform_random"],
            "anchor_forbidden_length": None,
            "broken_sampled_witnesses_by_length": [broken, 0],
            "removed_edge_load_sum_by_length": [broken + 1, 1],
            "removed_edge_load_max_by_length": [broken, 1],
            "minimum_distance_between_removed_edges": 1,
            "mean_distance_between_removed_edges": 1.5,
            "minimum_preexisting_distance_for_new_edges": 2,
            "mean_preexisting_distance_for_new_edges": 2.5,
            "local_triangle_risk": risk,
            "local_c4_risk": risk + 1,
            "reconnection_span": 3.0,
        }

    return context, (p(1, 0, 2, 2), p(2, 1, 1, 3), p(3, 2, 0, 4))


def _make_pool(proposals: Sequence[ScientificProposal]) -> Any:
    from mutation_forge.models import RewritePlan
    from mutation_forge.proposals.k_switch import ProposalCandidate, ProposalPool

    candidates = tuple(
        ProposalCandidate(
            RewritePlan((), (), str(p["operator_family"]), {"proposal_id": p["proposal_id"]}),
            cast(Any, p),
        )
        for p in proposals
    )
    payload = [{"proposal": c.payload, "removed_edges": (), "added_edges": ()} for c in candidates]
    return ProposalPool(
        "stage2b.pool.v1",
        candidates,
        canonical_hash(payload),
        len(candidates),
        {},
        0,
        len(candidates),
        {"uniform_random": len(candidates)},
        {str(p["k"]): 1 for p in proposals},
        {},
        0,
        0,
    )


def _behavior(
    source: str, limits: SandboxLimits, smoke_calls: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if smoke_calls != 10_000:
        raise ValueError("Stage 3 behavior requires exactly 10,000 smoke calls")
    validation = validate_policy(source, limits)
    if not validation.valid:
        raise ValueError("static validation failed")
    context, proposals = _probe_inputs()
    pool = _make_pool(proposals)
    before = canonical_hash((context, pool.as_dict()))
    ranker = SourceRanker("stage3-candidate", source, limits)
    try:
        ranked = ranker.rank(context, pool)
        if ranked.timeout:
            raise ValueError("worker_timeout")
        if ranked.crash:
            raise ValueError("worker_crash")
        if ranked.protocol:
            raise ValueError("worker_protocol")
        if ranked.exception:
            raise ValueError("runtime_exception")
        if not ranked.ranked:
            raise ValueError("finite_probe")
        if any(
            not isinstance(r.priority, (int, float)) or not math.isfinite(float(r.priority))
            for r in ranked.ranked
        ):
            raise ValueError("finite_probe")
        base = (
            ranked.as_dict()
            if hasattr(ranked, "as_dict")
            else {
                "rank_order": [r.proposal_id for r in ranked.ranked],
                "selected_proposal_id": ranked.selected_proposal_id,
                "exception": ranked.exception,
                "timeout": ranked.timeout,
                "crash": ranked.crash,
                "protocol": ranked.protocol,
            }
        )
        base = _strip_timing(base)
        base["schema_version"] = "stage3.behavior.v1"
        base["signature_sha256"] = canonical_hash(base)
        # SourceRanker owns one persistent PolicyWorker.  Reuse it for smoke calls.
        worker = getattr(ranker, "_worker", None)
        for _ in range(max(0, smoke_calls)):
            if worker is not None:
                result = worker.call(context, proposals[0])
                if (
                    result.status != "ok"
                    or result.priority is None
                    or not math.isfinite(float(result.priority))
                ):
                    raise ValueError("runtime_exception")
            else:
                check = ranker.rank(context, pool)
                if check.exception or check.timeout or check.crash or check.protocol:
                    raise ValueError("runtime_exception")
        after = canonical_hash((context, pool.as_dict()))
        if before != after:
            raise ValueError("input_mutation")
        telemetry = ranker.telemetry()
    finally:
        ranker.close()
    return base, {
        "behavior_probe": telemetry,
        "persistent_smoke": {"calls": smoke_calls},
        "smoke_calls": smoke_calls,
    }


def _diagnostics(errors: Sequence[Mapping[str, Any]], limit: int) -> tuple[Mapping[str, Any], ...]:
    result = []
    for error in errors:
        code = str(error.get("code", ""))
        if code in _REPAIRABLE:
            result.append({"code": code, "message": str(error.get("message", ""))[:256]})
    return tuple(result[: max(0, limit)])


def _repairable_errors(errors: Sequence[Mapping[str, Any]]) -> bool:
    """Repairs are restricted to output/schema/AST failures only.

    A mixed result (for example malformed output plus a usage or transport
    failure) is infrastructure-tainted and must remain terminal rather than
    consuming a repair turn.
    """
    codes = {str(error.get("code", "")) for error in errors}
    return bool(codes) and codes.issubset(_REPAIRABLE)


class OneShotGenerator:
    def __init__(
        self,
        provider: GenerationProvider | None = None,
        *,
        config: GenerationConfig | None = None,
        limits: SandboxLimits | None = None,
        artifacts: GenerationArtifacts | None = None,
        existing_sources: Sequence[str] = (),
        ranker_fixtures: Sequence[str] = (),
        slot_requests: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if provider is None:
            from .app_server import AppServerGenerationProvider

            provider = AppServerGenerationProvider()
        self.provider = provider
        self.config = config or GenerationConfig()
        self.limits = limits or SandboxLimits()
        self.artifacts = artifacts
        self.existing_sources = tuple(existing_sources) + tuple(ranker_fixtures)
        self.slot_requests = dict(slot_requests or {})
        if self.slot_requests and tuple(sorted(self.slot_requests)) != SLOTS:
            raise ValueError("slot requests must cover exactly slot-00 through slot-07")
        for slot in SLOTS:
            request = self._request(slot)
            if request.get("model", self.config.model) != "gpt-5.6-luna":
                raise ValueError(f"{slot} request model must be gpt-5.6-luna")
            if request.get("effort", self.config.effort) != "high":
                raise ValueError(f"{slot} request effort must be high")
            if (
                request.get("protocol_version", self.config.protocol_version)
                != self.config.protocol_version
            ):
                raise ValueError(
                    f"{slot} request protocol does not match the frozen protocol"
                )
        self._active = self._max_active = self._attempts = self._completed = (
            self._accepted_turns
        ) = 0
        self._lock = threading.Lock()

    def _request(self, slot: str) -> dict[str, Any]:
        if slot in self.slot_requests:
            request = dict(self.slot_requests[slot])
        else:
            request = {
                "slot": slot,
                "model": self.config.model,
                "effort": self.config.effort,
                "protocol_version": self.config.protocol_version,
                "prompt": f"Write one priority policy for {slot}.",
                "system_prompt": self.config.system_prompt,
                "output_schema": {"type": "object"},
            }
        request.setdefault("model", self.config.model)
        request.setdefault("effort", self.config.effort)
        request.setdefault("protocol_version", self.config.protocol_version)
        if self.artifacts is not None:
            request.setdefault("artifact_dir", str(self.artifacts.root / "slots" / slot))
            request.setdefault("artifact_prefix", slot)
            request.setdefault("artifact_root", str(self.artifacts.root))
        return request

    def _invoke(
        self,
        request: Mapping[str, Any],
        *,
        repair: bool = False,
        diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> Turn:
        with self._lock:
            self._active += 1
            self._attempts += 1
            self._max_active = max(self._max_active, self._active)
        try:
            if repair and hasattr(self.provider, "repair"):
                value = self.provider.repair(request, tuple(diagnostics))
            else:
                value = self.provider.generate(
                    {**request, "repair": repair, "diagnostics": list(diagnostics)}
                )
            turn = _turn(value)
            with self._lock:
                self._completed += 1
                if turn.accepted and turn.content:
                    self._accepted_turns += 1
            return turn
        finally:
            with self._lock:
                self._active -= 1

    def _assess(
        self,
        slot: str,
        turn: Turn,
        *,
        request: Mapping[str, Any] | None = None,
        repairs: int = 0,
        repair_record: Mapping[str, Any] | None = None,
    ) -> SlotResult:
        expected_request = request or self._request(slot)
        try:
            raw = (
                json.dumps(turn.response, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                if not isinstance(turn.response, str)
                else turn.response
            )
        except Exception:
            raw = repr(turn.response)
        usage = dict(turn.usage)
        # Persist the complete usage shape even when an offline provider omits
        # the optional cache-write counter (the App Server always emits it).
        usage.setdefault("cacheWriteInputTokens", 0)
        initial = {
            "status": turn.status,
            "accepted": turn.accepted,
            "charged": turn.charged,
            "content": turn.content,
            "usage": usage,
            "request_id": turn.request_id,
            "thread_id": turn.thread_id,
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "model": turn.model,
            "turn_effort": turn.effort,
            "effort": expected_request.get("effort", self.config.effort),
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "canonical_response_sha256": _value_hash(turn.response),
            "transport_sha256": turn.transport_sha256,
            "appserver_doctor_sha256": turn.appserver_doctor_sha256,
        }
        errors: list[Mapping[str, Any]] = []
        if turn.status != "completed":
            errors.append({"code": "turn_status", "message": f"turn status was {turn.status}"})
        elif not _usage_complete(turn.usage):
            errors.append(
                {"code": "usage_missing", "message": "completed turn omitted exact final usage"}
            )
        elif not turn.accepted or not turn.content:
            errors.append(
                {
                    "code": "turn_provenance",
                    "message": "completed turn must be accepted and contain content",
                }
            )
        elif turn.charged != (cast(int, turn.usage["totalTokens"]) > 0):
            errors.append(
                {
                    "code": "usage_provenance",
                    "message": "charged flag disagrees with exact totalTokens",
                }
            )
        elif turn.model != expected_request.get("model", self.config.model):
            errors.append(
                {
                    "code": "turn_provenance",
                    "message": "turn model does not match the immutable request",
                }
            )
        elif turn.model != "gpt-5.6-luna":
            errors.append({"code": "turn_provenance", "message": "turn model is not frozen"})
        elif turn.effort is not None and turn.effort != "high":
            errors.append({"code": "turn_provenance", "message": "turn effort is not frozen"})
        elif expected_request.get("effort", self.config.effort) != "high":
            errors.append({"code": "turn_provenance", "message": "request effort is not frozen"})
        elif not all(
            isinstance(getattr(turn, key), (str, int))
            and not isinstance(getattr(turn, key), bool)
            and bool(getattr(turn, key))
            for key in ("request_id", "thread_id", "turn_id")
        ):
            errors.append(
                {
                    "code": "turn_provenance",
                    "message": "completed turn omitted one or more request identifiers",
                }
            )
        elif turn.appserver_doctor_sha256 != expected_request.get("appserver_doctor_sha256"):
            errors.append(
                {
                    "code": "turn_provenance",
                    "message": "turn doctor provenance does not match the request",
                }
            )
        generated = None
        source = None
        try:
            generated = _parse_envelope(turn.response)
            source = generated.source
        except Exception as error:
            errors.append({"code": "structured_output", "message": str(error)[:256]})
        validation: ValidationResult | None = None
        behavior: Mapping[str, Any] = {}
        telemetry: Mapping[str, Any] = {}
        if source is not None:
            validation = validate_policy(source, self.limits)
            if not validation.valid:
                errors.extend(e.as_dict() for e in validation.errors)
            else:
                try:
                    behavior, telemetry = _behavior(source, self.limits, self.config.smoke_calls)
                except Exception as error:
                    error_code = str(error)
                    errors.append(
                        {
                            "code": (
                                error_code
                                if error_code in _REPAIRABLE | _RUNTIME_FAILURES
                                else "finite_probe"
                            ),
                            "message": str(error)[:256],
                        }
                    )
        if source is None or errors or validation is None or not validation.valid:
            return SlotResult(
                slot,
                "failed",
                errors=tuple(errors),
                repairs=repairs,
                usage=usage,
                initial=initial,
                repair=repair_record,
                response=turn.response,
            )
        generated = cast(GeneratedPolicy, generated)
        identity = validation.identity
        candidate = Candidate(
            slot,
            source,
            _sha_source(source),
            identity.normalized_ast_sha256 or "",
            behavior,
            telemetry,
            {
                "candidate_id": f"candidate-{slot}",
                "slot": slot,
                "model": turn.model,
                "protocol_version": self.config.protocol_version,
                "effort": self.config.effort,
                "request_id": turn.request_id,
                "thread_id": turn.thread_id,
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "transport_sha256": turn.transport_sha256,
                "appserver_doctor_sha256": turn.appserver_doctor_sha256,
                "usage": usage,
                "accepted": turn.accepted,
                "charged": turn.charged,
                "content": turn.content,
                "usage_final": turn.usage.get("final"),
                "usage_partial": turn.usage.get("partial"),
                **_prompt_identity(expected_request),
                "repair_count": repairs,
                "repair": dict(repair_record) if repair_record is not None else None,
                "initial_request_id": (
                    repair_record.get("initial_request_id") if repair_record else turn.request_id
                ),
                "design_summary": generated.design_summary,
                "used_fields": list(generated.used_fields),
                "assumptions": list(generated.assumptions),
            },
        )
        return SlotResult(
            slot,
            "duplicate_or_valid",
            candidate=candidate,
            repairs=repairs,
            usage=usage,
            initial=initial,
            repair=repair_record,
            response=turn.response,
        )

    def run(self, *, run_id: str = "stage3") -> GenerationResult:
        if self.artifacts:
            self.artifacts.start({"run_id": run_id, "status": "failed", "slots": []})
        initial: dict[str, Turn] = {}
        repairs: dict[str, Turn] = {}
        diagnostics: dict[str, tuple[Mapping[str, Any], ...]] = {}
        results: dict[str, SlotResult] = {}
        try:
            with ThreadPoolExecutor(max_workers=8, thread_name_prefix="stage3-slot") as pool:
                futures = {pool.submit(self._invoke, self._request(slot)): slot for slot in SLOTS}
                for future in as_completed(futures):
                    slot = futures[future]
                    try:
                        initial[slot] = future.result()
                    except BaseException as error:
                        results[slot] = SlotResult(
                            slot,
                            "failed",
                            errors=({"code": "provider_error", "message": str(error)[:256]},),
                        )
            for slot in SLOTS:
                if slot in initial:
                    results[slot] = self._assess(
                        slot, initial[slot], request=self._request(slot)
                    )
                    if (
                        results[slot].status == "failed"
                        and initial[slot].status == "completed"
                        and initial[slot].accepted
                        and initial[slot].content
                    ):
                        d = _diagnostics(results[slot].errors, self.config.max_repair_diagnostics)
                        if d and _repairable_errors(results[slot].errors):
                            diagnostics[slot] = d
            with ThreadPoolExecutor(max_workers=8, thread_name_prefix="stage3-repair") as pool:
                futures = {
                    pool.submit(self._invoke, self._request(slot), repair=True, diagnostics=d): slot
                    for slot, d in diagnostics.items()
                }
                for future in as_completed(futures):
                    slot = futures[future]
                    try:
                        repairs[slot] = future.result()
                    except BaseException as error:
                        results[slot] = replace(
                            results[slot],
                            repairs=1,
                            errors=(
                                *results[slot].errors,
                                {"code": "repair_provider_error", "message": str(error)[:256]},
                            ),
                        )
            for slot, turn in repairs.items():
                repair_usage = dict(turn.usage)
                repair_usage.setdefault("cacheWriteInputTokens", 0)
                record = {
                    "status": turn.status,
                    "accepted": turn.accepted,
                    "charged": turn.charged,
                    "content": turn.content,
                    "usage": repair_usage,
                    "request_id": turn.request_id,
                    "thread_id": turn.thread_id,
                    "session_id": turn.session_id,
                    "turn_id": turn.turn_id,
                    "model": turn.model,
                    "effort": turn.effort,
                    "transport_sha256": turn.transport_sha256,
                    "appserver_doctor_sha256": turn.appserver_doctor_sha256,
                    "diagnostics": list(diagnostics[slot]),
                    "response_sha256": _value_hash(turn.response),
                    "initial_request_id": results[slot].initial.get("request_id"),
                    "initial_response_sha256": results[slot].initial.get(
                        "canonical_response_sha256"
                    ),
                }
                assessed = self._assess(
                    slot,
                    turn,
                    request=self._request(slot),
                    repairs=1,
                    repair_record=record,
                )
                results[slot] = assessed
        except BaseException as error:
            for slot in SLOTS:
                results.setdefault(
                    slot,
                    SlotResult(
                        slot,
                        "interrupted",
                        errors=({"code": "interrupted", "message": str(error)[:256]},),
                    ),
                )
        ordered = tuple(results.get(slot, SlotResult(slot, "interrupted")) for slot in SLOTS)
        seen_ast: dict[str, str] = {}
        seen_src: dict[str, str] = {}
        unique: list[Candidate] = []
        rewritten: list[SlotResult] = []
        for source in self.existing_sources:
            try:
                valid = validate_policy(source, self.limits)
                if valid.identity.normalized_ast_sha256:
                    seen_ast[valid.identity.normalized_ast_sha256] = "baseline"
                seen_src[_sha_source(source)] = "baseline"
            except Exception:
                pass
        for result in ordered:
            candidate = result.candidate
            if candidate is None:
                rewritten.append(result)
                continue
            duplicate_of = seen_ast.get(candidate.normalized_ast_sha256) or seen_src.get(
                candidate.source_sha256
            )
            if duplicate_of is not None:
                candidate = replace(candidate, duplicate=True, duplicate_of=duplicate_of)
                rewritten.append(replace(result, status="duplicate", candidate=candidate))
            else:
                seen_ast[candidate.normalized_ast_sha256] = candidate.slot
                seen_src[candidate.source_sha256] = candidate.slot
                unique.append(candidate)
                rewritten.append(replace(result, status="accepted", candidate=candidate))
        usage_keys = (
            "inputTokens",
            "cachedInputTokens",
            "cacheWriteInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        )
        turns = tuple(initial.values()) + tuple(repairs.values())
        campaign_complete = (
            len(initial) == 8
            and len(repairs) == len(diagnostics)
            and self._attempts <= 16
            and all(
                self._request(slot).get("model") == "gpt-5.6-luna"
                and self._request(slot).get("effort") == "high"
                for slot in SLOTS
            )
            and all(
                result.candidate is not None and result.status in {"accepted", "duplicate"}
                for result in rewritten
            )
            and all(
                result.candidate is not None
                and result.candidate.worker_telemetry.get("smoke_calls") == 10_000
                and cast(
                    Mapping[str, Any],
                    result.candidate.worker_telemetry.get("persistent_smoke", {}),
                ).get("calls")
                == 10_000
                for result in rewritten
                if result.candidate is not None
            )
            and all(
                t.status == "completed"
                and t.accepted
                and t.content
                and _usage_complete(t.usage)
                and t.charged == (cast(int, t.usage["totalTokens"]) > 0)
                for t in turns
            )
        )
        summary: dict[str, Any] = {
            "schema_version": "stage3.generation.v1",
            "run_id": run_id,
            "status": "completed" if campaign_complete else "failed",
            "slots": [
                {
                    "slot": r.slot,
                    "status": r.status,
                    "repairs": r.repairs,
                    "errors": list(r.errors),
                    "duplicate": bool(r.candidate and r.candidate.duplicate),
                    "source_sha256": r.candidate.source_sha256 if r.candidate else None,
                    "normalized_ast_sha256": r.candidate.normalized_ast_sha256
                    if r.candidate
                    else None,
                }
                for r in rewritten
            ],
            "unique_count": len(unique),
            "initial_turn_count": len(initial),
            "repair_turn_count": len(repairs),
            "total_live_turns": len(turns),
            "provider_attempts": self._attempts,
            "completed_turns": self._completed,
            "model_turns": self._completed,
            "accepted_model_turns": self._accepted_turns,
            "exact_usage_complete": campaign_complete,
            "usage_totals": {k: sum(int(t.usage.get(k, 0)) for t in turns) for k in usage_keys},
            "initial_max_active": min(self._max_active, 8),
            "max_active": min(self._max_active, 8),
            "timing": {},
        }
        summary["canonical_generation_sha256"] = canonical_hash(_strip_timing(summary))
        if self.artifacts:
            for name, value in (
                (
                    "freeze.json",
                    {
                        "schema_version": "stage3.freeze.v1",
                        "slots": list(SLOTS),
                        "max_concurrency": 8,
                    },
                ),
                (
                    "environment.json",
                    {"python": platform.python_version(), "platform": platform.platform()},
                ),
                ("slots.json", summary["slots"]),
                (
                    "generation_config.json",
                    {
                        "model": self.config.model,
                        "effort": self.config.effort,
                        "protocol_version": self.config.protocol_version,
                        "smoke_calls": self.config.smoke_calls,
                        "sandbox_limits": asdict(self.limits),
                    },
                ),
            ):
                if not (self.artifacts.root / name).exists():
                    self.artifacts.write(name, value)
            for result in rewritten:
                base = f"slots/{result.slot}"
                self.artifacts.write(f"{base}/request.json", self._request(result.slot))
                self.artifacts.write(
                    f"{base}/events.json",
                    [
                        {
                            "event": "terminal",
                            "status": result.status,
                            "initial": dict(result.initial),
                            "repair": result.repair,
                        }
                    ],
                )
                self.artifacts.write(f"{base}/usage.json", _artifact_value(dict(result.usage)))
                self.artifacts.write(f"{base}/response.json", _artifact_value(result.response))
                self.artifacts.write(
                    f"{base}/validation.json",
                    {"errors": list(result.errors), "status": result.status},
                )
                if result.candidate:
                    self.artifacts.write_text(f"{base}/source.py", result.candidate.source)
                    with suppress(Exception):
                        self.artifacts.write(
                            f"{base}/canonical_response.json",
                            _parse_envelope(result.response).as_dict(),
                        )
                    self.artifacts.write(
                        f"{base}/identity.json",
                        validate_policy(result.candidate.source, self.limits).identity.as_dict(),
                    )
                    self.artifacts.write(
                        f"{base}/behavior.json", {"signature": result.candidate.behavior_signature}
                    )
                    self.artifacts.write(
                        f"{base}/worker_telemetry.json", result.candidate.worker_telemetry
                    )
                    self.artifacts.write(f"{base}/provenance.json", result.candidate.provenance)
                if result.repair:
                    self.artifacts.write(f"{base}/repair.json", result.repair)
            self.artifacts.finish(cast(str, summary["status"]), summary)
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()
        return GenerationResult(
            cast(str, summary["status"]), tuple(rewritten), tuple(unique), summary
        )


GenerationCoordinator = OneShotGenerator
GenerationOrchestrator = OneShotGenerator


def generate_once(provider: GenerationProvider, **kwargs: Any) -> GenerationResult:
    return OneShotGenerator(provider, **kwargs).run()


__all__ = [
    "Candidate",
    "GenerationArtifacts",
    "GenerationConfig",
    "GenerationCoordinator",
    "GenerationOrchestrator",
    "GenerationProvider",
    "GenerationResult",
    "OneShotGenerator",
    "SLOTS",
    "SlotResult",
    "Turn",
    "generate_once",
    "parse_envelope",
    "replay_generation",
]
