"""Stage-independent native generation orchestration.

This module is deliberately a small boundary around an injected provider.  It
does not know about a campaign freeze or a particular search stage: callers
choose the number of generations, population, turn budget and selection policy.
Provider envelopes are retained in checkpoints, making retries and resumes
idempotent.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.validation import (
    ValidationResult,
    render_policy_validator_contract,
    validate_policy,
)

from .json_io import read_json, write_json

GENERATION_SCHEMA_VERSION = "mforge.experiment.generation.v2"


class _InterruptibleThreadPoolExecutor(ThreadPoolExecutor):
    """Stop provider work before waiting for workers during Ctrl-C cleanup.

    ``ThreadPoolExecutor``'s context manager always calls ``shutdown(wait=True)``
    when leaving the ``with`` block.  That is normally useful, but it makes a
    signal received while a provider turn is running wait indefinitely for the
    provider process.  Native providers expose ``close`` specifically to stop
    those processes; call it after cancelling queued futures and before joining
    the worker threads.
    """

    def __init__(
        self,
        *args: Any,
        on_interrupt: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_interrupt = on_interrupt

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Literal[False]:
        if exc_type is None:
            self.shutdown(wait=True)
            return False

        # Do not wait for queued provider calls.  The callback closes active
        # transports, after which joining the workers is bounded by their
        # normal cleanup rather than by the provider's response timeout.
        self.shutdown(wait=False, cancel_futures=True)
        if self._on_interrupt is not None:
            with suppress(Exception):
                self._on_interrupt()
        self.shutdown(wait=True)
        return False


@dataclass(frozen=True, slots=True)
class _GeneratedPolicy:
    source: str


class _GeneratedPolicyError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        self.code = code
        super().__init__(message)


class _GracefulStopBoundary(Exception):
    """Stop a slot before it starts its next stage."""


def _parse_generated_policy(value: object) -> _GeneratedPolicy:
    if isinstance(value, (str, bytes, bytearray)):
        try:
            value = json.loads(
                value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = _GeneratedPolicyError(str(exc), "invalid_json")
            raise error from exc
    if not isinstance(value, Mapping):
        error = _GeneratedPolicyError("generated policy must be a JSON object", "invalid_output")
        raise error
    source = value.get("source")
    if not isinstance(source, str) or not source.strip():
        error = _GeneratedPolicyError(
            "generated policy source must be a non-empty string", "invalid_output"
        )
        raise error
    return _GeneratedPolicy(source)


class GenerationProvider(Protocol):
    def generate(self, request: Mapping[str, Any]) -> Any: ...


def _safe(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except Exception:
        return repr(value)
    return value


def _validate_generation_state(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported native generation checkpoint schema: {value.get('schema_version')!r}. "
            f"This runtime accepts only {GENERATION_SCHEMA_VERSION}. Create a fresh workspace."
        )
    if not isinstance(value.get("campaign_id"), str) or not value.get("campaign_id"):
        raise ValueError("native generation checkpoint campaign_id is required")
    for name in ("slots", "callbacks"):
        if not isinstance(value.get(name), Mapping):
            raise ValueError(f"native generation checkpoint {name} must be an object")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderResult:
    response: Any = None
    status: str = "completed"
    accepted: bool = True
    charged: bool = True
    content: bool = True
    uncharged: bool = False
    unauthorized_tool_approval: bool = False
    usage: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | int | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    session_id: str | None = None
    provider_request_id: str | None = None
    provider_thread_id: str | None = None
    provider_turn_id: str | None = None
    provider_duration_ms: int | None = None
    error: str | None = None
    retained: bool = False
    validation: Mapping[str, Any] = field(default_factory=dict)
    identity: Mapping[str, Any] = field(default_factory=dict)
    behavior: Mapping[str, Any] = field(default_factory=dict)
    worker_telemetry: Mapping[str, Any] = field(default_factory=dict)
    canonical_response: Mapping[str, Any] = field(default_factory=dict)
    metadata_validation: Mapping[str, Any] = field(default_factory=dict)
    response_projection_valid: bool | None = None
    response_diagnostics: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_value(cls, value: Any) -> ProviderResult:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            response = value.get("response", value.get("output"))
            usage = value.get("usage", {})
            return cls(
                response=response,
                status=str(value.get("status", "completed")),
                accepted=bool(value.get("accepted", True)),
                charged=bool(value.get("charged", True)),
                content=bool(value.get("content", response is not None)),
                uncharged=bool(value.get("uncharged", value.get("app_server_uncharged", False))),
                unauthorized_tool_approval=bool(value.get("unauthorized_tool_approval", False)),
                usage=cast(Mapping[str, Any], usage) if isinstance(usage, Mapping) else {},
                request_id=value.get("request_id"),
                thread_id=value.get("thread_id"),
                turn_id=value.get("turn_id"),
                session_id=value.get("session_id"),
                provider_request_id=value.get("provider_request_id"),
                provider_thread_id=value.get("provider_thread_id"),
                provider_turn_id=value.get("provider_turn_id"),
                provider_duration_ms=(
                    value.get("provider_duration_ms")
                    if isinstance(value.get("provider_duration_ms"), int)
                    and not isinstance(value.get("provider_duration_ms"), bool)
                    else None
                ),
                error=str(value.get("error")) if value.get("error") is not None else None,
                retained=value.get("retained") is True,
                validation=cast(Mapping[str, Any], value.get("validation", {}))
                if isinstance(value.get("validation"), Mapping)
                else {},
                identity=cast(Mapping[str, Any], value.get("identity", {}))
                if isinstance(value.get("identity"), Mapping)
                else {},
                behavior=cast(Mapping[str, Any], value.get("behavior", {}))
                if isinstance(value.get("behavior"), Mapping)
                else {},
                worker_telemetry=cast(Mapping[str, Any], value.get("worker_telemetry", {}))
                if isinstance(value.get("worker_telemetry"), Mapping)
                else {},
                canonical_response=cast(Mapping[str, Any], value.get("canonical_response", {}))
                if isinstance(value.get("canonical_response"), Mapping)
                else {},
                metadata_validation=cast(Mapping[str, Any], value.get("metadata_validation", {}))
                if isinstance(value.get("metadata_validation"), Mapping)
                else {},
                response_projection_valid=(
                    bool(value["response_projection_valid"])
                    if isinstance(value.get("response_projection_valid"), bool)
                    else None
                ),
                response_diagnostics=tuple(
                    item
                    for item in value.get("response_diagnostics", ())
                    if isinstance(item, Mapping)
                )
                if isinstance(value.get("response_diagnostics"), Sequence)
                and not isinstance(value.get("response_diagnostics"), (str, bytes, bytearray))
                else (),
            )
        if all(hasattr(value, name) for name in ("response", "status", "usage")):
            return cls(
                response=value.response,
                status=str(value.status),
                accepted=bool(getattr(value, "accepted", True)),
                charged=bool(getattr(value, "charged", True)),
                content=bool(getattr(value, "content", value.response is not None)),
                uncharged=bool(getattr(value, "uncharged", False)),
                unauthorized_tool_approval=bool(
                    getattr(value, "unauthorized_tool_approval", False)
                ),
                usage=cast(Mapping[str, Any], value.usage)
                if isinstance(value.usage, Mapping)
                else {},
                request_id=getattr(value, "request_id", None),
                thread_id=getattr(value, "thread_id", None),
                turn_id=getattr(value, "turn_id", None),
                session_id=getattr(value, "session_id", None),
                provider_request_id=getattr(value, "provider_request_id", None),
                provider_thread_id=getattr(value, "provider_thread_id", None),
                provider_turn_id=getattr(value, "provider_turn_id", None),
                provider_duration_ms=(
                    getattr(value, "provider_duration_ms", None)
                    if isinstance(getattr(value, "provider_duration_ms", None), int)
                    and not isinstance(getattr(value, "provider_duration_ms", None), bool)
                    else None
                ),
                error=getattr(value, "error", None),
                retained=bool(getattr(value, "retained", False)),
            )
        return cls(response=value, charged=False, usage={})

    def as_dict(self) -> dict[str, Any]:
        return {
            "response": _safe(self.response),
            "status": self.status,
            "accepted": self.accepted,
            "charged": self.charged,
            "content": self.content,
            "uncharged": self.uncharged,
            "unauthorized_tool_approval": self.unauthorized_tool_approval,
            "usage": _safe(dict(self.usage)),
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "provider_request_id": self.provider_request_id,
            "provider_thread_id": self.provider_thread_id,
            "provider_turn_id": self.provider_turn_id,
            "provider_duration_ms": self.provider_duration_ms,
            "error": self.error,
            "retained": self.retained,
            "validation": _safe(dict(self.validation)),
            "identity": _safe(dict(self.identity)),
            "behavior": _safe(dict(self.behavior)),
            "worker_telemetry": _safe(dict(self.worker_telemetry)),
            "canonical_response": _safe(dict(self.canonical_response)),
            "metadata_validation": _safe(dict(self.metadata_validation)),
            "response_projection_valid": self.response_projection_valid,
            "response_diagnostics": [_safe(dict(item)) for item in self.response_diagnostics],
        }


RawProviderResult = ProviderResult
GenerationTurn = ProviderResult


def request_idempotency_key(
    campaign: str,
    generation: int,
    slot: str,
    parent: str,
    brief: str,
    prompt_hash: str,
    phase: str = "initial",
    repair_attempt: int = 0,
) -> str:
    identity = {
        "campaign": campaign,
        "generation": generation,
        "slot": slot,
        "parent": parent,
        "brief": brief,
        "prompt_hash": prompt_hash,
        "phase": phase,
    }
    if repair_attempt:
        identity["repair_attempt"] = repair_attempt
    return _hash(identity)


make_idempotency_key = request_idempotency_key


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    campaign_id: str
    generation: int
    slot: str
    parent_id: str
    brief_id: str
    prompt: str
    prompt_hash: str
    idempotency_key: str
    phase: str = "initial"
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    parent_source: str = ""
    parent_metadata: Mapping[str, Any] = field(default_factory=dict)
    search_feedback: str = ""
    archive_context: str = ""
    model: str = "gpt-5.6-luna"
    effort: str = "high"
    system_prompt: str = "Return one generated policy JSON object."
    output_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    repair_prompt: str = "Repair the generated policy using the diagnostics."
    repair_attempt: int = 0
    max_repairs: int = 0
    remaining_repairs: int = 0

    @property
    def request_idempotency_key(self) -> str:
        return self.idempotency_key

    def as_dict(self) -> dict[str, Any]:
        value = {
            "campaign_id": self.campaign_id,
            "generation": self.generation,
            "slot": self.slot,
            "parent_id": self.parent_id,
            "brief_id": self.brief_id,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "idempotency_key": self.idempotency_key,
            "request_idempotency_key": self.idempotency_key,
            "phase": self.phase,
            "parent_source": self.parent_source,
            "parent_metadata": dict(self.parent_metadata),
            "search_feedback": self.search_feedback,
            "archive_context": self.archive_context,
            "model": self.model,
            "effort": self.effort,
            "system_prompt": self.system_prompt,
            "output_schema": dict(self.output_schema),
            "repair_prompt": self.repair_prompt,
            "repair_attempt": self.repair_attempt,
            "max_repairs": self.max_repairs,
            "remaining_repairs": self.remaining_repairs,
        }
        if self.phase == "repair":
            value["diagnostics"] = [dict(item) for item in self.diagnostics]
        return value

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> GenerationRequest:
        return cls(
            campaign_id=str(value.get("campaign_id", "native")),
            generation=int(value.get("generation", 0)),
            slot=str(value.get("slot", "")),
            parent_id=str(value.get("parent_id", "")),
            brief_id=str(value.get("brief_id", "")),
            prompt=str(value.get("prompt", "")),
            prompt_hash=str(value.get("prompt_hash", "")),
            idempotency_key=str(
                value.get("idempotency_key", value.get("request_idempotency_key", ""))
            ),
            phase=str(value.get("phase", "initial")),
            diagnostics=tuple(
                item for item in value.get("diagnostics", ()) if isinstance(item, Mapping)
            ),
            parent_source=str(value.get("parent_source", "")),
            parent_metadata=dict(value.get("parent_metadata", {}))
            if isinstance(value.get("parent_metadata", {}), Mapping)
            else {},
            search_feedback=str(value.get("search_feedback", "")),
            archive_context=str(value.get("archive_context", "")),
            model=str(value.get("model", "gpt-5.6-luna")),
            effort=str(value.get("effort", "high")),
            system_prompt=str(
                value.get("system_prompt", "Return one generated policy JSON object.")
            ),
            output_schema=(
                cast(Mapping[str, Any], value.get("output_schema"))
                if isinstance(value.get("output_schema"), Mapping)
                else {"type": "object"}
            ),
            repair_prompt=str(
                value.get("repair_prompt", "Repair the generated policy using the diagnostics.")
            ),
            repair_attempt=int(value.get("repair_attempt", 0)),
            max_repairs=int(value.get("max_repairs", 0)),
            remaining_repairs=int(value.get("remaining_repairs", 0)),
        )


Request = GenerationRequest


@dataclass(frozen=True, slots=True)
class Candidate:
    source: str
    source_sha256: str
    normalized_ast_sha256: str
    generation: int
    slot: str
    parent_id: str
    behavior_signature: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | int | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    repair: bool = False
    duplicate_of: str | None = None
    source_identity: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unique(self) -> bool:
        return self.duplicate_of is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "normalized_ast_sha256": self.normalized_ast_sha256,
            "generation": self.generation,
            "slot": self.slot,
            "parent_id": self.parent_id,
            "behavior_signature": _safe(dict(self.behavior_signature)),
            "usage": _safe(dict(self.usage)),
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "repair": self.repair,
            "duplicate_of": self.duplicate_of,
            "source_identity": _safe(dict(self.source_identity)),
        }


@dataclass(frozen=True, slots=True)
class SlotResult:
    generation: int
    slot: str
    parent_id: str
    status: str
    candidate: Candidate | None = None
    errors: tuple[Mapping[str, Any], ...] = ()
    repairs: int = 0
    initial: Mapping[str, Any] = field(default_factory=dict)
    repair: Mapping[str, Any] | None = None
    request: Mapping[str, Any] = field(default_factory=dict)
    raw_result: Mapping[str, Any] = field(default_factory=dict)
    duplicate_of: str | None = None
    initial_request: Mapping[str, Any] = field(default_factory=dict)
    repair_idempotency_keys: tuple[str, ...] = ()
    remaining_repairs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "slot": self.slot,
            "parent_id": self.parent_id,
            "status": self.status,
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "errors": [_safe(dict(item)) for item in self.errors],
            "repairs": self.repairs,
            "initial": _safe(dict(self.initial)),
            "repair": _safe(self.repair),
            "request": _safe(dict(self.request)),
            "raw_result": _safe(dict(self.raw_result)),
            "duplicate_of": self.duplicate_of,
            "initial_request": _safe(dict(self.initial_request)),
            "repair_idempotency_keys": list(self.repair_idempotency_keys),
            "remaining_repairs": self.remaining_repairs,
        }


def _slot_failure_is_retryable(value: Mapping[str, Any]) -> bool:
    if str(value.get("status", "")) != "failed":
        return False
    raw = value.get("raw_result")
    if not isinstance(raw, Mapping):
        raw = (
            value.get("repair")
            if isinstance(value.get("repair"), Mapping)
            else value.get("initial")
        )
    if not isinstance(raw, Mapping):
        return True
    usage = raw.get("usage")
    total_tokens = usage.get("totalTokens") if isinstance(usage, Mapping) else None
    charged = raw.get("charged") is True or (
        isinstance(total_tokens, int) and not isinstance(total_tokens, bool) and total_tokens > 0
    )
    return not charged


@dataclass(frozen=True, slots=True)
class GenerationResult:
    status: str
    generations: tuple[tuple[SlotResult, ...], ...]
    slots: tuple[SlotResult, ...]
    unique_candidates: tuple[Candidate, ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    campaign_id: str = "native"
    generations: int | None = 1
    slots: int = 8
    population_size: int | None = None
    max_workers: int = 8
    concurrency: int | None = None
    max_model_turns: int | None = None
    prior_model_turns: int = 0
    max_repairs: int = 1
    model: str = "gpt-5.6-luna"
    effort: str = "high"
    system_prompt: str = "Return one generated policy JSON object."
    output_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    repair_prompt: str = "Repair the generated policy using the diagnostics."
    sandbox_limits: SandboxLimits = field(default_factory=SandboxLimits)
    scientific_contract: bool = False
    max_repair_diagnostics: int = 8
    checkpoint_path: Path | None = None
    require_usage: bool = False
    turn_timeout_seconds: float = 120.0
    infrastructure_retry_limit: int = 3
    infrastructure_retry_backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        population = self.population_size if self.population_size is not None else self.slots
        if isinstance(population, bool) or not isinstance(population, int) or population <= 0:
            raise ValueError("population_size/slots must be a positive integer")
        if self.generations is not None and (
            isinstance(self.generations, bool)
            or not isinstance(self.generations, int)
            or self.generations <= 0
        ):
            raise ValueError("generations must be a positive integer or None")
        workers = self.concurrency if self.concurrency is not None else self.max_workers
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise ValueError("concurrency/max_workers must be a positive integer")
        if (
            isinstance(self.max_repairs, bool)
            or not isinstance(self.max_repairs, int)
            or self.max_repairs < 0
        ):
            raise ValueError("max_repairs must be non-negative")
        if self.max_model_turns is not None and (
            isinstance(self.max_model_turns, bool) or self.max_model_turns < 0
        ):
            raise ValueError("max_model_turns must be non-negative")
        if (
            isinstance(self.prior_model_turns, bool)
            or not isinstance(self.prior_model_turns, int)
            or self.prior_model_turns < 0
        ):
            raise ValueError("prior_model_turns must be non-negative")
        if (
            isinstance(self.turn_timeout_seconds, bool)
            or not isinstance(self.turn_timeout_seconds, int | float)
            or self.turn_timeout_seconds <= 0
        ):
            raise ValueError("turn_timeout_seconds must be positive")
        if (
            isinstance(self.infrastructure_retry_limit, bool)
            or not isinstance(self.infrastructure_retry_limit, int)
            or self.infrastructure_retry_limit < 0
        ):
            raise ValueError("infrastructure_retry_limit must be non-negative")
        if (
            isinstance(self.infrastructure_retry_backoff_seconds, bool)
            or not isinstance(self.infrastructure_retry_backoff_seconds, int | float)
            or self.infrastructure_retry_backoff_seconds < 0
        ):
            raise ValueError("infrastructure_retry_backoff_seconds must be non-negative")
        object.__setattr__(self, "slots", population)
        object.__setattr__(self, "max_workers", workers)


_REPAIRABLE = frozenset(
    {
        "structured_output",
        "invalid_json",
        "invalid_output",
        "invalid_keys",
        "invalid_schema",
        "invalid_schema_version",
        "invalid_array",
        "duplicate_value",
        "unknown_field",
        "used_fields_mismatch",
        "invalid_string",
        "syntax_error",
        "forbidden_syntax",
        "forbidden_input_field",
        "proposal_signal_required",
        "wrong_signature",
        "wrong_function_name",
        "return_contract",
        "static_loop_bound",
        "loop_bound",
        "non_finite_literal",
        "string_too_large",
        "top_level_contract",
        "behavior_error",
    }
)


def infrastructure_retry_allowed(
    result: ProviderResult | Mapping[str, Any] | BaseException,
) -> bool:
    if isinstance(result, BaseException):
        return False
    value = ProviderResult.from_value(result)
    token_count = sum(
        v
        for k, v in value.usage.items()
        if k.lower().endswith("tokens") and isinstance(v, int) and not isinstance(v, bool)
    )
    return (
        value.status.lower() in {"infrastructure", "transport_error", "unavailable", "retryable"}
        and not value.accepted
        and not value.charged
        and not value.content
        and value.uncharged
        and not value.unauthorized_tool_approval
        and token_count == 0
        and value.response in (None, "", {}, [])
    )


can_retry_infrastructure = infrastructure_retry_allowed


class GenerationCoordinator:
    """Run configurable mutation waves with bounded concurrent provider calls."""

    def __init__(
        self,
        provider: GenerationProvider,
        *,
        config: GenerationConfig | None = None,
        campaign_id: str | None = None,
        parent_assignments: Any = None,
        briefs: Any = None,
        prompt_renderer: Any = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_store: Any = None,
        checkpoint_hook: Callable[[Mapping[str, Any]], Any] | None = None,
        resume_hook: Callable[[], Mapping[str, Any] | None] | None = None,
        existing_sources: Sequence[str] = (),
        archive: Any = None,
        parent_selector: Any = None,
        selection_callback: Any = None,
        candidate_callback: Any = None,
        parent_sources: Mapping[str, str] | None = None,
        parent_records: Mapping[str, Any] | None = None,
        search_feedback: Any = "",
        archive_context: Any = "",
        retry_infrastructure: bool = False,
        budget_exhausted: Callable[[], bool | str | None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        behavior_evaluator: Any = None,
        resume_slot_validator: Callable[[GenerationRequest, Mapping[str, Any]], bool] | None = None,
        observer: Any = None,
        event_callback: Any = None,
    ) -> None:
        self.provider = provider
        self.config = config or GenerationConfig(
            campaign_id=campaign_id or "native",
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        )
        if campaign_id is not None:
            self.config = replace(self.config, campaign_id=campaign_id)
        if checkpoint_path is not None:
            self.config = replace(self.config, checkpoint_path=Path(checkpoint_path))
        self.parent_assignments, self.briefs, self.prompt_renderer = (
            parent_assignments,
            briefs,
            prompt_renderer,
        )
        self.checkpoint_store, self.checkpoint_hook, self.resume_hook = (
            checkpoint_store,
            checkpoint_hook,
            resume_hook,
        )
        self.existing_sources, self.archive = tuple(existing_sources), archive
        self.parent_selector, self.selection_callback, self.candidate_callback = (
            parent_selector,
            selection_callback,
            candidate_callback,
        )
        self.parent_sources, self.parent_records = (
            dict(parent_sources or {}),
            dict(parent_records or {}),
        )
        self.search_feedback, self.archive_context = search_feedback, archive_context
        self.retry_infrastructure, self.behavior_evaluator = (
            retry_infrastructure,
            behavior_evaluator,
        )
        self.resume_slot_validator = resume_slot_validator
        self.budget_exhausted = budget_exhausted
        self.stop_requested = stop_requested
        self.observer = observer if observer is not None else event_callback
        self._checkpoint_file = self.config.checkpoint_path

    def _graceful_stop_requested(self) -> bool:
        return bool(self.stop_requested is not None and self.stop_requested())

    def _invoke_before_stop(self, request: GenerationRequest) -> ProviderResult:
        if self._graceful_stop_requested():
            raise _GracefulStopBoundary
        return self._invoke(request)

    def _notify_candidate(self, generation: int, candidate: Candidate, result: SlotResult) -> None:
        """Notify a streaming consumer as soon as a candidate is accepted.

        Selection still runs only after the complete generation is assembled.  This
        hook is intentionally separate so expensive downstream evaluation can begin
        while the remaining slots are still being validated or repaired.
        """

        callback = self.candidate_callback
        if not callable(callback):
            return
        try:
            callback(generation, candidate, result)
        except TypeError:
            callback(candidate)

    def _emit(self, event_type: str, **payload: Any) -> None:
        """Best-effort callback at the native execution boundary.

        Output observers must never turn a successful provider/evaluation turn
        into a failed experiment.  KeyboardInterrupt remains intentionally
        visible to the coordinator; observer failures are operationally
        isolated and represented by the next durable checkpoint/event.
        """

        callback = self.observer
        if not callable(callback):
            return
        try:
            callback(event_type, payload)
        except TypeError:
            try:
                callback(event_type, **payload)
            except Exception:
                return
        except Exception:
            return

    @staticmethod
    def _usage_payload(raw: ProviderResult) -> dict[str, Any]:
        usage = dict(raw.usage) if isinstance(raw.usage, Mapping) else {}
        return {
            "usage": usage,
            "inputTokens": usage.get("inputTokens"),
            "cachedInputTokens": usage.get("cachedInputTokens"),
            "outputTokens": usage.get("outputTokens"),
            "reasoningOutputTokens": usage.get("reasoningOutputTokens"),
            "totalTokens": usage.get("totalTokens"),
            "usage_quality": (
                "exact"
                if usage.get("final") is True and usage.get("partial") is False
                else "partial"
                if usage
                else "unknown"
            ),
        }

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(f"slot-{index:02d}" for index in range(self.config.slots))

    def _context(self, value: Any, generation: int, slot: str, parent: str) -> str:
        if callable(value):
            try:
                value = value(generation, slot, parent)
            except TypeError:
                value = value(generation, slot)
        if isinstance(value, Mapping):
            value = value.get(generation, value.get(str(generation), value))
            if isinstance(value, Mapping):
                value = value.get(slot, value.get(str(slot), ""))
        return str(value or "")

    def _parents(self, generation: int) -> dict[str, str]:
        raw = self.parent_assignments
        if raw is None and callable(self.parent_selector):
            raw = self.parent_selector(generation)
        slots = self.slots
        if isinstance(raw, Mapping):
            value = raw.get(generation, raw.get(str(generation), raw))
            if isinstance(value, Mapping):
                return {
                    slot: str(value.get(slot, value.get(str(i), f"parent-{generation}-{slot}")))
                    for i, slot in enumerate(slots)
                }
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return {slot: str(value[i]) for i, slot in enumerate(slots) if i < len(value)} | {
                    slot: f"parent-{generation}-{slot}" for slot in slots[len(value) :]
                }
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            value = (
                raw[generation]
                if len(raw) == self.config.generations and isinstance(raw[generation], Sequence)
                else raw
            )
            return {slot: str(value[i]) for i, slot in enumerate(slots) if i < len(value)} | {
                slot: f"parent-{generation}-{slot}" for slot in slots[len(value) :]
            }
        return {slot: f"parent-{generation}-{slot}" for slot in slots}

    def _brief(self, generation: int, slot: str) -> Any:
        value = self.briefs
        if value is None:
            return f"mutation brief generation {generation} slot {slot}"
        if isinstance(value, Mapping):
            value = value.get(generation, value.get(str(generation), value))
            return (
                value.get(slot, value.get(str(slot), "")) if isinstance(value, Mapping) else value
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == self.config.generations and isinstance(value[generation], Sequence):
                value = value[generation]
            return value[int(slot[5:])]
        return value

    def _parent_source_metadata(self, parent: str) -> tuple[str, Mapping[str, Any]]:
        source = self.parent_sources.get(parent, "")
        record = self.parent_records.get(parent)
        if isinstance(record, Mapping):
            return str(record.get("source", source)), dict(record)
        if record is not None:
            return str(getattr(record, "source", source)), {}
        return source, {}

    def build_request(
        self,
        generation: int,
        slot: str,
        parent: str,
        *,
        phase: str = "initial",
        diagnostics: Sequence[Mapping[str, Any]] = (),
        repair_source: str = "",
        repair_attempt: int = 0,
    ) -> GenerationRequest:
        brief = self._brief(generation, slot)
        parent_source, parent_metadata = self._parent_source_metadata(parent)
        feedback, archive_context = (
            self._context(self.search_feedback, generation, slot, parent),
            self._context(self.archive_context, generation, slot, parent),
        )
        render_values = {
            "brief": brief,
            "parent_id": parent,
            "parent_source": parent_source,
            "parent_metadata": parent_metadata,
            "search_feedback": feedback,
            "archive_context": archive_context,
            "generation": generation,
            "slot": slot,
            "phase": phase,
            "diagnostics": tuple(diagnostics),
            "repair_source": repair_source,
            "repair_prompt": self.config.repair_prompt,
            "repair_attempt": repair_attempt,
            "max_repairs": self.config.max_repairs,
            "remaining_repairs": max(0, self.config.max_repairs - repair_attempt),
        }
        if self.prompt_renderer is not None:
            try:
                rendered = self.prompt_renderer(**render_values)
            except TypeError:
                rendered = self.prompt_renderer(brief)
        else:
            rendered = brief
        # ``prompt`` is the final model-facing string.  Metadata remains in
        # GenerationRequest.as_dict()/request.json and is never substituted
        # with a compact JSON envelope in the prompt itself.
        prompt = str(rendered)
        if phase == "repair" and self.prompt_renderer is None:
            diagnostics_json = json.dumps(
                [dict(item) for item in diagnostics],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            prompt = (
                f"{self.config.repair_prompt.strip()}\n\n"
                f"Repair attempt {repair_attempt} of {self.config.max_repairs}; "
                f"{max(0, self.config.max_repairs - repair_attempt)} repairs remain "
                "after this attempt.\n\n"
                f"## Previous generated source\n\n```python\n{repair_source}\n```\n\n"
                f"## Repair diagnostics\n\n```json\n{diagnostics_json}\n```"
            )
        validator_contract = render_policy_validator_contract(
            self.config.sandbox_limits,
            scientific=self.config.scientific_contract,
        )
        prompt = f"{prompt.rstrip()}\n\n{validator_contract}\n"
        prompt_hash, brief_id = _hash(prompt), _hash(brief)
        key = request_idempotency_key(
            self.config.campaign_id,
            generation,
            slot,
            parent,
            brief_id,
            prompt_hash,
            phase,
            repair_attempt,
        )
        return GenerationRequest(
            self.config.campaign_id,
            generation,
            slot,
            parent,
            brief_id,
            prompt,
            prompt_hash,
            key,
            phase,
            tuple(diagnostics),
            parent_source,
            parent_metadata,
            feedback,
            archive_context,
            self.config.model,
            self.config.effort,
            self.config.system_prompt,
            self.config.output_schema,
            self.config.repair_prompt,
            repair_attempt,
            self.config.max_repairs,
            max(0, self.config.max_repairs - repair_attempt),
        )

    def _invoke(self, request: GenerationRequest) -> ProviderResult:
        payload = request.as_dict()
        heartbeat_stop = threading.Event()
        turn_started = time.monotonic()
        self._emit(
            "provider_turn_started",
            generation=request.generation,
            slot=request.slot,
            phase=request.phase,
            parent_id=request.parent_id,
            model=request.model,
            effort=request.effort,
            idempotency_key=request.idempotency_key,
            repair_attempt=request.repair_attempt,
            provider_turn_state="running",
            timeout_seconds=self.config.turn_timeout_seconds,
        )

        def heartbeat() -> None:
            while not heartbeat_stop.wait(1.0):
                self._emit(
                    "repair_activity" if request.phase == "repair" else "provider_turn_activity",
                    generation=request.generation,
                    slot=request.slot,
                    phase=request.phase,
                    parent_id=request.parent_id,
                    idempotency_key=request.idempotency_key,
                    repair_attempt=request.repair_attempt,
                    operation_elapsed_seconds=time.monotonic() - turn_started,
                    timeout_seconds=self.config.turn_timeout_seconds,
                    provider_turn_state="running",
                )

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"native-heartbeat-{request.slot}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            if request.phase == "repair" and callable(getattr(self.provider, "repair", None)):
                value = self.provider.repair(payload, tuple(request.diagnostics))  # type: ignore[attr-defined]
            else:
                value = self.provider.generate(payload)
            result = ProviderResult.from_value(value)
            self._emit(
                "provider_turn_completed"
                if result.status.lower() == "completed"
                else "provider_turn_failed",
                generation=request.generation,
                slot=request.slot,
                phase=request.phase,
                parent_id=request.parent_id,
                status=result.status,
                accepted=result.accepted,
                content=result.content,
                charged=result.charged,
                uncharged=result.uncharged,
                retained=bool(isinstance(value, Mapping) and value.get("retained") is True),
                **self._usage_payload(result),
                provider_request_id=result.provider_request_id,
                provider_thread_id=result.provider_thread_id or result.thread_id,
                provider_turn_id=result.provider_turn_id or result.turn_id,
                provider_duration_ms=result.provider_duration_ms,
                operation_elapsed_seconds=max(0.0, time.monotonic() - turn_started),
                error=result.error,
                idempotency_key=request.idempotency_key,
                repair_attempt=request.repair_attempt,
            )
            return result
        except KeyboardInterrupt:
            self._emit(
                "provider_turn_failed",
                generation=request.generation,
                slot=request.slot,
                phase=request.phase,
                parent_id=request.parent_id,
                status="interrupted",
                error="KeyboardInterrupt",
                idempotency_key=request.idempotency_key,
                repair_attempt=request.repair_attempt,
            )
            raise
        except BaseException as exc:
            evidence = getattr(exc, "evidence", {})
            result = ProviderResult.from_value(
                {
                    "status": "infrastructure",
                    "accepted": False,
                    "charged": False,
                    "content": False,
                    "uncharged": True,
                    **(dict(evidence) if isinstance(evidence, Mapping) else {}),
                    "error": str(exc),
                }
            )
            self._emit(
                "provider_turn_failed",
                generation=request.generation,
                slot=request.slot,
                phase=request.phase,
                parent_id=request.parent_id,
                status=result.status,
                accepted=result.accepted,
                content=result.content,
                charged=result.charged,
                uncharged=result.uncharged,
                **self._usage_payload(result),
                provider_request_id=result.provider_request_id,
                provider_thread_id=result.provider_thread_id or result.thread_id,
                provider_turn_id=result.provider_turn_id or result.turn_id,
                provider_duration_ms=result.provider_duration_ms,
                operation_elapsed_seconds=max(0.0, time.monotonic() - turn_started),
                error=result.error,
                idempotency_key=request.idempotency_key,
                repair_attempt=request.repair_attempt,
            )
            return result
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not threading.current_thread():
                heartbeat_thread.join(timeout=0.2)

    def _assess(
        self, request: GenerationRequest, raw: ProviderResult, *, repair: bool = False
    ) -> tuple[Candidate | None, tuple[Mapping[str, Any], ...]]:
        errors: list[Mapping[str, Any]] = []
        if raw.status.lower() != "completed":
            errors.append({"code": "provider_status", "message": raw.status})
        if not raw.accepted or not raw.content:
            errors.append(
                {"code": "turn_provenance", "message": "turn was not accepted/contentful"}
            )
        if raw.response_diagnostics:
            errors.extend(raw.response_diagnostics)
        elif raw.response_projection_valid is False:
            errors.append(
                {
                    "code": "invalid_output",
                    "message": "response does not satisfy the generated-policy contract",
                }
            )
        metadata_status = str(raw.metadata_validation.get("status", ""))
        if metadata_status == "mismatch":
            metadata_errors = raw.metadata_validation.get("errors")
            metadata_error_added = False
            if isinstance(metadata_errors, Sequence) and not isinstance(
                metadata_errors, (str, bytes, bytearray)
            ):
                for item in metadata_errors:
                    if isinstance(item, Mapping):
                        errors.append(item)
                        metadata_error_added = True
            if not metadata_error_added:
                errors.append(
                    {
                        "code": "used_fields_mismatch",
                        "message": "declared used_fields do not match validated source",
                    }
                )
        response = raw.response
        if isinstance(response, (str, bytes, bytearray)):
            try:
                response = json.loads(
                    response.decode("utf-8")
                    if isinstance(response, (bytes, bytearray))
                    else response
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append({"code": "invalid_json", "message": str(exc)[:256]})
        source: str | None = None
        try:
            value = _parse_generated_policy(response)
            source = value.source
        except Exception as exc:
            errors.append(
                {"code": getattr(exc, "code", "structured_output"), "message": str(exc)[:256]}
            )
        validation: ValidationResult | None = None
        behavior: Mapping[str, Any] = {}
        if source is not None:
            if self._graceful_stop_requested():
                raise _GracefulStopBoundary
            self._emit(
                "validation_started",
                generation=request.generation,
                slot=request.slot,
                phase=request.phase,
                parent_id=request.parent_id,
            )
            validation = validate_policy(
                source,
                self.config.sandbox_limits,
                scientific=self.config.scientific_contract,
            )
            validation_diagnostics = [
                item.as_dict() for item in validation.errors[: self.config.max_repair_diagnostics]
            ]
            validation_codes = [str(item["code"]) for item in validation_diagnostics]
            self._emit(
                "validation_completed",
                generation=request.generation,
                slot=request.slot,
                phase=request.phase,
                parent_id=request.parent_id,
                valid=validation.valid,
                validation_status="valid" if validation.valid else "invalid",
                validation_errors=len(validation.errors),
                schema_valid=True,
                parse_outcome="valid",
                schema_outcome="valid",
                validation_codes=validation_codes,
                diagnostics=validation_diagnostics,
                error=", ".join(validation_codes) if validation_codes else None,
            )
            if self._graceful_stop_requested():
                raise _GracefulStopBoundary
            if not validation.valid:
                errors.extend(
                    {**item.as_dict(), "repair_class": "ast"} for item in validation.errors
                )
            elif raw.behavior:
                self._emit(
                    "behavior_probe_started",
                    generation=request.generation,
                    slot=request.slot,
                    phase=request.phase,
                    parent_id=request.parent_id,
                )
                if raw.behavior.get("status") == "failed":
                    message = str(raw.behavior.get("error", "behavior probe failed"))[:256]
                    errors.append({"code": "behavior_error", "message": message})
                    self._emit(
                        "behavior_probe_completed",
                        generation=request.generation,
                        slot=request.slot,
                        phase=request.phase,
                        parent_id=request.parent_id,
                        status="failed",
                        valid=False,
                        error=message,
                    )
                else:
                    behavior = raw.behavior
                    self._emit(
                        "behavior_probe_completed",
                        generation=request.generation,
                        slot=request.slot,
                        phase=request.phase,
                        parent_id=request.parent_id,
                        status="completed",
                        valid=True,
                    )
            elif callable(self.behavior_evaluator):
                self._emit(
                    "behavior_probe_started",
                    generation=request.generation,
                    slot=request.slot,
                    phase=request.phase,
                    parent_id=request.parent_id,
                )
                try:
                    evaluated = self.behavior_evaluator(source, self.config.sandbox_limits)
                    behavior = evaluated[0] if isinstance(evaluated, tuple) else evaluated
                    self._emit(
                        "behavior_probe_completed",
                        generation=request.generation,
                        slot=request.slot,
                        phase=request.phase,
                        parent_id=request.parent_id,
                        status="completed",
                        valid=True,
                    )
                except Exception as exc:
                    errors.append({"code": "behavior_error", "message": str(exc)[:256]})
                    self._emit(
                        "behavior_probe_completed",
                        generation=request.generation,
                        slot=request.slot,
                        phase=request.phase,
                        parent_id=request.parent_id,
                        status="failed",
                        valid=False,
                        error=str(exc)[:256],
                    )
        else:
            validation_diagnostics = [
                {
                    "code": str(item.get("code", "")),
                    "message": str(item.get("message", ""))[:256],
                }
                for item in errors[: self.config.max_repair_diagnostics]
            ]
            validation_codes = [str(item["code"]) for item in validation_diagnostics]
            self._emit(
                "validation_completed",
                generation=request.generation,
                slot=request.slot,
                phase=request.phase,
                parent_id=request.parent_id,
                valid=False,
                validation_status="invalid",
                validation_errors=len(errors),
                schema_valid=False,
                parse_outcome="invalid",
                schema_outcome="invalid",
                validation_codes=validation_codes,
                diagnostics=validation_diagnostics,
                error=", ".join(validation_codes) if validation_codes else "validation failed",
            )
        if self._graceful_stop_requested():
            raise _GracefulStopBoundary
        diagnostics_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
        for item in errors:
            code = str(item.get("code", ""))
            message = str(item.get("message", ""))[:256]
            if code in _REPAIRABLE or item.get("repair_class") == "ast":
                diagnostics_by_identity.setdefault(
                    (code, message),
                    {"code": code, "message": message},
                )
        diagnostics = tuple(diagnostics_by_identity.values())[: self.config.max_repair_diagnostics]
        if errors or validation is None or not validation.valid or source is None:
            return None, diagnostics
        identity = validation.identity
        return Candidate(
            source,
            identity.source_sha256,
            identity.normalized_ast_sha256 or "",
            request.generation,
            request.slot,
            request.parent_id,
            behavior,
            raw.usage,
            raw.request_id,
            raw.thread_id,
            raw.turn_id,
            repair,
            None,
            identity.as_dict(),
        ), diagnostics

    @staticmethod
    def _repair_allowed(raw: ProviderResult) -> bool:
        """Repair malformed content, never replace a terminal provider turn."""

        return raw.status.lower() == "completed" and raw.accepted and raw.content

    def _slot_status(
        self,
        candidate: Candidate | None,
        diagnostics: Sequence[Mapping[str, Any]],
        raw: ProviderResult,
        *,
        repairs: int,
    ) -> str:
        if candidate is not None:
            return "accepted"
        if diagnostics and self._repair_allowed(raw):
            return "repair_pending" if repairs < self.config.max_repairs else "invalid"
        if raw.status.lower() == "completed" and raw.accepted and raw.content:
            return "invalid"
        return "failed"

    def _wait_before_infrastructure_retry(self, retry_number: int) -> None:
        """Apply bounded exponential backoff before an uncharged retry."""

        delay = min(
            60.0,
            float(self.config.infrastructure_retry_backoff_seconds)
            * (2 ** max(0, retry_number - 1)),
        )
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _invalid_source(raw: Mapping[str, Any]) -> str:
        response = raw.get("response")
        if isinstance(response, Mapping) and isinstance(response.get("source"), str):
            return cast(str, response["source"])
        return (
            response.decode("utf-8", "replace")
            if isinstance(response, (bytes, bytearray))
            else str(response or "")
        )

    def _load(self) -> dict[str, Any]:
        if self.resume_hook is not None:
            value = self.resume_hook()
            if not isinstance(value, Mapping):
                raise ValueError("native generation resume state must be an object")
            state = dict(value)
            _validate_generation_state(state)
            return state
        if self._checkpoint_file is not None and self._checkpoint_file.exists():
            try:
                value = read_json(self._checkpoint_file)
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError("cannot read native generation checkpoint") from exc
            if not isinstance(value, dict):
                raise ValueError("native generation checkpoint must be an object")
            _validate_generation_state(value)
            return value
        return {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "campaign_id": self.config.campaign_id,
            "slots": {},
            "callbacks": {},
        }

    def _save(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(payload)
        if self.checkpoint_store is not None:
            self.checkpoint_store.save(
                {
                    "generation": payload.get("generation", 0),
                    "slots": payload.get("slots", {}),
                    "summary": payload.get("summary", {}),
                }
            )
        if self._checkpoint_file is None:
            return
        write_json(self._checkpoint_file, _safe(payload), indent=2)
        self._emit(
            "checkpoint_written",
            checkpoint=str(self._checkpoint_file),
            generation=payload.get("generation", 0),
            completed_slots=sum(
                1
                for item in cast(Mapping[str, Any], payload.get("slots", {})).values()
                if isinstance(item, Mapping)
                and str(item.get("status", "")) in {"accepted", "duplicate", "failed", "invalid"}
            ),
            durable=True,
        )

    @staticmethod
    def _from_slot(value: Mapping[str, Any]) -> SlotResult:
        raw = value.get("candidate")
        candidate = None
        if isinstance(raw, Mapping):
            candidate = Candidate(
                str(raw.get("source", "")),
                str(raw.get("source_sha256", "")),
                str(raw.get("normalized_ast_sha256", "")),
                int(raw.get("generation", value.get("generation", 0))),
                str(raw.get("slot", value.get("slot", ""))),
                str(raw.get("parent_id", value.get("parent_id", ""))),
                cast(Mapping[str, Any], raw.get("behavior_signature", {})),
                cast(Mapping[str, Any], raw.get("usage", {})),
                raw.get("request_id"),
                raw.get("thread_id"),
                raw.get("turn_id"),
                bool(raw.get("repair", False)),
                raw.get("duplicate_of"),
                cast(Mapping[str, Any], raw.get("source_identity", {})),
            )
        return SlotResult(
            int(value.get("generation", 0)),
            str(value.get("slot", "")),
            str(value.get("parent_id", "")),
            str(value.get("status", "failed")),
            candidate,
            tuple(item for item in value.get("errors", ()) if isinstance(item, Mapping)),
            int(value.get("repairs", 0)),
            cast(Mapping[str, Any], value.get("initial", {})),
            cast(Mapping[str, Any] | None, value.get("repair")),
            cast(Mapping[str, Any], value.get("request", {})),
            cast(Mapping[str, Any], value.get("raw_result", {})),
            value.get("duplicate_of"),
            cast(Mapping[str, Any], value.get("initial_request", {})),
            tuple(
                str(item)
                for item in value.get("repair_idempotency_keys", ())
                if isinstance(item, str) and item
            ),
            int(value.get("remaining_repairs", 0)),
        )

    @staticmethod
    def _recovered_event_payload(result: SlotResult) -> dict[str, Any]:
        evidence = (
            result.repair
            if isinstance(result.repair, Mapping) and result.repair
            else result.initial
            if result.initial
            else result.raw_result
        )
        provider_result = ProviderResult.from_value(
            evidence if isinstance(evidence, Mapping) else {}
        )
        error: str | None = None
        codes = [
            str(item.get("code", ""))
            for item in result.errors
            if isinstance(item.get("code"), str) and item.get("code")
        ]
        if codes:
            error = ", ".join(codes)
        elif isinstance(evidence, Mapping):
            behavior = evidence.get("behavior")
            if isinstance(behavior, Mapping) and behavior.get("status") == "failed":
                reason = behavior.get("error")
                error = f"behavior probe: {reason or 'failed'}"
            elif isinstance(evidence.get("error"), str) and evidence.get("error"):
                error = str(evidence["error"])
        if error is None and result.status == "invalid":
            error = "invalid candidate"
        candidate_id = (
            f"g{result.generation:04d}-{result.slot}" if result.candidate is not None else None
        )
        charged = (
            evidence.get("charged")
            if isinstance(evidence, Mapping) and isinstance(evidence.get("charged"), bool)
            else None
        )
        return {
            "error": error,
            "candidate_id": candidate_id,
            "validation_status": ("passed" if result.candidate is not None else "unknown"),
            "probe_status": ("passed" if result.candidate is not None else "unknown"),
            "charged": charged,
            "provider_duration_ms": provider_result.provider_duration_ms,
            **GenerationCoordinator._usage_payload(provider_result),
            "content": (
                evidence.get("content")
                if isinstance(evidence, Mapping) and isinstance(evidence.get("content"), bool)
                else None
            ),
        }

    def run_request(
        self, request: GenerationRequest, *, allow_repair: bool = True, retained_result: Any = None
    ) -> SlotResult:
        raw = (
            ProviderResult.from_value(retained_result)
            if retained_result is not None
            else self._invoke(request)
        )
        candidate, diagnostics = self._assess(request, raw)
        result = SlotResult(
            generation=request.generation,
            slot=request.slot,
            parent_id=request.parent_id,
            status=self._slot_status(candidate, diagnostics, raw, repairs=0),
            candidate=candidate,
            errors=tuple(diagnostics if not candidate else ()),
            repairs=0,
            initial=raw.as_dict(),
            request=request.as_dict(),
            raw_result=raw.as_dict(),
            initial_request=request.as_dict(),
            remaining_repairs=self.config.max_repairs,
        )
        if not (allow_repair and result.status == "repair_pending"):
            return result
        repair_request = self.build_request(
            request.generation,
            request.slot,
            request.parent_id,
            phase="repair",
            diagnostics=diagnostics,
            repair_source=self._invalid_source(raw.as_dict()),
            repair_attempt=1,
        )
        self._emit(
            "repair_started",
            generation=request.generation,
            slot=request.slot,
            parent_id=request.parent_id,
            phase="repair",
            diagnostics=list(diagnostics),
            repair_attempt=1,
            remaining_repairs=max(0, self.config.max_repairs - 1),
        )
        repair_raw = self._invoke(repair_request)
        repaired, repair_diagnostics = self._assess(repair_request, repair_raw, repair=True)
        repaired_status = self._slot_status(repaired, repair_diagnostics, repair_raw, repairs=1)
        self._emit(
            "repair_completed",
            generation=request.generation,
            slot=request.slot,
            parent_id=request.parent_id,
            phase="repair",
            status=repaired_status,
            repair_state=(
                "accepted"
                if repaired_status == "accepted"
                else "repair_pending"
                if repaired_status == "repair_pending"
                else "repair_failed"
            ),
            repairs=1,
            remaining_repairs=max(0, self.config.max_repairs - 1),
            validation_codes=[str(item.get("code", "")) for item in repair_diagnostics],
        )
        return SlotResult(
            generation=request.generation,
            slot=request.slot,
            parent_id=request.parent_id,
            status=repaired_status,
            candidate=repaired,
            errors=tuple(repair_diagnostics if not repaired else ()),
            repairs=1,
            initial=raw.as_dict(),
            repair=repair_raw.as_dict(),
            request=repair_request.as_dict(),
            raw_result=repair_raw.as_dict(),
            initial_request=request.as_dict(),
            repair_idempotency_keys=(repair_request.idempotency_key,),
            remaining_repairs=max(0, self.config.max_repairs - 1),
        )

    def run(self, *, resume: bool = True) -> GenerationResult:
        state: dict[str, Any] = (
            self._load()
            if resume
            else {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "campaign_id": self.config.campaign_id,
                "slots": {},
                "callbacks": {},
            }
        )
        slots_state = cast(dict[str, Any], state.setdefault("slots", {}))
        callbacks = cast(dict[str, Any], state.setdefault("callbacks", {}))

        def cached_requires_provider(value: Mapping[str, Any]) -> bool:
            """Return whether cached state still needs a provider artifact.

            A terminal slot result is already a durable scientific result in
            the generation checkpoint.  It must be recoverable even when an
            older provider manifest used a different idempotency key.  Only
            in-flight/retryable states need the provider-artifact validator;
            otherwise a safe resume would redo completed slots.
            """

            status = str(value.get("status", ""))
            return status in {
                "pending",
                "repair_running",
            } or _slot_failure_is_retryable(value)

        def cached_for_request(request: GenerationRequest) -> Mapping[str, Any] | None:
            """Find durable slot state even when a mutable prompt changed.

            Runtime controls are intentionally mutable.  They are included in
            the rendered prompt, so changing (for example) ``thread_count``
            changes the request hash even though the completed slot result is
            still valid.  Prefer the exact key, then match the durable slot
            identity from the checkpoint rather than regenerating the turn.
            """

            direct = slots_state.get(request.idempotency_key)
            if isinstance(direct, Mapping):
                return direct
            matches: list[Mapping[str, Any]] = []
            for value in slots_state.values():
                if not isinstance(value, Mapping):
                    continue
                if value.get("generation") != request.generation:
                    continue
                if str(value.get("slot", "")) != request.slot:
                    continue
                if str(value.get("parent_id", "")) != request.parent_id:
                    continue
                stored_request = value.get("initial_request")
                if not isinstance(stored_request, Mapping):
                    stored_request = value.get("request")
                if (
                    isinstance(stored_request, Mapping)
                    and str(stored_request.get("phase", "initial")) != request.phase
                ):
                    continue
                matches.append(value)
            if not matches:
                return None
            ranks = {
                "accepted": 100,
                "duplicate": 90,
                "invalid": 80,
                "failed": 70,
                "repair_pending": 60,
                "repair_running": 40,
                "stopped": 20,
                "pending": 10,
            }
            matches.sort(key=lambda item: ranks.get(str(item.get("status", "")), 0), reverse=True)
            return matches[0]

        seen: dict[str, str] = {}
        for index, source in enumerate(self.existing_sources):
            identity = validate_policy(
                source,
                self.config.sandbox_limits,
                scientific=self.config.scientific_contract,
            ).identity
            if identity.normalized_ast_sha256:
                seen[identity.normalized_ast_sha256] = f"existing-{index}"
        generations: list[tuple[SlotResult, ...]] = []
        unique: list[Candidate] = []
        stored_turns = state.get("model_turns_used", 0)
        turns = max(
            self.config.prior_model_turns,
            (
                int(stored_turns)
                if isinstance(stored_turns, int) and not isinstance(stored_turns, bool)
                else 0
            ),
        )
        live_turns = 0
        repairs = 0
        recovered = 0
        stopped = False
        stopped_reason: str | None = None
        infrastructure_failed = False
        state["model_turns_used"] = turns

        def reserve_model_turn() -> None:
            nonlocal turns, live_turns
            turns += 1
            live_turns += 1
            state["model_turns_used"] = turns
            self._save(state)

        def release_reserved_turn() -> None:
            nonlocal turns, live_turns
            turns = max(self.config.prior_model_turns, turns - 1)
            live_turns = max(0, live_turns - 1)
            state["model_turns_used"] = turns
            self._save(state)

        def budget_reason() -> str | None:
            if self.config.max_model_turns is not None and turns >= self.config.max_model_turns:
                return "max_model_turns"
            if self.budget_exhausted is not None:
                external_reason = self.budget_exhausted()
                if isinstance(external_reason, str) and external_reason:
                    return external_reason
                if external_reason:
                    return "wall_seconds"
            return None

        stored_next_generation = state.get("next_generation")
        if isinstance(stored_next_generation, int) and stored_next_generation >= 0:
            first_generation = stored_next_generation
        else:
            previous_summary = state.get("summary")
            previous_generation = state.get("generation")
            first_generation = (
                int(previous_generation) + 1
                if isinstance(previous_summary, Mapping)
                and previous_summary.get("status") == "completed"
                and isinstance(previous_generation, int)
                and previous_generation >= 0
                else 0
            )
        logical_slots: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
        for value in slots_state.values():
            if not isinstance(value, Mapping):
                continue
            value_generation = value.get("generation")
            value_slot = value.get("slot")
            if (
                isinstance(value_generation, int)
                and not isinstance(value_generation, bool)
                and isinstance(value_slot, str)
            ):
                logical_slots.setdefault((value_generation, value_slot), []).append(value)
        status_rank = {
            "accepted": 100,
            "duplicate": 90,
            "invalid": 80,
            "failed": 70,
            "repair_pending": 60,
            "repair_running": 40,
            "stopped": 20,
            "pending": 10,
        }
        authoritative_slots = [
            max(
                values,
                key=lambda item: status_rank.get(str(item.get("status", "")), 0),
            )
            for values in logical_slots.values()
        ]
        retry_generations = {
            int(value["generation"])
            for value in authoritative_slots
            if (
                _slot_failure_is_retryable(value)
                or str(value.get("status", ""))
                in {"repair_pending", "repair_running", "stopped"}
            )
        }
        if retry_generations and min(retry_generations) < first_generation:
            first_generation = min(retry_generations)
            state["next_generation"] = first_generation
            for collection_name in ("callbacks", "selection"):
                collection = state.get(collection_name)
                if isinstance(collection, dict):
                    for key in tuple(collection):
                        if str(key).isdigit() and int(key) >= first_generation:
                            collection.pop(key, None)
            self._save(state)
        state.setdefault("next_generation", first_generation)
        generation_limit = self.config.generations
        previous_selection = state.get("selection")
        if (
            self.parent_assignments is None
            and first_generation > 0
            and isinstance(previous_selection, Mapping)
        ):
            selected = previous_selection.get(str(first_generation - 1))
            if isinstance(selected, Mapping):
                self.parent_assignments = {first_generation: dict(selected)}
            elif isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
                self.parent_assignments = {first_generation: list(selected)}
        generation = first_generation
        while generation_limit is None or generation < generation_limit:
            boundary = budget_reason()
            has_retained_generation_state = any(
                isinstance(value, Mapping) and value.get("generation") == generation
                for value in slots_state.values()
            )
            if boundary is not None and not has_retained_generation_state:
                stopped = True
                stopped_reason = boundary
                break
            self._emit(
                "generation_started",
                generation=generation,
                generation_limit=generation_limit,
                population_size=self.config.slots,
                configured_concurrency=self.config.max_workers,
                effective_concurrency=self.config.max_workers,
                max_model_turns=self.config.max_model_turns,
                model=self.config.model,
                effort=self.config.effort,
                phase="initial",
            )
            parents = self._parents(generation)
            results: dict[str, SlotResult] = {}
            futures: dict[Any, tuple[str, GenerationRequest, int]] = {}
            initial_keys: dict[str, str] = {}
            close_provider = getattr(self.provider, "close", None)
            with _InterruptibleThreadPoolExecutor(
                max_workers=self.config.max_workers,
                thread_name_prefix=f"native-g{generation}",
                on_interrupt=close_provider if callable(close_provider) else None,
            ) as pool:
                for slot in self.slots:
                    request = self.build_request(generation, slot, parents[slot])
                    initial_keys[slot] = request.idempotency_key
                    self._emit(
                        "slot_queued",
                        generation=generation,
                        generation_limit=generation_limit,
                        slot=slot,
                        parent_id=parents[slot],
                        parent_status=("root" if parents[slot].startswith("parent-") else "parent"),
                        phase=request.phase,
                        status="queued",
                        completed_slots=len(results),
                        population_size=self.config.slots,
                    )
                    cached = cached_for_request(request)
                    if isinstance(cached, Mapping) and cached.get("status") in {
                        "repair_pending",
                        "stopped",
                    }:
                        retained_initial = cached.get("initial")
                        if not isinstance(retained_initial, Mapping):
                            retained_initial = cached.get("raw_result")
                        if isinstance(retained_initial, Mapping) and retained_initial:
                            reassessed = self.run_request(
                                request,
                                allow_repair=False,
                                retained_result=retained_initial,
                            )
                            if reassessed.status == "accepted":
                                cached = reassessed.as_dict()
                                slots_state[request.idempotency_key] = cached
                                self._save(state)
                    if (
                        isinstance(cached, Mapping)
                        and cached_requires_provider(cached)
                        and self.resume_slot_validator is not None
                        and not self.resume_slot_validator(request, cached)
                    ):
                        cached = None
                    if (
                        isinstance(cached, Mapping)
                        and cached.get("status")
                        not in {"pending", "budget_exhausted", "stopped"}
                        and not _slot_failure_is_retryable(cached)
                    ):
                        results[slot] = self._from_slot(cached)
                        recovered += 1
                        self._emit(
                            "slot_queued",
                            generation=generation,
                            slot=slot,
                            parent_id=parents[slot],
                            phase=request.phase,
                            status="recovered",
                            recovered=True,
                            recovered_status=results[slot].status,
                            repairs=results[slot].repairs,
                            remaining_repairs=results[slot].remaining_repairs,
                            validation_codes=[
                                str(error.get("code", "")) for error in results[slot].errors
                            ],
                            **self._recovered_event_payload(results[slot]),
                            completed_slots=len(results),
                            population_size=self.config.slots,
                        )
                        recovered_candidate = results[slot].candidate
                        if recovered_candidate is not None:
                            self._notify_candidate(
                                generation,
                                recovered_candidate,
                                results[slot],
                            )
                        continue
                    exhausted_reason = budget_reason()
                    if exhausted_reason is not None:
                        results[slot] = SlotResult(
                            generation,
                            slot,
                            parents[slot],
                            (
                                "stopped"
                                if exhausted_reason == "operator_stop"
                                else "budget_exhausted"
                            ),
                            request=request.as_dict(),
                        )
                        stopped = True
                        stopped_reason = exhausted_reason
                        self._emit(
                            "budget_boundary_reached",
                            generation=generation,
                            slot=slot,
                            reason=exhausted_reason,
                            max_model_turns=self.config.max_model_turns,
                            completed_turns=turns,
                        )
                        continue
                    reserve_model_turn()
                    futures[pool.submit(self._invoke_before_stop, request)] = (
                        slot,
                        request,
                        0,
                    )
                while futures:
                    future = next(as_completed(tuple(futures)))
                    slot, request, retry_count = futures.pop(future)
                    try:
                        raw = future.result()
                    except _GracefulStopBoundary:
                        release_reserved_turn()
                        results[slot] = SlotResult(
                            generation=generation,
                            slot=slot,
                            parent_id=request.parent_id,
                            status="stopped",
                            request=request.as_dict(),
                            initial_request=request.as_dict(),
                            remaining_repairs=self.config.max_repairs,
                        )
                        slots_state[request.idempotency_key] = results[slot].as_dict()
                        self._save(state)
                        stopped = True
                        stopped_reason = "operator_stop"
                        continue
                    if raw.retained:
                        release_reserved_turn()
                        recovered += 1
                    uncharged_infrastructure = not raw.retained and infrastructure_retry_allowed(
                        raw
                    )
                    retryable_infrastructure = (
                        self.retry_infrastructure and uncharged_infrastructure
                    )
                    if uncharged_infrastructure:
                        # An uncharged infrastructure failure did not consume a
                        # model turn; keep it out of the cumulative turn budget.
                        release_reserved_turn()
                    if (
                        retryable_infrastructure
                        and retry_count < self.config.infrastructure_retry_limit
                    ):
                        exhausted_reason = budget_reason()
                        if exhausted_reason is None:
                            self._wait_before_infrastructure_retry(retry_count + 1)
                            exhausted_reason = budget_reason()
                        if exhausted_reason is None:
                            reserve_model_turn()
                            futures[pool.submit(self._invoke_before_stop, request)] = (
                                slot,
                                request,
                                retry_count + 1,
                            )
                            self._emit(
                                "slot_queued",
                                generation=generation,
                                slot=slot,
                                parent_id=request.parent_id,
                                phase=request.phase,
                                status="retrying",
                                retry_count=retry_count + 1,
                                retry_limit=self.config.infrastructure_retry_limit,
                                remaining_model_turns=(
                                    max(0, self.config.max_model_turns - turns)
                                    if self.config.max_model_turns is not None
                                    else None
                                ),
                            )
                            continue
                        results[slot] = SlotResult(
                            generation=generation,
                            slot=slot,
                            parent_id=request.parent_id,
                            status=(
                                "stopped"
                                if exhausted_reason == "operator_stop"
                                else "budget_exhausted"
                            ),
                            initial=raw.as_dict(),
                            request=request.as_dict(),
                            raw_result=raw.as_dict(),
                            initial_request=request.as_dict(),
                            remaining_repairs=self.config.max_repairs,
                        )
                        slots_state[request.idempotency_key] = results[slot].as_dict()
                        self._save(state)
                        stopped = True
                        stopped_reason = exhausted_reason
                        self._emit(
                            "budget_boundary_reached",
                            generation=generation,
                            slot=slot,
                            reason=exhausted_reason,
                            max_model_turns=self.config.max_model_turns,
                            completed_turns=turns,
                        )
                        continue
                    if self._graceful_stop_requested():
                        results[slot] = SlotResult(
                            generation=generation,
                            slot=slot,
                            parent_id=request.parent_id,
                            status="stopped",
                            initial=raw.as_dict(),
                            request=request.as_dict(),
                            raw_result=raw.as_dict(),
                            initial_request=request.as_dict(),
                            remaining_repairs=self.config.max_repairs,
                        )
                        slots_state[request.idempotency_key] = results[slot].as_dict()
                        self._save(state)
                        stopped = True
                        stopped_reason = "operator_stop"
                        continue
                    try:
                        candidate, diagnostics = self._assess(request, raw)
                    except _GracefulStopBoundary:
                        results[slot] = SlotResult(
                            generation=generation,
                            slot=slot,
                            parent_id=request.parent_id,
                            status="stopped",
                            initial=raw.as_dict(),
                            request=request.as_dict(),
                            raw_result=raw.as_dict(),
                            initial_request=request.as_dict(),
                            remaining_repairs=self.config.max_repairs,
                        )
                        slots_state[request.idempotency_key] = results[slot].as_dict()
                        self._save(state)
                        stopped = True
                        stopped_reason = "operator_stop"
                        continue
                    results[slot] = SlotResult(
                        generation=generation,
                        slot=slot,
                        parent_id=request.parent_id,
                        status=self._slot_status(candidate, diagnostics, raw, repairs=0),
                        candidate=candidate,
                        errors=tuple(diagnostics if not candidate else ()),
                        repairs=0,
                        initial=raw.as_dict(),
                        request=request.as_dict(),
                        raw_result=raw.as_dict(),
                        initial_request=request.as_dict(),
                        remaining_repairs=self.config.max_repairs,
                    )
                    slots_state[request.idempotency_key] = results[slot].as_dict()
                    self._save(state)
                    if self._graceful_stop_requested():
                        stopped = True
                        stopped_reason = "operator_stop"
                    elif candidate is not None:
                        self._notify_candidate(generation, candidate, results[slot])
                    self._emit(
                        "slot_queued",
                        generation=generation,
                        slot=slot,
                        parent_id=request.parent_id,
                        phase=request.phase,
                        status=results[slot].status,
                        completed_slots=len(results),
                        population_size=self.config.slots,
                        remaining_model_turns=(
                            max(0, self.config.max_model_turns - turns)
                            if self.config.max_model_turns is not None
                            else None
                        ),
                    )
            for slot in self.slots:
                item = results.get(slot)
                initial_key = initial_keys[slot]
                if (
                    item is not None
                    and item.status == "repair_pending"
                    and item.repairs >= self.config.max_repairs
                ):
                    item = replace(
                        item,
                        status="invalid",
                        remaining_repairs=0,
                    )
                    results[slot] = item
                    slots_state[initial_key] = item.as_dict()
                    self._save(state)
                    self._emit(
                        "repair_completed",
                        generation=generation,
                        slot=slot,
                        parent_id=item.parent_id,
                        phase="repair",
                        status="invalid",
                        repair_state="repair_failed",
                        repairs=item.repairs,
                        remaining_repairs=0,
                        retained=True,
                        validation_codes=[str(error.get("code", "")) for error in item.errors],
                    )
                repair_boundary = (
                    budget_reason()
                    if item is not None and item.status == "repair_pending"
                    else None
                )
                if repair_boundary is not None:
                    stopped = True
                    stopped_reason = repair_boundary
                while (
                    item is not None
                    and item.status in {"repair_pending", "repair_running"}
                    and item.errors
                    and (
                        item.status == "repair_running"
                        or self.config.max_model_turns is None
                        or turns < self.config.max_model_turns
                    )
                    and (self.budget_exhausted is None or not self.budget_exhausted())
                ):
                    resuming_repair = item.status == "repair_running"
                    if resuming_repair:
                        req = GenerationRequest.from_value(item.request)
                        repair_attempt = item.repairs
                        repair_keys = item.repair_idempotency_keys or (req.idempotency_key,)
                    else:
                        if item.repairs >= self.config.max_repairs:
                            item = replace(
                                item,
                                status="invalid",
                                remaining_repairs=0,
                            )
                            results[slot] = item
                            slots_state[initial_key] = item.as_dict()
                            self._save(state)
                            break
                        repair_attempt = item.repairs + 1
                        req = self.build_request(
                            generation,
                            slot,
                            item.parent_id,
                            phase="repair",
                            diagnostics=item.errors,
                            repair_source=self._invalid_source(item.raw_result),
                            repair_attempt=repair_attempt,
                        )
                        repair_keys = (*item.repair_idempotency_keys, req.idempotency_key)
                        item = replace(
                            item,
                            status="repair_running",
                            repairs=repair_attempt,
                            request=req.as_dict(),
                            repair_idempotency_keys=repair_keys,
                            remaining_repairs=max(0, self.config.max_repairs - repair_attempt),
                        )
                        results[slot] = item
                        slots_state[initial_key] = item.as_dict()
                        slots_state[req.idempotency_key] = item.as_dict()
                        self._save(state)
                    self._emit(
                        "repair_started",
                        generation=generation,
                        slot=slot,
                        parent_id=item.parent_id,
                        phase="repair",
                        diagnostics=list(item.errors),
                        repair_attempt=repair_attempt,
                        max_repairs=self.config.max_repairs,
                        remaining_repairs=max(0, self.config.max_repairs - repair_attempt),
                        retained=resuming_repair,
                    )
                    cached = cached_for_request(req)
                    if (
                        isinstance(cached, Mapping)
                        and cached_requires_provider(cached)
                        and self.resume_slot_validator is not None
                        and not self.resume_slot_validator(req, cached)
                    ):
                        cached = None
                    if isinstance(cached, Mapping) and cached.get("status") not in {
                        "pending",
                        "repair_pending",
                        "repair_running",
                    }:
                        repaired = self._from_slot(cached)
                        recovered += 1
                        repaired_retained = True
                    else:
                        reserve_model_turn()
                        raw = self._invoke(req)
                        repaired_retained = raw.retained
                        if raw.retained:
                            release_reserved_turn()
                            recovered += 1
                        else:
                            if not infrastructure_retry_allowed(raw):
                                repairs += 1
                        if infrastructure_retry_allowed(raw):
                            # Uncharged failures are safe to retry and do not
                            # consume the model-turn budget.
                            release_reserved_turn()
                        if self._graceful_stop_requested():
                            deferred = replace(
                                item,
                                status="repair_running",
                                repair=raw.as_dict(),
                                raw_result=raw.as_dict(),
                            )
                            for state_key in (initial_key, *repair_keys):
                                slots_state[state_key] = deferred.as_dict()
                            self._save(state)
                            results[slot] = deferred
                            item = deferred
                            stopped = True
                            stopped_reason = "operator_stop"
                            break
                        retry_count = 0
                        retry_deferred = False
                        while (
                            self.retry_infrastructure
                            and not raw.retained
                            and infrastructure_retry_allowed(raw)
                            and retry_count < self.config.infrastructure_retry_limit
                        ):
                            exhausted_reason = budget_reason()
                            if exhausted_reason is None:
                                self._wait_before_infrastructure_retry(retry_count + 1)
                                exhausted_reason = budget_reason()
                            if exhausted_reason is not None:
                                deferred = replace(
                                    item,
                                    status="repair_running",
                                    repair=raw.as_dict(),
                                    raw_result=raw.as_dict(),
                                )
                                for state_key in (initial_key, *repair_keys):
                                    slots_state[state_key] = deferred.as_dict()
                                self._save(state)
                                results[slot] = deferred
                                item = deferred
                                stopped = True
                                stopped_reason = exhausted_reason
                                retry_deferred = True
                                self._emit(
                                    "repair_retry_deferred",
                                    generation=generation,
                                    slot=slot,
                                    parent_id=item.parent_id,
                                    phase="repair",
                                    repair_attempt=repair_attempt,
                                    reason=exhausted_reason,
                                    retry_count=retry_count,
                                    retry_limit=self.config.infrastructure_retry_limit,
                                )
                                break
                            retry_count += 1
                            self._emit(
                                "repair_retrying",
                                generation=generation,
                                slot=slot,
                                parent_id=item.parent_id,
                                phase="repair",
                                repair_attempt=repair_attempt,
                                retry_count=retry_count,
                                retry_limit=self.config.infrastructure_retry_limit,
                            )
                            reserve_model_turn()
                            raw = self._invoke(req)
                            repaired_retained = raw.retained
                            if raw.retained:
                                release_reserved_turn()
                                recovered += 1
                            else:
                                if not infrastructure_retry_allowed(raw):
                                    repairs += 1
                            if infrastructure_retry_allowed(raw):
                                release_reserved_turn()
                        if retry_deferred:
                            break
                        if (
                            self.retry_infrastructure
                            and not raw.retained
                            and infrastructure_retry_allowed(raw)
                        ):
                            deferred = replace(
                                item,
                                status="repair_running",
                                repair=raw.as_dict(),
                                raw_result=raw.as_dict(),
                            )
                            for state_key in (initial_key, *repair_keys):
                                slots_state[state_key] = deferred.as_dict()
                            self._save(state)
                            results[slot] = deferred
                            item = deferred
                            self._emit(
                                "repair_retry_exhausted",
                                generation=generation,
                                slot=slot,
                                parent_id=item.parent_id,
                                phase="repair",
                                repair_attempt=repair_attempt,
                                retry_count=retry_count,
                                retry_limit=self.config.infrastructure_retry_limit,
                            )
                            break
                        try:
                            candidate, diagnostics = self._assess(req, raw, repair=True)
                        except _GracefulStopBoundary:
                            deferred = replace(
                                item,
                                status="repair_running",
                                repair=raw.as_dict(),
                                raw_result=raw.as_dict(),
                            )
                            for state_key in (initial_key, *repair_keys):
                                slots_state[state_key] = deferred.as_dict()
                            self._save(state)
                            results[slot] = deferred
                            item = deferred
                            stopped = True
                            stopped_reason = "operator_stop"
                            break
                        status = self._slot_status(
                            candidate,
                            diagnostics,
                            raw,
                            repairs=repair_attempt,
                        )
                        repaired = SlotResult(
                            generation=generation,
                            slot=slot,
                            parent_id=item.parent_id,
                            status=status,
                            candidate=candidate,
                            errors=tuple(diagnostics if not candidate else ()),
                            repairs=repair_attempt,
                            initial=item.initial,
                            repair=raw.as_dict(),
                            request=req.as_dict(),
                            raw_result=raw.as_dict(),
                            initial_request=item.initial_request,
                            repair_idempotency_keys=repair_keys,
                            remaining_repairs=max(0, self.config.max_repairs - repair_attempt),
                        )
                    for state_key in (initial_key, *repair_keys):
                        slots_state[state_key] = repaired.as_dict()
                    self._save(state)
                    self._emit(
                        "repair_completed",
                        generation=generation,
                        slot=slot,
                        parent_id=item.parent_id,
                        phase="repair",
                        status=repaired.status,
                        repair_state=(
                            "accepted"
                            if repaired.status == "accepted"
                            else "repair_pending"
                            if repaired.status == "repair_pending"
                            else "repair_failed"
                        ),
                        repairs=repaired.repairs,
                        remaining_repairs=repaired.remaining_repairs,
                        retained=repaired_retained,
                        validation_codes=[str(error.get("code", "")) for error in repaired.errors],
                    )
                    results[slot] = repaired
                    item = repaired
                    if self._graceful_stop_requested():
                        stopped = True
                        stopped_reason = "operator_stop"
                    elif repaired.candidate is not None:
                        self._notify_candidate(generation, repaired.candidate, repaired)
            ordered: list[SlotResult] = []
            generation_candidates: list[Candidate] = []
            for slot in self.slots:
                item = results.get(slot, SlotResult(generation, slot, parents[slot], "failed"))
                candidate = item.candidate
                if candidate is not None:
                    duplicate = seen.get(candidate.normalized_ast_sha256)
                    if duplicate is not None:
                        candidate = replace(candidate, duplicate_of=duplicate)
                        item = replace(
                            item, status="duplicate", candidate=candidate, duplicate_of=duplicate
                        )
                        self._emit(
                            "candidate_archived",
                            generation=generation,
                            slot=slot,
                            parent_id=item.parent_id,
                            candidate_id=duplicate,
                            status="duplicate",
                            duplicate_of=duplicate,
                            archive_size=len(seen),
                        )
                    else:
                        key = f"g{generation:04d}-{slot}"
                        seen[candidate.normalized_ast_sha256] = key
                        unique.append(candidate)
                        generation_candidates.append(candidate)
                        self.parent_sources[key] = candidate.source
                        self.parent_records[key] = candidate.as_dict()
                        if self.archive is not None and callable(
                            getattr(self.archive, "append", None)
                        ):
                            self.archive.append(
                                {
                                    "program_id": key,
                                    "source": candidate.source,
                                    "source_sha256": candidate.source_sha256,
                                    "normalized_ast_sha256": candidate.normalized_ast_sha256,
                                    "generation": generation,
                                    "slot": slot,
                                    "parent_id": candidate.parent_id,
                                    "usage": dict(candidate.usage),
                                }
                            )
                        self._emit(
                            "candidate_archived",
                            generation=generation,
                            slot=slot,
                            parent_id=item.parent_id,
                            candidate_id=key,
                            status="accepted",
                            source_sha256=candidate.source_sha256,
                            normalized_ast_sha256=candidate.normalized_ast_sha256,
                            source_lines=len(candidate.source.splitlines()),
                            archive_size=len(seen),
                        )
                elif item.status in {"failed", "invalid", "budget_exhausted", "stopped"}:
                    self._emit(
                        "candidate_archived",
                        generation=generation,
                        slot=slot,
                        parent_id=item.parent_id,
                        status=("invalid" if item.status in {"failed", "invalid"} else item.status),
                        errors=list(item.errors),
                        archive_size=len(seen),
                    )
                ordered.append(item)
            generations.append(tuple(ordered))
            state["generation"] = generation
            self._save(state)
            generation_retryable = any(
                _slot_failure_is_retryable(item.as_dict())
                or item.status in {"repair_pending", "repair_running"}
                for item in ordered
            )
            if (
                not stopped
                and not generation_retryable
                and self.selection_callback is not None
                and str(generation) not in callbacks
            ):
                try:
                    selected = self.selection_callback(
                        generation, tuple(generation_candidates), tuple(ordered)
                    )
                except TypeError:
                    selected = self.selection_callback(tuple(generation_candidates))
                if selected is not None:
                    state.setdefault("selection", {})[str(generation)] = _safe(selected)
                    # A selector may directly provide the next parent vector.
                    # Preserve it as an in-memory assignment while retaining
                    # the callback result in the checkpoint for auditability.
                    if generation_limit is None or generation + 1 < generation_limit:
                        if isinstance(selected, Mapping):
                            self.parent_assignments = {generation + 1: dict(selected)}
                        elif isinstance(selected, Sequence) and not isinstance(
                            selected, (str, bytes)
                        ):
                            parent_ids: list[str] = []
                            for _index, value in enumerate(selected):
                                if isinstance(value, Candidate):
                                    parent_ids.append(f"g{value.generation:04d}-{value.slot}")
                                else:
                                    parent_ids.append(str(value))
                            self.parent_assignments = {generation + 1: parent_ids}
                    self._emit(
                        "selection_completed",
                        generation=generation,
                        selected_parents=_safe(selected),
                        elite_size=(
                            len(selected)
                            if isinstance(selected, Sequence)
                            and not isinstance(selected, (str, bytes))
                            else len(selected)
                            if isinstance(selected, Mapping)
                            else None
                        ),
                    )
                callbacks[str(generation)] = {"status": "completed"}
                self._save(state)
            if not stopped and not generation_retryable:
                state["next_generation"] = generation + 1
                self._save(state)
            if generation_retryable and not stopped:
                infrastructure_failed = True
            self._emit(
                "generation_completed",
                generation=generation,
                generation_limit=generation_limit,
                completed_slots=len(ordered),
                population_size=self.config.slots,
                accepted_candidates=sum(
                    1 for group in generations for item in group if item.status == "accepted"
                ),
                invalid_candidates=sum(
                    1
                    for group in generations
                    for item in group
                    if item.status in {"failed", "invalid"}
                ),
                duplicate_candidates=sum(
                    1 for group in generations for item in group if item.status == "duplicate"
                ),
                recovered_work=recovered,
                stopped=stopped or infrastructure_failed,
            )
            if stopped or infrastructure_failed:
                break
            generation += 1
        flat = tuple(item for group in generations for item in group)
        if (
            stopped_reason is None
            and any(item.status == "failed" for item in flat)
            and budget_reason() == "max_model_turns"
        ):
            stopped_reason = "max_model_turns"
        status = (
            "infrastructure_failed"
            if infrastructure_failed
            else "stopped"
            if stopped_reason == "operator_stop"
            else "budget_exhausted"
            if stopped_reason == "max_model_turns"
            else "budget_exhausted"
            if stopped
            else "completed"
        )
        summary = {
            "status": status,
            "generation_count": len(generations),
            "completed_generation_count": int(state.get("next_generation", first_generation)),
            "first_generation": first_generation,
            "generation_limit": generation_limit,
            "slots_per_generation": self.config.slots,
            "initial_turn_count": live_turns - repairs,
            "repair_turn_count": repairs,
            "total_live_turns": live_turns,
            "cumulative_model_turns": turns,
            "remaining_model_turns": (
                max(0, self.config.max_model_turns - turns)
                if self.config.max_model_turns is not None
                else None
            ),
            "recovered_turn_count": recovered,
            "unique_count": len(unique),
            "max_model_turns": self.config.max_model_turns,
            "stop_reason": (
                stopped_reason
                or ("infrastructure_failed" if infrastructure_failed else "generation_limit")
            ),
            "checkpoint": str(self._checkpoint_file) if self._checkpoint_file else None,
        }
        state["summary"] = summary
        self._save(state)
        return GenerationResult(status, tuple(generations), flat, tuple(unique), summary)


NativeGenerationCoordinator = GenerationCoordinator
GenerationOrchestrator = GenerationCoordinator
EvolutionCoordinator = GenerationCoordinator


def generate(provider: GenerationProvider, **kwargs: Any) -> GenerationResult:
    return GenerationCoordinator(provider, **kwargs).run()


generate_wave = generate


__all__ = [
    "Candidate",
    "EvolutionCoordinator",
    "GenerationConfig",
    "GenerationCoordinator",
    "NativeGenerationCoordinator",
    "GenerationOrchestrator",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTurn",
    "ProviderResult",
    "RawProviderResult",
    "Request",
    "SlotResult",
    "can_retry_infrastructure",
    "generate",
    "generate_wave",
    "infrastructure_retry_allowed",
    "make_idempotency_key",
    "request_idempotency_key",
]
