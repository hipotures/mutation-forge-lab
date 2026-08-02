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
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, cast

from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.validation import ValidationResult, validate_policy


@dataclass(frozen=True, slots=True)
class _GeneratedPolicy:
    source: str


class _GeneratedPolicyError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        self.code = code
        super().__init__(message)


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
        return value
    except Exception:
        return repr(value)


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
    error: str | None = None

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
                error=str(value.get("error")) if value.get("error") is not None else None,
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
                error=getattr(value, "error", None),
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
            "error": self.error,
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
) -> str:
    return _hash(
        {
            "campaign": campaign,
            "generation": generation,
            "slot": slot,
            "parent": parent,
            "brief": brief,
            "prompt_hash": prompt_hash,
            "phase": phase,
        }
    )


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
        }


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
    generations: int = 1
    slots: int = 8
    population_size: int | None = None
    max_workers: int = 8
    concurrency: int | None = None
    max_model_turns: int | None = None
    max_repairs: int = 1
    model: str = "gpt-5.6-luna"
    effort: str = "high"
    system_prompt: str = "Return one generated policy JSON object."
    output_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    repair_prompt: str = "Repair the generated policy using the diagnostics."
    sandbox_limits: SandboxLimits = field(default_factory=SandboxLimits)
    max_repair_diagnostics: int = 8
    checkpoint_path: Path | None = None
    require_usage: bool = False

    def __post_init__(self) -> None:
        population = self.population_size if self.population_size is not None else self.slots
        if isinstance(population, bool) or not isinstance(population, int) or population <= 0:
            raise ValueError("population_size/slots must be a positive integer")
        if (
            isinstance(self.generations, bool)
            or not isinstance(self.generations, int)
            or self.generations <= 0
        ):
            raise ValueError("generations must be a positive integer")
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
        object.__setattr__(self, "slots", population)
        object.__setattr__(self, "max_workers", workers)


_REPAIRABLE = frozenset(
    {
        "structured_output",
        "invalid_json",
        "invalid_output",
        "invalid_keys",
        "invalid_schema_version",
        "invalid_array",
        "duplicate_value",
        "unknown_field",
        "invalid_string",
        "syntax_error",
        "forbidden_syntax",
        "wrong_signature",
        "wrong_function_name",
        "return_contract",
        "static_loop_bound",
        "loop_bound",
        "non_finite_literal",
        "string_too_large",
        "top_level_contract",
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
        parent_sources: Mapping[str, str] | None = None,
        parent_records: Mapping[str, Any] | None = None,
        search_feedback: Any = "",
        archive_context: Any = "",
        retry_infrastructure: bool = False,
        behavior_evaluator: Any = None,
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
        self.parent_selector, self.selection_callback = parent_selector, selection_callback
        self.parent_sources, self.parent_records = (
            dict(parent_sources or {}),
            dict(parent_records or {}),
        )
        self.search_feedback, self.archive_context = search_feedback, archive_context
        self.retry_infrastructure, self.behavior_evaluator = (
            retry_infrastructure,
            behavior_evaluator,
        )
        self._checkpoint_file = self.config.checkpoint_path

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(f"slot-{index:02d}" for index in range(self.config.slots))

    def _context(self, value: Any, generation: int, slot: str) -> str:
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
    ) -> GenerationRequest:
        brief = self._brief(generation, slot)
        if self.prompt_renderer is not None:
            try:
                rendered = self.prompt_renderer(
                    brief=brief, parent_id=parent, generation=generation, slot=slot
                )
            except TypeError:
                rendered = self.prompt_renderer(brief)
        else:
            rendered = brief
        parent_source, parent_metadata = self._parent_source_metadata(parent)
        feedback, archive_context = (
            self._context(self.search_feedback, generation, slot),
            self._context(self.archive_context, generation, slot),
        )
        payload: dict[str, Any]
        if phase == "repair":
            payload = {"source": repair_source, "diagnostics": [dict(item) for item in diagnostics]}
        else:
            payload = {
                "brief": rendered,
                "parent_id": parent,
                "parent_source": parent_source,
                "parent_metadata": parent_metadata,
                "search_feedback": feedback,
                "archive_context": archive_context,
            }
        prompt = json.dumps(_safe(payload), sort_keys=True, separators=(",", ":"))
        prompt_hash, brief_id = _hash(prompt), _hash(brief)
        key = request_idempotency_key(
            self.config.campaign_id, generation, slot, parent, brief_id, prompt_hash, phase
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
        )

    def _invoke(self, request: GenerationRequest) -> ProviderResult:
        payload = request.as_dict()
        try:
            if request.phase == "repair" and callable(getattr(self.provider, "repair", None)):
                value = self.provider.repair(payload, tuple(request.diagnostics))  # type: ignore[attr-defined]
            else:
                value = self.provider.generate(payload)
            return ProviderResult.from_value(value)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            evidence = getattr(exc, "evidence", {})
            return ProviderResult.from_value(
                {
                    "status": "infrastructure",
                    "accepted": False,
                    "charged": False,
                    "content": False,
                    "uncharged": False,
                    **(dict(evidence) if isinstance(evidence, Mapping) else {}),
                    "error": str(exc),
                }
            )

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
            validation = validate_policy(source, self.config.sandbox_limits)
            if not validation.valid:
                errors.extend(
                    {**item.as_dict(), "repair_class": "ast"} for item in validation.errors
                )
            elif callable(self.behavior_evaluator):
                try:
                    evaluated = self.behavior_evaluator(source, self.config.sandbox_limits)
                    behavior = evaluated[0] if isinstance(evaluated, tuple) else evaluated
                except Exception as exc:
                    errors.append({"code": "behavior_error", "message": str(exc)[:256]})
        diagnostics = tuple(
            {"code": str(item.get("code", "")), "message": str(item.get("message", ""))[:256]}
            for item in errors
            if str(item.get("code", "")) in _REPAIRABLE or item.get("repair_class") == "ast"
        )[: self.config.max_repair_diagnostics]
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
            if isinstance(value, Mapping):
                return dict(value)
        if self._checkpoint_file is not None and self._checkpoint_file.exists():
            try:
                value = json.loads(self._checkpoint_file.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, ValueError, TypeError):
                pass
        return {
            "schema_version": "mforge.experiment.generation.v1",
            "campaign_id": self.config.campaign_id,
            "slots": {},
            "callbacks": {},
        }

    def _save(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(payload)
        if self.checkpoint_store is not None:
            with suppress(Exception):
                self.checkpoint_store.save(
                    {
                        "generation": payload.get("generation", 0),
                        "slots": payload.get("slots", {}),
                        "summary": payload.get("summary", {}),
                    }
                )
        if self._checkpoint_file is None:
            return
        self._checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".generation-", suffix=".tmp", dir=self._checkpoint_file.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_safe(payload), handle, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._checkpoint_file)
        finally:
            Path(temporary).unlink(missing_ok=True)

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
        )

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
            request.generation,
            request.slot,
            request.parent_id,
            "accepted" if candidate else "failed",
            candidate,
            tuple(diagnostics if not candidate else ()),
            0,
            raw.as_dict(),
            None,
            request.as_dict(),
            raw.as_dict(),
        )
        if not (allow_repair and candidate is None and diagnostics and self._repair_allowed(raw)):
            return result
        repair_request = self.build_request(
            request.generation,
            request.slot,
            request.parent_id,
            phase="repair",
            diagnostics=diagnostics,
            repair_source=self._invalid_source(raw.as_dict()),
        )
        repair_raw = self._invoke(repair_request)
        repaired, repair_diagnostics = self._assess(repair_request, repair_raw, repair=True)
        return SlotResult(
            request.generation,
            request.slot,
            request.parent_id,
            "accepted" if repaired else "failed",
            repaired,
            tuple(repair_diagnostics if not repaired else ()),
            1,
            raw.as_dict(),
            repair_raw.as_dict(),
            repair_request.as_dict(),
            repair_raw.as_dict(),
        )

    def run(self, *, resume: bool = True) -> GenerationResult:
        state: dict[str, Any] = (
            self._load()
            if resume
            else {
                "schema_version": "mforge.experiment.generation.v1",
                "campaign_id": self.config.campaign_id,
                "slots": {},
                "callbacks": {},
            }
        )
        slots_state = cast(dict[str, Any], state.setdefault("slots", {}))
        callbacks = cast(dict[str, Any], state.setdefault("callbacks", {}))
        seen: dict[str, str] = {}
        for index, source in enumerate(self.existing_sources):
            identity = validate_policy(source, self.config.sandbox_limits).identity
            if identity.normalized_ast_sha256:
                seen[identity.normalized_ast_sha256] = f"existing-{index}"
        generations: list[tuple[SlotResult, ...]] = []
        unique: list[Candidate] = []
        turns = 0
        repairs = 0
        recovered = 0
        stopped = False
        for generation in range(self.config.generations):
            parents = self._parents(generation)
            results: dict[str, SlotResult] = {}
            futures: dict[Any, tuple[str, GenerationRequest]] = {}
            with ThreadPoolExecutor(
                max_workers=self.config.max_workers, thread_name_prefix=f"native-g{generation}"
            ) as pool:
                for slot in self.slots:
                    request = self.build_request(generation, slot, parents[slot])
                    cached = slots_state.get(request.idempotency_key)
                    if isinstance(cached, Mapping) and cached.get("status") not in {
                        "pending",
                        "budget_exhausted",
                    }:
                        results[slot] = self._from_slot(cached)
                        recovered += 1
                        continue
                    if (
                        self.config.max_model_turns is not None
                        and turns >= self.config.max_model_turns
                    ):
                        results[slot] = SlotResult(
                            generation,
                            slot,
                            parents[slot],
                            "budget_exhausted",
                            request=request.as_dict(),
                        )
                        stopped = True
                        continue
                    futures[pool.submit(self._invoke, request)] = (slot, request)
                    turns += 1
                for future in as_completed(futures):
                    slot, request = futures[future]
                    raw = future.result()
                    if (
                        self.retry_infrastructure
                        and infrastructure_retry_allowed(raw)
                        and (
                            self.config.max_model_turns is None
                            or turns < self.config.max_model_turns
                        )
                    ):
                        raw = self._invoke(request)
                        turns += 1
                    candidate, diagnostics = self._assess(request, raw)
                    can_repair = self._repair_allowed(raw)
                    results[slot] = SlotResult(
                        generation,
                        slot,
                        request.parent_id,
                        "accepted"
                        if candidate
                        else ("repair_pending" if diagnostics and can_repair else "failed"),
                        candidate,
                        tuple(diagnostics if not candidate else ()),
                        0,
                        raw.as_dict(),
                        None,
                        request.as_dict(),
                        raw.as_dict(),
                    )
                    slots_state[request.idempotency_key] = results[slot].as_dict()
                    self._save(state)
            for slot in self.slots:
                item = results.get(slot)
                repair_attempt = 0
                if (
                    item is not None
                    and item.status == "repair_pending"
                    and self.config.max_model_turns is not None
                    and turns >= self.config.max_model_turns
                ):
                    stopped = True
                while (
                    item is not None
                    and item.status == "repair_pending"
                    and item.errors
                    and repair_attempt < self.config.max_repairs
                    and (self.config.max_model_turns is None or turns < self.config.max_model_turns)
                ):
                    req = self.build_request(
                        generation,
                        slot,
                        item.parent_id,
                        phase="repair",
                        diagnostics=item.errors,
                        repair_source=self._invalid_source(item.raw_result),
                    )
                    cached = slots_state.get(req.idempotency_key)
                    if isinstance(cached, Mapping) and cached.get("status") not in {
                        "pending",
                        "repair_pending",
                    }:
                        repaired = self._from_slot(cached)
                        recovered += 1
                    else:
                        raw = self._invoke(req)
                        turns += 1
                        repairs += 1
                        candidate, diagnostics = self._assess(req, raw, repair=True)
                        can_repair = self._repair_allowed(raw)
                        repaired = SlotResult(
                            generation,
                            slot,
                            item.parent_id,
                            "accepted"
                            if candidate
                            else ("repair_pending" if diagnostics and can_repair else "failed"),
                            candidate,
                            tuple(diagnostics if not candidate else ()),
                            item.repairs + 1,
                            item.initial,
                            raw.as_dict(),
                            req.as_dict(),
                            raw.as_dict(),
                        )
                        slots_state[req.idempotency_key] = repaired.as_dict()
                        slots_state[str(item.request.get("idempotency_key", ""))] = (
                            repaired.as_dict()
                        )
                        self._save(state)
                    results[slot] = repaired
                    item = repaired
                    repair_attempt += 1
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
                ordered.append(item)
            generations.append(tuple(ordered))
            state["generation"] = generation
            self._save(state)
            if self.selection_callback is not None and str(generation) not in callbacks:
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
                    if generation + 1 < self.config.generations:
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
                callbacks[str(generation)] = {"status": "completed"}
                self._save(state)
            if stopped:
                break
        flat = tuple(item for group in generations for item in group)
        status = "budget_exhausted" if stopped else "completed"
        summary = {
            "status": status,
            "generation_count": len(generations),
            "slots_per_generation": self.config.slots,
            "initial_turn_count": turns - repairs,
            "repair_turn_count": repairs,
            "total_live_turns": turns,
            "recovered_turn_count": recovered,
            "unique_count": len(unique),
            "max_model_turns": self.config.max_model_turns,
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
