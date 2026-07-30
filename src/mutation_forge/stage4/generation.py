"""Deterministic four-generation mutation waves.

The module deliberately knows nothing about a model transport.  A provider is a
small object with ``generate(request)`` (and, optionally, ``repair``); this makes
the campaign testable with a fake and leaves App Server adaptation to the command
layer.  Checkpoints contain the complete provider envelope, so a process restart
never spends a completed initial or repair turn twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, cast

from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.validation import ValidationResult, validate_policy

from .checkpoint import Checkpoint, CheckpointStore
from .contracts import GeneratedPolicy, parse_generated_policy
from .prompts import render_repair_prompt, render_request_prompt

SLOTS: tuple[str, ...] = tuple(f"slot-{i:02d}" for i in range(8))
GENERATIONS = 4
SMOKE_CALLS = 10_000


class GenerationProvider(Protocol):
    def generate(self, request: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Transport-independent raw turn envelope.

    ``accepted``, ``charged`` and ``content`` are explicit because infrastructure
    retry safety must never be inferred from an exception message.
    """

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
        # Stage 3's immutable ``Turn`` is accepted as a transport adapter too;
        # importing it here would couple Stage 4 to that transport module.
        if all(hasattr(value, name) for name in ("response", "status", "usage")):
            return cls(
                response=value.response,
                status=str(value.status),
                accepted=bool(getattr(value, "accepted", True)),
                charged=bool(getattr(value, "charged", True)),
                content=bool(getattr(value, "content", value.response is not None)),
                uncharged=bool(
                    getattr(value, "uncharged", getattr(value, "app_server_uncharged", False))
                ),
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
                error=getattr(value, "error", None),
            )
        if isinstance(value, Mapping):
            return cls(
                response=value.get("response", value.get("output")),
                status=str(value.get("status", "completed")),
                accepted=bool(value.get("accepted", True)),
                charged=bool(value.get("charged", True)),
                content=bool(value.get("content", value.get("response") is not None)),
                uncharged=bool(
                    value.get("uncharged", value.get("app_server_uncharged", False))
                ),
                unauthorized_tool_approval=bool(
                    value.get("unauthorized_tool_approval", False)
                ),
                usage=cast(Mapping[str, Any], value.get("usage", {}))
                if isinstance(value.get("usage", {}), Mapping)
                else {},
                request_id=value.get("request_id"),
                thread_id=value.get("thread_id"),
                turn_id=value.get("turn_id"),
                session_id=value.get("session_id"),
                provider_request_id=value.get("provider_request_id"),
                provider_thread_id=value.get("provider_thread_id"),
                provider_turn_id=value.get("provider_turn_id"),
                error=str(value.get("error")) if value.get("error") is not None else None,
            )
        # A provider may return a bare generated envelope in offline tests.
        return cls(response=value, usage={"totalTokens": 0}, charged=False)

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
    appserver_doctor_sha256: str | None = None

    @property
    def request_idempotency_key(self) -> str:
        return self.idempotency_key

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
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
            "appserver_doctor_sha256": self.appserver_doctor_sha256,
        }
        if self.phase == "repair":
            value["diagnostics"] = [dict(item) for item in self.diagnostics]
        return value


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
            "errors": [_safe(dict(e)) for e in self.errors],
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
    campaign_id: str = "stage4"
    generations: int = GENERATIONS
    slots: int = 8
    max_workers: int = 8
    smoke_calls: int = SMOKE_CALLS
    max_repair_diagnostics: int = 8
    sandbox_limits: SandboxLimits = field(default_factory=SandboxLimits)
    checkpoint_path: Path | None = None
    model: str = "gpt-5.6-luna"
    effort: str = "high"
    appserver_doctor_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.generations != GENERATIONS or self.slots != 8 or self.max_workers != 8:
            raise ValueError(
                "Stage 4 is frozen at four generations of eight slots with eight workers"
            )
        if self.smoke_calls != SMOKE_CALLS:
            raise ValueError("Stage 4 requires exactly 10,000 smoke calls")
        if self.model != "gpt-5.6-luna" or self.effort != "high":
            raise ValueError("Stage 4 requires gpt-5.6-luna/high")
        if self.appserver_doctor_sha256 is not None and len(self.appserver_doctor_sha256) != 64:
            raise ValueError("Stage 4 App Server doctor SHA-256 is invalid")


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


def _usage_complete(usage: Mapping[str, Any]) -> bool:
    required = (
        "inputTokens",
        "cachedInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    return (
        usage.get("final") is True
        and usage.get("partial") is False
        and all(
            isinstance(usage.get(name), int)
            and not isinstance(usage.get(name), bool)
            and int(usage[name]) >= 0
            for name in required
        )
    )


def infrastructure_retry_allowed(
    result: ProviderResult | Mapping[str, Any] | BaseException,
) -> bool:
    """Return true only for a proven pre-turn transport failure."""
    if isinstance(result, BaseException):
        return False
    value = ProviderResult.from_value(result)
    usage = value.usage
    total = sum(
        int(v)
        for k, v in usage.items()
        if k.lower().endswith("tokens") and isinstance(v, int) and not isinstance(v, bool)
    )
    return (
        value.status.lower() in {"infrastructure", "transport_error", "unavailable", "retryable"}
        and not value.accepted
        and not value.charged
        and not value.content
        and value.uncharged
        and total == 0
        and value.response in (None, "", {}, [])
    )


can_retry_infrastructure = infrastructure_retry_allowed


def _behavior(
    source: str, limits: SandboxLimits, smoke_calls: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from mutation_forge.stage3.generation import _behavior as stage3_behavior

    return stage3_behavior(source, limits, smoke_calls)


_REPAIRABLE = frozenset(
    {
        "invalid_json",
        "invalid_output",
        "invalid_keys",
        "structured_output",
        "invalid_schema_version",
        "invalid_array",
        "duplicate_value",
        "unknown_field",
        "response_too_large",
        "syntax_error",
        "forbidden_syntax",
        "wrong_signature",
        "return_contract",
        "static_loop_bound",
        "loop_bound",
        "non_finite_literal",
        "string_too_large",
        "invalid_string",
    }
)


def _render(brief: Any, parent: str, generation: int, slot: str) -> tuple[str, str]:
    rendered = brief(parent=parent, generation=generation, slot=slot) if callable(brief) else brief
    if isinstance(rendered, Mapping):
        # Renderer output is intentionally the sole initial prompt payload.
        rendered = json.dumps(_safe(dict(rendered)), sort_keys=True, separators=(",", ":"))
    text = str(rendered)
    return text, _hash(text)


class GenerationCoordinator:
    """Run exactly four ordered mutation waves with resumable slot checkpoints."""

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
        existing_sources: Sequence[str] = (),
        archive: Any = None,
        parent_selector: Any = None,
        retry_infrastructure: bool = False,
        parent_sources: Mapping[str, str] | None = None,
        parent_records: Mapping[str, Any] | None = None,
        search_feedback: Any = "",
        archive_context: Any = "",
        checkpoint_store: CheckpointStore | None = None,
        generation_completed: Callable[[int, tuple[SlotResult, ...], tuple[Candidate, ...]], Any]
        | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or GenerationConfig(
            campaign_id=campaign_id or "stage4",
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        )
        if campaign_id is not None:
            self.config = replace(self.config, campaign_id=campaign_id)
        self.parent_assignments, self.briefs = parent_assignments, briefs
        self.prompt_renderer, self.existing_sources, self.archive = (
            prompt_renderer,
            tuple(existing_sources),
            archive,
        )
        self.parent_selector, self.retry_infrastructure = parent_selector, retry_infrastructure
        self.parent_sources = dict(parent_sources or {})
        self.parent_records = dict(parent_records or {})
        self.search_feedback, self.archive_context = search_feedback, archive_context
        self.checkpoint_store = checkpoint_store
        self.generation_completed = generation_completed
        self._lock = threading.Lock()
        checkpoint_value = checkpoint_path or self.config.checkpoint_path
        self._checkpoint_file = Path(checkpoint_value) if checkpoint_value is not None else None

    def _parent_source_metadata(self, parent_id: str) -> tuple[str, Mapping[str, Any]]:
        source = self.parent_sources.get(parent_id, "")
        record = self.parent_records.get(parent_id)
        if record is not None:
            if isinstance(record, Mapping):
                source = str(record.get("source", source))
                return source, dict(record)
            source = str(getattr(record, "source", source))
            as_dict = getattr(record, "as_dict", None)
            metadata = as_dict() if callable(as_dict) else {}
            return source, metadata if isinstance(metadata, Mapping) else {}
        return source, {}

    @staticmethod
    def _context_value(value: Any, generation: int, slot: str) -> str:
        if isinstance(value, Mapping):
            selected = value.get(generation, value.get(str(generation), value))
            if isinstance(selected, Mapping):
                selected = selected.get(slot, selected.get(str(slot), ""))
            return str(selected)
        return str(value or "")

    def _parents(self, generation: int) -> dict[str, str]:
        raw = self.parent_assignments
        if raw is None and self.parent_selector is not None:
            raw = self.parent_selector(generation)
        if isinstance(raw, Mapping):
            value = (
                raw.get(generation, raw.get(str(generation), raw))
                if generation in raw or str(generation) in raw
                else raw
            )
            if isinstance(value, Mapping):
                return {
                    slot: str(value.get(slot, value.get(str(i), "parent-" + slot)))
                    for i, slot in enumerate(SLOTS)
                }
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return {slot: str(value[i]) for i, slot in enumerate(SLOTS)}
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            # Either one eight-parent vector or one vector per generation.
            value = (
                raw[generation] if len(raw) == 4 and isinstance(raw[generation], Sequence) else raw
            )
            return {slot: str(value[i]) for i, slot in enumerate(SLOTS)}
        return {slot: f"parent-{generation}-{slot}" for slot in SLOTS}

    def _brief(self, generation: int, slot: str) -> Any:
        raw = self.briefs
        if raw is None:
            return f"mutation brief generation {generation} slot {slot}"
        if isinstance(raw, Mapping):
            value = raw.get(generation, raw.get(str(generation), raw))
            if isinstance(value, Mapping):
                return value.get(slot, value.get(str(slot), ""))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return value[int(slot[-2:])]
            return value
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) == 4 and isinstance(raw[generation], Sequence):
                return raw[generation][int(slot[-2:])]
            return raw[int(slot[-2:])]
        return raw

    def _load(self) -> dict[str, Any]:
        if self._checkpoint_file is None or not self._checkpoint_file.exists():
            return {
                "schema_version": "stage4.checkpoint.v1",
                "campaign_id": self.config.campaign_id,
                "slots": {},
            }
        try:
            value = json.loads(self._checkpoint_file.read_text(encoding="utf-8"))
            return (
                value
                if isinstance(value, dict)
                else {
                    "schema_version": "stage4.checkpoint.v1",
                    "campaign_id": self.config.campaign_id,
                    "slots": {},
                }
            )
        except (OSError, ValueError, TypeError):
            return {
                "schema_version": "stage4.checkpoint.v1",
                "campaign_id": self.config.campaign_id,
                "slots": {},
            }

    def _save(self, state: dict[str, Any]) -> None:
        if self._checkpoint_file is None:
            return
        self._checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._checkpoint_file.with_suffix(self._checkpoint_file.suffix + ".tmp")
        tmp.write_text(json.dumps(_safe(state), sort_keys=True, indent=2), encoding="utf-8")
        tmp.replace(self._checkpoint_file)

    def _emit_checkpoint(
        self,
        generation: int,
        parents: Mapping[str, str],
        state: Mapping[str, Any],
        results: Mapping[str, SlotResult],
    ) -> None:
        if self.checkpoint_store is None:
            return
        slots = tuple(
            slot for slot in SLOTS if slot in results and results[slot].status != "pending"
        )
        keys = tuple(sorted(cast(Mapping[str, Any], state.get("slots", {}))))
        usage: dict[str, Any] = {}
        for result in results.values():
            raw_usage = result.raw_result.get("usage", {})
            if isinstance(raw_usage, Mapping):
                for name, value in raw_usage.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        usage[name] = int(usage.get(name, 0)) + value
        self.checkpoint_store.save(
            Checkpoint(
                generation=generation,
                completed_slots=slots,
                pending_slots=tuple(slot for slot in SLOTS if slot not in slots),
                parent_assignments=dict(parents),
                request_idempotency_keys=keys,
                turn_count=len(keys),
                usage=usage,
                token_count=int(usage.get("totalTokens", 0)),
            )
        )

    def _request(
        self,
        generation: int,
        slot: str,
        parent: str,
        phase: str,
        diagnostics: Sequence[Mapping[str, Any]] = (),
        repair_source: str = "",
    ) -> GenerationRequest:
        brief = self._brief(generation, slot)
        rendered = brief
        if self.prompt_renderer is not None:
            try:
                rendered = self.prompt_renderer(
                    brief=brief, parent_id=parent, generation=generation, slot=slot
                )
            except TypeError:
                rendered = self.prompt_renderer(brief)
        parent_source, parent_metadata = self._parent_source_metadata(parent)
        feedback = self._context_value(self.search_feedback, generation, slot)
        archive_context = self._context_value(self.archive_context, generation, slot)
        if phase == "initial":
            prompt = render_request_prompt(
                slot_id=slot,
                brief=str(rendered),
                parent_source=parent_source,
                parent_metadata=dict(parent_metadata),
                search_feedback=feedback,
                archive_context=archive_context,
            )
            phash = _hash(prompt)
        else:
            prompt = render_repair_prompt(
                diagnostics={"schema": list(diagnostics), "ast": list(diagnostics)},
                source=repair_source,
            )
            phash = _hash(prompt)
        brief_id = _hash(brief)
        key = request_idempotency_key(
            self.config.campaign_id, generation, slot, parent, brief_id, phash, phase
        )
        return GenerationRequest(
            self.config.campaign_id,
            generation,
            slot,
            parent,
            brief_id,
            prompt,
            phash,
            key,
            phase,
            tuple(diagnostics),
            parent_source,
            parent_metadata,
            feedback,
            archive_context,
            self.config.model,
            self.config.effort,
            self.config.appserver_doctor_sha256,
        )

    def _invoke(self, request: GenerationRequest) -> ProviderResult:
        payload = request.as_dict()
        if request.phase == "repair" and callable(getattr(self.provider, "repair", None)):
            value = self.provider.repair(payload, tuple(request.diagnostics))  # type: ignore[attr-defined]
        else:
            value = self.provider.generate(payload)
        return ProviderResult.from_value(value)

    def _assess(
        self, req: GenerationRequest, raw: ProviderResult, *, repair: bool = False
    ) -> tuple[Candidate | None, tuple[Mapping[str, Any], ...]]:
        errors: list[Mapping[str, Any]] = []
        if raw.status != "completed":
            errors.append({"code": "provider_status", "message": raw.status})
        elif not _usage_complete(raw.usage):
            errors.append(
                {"code": "usage_missing", "message": "completed turn omitted exact final usage"}
            )
        if not raw.accepted or not raw.content:
            errors.append(
                {"code": "turn_provenance", "message": "turn was not accepted/contentful"}
            )
        response_value = raw.response
        if isinstance(response_value, (str, bytes, bytearray)):
            try:
                response_value = json.loads(
                    response_value.decode("utf-8")
                    if isinstance(response_value, (bytes, bytearray))
                    else response_value
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append({"code": "invalid_json", "message": str(exc)[:256]})
        try:
            value: GeneratedPolicy = parse_generated_policy(
                response_value,
                validate_source=False,
            )
            source = value.source
        except Exception as exc:
            errors.append(
                {"code": getattr(exc, "code", "structured_output"), "message": str(exc)[:256]}
            )
            source = None
        validation: ValidationResult | None = None
        behavior: Mapping[str, Any] = {}
        telemetry: Mapping[str, Any] = {}
        if source is not None:
            validation = validate_policy(source, self.config.sandbox_limits)
            if not validation.valid:
                errors.extend(
                    {**error.as_dict(), "repair_class": "ast"} for error in validation.errors
                )
            else:
                try:
                    behavior, telemetry = _behavior(
                        source, self.config.sandbox_limits, self.config.smoke_calls
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "code": str(exc)
                            if str(exc)
                            in {
                                "input_mutation",
                                "finite_probe",
                                "runtime_exception",
                                "worker_timeout",
                                "worker_crash",
                                "worker_protocol",
                            }
                            else "finite_probe",
                            "message": str(exc)[:256],
                        }
                    )
        diagnostics = tuple(
            {"code": str(item.get("code", "")), "message": str(item.get("message", ""))[:256]}
            for item in errors
            if str(item.get("code", "")) in _REPAIRABLE or item.get("repair_class") == "ast"
        )
        if errors or validation is None or not validation.valid or source is None:
            return None, diagnostics
        identity = validation.identity
        return Candidate(
            source,
            identity.source_sha256,
            identity.normalized_ast_sha256 or "",
            req.generation,
            req.slot,
            req.parent_id,
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
    def _invalid_source(raw_result: Mapping[str, Any]) -> str:
        response = raw_result.get("response")
        if isinstance(response, Mapping):
            source = response.get("source")
            return (
                source
                if isinstance(source, str)
                else json.dumps(_safe(dict(response)), sort_keys=True)
            )
        if isinstance(response, (str, bytes, bytearray)):
            return (
                response.decode("utf-8", "replace")
                if isinstance(response, (bytes, bytearray))
                else response
            )
        return ""

    @staticmethod
    def _write_archive_source(archive: Any, program_id: str, source: str) -> str | None:
        root = getattr(archive, "root", None)
        if root is None:
            return None
        root = Path(root)
        relative = Path("sources") / f"{program_id}.py"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".source-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(source)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return relative.as_posix()

    def _slot_from_checkpoint(self, value: Mapping[str, Any]) -> SlotResult:
        candidate_raw = value.get("candidate")
        candidate = None
        if isinstance(candidate_raw, Mapping):
            candidate = Candidate(
                source=str(candidate_raw.get("source", "")),
                source_sha256=str(candidate_raw.get("source_sha256", "")),
                normalized_ast_sha256=str(candidate_raw.get("normalized_ast_sha256", "")),
                generation=int(candidate_raw.get("generation", value.get("generation", 0))),
                slot=str(candidate_raw.get("slot", value.get("slot", ""))),
                parent_id=str(candidate_raw.get("parent_id", value.get("parent_id", ""))),
                behavior_signature=cast(
                    Mapping[str, Any], candidate_raw.get("behavior_signature", {})
                ),
                usage=cast(Mapping[str, Any], candidate_raw.get("usage", {})),
                request_id=candidate_raw.get("request_id"),
                thread_id=candidate_raw.get("thread_id"),
                turn_id=candidate_raw.get("turn_id"),
                repair=bool(candidate_raw.get("repair", False)),
                duplicate_of=candidate_raw.get("duplicate_of"),
                source_identity=cast(Mapping[str, Any], candidate_raw.get("source_identity", {})),
            )
        return SlotResult(
            int(value.get("generation", 0)),
            str(value.get("slot", "")),
            str(value.get("parent_id", "")),
            str(value.get("status", "failed")),
            candidate,
            tuple(
                cast(Mapping[str, Any], x)
                for x in value.get("errors", [])
                if isinstance(x, Mapping)
            ),
            int(value.get("repairs", 0)),
            cast(Mapping[str, Any], value.get("initial", {})),
            cast(Mapping[str, Any] | None, value.get("repair")),
            cast(Mapping[str, Any], value.get("request", {})),
            cast(Mapping[str, Any], value.get("raw_result", {})),
            value.get("duplicate_of"),
        )

    def run(self, *, resume: bool = True) -> GenerationResult:
        state = (
            self._load()
            if resume
            else {
                "schema_version": "stage4.checkpoint.v1",
                "campaign_id": self.config.campaign_id,
                "slots": {},
            }
        )
        state.setdefault("slots", {})
        callbacks = cast(dict[str, Any], state.setdefault("callbacks", {}))
        slots_state = cast(dict[str, Any], state["slots"])
        all_generations: list[tuple[SlotResult, ...]] = []
        seen: dict[str, str] = {}
        # Existing archive/seed sources participate in duplicate detection but
        # never consume a live turn.
        for index, source in enumerate(self.existing_sources):
            try:
                identity = validate_policy(source, self.config.sandbox_limits).identity
                if identity.normalized_ast_sha256:
                    seen[identity.normalized_ast_sha256] = f"existing-{index}"
            except Exception:
                continue
        unique: list[Candidate] = []
        initial_calls = repair_calls = 0
        for generation in range(GENERATIONS):
            parents = self._parents(generation)  # fixed before worker submission
            results: dict[str, SlotResult] = {}
            futures: dict[Any, tuple[str, GenerationRequest]] = {}
            with ThreadPoolExecutor(
                max_workers=8, thread_name_prefix=f"stage4-g{generation}"
            ) as pool:
                for slot in SLOTS:
                    parent = parents[slot]
                    req = self._request(generation, slot, parent, "initial")
                    key = req.idempotency_key
                    cached = slots_state.get(key)
                    if isinstance(cached, Mapping) and cached.get("status") != "pending":
                        results[slot] = self._slot_from_checkpoint(cached)
                        continue
                    futures[pool.submit(self._invoke, req)] = (slot, req)
                    initial_calls += 1
                for future in as_completed(futures):
                    slot, req = futures[future]
                    try:
                        raw = future.result()
                    except BaseException as exc:
                        evidence = getattr(exc, "evidence", {})
                        raw = ProviderResult.from_value(
                            {
                                "accepted": False,
                                "charged": False,
                                "content": False,
                                "uncharged": False,
                                **(dict(evidence) if isinstance(evidence, Mapping) else {}),
                                "status": "infrastructure",
                                "error": str(exc),
                            }
                        )
                    if self.retry_infrastructure and infrastructure_retry_allowed(raw):
                        # This is the same idempotent request, never a slot
                        # replacement.  Retain the pre-turn evidence alongside
                        # the eventual result for accounting/audit consumers.
                        try:
                            retried = self._invoke(req)
                            raw = replace(
                                retried,
                                error=(
                                    f"infrastructure_retry:{raw.error}"
                                    if raw.error
                                    else "infrastructure_retry"
                                ),
                            )
                        except BaseException as exc:
                            evidence = getattr(exc, "evidence", {})
                            raw = ProviderResult.from_value(
                                {
                                    "accepted": False,
                                    "charged": False,
                                    "content": False,
                                    "uncharged": False,
                                    **(
                                        dict(evidence)
                                        if isinstance(evidence, Mapping)
                                        else {}
                                    ),
                                    "status": "infrastructure",
                                    "error": f"infrastructure_retry:{exc}",
                                }
                            )
                    candidate, diagnostics = self._assess(req, raw)
                    slot_result = SlotResult(
                        generation,
                        slot,
                        req.parent_id,
                        "accepted" if candidate else "failed",
                        candidate,
                        tuple(diagnostics if not candidate else ()),
                        0,
                        initial=raw.as_dict(),
                        raw_result=raw.as_dict(),
                        request=req.as_dict(),
                    )
                    if (
                        candidate is None
                        and diagnostics
                        and raw.status == "completed"
                        and raw.accepted
                        and raw.content
                        and _usage_complete(raw.usage)
                    ):
                        slot_result = replace(slot_result, status="repair_pending")
                    results[slot] = slot_result
                    slots_state[req.idempotency_key] = slot_result.as_dict()
                    self._save(state)
                    self._emit_checkpoint(generation, parents, state, results)
            # Repairs are sequentially integrated, with at most one per slot.
            for slot in SLOTS:
                result = results.get(slot)
                if result is None:
                    continue
                if result.status == "repair_pending" and result.errors:
                    req = self._request(
                        generation,
                        slot,
                        result.parent_id,
                        "repair",
                        result.errors,
                        self._invalid_source(result.raw_result),
                    )
                    cached = slots_state.get(req.idempotency_key)
                    if isinstance(cached, Mapping) and cached.get("status") not in {
                        "pending",
                        "repair_pending",
                    }:
                        repaired = self._slot_from_checkpoint(cached)
                    else:
                        repair_calls += 1
                        try:
                            raw = self._invoke(req)
                        except BaseException as exc:
                            evidence = getattr(exc, "evidence", {})
                            raw = ProviderResult.from_value(
                                {
                                    "accepted": False,
                                    "charged": False,
                                    "content": False,
                                    "uncharged": False,
                                    **(
                                        dict(evidence)
                                        if isinstance(evidence, Mapping)
                                        else {}
                                    ),
                                    "status": "infrastructure",
                                    "error": str(exc),
                                }
                            )
                        candidate, diagnostics = self._assess(req, raw, repair=True)
                        repaired = SlotResult(
                            generation,
                            slot,
                            result.parent_id,
                            "accepted" if candidate else "failed",
                            candidate,
                            tuple(diagnostics if not candidate else ()),
                            1,
                            result.initial or result.raw_result,
                            raw.as_dict(),
                            req.as_dict(),
                            raw.as_dict(),
                        )
                        slots_state[req.idempotency_key] = repaired.as_dict()
                        # The initial turn is complete even when its candidate
                        # required a repair; mark that idempotency key terminal
                        # as well so resume never repeats the initial request.
                        initial_key = str(result.request.get("idempotency_key", ""))
                        if initial_key:
                            slots_state[initial_key] = repaired.as_dict()
                        self._save(state)
                        self._emit_checkpoint(generation, parents, state, results)
                    results[slot] = repaired
            ordered: list[SlotResult] = []
            for slot in SLOTS:
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
                        seen[candidate.normalized_ast_sha256] = f"g{generation}:{slot}"
                        unique.append(candidate)
                        if self.archive is not None and hasattr(self.archive, "append"):
                            program_id = f"g{generation}-{slot}-{candidate.source_sha256[:12]}"
                            source_path = self._write_archive_source(
                                self.archive, program_id, candidate.source
                            )
                            if source_path is None:
                                source_path = f"sources/{program_id}.py"
                            self.archive.append(
                                {
                                    "program_id": program_id,
                                    "source_path": source_path,
                                    "source_sha256": candidate.source_sha256,
                                    "normalized_ast_sha256": candidate.normalized_ast_sha256,
                                    "behavior_signature": candidate.behavior_signature,
                                    "generation": generation,
                                    "slot": slot,
                                    "parent_id": candidate.parent_id,
                                    "request_id": str(candidate.request_id)
                                    if candidate.request_id is not None
                                    else None,
                                    "thread_id": candidate.thread_id,
                                    "turn_id": candidate.turn_id,
                                    "usage": dict(candidate.usage),
                                    "validation_status": "valid",
                                    "probe_status": "passed",
                                    "smoke_10k_status": "passed",
                                }
                            )
                ordered.append(item)
            all_generations.append(tuple(ordered))
            self._emit_checkpoint(generation, parents, state, results)
            if self.generation_completed is not None and str(generation) not in callbacks:
                # Callback completion is committed only after it returns.  If
                # it raises, model-turn checkpoints remain durable and resume
                # retries this callback without submitting another request.
                self.generation_completed(generation, tuple(ordered), tuple(unique))
                callbacks[str(generation)] = {"status": "completed"}
                self._save(state)
        flat = tuple(item for generation in all_generations for item in generation)
        summary = {
            "status": "completed",
            "generation_count": GENERATIONS,
            "slots_per_generation": 8,
            "initial_turn_count": initial_calls,
            "repair_turn_count": repair_calls,
            "total_live_turns": initial_calls + repair_calls,
            "accepted_live_turns": sum(
                envelope.get("accepted") is True
                for item in flat
                for envelope in (item.initial, item.repair or {})
                if isinstance(envelope, Mapping)
            ),
            "unique_count": len(unique),
            "max_live_turns": 64,
            "checkpoint": str(self._checkpoint_file) if self._checkpoint_file else None,
        }
        state["summary"] = summary
        self._save(state)
        return GenerationResult("completed", tuple(all_generations), flat, tuple(unique), summary)


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
    "GenerationOrchestrator",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResult",
    "GenerationTurn",
    "ProviderResult",
    "RawProviderResult",
    "Request",
    "SlotResult",
    "SLOTS",
    "GENERATIONS",
    "SMOKE_CALLS",
    "can_retry_infrastructure",
    "generate",
    "generate_wave",
    "infrastructure_retry_allowed",
    "make_idempotency_key",
    "request_idempotency_key",
]
