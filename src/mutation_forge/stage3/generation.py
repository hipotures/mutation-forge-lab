"""Deterministic, provider-injected one-shot Stage 3 generation.

This module deliberately models a provider as a small protocol.  Production
transport can be added later; tests use a local provider without model calls.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, cast

from mutation_forge.sandbox.contracts import RankerContext, RankerProposal, SandboxLimits
from mutation_forge.sandbox.validation import ValidationResult, validate_policy
from mutation_forge.sandbox.worker import PolicyWorker

from .artifacts import GenerationArtifacts, canonical_hash, replay_generation
from .contracts import GeneratedPolicy, parse_generated_policy

SLOTS: tuple[str, ...] = tuple(f"slot-{i:02d}" for i in range(8))
_REPAIRABLE = frozenset(
    {
        "structured_output",
        "syntax_error",
        "forbidden_syntax",
        "wrong_signature",
        "wrong_function_name",
        "return_contract",
        "static_loop_bound",
    }
)


class GenerationProvider(Protocol):
    def generate(self, request: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model: str = "injected"
    effort: str = "default"
    protocol_version: str = "stage3.generation.v1"
    smoke_calls: int = 10_000
    max_repair_diagnostics: int = 8
    allow_infrastructure_retry: bool = True


@dataclass(frozen=True, slots=True)
class Turn:
    response: Any
    accepted: bool
    charged: bool
    content: bool
    usage: Mapping[str, Any]
    status: str
    request_id: str | None = None
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    model: str | None = None
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


def parse_envelope(response: Any) -> str:
    """Strictly parse the provider envelope and return source unchanged."""
    return _parse_envelope(response).source


def _parse_envelope(response: Any) -> GeneratedPolicy:
    value: Any = response
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("response envelope must be a JSON object")
    return parse_generated_policy(value)


def _turn(value: Any) -> Turn:
    if isinstance(value, Turn):
        return value
    if isinstance(value, Mapping):
        required = {"response", "accepted", "charged", "content", "usage", "status"}
        if not required.issubset(value):
            raise ValueError(f"provider turn is missing {sorted(required.difference(value))}")
        if any(not isinstance(value[name], bool) for name in ("accepted", "charged", "content")):
            raise ValueError("provider billing flags must be booleans")
        usage = value["usage"]
        if not isinstance(usage, Mapping):
            raise ValueError("provider usage must be an object")
        return Turn(
            response=value["response"],
            accepted=cast(bool, value["accepted"]),
            charged=cast(bool, value["charged"]),
            content=cast(bool, value["content"]),
            usage=cast(Mapping[str, Any], usage),
            status=str(value["status"]),
            request_id=value.get("request_id"),
            thread_id=value.get("thread_id"),
            session_id=value.get("session_id"),
            turn_id=value.get("turn_id"),
            model=value.get("model"),
            error=value.get("error"),
        )
    raise ValueError("provider must return an explicit Turn envelope")


def _usage_complete(usage: Mapping[str, Any]) -> bool:
    required = {
        "inputTokens",
        "cachedInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    }
    if not required.issubset(usage):
        return False
    return all(
        isinstance(usage[key], int)
        and not isinstance(usage[key], bool)
        and cast(int, usage[key]) >= 0
        for key in required
        | ({"cacheWriteInputTokens"} if "cacheWriteInputTokens" in usage else set())
    )


def _call_provider(
    provider: Any,
    request: Mapping[str, Any],
    *,
    repair: bool = False,
    diagnostics: Sequence[Mapping[str, Any]] = (),
) -> Turn:
    if repair and hasattr(provider, "repair"):
        return _turn(provider.repair(request, tuple(diagnostics)))
    return _turn(
        provider.generate(
            {
                **request,
                "repair": repair,
                "diagnostics": list(diagnostics),
            }
        )
    )


def _probe_inputs() -> tuple[RankerContext, tuple[RankerProposal, ...]]:
    context: RankerContext = {
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

    def proposal(index: int, *, risk: int, broken: int, k: int) -> RankerProposal:
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

    return context, (
        proposal(1, risk=0, broken=2, k=2),
        proposal(2, risk=1, broken=1, k=3),
        proposal(3, risk=2, broken=0, k=4),
    )


def _behavior(
    source: str,
    limits: SandboxLimits,
    smoke_calls: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    validation = validate_policy(source, limits)
    if not validation.valid:
        raise ValueError("static validation failed")
    ctx, proposals = _probe_inputs()
    worker = PolicyWorker(source, limits)
    try:
        probe_records: list[dict[str, Any]] = []
        for proposal in proposals:
            result = worker.call(ctx, proposal)
            if result.status != "ok" or result.priority is None:
                raise ValueError("finite_probe")
            probe_records.append(
                {
                    "proposal_id": proposal["proposal_id"],
                    "priority": result.priority,
                    "finite": True,
                }
            )
        ranked = sorted(
            probe_records,
            key=lambda record: (-cast(float, record["priority"]), record["proposal_id"]),
        )
        signature_base = {
            "schema_version": "stage3.behavior.v1",
            "priorities": probe_records,
            "rank_order": [record["proposal_id"] for record in ranked],
            "selected_proposal_id": ranked[0]["proposal_id"],
            "exception": False,
            "timeout": False,
            "crash": False,
            "protocol": False,
        }
        behavior = {
            **signature_base,
            "signature_sha256": canonical_hash(signature_base),
        }
        behavior_telemetry = worker.telemetry()
        # A separate persistent-worker smoke; do not recreate the worker.
        for _ in range(max(0, smoke_calls)):
            result = worker.call(ctx, proposals[0])
            if result.status != "ok" or result.priority is None:
                raise ValueError("runtime_exception")
        smoke_telemetry = worker.telemetry()
    finally:
        worker.close()
    return behavior, {
        "behavior_probe": behavior_telemetry,
        "persistent_smoke": smoke_telemetry,
        "smoke_calls": smoke_calls,
    }


def _diagnostics(errors: Sequence[Mapping[str, Any]], limit: int) -> tuple[Mapping[str, Any], ...]:
    allowed: list[Mapping[str, Any]] = []
    for error in errors:
        code = str(error.get("code", ""))
        if code in _REPAIRABLE:
            allowed.append({"code": code, "message": str(error.get("message", ""))[:256]})
    return tuple(allowed[:limit])


class OneShotGenerator:
    def __init__(
        self,
        provider: GenerationProvider,
        *,
        config: GenerationConfig | None = None,
        limits: SandboxLimits | None = None,
        artifacts: GenerationArtifacts | None = None,
        existing_sources: Sequence[str] = (),
        ranker_fixtures: Sequence[str] = (),
        slot_requests: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or GenerationConfig()
        self.limits = limits or SandboxLimits()
        self.artifacts = artifacts
        fixture_root = Path(__file__).resolve().parents[3] / "fixtures" / "rankers"
        auto = tuple(path.read_text(encoding="utf-8") for path in sorted(fixture_root.glob("*.py")))
        self.existing_sources = tuple(existing_sources) + tuple(ranker_fixtures) + auto
        self.slot_requests = dict(slot_requests or {})
        if self.slot_requests and tuple(sorted(self.slot_requests)) != SLOTS:
            raise ValueError("slot requests must cover exactly slot-00 through slot-07")
        self._active = 0
        self._max_active = 0
        self._lock = threading.Lock()

    def _request(self, slot: str) -> dict[str, Any]:
        if slot in self.slot_requests:
            return dict(self.slot_requests[slot])
        return {
            "slot": slot,
            "model": self.config.model,
            "effort": self.config.effort,
            "protocol_version": self.config.protocol_version,
            # Deliberately no benchmark, baseline, oracle, or other-candidate data.
            "prompt": {"slot": slot, "task": "write one priority policy"},
        }

    def _invoke(
        self,
        request: Mapping[str, Any],
        *,
        repair: bool = False,
        diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> Turn:
        with self._lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)
        try:
            try:
                return _call_provider(
                    self.provider, request, repair=repair, diagnostics=diagnostics
                )
            except Exception as error:
                # Retry only a demonstrably unaccepted, uncharged, content-free turn.
                if not self.config.allow_infrastructure_retry or repair:
                    raise
                if not (
                    all(hasattr(error, name) for name in ("accepted", "usage", "content"))
                    and cast(Any, error).accepted is False
                    and not cast(Any, error).usage
                    and cast(Any, error).content is False
                ):
                    raise
                return _call_provider(self.provider, request, repair=False, diagnostics=())
        finally:
            with self._lock:
                self._active -= 1

    def _assess(
        self,
        slot: str,
        turn: Turn,
        *,
        repairs: int = 0,
        repair_record: Mapping[str, Any] | None = None,
    ) -> SlotResult:
        raw_response = (
            turn.response
            if isinstance(turn.response, str)
            else json.dumps(
                turn.response,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        initial = {
            "status": turn.status,
            "accepted": turn.accepted,
            "charged": turn.charged,
            "content": turn.content,
            "usage": dict(turn.usage),
            "request_id": turn.request_id,
            "thread_id": turn.thread_id,
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            "canonical_response_sha256": canonical_hash(turn.response),
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
        source: str | None = None
        generated: GeneratedPolicy | None = None
        try:
            generated = _parse_envelope(turn.response)
            source = generated.source
        except Exception as error:
            errors.append({"code": "structured_output", "message": str(error)[:256]})
        validation: ValidationResult | None = None
        behavior: Mapping[str, Any] = {}
        worker_telemetry: Mapping[str, Any] = {}
        if source is not None:
            validation = validate_policy(source, self.limits)
            if not validation.valid:
                errors.extend(error.as_dict() for error in validation.errors)
            else:
                try:
                    behavior, worker_telemetry = _behavior(
                        source,
                        self.limits,
                        self.config.smoke_calls,
                    )
                except Exception as error:
                    code = str(error) if str(error) in _REPAIRABLE else "finite_probe"
                    errors.append({"code": code, "message": str(error)[:256]})
        if source is None or errors or validation is None or not validation.valid:
            return SlotResult(
                slot,
                "failed",
                errors=tuple(errors),
                repairs=repairs,
                usage=turn.usage,
                initial=initial,
                repair=repair_record,
                response=turn.response,
            )
        identity = validation.identity
        candidate = Candidate(
            slot,
            source,
            _sha_source(source),
            identity.normalized_ast_sha256 or "",
            behavior,
            worker_telemetry,
            {
                "candidate_id": f"candidate-{slot}",
                "slot": slot,
                "model": turn.model or self.config.model,
                "protocol_version": self.config.protocol_version,
                "request_id": turn.request_id,
                "thread_id": turn.thread_id,
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "usage": dict(turn.usage),
                "design_summary": generated.design_summary if generated else "",
                "used_fields": list(generated.used_fields) if generated else [],
                "assumptions": list(generated.assumptions) if generated else [],
            },
        )
        return SlotResult(
            slot,
            "duplicate_or_valid",
            candidate=candidate,
            repairs=repairs,
            usage=turn.usage,
            initial=initial,
            repair=repair_record,
            response=turn.response,
        )

    def run(self, *, run_id: str = "stage3") -> GenerationResult:
        if self.artifacts:
            self.artifacts.start({"run_id": run_id, "status": "failed", "slots": []})
        initial_turns: dict[str, Turn] = {}
        repair_turns: dict[str, Turn] = {}
        repair_inputs: dict[str, tuple[Mapping[str, Any], ...]] = {}
        results: dict[str, SlotResult] = {}
        initial_max_active = 0
        try:
            with ThreadPoolExecutor(max_workers=8, thread_name_prefix="stage3-slot") as executor:
                futures = {
                    executor.submit(self._invoke, self._request(slot)): slot for slot in SLOTS
                }
                for future in as_completed(futures):
                    slot = futures[future]
                    try:
                        initial_turns[slot] = future.result()
                    except BaseException as error:
                        results[slot] = SlotResult(
                            slot,
                            "failed",
                            errors=({"code": "provider_error", "message": str(error)[:256]},),
                        )
            initial_max_active = self._max_active
            # The complete initial wave is observed before any repair can begin.
            for slot in SLOTS:
                if slot in initial_turns:
                    results[slot] = self._assess(slot, initial_turns[slot])

            for slot, result in results.items():
                if result.status != "failed" or slot not in initial_turns:
                    continue
                diagnostics = _diagnostics(
                    result.errors,
                    self.config.max_repair_diagnostics,
                )
                if diagnostics:
                    repair_inputs[slot] = diagnostics
            with ThreadPoolExecutor(
                max_workers=8,
                thread_name_prefix="stage3-repair",
            ) as executor:
                futures = {
                    executor.submit(
                        self._invoke,
                        self._request(slot),
                        repair=True,
                        diagnostics=diagnostics,
                    ): slot
                    for slot, diagnostics in repair_inputs.items()
                }
                for future in as_completed(futures):
                    slot = futures[future]
                    try:
                        repair_turns[slot] = future.result()
                    except BaseException as error:
                        previous = results[slot]
                        results[slot] = SlotResult(
                            slot,
                            "failed",
                            errors=(
                                *previous.errors,
                                {"code": "repair_provider_error", "message": str(error)[:256]},
                            ),
                            repairs=1,
                            usage=previous.usage,
                            initial=previous.initial,
                            repair={
                                "status": "failed",
                                "diagnostics": list(repair_inputs[slot]),
                            },
                            response=previous.response,
                        )
            for slot, repair_turn in repair_turns.items():
                previous = results[slot]
                repair_record = {
                    "status": repair_turn.status,
                    "accepted": repair_turn.accepted,
                    "charged": repair_turn.charged,
                    "content": repair_turn.content,
                    "usage": dict(repair_turn.usage),
                    "request_id": repair_turn.request_id,
                    "thread_id": repair_turn.thread_id,
                    "session_id": repair_turn.session_id,
                    "turn_id": repair_turn.turn_id,
                    "diagnostics": list(repair_inputs[slot]),
                    "response_sha256": canonical_hash(repair_turn.response),
                }
                repaired = self._assess(
                    slot,
                    repair_turn,
                    repairs=1,
                    repair_record=repair_record,
                )
                results[slot] = SlotResult(
                    slot,
                    repaired.status,
                    repaired.candidate,
                    repaired.errors,
                    1,
                    {
                        "initial": dict(previous.usage),
                        "repair": dict(repair_turn.usage),
                    },
                    repaired.denied,
                    previous.initial,
                    repair_record,
                    repair_turn.response,
                )
        except BaseException:
            for slot in SLOTS:
                results.setdefault(slot, SlotResult(slot, "interrupted"))
        ordered = tuple(results.get(slot, SlotResult(slot, "interrupted")) for slot in SLOTS)
        seen = {_sha_source(source): "fixture" for source in self.existing_sources}
        seen_ast: dict[str, str] = {}
        for source in self.existing_sources:
            valid = validate_policy(source, self.limits)
            if valid.identity.normalized_ast_sha256:
                seen_ast[valid.identity.normalized_ast_sha256] = "fixture"
        unique: list[Candidate] = []
        rewritten: list[SlotResult] = []
        for result in ordered:
            candidate = result.candidate
            if candidate is None:
                rewritten.append(result)
                continue
            duplicate_of = seen_ast.get(candidate.normalized_ast_sha256) or seen.get(
                candidate.source_sha256
            )
            if duplicate_of is not None:
                candidate = replace(
                    candidate,
                    duplicate=True,
                    duplicate_of=duplicate_of,
                )
                rewritten.append(
                    SlotResult(
                        result.slot,
                        "duplicate",
                        candidate,
                        result.errors,
                        result.repairs,
                        result.usage,
                        result.denied,
                        result.initial,
                        result.repair,
                        result.response,
                    )
                )
            else:
                seen_ast[candidate.normalized_ast_sha256] = candidate.slot
                seen[candidate.source_sha256] = candidate.slot
                unique.append(candidate)
                rewritten.append(
                    SlotResult(
                        result.slot,
                        "accepted",
                        candidate,
                        result.errors,
                        result.repairs,
                        result.usage,
                        result.denied,
                        result.initial,
                        result.repair,
                        result.response,
                    )
                )
        all_turns = (*initial_turns.values(), *repair_turns.values())
        campaign_complete = (
            len(initial_turns) == len(SLOTS)
            and len(repair_turns) == len(repair_inputs)
            and all(
                turn.status == "completed"
                and turn.accepted
                and turn.content
                and _usage_complete(turn.usage)
                and turn.charged == (cast(int, turn.usage["totalTokens"]) > 0)
                for turn in all_turns
            )
        )
        usage_keys = (
            "inputTokens",
            "cachedInputTokens",
            "cacheWriteInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        )
        usage_totals = {
            key: sum(cast(int, turn.usage.get(key, 0)) for turn in all_turns) for key in usage_keys
        }
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
            "initial_turn_count": len(initial_turns),
            "repair_turn_count": len(repair_turns),
            "total_live_turns": len(initial_turns) + len(repair_turns),
            "exact_usage_complete": campaign_complete,
            "usage_totals": usage_totals,
            "initial_max_active": initial_max_active,
            "max_active": self._max_active,
            "timing": {},
        }
        if self.artifacts:
            if not (self.artifacts.root / "freeze.json").exists():
                self.artifacts.write(
                    "freeze.json",
                    {
                        "schema_version": "stage3.freeze.v1",
                        "slots": list(SLOTS),
                        "max_concurrency": 8,
                    },
                )
            if not (self.artifacts.root / "environment.json").exists():
                self.artifacts.write(
                    "environment.json",
                    {
                        "python": __import__("platform").python_version(),
                        "platform": __import__("platform").platform(),
                    },
                )
            self.artifacts.write("slots.json", summary["slots"])
            if not (self.artifacts.root / "generation_config.json").exists():
                self.artifacts.write(
                    "generation_config.json",
                    {
                        "model": self.config.model,
                        "effort": self.config.effort,
                        "protocol_version": self.config.protocol_version,
                        "smoke_calls": self.config.smoke_calls,
                        "sandbox_limits": asdict(self.limits),
                    },
                )
            if not (self.artifacts.root / "prompt_bundle.json").exists():
                self.artifacts.write(
                    "prompt_bundle.json",
                    {
                        "protocol_version": self.config.protocol_version,
                        "prompt": "write one priority policy",
                    },
                )
            if not (self.artifacts.root / "development_manifest.json").exists():
                self.artifacts.write(
                    "development_manifest.json",
                    {"stage": 3, "provider": "injected", "run_id": run_id},
                )
            for result in rewritten:
                slot_path = f"slots/{result.slot}"
                self.artifacts.write(f"{slot_path}/request.json", self._request(result.slot))
                self.artifacts.write(
                    f"{slot_path}/events.json",
                    [
                        {
                            "event": "terminal",
                            "status": result.status,
                            "initial": dict(result.initial),
                            "repair": result.repair,
                        }
                    ],
                )
                self.artifacts.write(f"{slot_path}/usage.json", dict(result.usage))
                self.artifacts.write(f"{slot_path}/response.json", result.response)
                self.artifacts.write(
                    f"{slot_path}/validation.json",
                    {"errors": list(result.errors), "status": result.status},
                )
                if result.candidate:
                    parsed = _parse_envelope(result.response)
                    validation = validate_policy(result.candidate.source, self.limits)
                    self.artifacts.write(
                        f"{slot_path}/validation.json",
                        {
                            **validation.as_dict(),
                            "status": result.status,
                            "errors": list(result.errors),
                        },
                    )
                    self.artifacts.write_text(f"{slot_path}/source.py", result.candidate.source)
                    self.artifacts.write(
                        f"{slot_path}/canonical_response.json",
                        parsed.as_dict(),
                    )
                    self.artifacts.write(
                        f"{slot_path}/identity.json",
                        validation.identity.as_dict(),
                    )
                    self.artifacts.write(
                        f"{slot_path}/provenance.json",
                        result.candidate.provenance,
                    )
                    self.artifacts.write(
                        f"{slot_path}/behavior.json",
                        {"signature": result.candidate.behavior_signature},
                    )
                    self.artifacts.write(
                        f"{slot_path}/worker_telemetry.json",
                        result.candidate.worker_telemetry,
                    )
                if result.repair:
                    self.artifacts.write(f"{slot_path}/repair.json", result.repair)
            self.artifacts.finish(cast(str, summary["status"]), summary)
        close_provider = getattr(self.provider, "close", None)
        if callable(close_provider):
            close_provider()
        return GenerationResult(
            cast(str, summary["status"]), tuple(rewritten), tuple(unique), summary
        )


def generate_once(provider: GenerationProvider, **kwargs: Any) -> GenerationResult:
    return OneShotGenerator(provider, **kwargs).run()


__all__ = [
    "Candidate",
    "GenerationArtifacts",
    "GenerationConfig",
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

GenerationOrchestrator = OneShotGenerator
