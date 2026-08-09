"""Bounded sustained ordinary-Python search with concurrent evaluation."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, cast

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue

from . import search as core
from .validation import normalize_source_newlines, validate_python_policy_response

M9_SEARCH_PROTOCOL_ID = "mforge.native.python_scientific_search.v1"
M9_PREPARED_CANDIDATE_PROTOCOL_ID = (
    "mforge.native.python_scientific_search_prepared_candidate.v1"
)
M9_RUNTIME_PROTOCOL_ID = "mforge.native.python_scientific_search_runtime.v1"
M9_REPORT_PROTOCOL_ID = "mforge.native.python_scientific_search_report.v1"
M9_RUNTIME_FILENAME = "m9-runtime.json.gz"
M9_REPORT_FILENAME = "m9-report.json.gz"
M9_STOP_FILENAME = "m9-stop.json.gz"
M9_PREPARED_FILENAME = "prepared-candidate.json.gz"
M9_MAX_CONSECUTIVE_PROVIDER_FAILURES = 3


class _ProviderTurnBudgetExhausted(core.M5SearchError):
    """The durable M9 provider reservation budget is terminally exhausted."""


@dataclass(frozen=True, slots=True)
class ScientificSearchOptionsV1:
    """Immutable host-owned controls for one sustained campaign."""

    generation_limit: int
    evaluator_workers: int
    provider_concurrency: int
    wall_seconds: float
    provider_program_turn_limit: int
    stop_on_verified: bool
    resume_enabled: bool
    replace_terminal_slots: bool

    def __post_init__(self) -> None:
        if not 1 <= self.generation_limit <= 64:
            raise ValueError("generation_limit must be between 1 and 64")
        if not 1 <= self.evaluator_workers <= 12:
            raise ValueError("evaluator_workers must be between 1 and 12")
        if self.provider_concurrency != 1:
            raise ValueError("provider_concurrency must remain exactly one")
        if not 1 <= self.wall_seconds <= 8 * 60 * 60:
            raise ValueError("wall_seconds must be between 1 and 28,800")
        if not 1 <= self.provider_program_turn_limit <= 64:
            raise ValueError(
                "provider_program_turn_limit must be between 1 and 64"
            )
        if not self.stop_on_verified:
            raise ValueError("stop_on_verified must remain enabled")
        if not self.resume_enabled:
            raise ValueError("resume_enabled must remain enabled")
        if self.replace_terminal_slots:
            raise ValueError("terminal slots must never be replaced")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "generation_limit": self.generation_limit,
            "evaluator_workers": self.evaluator_workers,
            "provider_concurrency": self.provider_concurrency,
            "wall_seconds": self.wall_seconds,
            "provider_program_turn_limit": self.provider_program_turn_limit,
            "stop_on_verified": self.stop_on_verified,
            "resume_enabled": self.resume_enabled,
            "replace_terminal_slots": self.replace_terminal_slots,
        }


class _RuntimeTelemetry:
    """Mutable timing/status telemetry excluded from scientific identity."""

    def __init__(
        self,
        root: Path,
        options: ScientificSearchOptionsV1,
    ) -> None:
        self._path = root / M9_RUNTIME_FILENAME
        self._lock = threading.Lock()
        self._resume_started = time.monotonic()
        if self._path.is_file():
            raw = read_json(self._path)
            if not isinstance(raw, Mapping):
                raise core.M5InfrastructureError("M9 runtime telemetry is malformed")
            state = dict(raw)
            if (
                state.get("protocol_id") != M9_RUNTIME_PROTOCOL_ID
                or state.get("options") != options.as_dict()
            ):
                raise core.M5InfrastructureError(
                    "M9 runtime telemetry options changed on resume"
                )
            state["resume_attempts"] = int(state.get("resume_attempts", 0)) + 1
            state["active_evaluators"] = 0
            state["queued_evaluations"] = 0
        else:
            state = {
                "protocol_id": M9_RUNTIME_PROTOCOL_ID,
                "options": options.as_dict(),
                "campaign_started_epoch_seconds": time.time(),
                "active_elapsed_seconds": 0.0,
                "resume_attempts": 0,
                "provider_turns_submitted": 0,
                "provider_turn_keys": [],
                "provider_wait_seconds": 0.0,
                "persistence_seconds": 0.0,
                "evaluator_busy_seconds": 0.0,
                "evaluator_queue_wait_seconds": 0.0,
                "active_evaluators": 0,
                "peak_active_evaluators": 0,
                "queued_evaluations": 0,
                "completed_evaluations": 0,
                "failed_evaluations": 0,
                "evaluator_instances": 0,
                "first_valid_program_seconds": None,
                "last_scientific_improvement_epoch_seconds": None,
                "best_candidate_id": None,
                "best_fitness_interval": None,
                "last_boundary": None,
                "terminal_reason": None,
            }
        self._state = state
        self._persist_locked()

    def _elapsed_locked(self) -> float:
        return float(self._state["active_elapsed_seconds"]) + (
            time.monotonic() - self._resume_started
        )

    def _payload_locked(self) -> dict[str, Any]:
        return {
            **self._state,
            "active_elapsed_seconds": self._elapsed_locked(),
            "updated_epoch_seconds": time.time(),
        }

    def _persist_locked(self) -> None:
        write_json(self._path, self._payload_locked())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._payload_locked()

    def boundary(self, value: str) -> None:
        with self._lock:
            self._state["last_boundary"] = value
            self._persist_locked()

    def wall_expired(self, limit: float) -> bool:
        with self._lock:
            return self._elapsed_locked() >= limit

    def wall_remaining(self, limit: float) -> float:
        with self._lock:
            return max(0.0, limit - self._elapsed_locked())

    def provider_turn_available(self, limit: int, *, key: str | None = None) -> bool:
        with self._lock:
            keys = self._provider_turn_keys_locked()
            return (key is not None and key in keys) or len(keys) < limit

    def _provider_turn_keys_locked(self) -> list[str]:
        value = self._state.get("provider_turn_keys", [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise core.M5InfrastructureError(
                "M9 provider turn reservations are malformed"
            )
        return cast(list[str], value)

    def provider_started(self, key: str, limit: int) -> bool:
        with self._lock:
            keys = self._provider_turn_keys_locked()
            if key in keys:
                return False
            if len(keys) >= limit:
                raise _ProviderTurnBudgetExhausted(
                    "provider program-turn budget exhausted"
                )
            keys.append(key)
            self._state["provider_turn_keys"] = keys
            self._state["provider_turns_submitted"] = len(keys)
            self._persist_locked()
            return True

    def provider_finished(self, elapsed: float) -> None:
        with self._lock:
            self._state["provider_wait_seconds"] = (
                float(self._state["provider_wait_seconds"]) + elapsed
            )
            self._persist_locked()

    def timed_persist(self, operation: Callable[[], None]) -> None:
        started = time.monotonic()
        operation()
        elapsed = time.monotonic() - started
        with self._lock:
            self._state["persistence_seconds"] = (
                float(self._state["persistence_seconds"]) + elapsed
            )
            self._persist_locked()

    def first_valid_program(self) -> None:
        with self._lock:
            if self._state["first_valid_program_seconds"] is None:
                self._state["first_valid_program_seconds"] = self._elapsed_locked()
                self._persist_locked()

    def evaluator_created(self) -> None:
        with self._lock:
            self._state["evaluator_instances"] = (
                int(self._state["evaluator_instances"]) + 1
            )
            self._persist_locked()

    def evaluator_queued(self) -> None:
        with self._lock:
            self._state["queued_evaluations"] = (
                int(self._state["queued_evaluations"]) + 1
            )
            self._persist_locked()

    def evaluator_started(self, queue_wait_seconds: float) -> None:
        with self._lock:
            self._state["queued_evaluations"] = max(
                0,
                int(self._state["queued_evaluations"]) - 1,
            )
            active = int(self._state["active_evaluators"]) + 1
            self._state["active_evaluators"] = active
            self._state["evaluator_queue_wait_seconds"] = (
                float(self._state["evaluator_queue_wait_seconds"])
                + queue_wait_seconds
            )
            self._state["peak_active_evaluators"] = max(
                active,
                int(self._state["peak_active_evaluators"]),
            )
            self._persist_locked()

    def evaluator_finished(self, elapsed: float, *, failed: bool) -> None:
        with self._lock:
            self._state["active_evaluators"] = max(
                0,
                int(self._state["active_evaluators"]) - 1,
            )
            self._state["evaluator_busy_seconds"] = (
                float(self._state["evaluator_busy_seconds"]) + elapsed
            )
            key = "failed_evaluations" if failed else "completed_evaluations"
            self._state[key] = int(self._state[key]) + 1
            self._persist_locked()

    def scientific_improvement(
        self,
        *,
        candidate_id: str,
        fitness_interval: Mapping[str, Any],
    ) -> None:
        with self._lock:
            self._state["last_scientific_improvement_epoch_seconds"] = time.time()
            self._state["best_candidate_id"] = candidate_id
            self._state["best_fitness_interval"] = dict(fitness_interval)
            self._persist_locked()

    def finish(self, reason: str) -> None:
        with self._lock:
            self._state["active_elapsed_seconds"] = self._elapsed_locked()
            self._resume_started = time.monotonic()
            self._state["active_evaluators"] = 0
            self._state["queued_evaluations"] = 0
            self._state["terminal_reason"] = reason
            self._persist_locked()

    def clear_terminal_reason(self) -> None:
        with self._lock:
            self._state["terminal_reason"] = None
            self._persist_locked()


@dataclass(frozen=True, slots=True)
class _EvaluationOutcome:
    payloads: tuple[dict[str, Any], ...]
    failure_type: str | None = None
    failure_message: str | None = None
    failure_case_id: str | None = None


class _ConcurrentEvaluatorPool:
    """Thread pool whose workers each own one evaluator/backend chain."""

    def __init__(
        self,
        *,
        workers: int,
        evaluator_factory: Callable[[], core.M5ScientificEvaluator],
        telemetry: _RuntimeTelemetry,
    ) -> None:
        self._factory = evaluator_factory
        self._telemetry = telemetry
        self._local = threading.local()
        self._owned: list[core.M5ScientificEvaluator] = []
        self._owned_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mforge-m9-evaluator",
        )

    def _evaluator(self) -> core.M5ScientificEvaluator:
        evaluator = getattr(self._local, "evaluator", None)
        if evaluator is None:
            evaluator = self._factory()
            self._local.evaluator = evaluator
            with self._owned_lock:
                self._owned.append(evaluator)
            self._telemetry.evaluator_created()
        return cast(core.M5ScientificEvaluator, evaluator)

    def submit(
        self,
        *,
        source: str,
        panel: tuple[core.DevelopmentCaseV1, ...],
        candidate_id: str,
        slot_dir: Path,
    ) -> Future[_EvaluationOutcome]:
        self._telemetry.evaluator_queued()
        queued_at = time.monotonic()

        def evaluate() -> _EvaluationOutcome:
            started = time.monotonic()
            self._telemetry.evaluator_started(started - queued_at)
            payloads: list[dict[str, Any]] = []
            failed = False
            try:
                try:
                    evaluator = self._evaluator()
                except Exception as error:
                    failed = True
                    return _EvaluationOutcome(
                        (),
                        type(error).__name__,
                        str(error)[:1024],
                        panel[0].case_id,
                    )
                for case in panel:
                    evaluation_path = (
                        slot_dir / "evaluations" / f"{case.case_id}.json.gz"
                    )
                    if evaluation_path.is_file():
                        payloads.append(core._load_mapping(evaluation_path))
                        continue
                    try:
                        payloads.append(
                            dict(
                                evaluator.evaluate(
                                    source=source,
                                    case=case,
                                    candidate_id=candidate_id,
                                )
                            )
                        )
                    except Exception as error:
                        failed = True
                        return _EvaluationOutcome(
                            tuple(payloads),
                            type(error).__name__,
                            str(error)[:1024],
                            case.case_id,
                        )
                return _EvaluationOutcome(tuple(payloads))
            finally:
                self._telemetry.evaluator_finished(
                    time.monotonic() - started,
                    failed=failed,
                )

        return self._executor.submit(evaluate)

    def close(self) -> Exception | None:
        errors: list[Exception] = []
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        except Exception as error:
            errors.append(error)
        for evaluator in self._owned:
            close = getattr(evaluator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        return errors[0] if errors else None


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        retained = read_json(path)
        if retained != dict(value):
            raise core.M5InfrastructureError(
                f"immutable M9 metadata changed: {path}"
            )
        return
    write_json(path, value, exclusive=True)


def _prepared_path(slot_dir: Path) -> Path:
    return slot_dir / M9_PREPARED_FILENAME


def _verify_prepared(
    *,
    root: Path,
    path: Path,
    panel: tuple[core.DevelopmentCaseV1, ...],
    slot_plan: core.SlotPlanV1,
    search_memory_sha256: str,
) -> dict[str, Any]:
    value = core._load_mapping(path)
    expected_generation = int(path.parent.parent.name.removeprefix("generation-"))
    expected_id = core._candidate_id(expected_generation, slot_plan.slot)
    if (
        value.get("protocol_id") != M9_PREPARED_CANDIDATE_PROTOCOL_ID
        or value.get("status") != "evaluation_pending"
        or value.get("candidate_id") != expected_id
        or value.get("generation") != expected_generation
        or value.get("slot") != slot_plan.slot
        or value.get("kind") != slot_plan.kind
        or value.get("parent_candidate_id") != slot_plan.parent_candidate_id
        or value.get("panel_hash") != core.panel_hash(panel)
        or value.get("panel_case_ids") != [item.case_id for item in panel]
        or value.get("search_memory_sha256") != search_memory_sha256
    ):
        raise core.M5InfrastructureError(
            f"retained prepared candidate identity changed: {path}"
        )
    candidates = core._all_candidates(root)
    parent = next(
        (
            item
            for item in candidates
            if item.get("candidate_id") == slot_plan.parent_candidate_id
        ),
        None,
    )
    if slot_plan.kind == "child" and parent is None:
        raise core.M5InfrastructureError("prepared child parent is unavailable")
    if (
        value.get("parent_program_hash")
        != (parent.get("program_hash") if parent is not None else None)
        or value.get("parent_behavior_signature")
        != (
            parent.get("behavior_signature")
            if parent is not None
            else None
        )
    ):
        raise core.M5InfrastructureError("prepared parent identity changed")
    attempts_raw = value.get("provider_attempts")
    if not isinstance(attempts_raw, Sequence) or isinstance(
        attempts_raw, str | bytes
    ):
        raise core.M5InfrastructureError("prepared provider attempts are malformed")
    attempts = [
        core.M5ProviderResultV1.from_dict(item)
        for item in attempts_raw
        if isinstance(item, Mapping)
    ]
    if len(attempts) != len(attempts_raw) or not attempts:
        raise core.M5InfrastructureError("prepared provider evidence changed")
    for previous, current in zip(attempts, attempts[1:], strict=False):
        if (
            current.context.thread_id != previous.context.thread_id
            or current.context.included_turn_ids[:-1]
            != previous.context.included_turn_ids
        ):
            raise core.M5InfrastructureError(
                "prepared provider attempt lineage changed"
            )
    provider_context = value.get("provider_context")
    if (
        not isinstance(provider_context, Mapping)
        or core.M5ProviderContextV1.from_dict(provider_context)
        != attempts[-1].context
    ):
        raise core.M5InfrastructureError(
            "prepared provider context changed"
        )
    validation = validate_python_policy_response(attempts[-1].response_text)
    if (
        not validation.valid
        or validation.response is None
        or validation.identity is None
        or validation.identity.program_hash is None
        or value.get("validation") != validation.as_dict()
    ):
        raise core.M5InfrastructureError("prepared candidate no longer validates")
    source = normalize_source_newlines(validation.response.source)
    source_path_value = value.get("source_path")
    if not isinstance(source_path_value, str):
        raise core.M5InfrastructureError("prepared source path is missing")
    source_path = (root / source_path_value).resolve()
    if (
        not source_path.is_relative_to(root.resolve())
        or not source_path.is_file()
        or source_path.read_text(encoding="utf-8") != source
        or value.get("source") != source
        or value.get("program_hash") != validation.identity.program_hash
        or value.get("source_sha256") != validation.identity.source_sha256
        or value.get("canonical_ast_sha256")
        != validation.identity.canonical_ast_sha256
    ):
        raise core.M5InfrastructureError("prepared source identity changed")
    return value


@dataclass(slots=True)
class _PendingCommit:
    slot_plan: core.SlotPlanV1
    candidate_id: str
    slot_dir: Path
    terminal_payload: dict[str, Any] | None = None
    prepared: dict[str, Any] | None = None
    future: Future[_EvaluationOutcome] | None = None
    retained_terminal: bool = False

    @property
    def already_terminal(self) -> bool:
        return self.retained_terminal

    @property
    def unprocessed(self) -> bool:
        return (
            not self.retained_terminal
            and self.terminal_payload is None
            and self.prepared is None
            and self.future is None
        )


def _provider_call(
    *,
    telemetry: _RuntimeTelemetry,
    options: ScientificSearchOptionsV1,
    key: str,
    durable_result_path: Path,
    operation: Callable[[], core.M5ProviderResultV1],
) -> core.M5ProviderResultV1:
    newly_reserved = telemetry.provider_started(
        key,
        options.provider_program_turn_limit,
    )
    if not newly_reserved and not durable_result_path.is_file():
        raise core.M5SearchError(
            "interrupted provider turn has no durable result and will not repeat"
        )
    started = time.monotonic()
    try:
        return operation()
    finally:
        telemetry.provider_finished(time.monotonic() - started)


def _candidate_base(
    *,
    candidate_id: str,
    generation: int,
    slot_plan: core.SlotPlanV1,
    parent: Mapping[str, Any] | None,
    panel: tuple[core.DevelopmentCaseV1, ...],
    memory: Mapping[str, Any],
    provider_result: core.M5ProviderResultV1 | None,
    attempts: Sequence[Mapping[str, JsonValue]],
    repairs: int,
    prompt: str,
) -> dict[str, Any]:
    return {
        "protocol_id": core.M5_CANDIDATE_PROTOCOL_ID,
        "candidate_id": candidate_id,
        "generation": generation,
        "slot": slot_plan.slot,
        "kind": slot_plan.kind,
        "parent_candidate_id": slot_plan.parent_candidate_id,
        "parent_program_hash": (
            parent.get("program_hash") if parent is not None else None
        ),
        "parent_behavior_signature": (
            parent.get("behavior_signature") if parent is not None else None
        ),
        "panel_hash": slot_plan.panel_hash,
        "panel_case_ids": [item.case_id for item in panel],
        "search_memory_sha256": memory["sha256"],
        "provider_context": (
            provider_result.context.as_dict()
            if provider_result is not None
            else None
        ),
        "provider_attempts": list(attempts),
        "repairs": repairs,
        "usage": core._attempt_usage_total(attempts),
        "duration_ms": sum(
            core._json_nonnegative_int(
                attempt.get("duration_ms"),
                field="duration_ms",
            )
            for attempt in attempts
        ),
        "warnings": sum(
            core._json_nonnegative_int(
                attempt.get("warnings"),
                field="warnings",
            )
            for attempt in attempts
        ),
        "request_prompt_bytes": len(prompt.encode("utf-8")),
    }


def _provider_failure(
    *,
    base: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        **base,
        "status": "provider_failed",
        "program_hash": None,
        "behavior_signature": None,
        "behavior_profile": None,
        "duplicate_of": None,
        "evaluation_case_count": 0,
        "failure": {
            "type": type(error).__name__,
            "message": str(error)[:1024],
        },
    }


def _contract_invalid(
    *,
    base: Mapping[str, Any],
    validation: Mapping[str, Any],
    repair_skipped: str | None = None,
) -> dict[str, Any]:
    value = {
        **base,
        "status": "contract_invalid",
        "validation": dict(validation),
        "program_hash": None,
        "behavior_signature": None,
        "behavior_profile": None,
        "duplicate_of": None,
        "evaluation_case_count": 0,
    }
    if repair_skipped is not None:
        value["repair_skipped"] = repair_skipped
    return value


def _prepared_candidate(
    *,
    base: Mapping[str, Any],
    validation: Mapping[str, Any],
    source: str,
    source_path: Path,
    root: Path,
    program_hash: str,
    source_sha256: str,
    canonical_ast_sha256: str,
) -> dict[str, Any]:
    return {
        **base,
        "protocol_id": M9_PREPARED_CANDIDATE_PROTOCOL_ID,
        "status": "evaluation_pending",
        "validation": dict(validation),
        "source": source,
        "source_path": str(source_path.relative_to(root)),
        "source_sha256": source_sha256,
        "canonical_ast_sha256": canonical_ast_sha256,
        "program_hash": program_hash,
    }


def _interval_fraction(value: object, bound: str) -> Fraction:
    if not isinstance(value, Mapping):
        return Fraction()
    item = value.get(bound)
    if not isinstance(item, Mapping):
        return Fraction()
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def _is_improvement(
    candidate: Mapping[str, Any],
    retained_best: object,
) -> bool:
    profile = candidate.get("behavior_profile")
    if not isinstance(profile, Mapping):
        return False
    interval = profile.get("fitness_interval")
    if retained_best is None:
        return True
    candidate_key = (
        _interval_fraction(interval, "lower"),
        _interval_fraction(interval, "upper"),
        str(candidate.get("program_hash", "")),
    )
    best_key = (
        _interval_fraction(retained_best, "lower"),
        _interval_fraction(retained_best, "upper"),
        "",
    )
    return candidate_key[:2] > best_key[:2]


def _commit_pending(
    *,
    pending: _PendingCommit,
    root: Path,
    panel: tuple[core.DevelopmentCaseV1, ...],
    telemetry: _RuntimeTelemetry,
    block: bool,
    boundary_hook: Callable[[str], None] | None,
) -> tuple[bool, core.M5InfrastructureError | None] | None:
    if pending.already_terminal:
        return False, None
    if pending.unprocessed:
        return None
    candidate_path = pending.slot_dir / "candidate.json.gz"
    if pending.terminal_payload is not None:
        terminal_payload = pending.terminal_payload
        telemetry.timed_persist(
            partial(_write_or_verify, candidate_path, terminal_payload)
        )
        if boundary_hook is not None:
            boundary_hook(f"{pending.candidate_id}_committed")
        pending.terminal_payload = None
        return False, None
    if pending.future is None or pending.prepared is None:
        raise core.M5InfrastructureError("M9 pending candidate is incomplete")
    if not block and not pending.future.done():
        return None
    outcome = pending.future.result()
    for case, evaluation in zip(panel, outcome.payloads, strict=False):
        evaluation_path = (
            pending.slot_dir / "evaluations" / f"{case.case_id}.json.gz"
        )
        telemetry.timed_persist(
            partial(_write_or_verify, evaluation_path, evaluation)
        )
        if boundary_hook is not None:
            boundary_hook(
                f"{pending.candidate_id}_evaluation_{case.case_id}"
            )
    base = {
        key: value
        for key, value in pending.prepared.items()
        if key
        not in {
            "protocol_id",
            "status",
        }
    }
    base["protocol_id"] = core.M5_CANDIDATE_PROTOCOL_ID
    if outcome.failure_type is not None:
        candidate = {
            **base,
            "status": "evaluation_infrastructure_failure",
            "behavior_signature": None,
            "behavior_profile": None,
            "duplicate_of": None,
            "evaluation_case_count": len(outcome.payloads),
            "failure": {
                "type": outcome.failure_type,
                "message": outcome.failure_message,
                "case_id": outcome.failure_case_id,
            },
        }
        telemetry.timed_persist(
            partial(_write_or_verify, candidate_path, candidate)
        )
        if boundary_hook is not None:
            boundary_hook(f"{pending.candidate_id}_committed")
        pending.prepared = None
        pending.future = None
        return (
            False,
            core.M5InfrastructureError(
                "development evaluation failed for "
                f"{pending.candidate_id}/{outcome.failure_case_id}: "
                f"{outcome.failure_type}"
            ),
        )
    if len(outcome.payloads) != len(panel):
        raise core.M5InfrastructureError(
            "M9 evaluator returned an incomplete development panel"
        )
    behavior_profile = core.aggregate_behavior(outcome.payloads)
    behavior_signature = str(behavior_profile["behavior_signature"])
    duplicate_of = core._seen_duplicates(
        core._all_candidates(root),
        program_hash=str(base["program_hash"]),
        behavior_signature=behavior_signature,
    )
    candidate = {
        **base,
        "status": "duplicate" if duplicate_of is not None else "evaluated",
        "behavior_signature": behavior_signature,
        "behavior_profile": behavior_profile,
        "control_flow": core.python_control_flow_summary(str(base["source"])),
        "duplicate_of": duplicate_of,
        "evaluation_case_count": len(outcome.payloads),
        "exact_verified": behavior_profile["exact_verified"],
    }
    telemetry.timed_persist(
        partial(_write_or_verify, candidate_path, candidate)
    )
    runtime = telemetry.snapshot()
    if _is_improvement(candidate, runtime.get("best_fitness_interval")):
        interval = behavior_profile.get("fitness_interval")
        if isinstance(interval, Mapping):
            telemetry.scientific_improvement(
                candidate_id=pending.candidate_id,
                fitness_interval=interval,
            )
    if boundary_hook is not None:
        boundary_hook(f"{pending.candidate_id}_committed")
    pending.prepared = None
    pending.future = None
    return behavior_profile["exact_verified"] is True, None


def _child_mutation_proofs(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, JsonValue]]:
    by_id = {str(item["candidate_id"]): item for item in candidates}
    result: list[dict[str, JsonValue]] = []
    for candidate in candidates:
        if candidate.get("kind") != "child":
            continue
        parent = by_id.get(str(candidate.get("parent_candidate_id")), {})
        result.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "parent_candidate_id": (
                    str(candidate["parent_candidate_id"])
                    if candidate.get("parent_candidate_id") is not None
                    else None
                ),
                "source_changed": (
                    isinstance(candidate.get("source"), str)
                    and isinstance(parent.get("source"), str)
                    and candidate.get("source") != parent.get("source")
                ),
                "program_changed": (
                    isinstance(candidate.get("program_hash"), str)
                    and isinstance(parent.get("program_hash"), str)
                    and candidate.get("program_hash")
                    != parent.get("program_hash")
                ),
                "semantic_behavior_changed": (
                    isinstance(candidate.get("behavior_signature"), str)
                    and isinstance(parent.get("behavior_signature"), str)
                    and candidate.get("behavior_signature")
                    != parent.get("behavior_signature")
                ),
            }
        )
    return result


def _report(
    *,
    root: Path,
    panel: tuple[core.DevelopmentCaseV1, ...],
    provider_model: str,
    provider_effort: str,
    anchor_result: core.M5ProviderResultV1,
    options: ScientificSearchOptionsV1,
    runtime: Mapping[str, Any],
    stop_reason: str,
) -> dict[str, Any]:
    candidates = core._all_candidates(root)
    manifests = {
        int(path.parent.name.removeprefix("generation-")): core._load_mapping(
            path
        )
        for path in sorted(
            root.glob("generations/generation-*/manifest.json.gz")
        )
    }
    generations = sorted(manifests)
    planned_candidate_count = sum(
        len(cast(Sequence[object], manifest.get("slots", ())))
        for manifest in manifests.values()
    )
    statuses = Counter(str(item.get("status")) for item in candidates)
    proofs = _child_mutation_proofs(candidates)
    runtime_payload = dict(runtime)
    profiles = [
        cast(Mapping[str, Any], item["behavior_profile"])
        for item in candidates
        if isinstance(item.get("behavior_profile"), Mapping)
    ]
    exact_verified = any(
        profile.get("exact_verified") is True for profile in profiles
    )
    report: dict[str, Any] = {
        "protocol_id": M9_REPORT_PROTOCOL_ID,
        "search_protocol_id": M9_SEARCH_PROTOCOL_ID,
        "status": "completed",
        "stop_reason": stop_reason,
        "generation_count": len(generations),
        "generation_limit": options.generation_limit,
        "population_size": core.POPULATION_SIZE,
        "planned_candidate_count": planned_candidate_count,
        "candidate_count": len(candidates),
        "pending_candidate_count": max(
            0, planned_candidate_count - len(candidates)
        ),
        "candidate_status_counts": dict(sorted(statuses.items())),
        "generation_status_counts": {
            str(generation): dict(
                sorted(
                    Counter(
                        str(item.get("status"))
                        for item in candidates
                        if int(item["generation"]) == generation
                    ).items()
                )
            )
            for generation in generations
        },
        "generation_allocations": {
            str(generation): {
                "children": sum(
                    slot.get("kind") == "child"
                    for slot in cast(
                        Sequence[Mapping[str, Any]],
                        manifests[generation].get("slots", ()),
                    )
                ),
                "roots": sum(
                    slot.get("kind") == "root"
                    for slot in cast(
                        Sequence[Mapping[str, Any]],
                        manifests[generation].get("slots", ()),
                    )
                ),
            }
            for generation in generations
        },
        "generation_manifest_hashes": {
            str(generation): core._load_mapping(
                root
                / "generations"
                / f"generation-{generation:04d}"
                / "manifest.json.gz"
            )["sha256"]
            for generation in generations
        },
        "search_memory_hashes": {
            str(generation): core._load_mapping(
                root
                / "generations"
                / f"generation-{generation:04d}"
                / "search-memory.json.gz"
            )["sha256"]
            for generation in generations
            if (
                root
                / "generations"
                / f"generation-{generation:04d}"
                / "search-memory.json.gz"
            ).is_file()
        },
        "lineage": [
            {
                "candidate_id": item["candidate_id"],
                "parent_candidate_id": item.get("parent_candidate_id"),
                "parent_program_hash": item.get("parent_program_hash"),
                "parent_behavior_signature": item.get(
                    "parent_behavior_signature"
                ),
                "generation": item["generation"],
                "slot": item["slot"],
                "kind": item["kind"],
                "program_hash": item.get("program_hash"),
                "behavior_signature": item.get("behavior_signature"),
                "provider_context": item.get("provider_context"),
            }
            for item in candidates
        ],
        "provider_order": [item["candidate_id"] for item in candidates],
        "evaluation_order": [
            {
                "candidate_id": item["candidate_id"],
                "case_id": case.case_id,
            }
            for item in candidates
            for case in panel
            if (
                root
                / "generations"
                / f"generation-{int(item['generation']):04d}"
                / str(item["slot"])
                / "evaluations"
                / f"{case.case_id}.json.gz"
            ).is_file()
        ],
        "child_mutation_proofs": proofs,
        "behavior_profiles": {
            str(item["candidate_id"]): item.get("behavior_profile")
            for item in candidates
        },
        "duplicates": [
            {
                "candidate_id": item["candidate_id"],
                "duplicate_of": item["duplicate_of"],
            }
            for item in candidates
            if item.get("duplicate_of") is not None
        ],
        "usage": core._usage_with_anchor(candidates, anchor_result.usage),
        "provider_turns": 1
        + int(runtime_payload.get("provider_turns_submitted", 0)),
        "specification_anchor_turns": 1,
        "candidate_program_turns": int(
            runtime_payload.get("provider_turns_submitted", 0)
        ),
        "repair_turns": sum(int(item.get("repairs", 0)) for item in candidates),
        "provider_accounting": {
            "model": provider_model,
            "effort": provider_effort,
            "warnings": anchor_result.warnings
            + sum(int(item.get("warnings", 0)) for item in candidates),
            "duration_ms": anchor_result.duration_ms
            + sum(int(item.get("duration_ms", 0)) for item in candidates),
            "program_turn_limit": options.provider_program_turn_limit,
        },
        "exact_verified": exact_verified,
        "exact_verification": {
            "authority": "exact_verifier_only",
            "submissions": sum(
                int(profile.get("exact_verifier_submissions", 0))
                for profile in profiles
            ),
            "records": sum(
                int(profile.get("exact_verifier_records", 0))
                for profile in profiles
            ),
            "verified": exact_verified,
            "queue": 0,
        },
        "equal_panel_hash": core.panel_hash(panel),
        "equal_development_budget": {
            "case_count": len(panel),
            "case_ids": [item.case_id for item in panel],
            "all_evaluated_candidates_complete": all(
                int(item.get("evaluation_case_count", 0)) == len(panel)
                for item in candidates
                if item.get("status") in {"evaluated", "duplicate"}
            ),
            "held_out_evidence_used": False,
        },
        "sequential": False,
        "provider_concurrency": options.provider_concurrency,
        "evaluator_workers": options.evaluator_workers,
        "preview_active": True,
        "native_v2_default": True,
        "dsl_runtime_used": False,
        "safe_api_expanded": True,
        "api_expressiveness": {
            "edge_scoped_k_switch_selector": True,
            "edge_scoped_fanout_selector": True,
            "edge_scoped_relocation_selector": True,
        },
        "runtime": runtime_payload,
        "acceptance_checks": {
            "generation_zero_eight_roots": sum(
                slot.get("kind") == "root"
                for slot in cast(
                    Sequence[Mapping[str, Any]],
                    manifests.get(0, {}).get("slots", ()),
                )
            )
            == core.POPULATION_SIZE,
            "later_generations_four_children_four_roots": all(
                (
                    sum(
                        slot.get("kind") == "child"
                        for slot in cast(
                            Sequence[Mapping[str, Any]],
                            manifests[generation].get("slots", ()),
                        )
                    )
                    == core.CHILD_SLOTS
                    and sum(
                        slot.get("kind") == "root"
                        for slot in cast(
                            Sequence[Mapping[str, Any]],
                            manifests[generation].get("slots", ()),
                        )
                    )
                    == core.ROOT_SLOTS
                )
                for generation in generations
                if generation > 0
            ),
            "terminal_slots_not_replaced": options.replace_terminal_slots
            is False,
            "equal_development_panel_and_budget": all(
                int(item.get("evaluation_case_count", 0)) == len(panel)
                for item in candidates
                if item.get("status") in {"evaluated", "duplicate"}
            ),
            "exact_verifier_only_authority": True,
            "provider_program_turn_budget_respected": int(
                runtime_payload.get("provider_turns_submitted", 0)
            )
            <= options.provider_program_turn_limit,
        },
    }
    return report


def _panel_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[core.DevelopmentCaseV1, ...]:
    raw_panel = protocol.get("panel")
    if not isinstance(raw_panel, Sequence) or isinstance(
        raw_panel, str | bytes
    ):
        raise core.M5InfrastructureError("M9 protocol panel is malformed")
    panel: list[core.DevelopmentCaseV1] = []
    for raw_case in raw_panel:
        if not isinstance(raw_case, Mapping):
            raise core.M5InfrastructureError(
                "M9 protocol panel case is malformed"
            )
        required_ints = (
            "order",
            "graph_seed",
            "policy_seed",
            "horizon",
            "witness_cap",
        )
        if (
            not isinstance(raw_case.get("case_id"), str)
            or any(
                not isinstance(raw_case.get(key), int)
                or isinstance(raw_case.get(key), bool)
                for key in required_ints
            )
        ):
            raise core.M5InfrastructureError(
                "M9 protocol panel case fields changed"
            )
        raw_lengths = raw_case.get("forbidden_lengths")
        if (
            not isinstance(raw_lengths, Sequence)
            or isinstance(raw_lengths, str | bytes)
            or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in raw_lengths
            )
        ):
            raise core.M5InfrastructureError(
                "M9 protocol forbidden lengths changed"
            )
        panel.append(
            core.DevelopmentCaseV1(
                case_id=str(raw_case["case_id"]),
                order=int(raw_case["order"]),
                graph_seed=int(raw_case["graph_seed"]),
                policy_seed=int(raw_case["policy_seed"]),
                horizon=int(raw_case["horizon"]),
                witness_cap=int(raw_case["witness_cap"]),
                forbidden_lengths=tuple(cast(Sequence[int], raw_lengths)),
            )
        )
    result = tuple(panel)
    if (
        not result
        or protocol.get("panel_hash") != core.panel_hash(result)
    ):
        raise core.M5InfrastructureError("M9 protocol panel identity changed")
    return result


def finalize_budget_limited_search(
    *,
    workspace: str | Path,
    options: ScientificSearchOptionsV1,
) -> dict[str, Any] | None:
    """Finalize retained M9 evidence after the provider budget is exhausted."""

    root = Path(workspace)
    report_path = root / M9_REPORT_FILENAME
    if report_path.is_file():
        report = core._load_mapping(report_path)
        if report.get("protocol_id") != M9_REPORT_PROTOCOL_ID:
            raise core.M5InfrastructureError("M9 report identity changed")
        return report
    runtime_path = root / M9_RUNTIME_FILENAME
    if not runtime_path.is_file():
        return None
    runtime = core._load_mapping(runtime_path)
    submitted = runtime.get("provider_turns_submitted")
    if (
        not isinstance(submitted, int)
        or isinstance(submitted, bool)
        or submitted < options.provider_program_turn_limit
    ):
        return None
    if submitted != options.provider_program_turn_limit:
        raise core.M5InfrastructureError(
            "M9 provider turn accounting exceeded its budget"
        )
    keys = runtime.get("provider_turn_keys")
    if (
        not isinstance(keys, Sequence)
        or isinstance(keys, str | bytes)
        or len(keys) != submitted
        or not all(isinstance(item, str) and item for item in keys)
        or len(set(cast(Sequence[str], keys))) != submitted
    ):
        raise core.M5InfrastructureError(
            "M9 provider reservation evidence changed"
        )
    protocol = core._load_mapping(root / "protocol.json.gz")
    expected_protocol = {
        "protocol_id": M9_SEARCH_PROTOCOL_ID,
        "generation_limit": options.generation_limit,
        "provider_concurrency": options.provider_concurrency,
        "evaluator_workers": options.evaluator_workers,
        "wall_seconds": options.wall_seconds,
        "provider_program_turn_limit": options.provider_program_turn_limit,
        "replace_terminal_slots": options.replace_terminal_slots,
        "resume_enabled": options.resume_enabled,
        "stop_on_verified": options.stop_on_verified,
        "native_v2_default": True,
        "dsl_runtime_used": False,
    }
    if any(
        protocol.get(key) != value
        for key, value in expected_protocol.items()
    ):
        raise core.M5InfrastructureError("M9 protocol options changed")
    provider_model = protocol.get("model")
    provider_effort = protocol.get("effort")
    if not isinstance(provider_model, str) or not isinstance(
        provider_effort, str
    ):
        raise core.M5InfrastructureError(
            "M9 provider accounting identity changed"
        )
    provenance = core._load_mapping(
        root / "acceptance-provenance.json.gz"
    )
    if not provenance:
        raise core.M5InfrastructureError(
            "M9 acceptance provenance is unavailable"
        )
    anchor = core.M5ProviderResultV1.from_dict(
        core._load_mapping(root / "anchor.json.gz")
    )
    panel = _panel_from_protocol(protocol)
    manifest_paths = sorted(
        root.glob("generations/generation-*/manifest.json.gz")
    )
    generations = [
        int(path.parent.name.removeprefix("generation-"))
        for path in manifest_paths
    ]
    if not generations or generations != list(range(len(generations))):
        raise core.M5InfrastructureError(
            "M9 retained generations are not contiguous"
        )
    missing_memory_generations: list[int] = []
    for generation, manifest_path in zip(
        generations, manifest_paths, strict=True
    ):
        previous = (
            core._generation_candidates(root, generation - 1)
            if generation > 0
            else []
        )
        manifest = core.build_generation_manifest(
            generation=generation,
            panel=panel,
            previous_candidates=previous,
        )
        if core._load_mapping(manifest_path) != manifest.as_dict():
            raise core.M5InfrastructureError(
                "M9 retained generation manifest changed"
            )
        generation_dir = manifest_path.parent
        memory_path = generation_dir / "search-memory.json.gz"
        if memory_path.is_file():
            memory_candidates = [
                item
                for item in core._all_candidates(root)
                if int(item.get("generation", -1)) < generation
            ]
            memory = core.build_search_memory(memory_candidates)
            if core._load_mapping(memory_path) != memory:
                raise core.M5InfrastructureError(
                    "M9 retained Search Memory changed"
                )
            memory_sha256 = str(memory["sha256"])
        else:
            missing_memory_generations.append(generation)
            if generation != generations[-1]:
                raise core.M5InfrastructureError(
                    "M9 retained Search Memory is missing"
                )
            memory_sha256 = ""
        for slot_plan in manifest.slots:
            slot_dir = generation_dir / slot_plan.slot
            candidate_path = slot_dir / "candidate.json.gz"
            prepared_path = slot_dir / M9_PREPARED_FILENAME
            if candidate_path.is_file():
                if not memory_sha256:
                    raise core.M5InfrastructureError(
                        "M9 candidate has no retained Search Memory"
                    )
                core._verify_retained_candidate(
                    root=root,
                    path=candidate_path,
                    panel=panel,
                    slot_plan=slot_plan,
                    search_memory_sha256=memory_sha256,
                )
            elif prepared_path.is_file():
                raise core.M5InfrastructureError(
                    "M9 budget finalization found a prepared candidate"
                )
    candidates = core._all_candidates(root)
    attempts = sum(
        len(cast(Sequence[object], item.get("provider_attempts", ())))
        for item in candidates
    )
    provider_results = list(
        root.glob(
            "generations/generation-*/slot-*/provider-*/"
            "m5-provider-result.json.gz"
        )
    )
    if attempts != submitted or len(provider_results) != submitted:
        raise core.M5InfrastructureError(
            "M9 retained provider evidence is incomplete"
        )
    runtime_payload = {
        **runtime,
        "active_evaluators": 0,
        "queued_evaluations": 0,
        "terminal_reason": "provider_turn_budget",
        "updated_epoch_seconds": time.time(),
    }
    write_json(runtime_path, runtime_payload)
    report = _report(
        root=root,
        panel=panel,
        provider_model=provider_model,
        provider_effort=provider_effort,
        anchor_result=anchor,
        options=options,
        runtime=runtime_payload,
        stop_reason="provider_turn_budget",
    )
    report["recovery"] = {
        "offline_finalized": True,
        "provider_or_backend_started": False,
        "missing_search_memory_generations": (
            missing_memory_generations
        ),
    }
    _write_or_verify(report_path, report)
    return report


def run_sustained_search(
    *,
    provider: core.M5SearchProvider,
    evaluator_factory: Callable[[], core.M5ScientificEvaluator],
    workspace: str | Path,
    panel: tuple[core.DevelopmentCaseV1, ...],
    system_prompt: str,
    specification_prompt: str,
    specification_ack_schema: Mapping[str, Any],
    policy_schema: Mapping[str, Any],
    options: ScientificSearchOptionsV1,
    provider_turn_timeout_seconds: float,
    operator_stop: Callable[[], bool] | None = None,
    boundary_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run or resume one bounded, provider-overlapped scientific campaign."""

    if not panel:
        raise ValueError("development panel must not be empty")
    if not 0 < provider_turn_timeout_seconds < options.wall_seconds:
        raise ValueError(
            "provider turn timeout must be positive and below the wall budget"
        )
    core._assert_model_prompt_hygiene(system_prompt)
    core._assert_model_prompt_hygiene(specification_prompt)
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol_id": M9_SEARCH_PROTOCOL_ID,
        "population_size": core.POPULATION_SIZE,
        "generation_limit": options.generation_limit,
        "generation_zero_roots": core.POPULATION_SIZE,
        "later_child_slots": core.CHILD_SLOTS,
        "later_root_slots": core.ROOT_SLOTS,
        "panel_hash": core.panel_hash(panel),
        "panel": [item.as_dict() for item in panel],
        "model": provider.model,
        "effort": provider.effort,
        "provider_concurrency": options.provider_concurrency,
        "evaluator_workers": options.evaluator_workers,
        "wall_seconds": options.wall_seconds,
        "provider_program_turn_limit": options.provider_program_turn_limit,
        "provider_turn_timeout_seconds": provider_turn_timeout_seconds,
        "max_consecutive_provider_failures": (
            M9_MAX_CONSECUTIVE_PROVIDER_FAILURES
        ),
        "replace_terminal_slots": options.replace_terminal_slots,
        "resume_enabled": options.resume_enabled,
        "stop_on_verified": options.stop_on_verified,
        "safe_api_expanded": True,
        "preview_active": True,
        "native_v2_default": True,
        "dsl_runtime_used": False,
    }
    _write_or_verify(root / "protocol.json.gz", protocol)
    if boundary_hook is not None:
        boundary_hook("protocol_persisted")
    telemetry = _RuntimeTelemetry(root, options)
    telemetry.clear_terminal_reason()
    pool = _ConcurrentEvaluatorPool(
        workers=options.evaluator_workers,
        evaluator_factory=evaluator_factory,
        telemetry=telemetry,
    )
    stop_reason = "generation_budget"
    anchor_result: core.M5ProviderResultV1 | None = None
    close_error: Exception | None = None
    try:
        anchor_result = provider.ensure_specification_anchor(
            prompt=specification_prompt,
            system_prompt=system_prompt,
            output_schema=specification_ack_schema,
            artifact_dir=root / "provider" / "specification-anchor",
        )
        core._assert_provider_turn_boundary(anchor_result, expected_history=())
        anchor = anchor_result.context
        core._write_exclusive_or_verify(
            root / "anchor.json.gz",
            anchor_result.as_dict(),
        )
        if boundary_hook is not None:
            boundary_hook("anchor_persisted")
        halt = False
        exact_verified = False
        consecutive_provider_failures = 0
        for generation in range(options.generation_limit):
            previous = (
                core._generation_candidates(root, generation - 1)
                if generation > 0
                else []
            )
            manifest = core.build_generation_manifest(
                generation=generation,
                panel=panel,
                previous_candidates=previous,
            )
            generation_dir = (
                root / "generations" / f"generation-{generation:04d}"
            )
            core._write_exclusive_or_verify(
                generation_dir / "manifest.json.gz",
                manifest.as_dict(),
            )
            if boundary_hook is not None:
                boundary_hook(f"generation_{generation}_manifest")
            memory_candidates = [
                item
                for item in core._all_candidates(root)
                if int(item.get("generation", -1)) < generation
            ]
            memory = core.build_search_memory(memory_candidates)
            core._write_exclusive_or_verify(
                generation_dir / "search-memory.json.gz",
                memory,
            )
            if boundary_hook is not None:
                boundary_hook(f"generation_{generation}_search_memory")
            prior_candidates = core._all_candidates(root)
            by_id = {
                str(item["candidate_id"]): item for item in prior_candidates
            }
            entries = [
                _PendingCommit(
                    slot_plan=slot_plan,
                    candidate_id=core._candidate_id(
                        generation,
                        slot_plan.slot,
                    ),
                    slot_dir=core._candidate_path(
                        root,
                        generation,
                        slot_plan.slot,
                    ),
                )
                for slot_plan in manifest.slots
            ]
            commit_cursor = 0

            def commit_ready(
                *,
                block: bool,
                generation_entries: list[_PendingCommit] = entries,
            ) -> None:
                nonlocal commit_cursor, exact_verified
                while commit_cursor < len(generation_entries):
                    outcome = _commit_pending(
                        pending=generation_entries[commit_cursor],
                        root=root,
                        panel=panel,
                        telemetry=telemetry,
                        block=block,
                        boundary_hook=boundary_hook,
                    )
                    if outcome is None:
                        return
                    verified, infrastructure_error = outcome
                    commit_cursor += 1
                    exact_verified |= verified
                    if infrastructure_error is not None:
                        raise infrastructure_error

            for index, (slot_plan, pending) in enumerate(
                zip(manifest.slots, entries, strict=True)
            ):
                candidate_path = pending.slot_dir / "candidate.json.gz"
                prepared_path = _prepared_path(pending.slot_dir)
                if candidate_path.is_file():
                    retained_candidate = core._verify_retained_candidate(
                        root=root,
                        path=candidate_path,
                        panel=panel,
                        slot_plan=slot_plan,
                        search_memory_sha256=str(memory["sha256"]),
                    )
                    exact_verified |= (
                        retained_candidate.get("exact_verified") is True
                    )
                    pending.retained_terminal = True
                    commit_ready(block=False)
                    if exact_verified:
                        stop_reason = "exact_verified_counterexample"
                        halt = True
                        break
                    continue
                if prepared_path.is_file():
                    prepared = _verify_prepared(
                        root=root,
                        path=prepared_path,
                        panel=panel,
                        slot_plan=slot_plan,
                        search_memory_sha256=str(memory["sha256"]),
                    )
                    pending.prepared = prepared
                    pending.future = pool.submit(
                        source=str(prepared["source"]),
                        panel=panel,
                        candidate_id=pending.candidate_id,
                        slot_dir=pending.slot_dir,
                    )
                    commit_ready(block=False)
                    continue
                if operator_stop is not None and operator_stop():
                    commit_ready(block=True)
                    stop_reason = "operator_stop"
                    raise core.M5OperatorStop("operator stop requested")
                if telemetry.wall_expired(options.wall_seconds):
                    stop_reason = "wall_clock_budget"
                    halt = True
                    break
                if (
                    telemetry.wall_remaining(options.wall_seconds)
                    < provider_turn_timeout_seconds
                ):
                    stop_reason = "wall_clock_budget"
                    halt = True
                    break
                initial_key = f"{slot_plan.request_key}-initial"
                if not telemetry.provider_turn_available(
                    options.provider_program_turn_limit,
                    key=initial_key,
                ):
                    stop_reason = "provider_turn_budget"
                    halt = True
                    break
                parent: Mapping[str, Any] | None = None
                prompt: str
                provider_result: core.M5ProviderResultV1 | None = None
                attempts: list[Mapping[str, JsonValue]] = []
                prompt = ""
                try:
                    if slot_plan.kind == "root":
                        prompt = core.build_root_prompt(memory)
                        core._assert_model_prompt_hygiene(prompt)
                        provider_result = _provider_call(
                            telemetry=telemetry,
                            options=options,
                            key=initial_key,
                            durable_result_path=(
                                pending.slot_dir
                                / "provider-initial"
                                / "m5-provider-result.json.gz"
                            ),
                            operation=partial(
                                provider.generate_root,
                                anchor=anchor,
                                generation=generation,
                                slot=slot_plan.slot,
                                prompt=prompt,
                                system_prompt=system_prompt,
                                output_schema=policy_schema,
                                idempotency_key=slot_plan.request_key,
                                artifact_dir=(
                                    pending.slot_dir / "provider-initial"
                                ),
                            ),
                        )
                        core._assert_provider_turn_boundary(
                            provider_result,
                            expected_history=anchor.included_turn_ids,
                        )
                    else:
                        parent_id = slot_plan.parent_candidate_id
                        if parent_id is None or parent_id not in by_id:
                            raise core.M5InfrastructureError(
                                "frozen child parent is unavailable"
                            )
                        parent = by_id[parent_id]
                        parent_source = parent.get("source")
                        parent_profile = parent.get("behavior_profile")
                        if not isinstance(parent_source, str) or not isinstance(
                            parent_profile,
                            Mapping,
                        ):
                            raise core.M5InfrastructureError(
                                "selected parent evidence is incomplete"
                            )
                        prompt = core.build_child_prompt(
                            parent_source=parent_source,
                            parent_profile=parent_profile,
                        )
                        core._assert_model_prompt_hygiene(prompt)
                        parent_context = core._provider_context(parent)
                        provider_result = _provider_call(
                            telemetry=telemetry,
                            options=options,
                            key=initial_key,
                            durable_result_path=(
                                pending.slot_dir
                                / "provider-initial"
                                / "m5-provider-result.json.gz"
                            ),
                            operation=partial(
                                provider.generate_child,
                                parent=parent_context,
                                generation=generation,
                                slot=slot_plan.slot,
                                prompt=prompt,
                                system_prompt=system_prompt,
                                output_schema=policy_schema,
                                idempotency_key=slot_plan.request_key,
                                artifact_dir=(
                                    pending.slot_dir / "provider-initial"
                                ),
                            ),
                        )
                        core._assert_provider_turn_boundary(
                            provider_result,
                            expected_history=parent_context.included_turn_ids,
                        )
                except Exception as error:
                    if isinstance(error, _ProviderTurnBudgetExhausted):
                        commit_ready(block=True)
                        raise
                    if isinstance(error, core.M5InfrastructureError):
                        raise
                    base = _candidate_base(
                        candidate_id=pending.candidate_id,
                        generation=generation,
                        slot_plan=slot_plan,
                        parent=parent,
                        panel=panel,
                        memory=memory,
                        provider_result=None,
                        attempts=(),
                        repairs=0,
                        prompt=prompt,
                    )
                    pending.terminal_payload = _provider_failure(
                        base=base,
                        error=error,
                    )
                    commit_ready(block=False)
                    consecutive_provider_failures += 1
                    if (
                        consecutive_provider_failures
                        >= M9_MAX_CONSECUTIVE_PROVIDER_FAILURES
                    ):
                        commit_ready(block=True)
                        raise core.M5InfrastructureError(
                            "provider unavailable after three consecutive "
                            "terminal slot failures"
                        ) from error
                    continue
                if boundary_hook is not None:
                    boundary_hook(f"{pending.candidate_id}_provider")
                attempts.append(provider_result.as_dict())
                validation = validate_python_policy_response(
                    provider_result.response_text
                )
                repairs = 0
                if not validation.valid:
                    repair_key = f"{slot_plan.request_key}-repair-01"
                    if telemetry.provider_turn_available(
                        options.provider_program_turn_limit,
                        key=repair_key,
                    ) and (
                        telemetry.wall_remaining(options.wall_seconds)
                        >= provider_turn_timeout_seconds
                    ):
                        previous_result = provider_result
                        repair_prompt = core.build_repair_prompt(
                            [
                                item.as_dict()
                                for item in validation.diagnostics[:32]
                            ]
                        )
                        core._assert_model_prompt_hygiene(repair_prompt)
                        try:
                            provider_result = _provider_call(
                                telemetry=telemetry,
                                options=options,
                                key=repair_key,
                                durable_result_path=(
                                    pending.slot_dir
                                    / "provider-repair-01"
                                    / "m5-provider-result.json.gz"
                                ),
                                operation=partial(
                                    provider.repair,
                                    previous=previous_result,
                                    generation=generation,
                                    slot=slot_plan.slot,
                                    prompt=repair_prompt,
                                    system_prompt=system_prompt,
                                    output_schema=policy_schema,
                                    idempotency_key=(
                                        slot_plan.request_key + "-repair-01"
                                    ),
                                    artifact_dir=(
                                        pending.slot_dir
                                        / "provider-repair-01"
                                    ),
                                ),
                            )
                            core._assert_provider_turn_boundary(
                                provider_result,
                                expected_history=(
                                    previous_result.context.included_turn_ids
                                ),
                                expected_thread_id=(
                                    previous_result.context.thread_id
                                ),
                            )
                        except Exception as error:
                            if isinstance(
                                error, _ProviderTurnBudgetExhausted
                            ):
                                commit_ready(block=True)
                                raise
                            if isinstance(error, core.M5InfrastructureError):
                                raise
                            base = _candidate_base(
                                candidate_id=pending.candidate_id,
                                generation=generation,
                                slot_plan=slot_plan,
                                parent=parent,
                                panel=panel,
                                memory=memory,
                                provider_result=previous_result,
                                attempts=attempts,
                                repairs=1,
                                prompt=prompt,
                            )
                            pending.terminal_payload = _provider_failure(
                                base=base,
                                error=error,
                            )
                            commit_ready(block=False)
                            consecutive_provider_failures += 1
                            if (
                                consecutive_provider_failures
                                >= M9_MAX_CONSECUTIVE_PROVIDER_FAILURES
                            ):
                                commit_ready(block=True)
                                raise core.M5InfrastructureError(
                                    "provider unavailable after three "
                                    "consecutive terminal slot failures"
                                ) from error
                            continue
                        attempts.append(provider_result.as_dict())
                        validation = validate_python_policy_response(
                            provider_result.response_text
                        )
                        repairs = 1
                base = _candidate_base(
                    candidate_id=pending.candidate_id,
                    generation=generation,
                    slot_plan=slot_plan,
                    parent=parent,
                    panel=panel,
                    memory=memory,
                    provider_result=provider_result,
                    attempts=attempts,
                    repairs=repairs,
                    prompt=prompt,
                )
                consecutive_provider_failures = 0
                if (
                    not validation.valid
                    or validation.response is None
                    or validation.identity is None
                    or validation.identity.program_hash is None
                ):
                    skipped = None
                    if repairs == 0:
                        skipped = (
                            "wall_clock_budget"
                            if telemetry.wall_expired(options.wall_seconds)
                            else "provider_turn_budget"
                        )
                    pending.terminal_payload = _contract_invalid(
                        base=base,
                        validation=validation.as_dict(),
                        repair_skipped=skipped,
                    )
                    commit_ready(block=False)
                    continue
                source = normalize_source_newlines(
                    validation.response.source
                )
                program_hash = validation.identity.program_hash
                source_path = root / "sources" / f"{program_hash}.py"
                telemetry.timed_persist(
                    partial(
                        core._write_source_exclusive_or_verify,
                        source_path,
                        source,
                    )
                )
                if boundary_hook is not None:
                    boundary_hook(f"{pending.candidate_id}_source_persisted")
                canonical_ast_sha256 = (
                    validation.identity.canonical_ast_sha256
                )
                if not isinstance(canonical_ast_sha256, str):
                    raise core.M5InfrastructureError(
                        "validated program omitted canonical AST identity"
                    )
                prepared = _prepared_candidate(
                    base=base,
                    validation=validation.as_dict(),
                    source=source,
                    source_path=source_path,
                    root=root,
                    program_hash=program_hash,
                    source_sha256=validation.identity.source_sha256,
                    canonical_ast_sha256=canonical_ast_sha256,
                )
                telemetry.timed_persist(
                    partial(_write_or_verify, prepared_path, prepared)
                )
                telemetry.first_valid_program()
                pending.prepared = prepared
                pending.future = pool.submit(
                    source=source,
                    panel=panel,
                    candidate_id=pending.candidate_id,
                    slot_dir=pending.slot_dir,
                )
                if boundary_hook is not None:
                    boundary_hook(f"{pending.candidate_id}_evaluation_queued")
                commit_ready(block=False)
                if exact_verified:
                    stop_reason = "exact_verified_counterexample"
                    halt = True
                    break
                if index == len(entries) - 1:
                    commit_ready(block=True)
            commit_ready(block=True)
            if exact_verified:
                stop_reason = "exact_verified_counterexample"
                halt = True
            if halt:
                break
        telemetry.boundary("report_pending")
    except _ProviderTurnBudgetExhausted:
        stop_reason = "provider_turn_budget"
        telemetry.boundary("provider_turn_budget_exhausted")
    except (core.M5InfrastructureError, core.M5OperatorStop) as error:
        reason = (
            "operator_stop"
            if isinstance(error, core.M5OperatorStop)
            else "infrastructure_failure"
        )
        telemetry.finish(reason)
        write_json(
            root / M9_STOP_FILENAME,
            {
                "protocol_id": M9_REPORT_PROTOCOL_ID,
                "status": reason,
                "error_type": type(error).__name__,
                "error": str(error)[:2048],
                "candidate_count": len(core._all_candidates(root)),
                "panel_hash": core.panel_hash(panel),
                "resumable": True,
            },
        )
        raise
    finally:
        close_error = pool.close()
    if close_error is not None:
        telemetry.finish("infrastructure_failure")
        cleanup_failure = core.M5InfrastructureError(
            f"evaluator cleanup failed: {type(close_error).__name__}"
        )
        write_json(
            root / M9_STOP_FILENAME,
            {
                "protocol_id": M9_REPORT_PROTOCOL_ID,
                "status": "infrastructure_failure",
                "error_type": type(cleanup_failure).__name__,
                "error": str(cleanup_failure),
                "candidate_count": len(core._all_candidates(root)),
                "panel_hash": core.panel_hash(panel),
                "resumable": True,
            },
        )
        raise cleanup_failure from close_error
    if anchor_result is None:
        raise core.M5InfrastructureError("M9 specification anchor is missing")
    telemetry.finish(stop_reason)
    report = _report(
        root=root,
        panel=panel,
        provider_model=provider.model,
        provider_effort=provider.effort,
        anchor_result=anchor_result,
        options=options,
        runtime=telemetry.snapshot(),
        stop_reason=stop_reason,
    )
    write_json(root / M9_REPORT_FILENAME, report)
    if boundary_hook is not None:
        boundary_hook("report_persisted")
    return report


__all__ = [
    "M9_REPORT_FILENAME",
    "M9_REPORT_PROTOCOL_ID",
    "M9_RUNTIME_FILENAME",
    "M9_SEARCH_PROTOCOL_ID",
    "M9_STOP_FILENAME",
    "ScientificSearchOptionsV1",
    "finalize_budget_limited_search",
    "run_sustained_search",
]
