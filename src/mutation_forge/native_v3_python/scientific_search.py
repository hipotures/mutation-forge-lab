"""Bounded sustained ordinary-Python search with concurrent supply and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from functools import partial
from itertools import count
from pathlib import Path
from typing import Any, Protocol, cast

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3.canonical import canonical_json_bytes

from . import search as core
from .scientific_evaluation import (
    ScientificEvaluationOptionsV1,
    workload_projection,
)
from .validation import normalize_source_newlines, validate_python_policy_response

M10_SEARCH_PROTOCOL_ID = "mforge.native.python_scientific_search.v2"
M10_PREPARED_CANDIDATE_PROTOCOL_ID = "mforge.native.python_scientific_search_prepared_candidate.v2"
M10_RUNTIME_PROTOCOL_ID = "mforge.native.python_scientific_search_runtime.v2"
M10_REPORT_PROTOCOL_ID = "mforge.native.python_scientific_search_report.v2"
M10_RUNTIME_FILENAME = "m10-runtime.json.gz"
M10_REPORT_FILENAME = "m10-report.json.gz"
M10_STOP_FILENAME = "m10-stop.json.gz"
M10_PREPARED_FILENAME = "prepared-candidate.json.gz"
M10_BASELINE_FILENAME = "generation-baselines.json.gz"
M10_BASELINE_PROTOCOL_ID = "mforge.native.python_generation_baselines.v1"
M10_BASELINE_RESULT_FILENAME = "baseline-result.json.gz"
M10_BASELINE_RESULT_PROTOCOL_ID = "mforge.native.python_generation_baseline.v1"


class _ProviderTurnBudgetExhausted(core.M5SearchError):
    """The durable M10 provider reservation budget is terminally exhausted."""


class M10ScientificEvaluator(core.M5ScientificEvaluator, Protocol):
    def evaluate_baseline(
        self,
        *,
        baseline: str,
        case: core.DevelopmentCaseV1,
        generation: int,
    ) -> Mapping[str, JsonValue]: ...


@dataclass(frozen=True, slots=True)
class ScientificSearchOptionsV2:
    """Immutable host-owned controls for one sustained campaign."""

    generation_limit: int | None
    evaluator_workers: int
    provider_concurrency: int
    wall_seconds: float | None
    primary_program_slots: int | None
    repair_turn_limit: int | None
    provider_total_turn_limit: int | None
    validated_queue_target: int
    validated_queue_capacity: int
    stop_on_verified: bool
    resume_enabled: bool
    replace_terminal_slots: bool
    evaluation: ScientificEvaluationOptionsV1
    max_total_tokens_per_hour: int | None = None

    def __post_init__(self) -> None:
        if self.generation_limit is not None and self.generation_limit < 1:
            raise ValueError("generation_limit must be positive when configured")
        if not 1 <= self.evaluator_workers <= 12:
            raise ValueError("evaluator_workers must be between 1 and 12")
        if not 1 <= self.provider_concurrency <= 4:
            raise ValueError("provider_concurrency must be between 1 and 4")
        if self.wall_seconds is not None and self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive when configured")
        if self.max_total_tokens_per_hour is not None and self.max_total_tokens_per_hour < 1:
            raise ValueError("max_total_tokens_per_hour must be positive when configured")
        if (
            self.generation_limit is not None
            and self.primary_program_slots != self.generation_limit * core.POPULATION_SIZE
        ):
            raise ValueError(
                "primary_program_slots must provide exactly eight slots per generation"
            )
        if self.generation_limit is None and self.primary_program_slots is not None:
            raise ValueError("primary_program_slots must be omitted for unlimited generations")
        if self.repair_turn_limit is not None and self.repair_turn_limit < 0:
            raise ValueError("repair_turn_limit cannot be negative")
        if (
            self.primary_program_slots is not None
            and self.repair_turn_limit is not None
            and self.provider_total_turn_limit
            != self.primary_program_slots + self.repair_turn_limit
        ):
            raise ValueError("provider_total_turn_limit must equal primary slots plus repairs")
        if (
            self.primary_program_slots is None or self.repair_turn_limit is None
        ) and self.provider_total_turn_limit is not None:
            raise ValueError(
                "provider_total_turn_limit must be omitted when either provider budget is unlimited"
            )
        if self.validated_queue_target != 2 * self.evaluator_workers:
            raise ValueError("validated_queue_target must equal twice evaluator_workers")
        if self.validated_queue_capacity != 4 * self.evaluator_workers:
            raise ValueError("validated_queue_capacity must equal four times evaluator_workers")
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
            "max_total_tokens_per_hour": self.max_total_tokens_per_hour,
            "primary_program_slots": self.primary_program_slots,
            "repair_turn_limit": self.repair_turn_limit,
            "provider_total_turn_limit": self.provider_total_turn_limit,
            "validated_queue_target": self.validated_queue_target,
            "validated_queue_capacity": self.validated_queue_capacity,
            "stop_on_verified": self.stop_on_verified,
            "resume_enabled": self.resume_enabled,
            "replace_terminal_slots": self.replace_terminal_slots,
            "evaluation": self.evaluation.as_dict(),
        }

    def scientific_identity_dict(self) -> dict[str, JsonValue]:
        """Return experiment semantics without per-invocation execution controls."""

        return {
            "generation_limit": self.generation_limit,
            "primary_program_slots": self.primary_program_slots,
            "repair_turn_limit": self.repair_turn_limit,
            "provider_total_turn_limit": self.provider_total_turn_limit,
            "stop_on_verified": self.stop_on_verified,
            "resume_enabled": self.resume_enabled,
            "replace_terminal_slots": self.replace_terminal_slots,
            "evaluation": self.evaluation.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScientificResumeBudgetV1:
    """Transient provider budget for one incomplete generation resume."""

    expected_pending_primary_slots: int
    max_new_repair_turns: int

    def __post_init__(self) -> None:
        if not 1 <= self.expected_pending_primary_slots <= core.POPULATION_SIZE:
            raise ValueError("expected_pending_primary_slots must be between 1 and 8")
        if not 0 <= self.max_new_repair_turns <= core.POPULATION_SIZE:
            raise ValueError("max_new_repair_turns must be between 0 and 8")


def hourly_token_usage(
    root: Path,
    limit: int | None,
    *,
    now: datetime | None = None,
) -> dict[str, JsonValue]:
    """Return rolling one-hour usage from durable provider results."""

    current = now or datetime.now(UTC)
    current = current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)
    cutoff = current - timedelta(hours=1)
    runtime_path = root / M10_RUNTIME_FILENAME
    if runtime_path.is_file():
        runtime = read_json(runtime_path)
        started = (
            runtime.get("campaign_started_epoch_seconds") if isinstance(runtime, Mapping) else None
        )
        if isinstance(started, int | float) and not isinstance(started, bool):
            cutoff = max(
                cutoff,
                datetime.fromtimestamp(float(started), UTC),
            )
    charges: list[tuple[datetime, int]] = []
    for path in root.glob("**/m5-provider-result.json.gz"):
        charged_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if not cutoff < charged_at <= current:
            continue
        payload = read_json(path)
        usage = payload.get("usage") if isinstance(payload, Mapping) else None
        tokens = usage.get("totalTokens") if isinstance(usage, Mapping) else None
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
            charges.append((charged_at, tokens))
    charges.sort()
    used = sum(tokens for _, tokens in charges)
    reached = limit is not None and used >= limit
    retry_after: str | None = None
    if reached and limit is not None:
        remaining = used
        for charged_at, tokens in charges:
            remaining -= tokens
            if remaining < limit:
                retry_after = (charged_at + timedelta(hours=1)).isoformat()
                break
    return {
        "hourly_token_limit": limit,
        "hourly_tokens_used": used,
        "hourly_tokens_remaining": (None if limit is None else max(0, limit - used)),
        "hourly_window_seconds": 3600,
        "hourly_limit_reached": reached,
        "hourly_retry_after": retry_after,
    }


@dataclass(frozen=True, slots=True)
class GenerationSnapshotV1:
    """Immutable model-facing and host scheduling identity for one generation."""

    generation: int
    slots: tuple[Mapping[str, JsonValue], ...]
    search_memory_projection: Mapping[str, JsonValue]
    model: str
    effort: str
    prompt_versions: Mapping[str, JsonValue]
    output_schema: Mapping[str, JsonValue]
    panel: tuple[Mapping[str, JsonValue], ...]
    budgets: Mapping[str, JsonValue]

    def as_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "schema_version": "mforge.native.python_generation_snapshot.v1",
            "generation": self.generation,
            "slots": cast(JsonValue, [dict(slot) for slot in self.slots]),
            "search_memory_projection": cast(JsonValue, dict(self.search_memory_projection)),
            "model": self.model,
            "effort": self.effort,
            "prompt_versions": cast(JsonValue, dict(self.prompt_versions)),
            "output_schema": cast(JsonValue, dict(self.output_schema)),
            "panel": cast(JsonValue, [dict(case) for case in self.panel]),
            "budgets": cast(JsonValue, dict(self.budgets)),
        }
        payload["sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        return payload


def _generation_snapshot(
    *,
    generation: int,
    manifest: core.GenerationManifestV1,
    memory: Mapping[str, Any],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    anchor: core.M5ProviderContextV1,
    model: str,
    effort: str,
    system_prompt: str,
    policy_schema: Mapping[str, Any],
    panel: tuple[core.DevelopmentCaseV1, ...],
    options: ScientificSearchOptionsV2,
) -> dict[str, JsonValue]:
    model_projection = memory.get("model_projection")
    if not isinstance(model_projection, Mapping):
        raise core.M5InfrastructureError("generation Search Memory has no model projection")
    slots: list[Mapping[str, JsonValue]] = []
    for slot_plan in manifest.slots:
        parent_turn_id: str | None = anchor.turn_id
        parent_thread_id: str | None = anchor.thread_id
        parent_turn_ids: list[str] = list(anchor.included_turn_ids)
        if slot_plan.parent_candidate_id is not None:
            parent = candidates_by_id.get(slot_plan.parent_candidate_id)
            if parent is None:
                raise core.M5InfrastructureError("generation snapshot parent is unavailable")
            context = core._provider_context(parent)
            parent_turn_id = context.turn_id
            parent_thread_id = context.thread_id
            parent_turn_ids = list(context.included_turn_ids)
        slots.append(
            {
                "candidate_id": core._candidate_id(generation, slot_plan.slot),
                "slot": slot_plan.slot,
                "kind": slot_plan.kind,
                "parent_candidate_id": slot_plan.parent_candidate_id,
                "parent_thread_id": parent_thread_id,
                "parent_turn_id": parent_turn_id,
                "parent_included_turn_ids": cast(JsonValue, parent_turn_ids),
                "request_key": slot_plan.request_key,
            }
        )
    snapshot = GenerationSnapshotV1(
        generation=generation,
        slots=tuple(slots),
        search_memory_projection=cast(Mapping[str, JsonValue], model_projection),
        model=model,
        effort=effort,
        prompt_versions={
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "root": "mforge.native.python_root_prompt.v2",
            "child": "mforge.native.python_child_prompt.v2",
            "repair": "mforge.native.python_repair_prompt.v2",
        },
        output_schema=cast(Mapping[str, JsonValue], policy_schema),
        panel=tuple(case.as_dict() for case in panel),
        budgets={
            "generation_limit": options.generation_limit,
            "primary_program_slots": options.primary_program_slots,
            "repair_turn_limit": options.repair_turn_limit,
            "provider_total_turn_limit": options.provider_total_turn_limit,
            "stop_on_verified": options.stop_on_verified,
            "resume_enabled": options.resume_enabled,
            "replace_terminal_slots": options.replace_terminal_slots,
        },
    ).as_dict()
    core._assert_model_prompt_hygiene(
        json.dumps(
            snapshot["search_memory_projection"],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return snapshot


def _snapshot_patterns(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    projection = snapshot.get("search_memory_projection")
    if not isinstance(projection, Mapping):
        raise core.M5SearchError("generation snapshot memory is malformed")
    return {
        "successful_patterns": projection.get("successful_patterns", []),
        "tested_patterns": projection.get("tested_patterns", []),
    }


def _root_prompt(snapshot: Mapping[str, Any]) -> str:
    return (
        "Generate one complete fresh ordinary-Python policy for generation "
        f"{snapshot['generation']} with active_parent=null. Return only the "
        "retained two-field response "
        "envelope. Do not describe or predict fitness.\n\n"
        "Bounded host-derived patterns:\n"
        + json.dumps(
            _snapshot_patterns(snapshot),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _child_prompt(
    *,
    snapshot: Mapping[str, Any],
    parent_source: str,
    parent_profile: Mapping[str, Any],
) -> str:
    return (
        "Return one complete replacement ordinary-Python policy, not a patch. "
        "Mutate the exact parent using the compact host-derived feedback and "
        "bounded patterns below.\n\nExact parent source:\n```python\n"
        + parent_source
        + "\n```\n\nCompact evaluation feedback:\n"
        + json.dumps(
            core._feedback(parent_profile),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n\nBounded host-derived patterns:\n"
        + json.dumps(
            _snapshot_patterns(snapshot),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _repair_prompt(
    *,
    snapshot: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
) -> str:
    bounded = [
        {
            "code": str(item.get("code", "INVALID"))[:128],
            "path": str(item.get("path", "/"))[:512],
            "message": str(item.get("message", "invalid response"))[:512],
            "line": item.get("line") if isinstance(item.get("line"), int) else None,
            "column": (item.get("column") if isinstance(item.get("column"), int) else None),
        }
        for item in diagnostics[:16]
    ]
    return (
        "Return one complete replacement ordinary-Python source in the retained "
        "two-field envelope, not a patch. Repair only these concise host "
        f"validator diagnostics for generation {snapshot['generation']}:\n"
        + json.dumps(
            bounded,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


class _RuntimeTelemetry:
    """Mutable timing/status telemetry excluded from scientific identity."""

    def __init__(
        self,
        root: Path,
        options: ScientificSearchOptionsV2,
        resume_budget: ScientificResumeBudgetV1 | None = None,
    ) -> None:
        self._path = root / M10_RUNTIME_FILENAME
        self._lock = threading.Lock()
        self._resume_started = time.monotonic()
        self._resume_started_epoch = time.time()
        if self._path.is_file():
            raw = read_json(self._path)
            if not isinstance(raw, Mapping):
                raise core.M5InfrastructureError("M10 runtime telemetry is malformed")
            state = dict(raw)
            state["protocol_id"] = M10_RUNTIME_PROTOCOL_ID
            state["options"] = options.as_dict()
            state["resume_attempts"] = int(state.get("resume_attempts", 0)) + 1
            state["active_evaluators"] = 0
            state["queued_evaluations"] = 0
            state["active_provider_turns"] = 0
            state["active_evaluation_work"] = {}
            # Runtime counters describe the current invocation.  Scientific
            # lineage remains retained above, but stale in-flight UI rows must
            # not be presented as work started by this invocation.
            state["evaluation_progress"] = {}
            state["evaluation_cases_completed"] = 0
            state["evaluation_cases_total"] = 0
            state["baseline_evaluation_cases_completed"] = 0
            state["baseline_evaluation_cases_total"] = 0
            state["candidate_evaluation_cases_completed"] = 0
            state["candidate_evaluation_cases_total"] = 0
        else:
            state = {
                "protocol_id": M10_RUNTIME_PROTOCOL_ID,
                "options": options.as_dict(),
                "campaign_started_epoch_seconds": time.time(),
                "active_elapsed_seconds": 0.0,
                "resume_attempts": 0,
                "primary_slot_keys": [],
                "repair_turn_keys": [],
                "provider_started_keys": [],
                "provider_turns_submitted": 0,
                "primary_turns_submitted": 0,
                "repair_turns_submitted": 0,
                "provider_wait_seconds": 0.0,
                "provider_active_wall_seconds": 0.0,
                "active_provider_turns": 0,
                "peak_active_provider_turns": 0,
                "provider_concurrency_timeline": [],
                "resume_started_epoch_seconds": self._resume_started_epoch,
                "persistence_seconds": 0.0,
                "evaluator_busy_seconds": 0.0,
                "evaluator_queue_wait_seconds": 0.0,
                "active_evaluators": 0,
                "peak_active_evaluators": 0,
                "queued_evaluations": 0,
                "peak_queued_evaluations": 0,
                "completed_evaluations": 0,
                "failed_evaluations": 0,
                "evaluator_instances": 0,
                "evaluation_progress": {},
                "active_evaluation_work": {},
                "evaluation_cases_completed": 0,
                "evaluation_cases_total": 0,
                "baseline_evaluation_cases_completed": 0,
                "baseline_evaluation_cases_total": 0,
                "candidate_evaluation_cases_completed": 0,
                "candidate_evaluation_cases_total": 0,
                "first_valid_program_seconds": None,
                "last_scientific_improvement_epoch_seconds": None,
                "best_candidate_id": None,
                "best_fitness_interval": None,
                "last_boundary": None,
                "terminal_reason": None,
            }
        self._state = state
        self._state["resume_started_epoch_seconds"] = self._resume_started_epoch
        self._resume_budget = resume_budget
        current_started = self._string_keys_locked("provider_started_keys")
        current_repairs = self._string_keys_locked("repair_turn_keys")
        if resume_budget is None:
            baseline_started = current_started
            baseline_repairs = current_repairs
        else:
            expected_guard = {
                "protocol_id": "mforge.native.python.resume_budget.v1",
                "expected_pending_primary_slots": (resume_budget.expected_pending_primary_slots),
                "max_new_repair_turns": resume_budget.max_new_repair_turns,
            }
            raw_guard = state.get("resume_budget_guard")
            if raw_guard is None:
                guard: dict[str, Any] = {
                    **expected_guard,
                    "provider_started_baseline": list(current_started),
                    "repair_turn_baseline": list(current_repairs),
                }
                state["resume_budget_guard"] = guard
            elif not isinstance(raw_guard, Mapping):
                raise core.M5InfrastructureError("M10 resume budget guard is malformed")
            else:
                guard = dict(raw_guard)
                if any(guard.get(key) != value for key, value in expected_guard.items()):
                    raise core.M5InfrastructureError(
                        "M10 resume budget guard changed across attempts"
                    )
            guard_started = guard.get("provider_started_baseline")
            guard_repairs = guard.get("repair_turn_baseline")
            if (
                not isinstance(guard_started, list)
                or not all(isinstance(item, str) and item for item in guard_started)
                or not isinstance(guard_repairs, list)
                or not all(isinstance(item, str) and item for item in guard_repairs)
            ):
                raise core.M5InfrastructureError("M10 resume budget baseline is malformed")
            baseline_started = cast(list[str], guard_started)
            baseline_repairs = cast(list[str], guard_repairs)
        self._started_at_resume = frozenset(baseline_started)
        self._repairs_at_resume = frozenset(baseline_repairs)
        self._provider_active_started: float | None = None
        self._persist_locked()

    def _elapsed_locked(self) -> float:
        return float(self._state["active_elapsed_seconds"]) + (
            time.monotonic() - self._resume_started
        )

    def _payload_locked(self) -> dict[str, Any]:
        return {
            **self._state,
            "active_elapsed_seconds": self._elapsed_locked(),
            "current_run_elapsed_seconds": max(
                0.0,
                time.monotonic() - self._resume_started,
            ),
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
            return (time.monotonic() - self._resume_started) >= limit

    def wall_remaining(self, limit: float) -> float:
        with self._lock:
            return max(0.0, limit - (time.monotonic() - self._resume_started))

    def _string_keys_locked(self, field: str) -> list[str]:
        value = self._state.get(field, [])
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise core.M5InfrastructureError(f"M10 {field} reservations are malformed")
        return cast(list[str], value)

    def reserve_primary_generation(
        self,
        keys: Sequence[str],
        *,
        limit: int | None,
    ) -> None:
        if len(keys) != core.POPULATION_SIZE or len(set(keys)) != len(keys):
            raise core.M5InfrastructureError("a generation must reserve eight unique primary slots")
        with self._lock:
            retained = self._string_keys_locked("primary_slot_keys")
            new = [key for key in keys if key not in retained]
            if limit is not None and len(retained) + len(new) > limit:
                raise _ProviderTurnBudgetExhausted("fewer than eight primary program slots remain")
            retained.extend(new)
            self._state["primary_slot_keys"] = retained
            self._persist_locked()

    def reserve_repair(self, key: str, *, limit: int | None) -> bool:
        with self._lock:
            repairs = self._string_keys_locked("repair_turn_keys")
            if key in repairs:
                return True
            if (
                self._resume_budget is not None
                and len(set(repairs).difference(self._repairs_at_resume))
                >= self._resume_budget.max_new_repair_turns
            ):
                return False
            if limit is not None and len(repairs) >= limit:
                return False
            repairs.append(key)
            self._state["repair_turn_keys"] = repairs
            self._persist_locked()
            return True

    def primary_was_started(self, key: str) -> bool:
        with self._lock:
            return key in self._string_keys_locked("provider_started_keys")

    def admit_primary_retry(
        self,
        key: str,
        *,
        limit: int | None,
        durable_result_exists: Callable[[int], bool],
    ) -> tuple[str, int]:
        with self._lock:
            primary = self._string_keys_locked("primary_slot_keys")
            started = self._string_keys_locked("provider_started_keys")
            if key not in primary or key not in started:
                raise core.M5InfrastructureError(
                    "only an interrupted admitted primary turn can be retried"
                )
            for attempt in range(1, core.POPULATION_SIZE + 1):
                retry_key = f"{key}-resume-{attempt:02d}"
                if retry_key in primary:
                    if retry_key not in started or durable_result_exists(attempt):
                        return retry_key, attempt
                    continue
                if retry_key not in primary:
                    if limit is not None and len(primary) >= limit:
                        raise _ProviderTurnBudgetExhausted(
                            "primary retry exceeds the frozen provider budget"
                        )
                    primary.append(retry_key)
                    self._state["primary_slot_keys"] = primary
                    retries = self._state.setdefault(
                        "interrupted_primary_retries",
                        {},
                    )
                    if not isinstance(retries, dict):
                        raise core.M5InfrastructureError(
                            "M10 interrupted primary retry evidence is malformed"
                        )
                    retries[retry_key] = key
                    self._persist_locked()
                    return retry_key, attempt
            raise core.M5InfrastructureError("interrupted primary retry attempts are exhausted")

    def provider_started(self, key: str, *, kind: str) -> bool:
        with self._lock:
            if kind not in {"primary", "repair"}:
                raise ValueError("provider turn kind is invalid")
            admitted = self._string_keys_locked(
                "primary_slot_keys" if kind == "primary" else "repair_turn_keys"
            )
            if key not in admitted:
                raise core.M5InfrastructureError(
                    "provider turn was not admitted by its frozen budget"
                )
            started = self._string_keys_locked("provider_started_keys")
            if key in started:
                return False
            if (
                self._resume_budget is not None
                and kind == "primary"
                and len(set(started).difference(self._started_at_resume).intersection(admitted))
                >= self._resume_budget.expected_pending_primary_slots
            ):
                raise _ProviderTurnBudgetExhausted("resume primary turn budget is exhausted")
            started.append(key)
            self._state["provider_started_keys"] = started
            self._state["provider_turns_submitted"] = len(started)
            count_field = (
                "primary_turns_submitted" if kind == "primary" else "repair_turns_submitted"
            )
            self._state[count_field] = int(self._state[count_field]) + 1
            active = int(self._state["active_provider_turns"]) + 1
            if active == 1:
                self._provider_active_started = time.monotonic()
            self._state["active_provider_turns"] = active
            self._state["peak_active_provider_turns"] = max(
                active,
                int(self._state["peak_active_provider_turns"]),
            )
            timeline = cast(
                list[JsonValue],
                self._state["provider_concurrency_timeline"],
            )
            timeline.append(
                cast(
                    JsonValue,
                    {
                        "key": key,
                        "kind": kind,
                        "started_epoch_seconds": time.time(),
                    },
                )
            )
            self._persist_locked()
            return True

    def provider_finished(
        self,
        elapsed: float,
        *,
        key: str,
        failed: bool,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._state["provider_wait_seconds"] = (
                float(self._state["provider_wait_seconds"]) + elapsed
            )
            active = max(0, int(self._state["active_provider_turns"]) - 1)
            self._state["active_provider_turns"] = active
            if active == 0 and self._provider_active_started is not None:
                self._state["provider_active_wall_seconds"] = (
                    float(self._state["provider_active_wall_seconds"])
                    + time.monotonic()
                    - self._provider_active_started
                )
                self._provider_active_started = None
            timeline = cast(
                list[JsonValue],
                self._state["provider_concurrency_timeline"],
            )
            for item in reversed(timeline):
                if (
                    isinstance(item, dict)
                    and item.get("key") == key
                    and "finished_epoch_seconds" not in item
                ):
                    item["finished_epoch_seconds"] = time.time()
                    item["duration_seconds"] = elapsed
                    item["failed"] = failed
                    if error:
                        item["error"] = error[:1024]
                    break
            self._persist_locked()

    def timed_persist(self, operation: Callable[[], None]) -> None:
        started = time.monotonic()
        operation()
        elapsed = time.monotonic() - started
        with self._lock:
            self._state["persistence_seconds"] = float(self._state["persistence_seconds"]) + elapsed
            self._persist_locked()

    def first_valid_program(self) -> None:
        with self._lock:
            if self._state["first_valid_program_seconds"] is None:
                self._state["first_valid_program_seconds"] = self._elapsed_locked()
                self._persist_locked()

    def evaluator_created(self) -> None:
        with self._lock:
            self._state["evaluator_instances"] = int(self._state["evaluator_instances"]) + 1
            self._persist_locked()

    def evaluator_queued(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("queued evaluator work cannot be negative")
        with self._lock:
            queued = int(self._state["queued_evaluations"]) + count
            self._state["queued_evaluations"] = queued
            self._state["peak_queued_evaluations"] = max(
                queued,
                int(self._state["peak_queued_evaluations"]),
            )
            self._persist_locked()

    def evaluator_started(
        self,
        queue_wait_seconds: float,
        *,
        owner: str,
        case_id: str,
        worker_id: str | None = None,
        process_id: int | None = None,
    ) -> None:
        with self._lock:
            self._state["queued_evaluations"] = max(
                0,
                int(self._state["queued_evaluations"]) - 1,
            )
            active = int(self._state["active_evaluators"]) + 1
            self._state["active_evaluators"] = active
            self._state["evaluator_queue_wait_seconds"] = (
                float(self._state["evaluator_queue_wait_seconds"]) + queue_wait_seconds
            )
            self._state["peak_active_evaluators"] = max(
                active,
                int(self._state["peak_active_evaluators"]),
            )
            active_work = self._state.setdefault("active_evaluation_work", {})
            if not isinstance(active_work, dict):
                raise core.M5InfrastructureError("M10 active evaluation work is malformed")
            identity = worker_id or str(threading.get_native_id())
            active_work[identity] = {
                "owner": owner,
                "case_id": case_id,
                "started_epoch_seconds": time.time(),
                "dispatch_thread_id": threading.get_native_id(),
                "process_id": process_id,
            }
            progress = self._state.setdefault("evaluation_progress", {})
            item = progress.get(owner) if isinstance(progress, dict) else None
            if isinstance(item, dict):
                item["queued"] = max(0, int(item.get("queued", 0)) - 1)
                item["running"] = int(item.get("running", 0)) + 1
                item["state"] = "running"
            self._persist_locked()

    def evaluator_cancelled(self, count: int, *, owner: str) -> None:
        if count < 0:
            raise ValueError("cancelled evaluator work cannot be negative")
        with self._lock:
            self._state["queued_evaluations"] = max(
                0,
                int(self._state["queued_evaluations"]) - count,
            )
            progress = self._state.setdefault("evaluation_progress", {})
            item = progress.get(owner) if isinstance(progress, dict) else None
            if isinstance(item, dict):
                item["queued"] = max(0, int(item.get("queued", 0)) - count)
            self._persist_locked()

    def evaluator_finished(
        self,
        elapsed: float,
        *,
        owner: str,
        failed: bool,
        worker_id: str | None = None,
    ) -> None:
        with self._lock:
            self._state["active_evaluators"] = max(
                0,
                int(self._state["active_evaluators"]) - 1,
            )
            self._state["evaluator_busy_seconds"] = (
                float(self._state["evaluator_busy_seconds"]) + elapsed
            )
            counter = "failed_evaluations" if failed else "completed_evaluations"
            self._state[counter] = int(self._state[counter]) + 1
            active_work = self._state.setdefault("active_evaluation_work", {})
            if isinstance(active_work, dict):
                active_work.pop(worker_id or str(threading.get_native_id()), None)
            progress = self._state.setdefault("evaluation_progress", {})
            item = progress.get(owner) if isinstance(progress, dict) else None
            if isinstance(item, dict):
                item["running"] = max(0, int(item.get("running", 0)) - 1)
            self._persist_locked()

    def evaluation_started(
        self,
        *,
        key: str,
        total: int,
        completed: int,
        queued: int,
    ) -> None:
        if not key or total < 0 or not 0 <= completed <= total or queued < 0:
            raise ValueError("evaluation progress identity or total is invalid")
        with self._lock:
            progress = self._state.setdefault("evaluation_progress", {})
            if not isinstance(progress, dict):
                raise core.M5InfrastructureError("M10 evaluation progress is malformed")
            progress[key] = {
                "completed": completed,
                "total": total,
                "queued": queued,
                "running": 0,
                "state": "queued" if queued else "terminal",
                "started_epoch_seconds": time.time(),
            }
            self._state["evaluation_cases_completed"] = (
                int(self._state.get("evaluation_cases_completed", 0)) + completed
            )
            self._state["evaluation_cases_total"] = (
                int(self._state.get("evaluation_cases_total", 0)) + total
            )
            owner = "baseline" if key.startswith("baseline:") else "candidate"
            completed_field = f"{owner}_evaluation_cases_completed"
            total_field = f"{owner}_evaluation_cases_total"
            self._state[completed_field] = int(self._state.get(completed_field, 0)) + completed
            self._state[total_field] = int(self._state.get(total_field, 0)) + total
            self._persist_locked()

    def evaluation_case_completed(self, *, key: str, completed: int) -> None:
        if completed < 0:
            raise ValueError("evaluation progress cannot be negative")
        with self._lock:
            progress = self._state.setdefault("evaluation_progress", {})
            if not isinstance(progress, dict):
                raise core.M5InfrastructureError("M10 evaluation progress is malformed")
            item = progress.get(key)
            if not isinstance(item, dict):
                return
            total = int(item.get("total", 0))
            value = min(completed, total)
            previous = int(item.get("completed", 0))
            item["completed"] = max(previous, value)
            self._state["evaluation_cases_completed"] = int(
                self._state.get("evaluation_cases_completed", 0)
            ) + max(0, item["completed"] - previous)
            owner = "baseline" if key.startswith("baseline:") else "candidate"
            completed_field = f"{owner}_evaluation_cases_completed"
            self._state[completed_field] = int(
                self._state.get(completed_field, 0)
            ) + max(0, item["completed"] - previous)
            self._persist_locked()

    def evaluation_finished(self, *, key: str) -> None:
        with self._lock:
            progress = self._state.setdefault("evaluation_progress", {})
            if not isinstance(progress, dict):
                raise core.M5InfrastructureError("M10 evaluation progress is malformed")
            item = progress.get(key)
            if isinstance(item, dict):
                item["queued"] = 0
                item["running"] = 0
                item["state"] = "terminal"
            self._persist_locked()

    def evaluation_committed(self, *, key: str) -> None:
        with self._lock:
            progress = self._state.setdefault("evaluation_progress", {})
            if not isinstance(progress, dict):
                raise core.M5InfrastructureError("M10 evaluation progress is malformed")
            progress.pop(key, None)
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
            self._state["active_provider_turns"] = 0
            self._state["active_evaluation_work"] = {}
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


@dataclass(slots=True)
class _PanelWork:
    key: str
    panel: tuple[core.DevelopmentCaseV1, ...]
    evaluation_paths: tuple[Path, ...]
    operation: Callable[[M10ScientificEvaluator, core.DevelopmentCaseV1], Mapping[str, JsonValue]]
    future: Future[_EvaluationOutcome]
    payloads: list[dict[str, Any] | None]
    queued_at: float
    remaining_indices: deque[int]
    finalize: Callable[[tuple[dict[str, Any], ...]], None] | None = None
    running: int = 0
    failure_type: str | None = None
    failure_message: str | None = None
    failure_case_id: str | None = None


def _evaluation_process_name(worker: int, owner: str) -> str:
    """Return one unique owner-aware Linux process name."""

    prefix = f"mf{worker:02d}-"
    candidate = re.fullmatch(r"candidate:g(\d+)-slot-(\d+)", owner)
    if candidate is not None:
        owner_name = f"g{int(candidate.group(1)):x}s{int(candidate.group(2)):02d}"
    elif owner == "baseline:random":
        owner_name = "rand"
    elif owner == "baseline:structural":
        owner_name = "struct"
    else:
        owner_name = "work"
    return f"{prefix}{owner_name}"[:15]


class _ConcurrentEvaluatorPool:
    """Bounded fair case scheduler with one private evaluator per worker."""

    def __init__(
        self,
        *,
        workers: int,
        queue_capacity: int,
        evaluator_factory: Callable[[], M10ScientificEvaluator],
        telemetry: _RuntimeTelemetry,
    ) -> None:
        self._factory = evaluator_factory
        self._telemetry = telemetry
        self._owned: list[M10ScientificEvaluator] = []
        self._owned_lock = threading.Lock()
        self._closing = False
        self._force_closing = False
        self._panel_capacity = threading.BoundedSemaphore(queue_capacity)
        self._condition = threading.Condition()
        self._runnable: deque[_PanelWork] = deque()
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                args=(index,),
                name=f"mforge-m10-evaluator-{index:02d}",
            )
            for index in range(workers)
        )
        for thread in self._threads:
            thread.start()

    def _worker(self, worker_index: int) -> None:
        evaluator: M10ScientificEvaluator | None = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closing or bool(self._runnable))
                if self._closing and (self._force_closing or not self._runnable):
                    break
                work = self._runnable.popleft()
                if work.failure_type is not None or not work.remaining_indices:
                    continue
                index = work.remaining_indices.popleft()
                work.running += 1
                if work.remaining_indices:
                    self._runnable.append(work)
                case = work.panel[index]
            failed = False
            payload: dict[str, Any] | None = None
            try:
                if evaluator is None:
                    evaluator = self._factory()
                    with self._owned_lock:
                        if self._force_closing:
                            close = getattr(evaluator, "close", None)
                            if callable(close):
                                try:
                                    close(force=True)
                                except TypeError:
                                    close()
                            raise RuntimeError("evaluator pool is closing")
                        self._owned.append(evaluator)
                    self._telemetry.evaluator_created()
                set_worker_name = getattr(evaluator, "set_worker_name", None)
                if callable(set_worker_name):
                    set_worker_name(_evaluation_process_name(worker_index, work.key))
                raw_worker_pid = getattr(evaluator, "worker_pid", None)
                worker_id = (
                    str(raw_worker_pid)
                    if isinstance(raw_worker_pid, int) and not isinstance(raw_worker_pid, bool)
                    else str(threading.get_native_id())
                )
                started = time.monotonic()
                failed = False
                self._telemetry.evaluator_started(
                    started - work.queued_at,
                    owner=work.key,
                    case_id=case.case_id,
                    worker_id=worker_id,
                    process_id=(
                        raw_worker_pid
                        if isinstance(raw_worker_pid, int)
                        and not isinstance(raw_worker_pid, bool)
                        else None
                    ),
                )
                try:
                    payload = dict(work.operation(evaluator, case))
                    self._telemetry.timed_persist(
                        partial(_write_or_verify, work.evaluation_paths[index], payload)
                    )
                except Exception:
                    failed = True
                    raise
                finally:
                    self._telemetry.evaluator_finished(
                        time.monotonic() - started,
                        owner=work.key,
                        failed=failed,
                        worker_id=worker_id,
                    )
            except Exception as error:
                failed = True
                with self._condition:
                    if work.failure_type is None:
                        work.failure_type = str(
                            getattr(error, "remote_type", type(error).__name__)
                        )
                        work.failure_message = str(error)[:1024]
                        work.failure_case_id = case.case_id
                        self._runnable = deque(item for item in self._runnable if item is not work)
                        cancelled = len(work.remaining_indices)
                        work.remaining_indices.clear()
                        self._telemetry.evaluator_cancelled(
                            cancelled,
                            owner=work.key,
                        )
            with self._condition:
                if payload is not None:
                    work.payloads[index] = payload
                    self._telemetry.evaluation_case_completed(
                        key=work.key,
                        completed=sum(item is not None for item in work.payloads),
                    )
                work.running -= 1
                if (
                    work.running == 0
                    and (work.failure_type is not None or not work.remaining_indices)
                ):
                    self._finish_work(work)
                self._condition.notify_all()

    def _finish_work(self, work: _PanelWork) -> None:
        payloads = tuple(item for item in work.payloads if item is not None)
        if work.failure_type is None and len(payloads) == len(work.panel):
            try:
                if work.finalize is not None:
                    work.finalize(payloads)
            except Exception as error:
                work.failure_type = type(error).__name__
                work.failure_message = str(error)[:1024]
        self._telemetry.evaluation_finished(key=work.key)
        work.future.set_result(
            _EvaluationOutcome(
                payloads,
                work.failure_type,
                work.failure_message,
                work.failure_case_id,
            )
        )
        self._panel_capacity.release()

    def _submit_panel(
        self,
        *,
        key: str,
        panel: tuple[core.DevelopmentCaseV1, ...],
        evaluation_paths: tuple[Path, ...],
        operation: Callable[
            [M10ScientificEvaluator, core.DevelopmentCaseV1],
            Mapping[str, JsonValue],
        ],
        finalize: Callable[[tuple[dict[str, Any], ...]], None] | None = None,
    ) -> Future[_EvaluationOutcome]:
        self._panel_capacity.acquire()
        future: Future[_EvaluationOutcome] = Future()
        payloads: list[dict[str, Any] | None] = [None] * len(panel)
        missing: deque[int] = deque()
        try:
            for index, path in enumerate(evaluation_paths):
                if path.is_file():
                    payloads[index] = core._load_mapping(path)
                else:
                    missing.append(index)
            self._telemetry.evaluation_started(
                key=key,
                total=len(panel),
                completed=len(panel) - len(missing),
                queued=len(missing),
            )
            self._telemetry.evaluator_queued(len(missing))
            work = _PanelWork(
                key=key,
                panel=panel,
                evaluation_paths=evaluation_paths,
                operation=operation,
                future=future,
                payloads=payloads,
                queued_at=time.monotonic(),
                remaining_indices=missing,
                finalize=finalize,
            )
            with self._condition:
                if self._closing:
                    raise RuntimeError("evaluator pool is closing")
                if missing:
                    self._runnable.append(work)
                    self._condition.notify_all()
                else:
                    self._finish_work(work)
            return future
        except BaseException:
            self._panel_capacity.release()
            raise

    def submit(
        self,
        *,
        source: str,
        panel: tuple[core.DevelopmentCaseV1, ...],
        candidate_id: str,
        slot_dir: Path,
    ) -> Future[_EvaluationOutcome]:
        paths = tuple(slot_dir / "evaluations" / f"{case.case_id}.json.gz" for case in panel)
        return self._submit_panel(
            key=f"candidate:{candidate_id}",
            panel=panel,
            evaluation_paths=paths,
            operation=lambda evaluator, case: evaluator.evaluate(
                source=source,
                case=case,
                candidate_id=candidate_id,
            ),
        )

    def submit_baseline(
        self,
        *,
        baseline: str,
        panel: tuple[core.DevelopmentCaseV1, ...],
        generation: int,
        generation_dir: Path,
    ) -> Future[_EvaluationOutcome]:
        baseline_dir = generation_dir / "baselines" / baseline
        paths = tuple(baseline_dir / "evaluations" / f"{case.case_id}.json.gz" for case in panel)

        def finalize(payloads: tuple[dict[str, Any], ...]) -> None:
            result = {
                "protocol_id": M10_BASELINE_RESULT_PROTOCOL_ID,
                "generation": generation,
                "baseline": baseline,
                "panel_hash": core.panel_hash(panel),
                "evaluation_case_count": len(payloads),
                "profile": core.aggregate_behavior(payloads),
            }
            self._telemetry.timed_persist(
                partial(
                    _write_or_verify,
                    baseline_dir / M10_BASELINE_RESULT_FILENAME,
                    result,
                )
            )

        return self._submit_panel(
            key=f"baseline:{baseline}",
            panel=panel,
            evaluation_paths=paths,
            operation=lambda evaluator, case: evaluator.evaluate_baseline(
                baseline=baseline,
                case=case,
                generation=generation,
            ),
            finalize=finalize,
        )

    def close(self, *, force: bool = False) -> Exception | None:
        errors: list[Exception] = []
        with self._condition:
            self._closing = True
            self._force_closing = force
            if force:
                self._runnable.clear()
            self._condition.notify_all()
        if force:
            with self._owned_lock:
                owned = tuple(self._owned)
            for evaluator in owned:
                close = getattr(evaluator, "close", None)
                if callable(close):
                    try:
                        try:
                            close(force=True)
                        except TypeError:
                            close()
                    except Exception as error:
                        errors.append(error)
        for thread in self._threads:
            thread.join()
        if not force:
            with self._owned_lock:
                owned = tuple(self._owned)
            for evaluator in owned:
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
            raise core.M5InfrastructureError(f"immutable M10 metadata changed: {path}")
        return
    write_json(path, value, exclusive=True)


def _prepared_path(slot_dir: Path) -> Path:
    return slot_dir / M10_PREPARED_FILENAME


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
        value.get("protocol_id") != M10_PREPARED_CANDIDATE_PROTOCOL_ID
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
        raise core.M5InfrastructureError(f"retained prepared candidate identity changed: {path}")
    candidates = core._all_candidates(root)
    parent = next(
        (item for item in candidates if item.get("candidate_id") == slot_plan.parent_candidate_id),
        None,
    )
    if slot_plan.kind == "child" and parent is None:
        raise core.M5InfrastructureError("prepared child parent is unavailable")
    if value.get("parent_program_hash") != (
        parent.get("program_hash") if parent is not None else None
    ) or value.get("parent_behavior_signature") != (
        parent.get("behavior_signature") if parent is not None else None
    ):
        raise core.M5InfrastructureError("prepared parent identity changed")
    attempts_raw = value.get("provider_attempts")
    if not isinstance(attempts_raw, Sequence) or isinstance(attempts_raw, str | bytes):
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
            or current.context.included_turn_ids[:-1] != previous.context.included_turn_ids
        ):
            raise core.M5InfrastructureError("prepared provider attempt lineage changed")
    provider_context = value.get("provider_context")
    if (
        not isinstance(provider_context, Mapping)
        or core.M5ProviderContextV1.from_dict(provider_context) != attempts[-1].context
    ):
        raise core.M5InfrastructureError("prepared provider context changed")
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
        or value.get("canonical_ast_sha256") != validation.identity.canonical_ast_sha256
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


@dataclass(frozen=True, slots=True)
class _ProviderTask:
    pending: _PendingCommit
    slot_plan: core.SlotPlanV1
    parent: Mapping[str, Any] | None
    prompt: str
    expected_history: tuple[str, ...]
    key: str
    durable_result_path: Path
    operation: Callable[[], core.M5ProviderResultV1]


def _provider_call(
    *,
    telemetry: _RuntimeTelemetry,
    key: str,
    kind: str,
    durable_result_path: Path,
    operation: Callable[[], core.M5ProviderResultV1],
) -> core.M5ProviderResultV1:
    newly_reserved = telemetry.provider_started(
        key,
        kind=kind,
    )
    if not newly_reserved and not durable_result_path.is_file():
        raise core.M5InfrastructureError(
            "interrupted provider turn has no durable result and will not repeat"
        )
    if not newly_reserved:
        return operation()
    started = time.monotonic()
    failed = True
    error_message: str | None = None
    try:
        result = operation()
        failed = False
        return result
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        telemetry.provider_finished(
            time.monotonic() - started,
            key=key,
            failed=failed,
            error=error_message,
        )


def _primary_provider_call(
    *,
    provider: core.M10SearchProvider,
    generation: int,
    slot: str,
    telemetry: _RuntimeTelemetry,
    key: str,
    durable_result_path: Path,
    operation: Callable[[], core.M5ProviderResultV1],
) -> core.M5ProviderResultV1:
    provider.await_primary_slot(generation=generation, slot=slot)
    return _provider_call(
        telemetry=telemetry,
        key=key,
        kind="primary",
        durable_result_path=durable_result_path,
        operation=operation,
    )


def _queue_valid_candidate(
    *,
    pending: _PendingCommit,
    root: Path,
    panel: tuple[core.DevelopmentCaseV1, ...],
    telemetry: _RuntimeTelemetry,
    pool: _ConcurrentEvaluatorPool,
    base: Mapping[str, Any],
    validation: Any,
    boundary_hook: Callable[[str], None] | None,
) -> bool:
    """Persist one validated source and enqueue its evaluation immediately."""

    if (
        not validation.valid
        or validation.response is None
        or validation.identity is None
        or validation.identity.program_hash is None
    ):
        return False
    source = normalize_source_newlines(validation.response.source)
    program_hash = validation.identity.program_hash
    source_path = root / "sources" / f"{program_hash}.py"
    telemetry.timed_persist(partial(core._write_source_exclusive_or_verify, source_path, source))
    if boundary_hook is not None:
        boundary_hook(f"{pending.candidate_id}_source_persisted")
    canonical_ast_sha256 = validation.identity.canonical_ast_sha256
    if not isinstance(canonical_ast_sha256, str):
        raise core.M5InfrastructureError("validated program omitted canonical AST identity")
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
    telemetry.timed_persist(partial(_write_or_verify, _prepared_path(pending.slot_dir), prepared))
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
    return True


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
        "parent_program_hash": (parent.get("program_hash") if parent is not None else None),
        "parent_behavior_signature": (
            parent.get("behavior_signature") if parent is not None else None
        ),
        "panel_hash": slot_plan.panel_hash,
        "panel_case_ids": [item.case_id for item in panel],
        "search_memory_sha256": memory["sha256"],
        "provider_context": (
            provider_result.context.as_dict() if provider_result is not None else None
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
        "protocol_id": M10_PREPARED_CANDIDATE_PROTOCOL_ID,
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
    force_stop: Callable[[], bool] | None = None,
) -> tuple[bool, core.M5InfrastructureError | None] | None:
    if pending.already_terminal:
        return False, None
    if pending.unprocessed:
        return None
    candidate_path = pending.slot_dir / "candidate.json.gz"
    if pending.terminal_payload is not None:
        terminal_payload = pending.terminal_payload
        telemetry.timed_persist(partial(_write_or_verify, candidate_path, terminal_payload))
        if boundary_hook is not None:
            boundary_hook(f"{pending.candidate_id}_committed")
        pending.terminal_payload = None
        return False, None
    if pending.future is None or pending.prepared is None:
        raise core.M5InfrastructureError("M10 pending candidate is incomplete")
    if not block and not pending.future.done():
        return None
    while block and not pending.future.done():
        if force_stop is not None and force_stop():
            raise core.M5OperatorStop("immediate operator stop requested")
        time.sleep(0.05)
    outcome = pending.future.result()
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
        telemetry.timed_persist(partial(_write_or_verify, candidate_path, candidate))
        telemetry.evaluation_committed(key=f"candidate:{pending.candidate_id}")
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
    evaluation_payloads = tuple(
        core._load_mapping(
            pending.slot_dir / "evaluations" / f"{case.case_id}.json.gz"
        )
        for case in panel
    )
    if len(evaluation_payloads) != len(panel):
        raise core.M5InfrastructureError("M10 evaluator returned an incomplete development panel")
    behavior_profile = core.aggregate_behavior(evaluation_payloads)
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
    telemetry.timed_persist(partial(_write_or_verify, candidate_path, candidate))
    telemetry.evaluation_committed(key=f"candidate:{pending.candidate_id}")
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


def _commit_generation_baselines(
    *,
    generation: int,
    generation_dir: Path,
    panel: tuple[core.DevelopmentCaseV1, ...],
    options: ScientificEvaluationOptionsV1,
    futures: Mapping[str, Future[_EvaluationOutcome]],
    telemetry: _RuntimeTelemetry,
    boundary_hook: Callable[[str], None] | None,
    force_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    profiles: dict[str, JsonValue] = {}
    for baseline in options.baselines:
        future = futures[baseline]
        while not future.done():
            if force_stop is not None and force_stop():
                raise core.M5OperatorStop("immediate operator stop requested")
            time.sleep(0.05)
        outcome = future.result()
        if outcome.failure_type is not None:
            raise core.M5InfrastructureError(
                "baseline evaluation failed for "
                f"generation {generation}/{baseline}/"
                f"{outcome.failure_case_id}: {outcome.failure_type}"
            )
        baseline_result = core._load_mapping(
            generation_dir
            / "baselines"
            / baseline
            / M10_BASELINE_RESULT_FILENAME
        )
        if (
            baseline_result.get("protocol_id") != M10_BASELINE_RESULT_PROTOCOL_ID
            or baseline_result.get("generation") != generation
            or baseline_result.get("baseline") != baseline
            or baseline_result.get("panel_hash") != core.panel_hash(panel)
            or baseline_result.get("evaluation_case_count") != len(panel)
            or not isinstance(baseline_result.get("profile"), Mapping)
        ):
            raise core.M5InfrastructureError(f"baseline {baseline} returned an incomplete panel")
        profile = cast(Mapping[str, Any], baseline_result["profile"])
        profiles[baseline] = cast(JsonValue, profile)
        if boundary_hook is not None:
            boundary_hook(f"generation_{generation}_baseline_{baseline}_evaluated")
    summary = {
        "protocol_id": M10_BASELINE_PROTOCOL_ID,
        "generation": generation,
        "panel_hash": core.panel_hash(panel),
        "workload": workload_projection(
            generation=generation,
            panel=panel,
            options=options,
        ),
        "baselines": profiles,
    }
    telemetry.timed_persist(
        partial(
            _write_or_verify,
            generation_dir / M10_BASELINE_FILENAME,
            summary,
        )
    )
    if boundary_hook is not None:
        boundary_hook(f"generation_{generation}_baselines_committed")
    for baseline in options.baselines:
        telemetry.evaluation_committed(key=f"baseline:{baseline}")
    return summary


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
                    and candidate.get("program_hash") != parent.get("program_hash")
                ),
                "semantic_behavior_changed": (
                    isinstance(candidate.get("behavior_signature"), str)
                    and isinstance(parent.get("behavior_signature"), str)
                    and candidate.get("behavior_signature") != parent.get("behavior_signature")
                ),
            }
        )
    return result


def _panel_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[core.DevelopmentCaseV1, ...]:
    raw_panel = manifest.get("panel")
    if not isinstance(raw_panel, Sequence) or isinstance(raw_panel, str | bytes):
        raise core.M5InfrastructureError("generation manifest panel is malformed")
    try:
        return tuple(
            core.DevelopmentCaseV1.from_dict(item)
            for item in raw_panel
            if isinstance(item, Mapping)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise core.M5InfrastructureError("generation manifest panel is invalid") from error


def _report(
    *,
    root: Path,
    provider_model: str,
    provider_effort: str,
    anchor_result: core.M5ProviderResultV1,
    options: ScientificSearchOptionsV2,
    runtime: Mapping[str, Any],
    stop_reason: str,
) -> dict[str, Any]:
    candidates = core._all_candidates(root)
    manifests = {
        int(path.parent.name.removeprefix("generation-")): core._load_mapping(path)
        for path in sorted(root.glob("generations/generation-*/manifest.json.gz"))
    }
    generations = sorted(manifests)
    panels = {
        generation: _panel_from_manifest(manifest) for generation, manifest in manifests.items()
    }
    baseline_summaries = {
        generation: core._load_mapping(
            root / "generations" / f"generation-{generation:04d}" / M10_BASELINE_FILENAME
        )
        for generation in generations
        if (root / "generations" / f"generation-{generation:04d}" / M10_BASELINE_FILENAME).is_file()
    }
    planned_candidate_count = sum(
        len(cast(Sequence[object], manifest.get("slots", ()))) for manifest in manifests.values()
    )
    statuses = Counter(str(item.get("status")) for item in candidates)
    proofs = _child_mutation_proofs(candidates)
    runtime_payload = dict(runtime)
    profiles = [
        cast(Mapping[str, Any], item["behavior_profile"])
        for item in candidates
        if isinstance(item.get("behavior_profile"), Mapping)
    ]
    exact_verified = any(profile.get("exact_verified") is True for profile in profiles)
    baseline_profiles = [
        profile
        for summary in baseline_summaries.values()
        for profile in (
            cast(Mapping[str, Any], summary.get("baselines", {})).values()
            if isinstance(summary.get("baselines"), Mapping)
            else ()
        )
        if isinstance(profile, Mapping)
    ]
    exact_verified |= any(profile.get("exact_verified") is True for profile in baseline_profiles)
    report: dict[str, Any] = {
        "protocol_id": M10_REPORT_PROTOCOL_ID,
        "search_protocol_id": M10_SEARCH_PROTOCOL_ID,
        "status": "completed",
        "stop_reason": stop_reason,
        "generation_count": len(generations),
        "generation_limit": options.generation_limit,
        "population_size": core.POPULATION_SIZE,
        "planned_candidate_count": planned_candidate_count,
        "candidate_count": len(candidates),
        "pending_candidate_count": max(0, planned_candidate_count - len(candidates)),
        "candidate_status_counts": dict(sorted(statuses.items())),
        "repaired_valid_count": sum(
            int(item.get("repairs", 0)) > 0 and item.get("status") in {"evaluated", "duplicate"}
            for item in candidates
        ),
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
                root / "generations" / f"generation-{generation:04d}" / "manifest.json.gz"
            )["sha256"]
            for generation in generations
        },
        "search_memory_hashes": {
            str(generation): core._load_mapping(
                root / "generations" / f"generation-{generation:04d}" / "search-memory.json.gz"
            )["sha256"]
            for generation in generations
            if (
                root / "generations" / f"generation-{generation:04d}" / "search-memory.json.gz"
            ).is_file()
        },
        "lineage": [
            {
                "candidate_id": item["candidate_id"],
                "parent_candidate_id": item.get("parent_candidate_id"),
                "parent_program_hash": item.get("parent_program_hash"),
                "parent_behavior_signature": item.get("parent_behavior_signature"),
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
            for case in panels[int(item["generation"])]
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
            str(item["candidate_id"]): item.get("behavior_profile") for item in candidates
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
        "provider_turns": 1 + int(runtime_payload.get("provider_turns_submitted", 0)),
        "specification_anchor_turns": 1,
        "candidate_program_turns": int(runtime_payload.get("provider_turns_submitted", 0)),
        "repair_turns": sum(int(item.get("repairs", 0)) for item in candidates),
        "provider_accounting": {
            "model": provider_model,
            "effort": provider_effort,
            "warnings": anchor_result.warnings
            + sum(int(item.get("warnings", 0)) for item in candidates),
            "duration_ms": anchor_result.duration_ms
            + sum(int(item.get("duration_ms", 0)) for item in candidates),
            "primary_program_slots": options.primary_program_slots,
            "repair_turn_limit": options.repair_turn_limit,
            "total_turn_limit": options.provider_total_turn_limit,
            "primary_turns_submitted": int(runtime_payload.get("primary_turns_submitted", 0)),
            "repair_turns_submitted": int(runtime_payload.get("repair_turns_submitted", 0)),
        },
        "exact_verified": exact_verified,
        "exact_verification": {
            "authority": "exact_verifier_only",
            "submissions": sum(
                int(profile.get("exact_verifier_submissions", 0))
                for profile in [*profiles, *baseline_profiles]
            ),
            "records": sum(
                int(profile.get("exact_verifier_records", 0))
                for profile in [*profiles, *baseline_profiles]
            ),
            "verified": exact_verified,
            "queue": 0,
        },
        "generation_workloads": {
            str(generation): workload_projection(
                generation=generation,
                panel=panels[generation],
                options=options.evaluation,
            )
            for generation in generations
        },
        "generation_baselines": {
            str(generation): summary for generation, summary in baseline_summaries.items()
        },
        "equal_development_budget": {
            "case_count_per_generation": {
                str(generation): len(panels[generation]) for generation in generations
            },
            "all_evaluated_candidates_complete": all(
                int(item.get("evaluation_case_count", 0)) == len(panels[int(item["generation"])])
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
            "terminal_slots_not_replaced": options.replace_terminal_slots is False,
            "equal_development_panel_and_budget": all(
                int(item.get("evaluation_case_count", 0)) == len(panels[int(item["generation"])])
                for item in candidates
                if item.get("status") in {"evaluated", "duplicate"}
            ),
            "generation_baselines_complete": (
                len(baseline_summaries) == len(generations)
                and all(
                    set(
                        cast(
                            Mapping[str, Any],
                            summary.get("baselines", {}),
                        )
                    )
                    == set(options.evaluation.baselines)
                    for summary in baseline_summaries.values()
                    if isinstance(summary.get("baselines"), Mapping)
                )
            ),
            "exact_verifier_only_authority": True,
            "provider_program_turn_budget_respected": (
                True
                if options.provider_total_turn_limit is None
                else int(runtime_payload.get("provider_turns_submitted", 0))
                <= options.provider_total_turn_limit
            ),
        },
    }
    return report


def resolve_resume_generation(
    *,
    root: Path,
    options: ScientificSearchOptionsV2,
    budget: ScientificResumeBudgetV1,
) -> tuple[int, tuple[str, ...]]:
    manifests: list[int] = []
    incomplete: list[int] = []
    pending_primary: dict[int, tuple[str, ...]] = {}
    generations = sorted(
        int(path.parent.name.removeprefix("generation-"))
        for path in root.glob("generations/generation-*/manifest.json.gz")
    )
    for generation in generations:
        generation_dir = root / "generations" / f"generation-{generation:04d}"
        if not (generation_dir / "manifest.json.gz").is_file():
            continue
        manifests.append(generation)
        terminal = 0
        pending: list[str] = []
        for slot_index in range(core.POPULATION_SIZE):
            slot = f"slot-{slot_index:02d}"
            slot_dir = core._candidate_path(root, generation, slot)
            if (slot_dir / "candidate.json.gz").is_file():
                terminal += 1
            elif not _prepared_path(slot_dir).is_file():
                pending.append(core._candidate_id(generation, slot))
        if terminal < core.POPULATION_SIZE:
            incomplete.append(generation)
            pending_primary[generation] = tuple(pending)
    if not manifests or len(incomplete) != 1:
        raise core.M5InfrastructureError(
            "current-generation resume requires exactly one incomplete existing generation"
        )
    generation = incomplete[0]
    if generation != max(manifests):
        raise core.M5InfrastructureError(
            "a later generation already exists after the incomplete generation"
        )
    for previous in range(generation):
        if previous not in manifests:
            raise core.M5InfrastructureError(
                "generation history is incomplete before the resume boundary"
            )
    pending_ids = pending_primary[generation]
    if len(pending_ids) != budget.expected_pending_primary_slots:
        raise core.M5InfrastructureError(
            "pending primary slot count changed: "
            f"expected {budget.expected_pending_primary_slots}, "
            f"found {len(pending_ids)}"
        )
    return generation, pending_ids


def _next_generation_to_run(root: Path) -> int:
    generations = sorted(
        int(path.parent.name.removeprefix("generation-"))
        for path in root.glob("generations/generation-*/manifest.json.gz")
    )
    if not generations:
        return 0
    if generations != list(range(generations[-1] + 1)):
        raise core.M5InfrastructureError("generation manifest history is incomplete")
    for generation in generations:
        generation_dir = root / "generations" / f"generation-{generation:04d}"
        terminal = len(core._generation_candidates(root, generation))
        baselines_complete = (generation_dir / M10_BASELINE_FILENAME).is_file()
        if terminal < core.POPULATION_SIZE or not baselines_complete:
            if generation != generations[-1]:
                raise core.M5InfrastructureError(
                    "a later generation exists after an incomplete generation"
                )
            return generation
    return generations[-1] + 1


def run_sustained_search(
    *,
    provider: core.M10SearchProvider,
    evaluator_factory: Callable[[], M10ScientificEvaluator],
    workspace: str | Path,
    panel_factory: Callable[[int], tuple[core.DevelopmentCaseV1, ...]],
    system_prompt: str,
    specification_prompt: str,
    specification_ack_schema: Mapping[str, Any],
    policy_schema: Mapping[str, Any],
    options: ScientificSearchOptionsV2,
    provider_turn_timeout_seconds: float,
    resume_budget: ScientificResumeBudgetV1 | None = None,
    operator_stop: Callable[[], bool] | None = None,
    force_stop: Callable[[], bool] | None = None,
    boundary_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run or resume one bounded, completion-order-independent campaign."""

    if provider_turn_timeout_seconds <= 0 or (
        options.wall_seconds is not None and provider_turn_timeout_seconds >= options.wall_seconds
    ):
        raise ValueError("provider turn timeout must be positive and below the wall budget")
    if provider.provider_concurrency != options.provider_concurrency:
        raise ValueError("provider concurrency differs from the frozen options")
    core._assert_model_prompt_hygiene(system_prompt)
    core._assert_model_prompt_hygiene(specification_prompt)
    root = Path(workspace)
    resume_generation: int | None = None
    if resume_budget is not None:
        resume_generation, _ = resolve_resume_generation(
            root=root,
            options=options,
            budget=resume_budget,
        )
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol_id": M10_SEARCH_PROTOCOL_ID,
        "population_size": core.POPULATION_SIZE,
        "generation_limit": options.generation_limit,
        "generation_zero_roots": core.POPULATION_SIZE,
        "later_child_slots": core.CHILD_SLOTS,
        "later_root_slots": core.ROOT_SLOTS,
        "model": provider.model,
        "effort": provider.effort,
        **options.scientific_identity_dict(),
        "replace_terminal_slots": False,
        "safe_api_expanded": True,
        "preview_active": True,
        "native_v2_default": True,
        "dsl_runtime_used": False,
    }
    _write_or_verify(root / "protocol.json.gz", protocol)
    if boundary_hook is not None:
        boundary_hook("protocol_persisted")
    telemetry = _RuntimeTelemetry(root, options, resume_budget)
    telemetry.clear_terminal_reason()
    pool = _ConcurrentEvaluatorPool(
        workers=options.evaluator_workers,
        queue_capacity=options.validated_queue_capacity,
        evaluator_factory=evaluator_factory,
        telemetry=telemetry,
    )
    provider_executor = ThreadPoolExecutor(
        # Keep the host-side provider workers equal to the configured number
        # of AI lanes.  Additional queued futures made the dashboard look as
        # if six or eight model connections were active, even though only two
        # provider lanes were allowed to enter the app-server.
        max_workers=options.provider_concurrency,
        thread_name_prefix="mforge-m10-provider",
    )
    stop_reason: str | None = None
    anchor_result: core.M5ProviderResultV1 | None = None
    exact_verified = False
    close_error: Exception | None = None
    provider_abort = False

    def close_provider_forcefully() -> None:
        close = getattr(provider, "close", None)
        if not callable(close):
            return
        try:
            close(force=True)
        except TypeError:
            close()

    def check_force_stop() -> None:
        if force_stop is None or not force_stop():
            return
        # Kill app-server workers before raising so an immediate dashboard
        # interrupt does not wait for an in-flight provider timeout.
        with suppress(Exception):
            close_provider_forcefully()
        raise core.M5OperatorStop("immediate operator stop requested")

    def wait_for_provider_result[T](
        future: Future[T],
        *,
        submitted_at: float | None = None,
    ) -> T:
        nonlocal provider_abort
        deadline = (submitted_at or time.monotonic()) + provider_turn_timeout_seconds
        while True:
            check_force_stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                provider_abort = True
                with suppress(Exception):
                    close_provider_forcefully()
                raise core.M5InfrastructureError(
                    f"provider turn exceeded {provider_turn_timeout_seconds:.0f}s"
                )
            try:
                return future.result(timeout=min(0.1, remaining))
            except FutureTimeoutError:
                continue

    def stop_payload(reason: str, error: Exception) -> dict[str, Any]:
        return {
            "protocol_id": M10_REPORT_PROTOCOL_ID,
            "status": reason,
            "error_type": type(error).__name__,
            "error": str(error)[:2048],
            "candidate_count": len(core._all_candidates(root)),
            "evaluation": options.evaluation.as_dict(),
            "resumable": True,
        }

    try:
        check_force_stop()
        anchor_future = provider_executor.submit(
            provider.ensure_specification_anchor,
            prompt=specification_prompt,
            system_prompt=system_prompt,
            output_schema=specification_ack_schema,
            artifact_dir=root / "provider" / "specification-anchor",
        )
        anchor_result = wait_for_provider_result(anchor_future)
        check_force_stop()
        core._assert_provider_turn_boundary(anchor_result, expected_history=())
        anchor = anchor_result.context
        core._write_exclusive_or_verify(root / "anchor.json.gz", anchor_result.as_dict())
        if boundary_hook is not None:
            boundary_hook("anchor_persisted")

        next_generation = _next_generation_to_run(root)
        generations = (
            range(resume_generation, resume_generation + 1)
            if resume_generation is not None
            else count(next_generation)
            if options.generation_limit is None
            else range(next_generation, options.generation_limit)
        )
        for generation in generations:
            panel = panel_factory(generation)
            if not panel:
                raise core.M5InfrastructureError(
                    f"generation {generation} development panel is empty"
                )
            if operator_stop is not None and operator_stop():
                raise core.M5OperatorStop("operator stop requested")
            if (
                hourly_token_usage(
                    root,
                    options.max_total_tokens_per_hour,
                )["hourly_limit_reached"]
                is True
            ):
                stop_reason = "hourly_token_limit"
                break
            if options.wall_seconds is not None and (
                telemetry.wall_expired(options.wall_seconds)
                or telemetry.wall_remaining(options.wall_seconds) < provider_turn_timeout_seconds
            ):
                stop_reason = "wall_clock_budget"
                break

            previous = core._generation_candidates(root, generation - 1) if generation > 0 else []
            manifest = core.build_generation_manifest(
                generation=generation,
                panel=panel,
                previous_candidates=previous,
            )
            if generation > 0 and manifest.all_root_fallback:
                stop_reason = "insufficient_valid_parents"
                break
            primary_keys = [f"{slot.request_key}-initial" for slot in manifest.slots]
            telemetry.reserve_primary_generation(
                primary_keys,
                limit=options.primary_program_slots,
            )

            generation_dir = root / "generations" / f"generation-{generation:04d}"
            core._write_exclusive_or_verify(generation_dir / "manifest.json.gz", manifest.as_dict())
            baseline_summary_path = generation_dir / M10_BASELINE_FILENAME
            retained_baselines = (
                core._load_mapping(baseline_summary_path)
                if baseline_summary_path.is_file()
                else None
            )
            baseline_futures = (
                {}
                if retained_baselines is not None
                else {
                    baseline: pool.submit_baseline(
                        baseline=baseline,
                        panel=panel,
                        generation=generation,
                        generation_dir=generation_dir,
                    )
                    for baseline in options.evaluation.baselines
                }
            )
            if boundary_hook is not None:
                boundary_hook(f"generation_{generation}_manifest")
            memory_candidates = [
                item
                for item in core._all_candidates(root)
                if int(item.get("generation", -1)) < generation
            ]
            memory = core.build_search_memory(memory_candidates)
            core._write_exclusive_or_verify(generation_dir / "search-memory.json.gz", memory)
            if boundary_hook is not None:
                boundary_hook(f"generation_{generation}_search_memory")
            prior_candidates = core._all_candidates(root)
            by_id = {str(item["candidate_id"]): item for item in prior_candidates}
            snapshot = _generation_snapshot(
                generation=generation,
                manifest=manifest,
                memory=memory,
                candidates_by_id=by_id,
                anchor=anchor,
                model=provider.model,
                effort=provider.effort,
                system_prompt=system_prompt,
                policy_schema=policy_schema,
                panel=panel,
                options=options,
            )
            _write_or_verify(generation_dir / "generation-snapshot.json.gz", snapshot)
            # Forking/resuming the provider lanes performs app-server
            # protocol calls too.  Keep it on the bounded provider executor;
            # otherwise the dashboard thread cannot observe q/force_stop and
            # a stalled fork can outlive the configured 300 s turn timeout.
            check_force_stop()
            prepare_started_at = time.monotonic()
            prepare_future = provider_executor.submit(
                provider.prepare_generation,
                snapshot=snapshot,
                anchor=anchor,
                artifact_dir=generation_dir / "provider",
            )
            wait_for_provider_result(
                prepare_future,
                submitted_at=prepare_started_at,
            )
            check_force_stop()
            if boundary_hook is not None:
                boundary_hook(f"generation_{generation}_snapshot")

            entries = [
                _PendingCommit(
                    slot_plan=slot_plan,
                    candidate_id=core._candidate_id(generation, slot_plan.slot),
                    slot_dir=core._candidate_path(root, generation, slot_plan.slot),
                )
                for slot_plan in manifest.slots
            ]
            commit_cursor = 0

            def commit_ready(
                *,
                block: bool,
                generation_entries: list[_PendingCommit] = entries,
                generation_panel: tuple[core.DevelopmentCaseV1, ...] = panel,
            ) -> None:
                nonlocal commit_cursor, exact_verified
                while commit_cursor < len(generation_entries):
                    outcome = _commit_pending(
                        pending=generation_entries[commit_cursor],
                        root=root,
                        panel=generation_panel,
                        telemetry=telemetry,
                        block=block,
                        boundary_hook=boundary_hook,
                        force_stop=force_stop,
                    )
                    if outcome is None:
                        return
                    verified, infrastructure_error = outcome
                    commit_cursor += 1
                    exact_verified |= verified
                    if infrastructure_error is not None:
                        raise infrastructure_error

            provider_futures: dict[str, Future[core.M5ProviderResultV1]] = {}
            provider_submitted_at: dict[str, float] = {}
            provider_task_list: list[_ProviderTask] = []
            for slot_plan, pending, initial_key in zip(
                manifest.slots, entries, primary_keys, strict=True
            ):
                candidate_path = pending.slot_dir / "candidate.json.gz"
                prepared_path = _prepared_path(pending.slot_dir)
                if candidate_path.is_file():
                    retained = core._verify_retained_candidate(
                        root=root,
                        path=candidate_path,
                        panel=panel,
                        slot_plan=slot_plan,
                        search_memory_sha256=str(memory["sha256"]),
                    )
                    exact_verified |= retained.get("exact_verified") is True
                    pending.retained_terminal = True
                    provider.release_primary_slot(
                        generation=generation,
                        slot=slot_plan.slot,
                    )
                    commit_ready(block=False)
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
                    provider.release_primary_slot(
                        generation=generation,
                        slot=slot_plan.slot,
                    )
                    commit_ready(block=False)
                    continue

                parent: Mapping[str, Any] | None = None
                provider_key = initial_key
                idempotency_key = slot_plan.request_key
                artifact_dir = pending.slot_dir / "provider-initial"
                durable_result_path = artifact_dir / "m5-provider-result.json.gz"
                if (
                    resume_budget is not None
                    and telemetry.primary_was_started(initial_key)
                    and not durable_result_path.is_file()
                ):
                    slot_dir = pending.slot_dir

                    def retry_result_exists(
                        attempt: int,
                        *,
                        retry_slot_dir: Path = slot_dir,
                    ) -> bool:
                        return (
                            retry_slot_dir
                            / f"provider-resume-{attempt:02d}"
                            / "m5-provider-result.json.gz"
                        ).is_file()

                    provider_key, retry_attempt = telemetry.admit_primary_retry(
                        initial_key,
                        limit=options.primary_program_slots,
                        durable_result_exists=retry_result_exists,
                    )
                    idempotency_key = provider_key
                    artifact_dir = pending.slot_dir / f"provider-resume-{retry_attempt:02d}"
                    durable_result_path = artifact_dir / "m5-provider-result.json.gz"
                if slot_plan.kind == "root":
                    prompt = _root_prompt(snapshot)
                    operation = partial(
                        provider.generate_root,
                        anchor=anchor,
                        generation=generation,
                        slot=slot_plan.slot,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        output_schema=policy_schema,
                        idempotency_key=idempotency_key,
                        artifact_dir=artifact_dir,
                    )
                    expected_history: tuple[str, ...] = ()
                else:
                    parent_id = slot_plan.parent_candidate_id
                    if parent_id is None or parent_id not in by_id:
                        raise core.M5InfrastructureError("frozen child parent is unavailable")
                    parent = by_id[parent_id]
                    parent_source = parent.get("source")
                    parent_profile = parent.get("behavior_profile")
                    if not isinstance(parent_source, str) or not isinstance(
                        parent_profile, Mapping
                    ):
                        raise core.M5InfrastructureError("selected parent evidence is incomplete")
                    prompt = _child_prompt(
                        snapshot=snapshot,
                        parent_source=parent_source,
                        parent_profile=parent_profile,
                    )
                    parent_context = core._provider_context(parent)
                    expected_history = parent_context.included_turn_ids
                    operation = partial(
                        provider.generate_child,
                        parent=parent_context,
                        generation=generation,
                        slot=slot_plan.slot,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        output_schema=policy_schema,
                        idempotency_key=idempotency_key,
                        artifact_dir=artifact_dir,
                    )
                core._assert_model_prompt_hygiene(prompt)
                task = _ProviderTask(
                    pending=pending,
                    slot_plan=slot_plan,
                    parent=parent,
                    prompt=prompt,
                    expected_history=expected_history,
                    key=provider_key,
                    durable_result_path=durable_result_path,
                    operation=operation,
                )
                provider_task_list.append(task)

            if (
                resume_budget is not None
                and len(provider_task_list) != resume_budget.expected_pending_primary_slots
            ):
                raise core.M5InfrastructureError(
                    "pending primary task identities changed after resume validation"
                )

            lane_tasks: dict[int, list[_ProviderTask]] = {
                lane: [] for lane in range(options.provider_concurrency)
            }
            for task in provider_task_list:
                lane = provider.primary_lane(
                    generation=generation,
                    slot=task.slot_plan.slot,
                )
                if lane not in lane_tasks:
                    raise core.M5InfrastructureError("provider returned an invalid frozen lane")
                lane_tasks[lane].append(task)
            next_provider_task = 0

            def submit_next_provider_task(
                *,
                task_list: list[_ProviderTask] = provider_task_list,
                future_map: dict[str, Future[core.M5ProviderResultV1]] = provider_futures,
                submitted_map: dict[str, float] = provider_submitted_at,
                executor: ThreadPoolExecutor = provider_executor,
                provider_instance: core.M10SearchProvider = provider,
                generation_number: int = generation,
            ) -> None:
                nonlocal next_provider_task
                if next_provider_task >= len(task_list):
                    return
                task = task_list[next_provider_task]
                next_provider_task += 1
                submitted_at = time.monotonic()
                future_map[task.pending.candidate_id] = executor.submit(
                    _primary_provider_call,
                    provider=provider_instance,
                    generation=generation_number,
                    slot=task.slot_plan.slot,
                    telemetry=telemetry,
                    key=task.key,
                    durable_result_path=task.durable_result_path,
                    operation=task.operation,
                )
                submitted_map[task.pending.candidate_id] = submitted_at

            # Keep at most one future per configured provider lane in flight.
            # The next task is submitted only after its predecessor has been
            # consumed and its lane released below.
            for _ in range(min(options.provider_concurrency, len(provider_task_list))):
                submit_next_provider_task()

            for task in provider_task_list:
                future = provider_futures[task.pending.candidate_id]
                pending = task.pending
                try:
                    provider_result = wait_for_provider_result(
                        future,
                        submitted_at=provider_submitted_at.get(pending.candidate_id),
                    )
                    check_force_stop()
                    if task.slot_plan.kind == "child":
                        core._assert_provider_turn_boundary(
                            provider_result,
                            expected_history=task.expected_history,
                        )
                    elif anchor.turn_id not in (provider_result.context.included_turn_ids):
                        raise core.M5InfrastructureError("fresh root lost the specification anchor")
                except core.M5OperatorStop:
                    raise
                except core.M5InfrastructureError:
                    raise
                except Exception as error:
                    base = _candidate_base(
                        candidate_id=pending.candidate_id,
                        generation=generation,
                        slot_plan=task.slot_plan,
                        parent=task.parent,
                        panel=panel,
                        memory=memory,
                        provider_result=None,
                        attempts=(),
                        repairs=0,
                        prompt=task.prompt,
                    )
                    pending.terminal_payload = _provider_failure(base=base, error=error)
                    provider.release_primary_slot(
                        generation=generation,
                        slot=task.slot_plan.slot,
                    )
                    commit_ready(block=False)
                    submit_next_provider_task()
                    continue
                if boundary_hook is not None:
                    boundary_hook(f"{pending.candidate_id}_provider")
                attempts: list[Mapping[str, JsonValue]] = [provider_result.as_dict()]
                validation = validate_python_policy_response(provider_result.response_text)
                base = _candidate_base(
                    candidate_id=pending.candidate_id,
                    generation=generation,
                    slot_plan=task.slot_plan,
                    parent=task.parent,
                    panel=panel,
                    memory=memory,
                    provider_result=provider_result,
                    attempts=attempts,
                    repairs=0,
                    prompt=task.prompt,
                )
                valid = _queue_valid_candidate(
                    pending=pending,
                    root=root,
                    panel=panel,
                    telemetry=telemetry,
                    pool=pool,
                    base=base,
                    validation=validation,
                    boundary_hook=boundary_hook,
                )
                if valid:
                    provider.release_primary_slot(
                        generation=generation,
                        slot=task.slot_plan.slot,
                    )
                    commit_ready(block=False)
                    submit_next_provider_task()
                    continue
                repair_key = f"{task.slot_plan.request_key}-repair-01"
                if not telemetry.reserve_repair(repair_key, limit=options.repair_turn_limit):
                    base = _candidate_base(
                        candidate_id=pending.candidate_id,
                        generation=generation,
                        slot_plan=task.slot_plan,
                        parent=task.parent,
                        panel=panel,
                        memory=memory,
                        provider_result=provider_result,
                        attempts=attempts,
                        repairs=0,
                        prompt=task.prompt,
                    )
                    pending.terminal_payload = _contract_invalid(
                        base=base,
                        validation=validation.as_dict(),
                        repair_skipped="repair_turn_budget",
                    )
                    provider.release_primary_slot(
                        generation=generation,
                        slot=task.slot_plan.slot,
                    )
                    commit_ready(block=False)
                    submit_next_provider_task()
                    continue
                repair_prompt = _repair_prompt(
                    snapshot=snapshot,
                    diagnostics=[item.as_dict() for item in validation.diagnostics],
                )
                core._assert_model_prompt_hygiene(repair_prompt)
                previous_result = provider_result
                operation = partial(
                    provider.repair,
                    previous=previous_result,
                    generation=generation,
                    slot=task.slot_plan.slot,
                    prompt=repair_prompt,
                    system_prompt=system_prompt,
                    output_schema=policy_schema,
                    idempotency_key=task.slot_plan.request_key + "-repair-01",
                    artifact_dir=pending.slot_dir / "provider-repair-01",
                )
                repair_submitted_at = time.monotonic()
                repair_future = provider_executor.submit(
                    _provider_call,
                    telemetry=telemetry,
                    key=repair_key,
                    kind="repair",
                    durable_result_path=(
                        pending.slot_dir / "provider-repair-01" / "m5-provider-result.json.gz"
                    ),
                    operation=operation,
                )
                try:
                    repaired_result = wait_for_provider_result(
                        repair_future,
                        submitted_at=repair_submitted_at,
                    )
                    check_force_stop()
                    core._assert_provider_turn_boundary(
                        repaired_result,
                        expected_history=(previous_result.context.included_turn_ids),
                        expected_thread_id=previous_result.context.thread_id,
                    )
                except core.M5OperatorStop:
                    raise
                except core.M5InfrastructureError:
                    raise
                except Exception as error:
                    base = _candidate_base(
                        candidate_id=pending.candidate_id,
                        generation=generation,
                        slot_plan=task.slot_plan,
                        parent=task.parent,
                        panel=panel,
                        memory=memory,
                        provider_result=previous_result,
                        attempts=attempts,
                        repairs=1,
                        prompt=task.prompt,
                    )
                    pending.terminal_payload = _provider_failure(base=base, error=error)
                    provider.release_primary_slot(
                        generation=generation,
                        slot=task.slot_plan.slot,
                    )
                    commit_ready(block=False)
                    submit_next_provider_task()
                    continue
                repaired_attempts = [
                    *attempts,
                    repaired_result.as_dict(),
                ]
                repaired_validation = validate_python_policy_response(repaired_result.response_text)
                base = _candidate_base(
                    candidate_id=pending.candidate_id,
                    generation=generation,
                    slot_plan=task.slot_plan,
                    parent=task.parent,
                    panel=panel,
                    memory=memory,
                    provider_result=repaired_result,
                    attempts=repaired_attempts,
                    repairs=1,
                    prompt=task.prompt,
                )
                if not _queue_valid_candidate(
                    pending=pending,
                    root=root,
                    panel=panel,
                    telemetry=telemetry,
                    pool=pool,
                    base=base,
                    validation=repaired_validation,
                    boundary_hook=boundary_hook,
                ):
                    pending.terminal_payload = _contract_invalid(
                        base=base,
                        validation=repaired_validation.as_dict(),
                    )
                provider.release_primary_slot(
                    generation=generation,
                    slot=task.slot_plan.slot,
                )
                commit_ready(block=False)
                submit_next_provider_task()

            commit_ready(block=True)
            if commit_cursor != core.POPULATION_SIZE:
                raise core.M5InfrastructureError("generation did not reach eight terminal slots")
            baseline_summary = (
                retained_baselines
                if retained_baselines is not None
                else _commit_generation_baselines(
                    generation=generation,
                    generation_dir=generation_dir,
                    panel=panel,
                    options=options.evaluation,
                    futures=baseline_futures,
                    telemetry=telemetry,
                    boundary_hook=boundary_hook,
                    force_stop=force_stop,
                )
            )
            if baseline_summary.get("panel_hash") != core.panel_hash(panel):
                raise core.M5InfrastructureError(
                    "retained baseline panel differs from the generation"
                )
            raw_baselines = baseline_summary.get("baselines")
            if not isinstance(raw_baselines, Mapping) or set(raw_baselines) != set(
                options.evaluation.baselines
            ):
                raise core.M5InfrastructureError("generation baseline summary is malformed")
            exact_verified |= any(
                isinstance(profile, Mapping) and profile.get("exact_verified") is True
                for profile in raw_baselines.values()
            )
            telemetry.boundary(f"generation_{generation}_committed")
            if exact_verified:
                stop_reason = "exact_verified_counterexample"
                break
            if resume_generation is not None:
                stop_reason = "resume_generation_complete"
                break
        else:
            stop_reason = "generation_budget"
        telemetry.boundary("report_pending")
    except _ProviderTurnBudgetExhausted:
        stop_reason = "provider_turn_budget"
        telemetry.boundary("provider_turn_budget_exhausted")
    except (core.M5InfrastructureError, core.M5OperatorStop) as error:
        reason = (
            "operator_stop" if isinstance(error, core.M5OperatorStop) else "infrastructure_failure"
        )
        telemetry.finish(reason)
        write_json(root / M10_STOP_FILENAME, stop_payload(reason, error))
        raise
    finally:
        immediate_stop = provider_abort or (force_stop is not None and force_stop())
        provider_executor.shutdown(
            wait=not immediate_stop,
            cancel_futures=immediate_stop,
        )
        close_error = pool.close(force=immediate_stop)
    if close_error is not None:
        telemetry.finish("infrastructure_failure")
        cleanup_failure = core.M5InfrastructureError(
            f"evaluator cleanup failed: {type(close_error).__name__}"
        )
        write_json(
            root / M10_STOP_FILENAME,
            stop_payload("infrastructure_failure", cleanup_failure),
        )
        raise cleanup_failure from close_error
    if anchor_result is None:
        raise core.M5InfrastructureError("M10 specification anchor is missing")
    if stop_reason is None:
        raise core.M5InfrastructureError("scientific search stopped without a reason")
    telemetry.finish(stop_reason)
    report = _report(
        root=root,
        provider_model=provider.model,
        provider_effort=provider.effort,
        anchor_result=anchor_result,
        options=options,
        runtime=telemetry.snapshot(),
        stop_reason=stop_reason,
    )
    report["hourly_token_usage"] = hourly_token_usage(
        root,
        options.max_total_tokens_per_hour,
    )
    write_json(root / M10_REPORT_FILENAME, report)
    if boundary_hook is not None:
        boundary_hook("report_persisted")
    return report


__all__ = [
    "M10_REPORT_FILENAME",
    "M10_REPORT_PROTOCOL_ID",
    "M10_RUNTIME_FILENAME",
    "M10_SEARCH_PROTOCOL_ID",
    "M10_STOP_FILENAME",
    "M10ScientificEvaluator",
    "ScientificResumeBudgetV1",
    "ScientificSearchOptionsV2",
    "resolve_resume_generation",
    "run_sustained_search",
]
