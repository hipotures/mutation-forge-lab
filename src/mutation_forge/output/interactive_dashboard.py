from __future__ import annotations

import _thread
import os
import resource
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Any, Literal, TextIO

from rich import box
from rich.align import Align
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from mutation_forge.events import Event
from mutation_forge.models import JsonValue
from mutation_forge.output.panel_copy import (
    copy_text_to_clipboard_osc52,
    render_panel_copy_text,
    save_panel_copy,
)

REFRESH_INTERVAL_SECONDS = 1.0
COPY_NOTICE_SECONDS = 6.0
PANEL_COPY_TMP_DIR = Path("/tmp")
DETAIL_TABS = (
    "Overview",
    "Lifecycle",
    "Validation",
    "Probe",
    "Tokens",
    "Artifacts",
    "Prompt preview",
    "Response preview",
)
TERMINAL_EVENTS = {
    "experiment_completed",
    "experiment_interrupted",
    "experiment_failed",
}
ACTIVE_STATES = {
    "starting",
    "model",
    "retrying",
    "repair",
    "validating",
    "probing",
    "evaluating",
}
STATE_STYLES = {
    "queued": "grey62",
    "starting": "cyan",
    "model": "cyan",
    "retrying": "yellow",
    "repair": "yellow",
    "validating": "blue",
    "probing": "magenta",
    "evaluating": "dark_orange",
    "accepted": "green",
    "duplicate": "dim blue",
    "invalid": "red",
    "failed": "bright_red",
    "recovered": "dim green",
    "budget": "yellow",
    "budget_exhausted": "yellow",
    "interrupted": "magenta",
}
LIFECYCLE_PHASES = (
    "queued",
    "provider",
    "response",
    "schema",
    "AST",
    "probe",
    "evaluation",
    "archived",
)
PANEL_COPY_KEYS = {
    "1": "header",
    "2": "progress",
    "3": "slots",
    "4": "performance",
    "5": "tokens",
    "6": "objective",
    "7": "activity",
    "8": "quick-view",
}
PANEL_COPY_WIDTHS = {
    "header": 150,
    "progress": 150,
    "slots": 150,
    "performance": 60,
    "tokens": 60,
    "objective": 60,
    "activity": 100,
    "quick-view": 80,
}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input: int | None = None
    cached: int | None = None
    output: int | None = None
    reasoning: int | None = None
    total: int | None = None
    quality: str = "unknown"


@dataclass(frozen=True, slots=True)
class LifecycleStep:
    phase: str
    status: str
    elapsed_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class DashboardSlot:
    slot: str
    generation: int
    parent: str = "root"
    phase: str = "queued"
    state: str = "queued"
    started_monotonic: float | None = None
    elapsed_seconds: float | None = None
    provider_request_id: str | None = None
    provider_thread_id: str | None = None
    provider_turn_id: str | None = None
    repairs: int = 0
    timeout_seconds: float | None = None
    usage: TokenUsage = TokenUsage()
    validation: str = "—"
    validation_message: str = ""
    probe: str = "—"
    probe_message: str = ""
    candidate: str = ""
    error: str = ""
    objective: float | None = None
    retryable: bool = False
    charged: bool | None = None
    lifecycle: tuple[LifecycleStep, ...] = ()
    artifacts: tuple[str, ...] = ()
    prompt_preview: str = ""
    response_preview: str = ""


@dataclass(frozen=True, slots=True)
class GenerationSlots:
    generation: int
    slots: tuple[DashboardSlot, ...]


@dataclass(frozen=True, slots=True)
class ActivityEntry:
    timestamp: str
    component: str
    severity: str
    message: str
    slot: str | None = None


@dataclass(frozen=True, slots=True)
class DashboardState:
    run_id: str = "initializing"
    session_id: str = "—"
    experiment_state: str = "starting"
    run_mode: str = "—"
    model: str = "—"
    effort: str = "—"
    phase: str = "initial"
    checkpoint: str = "—"
    started_at: str = "—"
    started_monotonic: float | None = None
    elapsed_seconds: float = 0.0
    wall_seconds: float | None = None
    generation: int = 0
    generation_limit: int | None = None
    displayed_generation: int = 0
    population_size: int = 8
    completed_slots: int = 0
    provider_turns_attempted: int = 0
    provider_turns_completed: int = 0
    max_model_turns: int | None = None
    active_provider_turns: int = 0
    configured_provider_concurrency: int | None = None
    evaluations_completed: int = 0
    evaluation_episodes_completed: int = 0
    evaluation_episodes_total: int | None = None
    evaluation_workers_active: int | None = None
    evaluation_workers_configured: int | None = None
    archive_size: int = 0
    accepted_candidates: int = 0
    invalid_candidates: int = 0
    failed_candidates: int = 0
    duplicate_candidates: int = 0
    current_objective: float | None = None
    best_objective: float | None = None
    best_candidate: str = "—"
    objective_direction: str = "maximize"
    baseline_random: float | None = None
    baseline_structural: float | None = None
    improvement_rate: float | None = None
    evaluation_rate: float | None = None
    episode_rate: float | None = None
    profiling_enabled: bool = False
    timing_profile: Mapping[str, JsonValue] | None = None
    cumulative_usage: TokenUsage = TokenUsage()
    session_usage: TokenUsage = TokenUsage()
    usage_seen: frozenset[tuple[str, str, str]] = frozenset()
    archive_seen: frozenset[tuple[str, str, str]] = frozenset()
    failed_slots_seen: frozenset[tuple[int, str]] = frozenset()
    generations: tuple[GenerationSlots, ...] = ()
    selected_index: int = 0
    view: str = "matrix"
    detail_tab: int = 0
    activity: tuple[ActivityEntry, ...] = ()
    logs: tuple[ActivityEntry, ...] = ()
    objective_history: tuple[float, ...] = ()
    search_query: str = ""
    search_editing: bool = False
    retry_confirmation: bool = False
    paused: bool = False
    status_message: str = ""


@dataclass(frozen=True, slots=True)
class DashboardAction:
    kind: Literal["quit", "pause", "resume", "retry", "copy"]
    slot: str | None = None
    panel: str | None = None


@dataclass(frozen=True, slots=True)
class DashboardKey:
    value: str


@dataclass(frozen=True, slots=True)
class DashboardCapabilities:
    pause: Callable[[bool], None] | None = None
    retry: Callable[[str], None] | None = None
    quit: Callable[[], None] | None = None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _usage(value: object, *, quality: object = None) -> TokenUsage:
    source = value if isinstance(value, Mapping) else {}
    return TokenUsage(
        input=_integer(source.get("inputTokens")),
        cached=_integer(source.get("cachedInputTokens")),
        output=_integer(source.get("outputTokens")),
        reasoning=_integer(source.get("reasoningOutputTokens")),
        total=_integer(source.get("totalTokens")),
        quality=str(quality or source.get("quality") or "unknown"),
    )


def _add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    def add(a: int | None, b: int | None) -> int | None:
        if a is None and b is None:
            return None
        return (a or 0) + (b or 0)

    return TokenUsage(
        input=add(left.input, right.input),
        cached=add(left.cached, right.cached),
        output=add(left.output, right.output),
        reasoning=add(left.reasoning, right.reasoning),
        total=add(left.total, right.total),
        quality=right.quality if right.quality != "unknown" else left.quality,
    )


def _slot_count(payload: Mapping[str, JsonValue], current: int) -> int:
    value = _integer(payload.get("population_size"))
    return max(1, value) if value is not None else current


def _empty_slots(generation: int, count: int) -> tuple[DashboardSlot, ...]:
    return tuple(
        DashboardSlot(slot=f"slot-{index:02d}", generation=generation)
        for index in range(count)
    )


def _generation_slots(state: DashboardState, generation: int) -> GenerationSlots:
    existing = next((item for item in state.generations if item.generation == generation), None)
    return existing or GenerationSlots(generation, _empty_slots(generation, state.population_size))


def _replace_generation(
    state: DashboardState,
    generation: int,
    slots: tuple[DashboardSlot, ...],
) -> DashboardState:
    replacement = GenerationSlots(generation, slots)
    values = [item for item in state.generations if item.generation != generation]
    values.append(replacement)
    values.sort(key=lambda item: item.generation)
    return replace(state, generations=tuple(values))


def _replace_slot(state: DashboardState, updated: DashboardSlot) -> DashboardState:
    group = _generation_slots(state, updated.generation)
    slots = list(group.slots)
    index = next((i for i, item in enumerate(slots) if item.slot == updated.slot), None)
    if index is None:
        slots.append(updated)
        slots.sort(key=lambda item: item.slot)
    else:
        slots[index] = updated
    return _replace_generation(state, updated.generation, tuple(slots))


def _selected_slot(state: DashboardState) -> DashboardSlot | None:
    group = _generation_slots(state, state.displayed_generation)
    if not group.slots:
        return None
    return group.slots[min(state.selected_index, len(group.slots) - 1)]


def _lifecycle(
    slot: DashboardSlot,
    phase: str,
    status: str,
    elapsed: float | None = None,
) -> tuple[LifecycleStep, ...]:
    values = [item for item in slot.lifecycle if item.phase != phase]
    values.append(LifecycleStep(phase, status, elapsed))
    order = {name.lower(): index for index, name in enumerate(LIFECYCLE_PHASES)}
    values.sort(key=lambda item: order.get(item.phase.lower(), len(order)))
    return tuple(values)


def _diagnostic(payload: Mapping[str, JsonValue]) -> tuple[str, str]:
    codes = payload.get("validation_codes")
    code_values = (
        [str(item) for item in codes if isinstance(item, str) and item]
        if isinstance(codes, list)
        else []
    )
    errors = payload.get("errors")
    messages: list[str] = []
    if isinstance(errors, list):
        for item in errors:
            if isinstance(item, Mapping):
                code = _text(item.get("code"))
                message = _text(item.get("message"))
                if code and code not in code_values:
                    code_values.append(code)
                if message:
                    messages.append(message)
            elif isinstance(item, str):
                messages.append(item)
    error = _text(payload.get("error"))
    if error:
        messages.append(error)
    return ", ".join(code_values), "; ".join(messages)


def _retryable_provider_failure(payload: Mapping[str, JsonValue]) -> bool:
    status = str(payload.get("status", "")).lower()
    usage = payload.get("usage")
    token_count = 0
    if isinstance(usage, Mapping):
        token_count = sum(
            value
            for key, value in usage.items()
            if isinstance(key, str)
            and key.lower().endswith("tokens")
            and isinstance(value, int)
            and not isinstance(value, bool)
        )
    return (
        status in {"infrastructure", "transport_error", "unavailable", "retryable"}
        and payload.get("accepted") is not True
        and payload.get("charged") is False
        and payload.get("content") is False
        and payload.get("uncharged") is True
        and payload.get("unauthorized_tool_approval") is not True
        and token_count == 0
        and payload.get("response") in (None, "", {}, [])
    )


def _event_activity(event: Event) -> ActivityEntry | None:
    payload = event.payload
    slot = _text(payload.get("slot"))
    event_type = event.event_type
    if event_type in {"provider_turn_activity", "repair_activity"}:
        elapsed = _number(payload.get("operation_elapsed_seconds"))
        message = f"waiting for provider response{f' ({elapsed:.0f}s)' if elapsed else ''}"
        return ActivityEntry(event.timestamp[11:19], "provider", "info", message, slot)
    if event_type == "provider_turn_failed":
        return ActivityEntry(
            event.timestamp[11:19],
            "provider",
            "error",
            _text(payload.get("error")) or "provider turn failed",
            slot,
        )
    if event_type in {"experiment_failed", "experiment_interrupted"}:
        return ActivityEntry(
            event.timestamp[11:19],
            "experiment",
            "error" if event_type.endswith("failed") else "warning",
            _text(payload.get("error")) or event_type.replace("_", " "),
            slot,
        )
    meaningful = {
        "preflight_started",
        "preflight_completed",
        "workspace_initialized",
        "workspace_resumed",
        "session_started",
        "generation_started",
        "slot_queued",
        "provider_turn_started",
        "provider_turn_completed",
        "repair_started",
        "repair_completed",
        "validation_completed",
        "behavior_probe_completed",
        "candidate_archived",
        "evaluation_started",
        "evaluation_completed",
        "evaluation_failed",
        "selection_completed",
        "checkpoint_written",
        "budget_boundary_reached",
        "experiment_completed",
    }
    if event_type not in meaningful:
        return None
    severity = "error" if event_type.endswith("failed") else "info"
    status = _text(payload.get("status"))
    message = event_type.replace("_", " ")
    if status:
        message += f" · {status}"
    return ActivityEntry(
        event.timestamp[11:19],
        event_type.split("_", 1)[0],
        severity,
        message,
        slot,
    )


def _global_payload(state: DashboardState, payload: Mapping[str, JsonValue]) -> DashboardState:
    values: dict[str, Any] = {}
    mappings = {
        "session_id": "session_id",
        "run_mode": "run_mode",
        "model": "model",
        "effort": "effort",
        "phase": "phase",
        "checkpoint": "checkpoint",
    }
    for source, target in mappings.items():
        text_value = _text(payload.get(source))
        if text_value is not None:
            values[target] = text_value
    integers = {
        "generation": "generation",
        "generation_limit": "generation_limit",
        "population_size": "population_size",
        "max_model_turns": "max_model_turns",
        "configured_concurrency": "configured_provider_concurrency",
        "effective_concurrency": "configured_provider_concurrency",
        "archive_size": "archive_size",
        "evaluations_completed": "evaluations_completed",
        "active_workers": "evaluation_workers_active",
        "worker_count": "evaluation_workers_configured",
    }
    for source, target in integers.items():
        integer_value = _integer(payload.get(source))
        if integer_value is not None:
            values[target] = integer_value
    numbers = {
        "configured_wall_seconds": "wall_seconds",
        "elapsed_seconds": "elapsed_seconds",
        "evaluations_per_second": "evaluation_rate",
        "episodes_per_second": "episode_rate",
        "ir": "improvement_rate",
        "improvement_rate": "improvement_rate",
    }
    for source, target in numbers.items():
        number_value = _number(payload.get(source))
        if number_value is not None:
            values[target] = number_value
    profile = payload.get("timing_profile")
    if isinstance(profile, Mapping):
        values["timing_profile"] = profile
        values["profiling_enabled"] = profile.get("enabled") is True
    profiling = payload.get("profiling_enabled")
    if isinstance(profiling, bool):
        values["profiling_enabled"] = profiling
    return replace(state, **values)


def reduce_dashboard_event(
    state: DashboardState,
    event: Event,
    *,
    monotonic: float | None = None,
) -> DashboardState:
    """Fold one canonical native event into an immutable presentation state."""

    now = time.monotonic() if monotonic is None else monotonic
    payload = event.payload
    state = _global_payload(replace(state, run_id=event.run_id), payload)
    event_type = event.event_type
    if event_type == "session_started":
        cumulative = _usage(payload.get("usage"))
        session = _usage(payload.get("session_usage"))
        model_turns_used = (
            _integer(payload.get("model_turns_used"))
            or _integer(payload.get("cumulative_provider_turns"))
            or 0
        )
        state = replace(
            state,
            run_id=_text(payload.get("experiment_id")) or event.run_id,
            experiment_state="running",
            started_at=event.timestamp[11:19],
            started_monotonic=now,
            cumulative_usage=cumulative,
            session_usage=session,
            provider_turns_attempted=model_turns_used,
            provider_turns_completed=(
                _integer(payload.get("cumulative_provider_turns")) or 0
            ),
        )
    elif event_type == "generation_started":
        generation = _integer(payload.get("generation")) or 0
        count = _slot_count(payload, state.population_size)
        state = replace(
            state,
            generation=generation,
            displayed_generation=generation,
            population_size=count,
            completed_slots=0,
            selected_index=0,
            view="matrix" if state.view == "details" else state.view,
        )
        state = _replace_generation(state, generation, _empty_slots(generation, count))
    elif event_type in {"slot_queued", "generation_completed"}:
        event_generation = _integer(payload.get("generation"))
        completed = _integer(payload.get("completed_slots"))
        if event_generation == state.generation and completed is not None:
            state = replace(
                state,
                completed_slots=min(
                    state.population_size,
                    max(state.completed_slots, completed),
                ),
            )
    elif event_type == "provider_turn_started":
        state = replace(
            state,
            active_provider_turns=state.active_provider_turns + 1,
            provider_turns_attempted=state.provider_turns_attempted + 1,
            phase="repair" if payload.get("phase") == "repair" else "provider",
        )
    elif event_type in {"provider_turn_completed", "provider_turn_failed"}:
        state = replace(
            state,
            active_provider_turns=max(0, state.active_provider_turns - 1),
            provider_turns_completed=(
                state.provider_turns_completed + (1 if event_type.endswith("completed") else 0)
            ),
        )
        usage_key = (
            str(payload.get("generation", "")),
            str(payload.get("slot", "")),
            str(
                payload.get(
                    "idempotency_key",
                    payload.get("provider_turn_id", payload.get("phase", "initial")),
                )
            ),
        )
        if (
            payload.get("retained") is not True
            and usage_key not in state.usage_seen
            and isinstance(payload.get("usage"), Mapping)
        ):
            delta = _usage(payload.get("usage"), quality=payload.get("usage_quality"))
            state = replace(
                state,
                cumulative_usage=_add_usage(state.cumulative_usage, delta),
                session_usage=_add_usage(state.session_usage, delta),
                usage_seen=state.usage_seen | {usage_key},
            )
        if event_type == "provider_turn_failed":
            failed_generation = _integer(payload.get("generation"))
            failed_slot = _text(payload.get("slot"))
            failed_key = (
                (failed_generation, failed_slot)
                if failed_generation is not None and failed_slot is not None
                else None
            )
            if (
                failed_key is not None
                and failed_key not in state.failed_slots_seen
                and str(payload.get("status", "")).lower()
                in {"infrastructure", "transport_error", "unavailable"}
            ):
                state = replace(
                    state,
                    failed_candidates=state.failed_candidates + 1,
                    failed_slots_seen=state.failed_slots_seen | {failed_key},
                )
    elif event_type == "evaluation_started":
        total = _integer(payload.get("evaluation_total"))
        state = replace(
            state,
            evaluation_episodes_completed=0,
            evaluation_episodes_total=total or state.evaluation_episodes_total,
        )
    elif event_type == "evaluation_progress":
        completed = _integer(payload.get("completed"))
        total = _integer(payload.get("total")) or _integer(payload.get("evaluation_total"))
        state = replace(
            state,
            evaluation_episodes_completed=max(
                state.evaluation_episodes_completed, completed or 0
            ),
            evaluation_episodes_total=total or state.evaluation_episodes_total,
        )
    elif event_type in {"evaluation_completed", "evaluation_failed"}:
        explicit_evaluations = _integer(payload.get("evaluations_completed"))
        state = replace(
            state,
            evaluations_completed=(
                explicit_evaluations
                if explicit_evaluations is not None
                else state.evaluations_completed + 1
            ),
        )
        current = _number(payload.get("current_objective", payload.get("mean_auc")))
        best = _number(payload.get("best_objective", payload.get("best_auc")))
        comparison = payload.get("baseline_comparison")
        random_baseline = state.baseline_random
        structural_baseline = state.baseline_structural
        if isinstance(comparison, Mapping):
            random_value = _number(comparison.get("random"))
            structural_value = _number(comparison.get("structural"))
            if random_value is not None:
                random_baseline = random_value
            if structural_value is not None:
                structural_baseline = structural_value
        history = state.objective_history
        if current is not None and event_type == "evaluation_completed":
            history = (*history[-59:], current)
        state = replace(
            state,
            current_objective=current if current is not None else state.current_objective,
            best_objective=best if best is not None else state.best_objective,
            best_candidate=_text(payload.get("best_candidate_id")) or state.best_candidate,
            baseline_random=random_baseline,
            baseline_structural=structural_baseline,
            objective_history=history,
        )
    elif event_type == "candidate_archived":
        status = _text(payload.get("status"))
        archive_key = (
            str(payload.get("generation", "")),
            str(payload.get("slot", payload.get("candidate_id", ""))),
            status or "",
        )
        if archive_key not in state.archive_seen:
            if status == "accepted":
                state = replace(state, accepted_candidates=state.accepted_candidates + 1)
            elif status == "invalid":
                state = replace(state, invalid_candidates=state.invalid_candidates + 1)
            elif status == "failed":
                state = replace(state, failed_candidates=state.failed_candidates + 1)
            elif status == "duplicate":
                state = replace(state, duplicate_candidates=state.duplicate_candidates + 1)
            state = replace(state, archive_seen=state.archive_seen | {archive_key})
    elif event_type == "budget_boundary_reached":
        state = replace(
            state,
            experiment_state=_text(payload.get("state")) or "idle",
            phase="budget",
        )
    elif event_type == "experiment_completed":
        state = replace(state, experiment_state="completed", phase="completed")
    elif event_type == "experiment_interrupted":
        state = replace(state, experiment_state="interrupted", phase="interrupted")
    elif event_type == "experiment_failed":
        state = replace(state, experiment_state="failed", phase="failed")

    slot_name = _text(payload.get("slot"))
    if slot_name is not None:
        event_generation = _integer(payload.get("generation"))
        generation = state.generation if event_generation is None else event_generation
        group = _generation_slots(state, generation)
        slot = next(
            (item for item in group.slots if item.slot == slot_name),
            DashboardSlot(slot=slot_name, generation=generation),
        )
        parent = _text(payload.get("parent_id")) or slot.parent
        phase = _text(payload.get("phase")) or slot.phase
        slot_state = _text(payload.get("status")) or slot.state
        lifecycle_phase = phase
        lifecycle_status = "running"
        started = slot.started_monotonic
        elapsed = slot.elapsed_seconds
        retryable = slot.retryable
        error = slot.error
        validation, validation_message = slot.validation, slot.validation_message
        probe, probe_message = slot.probe, slot.probe_message
        if event_type == "slot_queued":
            lifecycle_phase = "queued"
            lifecycle_status = "pass"
            if payload.get("status") == "retrying":
                slot_state = "retrying"
                error = ""
                retryable = False
        elif event_type == "provider_turn_started":
            slot_state = "repair" if phase == "repair" else "model"
            lifecycle_phase = "provider"
            started = now
        elif event_type == "provider_turn_completed":
            slot_state = "validating" if payload.get("accepted") is True else "failed"
            lifecycle_phase = "response"
            lifecycle_status = "pass" if payload.get("accepted") is True else "fail"
            elapsed = max(0.0, now - started) if started is not None else elapsed
        elif event_type == "provider_turn_failed":
            slot_state = "failed"
            lifecycle_phase = "response"
            lifecycle_status = "fail"
            retryable = _retryable_provider_failure(payload)
            error = _text(payload.get("error")) or "provider turn failed"
            elapsed = max(0.0, now - started) if started is not None else elapsed
        elif event_type == "repair_started":
            slot_state = "repair"
            lifecycle_phase = "provider"
            started = now
        elif event_type == "validation_started":
            slot_state = "validating"
            lifecycle_phase = "schema"
        elif event_type == "validation_completed":
            valid = payload.get("valid") is True
            slot_state = "probing" if valid else "invalid"
            lifecycle_phase = "schema"
            lifecycle_status = "pass" if valid else "fail"
            code, message = _diagnostic(payload)
            validation = "pass" if valid else code or "fail"
            validation_message = message
            if not valid:
                error = code or message or "validation failed"
        elif event_type == "behavior_probe_started":
            slot_state = "probing"
            lifecycle_phase = "probe"
        elif event_type == "behavior_probe_completed":
            valid = payload.get("valid") is True
            slot_state = "evaluating" if valid else "invalid"
            lifecycle_phase = "probe"
            lifecycle_status = "pass" if valid else "fail"
            code, message = _diagnostic(payload)
            probe = "pass" if valid else code or "fail"
            probe_message = message
            if not valid:
                error = code or message or "probe failed"
        elif event_type == "evaluation_started":
            slot_state = "evaluating"
            lifecycle_phase = "evaluation"
        elif event_type == "evaluation_completed":
            slot_state = "accepted"
            lifecycle_phase = "evaluation"
            lifecycle_status = "pass"
        elif event_type == "evaluation_failed":
            slot_state = "failed"
            lifecycle_phase = "evaluation"
            lifecycle_status = "fail"
            error = _text(payload.get("error")) or "evaluation failed"
        elif event_type == "candidate_archived":
            archived_state = _text(payload.get("status"))
            if not (
                archived_state == "invalid"
                and slot.state == "failed"
                and slot.retryable
            ):
                slot_state = archived_state or slot_state
            lifecycle_phase = "archived"
            lifecycle_status = "pass" if slot_state in {"accepted", "duplicate"} else "fail"
        usage = (
            _usage(payload.get("usage"), quality=payload.get("usage_quality"))
            if isinstance(payload.get("usage"), Mapping)
            else slot.usage
        )
        artifacts_value = payload.get("artifacts")
        artifacts = slot.artifacts
        if isinstance(artifacts_value, list):
            artifacts = tuple(str(item) for item in artifacts_value if isinstance(item, str))
        charged_value = payload.get("charged")
        charged = charged_value if isinstance(charged_value, bool) else slot.charged
        objective = _number(payload.get("current_objective"))
        if objective is None:
            objective = _number(payload.get("best_score"))
        if objective is None:
            objective = slot.objective
        updated = replace(
            slot,
            parent=parent,
            phase=phase,
            state=slot_state,
            started_monotonic=started,
            elapsed_seconds=elapsed,
            provider_request_id=(
                _text(payload.get("provider_request_id")) or slot.provider_request_id
            ),
            provider_thread_id=_text(payload.get("provider_thread_id")) or slot.provider_thread_id,
            provider_turn_id=_text(payload.get("provider_turn_id")) or slot.provider_turn_id,
            repairs=_integer(payload.get("repairs")) or slot.repairs,
            timeout_seconds=_number(payload.get("timeout_seconds")) or slot.timeout_seconds,
            usage=usage,
            validation=validation,
            validation_message=validation_message,
            probe=probe,
            probe_message=probe_message,
            candidate=_text(payload.get("candidate_id")) or slot.candidate,
            error=error,
            objective=objective,
            retryable=retryable,
            charged=charged,
            lifecycle=_lifecycle(slot, lifecycle_phase, lifecycle_status, elapsed),
            artifacts=artifacts,
            prompt_preview=_text(payload.get("prompt_preview")) or slot.prompt_preview,
            response_preview=_text(payload.get("response_preview")) or slot.response_preview,
        )
        state = _replace_slot(state, updated)

    activity = _event_activity(event)
    if activity is not None:
        recent = list(state.activity)
        if (
            event_type in {"provider_turn_activity", "repair_activity"}
            and recent
            and recent[0].slot == activity.slot
            and "waiting for provider" in recent[0].message
        ):
            recent[0] = activity
        else:
            recent.insert(0, activity)
        logs = (*state.logs[-199:], activity)
        state = replace(state, activity=tuple(recent[:10]), logs=logs)
    return state


def reduce_dashboard_key(
    state: DashboardState,
    key: str,
    *,
    pause_supported: bool = False,
    retry_supported: bool = False,
) -> tuple[DashboardState, DashboardAction | None]:
    """Reduce one decoded key without touching the experiment scheduler."""

    if state.retry_confirmation:
        if key.lower() == "y":
            slot = _selected_slot(state)
            if slot is not None and slot.retryable and retry_supported:
                return (
                    replace(
                        state,
                        retry_confirmation=False,
                        status_message=f"Retry requested for {slot.slot}",
                    ),
                    DashboardAction("retry", slot.slot),
                )
        return (
            replace(state, retry_confirmation=False, status_message="Retry cancelled"),
            None,
        )
    if state.search_editing:
        if key == "ESC":
            return replace(state, search_editing=False, status_message="Search cancelled"), None
        if key == "ENTER":
            return replace(state, search_editing=False, status_message="Filter applied"), None
        if key == "BACKSPACE":
            return replace(state, search_query=state.search_query[:-1]), None
        if len(key) == 1 and key.isprintable():
            return replace(state, search_query=state.search_query + key), None
        return state, None
    if key in PANEL_COPY_KEYS:
        panel = PANEL_COPY_KEYS[key]
        return (
            replace(state, status_message=f"Preparing panel {key} copy"),
            DashboardAction("copy", panel=panel),
        )
    if key in {"UP", "k", "DOWN", "j", "HOME", "END"}:
        group = _generation_slots(state, state.displayed_generation)
        last = max(0, len(group.slots) - 1)
        index = state.selected_index
        if key in {"UP", "k"}:
            index = max(0, index - 1)
        elif key in {"DOWN", "j"}:
            index = min(last, index + 1)
        elif key == "HOME":
            index = 0
        else:
            index = last
        return replace(state, selected_index=index, retry_confirmation=False), None
    if key in {"ENTER", "RIGHT"}:
        return replace(state, view="details", status_message=""), None
    if key in {"ESC", "LEFT"}:
        return replace(state, view="matrix", retry_confirmation=False), None
    if key == "TAB" and state.view == "details":
        return replace(state, detail_tab=(state.detail_tab + 1) % len(DETAIL_TABS)), None
    if key == "SHIFT_TAB" and state.view == "details":
        return replace(state, detail_tab=(state.detail_tab - 1) % len(DETAIL_TABS)), None
    if key in {"n", "N"}:
        generations = [item.generation for item in state.generations]
        if not generations:
            return replace(state, status_message="No retained generation"), None
        current = generations.index(state.displayed_generation)
        delta = -1 if key == "N" else 1
        target = generations[(current + delta) % len(generations)]
        return replace(
            state,
            displayed_generation=target,
            selected_index=0,
            view="matrix",
            status_message=f"Viewing generation {target}",
        ), None
    if key == "q":
        return (
            replace(state, experiment_state="stopping", status_message="Graceful stop requested"),
            DashboardAction("quit"),
        )
    if key == "p":
        if not pause_supported:
            return (
                replace(
                    state,
                    status_message="Pause unavailable: scheduler control not connected",
                ),
                None,
            )
        paused = not state.paused
        return (
            replace(
                state,
                paused=paused,
                experiment_state="paused" if paused else "running",
                status_message="Scheduling paused" if paused else "Scheduling resumed",
            ),
            DashboardAction("pause" if paused else "resume"),
        )
    if key == "r":
        slot = _selected_slot(state)
        if slot is None or not slot.retryable:
            return replace(state, status_message="Selected slot is not retryable"), None
        if not retry_supported:
            return (
                replace(
                    state,
                    status_message="Retry unavailable: scheduler control not connected",
                ),
                None,
            )
        return replace(state, retry_confirmation=True, status_message="Confirm retry [y/N]"), None
    if key == "c":
        return replace(state, view="config", status_message="Read-only locked configuration"), None
    if key == "l":
        return replace(state, view="logs", status_message="In-memory canonical event log"), None
    if key == "t":
        if not state.profiling_enabled or state.timing_profile is None:
            return replace(state, status_message="Profiling unavailable"), None
        return replace(state, view="top", status_message="Profiling top view"), None
    if key == "/":
        return replace(state, search_editing=True, status_message="Search presentation only"), None
    if key == "h":
        return replace(state, view="help", status_message="Help"), None
    return state, None


def reduce_dashboard(
    state: DashboardState,
    item: Event | DashboardKey,
    *,
    monotonic: float | None = None,
    pause_supported: bool = False,
    retry_supported: bool = False,
) -> tuple[DashboardState, DashboardAction | None]:
    """Single production reducer for canonical events and operator input."""

    if isinstance(item, Event):
        return reduce_dashboard_event(state, item, monotonic=monotonic), None
    return reduce_dashboard_key(
        state,
        item.value,
        pause_supported=pause_supported,
        retry_supported=retry_supported,
    )


class _TerminalInput:
    def __init__(self, stream: TextIO, callback: Callable[[str], None]) -> None:
        self.stream = stream
        self.callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._original: list[Any] | None = None

    def start(self) -> None:
        try:
            fd = self.stream.fileno()
        except (AttributeError, OSError):
            return
        if not self.stream.isatty():
            return
        self._fd = fd
        self._original = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._thread = threading.Thread(
            target=self._read_loop,
            name="mforge-dashboard-input",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        buffer = b""
        while not self._stop.is_set() and self._fd is not None:
            readable, _, _ = select.select([self._fd], [], [], 0.1)
            if not readable:
                if buffer == b"\x1b":
                    self.callback("ESC")
                    buffer = b""
                continue
            chunk = os.read(self._fd, 32)
            if not chunk:
                return
            buffer += chunk
            keys, buffer = _decode_keys(buffer)
            for key in keys:
                self.callback(key)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        if self._fd is not None and self._original is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)


def _decode_keys(value: bytes) -> tuple[list[str], bytes]:
    sequences = {
        b"\x1b[A": "UP",
        b"\x1b[B": "DOWN",
        b"\x1b[C": "RIGHT",
        b"\x1b[D": "LEFT",
        b"\x1b[H": "HOME",
        b"\x1b[F": "END",
        b"\x1b[1~": "HOME",
        b"\x1b[4~": "END",
        b"\x1b[5~": "PAGE_UP",
        b"\x1b[6~": "PAGE_DOWN",
        b"\x1b[Z": "SHIFT_TAB",
    }
    keys: list[str] = []
    while value:
        matched = next((raw for raw in sequences if value.startswith(raw)), None)
        if matched is not None:
            keys.append(sequences[matched])
            value = value[len(matched) :]
            continue
        if value.startswith(b"\x1b"):
            if len(value) == 1:
                return keys, value
            keys.append("ESC")
            value = value[1:]
            continue
        byte = value[:1]
        value = value[1:]
        if byte in {b"\r", b"\n"}:
            keys.append("ENTER")
        elif byte == b"\t":
            keys.append("TAB")
        elif byte in {b"\x7f", b"\x08"}:
            keys.append("BACKSPACE")
        else:
            keys.append(byte.decode("utf-8", errors="ignore"))
    return keys, b""


class InteractiveDashboardSink:
    """Optional Rich operator dashboard for native experiment events."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        input_stream: TextIO = sys.stdin,
        locked_config: Mapping[str, object] | None = None,
        capabilities: DashboardCapabilities | None = None,
        start_live: bool = True,
    ) -> None:
        self.console = console or Console()
        self.locked_config = dict(locked_config or {})
        self.capabilities = capabilities or DashboardCapabilities()
        self.state = DashboardState()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._dirty = threading.Event()
        self._rendered = threading.Event()
        self._rendered.set()
        self._closed = False
        self._pending_copy_action: str | None = None
        self._copy_notice_message: str | None = None
        self._copy_notice_until: float | None = None
        self._input = _TerminalInput(input_stream, self.handle_key)
        self.live = Live(
            self.render(),
            console=self.console,
            screen=self.console.is_terminal,
            auto_refresh=False,
            transient=False,
        )
        if start_live:
            self.live.start()
        self._live_started = start_live
        self._refresh_thread: threading.Thread | None = None
        if start_live:
            if self.console.is_terminal:
                self._input.start()
            self._refresh_thread = threading.Thread(
                target=self._refresh_loop,
                name="mforge-dashboard-refresh",
                daemon=True,
            )
            self._refresh_thread.start()

    def write(self, event: Event) -> None:
        with self._lock:
            self.state, _ = reduce_dashboard(self.state, event)
            self._rendered.clear()
            self._dirty.set()

    def handle_key(self, key: str) -> None:
        with self._lock:
            state, action = reduce_dashboard(
                self.state,
                DashboardKey(key),
                pause_supported=self.capabilities.pause is not None,
                retry_supported=self.capabilities.retry is not None,
            )
            self.state = state
            if action is not None and action.kind == "copy":
                self._pending_copy_action = action.panel
                action = None
            self._rendered.clear()
            self._dirty.set()
        if action is not None:
            self._dispatch(action)

    def _dispatch(self, action: DashboardAction) -> None:
        if action.kind == "quit":
            if self.capabilities.quit is not None:
                self.capabilities.quit()
            else:
                _thread.interrupt_main()
        elif action.kind in {"pause", "resume"} and self.capabilities.pause is not None:
            self.capabilities.pause(action.kind == "pause")
        elif (
            action.kind == "retry"
            and action.slot is not None
            and self.capabilities.retry is not None
        ):
            self.capabilities.retry(action.slot)

    def _refresh_loop(self) -> None:
        while not self._stop.is_set():
            changed = self._dirty.wait(REFRESH_INTERVAL_SECONDS)
            self._dirty.clear()
            with self._lock:
                notice_expired = self._expire_copy_notice_unlocked()
                active = self.state.experiment_state not in {
                    "completed",
                    "interrupted",
                    "failed",
                }
                if self._pending_copy_action is not None:
                    self._handle_pending_panel_copy_unlocked()
                    changed = True
                if changed or active or notice_expired:
                    self.live.update(self.render_unlocked(), refresh=True)
                self._rendered.set()

    def _handle_pending_panel_copy_unlocked(self) -> None:
        panel_name = self._pending_copy_action
        self._pending_copy_action = None
        if panel_name is None:
            return
        title, renderable = self._panel_copy_source(panel_name)
        text = render_panel_copy_text(
            title,
            renderable,
            width=PANEL_COPY_WIDTHS[panel_name],
        )
        try:
            path = save_panel_copy(
                panel_name,
                self.state.run_id,
                text,
                tmp_dir=PANEL_COPY_TMP_DIR,
            )
        except OSError as error:
            notice = f"Panel copy failed: {error.strerror or error.__class__.__name__}"
        else:
            try:
                osc52_sent = copy_text_to_clipboard_osc52(text)
            except OSError:
                osc52_sent = False
            notice = (
                f"OSC 52 sent · fallback {path}"
                if osc52_sent
                else f"OSC 52 unavailable · saved {path}"
            )
        self.state = replace(self.state, status_message=notice)
        self._copy_notice_message = notice
        self._copy_notice_until = time.monotonic() + COPY_NOTICE_SECONDS

    def _expire_copy_notice_unlocked(self) -> bool:
        if (
            self._copy_notice_until is None
            or time.monotonic() < self._copy_notice_until
        ):
            return False
        if self.state.status_message == self._copy_notice_message:
            self.state = replace(self.state, status_message="")
        self._copy_notice_message = None
        self._copy_notice_until = None
        return True

    def _panel_copy_source(self, panel_name: str) -> tuple[str, RenderableType]:
        if panel_name == "header":
            return "Experiment header", self._header(PANEL_COPY_WIDTHS[panel_name])
        if panel_name == "progress":
            return "Experiment progress", self._progress(
                PANEL_COPY_WIDTHS[panel_name],
                horizontal=True,
            )
        if panel_name == "slots":
            if self.state.view == "details":
                slot = _selected_slot(self.state)
                suffix = f" · {slot.slot}" if slot is not None else ""
                return (
                    f"Slot details{suffix}",
                    self._slot_details(PANEL_COPY_WIDTHS[panel_name], "copy"),
                )
            return (
                f"Slot matrix · generation {self.state.displayed_generation}",
                self._slot_matrix(PANEL_COPY_WIDTHS[panel_name], "full"),
            )
        if panel_name == "performance":
            return "Performance & IR", self._performance_panel()
        if panel_name == "tokens":
            return "Token Accounting", self._tokens_panel()
        if panel_name == "objective":
            return "Objective / Archive", self._objective_panel()
        if panel_name == "activity":
            return "Recent Activity", self._activity_panel("copy")
        if panel_name == "quick-view":
            return "Quick View", self._quick_view_panel("full")
        raise ValueError(f"Unknown panel copy target: {panel_name}")

    def render(self) -> Layout:
        with self._lock:
            return self.render_unlocked()

    def render_unlocked(self) -> Layout:
        width = max(40, self.console.size.width)
        height = max(12, self.console.size.height)
        mode = _responsive_mode(width, height)
        if self.state.view in {"config", "logs", "top", "help"}:
            return self._render_overlay(width, height, mode)
        if mode == "minimal":
            return self._render_minimal(width, height)
        return self._render_full_or_compact(width, height, mode)

    def _render_full_or_compact(self, width: int, height: int, mode: str) -> Layout:
        root = Layout(name="root")
        metric_rows = 1
        if mode == "full":
            root.split_column(
                Layout(_numbered_panel(self._header(width), "1"), name="header", size=5),
                Layout(
                    _numbered_panel(self._progress(width, horizontal=True), "2"),
                    name="progress",
                    size=5,
                ),
                Layout(name="main", size=13),
                Layout(name="metrics", size=14),
                Layout(name="bottom"),
                Layout(self._footer(width), name="footer", size=1),
            )
        else:
            metrics_size = min(13, max(3, height - 29))
            metric_rows = max(1, metrics_size - 2)
            root.split_column(
                Layout(
                    _numbered_panel(self._header(width, compact=True), "1"),
                    name="header",
                    size=4,
                ),
                Layout(
                    _numbered_panel(self._progress(width, horizontal=False), "2"),
                    name="progress",
                    size=7,
                ),
                Layout(name="main", size=13),
                Layout(name="metrics", size=metrics_size),
                Layout(name="bottom"),
                Layout(self._footer(width), name="footer", size=1),
            )
        root["main"].update(
            _numbered_panel(
                self._slot_details(width, mode)
                if self.state.view == "details"
                else self._slot_matrix(width, mode),
                "3",
            )
        )
        if mode == "full":
            panels = [
                Layout(_numbered_panel(self._performance_panel(), "4"), ratio=1),
                Layout(_numbered_panel(self._tokens_panel(), "5"), ratio=1),
                Layout(_numbered_panel(self._objective_panel(), "6"), ratio=1),
            ]
            if self.state.profiling_enabled and self.state.timing_profile is not None:
                panels.append(Layout(self._profiling_panel(), ratio=1))
            root["metrics"].split_row(*panels)
        else:
            root["metrics"].split_row(
                Layout(
                    _numbered_panel(
                        self._performance_panel(
                            compact=True,
                            row_limit=metric_rows,
                        ),
                        "4",
                    ),
                    ratio=1,
                ),
                Layout(
                    _numbered_panel(
                        self._tokens_panel(
                            compact=True,
                            row_limit=metric_rows,
                        ),
                        "5",
                    ),
                    ratio=1,
                ),
                Layout(
                    _numbered_panel(
                        self._objective_panel(
                            compact=True,
                            row_limit=metric_rows,
                        ),
                        "6",
                    ),
                    ratio=1,
                ),
            )
        root["bottom"].split_row(
            Layout(_numbered_panel(self._activity_panel(mode), "7"), ratio=3),
            Layout(_numbered_panel(self._quick_view_panel(mode), "8"), ratio=2),
        )
        return root

    def _render_minimal(self, width: int, height: int) -> Layout:
        root = Layout(name="root")
        root.split_column(
            Layout(_numbered_panel(self._header(width, minimal=True), "1"), size=3),
            Layout(
                _numbered_panel(self._progress(width, horizontal=False), "2"),
                size=7,
            ),
            Layout(_numbered_panel(self._minimal_slot(), "3"), size=5),
            Layout(_numbered_panel(self._activity_panel("minimal"), "7")),
            Layout(self._footer(width), size=1),
        )
        return root

    def _render_overlay(self, width: int, height: int, mode: str) -> Layout:
        root = Layout(name="root")
        header = self._header(
            width,
            compact=mode == "compact",
            minimal=mode == "minimal",
        )
        root.split_column(
            Layout(
                _numbered_panel(header, "1"),
                size=5 if mode == "full" else 4 if mode == "compact" else 3,
            ),
            Layout(
                _numbered_panel(
                    self._progress(width, horizontal=mode == "full"),
                    "2",
                ),
                size=5 if mode == "full" else 7,
            ),
            Layout(self._overlay_panel(width, height)),
            Layout(self._footer(width), size=1),
        )
        return root

    def _header(
        self,
        width: int,
        *,
        compact: bool = False,
        minimal: bool = False,
    ) -> Panel:
        state_style = {
            "running": "bold cyan",
            "paused": "bold yellow",
            "idle": "bold yellow",
            "completed": "bold green",
            "interrupted": "bold magenta",
            "failed": "bold red",
            "stopping": "bold yellow",
        }.get(self.state.experiment_state, "bold cyan")
        elapsed = self._elapsed()
        first = _parameter_line(
            (
                ("Run", self.state.run_id, None),
                ("State", self.state.experiment_state.upper(), state_style),
                (
                    "Gen",
                    f"{self.state.generation}/{_show(self.state.generation_limit)}",
                    None,
                ),
                ("Phase", self.state.phase, None),
            )
        )
        if minimal:
            return Panel(first, title="Mutation Forge Lab · Native experiment", border_style="cyan")
        evaluation_workers = _ratio(
            self.state.evaluation_workers_active,
            self.state.evaluation_workers_configured,
        )
        provider_concurrency = _ratio(
            self.state.active_provider_turns,
            self.state.configured_provider_concurrency,
        )
        second = _parameter_line(
            (
                ("Session", self.state.session_id, None),
                ("Mode", self.state.run_mode, None),
                ("Model", f"{self.state.model}/{self.state.effort}", None),
                ("Eval workers", evaluation_workers, None),
                ("Provider", provider_concurrency, None),
            )
        )
        if compact:
            return Panel(
                Group(Align.center(first), Align.center(second)),
                title="Mutation Forge Lab · Native experiment",
                border_style="cyan",
                padding=(0, 1),
            )
        third = _parameter_line(
            (
                ("Checkpoint", _safe_display_path(self.state.checkpoint), None),
                ("Started", self.state.started_at, None),
                ("Uptime", _duration(elapsed), None),
                ("Wall budget", _duration(self.state.wall_seconds), None),
            )
        )
        return Panel(
            Group(Align.center(first), Align.center(second), Align.center(third)),
            title="Mutation Forge Lab · Native experiment",
            border_style="cyan",
            padding=(0, 1),
        )

    def _progress(self, width: int, *, horizontal: bool) -> Panel:
        values = (
            ("Generation", self.state.generation, self.state.generation_limit),
            ("Slots Complete", self.state.completed_slots, self.state.population_size),
            (
                "Model Turn Budget",
                self.state.provider_turns_attempted,
                self.state.max_model_turns,
            ),
            (
                "Evaluation Progress",
                self.state.evaluation_episodes_completed,
                self.state.evaluation_episodes_total,
            ),
            (
                "Wall-time Budget",
                int(self._elapsed()),
                int(self.state.wall_seconds) if self.state.wall_seconds is not None else None,
            ),
        )
        renderables = [
            _progress_bar(
                label,
                current,
                total,
                width=8 if horizontal else max(12, width - 31),
                stacked=horizontal,
            )
            for label, current, total in values
        ]
        if horizontal:
            grid = Table.grid(expand=True)
            for _ in renderables:
                grid.add_column(ratio=1)
            grid.add_row(*renderables)
            content: RenderableType = grid
        else:
            content = Group(*renderables)
        return Panel(content, border_style="cyan", padding=(0, 1))

    def _slot_matrix(self, width: int, mode: str) -> Panel:
        group = _generation_slots(self.state, self.state.displayed_generation)
        table = Table(
            box=box.MINIMAL_HEAVY_HEAD,
            expand=True,
            padding=(0, 1),
            show_lines=False,
            row_styles=("", "dim"),
        )
        columns: list[tuple[str, Literal["left", "right"], int]] = [
            ("", "left", 1),
            ("slot", "left", 7),
            ("parent", "left", 13),
            ("phase", "left", 10),
            ("state", "left", 10),
            ("elapsed", "right", 7),
            ("in", "right", 7),
            ("out", "right", 7),
            ("total", "right", 7),
            ("validation", "left", 22),
            ("probe", "left", 14),
            ("candidate / error", "left", 24),
            ("objective ↑", "right", 12),
        ]
        if mode == "compact":
            visible = {
                "",
                "slot",
                "phase",
                "state",
                "elapsed",
                "total",
                "validation",
                "probe",
                "candidate / error",
            }
            columns = [
                item
                for item in columns
                if item[0] in visible
            ]
        for name, justify, max_width in columns:
            table.add_column(
                name,
                justify=justify,
                no_wrap=True,
                overflow="ellipsis",
                max_width=max_width,
            )
        for index, slot in enumerate(group.slots):
            selected = index == self.state.selected_index
            values = {
                "": "▶" if selected else "",
                "slot": slot.slot,
                "parent": _compact(slot.parent, 11),
                "phase": slot.phase,
                "state": Text(slot.state, style=STATE_STYLES.get(slot.state, "")),
                "elapsed": _duration(self._slot_elapsed(slot)),
                "in": _show(slot.usage.input),
                "out": _show(slot.usage.output),
                "total": _show(slot.usage.total),
                "validation": slot.validation,
                "probe": slot.probe,
                "candidate / error": slot.error or slot.candidate or "—",
                "objective ↑": _objective(slot.objective),
            }
            style = "bold on grey15" if selected else "bold" if slot.state in ACTIVE_STATES else ""
            table.add_row(*(values[name] for name, _, _ in columns), style=style)
        title = f"SLOT MATRIX ({len(group.slots)} total) · generation {group.generation}"
        if self.state.search_query:
            title += f" · filter {self.state.search_query!r}"
        return Panel(table, title=title, border_style="cyan", padding=(0, 0))

    def _minimal_slot(self) -> Panel:
        slot = _selected_slot(self.state)
        if slot is None:
            content = Text("No slot data", style="dim")
        else:
            content = Text()
            content.append(f"▶ {slot.slot}  ")
            content.append(slot.state, style=STATE_STYLES.get(slot.state, ""))
            content.append(
                f"  {slot.phase}  {_duration(self._slot_elapsed(slot))}  "
                f"tokens {_show(slot.usage.total)}\n"
            )
            content.append(slot.error or slot.candidate or "No result yet", style="dim")
        return Panel(content, title="SELECTED SLOT", border_style="cyan")

    def _slot_details(self, width: int, mode: str) -> Panel:
        slot = _selected_slot(self.state)
        if slot is None:
            return Panel("No selected slot", title="SLOT DETAILS", border_style="cyan")
        tab = DETAIL_TABS[self.state.detail_tab]
        tabs = Text("  ".join(f"[{name}]" if name == tab else name for name in DETAIL_TABS))
        body: RenderableType
        if tab == "Overview":
            body = _key_value_grid(
                (
                    ("slot", slot.slot),
                    ("parent/root", slot.parent),
                    ("generation", slot.generation),
                    ("state", slot.state),
                    ("current phase", slot.phase),
                    ("operation elapsed", _duration(self._slot_elapsed(slot))),
                    ("timeout", _duration(slot.timeout_seconds)),
                    ("provider request", slot.provider_request_id or "—"),
                    ("provider thread", slot.provider_thread_id or "—"),
                    ("provider turn", slot.provider_turn_id or "—"),
                    ("repairs", slot.repairs),
                    ("next action", _next_action(slot)),
                )
            )
        elif tab == "Lifecycle":
            table = Table.grid(expand=True)
            table.add_column()
            table.add_column()
            table.add_column(justify="right")
            known = {item.phase.lower(): item for item in slot.lifecycle}
            for phase in LIFECYCLE_PHASES:
                item = known.get(phase.lower())
                table.add_row(
                    phase,
                    item.status if item else "not-applicable",
                    _duration(item.elapsed_seconds) if item else "—",
                )
            body = table
        elif tab == "Validation":
            body = _key_value_grid(
                (("outcome/code", slot.validation), ("message", slot.validation_message or "—"))
            )
        elif tab == "Probe":
            body = _key_value_grid(
                (("outcome/code", slot.probe), ("message", slot.probe_message or "—"))
            )
        elif tab == "Tokens":
            body = _token_grid(slot.usage, charged=slot.charged)
        elif tab == "Artifacts":
            body = Text(
                "\n".join(_safe_display_path(item) for item in slot.artifacts)
                if slot.artifacts
                else "—",
                style="dim",
            )
        elif tab == "Prompt preview":
            body = Text(
                slot.prompt_preview
                if mode == "copy" and slot.prompt_preview
                else _bounded_preview(
                    slot.prompt_preview,
                    8 if mode == "full" else 5,
                ),
                style="" if slot.prompt_preview else "dim",
            )
        else:
            body = Text(
                slot.response_preview
                if mode == "copy" and slot.response_preview
                else _bounded_preview(
                    slot.response_preview,
                    8 if mode == "full" else 5,
                ),
                style="" if slot.response_preview else "dim",
            )
        return Panel(
            Group(tabs, Rule(style="cyan"), body),
            title=f"▶ SLOT DETAILS · {slot.slot}",
            border_style="cyan",
        )

    def _performance_panel(
        self,
        *,
        compact: bool = False,
        row_limit: int | None = None,
    ) -> Panel:
        elapsed = max(self._elapsed(), 0.001)
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            user, system = float(usage.ru_utime), float(usage.ru_stime)
        except (AttributeError, OSError):
            user, system = 0.0, 0.0
        rows: list[tuple[str, object]] = [
            ("eval/s", _rate(self.state.evaluation_rate)),
            ("episodes/s", _rate(self.state.episode_rate)),
            ("turn/s", _rate(self.state.provider_turns_completed / elapsed)),
            ("IR", _objective(self.state.improvement_rate)),
            (
                "eval workers",
                _ratio(
                    self.state.evaluation_workers_active,
                    self.state.evaluation_workers_configured,
                ),
            ),
            (
                "provider",
                _ratio(
                    self.state.active_provider_turns,
                    self.state.configured_provider_concurrency,
                ),
            ),
        ]
        if compact:
            rows = rows[: row_limit or 1]
        if not compact:
            rows.extend(
                (
                    ("wall/user/sys", f"{elapsed:.1f}/{user:.1f}/{system:.1f}s"),
                    ("active slots", sum(
                        slot.state in ACTIVE_STATES
                        for slot in _generation_slots(
                            self.state, self.state.generation
                        ).slots
                    )),
                )
            )
        return Panel(_key_value_grid(rows), title="Performance & IR", border_style="cyan")

    def _tokens_panel(
        self,
        *,
        compact: bool = False,
        row_limit: int | None = None,
    ) -> Panel:
        cumulative = self.state.cumulative_usage
        session = self.state.session_usage
        rows: list[tuple[str, object]] = [
            ("experiment total", _show(cumulative.total)),
            ("experiment input", _show(cumulative.input)),
            ("experiment cached", _show(cumulative.cached)),
            ("experiment output", _show(cumulative.output)),
            ("experiment reasoning", _show(cumulative.reasoning)),
            ("session total", _show(session.total)),
            ("session input", _show(session.input)),
            ("session cached", _show(session.cached)),
            ("session output", _show(session.output)),
            ("session reasoning", _show(session.reasoning)),
            ("usage quality", cumulative.quality),
        ]
        if compact:
            compact_rows = [
                rows[0],
                rows[5],
                rows[6],
                rows[8],
                rows[9],
                rows[10],
            ]
            rows = compact_rows[: row_limit or 1]
        return Panel(_key_value_grid(rows), title="Token Accounting", border_style="cyan")

    def _objective_panel(
        self,
        *,
        compact: bool = False,
        row_limit: int | None = None,
    ) -> Panel:
        rows: list[tuple[str, object]] = [
            ("direction", f"{self.state.objective_direction} ↑"),
            ("current", _objective(self.state.current_objective)),
            ("best", _objective(self.state.best_objective)),
            ("best candidate", self.state.best_candidate),
            ("random baseline", _objective(self.state.baseline_random)),
            ("structural baseline", _objective(self.state.baseline_structural)),
            ("accepted", self.state.accepted_candidates),
            ("invalid", self.state.invalid_candidates),
            ("failed", self.state.failed_candidates),
            ("duplicate", self.state.duplicate_candidates),
            ("archive size", self.state.archive_size),
        ]
        if compact:
            rows = rows[: row_limit or 1]
        return Panel(_key_value_grid(rows), title="Objective / Archive", border_style="cyan")

    def _profiling_panel(self) -> Panel:
        profile = self.state.timing_profile
        if not self.state.profiling_enabled or not isinstance(profile, Mapping):
            return Panel(
                Text("Profiling disabled", style="dim"),
                title="Profiling",
                border_style="cyan",
            )
        phases = profile.get("phase_seconds")
        calls = profile.get("phase_calls")
        if not isinstance(phases, Mapping):
            return Panel(
                Text("Waiting for profile data", style="dim"),
                title="Profiling",
                border_style="cyan",
            )
        rows = sorted(
            (
                (str(name), float(seconds))
                for name, seconds in phases.items()
                if isinstance(seconds, int | float) and not isinstance(seconds, bool)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
        measured = sum(value for _, value in rows) or 1.0
        table = Table.grid(expand=True)
        table.add_column(overflow="ellipsis")
        table.add_column(justify="right")
        table.add_column(justify="right")
        for name, seconds in rows:
            count = calls.get(name) if isinstance(calls, Mapping) else None
            table.add_row(name, f"{seconds / measured:5.1%}", _show(count))
        unattributed = _number(profile.get("unattributed_fraction"))
        if unattributed is not None:
            table.add_row("unattributed", f"{unattributed:5.1%}", "—")
        return Panel(table, title="Profiling · top-N", border_style="cyan")

    def _activity_panel(self, mode: str) -> Panel:
        if mode == "copy":
            limit = len(self.state.activity)
        elif mode == "full":
            limit = 8
        elif mode == "compact":
            limit = 4
        else:
            limit = 5
        query = self.state.search_query.lower()
        entries = [
            item
            for item in self.state.activity
            if not query
            or query in item.message.lower()
            or query in item.component.lower()
            or (item.slot is not None and query in item.slot.lower())
        ][:limit]
        table = Table.grid(expand=True)
        table.add_column(width=8, no_wrap=True)
        table.add_column(width=10, no_wrap=True)
        table.add_column(overflow="ellipsis")
        for item in entries:
            message = f"{item.slot} · {item.message}" if item.slot else item.message
            style = (
                "red"
                if item.severity == "error"
                else "yellow"
                if item.severity == "warning"
                else ""
            )
            table.add_row(item.timestamp, item.component, Text(message, style=style))
        if not entries:
            table.add_row("", "", Text("Waiting for meaningful events", style="dim"))
        title = "Recent Activity"
        if query:
            title += f" · filter {self.state.search_query!r}"
        return Panel(table, title=title, border_style="cyan")

    def _quick_view_panel(self, mode: str) -> Panel:
        group = _generation_slots(self.state, self.state.displayed_generation)
        active = sum(item.state in ACTIVE_STATES for item in group.slots)
        queued = sum(item.state == "queued" for item in group.slots)
        failed = sum(item.state == "failed" for item in group.slots)
        invalid = sum(item.state == "invalid" for item in group.slots)
        rows: tuple[tuple[str, object], ...] = (
            (
                "Gen / Turn / Slots",
                f"{self.state.displayed_generation} / "
                f"{self.state.provider_turns_attempted} / {len(group.slots)}",
            ),
            ("Best objective", _objective(self.state.best_objective)),
            ("Active / Queued", f"{active} / {queued}"),
            ("Failed / Invalid", f"{failed} / {invalid}"),
            ("Archive", self.state.archive_size),
        )
        if mode == "compact":
            rows = (rows[0],)
        summary = _key_value_grid(rows)
        if not self.state.objective_history:
            chart = Text("No evaluated objective history yet", style="dim")
        else:
            history = self.state.objective_history
            sparkline = _sparkline(history)
            if mode == "compact":
                chart = Text(
                    f"Objective n={len(history)}: "
                    f"{history[0]:.6f} → {history[-1]:.6f}  {sparkline}",
                    style="green",
                )
            else:
                chart = Text(
                    f"Objective history · oldest → latest · n={len(history)}\n"
                    f"min {min(history):.6f}  {sparkline}  max {max(history):.6f}",
                    style="green",
                )
        return Panel(Group(summary, chart), title="Quick View", border_style="cyan")

    def _overlay_panel(self, width: int, height: int) -> Panel:
        if self.state.view == "config":
            body = _flatten_config(self.locked_config)
            content: RenderableType = Text(body or "No locked configuration supplied", style="dim")
            title = "CONFIG · read-only"
        elif self.state.view == "logs":
            query = self.state.search_query.lower()
            entries = [
                item
                for item in self.state.logs
                if not query or query in item.message.lower() or query in item.component.lower()
            ]
            limit = max(3, height - 16)
            content = Text(
                "\n".join(
                    f"{item.timestamp} {item.component:<10} "
                    f"{(item.slot + ' ') if item.slot else ''}{item.message}"
                    for item in entries[-limit:]
                )
                or "No canonical events yet",
                style="dim" if not entries else "",
            )
            title = "LOGS · canonical events"
        elif self.state.view == "top":
            content = self._profiling_panel()
            title = "PROFILING · full top view"
        else:
            content = Text(
                "Navigation\n"
                "  ↑/k ↓/j  select slot    Home/End first/last\n"
                "  Enter/→ details         Esc/← matrix\n"
                "  Tab/Shift+Tab detail tab\n\n"
                "Actions\n"
                "  q graceful stop         p pause/resume scheduling\n"
                "  n/Shift+N generation    r confirmed retryable slot\n"
                "  c config  l logs  t top  / search  h help\n\n"
                "Panel copy\n"
                "  1–8 copy the numbered panel to OSC 52 and /tmp\n\n"
                "Metrics\n"
                "  IR uses completed authoritative evaluations only.\n"
                "  Unknown values are —, never inferred as zero.\n"
                "  Evaluation workers and provider concurrency are separate.",
            )
            title = "HELP · definitions and shortcuts"
        return Panel(content, title=title, border_style="cyan", padding=(0, 1))

    def _footer(self, width: int) -> Text:
        if self.state.search_editing:
            return Text(
                _compact(
                    f"Search: {self.state.search_query}_  Enter apply · Esc cancel",
                    width,
                ),
                style="reverse",
            )
        if self.state.retry_confirmation:
            return Text("Retry selected slot? [y/N]".ljust(width), style="reverse")
        labels = (
            (
                "[1–8] copy",
                "[q] quit",
                "[p] pause/resume",
                "[n] next gen",
                "[r] retry failed",
                "[c] config",
                "[l] logs",
                "[t] top",
                "[/] search",
                "[h] help",
            )
            if width >= 110
            else (
                "[1–8] copy",
                "[q]quit",
                "[p]pause",
                "[n]gen",
                "[r]retry",
                "[c]config",
                "[l]logs",
                "[t]top",
                "[/]search",
                "[h]help",
            )
        )
        footer = Text(style="reverse")
        for index, label in enumerate(labels):
            if index:
                footer.append("  ", style="reverse")
            disabled = (
                (label.startswith("[p]") and self.capabilities.pause is None)
                or (label.startswith("[r]") and self.capabilities.retry is None)
                or (label.startswith("[t]") and not self.state.profiling_enabled)
            )
            footer.append(label, style="reverse dim" if disabled else "reverse")
        status = f" · {self.state.status_message}" if self.state.status_message else ""
        available = max(0, width - len(status))
        footer.truncate(available, overflow="ellipsis")
        if status:
            footer.append(_compact(status, width - len(footer)), style="reverse")
        footer.append(" " * max(0, width - len(footer)), style="reverse")
        return footer

    def _elapsed(self) -> float:
        if self.state.started_monotonic is not None and self.state.experiment_state not in {
            "completed",
            "interrupted",
            "failed",
        }:
            return max(
                self.state.elapsed_seconds,
                time.monotonic() - self.state.started_monotonic,
            )
        return self.state.elapsed_seconds

    @staticmethod
    def _slot_elapsed(slot: DashboardSlot) -> float | None:
        if slot.started_monotonic is not None and slot.state in ACTIVE_STATES:
            return max(0.0, time.monotonic() - slot.started_monotonic)
        return slot.elapsed_seconds

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._input.close()
        if self._live_started:
            self._dirty.set()
            self._rendered.wait(timeout=1.0)
        self._stop.set()
        self._dirty.set()
        if (
            self._refresh_thread is not None
            and self._refresh_thread is not threading.current_thread()
        ):
            self._refresh_thread.join(timeout=1.0)
        if self._live_started:
            self.live.stop()


def _responsive_mode(width: int, height: int) -> Literal["full", "compact", "minimal"]:
    if width >= 140 and height >= 48:
        return "full"
    if width >= 110 and height >= 32:
        return "compact"
    return "minimal"


@dataclass(frozen=True, slots=True)
class _NumberedPanel:
    panel: Panel
    number: str

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        width = (
            options.max_width
            if self.panel.width is None
            else min(options.max_width, self.panel.width)
        )
        title = _numbered_panel_title(self.panel.title, self.number, width)
        numbered = Panel(
            self.panel.renderable,
            box=self.panel.box,
            title=title,
            title_align="left",
            subtitle=self.panel.subtitle,
            subtitle_align=self.panel.subtitle_align,
            safe_box=self.panel.safe_box,
            expand=self.panel.expand,
            style=self.panel.style,
            border_style=self.panel.border_style,
            width=self.panel.width,
            height=self.panel.height,
            padding=self.panel.padding,
            highlight=self.panel.highlight,
        )
        yield from console.render(numbered, options)


def _numbered_panel(panel: Panel, number: str) -> _NumberedPanel:
    return _NumberedPanel(panel, number)


def _numbered_panel_title(
    title: str | Text | None,
    number: str,
    panel_width: int,
) -> Text:
    width = max(1, panel_width - 6)
    characters = ["─"] * width
    characters[-1] = number
    if width >= 2:
        characters[-2] = " "
    if title is not None:
        plain_title = (
            title.plain if isinstance(title, Text) else Text.from_markup(title).plain
        ).strip()
        label = f" {plain_title} "
        available = max(0, width - 3)
        label = _compact(label, available)
        start = max(0, min((width - len(label)) // 2, available - len(label)))
        characters[start : start + len(label)] = label
    return Text("".join(characters))


def _progress_bar(
    label: str,
    current: int,
    total: int | None,
    *,
    width: int,
    stacked: bool,
) -> RenderableType:
    columns: list[Any] = []
    if not stacked:
        columns.append(TextColumn(label))
    columns.extend(
        (
            BarColumn(bar_width=width, complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("{task.fields[ratio]}"),
        )
    )
    progress = Progress(
        *columns,
        expand=not stacked,
    )
    if total is None:
        progress.add_task("", total=1, completed=0, ratio="—/—")
    else:
        progress.add_task(
            "",
            total=max(total, 1),
            completed=max(0, min(current, max(total, 1))),
            ratio=f"{current}/{total}",
        )
    return Group(Text(label), progress) if stacked else progress


def _parameter_line(
    groups: Sequence[tuple[str, object, str | None]],
) -> Text:
    line = Text()
    for index, (label, value, semantic_style) in enumerate(groups):
        if index:
            line.append("  ")
        style = semantic_style or ("grey62" if index % 2 == 0 else "")
        line.append(f"{label} {_show(value)}", style=style)
    return line


def _key_value_grid(rows: Sequence[tuple[str, object]]) -> Table:
    table = Table.grid(expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right", overflow="ellipsis")
    for label, value in rows:
        table.add_row(label, _show(value))
    return table


def _token_grid(usage: TokenUsage, *, charged: bool | None) -> Table:
    return _key_value_grid(
        (
            ("input", usage.input),
            ("cached input", usage.cached),
            ("output", usage.output),
            ("reasoning", usage.reasoning),
            ("authoritative total", usage.total),
            ("usage quality", usage.quality),
            ("charged", "yes" if charged is True else "no" if charged is False else "—"),
        )
    )


def _show(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _ratio(active: int | None, configured: int | None) -> str:
    if active is None or configured is None:
        return "—/—"
    return f"{active}/{configured}"


def _duration(value: float | int | None) -> str:
    if value is None:
        return "—"
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _objective(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _compact(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "…"


def _bounded_preview(value: str, lines: int) -> str:
    if not value:
        return "Unavailable from canonical event stream"
    values = value.splitlines()
    result = values[:lines]
    if len(values) > lines:
        result.append("…")
    return "\n".join(result)


def _sparkline(values: Sequence[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    low, high = min(values), max(values)
    if high == low:
        return blocks[3] * len(values)
    return "".join(blocks[round((value - low) * 7 / (high - low))] for value in values)


def _next_action(slot: DashboardSlot) -> str:
    if slot.retryable:
        return "retry available"
    if slot.state in ACTIVE_STATES:
        return "wait for safe phase boundary"
    if slot.state in {"accepted", "duplicate", "invalid"}:
        return "none"
    return "await scheduler"


def _flatten_config(value: Mapping[str, object], prefix: str = "") -> str:
    rows: list[str] = []
    for key in sorted(value):
        item = value[key]
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            rows.extend(_flatten_config(item, name).splitlines())
        else:
            rendered = _safe_display_path(item) if isinstance(item, str) else item
            rows.append(f"{name} = {rendered}")
    return "\n".join(row for row in rows if row)


def _safe_display_path(value: str) -> str:
    path = PurePath(value)
    if not path.is_absolute():
        return value
    parts = path.parts
    if "workspace" in parts:
        return "/".join(parts[parts.index("workspace") :])
    return path.name
