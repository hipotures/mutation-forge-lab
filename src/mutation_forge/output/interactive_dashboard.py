from __future__ import annotations

import _thread
import contextlib
import json
import os
import resource
import select
import sqlite3
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from pathlib import Path, PurePath
from typing import Any, Literal, TextIO, cast

from rich import box
from rich.align import Align
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from mutation_forge.events import Event
from mutation_forge.experiment.config import orders_for_generation
from mutation_forge.models import JsonValue
from mutation_forge.output.display_ids import compact_display_ids
from mutation_forge.output.panel_copy import (
    copy_text_to_clipboard_osc52,
    render_panel_copy_text,
    save_panel_copy,
)

REFRESH_INTERVAL_SECONDS = 1.0
PERSISTED_GENERATION_PAGE_SIZE = 10
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
    "batched": "cyan",
    "retrying": "yellow",
    "repair": "yellow",
    "validating": "blue",
    "probing": "magenta",
    "evaluating": "dark_orange",
    "stopping": "yellow",
    "accepted": "green",
    "duplicate": "dim blue",
    "invalid": "red",
    "failed": "bright_red",
    "recovered": "dim green",
    "budget": "yellow",
    "budget_exhausted": "yellow",
    "interrupted": "magenta",
    "stopped": "yellow",
}
STATE_ICONS = {
    "queued": "…",
    "starting": "▶",
    "model": "●",
    "retrying": "R",
    "repair": "✚",
    "validating": "V",
    "probing": "P",
    "evaluating": "▲",
    "accepted": "✓",
    "duplicate": "D",
    "invalid": "×",
    "failed": "!",
    "recovered": "↺",
    "budget": "B",
    "budget_exhausted": "X",
    "interrupted": "■",
    "stopping": "■",
    "stopped": "■",
}
PHASE_ICONS: dict[str, str] = {
    "initial": "◌",
    "repair": "↻",
    "development": "⋆",
    "replay": "↺",
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
ALL_PANELS_COPY_TARGET = "all-panels"
PANEL_COPY_WIDTHS = {
    "header": 150,
    "progress": 150,
    "slots": 240,
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
    cache_write: int | None = None
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
    phase_started_monotonic: float | None = None
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
    evaluation_id: str | None = None
    evaluation_completed: int = 0
    evaluation_total: int | None = None
    evaluation_pass: str | None = None
    evaluation_order: int | None = None
    graph_seed: int | None = None
    policy_seed: int | None = None
    evaluation_rate: float | None = None
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
    graph_mode: str = "—"
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
    hourly_token_limit: int | None = None
    hourly_tokens_used: int = 0
    hourly_limit_reached: bool = False
    hourly_retry_after: str | None = None
    active_provider_turns: int = 0
    configured_provider_concurrency: int | None = None
    evaluations_completed: int = 0
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
    native_v3_bottleneck: str = "—"
    provider_utilization: float | None = None
    evaluator_utilization: float | None = None
    provider_response_latency_seconds: float = 0.0
    programs_returned_per_call: float | None = None
    valid_programs_per_provider_minute: float | None = None
    candidate_queue_depth: int = 0
    evaluation_shard_queue_depth: int = 0
    verification_queue_depth: int = 0
    verification_backpressure_active: bool = False
    provider_starvation_seconds: float = 0.0
    provider_backpressure_seconds: float = 0.0
    generation_wall_share: float | None = None
    validation_wall_share: float | None = None
    evaluation_wall_share: float | None = None
    persistence_wall_share: float | None = None
    time_to_first_evaluation_seconds: float | None = None
    first_valid_ast_to_first_worker_seconds: float | None = None
    first_valid_ast_to_half_workers_seconds: float | None = None
    first_valid_ast_to_all_workers_seconds: float | None = None
    raw_graph_score_calls: int = 0
    unique_graph_scores: int = 0
    raw_graph_score_calls_per_second: float | None = None
    unique_graph_scores_per_second: float | None = None
    episodes_per_second: float | None = None
    accepted_rewrites_per_second: float | None = None
    accepted_rewrites: int = 0
    score_cache_hit_rate: float | None = None
    active_cpp_scorers: int = 0
    scorer_restarts: int = 0
    forbidden_fallback_count: int = 0
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
    slot_icon_mode: bool = False
    retry_confirmation: bool = False
    paused: bool = False
    status_message: str = ""
    counterexample_state: str = "none"
    counterexample_candidate: str = "—"
    counterexample_order: int | None = None
    counterexample_edges: int | None = None
    counterexample_minimum_degree: int | None = None
    counterexample_lengths: tuple[int, ...] = ()
    counterexample_primary: str = "not started"
    counterexample_independent: str = "not started"
    counterexample_certificate: str = "—"
    event_keys: frozenset[str] = frozenset()


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


def load_persisted_dashboard_state(
    experiment_root: str | Path,
    *,
    run_id: str,
    model: str = "—",
    effort: str = "—",
    generation_limit: int | None = None,
    population_size: int = 8,
    wall_seconds: float | None = None,
    hourly_token_limit: int | None = None,
    graph_mode: str = "—",
    generation_before: int | None = None,
    generation_page_size: int = PERSISTED_GENERATION_PAGE_SIZE,
) -> DashboardState:
    """Load one durable generation page without replaying historical events.

    The live sink used to start from an empty reducer and then receive every
    transition of the resumed session.  That made a continuation look as if
    it was going backwards through old ``queued``/``validating`` states and
    dropped objectives from earlier sessions.  This reader takes only the
    requested checkpoint/SQLite generation window, then live events continue
    from that snapshot.
    """

    root = Path(experiment_root)
    state = DashboardState(
        run_id=run_id,
        model=model,
        effort=effort,
        generation_limit=generation_limit,
        population_size=population_size,
        wall_seconds=wall_seconds,
        hourly_token_limit=hourly_token_limit,
        graph_mode=graph_mode,
    )
    checkpoint_generation = 0
    raw_slot_values: list[Mapping[str, Any]] = []

    store_path = root / "state.sqlite3"
    if store_path.is_file():
        try:
            database_uri = f"{store_path.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(database_uri, uri=True) as connection:
                latest_row = connection.execute(
                    "SELECT MAX(generation) FROM ("
                    "SELECT generation FROM candidates "
                    "UNION ALL SELECT generation FROM provider_turns"
                    ")"
                ).fetchone()
            latest_value = latest_row[0] if latest_row else None
            if isinstance(latest_value, int) and not isinstance(latest_value, bool):
                checkpoint_generation = max(checkpoint_generation, latest_value)
        except sqlite3.DatabaseError:
            pass

    page_size = max(1, generation_page_size)
    page_end = (
        checkpoint_generation
        if generation_before is None
        else min(checkpoint_generation, max(0, generation_before - 1))
    )
    page_start = max(0, page_end - page_size + 1)

    slot_records: dict[tuple[int, str], Mapping[str, Any]] = {}
    for raw in raw_slot_values:
        generation = cast(int, raw["generation"])
        if not page_start <= generation <= page_end:
            continue
        slot = cast(str, raw["slot"])
        key = (generation, slot)
        previous = slot_records.get(key)
        # Repair records supersede their initial record.  If both have the
        # same repair count, preserve the later checkpoint object.
        repairs = raw.get("repairs")
        previous_repairs_value = previous.get("repairs") if previous else -1
        previous_repairs = (
            previous_repairs_value
            if isinstance(previous_repairs_value, int)
            and not isinstance(previous_repairs_value, bool)
            else -1
        )
        if previous is None or (
            isinstance(repairs, int)
            and not isinstance(repairs, bool)
            and repairs >= int(previous_repairs)
        ):
            slot_records[key] = raw

    evaluations: dict[str, tuple[float, Mapping[str, Any], float | None]] = {}
    evaluation_history: list[tuple[str, float]] = []
    best_candidate: (
        tuple[
            str,
            tuple[float, Mapping[str, Any], float | None],
        ]
        | None
    ) = None
    slot_runtime_seconds: dict[tuple[int, str], float] = {}
    store: Any | None = None
    try:
        from mutation_forge.experiment.state import ExperimentStateStore

        if store_path.is_file():
            store = ExperimentStateStore(store_path)
            experiment = store.experiment()
            current_session = store.session()
            cumulative = store.cumulative()
            counts = store.counts()
            usage = _usage(store.token_usage())
            hourly = store.hourly_token_usage(hourly_token_limit)
            state = replace(
                state,
                experiment_state=str(experiment.get("state", state.experiment_state)),
                provider_turns_attempted=int(cumulative.get("provider_turns", 0)),
                provider_turns_completed=counts.get("provider_turns_completed", 0),
                cumulative_usage=usage,
                hourly_tokens_used=int(hourly.get("hourly_tokens_used", 0)),
                hourly_limit_reached=hourly.get("hourly_limit_reached") is True,
                hourly_retry_after=(
                    str(hourly["hourly_retry_after"]) if hourly.get("hourly_retry_after") else None
                ),
                archive_size=counts.get("candidate_count", 0),
                accepted_candidates=counts.get("unique_candidate_count", 0),
                evaluations_completed=counts.get("evaluation_count", 0),
            )
            if isinstance(current_session, Mapping):
                session_id = current_session.get("session_id")
                state = replace(
                    state,
                    session_id=str(session_id) if session_id else state.session_id,
                    started_at=str(current_session.get("started_at", ""))[11:19] or "—",
                )
            slot_started_at: dict[tuple[int, str], datetime] = {}
            for event_row in store.connection.execute(
                "SELECT event_type,timestamp,generation,slot FROM events "
                "WHERE event_type IN ("
                "'provider_turn_started','provider_turn_completed','provider_turn_failed',"
                "'repair_started','repair_completed','validation_started',"
                "'validation_completed','behavior_probe_started','behavior_probe_completed',"
                "'evaluation_started','evaluation_progress','evaluation_completed',"
                "'evaluation_failed','candidate_archived'"
                ") AND generation BETWEEN ? AND ? ORDER BY sequence",
                (page_start, page_end),
            ):
                try:
                    timestamp = datetime.fromisoformat(str(event_row["timestamp"]))
                except ValueError:
                    continue
                event_generation = event_row["generation"]
                event_slot = event_row["slot"]
                if (
                    not isinstance(event_generation, int)
                    or isinstance(event_generation, bool)
                    or not isinstance(event_slot, str)
                ):
                    continue
                key = (event_generation, event_slot)
                if key not in slot_started_at:
                    if event_row["event_type"] != "provider_turn_started":
                        continue
                    slot_started_at[key] = timestamp
                slot_runtime_seconds[key] = max(
                    slot_runtime_seconds.get(key, 0.0),
                    (timestamp - slot_started_at[key]).total_seconds(),
                )
            rows = store.connection.execute(
                "SELECT evaluations.candidate_id,evaluations.completed_at,"
                "evaluations.episode_count,evaluations.mean_auc,evaluations.best_auc,"
                "evaluations.baseline_auc_json,evaluations.improvement_rate,"
                "evaluations.elapsed_seconds "
                "FROM evaluations JOIN candidates "
                "ON candidates.candidate_id=evaluations.candidate_id "
                "WHERE evaluations.state='completed' "
                "AND candidates.generation BETWEEN ? AND ? "
                "ORDER BY evaluations.completed_at,evaluations.identity",
                (page_start, page_end),
            ).fetchall()
            for row in rows:
                candidate_id = str(row["candidate_id"] or "")
                try:
                    baseline_auc = json.loads(str(row["baseline_auc_json"]))
                except (TypeError, json.JSONDecodeError):
                    baseline_auc = {}
                metric = row["mean_auc"]
                if (
                    not candidate_id
                    or not isinstance(metric, (int, float))
                    or isinstance(metric, bool)
                ):
                    continue
                value = float(metric)
                summary = {
                    "episode_count": row["episode_count"],
                    "mean_auc": value,
                    "best_auc": row["best_auc"],
                    "baseline_auc": (baseline_auc if isinstance(baseline_auc, Mapping) else {}),
                    "improvement_rate": row["improvement_rate"],
                }
                evaluations[candidate_id] = (
                    value,
                    summary,
                    _number(row["elapsed_seconds"]),
                )

            history_rows = store.connection.execute(
                "SELECT completed_at,mean_auc "
                "FROM evaluations WHERE state='completed' "
                "AND mean_auc IS NOT NULL "
                "ORDER BY completed_at DESC,identity DESC LIMIT 60"
            ).fetchall()
            evaluation_history = [
                (str(row["completed_at"] or ""), float(row["mean_auc"]))
                for row in reversed(history_rows)
                if isinstance(row["mean_auc"], (int, float))
                and not isinstance(row["mean_auc"], bool)
            ]
            best_row = store.connection.execute(
                "SELECT candidate_id,mean_auc "
                "FROM evaluations WHERE state='completed' "
                "AND mean_auc IS NOT NULL "
                "ORDER BY mean_auc DESC,candidate_id LIMIT 1"
            ).fetchone()
            if (
                best_row is not None
                and isinstance(best_row["candidate_id"], str)
                and isinstance(best_row["mean_auc"], (int, float))
                and not isinstance(best_row["mean_auc"], bool)
            ):
                best_candidate = (
                    best_row["candidate_id"],
                    (float(best_row["mean_auc"]), {}, None),
                )

            # Rebuild slot rows from the durable candidate table so a
            # dashboard restart never loses a finished generation or its
            # objective values.
            candidate_rows = store.connection.execute(
                "SELECT candidate_id,generation,slot,parent_id,status,behavior_json "
                "FROM candidates WHERE generation BETWEEN ? AND ? "
                "ORDER BY generation,slot,candidate_id",
                (page_start, page_end),
            ).fetchall()
            for row in candidate_rows:
                candidate_id = str(row["candidate_id"] or "")
                generation = row["generation"]
                slot = row["slot"]
                if (
                    not candidate_id
                    or not isinstance(generation, int)
                    or isinstance(generation, bool)
                    or not isinstance(slot, str)
                ):
                    continue
                behavior: Mapping[str, Any] = {}
                try:
                    raw_behavior = json.loads(str(row["behavior_json"] or "{}"))
                    if isinstance(raw_behavior, Mapping):
                        behavior = raw_behavior
                except (TypeError, json.JSONDecodeError):
                    pass
                existing = slot_records.get((generation, slot), {})
                existing_candidate = existing.get("candidate")
                if isinstance(existing_candidate, Mapping):
                    continue
                slot_records[(generation, slot)] = {
                    "generation": generation,
                    "slot": slot,
                    "parent_id": str(row["parent_id"] or "root"),
                    "status": str(row["status"] or "accepted"),
                    "candidate": {"candidate_id": candidate_id},
                    "raw_result": {},
                    "request": {},
                    "errors": [],
                    "repairs": 0,
                    "behavior": behavior,
                }
    except (OSError, UnicodeError, sqlite3.DatabaseError, json.JSONDecodeError):
        # A missing or partially-written optional snapshot can be completed by
        # the live event stream.  Contract/schema errors intentionally escape:
        # v1 or malformed state must be rejected, never silently rendered as a
        # fresh experiment.
        pass
    finally:
        if store is not None:
            with contextlib.suppress(Exception):
                store.close()

    groups: dict[int, list[DashboardSlot]] = {}
    for (generation, slot_name), raw in slot_records.items():
        candidate_value = raw.get("candidate")
        candidate_id = (
            f"g{generation:04d}-{slot_name}" if isinstance(candidate_value, Mapping) else ""
        )
        metric = evaluations.get(candidate_id)
        status = str(raw.get("status", "queued"))
        if status in {"repair_pending", "repair_running"}:
            display_state = "repair"
        elif status in {"failed", "invalid", "duplicate", "accepted"}:
            display_state = status
        elif status in {"created", "queued"} and metric is not None:
            # SQLite may contain a completed evaluation written after the last
            # A metric is durable evidence that this slot is accepted even
            # when its candidate row still has the transient ``created``
            # status.
            display_state = "accepted"
        else:
            display_state = "queued"
        raw_result = raw.get("raw_result")
        result_mapping: Mapping[str, Any] = {}
        if isinstance(raw_result, Mapping):
            result_mapping = cast(Mapping[str, Any], raw_result)
        else:
            repair_value = raw.get("repair")
            if isinstance(repair_value, Mapping):
                result_mapping = cast(Mapping[str, Any], repair_value)
        usage_value = result_mapping.get("usage")
        if usage_value is None and isinstance(candidate_value, Mapping):
            usage_value = candidate_value.get("usage")
        duration_ms = result_mapping.get("provider_duration_ms")
        provider_duration_seconds = (
            float(duration_ms) / 1000.0
            if isinstance(duration_ms, int) and not isinstance(duration_ms, bool)
            else None
        )
        evaluation_elapsed = metric[2] if metric is not None else None
        duration_seconds = slot_runtime_seconds.get(
            (generation, slot_name),
            evaluation_elapsed if metric is not None else provider_duration_seconds,
        )
        raw_errors = raw.get("errors")
        error_message = ""
        if isinstance(raw_errors, list) and raw_errors:
            first_error = raw_errors[0]
            if isinstance(first_error, Mapping):
                error_message = str(first_error.get("message", ""))
        groups.setdefault(generation, []).append(
            DashboardSlot(
                slot=slot_name,
                generation=generation,
                parent=str(raw.get("parent_id", "root")),
                phase=(
                    "development"
                    if metric is not None
                    else str(raw.get("request", {}).get("phase", "initial"))
                    if isinstance(raw.get("request"), Mapping)
                    else "initial"
                ),
                state=display_state,
                elapsed_seconds=duration_seconds,
                usage=_usage(usage_value),
                validation="pass" if display_state in {"accepted", "duplicate"} else "—",
                probe="pass" if display_state in {"accepted", "duplicate"} else "—",
                candidate=candidate_id,
                error=error_message,
                objective=metric[0] if metric is not None else None,
                repairs=(
                    int(raw.get("repairs", 0))
                    if isinstance(raw.get("repairs", 0), int)
                    and not isinstance(raw.get("repairs", 0), bool)
                    else 0
                ),
                lifecycle=(
                    (LifecycleStep("evaluation", "pass", 0.0),)
                    if metric is not None
                    else (LifecycleStep("archived", "pass", 0.0),)
                ),
            )
        )
    if page_end == checkpoint_generation:
        groups.setdefault(
            checkpoint_generation,
            list(_empty_slots(checkpoint_generation, population_size)),
        )
    generation_groups = tuple(
        GenerationSlots(generation, tuple(sorted(slots, key=lambda item: item.slot)))
        for generation, slots in sorted(groups.items())
    )
    metrics = [value for _, value in evaluation_history]
    if best_candidate is None:
        best_candidate = max(evaluations.items(), key=lambda item: item[1][0], default=None)
    return replace(
        state,
        generation=checkpoint_generation,
        displayed_generation=page_end,
        generations=generation_groups,
        current_objective=metrics[-1] if metrics else None,
        best_objective=best_candidate[1][0] if best_candidate else None,
        best_candidate=best_candidate[0] if best_candidate else "—",
        objective_history=tuple(metrics[-60:]),
    )


def _merge_persisted_dashboard_state(
    live: DashboardState,
    persisted: DashboardState,
) -> DashboardState:
    """Add durable generations to a live reducer without rewinding it."""

    by_generation: dict[int, dict[str, DashboardSlot]] = {}
    for source in (persisted, live):
        for group in source.generations:
            slots = by_generation.setdefault(group.generation, {})
            for slot in group.slots:
                existing = slots.get(slot.slot)
                if (
                    existing is None
                    or slot.objective is not None
                    or slot.state in ACTIVE_STATES
                    or existing.objective is None
                ):
                    slots[slot.slot] = slot
    generations = tuple(
        GenerationSlots(
            generation,
            tuple(sorted(slots.values(), key=lambda item: item.slot)),
        )
        for generation, slots in sorted(by_generation.items())
    )
    history = live.objective_history or persisted.objective_history
    live_best = live.best_objective
    persisted_best = persisted.best_objective
    use_persisted_best = persisted_best is not None and (
        live_best is None or persisted_best > live_best
    )
    return replace(
        live,
        generations=generations,
        objective_history=history,
        best_objective=persisted_best if use_persisted_best else live_best,
        best_candidate=persisted.best_candidate if use_persisted_best else live.best_candidate,
        archive_size=max(live.archive_size, persisted.archive_size),
        accepted_candidates=max(live.accepted_candidates, persisted.accepted_candidates),
        evaluations_completed=max(
            live.evaluations_completed,
            persisted.evaluations_completed,
        ),
    )


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _rational_number(value: object) -> float | None:
    if not isinstance(value, Mapping):
        return _number(value)
    numerator = _integer(value.get("numerator"))
    denominator = _integer(value.get("denominator"))
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _provider_elapsed(payload: Mapping[str, JsonValue]) -> float | None:
    elapsed_ns = _integer(payload.get("operation_elapsed_ns"))
    if elapsed_ns is not None and elapsed_ns >= 0:
        return elapsed_ns / 1e9
    duration_ms = _number(payload.get("provider_duration_ms"))
    if duration_ms is not None and duration_ms >= 0:
        return duration_ms / 1000.0
    elapsed = _number(payload.get("operation_elapsed_seconds"))
    return elapsed if elapsed is not None and elapsed >= 0 else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _provider_call_slots(payload: Mapping[str, JsonValue]) -> tuple[str, ...]:
    encoded = _text(payload.get("slot_ids"))
    if encoded is None:
        return ()
    return tuple(value for item in encoded.split(",") if (value := item.strip()))


def _update_provider_call_slots(
    state: DashboardState,
    payload: Mapping[str, JsonValue],
    *,
    event_type: str,
    now: float,
) -> DashboardState:
    generation = _integer(payload.get("generation"))
    target_generation = state.generation if generation is None else generation
    elapsed = _provider_elapsed(payload)
    timeout_ns = _integer(payload.get("timeout_ns"))
    timeout = (
        timeout_ns / 1e9
        if timeout_ns is not None and timeout_ns >= 0
        else _number(payload.get("timeout_seconds"))
    )
    error_type = _text(payload.get("error_type"))
    error_message = _text(payload.get("error_message"))
    diagnostic = (
        f"{error_type}: {error_message}"
        if error_type is not None and error_message is not None
        else error_message or error_type or ""
    )
    call_id = _text(payload.get("call_id"))
    slot_names = _provider_call_slots(payload)
    for index, slot_name in enumerate(slot_names):
        is_call_representative = index == 0
        group = _generation_slots(state, target_generation)
        slot = next(
            (item for item in group.slots if item.slot == slot_name),
            DashboardSlot(slot=slot_name, generation=target_generation),
        )
        started = slot.started_monotonic
        phase_started = slot.phase_started_monotonic
        if event_type == "provider_call_started":
            started = now if is_call_representative and started is None else started
            phase_started = now if is_call_representative else None
            updated = replace(
                slot,
                phase="provider",
                state="model" if is_call_representative else "batched",
                started_monotonic=started,
                phase_started_monotonic=phase_started,
                elapsed_seconds=None,
                timeout_seconds=timeout,
                provider_request_id=call_id,
                lifecycle=_lifecycle(slot, "provider", "running"),
            )
        elif event_type == "provider_call_failed":
            updated = replace(
                slot,
                phase="response",
                state="failed" if is_call_representative else "batched",
                elapsed_seconds=elapsed if is_call_representative else None,
                phase_started_monotonic=None,
                error=(diagnostic or "provider call failed") if is_call_representative else None,
                lifecycle=(
                    _lifecycle(slot, "provider", "fail", elapsed)
                    if is_call_representative
                    else slot.lifecycle
                ),
            )
        else:
            updated = replace(
                slot,
                phase="response" if is_call_representative else "provider",
                state="validating" if is_call_representative else "batched",
                elapsed_seconds=elapsed if is_call_representative else None,
                phase_started_monotonic=None,
                lifecycle=(
                    _lifecycle(slot, "provider", "pass", elapsed)
                    if is_call_representative
                    else slot.lifecycle
                ),
            )
        state = _replace_slot(state, updated)
    return state


def _usage(value: object, *, quality: object = None) -> TokenUsage:
    source = value if isinstance(value, Mapping) else {}
    return TokenUsage(
        input=_integer(source.get("inputTokens")),
        cached=_integer(source.get("cachedInputTokens")),
        cache_write=_integer(source.get("cacheWriteInputTokens")),
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
        cache_write=add(left.cache_write, right.cache_write),
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
        DashboardSlot(slot=f"slot-{index:02d}", generation=generation) for index in range(count)
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


def _restored_gate_status(value: JsonValue | None) -> str:
    status = str(value or "").lower()
    if status in {"valid", "pass", "passed"}:
        return "pass"
    if status in {"invalid", "fail", "failed"}:
        return "fail"
    return "unknown"


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
    if event_type in {
        "provider_turn_activity",
        "repair_activity",
    }:
        elapsed = _provider_elapsed(payload)
        timeout_ns = _integer(payload.get("timeout_ns"))
        timeout = (
            timeout_ns / 1e9
            if timeout_ns is not None and timeout_ns >= 0
            else _number(payload.get("timeout_seconds"))
        )
        timing = ""
        if elapsed is not None:
            timing = (
                f" ({elapsed:.0f}/{timeout:.0f}s)"
                if timeout is not None
                else f" ({elapsed:.0f}s)"
            )
        return ActivityEntry(
            event.timestamp[11:19],
            "provider",
            "info",
            f"waiting{timing}",
            slot,
        )
    if event_type == "provider_turn_failed":
        return ActivityEntry(
            event.timestamp[11:19],
            "provider",
            "error",
            _text(payload.get("error")) or "provider turn failed",
            slot,
        )
    if event_type == "provider_call_failed":
        error_type = _text(payload.get("error_type"))
        error_message = _text(payload.get("error_message")) or "provider call failed"
        message = f"{error_type}: {error_message}" if error_type else error_message
        return ActivityEntry(
            event.timestamp[11:19],
            "provider",
            "error",
            message,
            _text(payload.get("call_id")),
        )
    if event_type in {
        "experiment_failed",
        "experiment_interrupted",
        "experiment_exhausted",
    }:
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
        "experiment_exhausted",
        "counterexample_candidate_found",
        "counterexample_primary_verification_started",
        "counterexample_primary_verification_completed",
        "counterexample_independent_verification_started",
        "counterexample_independent_verification_completed",
        "counterexample_verification_conflict",
        "counterexample_verified",
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


def _global_payload(
    state: DashboardState,
    payload: Mapping[str, JsonValue],
    *,
    include_elapsed: bool,
) -> DashboardState:
    values: dict[str, Any] = {}
    mappings = {
        "session_id": "session_id",
        "run_mode": "run_mode",
        "model": "model",
        "effort": "effort",
        "phase": "phase",
        "graph_mode": "graph_mode",
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
        "hourly_token_limit": "hourly_token_limit",
        "hourly_tokens_used": "hourly_tokens_used",
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
        "evaluations_per_second": "evaluation_rate",
        "ir": "improvement_rate",
        "improvement_rate": "improvement_rate",
    }
    if include_elapsed:
        numbers["elapsed_seconds"] = "elapsed_seconds"
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
    hourly_reached = payload.get("hourly_limit_reached")
    if isinstance(hourly_reached, bool):
        values["hourly_limit_reached"] = hourly_reached
    hourly_retry_after = _text(payload.get("hourly_retry_after"))
    if hourly_retry_after is not None:
        values["hourly_retry_after"] = hourly_retry_after
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
    event_type = event.event_type
    previous_generation = state.generation
    previous_displayed_generation = state.displayed_generation
    was_following_current_generation = previous_displayed_generation == previous_generation
    raw_idempotency_key = _text(payload.get("idempotency_key"))
    idempotency_key = (
        f"{event_type}:{raw_idempotency_key}" if raw_idempotency_key is not None else None
    )
    if idempotency_key is not None and idempotency_key in state.event_keys:
        return state
    if idempotency_key is not None:
        state = replace(
            state,
            event_keys=state.event_keys | {idempotency_key},
        )
    state = _global_payload(
        replace(state, run_id=event.run_id),
        payload,
        include_elapsed=event_type
        in {
            "session_started",
            "budget_boundary_reached",
            "experiment_completed",
            "experiment_exhausted",
            "experiment_interrupted",
            "experiment_failed",
        },
    )
    if event_type != "generation_started":
        current_generation = max(previous_generation, state.generation)
        state = replace(
            state,
            generation=current_generation,
            displayed_generation=(
                current_generation
                if current_generation > previous_generation and was_following_current_generation
                else previous_displayed_generation
            ),
        )
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
            provider_turns_completed=(_integer(payload.get("cumulative_provider_turns")) or 0),
        )
    elif event_type == "generation_started":
        generation = _integer(payload.get("generation")) or 0
        count = _slot_count(payload, state.population_size)
        retained_generation = next(
            (item for item in state.generations if item.generation == generation),
            None,
        )
        state = replace(
            state,
            generation=generation,
            displayed_generation=generation,
            population_size=count,
            completed_slots=0,
            selected_index=0,
            view="matrix" if state.view == "details" else state.view,
        )
        if retained_generation is None:
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
    elif event_type == "provider_call_started":
        state = replace(
            state,
            active_provider_turns=state.active_provider_turns + 1,
            provider_turns_attempted=state.provider_turns_attempted + 1,
            phase="provider",
        )
        state = _update_provider_call_slots(
            state,
            payload,
            event_type=event_type,
            now=now,
        )
    elif event_type in {"provider_call_completed", "provider_call_failed"}:
        failed = event_type == "provider_call_failed"
        error_type = _text(payload.get("error_type"))
        error_message = _text(payload.get("error_message"))
        diagnostic = (
            f"{error_type}: {error_message}"
            if error_type is not None and error_message is not None
            else error_message or error_type or "Provider call failed"
        )
        state = replace(
            state,
            active_provider_turns=max(0, state.active_provider_turns - 1),
            provider_turns_completed=state.provider_turns_completed + (0 if failed else 1),
            phase="provider error" if failed else state.phase,
            status_message=diagnostic if failed else state.status_message,
        )
        state = _update_provider_call_slots(
            state,
            payload,
            event_type=event_type,
            now=now,
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
                hourly_tokens_used=state.hourly_tokens_used + (delta.total or 0),
                hourly_limit_reached=(
                    state.hourly_token_limit is not None
                    and state.hourly_tokens_used + (delta.total or 0) >= state.hourly_token_limit
                ),
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
    elif event_type in {
        "verification_backpressure_started",
        "verification_backpressure_ended",
        "verification_job_queued",
        "verification_job_completed",
    }:
        verification_queue_depth = _integer(payload.get("verification_queue_depth"))
        state = replace(
            state,
            verification_queue_depth=(
                state.verification_queue_depth
                if verification_queue_depth is None
                else verification_queue_depth
            ),
            verification_backpressure_active=(
                event_type == "verification_backpressure_started"
                or (
                    state.verification_backpressure_active
                    and event_type != "verification_backpressure_ended"
                )
            ),
        )
    elif event_type == "native_v3_metrics":
        state = replace(
            state,
            native_v3_bottleneck=_text(payload.get("bottleneck")) or "—",
            provider_utilization=_rational_number(payload.get("provider_utilization")),
            evaluator_utilization=_rational_number(payload.get("evaluator_utilization")),
            provider_response_latency_seconds=(
                (_integer(payload.get("provider_response_latency_ns")) or 0) / 1_000_000_000
            ),
            programs_returned_per_call=_rational_number(payload.get("programs_returned_per_call")),
            valid_programs_per_provider_minute=_rational_number(
                payload.get("valid_programs_per_provider_minute")
            ),
            candidate_queue_depth=(_integer(payload.get("candidate_queue_depth")) or 0),
            evaluation_shard_queue_depth=(
                _integer(payload.get("evaluation_shard_queue_depth")) or 0
            ),
            provider_starvation_seconds=(
                (_integer(payload.get("cpu_idle_time_caused_by_provider_starvation_ns")) or 0)
                / 1_000_000_000
            ),
            provider_backpressure_seconds=(
                (
                    _integer(payload.get("provider_idle_time_caused_by_evaluation_backpressure_ns"))
                    or 0
                )
                / 1_000_000_000
            ),
            generation_wall_share=_rational_number(payload.get("generation_wall_share")),
            validation_wall_share=_rational_number(payload.get("validation_wall_share")),
            evaluation_wall_share=_rational_number(payload.get("evaluation_wall_share")),
            persistence_wall_share=_rational_number(payload.get("persistence_wall_share")),
            time_to_first_evaluation_seconds=(
                (_integer(payload.get("time_to_first_evaluation_ns")) or 0) / 1_000_000_000
                if payload.get("time_to_first_evaluation_ns") is not None
                else None
            ),
            first_valid_ast_to_first_worker_seconds=(
                (_integer(payload.get("first_valid_ast_to_first_worker_ns")) or 0) / 1_000_000_000
                if payload.get("first_valid_ast_to_first_worker_ns") is not None
                else None
            ),
            first_valid_ast_to_half_workers_seconds=(
                (_integer(payload.get("first_valid_ast_to_50_percent_workers_ns")) or 0)
                / 1_000_000_000
                if payload.get("first_valid_ast_to_50_percent_workers_ns") is not None
                else None
            ),
            first_valid_ast_to_all_workers_seconds=(
                (_integer(payload.get("first_valid_ast_to_all_workers_ns")) or 0) / 1_000_000_000
                if payload.get("first_valid_ast_to_all_workers_ns") is not None
                else None
            ),
            raw_graph_score_calls=(_integer(payload.get("raw_graph_score_calls")) or 0),
            unique_graph_scores=(_integer(payload.get("unique_graph_scores")) or 0),
            raw_graph_score_calls_per_second=_rational_number(
                payload.get("raw_graph_score_calls_per_second")
            ),
            unique_graph_scores_per_second=_rational_number(
                payload.get("unique_graph_scores_per_second")
            ),
            episodes_per_second=_rational_number(payload.get("episodes_per_second")),
            accepted_rewrites_per_second=_rational_number(
                payload.get("accepted_rewrites_per_second")
            ),
            accepted_rewrites=(_integer(payload.get("accepted_rewrites")) or 0),
            score_cache_hit_rate=_rational_number(payload.get("score_cache_hit_rate")),
            active_cpp_scorers=(_integer(payload.get("active_cpp_scorers")) or 0),
            scorer_restarts=_integer(payload.get("scorer_restarts")) or 0,
            forbidden_fallback_count=(_integer(payload.get("forbidden_fallback_count")) or 0),
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
        best_improves = best is not None and (
            state.best_objective is None or best > state.best_objective
        )
        state = replace(
            state,
            current_objective=current if current is not None else state.current_objective,
            best_objective=best if best_improves else state.best_objective,
            best_candidate=(
                _text(payload.get("best_candidate_id")) or state.best_candidate
                if best_improves
                else state.best_candidate
            ),
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
    elif event_type == "hourly_token_limit_reached":
        state = replace(state, phase="hourly token limit")
    elif event_type == "budget_boundary_reached":
        state = replace(
            state,
            experiment_state=_text(payload.get("state")) or "idle",
            phase="budget",
        )
    elif event_type == "experiment_completed":
        if state.counterexample_state != "conflict":
            state = replace(state, experiment_state="completed", phase="completed")
    elif event_type == "experiment_exhausted":
        state = replace(state, experiment_state="exhausted", phase="exhausted")
    elif event_type == "experiment_interrupted":
        state = replace(state, experiment_state="interrupted", phase="interrupted")
    elif event_type == "experiment_failed":
        if state.counterexample_state != "verified":
            state = replace(state, experiment_state="failed", phase="failed")
    elif event_type == "counterexample_candidate_found":
        if state.counterexample_state in {"verified", "conflict"}:
            return state
        lengths = payload.get("target_forbidden_lengths")
        state = replace(
            state,
            counterexample_state="candidate",
            counterexample_candidate=_text(payload.get("candidate_id")) or "—",
            counterexample_order=_integer(payload.get("order")),
            counterexample_edges=_integer(payload.get("edge_count")),
            counterexample_minimum_degree=_integer(payload.get("minimum_degree")),
            counterexample_lengths=(
                tuple(
                    item for item in lengths if isinstance(item, int) and not isinstance(item, bool)
                )
                if isinstance(lengths, list)
                else ()
            ),
            phase="exact verification",
        )
    elif event_type == "counterexample_primary_verification_started":
        if state.counterexample_state in {"verified", "conflict"}:
            return state
        state = replace(
            state,
            counterexample_state="primary_verifying",
            counterexample_primary="running",
            phase="exact verification",
        )
    elif event_type == "counterexample_primary_verification_completed":
        if state.counterexample_state in {"verified", "conflict"}:
            return state
        if state.counterexample_state not in {"candidate", "primary_verifying"}:
            return state
        state = replace(
            state,
            counterexample_state=(
                "primary_verified"
                if payload.get("status") == "VERIFIED" and payload.get("complete") is True
                else "candidate"
            ),
            counterexample_primary=(
                f"{payload.get('status', 'UNKNOWN')} · "
                f"{'complete' if payload.get('complete') is True else 'incomplete'}"
            ),
        )
    elif event_type == "counterexample_independent_verification_started":
        if state.counterexample_state in {"verified", "conflict"}:
            return state
        if state.counterexample_state not in {"primary_verified", "independent_verifying"}:
            return state
        state = replace(
            state,
            counterexample_state="independent_verifying",
            counterexample_independent="running",
            phase="exact verification",
        )
    elif event_type == "counterexample_independent_verification_completed":
        if state.counterexample_state in {"verified", "conflict"}:
            return state
        if state.counterexample_state not in {"primary_verified", "independent_verifying"}:
            return state
        state = replace(
            state,
            counterexample_state=(
                "independent_verified"
                if payload.get("status") == "VERIFIED" and payload.get("complete") is True
                else state.counterexample_state
            ),
            counterexample_independent=(
                f"{payload.get('status', 'UNKNOWN')} · "
                f"{'complete' if payload.get('complete') is True else 'incomplete'}"
            ),
        )
    elif event_type == "counterexample_verification_conflict":
        if state.counterexample_state == "verified":
            return state
        if state.counterexample_state not in {"primary_verified", "independent_verifying"}:
            return state
        state = replace(
            state,
            counterexample_state="conflict",
            experiment_state="failed",
            phase="verification conflict",
        )
    elif event_type == "counterexample_verified":
        if state.counterexample_state == "conflict":
            return state
        if state.counterexample_state != "independent_verified":
            return state
        state = replace(
            state,
            counterexample_state="verified",
            counterexample_candidate=(
                _text(payload.get("candidate_id")) or state.counterexample_candidate
            ),
            counterexample_certificate=(
                _text(payload.get("certificate")) or state.counterexample_certificate
            ),
            experiment_state="completed",
            phase="exact verification",
        )

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
        phase_started = slot.phase_started_monotonic
        elapsed = slot.elapsed_seconds
        lifecycle_elapsed: float | None = None
        retryable = slot.retryable
        error = slot.error
        validation, validation_message = slot.validation, slot.validation_message
        probe, probe_message = slot.probe, slot.probe_message
        evaluation_id = slot.evaluation_id
        evaluation_completed = slot.evaluation_completed
        evaluation_total = slot.evaluation_total
        evaluation_pass = slot.evaluation_pass
        evaluation_order = slot.evaluation_order
        graph_seed = slot.graph_seed
        policy_seed = slot.policy_seed
        evaluation_rate = slot.evaluation_rate
        if event_type == "slot_queued":
            lifecycle_phase = "queued"
            lifecycle_status = "pass"
            if payload.get("status") == "retrying":
                slot_state = "retrying"
                error = ""
                retryable = False
            elif payload.get("recovered") is True:
                slot_state = _text(payload.get("recovered_status")) or slot_state
                validation = _restored_gate_status(payload.get("validation_status"))
                probe = _restored_gate_status(payload.get("probe_status"))
                if slot_state in {"accepted", "duplicate"}:
                    error = ""
        elif event_type == "provider_turn_started":
            slot_state = "repair" if phase == "repair" else "model"
            lifecycle_phase = "provider"
            started = now if started is None else started
            phase_started = now
        elif event_type == "provider_turn_completed":
            slot_state = "validating" if payload.get("accepted") is True else "failed"
            lifecycle_phase = "response"
            lifecycle_status = "pass" if payload.get("accepted") is True else "fail"
            event_elapsed = _provider_elapsed(payload)
            lifecycle_elapsed = (
                event_elapsed
                if event_elapsed is not None
                else max(0.0, now - phase_started)
                if phase_started is not None
                else None
            )
            elapsed = (
                max(0.0, now - started)
                if started is not None
                else lifecycle_elapsed
                if lifecycle_elapsed is not None
                else elapsed
            )
            phase_started = None
            if payload.get("accepted") is True:
                error = ""
                retryable = False
        elif event_type == "provider_turn_failed":
            slot_state = "failed"
            lifecycle_phase = "response"
            lifecycle_status = "fail"
            retryable = _retryable_provider_failure(payload)
            error = _text(payload.get("error")) or "provider turn failed"
            event_elapsed = _provider_elapsed(payload)
            lifecycle_elapsed = (
                event_elapsed
                if event_elapsed is not None
                else max(0.0, now - phase_started)
                if phase_started is not None
                else None
            )
            elapsed = (
                max(0.0, now - started)
                if started is not None
                else lifecycle_elapsed
                if lifecycle_elapsed is not None
                else elapsed
            )
            phase_started = None
        elif event_type == "candidate_validated":
            slot_state = "evaluating"
            phase = "evaluation"
            lifecycle_phase = "schema"
            lifecycle_status = "pass"
            lifecycle_elapsed = 0.0
            validation = "pass"
            error = ""
        elif event_type == "repair_started":
            slot_state = "repair"
            lifecycle_phase = "provider"
            started = now if started is None else started
            phase_started = now
        elif event_type == "repair_completed":
            lifecycle_phase = "response"
            lifecycle_elapsed = max(0.0, now - phase_started) if phase_started is not None else None
            elapsed = (
                max(0.0, now - started)
                if started is not None
                else lifecycle_elapsed
                if lifecycle_elapsed is not None
                else elapsed
            )
            phase_started = None
            if slot_state == "accepted":
                lifecycle_status = "pass"
                error = ""
                retryable = False
            else:
                lifecycle_status = "fail"
        elif event_type == "validation_started":
            slot_state = "validating"
            lifecycle_phase = "schema"
            started = now if started is None else started
            phase_started = now
        elif event_type == "validation_completed":
            valid = payload.get("valid") is True
            slot_state = "probing" if valid else "invalid"
            lifecycle_phase = "schema"
            lifecycle_status = "pass" if valid else "fail"
            lifecycle_elapsed = max(0.0, now - phase_started) if phase_started is not None else None
            elapsed = (
                max(0.0, now - started)
                if started is not None
                else lifecycle_elapsed
                if lifecycle_elapsed is not None
                else elapsed
            )
            phase_started = None
            code, message = _diagnostic(payload)
            validation = "pass" if valid else code or "fail"
            validation_message = message
            error = "" if valid else code or message or "validation failed"
        elif event_type == "behavior_probe_started":
            slot_state = "probing"
            lifecycle_phase = "probe"
            started = now if started is None else started
            phase_started = now
        elif event_type == "behavior_probe_completed":
            valid = payload.get("valid") is True
            slot_state = "evaluating" if valid else "invalid"
            lifecycle_phase = "probe"
            lifecycle_status = "pass" if valid else "fail"
            lifecycle_elapsed = max(0.0, now - phase_started) if phase_started is not None else None
            elapsed = (
                max(0.0, now - started)
                if started is not None
                else lifecycle_elapsed
                if lifecycle_elapsed is not None
                else elapsed
            )
            phase_started = None
            code, message = _diagnostic(payload)
            probe = "pass" if valid else code or "fail"
            probe_message = message
            error = "" if valid else code or message or "probe failed"
        elif event_type == "evaluation_started":
            slot_state = "evaluating"
            lifecycle_phase = "evaluation"
            started = now if started is None else started
            phase_started = now
            evaluation_id = _text(payload.get("evaluation_id")) or evaluation_id
            evaluation_completed = 0
            evaluation_total = _integer(payload.get("evaluation_total")) or evaluation_total
            evaluation_pass = _text(payload.get("pass")) or phase
            evaluation_order = None
            graph_seed = None
            policy_seed = None
            evaluation_rate = None
        elif event_type == "evaluation_progress":
            slot_state = "evaluating"
            lifecycle_phase = "evaluation"
            evaluation_id = _text(payload.get("evaluation_id")) or evaluation_id
            completed_value = _integer(payload.get("completed"))
            if completed_value is not None:
                evaluation_completed = max(evaluation_completed, completed_value)
            evaluation_total = (
                _integer(payload.get("total"))
                or _integer(payload.get("evaluation_total"))
                or evaluation_total
            )
            evaluation_pass = _text(payload.get("pass")) or evaluation_pass
            evaluation_order = _integer(payload.get("order"))
            graph_seed = _integer(payload.get("graph_seed"))
            policy_seed = _integer(payload.get("policy_seed"))
            rate_value = _number(payload.get("evaluations_per_second"))
            if rate_value is not None:
                evaluation_rate = rate_value
        elif event_type == "evaluation_completed":
            slot_state = "accepted"
            lifecycle_phase = "evaluation"
            lifecycle_status = "pass"
            event_elapsed = _number(payload.get("elapsed_seconds"))
            lifecycle_elapsed = (
                event_elapsed
                if event_elapsed is not None
                else max(0.0, now - phase_started)
                if phase_started is not None
                else None
            )
            elapsed = (
                max(0.0, now - started)
                if started is not None
                else lifecycle_elapsed
                if lifecycle_elapsed is not None
                else elapsed
            )
            started = None
            phase_started = None
            evaluation_id = _text(payload.get("evaluation_id")) or evaluation_id
            if evaluation_total is not None:
                evaluation_completed = evaluation_total
            error = ""
        elif event_type == "evaluation_failed":
            slot_state = "failed"
            lifecycle_phase = "evaluation"
            lifecycle_status = "fail"
            event_elapsed = _number(payload.get("elapsed_seconds"))
            lifecycle_elapsed = (
                event_elapsed
                if event_elapsed is not None
                else max(0.0, now - phase_started)
                if phase_started is not None
                else None
            )
            elapsed = (
                max(0.0, now - started)
                if started is not None
                else lifecycle_elapsed
                if lifecycle_elapsed is not None
                else elapsed
            )
            started = None
            phase_started = None
            evaluation_id = _text(payload.get("evaluation_id")) or evaluation_id
            error = _text(payload.get("error")) or "evaluation failed"
        elif event_type == "candidate_archived":
            archived_state = _text(payload.get("status"))
            if not (archived_state == "invalid" and slot.state == "failed" and slot.retryable):
                slot_state = archived_state or slot_state
            lifecycle_phase = "archived"
            lifecycle_status = "pass" if slot_state in {"accepted", "duplicate"} else "fail"
            elapsed = max(0.0, now - started) if started is not None else elapsed
            started = None
            phase_started = None
            if slot_state in {"accepted", "duplicate"}:
                error = ""
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
            phase_started_monotonic=phase_started,
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
            evaluation_id=evaluation_id,
            evaluation_completed=evaluation_completed,
            evaluation_total=evaluation_total,
            evaluation_pass=evaluation_pass,
            evaluation_order=evaluation_order,
            graph_seed=graph_seed,
            policy_seed=policy_seed,
            evaluation_rate=evaluation_rate,
            lifecycle=_lifecycle(
                slot,
                lifecycle_phase,
                lifecycle_status,
                lifecycle_elapsed,
            ),
            artifacts=artifacts,
            prompt_preview=_text(payload.get("prompt_preview")) or slot.prompt_preview,
            response_preview=_text(payload.get("response_preview")) or slot.response_preview,
        )
        state = _replace_slot(state, updated)
        active_rates = [
            item.evaluation_rate
            for item in _generation_slots(state, state.generation).slots
            if item.state == "evaluating" and item.evaluation_rate is not None
        ]
        state = replace(
            state,
            evaluation_rate=sum(active_rates) if active_rates else None,
        )

    activity = _event_activity(event)
    if activity is not None:
        recent = list(state.activity)
        if event_type in {
            "provider_turn_activity",
            "repair_activity",
        }:
            recent = [
                item
                for item in recent
                if not (
                    item.component == "provider"
                    and item.slot == activity.slot
                    and item.message.startswith("waiting")
                )
            ]
            recent.insert(0, activity)
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
                        status_message=f"Retry requested for {compact_display_ids(slot.slot)}",
                    ),
                    DashboardAction("retry", slot.slot),
                )
        return (
            replace(state, retry_confirmation=False, status_message="Retry cancelled"),
            None,
        )
    if state.search_editing:
        if key == "ESC":
            return (
                replace(
                    state,
                    search_editing=False,
                    search_query="",
                    status_message="Filter cleared",
                ),
                None,
            )
        if key == "ENTER":
            return replace(state, search_editing=False, status_message="Filter applied"), None
        if key == "BACKSPACE":
            return replace(state, search_query=state.search_query[:-1]), None
        if len(key) == 1 and key.isprintable():
            return replace(state, search_query=state.search_query + key), None
        return state, None
    if key == "0":
        return (
            replace(state, status_message="Preparing all panels copy"),
            DashboardAction("copy", panel=ALL_PANELS_COPY_TARGET),
        )
    if key in PANEL_COPY_KEYS:
        panel = PANEL_COPY_KEYS[key]
        return (
            replace(state, status_message=f"Preparing panel {key} copy"),
            DashboardAction("copy", panel=panel),
        )
    if key.lower() == "i":
        enabled = not state.slot_icon_mode
        return (
            replace(
                state,
                slot_icon_mode=enabled,
                status_message=("Slot phase/state: icons" if enabled else "Slot phase/state: text"),
            ),
            None,
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
    if key == "ENTER":
        return replace(state, view="details", status_message=""), None
    if key == "ESC" and state.search_query:
        return replace(state, search_query="", status_message="Filter cleared"), None
    if key == "ESC":
        return (
            replace(
                state,
                view="matrix",
                retry_confirmation=False,
                status_message="" if state.view != "matrix" else state.status_message,
            ),
            None,
        )
    if key == "TAB" and state.view == "details":
        return replace(state, detail_tab=(state.detail_tab + 1) % len(DETAIL_TABS)), None
    if key == "SHIFT_TAB" and state.view == "details":
        return replace(state, detail_tab=(state.detail_tab - 1) % len(DETAIL_TABS)), None
    if key in {"LEFT", "RIGHT"}:
        generations = sorted(item.generation for item in state.generations)
        if not generations:
            return replace(state, status_message="No retained generation"), None
        current = (
            generations.index(state.displayed_generation)
            if state.displayed_generation in generations
            else len(generations) - 1
        )
        delta = -1 if key == "LEFT" else 1
        target = generations[max(0, min(len(generations) - 1, current + delta))]
        return replace(
            state,
            displayed_generation=target,
            selected_index=0,
            status_message=f"Viewing generation {_human_generation(target)}",
        ), None
    if key == "q":
        if state.experiment_state == "stopping":
            return (
                replace(state, status_message="Immediate interrupt requested"),
                DashboardAction("quit"),
            )
        return (
            replace(
                state,
                experiment_state="stopping",
                generations=tuple(
                    replace(
                        group,
                        slots=tuple(
                            replace(slot, state="stopping")
                            if group.generation == state.generation and slot.state == "queued"
                            else slot
                            for slot in group.slots
                        ),
                    )
                    for group in state.generations
                ),
                status_message=(
                    "Graceful stop requested · active stages will finish · "
                    "press q again to interrupt"
                ),
            ),
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
        return (
            replace(
                state,
                search_editing=True,
                search_query="",
                status_message="Search presentation only",
            ),
            None,
        )
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
        initial_state: DashboardState | None = None,
        persisted_loader: Callable[[int | None], DashboardState] | None = None,
        start_live: bool = True,
    ) -> None:
        self.console = console or Console()
        self.locked_config = dict(locked_config or {})
        self.capabilities = capabilities or DashboardCapabilities()
        self.state = initial_state or DashboardState()
        self._persisted_loader = persisted_loader
        self._history_exhausted = persisted_loader is None or any(
            group.generation == 0 for group in self.state.generations
        )
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
            generations = sorted(group.generation for group in self.state.generations)
            if (
                key == "LEFT"
                and self._persisted_loader is not None
                and not self._history_exhausted
                and generations
                and self.state.displayed_generation == generations[0]
            ):
                previous_oldest = generations[0]
                self.state = _merge_persisted_dashboard_state(
                    self.state,
                    self._persisted_loader(previous_oldest),
                )
                loaded_generations = [group.generation for group in self.state.generations]
                loaded_oldest = min(loaded_generations, default=previous_oldest)
                if loaded_oldest == 0 or loaded_oldest >= previous_oldest:
                    self._history_exhausted = True
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
        text = self._panel_copy_text(panel_name)
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

    def _panel_copy_text(self, panel_name: str) -> str:
        if panel_name == ALL_PANELS_COPY_TARGET:
            sections = []
            for key, numbered_panel_name in PANEL_COPY_KEYS.items():
                title, renderable = self._panel_copy_source(numbered_panel_name)
                sections.append(
                    render_panel_copy_text(
                        f"{key} · {title}",
                        renderable,
                        width=PANEL_COPY_WIDTHS[numbered_panel_name],
                    ).rstrip()
                )
            return "\n\n".join(sections) + "\n"
        title, renderable = self._panel_copy_source(panel_name)
        return render_panel_copy_text(
            title,
            renderable,
            width=PANEL_COPY_WIDTHS[panel_name],
        )

    def _expire_copy_notice_unlocked(self) -> bool:
        if self._copy_notice_until is None or time.monotonic() < self._copy_notice_until:
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
                suffix = f" · {compact_display_ids(slot.slot)}" if slot is not None else ""
                return (
                    f"Slot details{suffix}",
                    self._slot_details(PANEL_COPY_WIDTHS[panel_name], "copy"),
                )
            return (
                f"Slot matrix · generation {_human_generation(self.state.displayed_generation)}",
                self._slot_matrix(PANEL_COPY_WIDTHS[panel_name], "copy"),
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
            return (
                "Quick View",
                self._quick_view_panel(
                    "full",
                    content_width=PANEL_COPY_WIDTHS[panel_name] - 4,
                ),
            )
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
                    _numbered_panel(self._progress(width, horizontal=True), "2"),
                    name="progress",
                    size=5,
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
        quick_view_content_width = max(1, width - width // 2 - 4)
        root["bottom"].split_row(
            Layout(_numbered_panel(self._activity_panel(mode), "7"), ratio=1),
            Layout(
                _numbered_panel(
                    self._quick_view_panel(
                        mode,
                        content_width=quick_view_content_width,
                    ),
                    "8",
                ),
                ratio=1,
            ),
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
            "exhausted": "bold blue",
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
                    (
                        str(_human_generation(self.state.generation))
                        if self.state.generation_limit is None
                        else f"{_human_generation(self.state.generation)}/"
                        f"{self.state.generation_limit}"
                    ),
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
                ("Model", f"{self.state.model}:{self.state.effort}", None),
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
                ("Graph mode", self.state.graph_mode, None),
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
        current_slots = _generation_slots(self.state, self.state.generation).slots
        generating_slots = sum(
            slot.state in {"model", "batched", "repair"}
            for slot in current_slots
        )
        slot_label = "Slots Complete"
        token_label = "Token Budget"
        call_label = "call" if self.state.active_provider_turns == 1 else "calls"
        if generating_slots or self.state.active_provider_turns:
            slot_label = (
                f"Slots · {generating_slots} generating · "
                f"{self.state.active_provider_turns} {call_label}"
            )
            if self.state.hourly_tokens_used == 0:
                token_label = "Token Budget · usage pending"
        configured_values = (
            (
                "Generation",
                _human_generation(self.state.generation),
                self.state.generation_limit,
            ),
            (
                slot_label,
                self.state.completed_slots,
                self.state.population_size,
            ),
            (
                "Model Turn Budget",
                self.state.provider_turns_attempted,
                self.state.max_model_turns,
            ),
            (
                token_label,
                self.state.hourly_tokens_used,
                self.state.hourly_token_limit,
            ),
            (
                "Wall-time Budget",
                int(self._elapsed()),
                int(self.state.wall_seconds) if self.state.wall_seconds is not None else None,
            ),
        )
        values = tuple(item for item in configured_values if item[2] is not None)
        horizontal = horizontal and (len(values) <= 4 or width >= 140)
        if horizontal:
            column_width = max(
                1,
                (width - 4 - 2 * len(values) - max(0, len(values) - 1)) // len(values),
            )
            renderables = [
                _progress_bar(
                    _compact(label, column_width),
                    current,
                    total,
                    width=max(
                        1,
                        column_width
                        - len(f"{_compact_count(current)}/{_compact_count(cast(int, total))}")
                        - 6,
                    ),
                    stacked=True,
                )
                for label, current, total in values
            ]
            grid = Table.grid(expand=True)
            row: list[RenderableType] = []
            for index, renderable in enumerate(renderables):
                grid.add_column(ratio=1)
                row.append(Padding(Align.center(renderable), (0, 1)))
                if index < len(renderables) - 1:
                    grid.add_column(width=1, no_wrap=True)
                    row.append(Text("│\n│\n│", style="grey23"))
            grid.add_row(*row)
        else:
            renderables = [
                _progress_bar(
                    label,
                    current,
                    total,
                    width=max(4, min(12, width // 2 - 24)),
                    stacked=False,
                )
                for label, current, total in values
            ]
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=1, no_wrap=True)
            grid.add_column(ratio=1, no_wrap=True)
            for index in range(0, len(renderables), 2):
                grid.add_row(
                    renderables[index],
                    renderables[index + 1] if index + 1 < len(renderables) else Text(),
                )
        notices: list[RenderableType] = []
        if generating_slots or self.state.active_provider_turns:
            notices.append(
                Align.center(
                    Text(
                        f"Provider work active · {generating_slots} programs in "
                        f"{self.state.active_provider_turns} {call_label} · "
                        "token usage pending until response",
                        style="cyan",
                    )
                )
            )
        if self.state.hourly_limit_reached and self.state.hourly_retry_after:
            notices.append(
                Text(
                    f"Hourly token limit reached · retry after "
                    f"{_clock_time(self.state.hourly_retry_after)}",
                    style="bold red",
                )
            )
        content: RenderableType = Group(grid, *notices) if notices else grid
        return Panel(content, border_style="cyan", padding=(0, 1))

    def _slot_matrix(self, width: int, mode: str) -> Panel:
        group = _generation_slots(self.state, self.state.displayed_generation)
        icon_mode = self.state.slot_icon_mode and mode != "copy"
        table = Table(
            box=None if mode == "copy" else box.MINIMAL_HEAVY_HEAD,
            expand=True,
            padding=(0, 1),
            show_lines=False,
            row_styles=("", "dim"),
        )
        columns: list[tuple[str, Literal["left", "right"], int]] = [
            ("slot", "left", 8),
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
            ("goal ↑", "right", 6),
        ]
        if mode == "compact":
            visible = {
                "slot",
                "phase",
                "state",
                "elapsed",
                "total",
                "validation",
                "probe",
                "candidate / error",
            }
            columns = [item for item in columns if item[0] in visible]
        icon_state_width = (
            4
            if icon_mode and any(_slot_evaluation_percent(slot) is not None for slot in group.slots)
            else 1
        )
        for name, justify, max_width in columns:
            heading = (
                "P"
                if icon_mode and name == "phase"
                else "S"
                if icon_mode and name == "state"
                else name
            )
            column_width = (
                1
                if icon_mode and name == "phase"
                else icon_state_width
                if icon_mode and name == "state"
                else max_width
            )
            if mode == "copy":
                table.add_column(heading, justify=justify, no_wrap=True)
            else:
                table.add_column(
                    heading,
                    justify=justify,
                    no_wrap=True,
                    overflow="ellipsis",
                    max_width=column_width,
                )
        for index, slot in enumerate(group.slots):
            selected = index == self.state.selected_index
            evaluation_percent = _slot_evaluation_percent(slot)
            stopping_active = (
                self.state.experiment_state == "stopping" and slot.state in ACTIVE_STATES
            )
            state_style = STATE_STYLES.get(slot.state, "")
            if stopping_active:
                state_style = f"{state_style} blink".strip()
            values: dict[str, RenderableType] = {
                "slot": (
                    f"▶{compact_display_ids(slot.slot)}"
                    if selected
                    else f" {compact_display_ids(slot.slot)}"
                ),
                "parent": compact_display_ids(slot.parent),
                "phase": PHASE_ICONS.get(slot.phase, "?") if icon_mode else slot.phase,
                "state": Text(
                    f"{evaluation_percent}%"
                    if icon_mode and evaluation_percent is not None
                    else STATE_ICONS.get(slot.state, "?")
                    if icon_mode
                    else _slot_state_label(slot),
                    style=state_style,
                ),
                "elapsed": _duration(self._slot_elapsed(slot)),
                "in": _show(slot.usage.input),
                "out": _show(slot.usage.output),
                "total": _show(slot.usage.total),
                "validation": slot.validation,
                "probe": slot.probe,
                "candidate / error": compact_display_ids(slot.error or slot.candidate or "—"),
                "goal ↑": _objective(slot.objective),
            }
            style = "bold on grey15" if selected else "bold" if slot.state in ACTIVE_STATES else ""
            table.add_row(*(values[name] for name, _, _ in columns), style=style)
        title = (
            f"SLOT MATRIX ({len(group.slots)} total) · "
            f"generation {_human_generation(group.generation)}"
        )
        if self.state.search_query:
            title += f" · filter {self.state.search_query!r}"
        return Panel(table, title=title, border_style="cyan", padding=(0, 0))

    def _minimal_slot(self) -> Panel:
        slot = _selected_slot(self.state)
        if slot is None:
            content = Text("No slot data", style="dim")
        else:
            content = Text()
            content.append(f"▶ {compact_display_ids(slot.slot)}  ")
            content.append(
                _slot_state_label(slot),
                style=STATE_STYLES.get(slot.state, ""),
            )
            content.append(
                f"  {slot.phase}  {_duration(self._slot_elapsed(slot))}  "
                f"tokens {_show(slot.usage.total)}\n"
            )
            content.append(
                compact_display_ids(slot.error or slot.candidate or "No result yet"),
                style="dim",
            )
        return Panel(content, title="SELECTED SLOT", border_style="cyan")

    def _slot_details(self, width: int, mode: str) -> Panel:
        slot = _selected_slot(self.state)
        if slot is None:
            return Panel("No selected slot", title="SLOT DETAILS", border_style="cyan")
        tab = DETAIL_TABS[self.state.detail_tab]
        tabs = Text("  ".join(f"[{name}]" if name == tab else name for name in DETAIL_TABS))
        body: RenderableType
        if tab == "Overview":
            rows: list[tuple[str, object]] = [
                ("slot", compact_display_ids(slot.slot)),
                ("parent/root", compact_display_ids(slot.parent)),
                ("generation", _human_generation(slot.generation)),
                ("state", slot.state),
                ("current phase", slot.phase),
                ("elapsed", _duration(self._slot_elapsed(slot))),
            ]
            if slot.evaluation_id is not None or slot.state == "evaluating":
                progress = (
                    f"{slot.evaluation_completed} / {slot.evaluation_total}"
                    if slot.evaluation_total is not None
                    else f"{slot.evaluation_completed} / —"
                )
                eta = None
                if (
                    slot.evaluation_total is not None
                    and slot.evaluation_rate is not None
                    and slot.evaluation_rate > 0
                ):
                    eta = max(
                        0.0,
                        (slot.evaluation_total - slot.evaluation_completed) / slot.evaluation_rate,
                    )
                rows.extend(
                    (
                        (
                            "evaluation ID",
                            compact_display_ids(slot.evaluation_id or "—"),
                        ),
                        ("pass", slot.evaluation_pass or "—"),
                        ("progress", progress),
                        ("order", slot.evaluation_order),
                        ("graph seed", slot.graph_seed),
                        ("policy seed", slot.policy_seed),
                        (
                            "evaluations/s",
                            f"{slot.evaluation_rate:.3f}/s"
                            if slot.evaluation_rate is not None
                            else "—",
                        ),
                        ("ETA", _duration(eta)),
                    )
                )
            elif (
                slot.state in {"model", "repair"}
                or slot.provider_request_id is not None
                or slot.provider_thread_id is not None
                or slot.provider_turn_id is not None
            ):
                rows.extend(
                    (
                        ("timeout", _duration(slot.timeout_seconds)),
                        ("provider request", slot.provider_request_id or "—"),
                        ("provider thread", slot.provider_thread_id or "—"),
                        ("provider turn", slot.provider_turn_id or "—"),
                        ("repairs", slot.repairs),
                    )
                )
            elif slot.validation != "—" or slot.probe != "—":
                rows.extend(
                    (
                        ("validation", slot.validation),
                        ("probe", slot.probe),
                        ("repairs", slot.repairs),
                    )
                )
            rows.append(("next action", _next_action(slot)))
            body = _paired_key_value_grid(rows) if width >= 110 else _key_value_grid(rows)
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
            Group(tabs, body) if mode == "copy" else Group(tabs, Rule(style="cyan"), body),
            title=f"▶ SLOT DETAILS · {compact_display_ids(slot.slot)}",
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
            ("bottleneck", self.state.native_v3_bottleneck),
            ("episodes/s", _rate(self.state.evaluation_rate)),
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
            (
                "v3 provider/evaluator",
                f"{_objective(self.state.provider_utilization)} / "
                f"{_objective(self.state.evaluator_utilization)}",
            ),
            (
                "candidate/shard/verify",
                f"{self.state.candidate_queue_depth} / "
                f"{self.state.evaluation_shard_queue_depth} / "
                f"{self.state.verification_queue_depth}"
                + (" BP" if self.state.verification_backpressure_active else ""),
            ),
        ]
        if compact:
            rows = rows[: row_limit or 1]
        if not compact:
            rows.extend(
                (
                    ("wall/user/sys", f"{int(elapsed)}/{int(user)}/{int(system)}s"),
                    (
                        "active slots",
                        sum(
                            slot.state in ACTIVE_STATES
                            for slot in _generation_slots(self.state, self.state.generation).slots
                        ),
                    ),
                    (
                        "raw/unique scores",
                        f"{self.state.raw_graph_score_calls} / {self.state.unique_graph_scores}",
                    ),
                    (
                        "raw/unique score/s",
                        f"{_rate(self.state.raw_graph_score_calls_per_second)} / "
                        f"{_rate(self.state.unique_graph_scores_per_second)}",
                    ),
                    (
                        "episode/rewrite/s",
                        f"{_rate(self.state.episodes_per_second)} / "
                        f"{_rate(self.state.accepted_rewrites_per_second)}",
                    ),
                    ("accepted rewrites", self.state.accepted_rewrites),
                    (
                        "score cache hit",
                        _objective(self.state.score_cache_hit_rate),
                    ),
                    (
                        "C++/restart/fallback",
                        f"{self.state.active_cpp_scorers} / "
                        f"{self.state.scorer_restarts} / "
                        f"{self.state.forbidden_fallback_count}",
                    ),
                    (
                        "starved/backpressure",
                        f"{self.state.provider_starvation_seconds:.1f}s / "
                        f"{self.state.provider_backpressure_seconds:.1f}s",
                    ),
                    (
                        "provider latency/batch",
                        f"{self.state.provider_response_latency_seconds:.1f}s / "
                        f"{_objective(self.state.programs_returned_per_call)}",
                    ),
                    (
                        "valid/provider-min",
                        _objective(self.state.valid_programs_per_provider_minute),
                    ),
                    (
                        "gen/val/eval/persist",
                        f"{_objective(self.state.generation_wall_share)} / "
                        f"{_objective(self.state.validation_wall_share)} / "
                        f"{_objective(self.state.evaluation_wall_share)} / "
                        f"{_objective(self.state.persistence_wall_share)}",
                    ),
                    (
                        "first eval/1/50%/all",
                        f"{_objective(self.state.time_to_first_evaluation_seconds)} / "
                        f"{_objective(self.state.first_valid_ast_to_first_worker_seconds)} / "
                        f"{_objective(self.state.first_valid_ast_to_half_workers_seconds)} / "
                        f"{_objective(self.state.first_valid_ast_to_all_workers_seconds)}s",
                    ),
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
        rows: list[tuple[str, str, object]] = [
            ("experiment", "total", cumulative.total),
            ("", "input", cumulative.input),
            ("", "cached", cumulative.cached),
        ]
        if cumulative.cache_write:
            rows.append(("", "cache write", cumulative.cache_write))
        rows.extend(
            (
                ("", "output", cumulative.output),
                ("", "reasoning (in output)", cumulative.reasoning),
                ("session", "total", session.total),
                ("", "input", session.input),
                ("", "cached", session.cached),
            )
        )
        if session.cache_write:
            rows.append(("", "cache write", session.cache_write))
        rows.extend(
            (
                ("", "output", session.output),
                ("", "reasoning (in output)", session.reasoning),
                ("usage", "quality", cumulative.quality),
            )
        )
        if compact:
            compact_rows: list[tuple[str, str, object]] = [
                ("experiment", "total", cumulative.total),
                ("", "reasoning (in output)", cumulative.reasoning),
                ("", "input", session.input),
                ("", "output", session.output),
                ("", "reasoning (in output)", session.reasoning),
                ("usage", "quality", cumulative.quality),
            ]
            rows = compact_rows[: row_limit or 1]
        table = Table.grid(expand=True)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="dim", no_wrap=True)
        table.add_column(justify="right", overflow="ellipsis")
        for section, metric, value in rows:
            table.add_row(section, metric, _show(value))
        return Panel(table, title="Token Accounting", border_style="cyan")

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
            ("best candidate", compact_display_ids(self.state.best_candidate)),
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
            or (
                item.slot is not None
                and (query in item.slot.lower() or query in compact_display_ids(item.slot).lower())
            )
        ][:limit]
        table = Table.grid(expand=True)
        table.add_column(width=8, no_wrap=True)
        table.add_column(width=10, no_wrap=True)
        table.add_column(overflow="ellipsis")
        for item in entries:
            message = (
                f"{compact_display_ids(item.slot)} · {compact_display_ids(item.message)}"
                if item.slot
                else compact_display_ids(item.message)
            )
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

    def _quick_view_panel(
        self,
        mode: str,
        *,
        content_width: int | None = None,
    ) -> Panel:
        if self.state.counterexample_state != "none":
            verified = self.state.counterexample_state == "verified"
            rows: tuple[tuple[str, object], ...] = (
                (
                    "Candidate",
                    compact_display_ids(self.state.counterexample_candidate),
                ),
                ("Order", self.state.counterexample_order),
                ("Edges", self.state.counterexample_edges),
                ("Minimum degree", self.state.counterexample_minimum_degree),
                (
                    "Target lengths",
                    ", ".join(map(str, self.state.counterexample_lengths)) or "—",
                ),
                (
                    "Exact cycle counts",
                    (
                        ", ".join(f"C{length}=0" for length in self.state.counterexample_lengths)
                        if verified
                        else "pending"
                    ),
                ),
                ("Primary", self.state.counterexample_primary),
                ("Independent", self.state.counterexample_independent),
                ("Certificate", self.state.counterexample_certificate),
            )
            title = (
                "Quick View · VERIFIED COUNTEREXAMPLE"
                if verified
                else "Quick View · COUNTEREXAMPLE CANDIDATE"
            )
            return Panel(
                _key_value_grid(rows),
                title=title,
                border_style="bright_green" if verified else "yellow",
            )
        group = _generation_slots(self.state, self.state.displayed_generation)
        active = sum(item.state in ACTIVE_STATES for item in group.slots)
        queued = sum(item.state == "queued" for item in group.slots)
        failed = sum(item.state == "failed" for item in group.slots)
        invalid = sum(item.state == "invalid" for item in group.slots)
        evaluation_config = self.locked_config.get("evaluation")
        effective_orders: tuple[int, ...] = ()
        if isinstance(evaluation_config, Mapping):
            effective_orders = orders_for_generation(
                evaluation_config,
                self.state.displayed_generation,
            )
        orders = ", ".join(map(str, effective_orders)) or "—"
        rows = (
            (
                "Gen / Turn / Slots",
                f"{_human_generation(self.state.displayed_generation)} / "
                f"{self.state.provider_turns_attempted} / {len(group.slots)}",
            ),
            ("Orders", orders),
            ("Best objective", _objective(self.state.best_objective)),
            ("Active / Queued", f"{active} / {queued}"),
            ("Failed / Invalid", f"{failed} / {invalid}"),
            ("Archive", self.state.archive_size),
        )
        if mode == "compact":
            rows = rows[:2]
        if not self.state.objective_history:
            chart = Text("No evaluated objective history yet", style="dim")
        else:
            history = self.state.objective_history
            fixed_width = len("min 0.0000    max 0.0000")
            key_width = max(
                len("Objective history"),
                *(len(label) for label, _value in rows),
            )
            sparkline_width = (
                None if content_width is None else max(1, content_width - key_width - fixed_width)
            )
            sparkline = _sparkline(history, width=sparkline_width)
            chart_line = f"min {min(history):.4f}  {sparkline}  max {max(history):.4f}"
            chart = Text(chart_line, style="green")
        return Panel(
            _key_value_grid((*rows, ("Objective history", chart))),
            title="Quick View",
            border_style="cyan",
        )

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
                    f"{(compact_display_ids(item.slot) + ' ') if item.slot else ''}"
                    f"{compact_display_ids(item.message)}"
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
                "  Enter details           Esc matrix\n"
                "  ←/→ previous/next generation\n"
                "  Tab/Shift+Tab detail tab\n\n"
                "Actions\n"
                "  q stop after active stages; q again interrupts immediately\n"
                "                          p pause/resume scheduling\n"
                "  r confirmed retryable slot\n"
                "  i phase/state icons  c config  l logs  t top  / search  h help\n\n"
                "Panel copy\n"
                "  0 copy all numbered panels in order to OSC 52 and /tmp\n"
                "  1–8 copy one numbered panel to OSC 52 and /tmp\n\n"
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
        quit_label = "[q] interrupt" if self.state.experiment_state == "stopping" else "[q] stop"
        compact_quit_label = (
            "[q]interrupt" if self.state.experiment_state == "stopping" else "[q]stop"
        )
        pause_label = "[p] resume" if self.state.paused else "[p] pause"
        compact_pause_label = "[p]resume" if self.state.paused else "[p]pause"
        labels = (
            (
                "[0] all · [1–8] panel",
                quit_label,
                pause_label,
                "[←/→] gen",
                "[i] icons/text",
                "[r] retry failed",
                "[c] config",
                "[l] logs",
                "[t] top",
                "[/] search",
                "[h] help",
            )
            if width >= 110
            else (
                "[0]all · [1–8]panel",
                compact_quit_label,
                compact_pause_label,
                "[←/→]gen",
                "[i]icons",
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
                or (
                    label.startswith("[t]")
                    and (not self.state.profiling_enabled or self.state.timing_profile is None)
                )
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
        numbered = _UnpaddedTitlePanel(
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


class _UnpaddedTitlePanel(Panel):
    @property
    def _title(self) -> Text | None:
        if not self.title:
            return None
        title = Text.from_markup(self.title) if isinstance(self.title, str) else self.title.copy()
        title.end = ""
        title.plain = title.plain.replace("\n", " ")
        title.no_wrap = True
        title.expand_tabs()
        return title


def _numbered_panel(panel: Panel, number: str) -> _NumberedPanel:
    return _NumberedPanel(panel, number)


def _numbered_panel_title(
    title: str | Text | None,
    number: str,
    panel_width: int,
) -> Text:
    width = max(1, panel_width - 4)
    characters = ["─"] * width
    characters[-1] = " "
    if width >= 2:
        characters[-2] = number
    if width >= 3:
        characters[-3] = " "
    if title is not None:
        plain_title = (
            title.plain if isinstance(title, Text) else Text.from_markup(title).plain
        ).strip()
        label = f" {plain_title} "
        available = max(0, width - 4)
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
    critical: bool = False,
) -> RenderableType:
    columns: list[Any] = []
    if not stacked:
        columns.append(TextColumn(label))
    columns.extend(
        (
            BarColumn(
                bar_width=width,
                complete_style=(
                    "red"
                    if critical and total is not None and current >= total
                    else "yellow"
                    if critical and total is not None and current >= total * 0.8
                    else "cyan"
                ),
                finished_style="red" if critical else "green",
            ),
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
            ratio=f"{_compact_count(current)}/{_compact_count(total)}",
        )
    return Group(Text(label), progress) if stacked else progress


def _compact_count(value: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _clock_time(value: str) -> str:
    time_part = value.split("T", 1)[-1]
    return time_part[:8] if len(time_part) >= 8 else value


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
        table.add_row(label, value if isinstance(value, Text) else _show(value))
    return table


def _paired_key_value_grid(rows: Sequence[tuple[str, object]]) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right", overflow="ellipsis")
    table.add_column(style="dim", no_wrap=True)
    table.add_column(justify="right", overflow="ellipsis")
    for index in range(0, len(rows), 2):
        left_label, left_value = rows[index]
        right_label, right_value = rows[index + 1] if index + 1 < len(rows) else ("", "")
        table.add_row(
            left_label,
            _show(left_value),
            right_label,
            _show(right_value),
        )
    return table


def _token_grid(usage: TokenUsage, *, charged: bool | None) -> Table:
    return _key_value_grid(
        (
            ("input", usage.input),
            ("cached input", usage.cached),
            ("output", usage.output),
            ("reasoning (in output)", usage.reasoning),
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
    if isinstance(value, str):
        return compact_display_ids(value)
    return str(value)


def _human_generation(generation: int) -> int:
    return generation + 1


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
    if value is None:
        return "—"
    truncated = Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
    return f"{truncated:.4f}"


def _slot_state_label(slot: DashboardSlot) -> str:
    percent = _slot_evaluation_percent(slot)
    if percent is not None:
        return f"eval {percent}%"
    return slot.state


def _slot_evaluation_percent(slot: DashboardSlot) -> int | None:
    if slot.state != "evaluating" or slot.evaluation_total is None or slot.evaluation_total <= 0:
        return None
    return round(
        100 * min(slot.evaluation_completed, slot.evaluation_total) / slot.evaluation_total
    )


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


def _sparkline(values: Sequence[float], *, width: int | None = None) -> str:
    if width is not None and len(values) > width:
        values = (
            (values[-1],)
            if width == 1
            else tuple(
                values[round(index * (len(values) - 1) / (width - 1))] for index in range(width)
            )
        )
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
