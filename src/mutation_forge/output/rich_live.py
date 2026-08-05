from __future__ import annotations

import resource
import threading
import time
from typing import TextIO

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mutation_forge.events import Event
from mutation_forge.models import JsonValue
from mutation_forge.output.display_ids import compact_display_ids

REFRESH_INTERVAL_SECONDS = 1.0


class RichLiveSink:
    def __init__(self, *, console: Console | None = None, native: bool = False) -> None:
        self.console = console or Console()
        self.state: dict[str, JsonValue] = {
            "stage": "initializing",
            "latest_event": "none",
            "evaluations": 0,
            "episodes_completed": 0,
            "native": native,
            "state": "starting",
            "slot_states": {},
            "active_model_turns": 0,
            "archive_size": 0,
            "accepted_candidates": 0,
            "invalid_candidates": 0,
            "duplicate_candidates": 0,
            "evaluations_queued": 0,
            "evaluations_active": 0,
            "evaluations_completed": 0,
            "recovered_work": 0,
            "_usage_cumulative": {},
            "_usage_session": {},
            "_archive_seen": {},
        }
        self._native_mode = native
        self._state_lock = threading.RLock()
        self._slot_details: dict[str, dict[str, JsonValue]] = {}
        self._recent_events: list[str] = []
        self._session_started_monotonic: float | None = None
        self._session_cpu_start: tuple[float, float] | None = None
        self._active_operation: dict[str, object] | None = None
        self._last_activity_monotonic: float | None = None
        self._last_activity_label = "waiting"
        self._usage_seen: set[tuple[str, str, str]] = set()
        self._event_keys: set[str] = set()
        self._source_lines = 0
        self.live = Live(
            self._render(),
            console=self.console,
            auto_refresh=False,
            transient=False,
        )
        self.live.start()
        self._last_refresh = time.monotonic()
        self._native_first_refresh = True
        self._refresh_stop = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        if self.console.is_terminal:
            self._refresh_thread = threading.Thread(
                target=self._refresh_loop,
                name="mforge-dashboard-refresh",
                daemon=True,
            )
            self._refresh_thread.start()

    def write(self, event: Event) -> None:
        with self._state_lock:
            self._write(event)

    def _write(self, event: Event) -> None:
        raw_idempotency_key = event.payload.get("idempotency_key")
        if isinstance(raw_idempotency_key, str):
            idempotency_key = f"{event.event_type}:{raw_idempotency_key}"
            if idempotency_key in self._event_keys:
                return
            self._event_keys.add(idempotency_key)
        if event.event_type == "generation_started":
            self._slot_details.clear()
            self.state["slot_states"] = {}
        self.state.update(event.payload)
        self.state["latest_event"] = event.event_type
        self.state["run_id"] = event.run_id
        if event.event_type in {
            "provider_turn_started",
            "provider_turn_activity",
            "provider_call_started",
            "provider_call_activity",
        }:
            self.state["phase"] = "repair" if event.payload.get("phase") == "repair" else "provider"
        elif event.event_type == "repair_started":
            self.state["phase"] = "repair"
        elif event.event_type in {
            "validation_started",
            "behavior_probe_started",
            "evaluation_started",
            "selection_started",
        }:
            self.state["phase"] = event.event_type.removesuffix("_started")
        if event.event_type == "session_started":
            self._session_started_monotonic = time.monotonic()
        native_event = event.event_type in {
            "preflight_started",
            "preflight_completed",
            "workspace_initialized",
            "workspace_resumed",
            "session_started",
            "generation_started",
            "generation_completed",
            "slot_queued",
            "provider_turn_started",
            "provider_turn_activity",
            "provider_turn_completed",
            "provider_turn_failed",
            "provider_call_started",
            "provider_call_activity",
            "provider_call_completed",
            "provider_call_failed",
            "candidate_validated",
            "repair_started",
            "repair_activity",
            "repair_completed",
            "validation_started",
            "validation_completed",
            "behavior_probe_started",
            "behavior_probe_completed",
            "candidate_archived",
            "evaluation_started",
            "evaluation_progress",
            "evaluation_completed",
            "evaluation_failed",
            "selection_started",
            "selection_completed",
            "budget_boundary_reached",
            "experiment_exhausted",
            "experiment_completed",
            "experiment_interrupted",
            "experiment_failed",
            "counterexample_candidate_found",
            "counterexample_primary_verification_started",
            "counterexample_primary_verification_completed",
            "counterexample_independent_verification_started",
            "counterexample_independent_verification_completed",
            "counterexample_verification_conflict",
            "counterexample_verified",
        }
        if native_event:
            self.state["native"] = True
            self._update_counterexample(event)
            self._update_native_counters(event)
            if event.event_type.startswith("provider_call_"):
                self._update_native_provider_call_slots(event)
            else:
                self._update_native_slot(event)
            self._record_recent_event(event)
            self._track_native_activity(event)
        if event.event_type == "session_started":
            self.state["state"] = "running"
        elif event.event_type == "experiment_completed":
            self.state["state"] = "completed"
        elif event.event_type == "experiment_exhausted":
            self.state["state"] = "exhausted"
        elif event.event_type == "experiment_interrupted":
            self.state["state"] = "interrupted"
        elif event.event_type == "experiment_failed":
            self.state["state"] = "failed"
        elif event.event_type == "budget_boundary_reached" and event.payload.get("state") in {
            "idle",
            "budget_exhausted",
        }:
            self.state["state"] = "idle"
        if event.event_type == "baseline_started":
            self.state["stage"] = "baseline"
        elif event.event_type in {"run_completed", "experiment_completed"}:
            self.state["stage"] = "completed"
        elif event.event_type in {"run_failed", "experiment_failed"}:
            self.state["stage"] = "failed"
        now = time.monotonic()
        terminal = event.event_type in {
            "run_completed",
            "run_failed",
            "experiment_completed",
            "experiment_exhausted",
            "experiment_interrupted",
            "experiment_failed",
            "counterexample_verified",
            "counterexample_verification_conflict",
        } or (
            event.event_type == "budget_boundary_reached"
            and event.payload.get("state") in {"idle", "budget_exhausted"}
        )
        immediate_native = native_event and (
            self._native_first_refresh
            or event.event_type
            in {
                "session_started",
                "generation_started",
                "provider_turn_started",
                "repair_started",
                "validation_started",
                "behavior_probe_started",
                "evaluation_started",
            }
        )
        if terminal or immediate_native or now - self._last_refresh >= REFRESH_INTERVAL_SECONDS:
            self.live.update(self._render(), refresh=True)
            self._last_refresh = now
            if native_event:
                self._native_first_refresh = False

    def _refresh_loop(self) -> None:
        while not self._refresh_stop.wait(REFRESH_INTERVAL_SECONDS):
            try:
                with self._state_lock:
                    self._update_live_rates()
                    self.live.update(self._render_unlocked(), refresh=True)
            except Exception:
                return

    def _update_live_rates(self) -> None:
        if self._session_started_monotonic is None:
            return
        elapsed = max(0.0, time.monotonic() - self._session_started_monotonic)
        self.state["elapsed_seconds"] = elapsed
        wall = self.state.get("configured_wall_seconds")
        if isinstance(wall, int | float) and not isinstance(wall, bool):
            self.state["remaining_seconds"] = max(0.0, float(wall) - elapsed)
        completed = self.state.get("evaluations_completed")
        if isinstance(completed, int) and not isinstance(completed, bool) and elapsed > 0:
            self.state["evaluations_per_second"] = completed / elapsed
        episodes = self.state.get("episodes_completed")
        if isinstance(episodes, int) and not isinstance(episodes, bool) and elapsed > 0:
            self.state["episodes_per_second"] = episodes / elapsed
        turns = self.state.get("provider_turns_completed")
        if isinstance(turns, int) and not isinstance(turns, bool) and elapsed > 0:
            self.state["turns_per_minute"] = turns * 60.0 / elapsed
        if elapsed > 0:
            self.state["source_lines_per_second"] = self._source_lines / elapsed
        try:
            user, system = resource.getrusage(resource.RUSAGE_SELF)[:2]
        except (AttributeError, OSError):
            user, system = 0.0, 0.0
        if self._session_cpu_start is not None:
            self.state["user_seconds"] = max(0.0, float(user) - self._session_cpu_start[0])
            self.state["system_seconds"] = max(0.0, float(system) - self._session_cpu_start[1])
        if self._last_activity_monotonic is not None:
            self.state["last_activity_age_seconds"] = max(
                0.0, time.monotonic() - self._last_activity_monotonic
            )

    def _track_native_activity(self, event: Event) -> None:
        """Keep heartbeat state independent of domain-event arrival rate."""

        payload = event.payload
        event_type = event.event_type
        now = time.monotonic()
        if event_type == "session_started":
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self._session_cpu_start = (float(usage.ru_utime), float(usage.ru_stime))
            self._last_activity_monotonic = now
            self._last_activity_label = "session"
        if event_type in {
            "provider_turn_started",
            "provider_call_started",
            "repair_started",
        }:
            slot = payload.get("slot", payload.get("slot_ids"))
            phase = "repair" if event_type == "repair_started" else "provider"
            timeout_ns = payload.get("timeout_ns")
            timeout = (
                float(timeout_ns) / 1e9
                if isinstance(timeout_ns, int) and not isinstance(timeout_ns, bool)
                else payload.get("timeout_seconds", 120.0)
            )
            self._active_operation = {
                "phase": phase,
                "slot": slot if isinstance(slot, str) else "?",
                "generation": payload.get("generation"),
                "started": now,
                "timeout": timeout,
                "thread": payload.get("provider_thread_id"),
                "turn": payload.get("provider_turn_id"),
            }
            self._last_activity_monotonic = now
            self._last_activity_label = event_type.removesuffix("_started")
        elif event_type in {
            "provider_turn_activity",
            "provider_call_activity",
            "repair_activity",
        }:
            elapsed_ns = payload.get("operation_elapsed_ns")
            elapsed_value = payload.get("operation_elapsed_seconds")
            operation_elapsed = (
                float(elapsed_ns) / 1e9
                if isinstance(elapsed_ns, int) and not isinstance(elapsed_ns, bool)
                else float(elapsed_value)
                if isinstance(elapsed_value, (int, float)) and not isinstance(elapsed_value, bool)
                else 0.0
            )
            timeout_ns = payload.get("timeout_ns")
            timeout = (
                float(timeout_ns) / 1e9
                if isinstance(timeout_ns, int) and not isinstance(timeout_ns, bool)
                else payload.get("timeout_seconds", 120.0)
            )
            if self._active_operation is None:
                self._active_operation = {
                    "phase": "repair" if event_type == "repair_activity" else "provider",
                    "slot": payload.get("slot", payload.get("slot_ids", "?")),
                    "generation": payload.get("generation"),
                    "started": now - operation_elapsed,
                    "timeout": timeout,
                }
            elif event_type == "provider_call_activity":
                self._active_operation["started"] = now - operation_elapsed
            self._last_activity_monotonic = now
            self._last_activity_label = event_type.removesuffix("_activity")
            for key in ("provider_thread_id", "provider_turn_id"):
                value = payload.get(key)
                if value not in (None, "") and self._active_operation is not None:
                    self._active_operation[key.removeprefix("provider_")] = value
        elif event_type in {
            "provider_turn_completed",
            "provider_turn_failed",
            "provider_call_completed",
            "provider_call_failed",
            "repair_completed",
        }:
            self._last_activity_monotonic = now
            self._last_activity_label = event_type.removesuffix("_completed").removesuffix(
                "_failed"
            )
            calls_in_flight = payload.get("provider_calls_in_flight")
            keep_provider_call = (
                event_type == "provider_call_completed"
                and isinstance(calls_in_flight, int)
                and not isinstance(calls_in_flight, bool)
                and calls_in_flight > 0
            )
            if event_type != "repair_completed" and not keep_provider_call:
                self._active_operation = None
        elif event_type in {
            "validation_started",
            "behavior_probe_started",
            "evaluation_started",
        }:
            phase = str(payload.get("phase", event_type.removesuffix("_started")))
            self._active_operation = {
                "phase": phase,
                "slot": payload.get("slot", "?"),
                "generation": payload.get("generation"),
                "started": now,
                "timeout": payload.get("timeout_seconds"),
            }
            self._last_activity_monotonic = now
            self._last_activity_label = phase
        elif event_type in {
            "validation_completed",
            "behavior_probe_completed",
            "evaluation_completed",
            "evaluation_failed",
        }:
            self._last_activity_monotonic = now
            self._active_operation = None

    def _live_elapsed_seconds(self) -> float:
        value = self.state.get("elapsed_seconds")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return max(0.0, float(value))
        if self._session_started_monotonic is not None:
            return max(0.0, time.monotonic() - self._session_started_monotonic)
        return 0.0

    def _update_counterexample(self, event: Event) -> None:
        payload = event.payload
        event_type = event.event_type
        if event_type == "counterexample_candidate_found":
            self.state["counterexample_state"] = "candidate"
            self.state["counterexample_candidate"] = payload.get("candidate_id")
            self.state["counterexample_order"] = payload.get("order")
            self.state["counterexample_edges"] = payload.get("edge_count")
            self.state["counterexample_minimum_degree"] = payload.get("minimum_degree")
            self.state["counterexample_lengths"] = payload.get("target_forbidden_lengths")
            self.state["phase"] = "exact verification"
        elif event_type == "counterexample_primary_verification_started":
            self.state["counterexample_state"] = "primary_verifying"
            self.state["counterexample_primary"] = "running"
        elif event_type == "counterexample_primary_verification_completed":
            self.state["counterexample_primary"] = (
                f"{payload.get('status', 'UNKNOWN')} · "
                f"{'complete' if payload.get('complete') is True else 'incomplete'}"
            )
            if payload.get("status") == "VERIFIED" and payload.get("complete") is True:
                self.state["counterexample_state"] = "primary_verified"
        elif event_type == "counterexample_independent_verification_started":
            self.state["counterexample_state"] = "independent_verifying"
            self.state["counterexample_independent"] = "running"
        elif event_type == "counterexample_independent_verification_completed":
            self.state["counterexample_independent"] = (
                f"{payload.get('status', 'UNKNOWN')} · "
                f"{'complete' if payload.get('complete') is True else 'incomplete'}"
            )
        elif event_type == "counterexample_verification_conflict":
            self.state["counterexample_state"] = "conflict"
        elif event_type == "counterexample_verified":
            self.state["counterexample_state"] = "verified"
            self.state["counterexample_candidate"] = payload.get("candidate_id")
            self.state["counterexample_certificate"] = payload.get("certificate")

    def _update_native_counters(self, event: Event) -> None:
        """Accumulate counters when an event carries only a local delta."""

        payload = event.payload

        def integer(name: str, default: int = 0) -> int:
            value = self.state.get(name, default)
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else default

        def add(name: str, amount: int = 1) -> None:
            self.state[name] = integer(name) + amount

        if event.event_type == "generation_started":
            self.state["completed_slots"] = 0
            self.state["active_model_turns"] = 0
        elif event.event_type == "session_started":
            usage = payload.get("usage")
            if isinstance(usage, dict):
                self.state["_usage_cumulative"] = dict(usage)
            session_usage = payload.get("session_usage")
            self.state["_usage_session"] = (
                dict(session_usage) if isinstance(session_usage, dict) else {}
            )
            for key in (
                "cumulative_provider_turns",
                "cumulative_evaluations",
                "cumulative_candidates",
                "cumulative_tokens",
            ):
                if key in payload:
                    self.state[key] = payload[key]
        elif event.event_type == "slot_queued":
            slot = payload.get("slot")
            status = payload.get("status")
            slots = self.state.get("slot_states")
            if isinstance(slot, str) and isinstance(status, str) and isinstance(slots, dict):
                slots[slot] = status
            completed = payload.get("completed_slots")
            if isinstance(completed, int) and not isinstance(completed, bool):
                self.state["completed_slots"] = max(integer("completed_slots"), completed)
            if payload.get("recovered") is True:
                add("recovered_work")
        elif event.event_type == "provider_turn_started":
            self.state["active_model_turns"] = integer("active_model_turns") + 1
            add("provider_turns_attempted")
        elif event.event_type == "provider_call_started":
            in_flight = payload.get("provider_calls_in_flight")
            self.state["active_model_turns"] = (
                in_flight
                if isinstance(in_flight, int) and not isinstance(in_flight, bool)
                else integer("active_model_turns") + 1
            )
            add("provider_turns_attempted")
        elif event.event_type == "provider_call_activity":
            in_flight = payload.get("provider_calls_in_flight")
            if isinstance(in_flight, int) and not isinstance(in_flight, bool):
                self.state["active_model_turns"] = in_flight
        elif event.event_type in {"provider_call_completed", "provider_call_failed"}:
            in_flight = payload.get("provider_calls_in_flight")
            self.state["active_model_turns"] = (
                in_flight
                if isinstance(in_flight, int) and not isinstance(in_flight, bool)
                else max(0, integer("active_model_turns") - 1)
            )
            if event.event_type == "provider_call_completed":
                add("provider_turns_completed")
                add("responses_received")
            else:
                self.state["error_summary"] = payload.get(
                    "error_message",
                    "provider call failed",
                )
        elif event.event_type in {"provider_turn_completed", "provider_turn_failed"}:
            self.state["active_model_turns"] = max(0, integer("active_model_turns") - 1)
            if payload.get("retained") is True:
                # A recovered terminal turn is observed by the coordinator
                # but does not represent new provider work in this session.
                self.state["provider_turns_attempted"] = max(
                    0, integer("provider_turns_attempted") - 1
                )
            elif event.event_type == "provider_turn_completed":
                add("provider_turns_completed")
                add("responses_received")
            if event.event_type == "provider_turn_failed" and payload.get("charged") is True:
                add("charged_failed_turns")
            usage = payload.get("usage")
            if isinstance(usage, dict) and payload.get("retained") is not True:
                current = self.state.get("_usage_cumulative")
                cumulative = dict(current) if isinstance(current, dict) else {}
                session_current = self.state.get("_usage_session")
                session_usage = dict(session_current) if isinstance(session_current, dict) else {}
                usage_key = (
                    str(payload.get("generation", "")),
                    str(payload.get("slot", "")),
                    str(
                        payload.get(
                            "idempotency_key",
                            payload.get(
                                "provider_turn_id",
                                f"{payload.get('phase', 'initial')}:"
                                f"{payload.get('repair_attempt', 0)}",
                            ),
                        )
                    ),
                )
                duplicate_usage = usage_key in self._usage_seen
                self._usage_seen.add(usage_key)
                for key in (
                    "inputTokens",
                    "cachedInputTokens",
                    "outputTokens",
                    "reasoningOutputTokens",
                    "totalTokens",
                ):
                    value = usage.get(key)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and not duplicate_usage
                    ):
                        prior = cumulative.get(key, 0)
                        cumulative[key] = (
                            prior + value
                            if isinstance(prior, int) and not isinstance(prior, bool)
                            else value
                        )
                        session_prior = session_usage.get(key, 0)
                        session_usage[key] = (
                            session_prior + value
                            if (
                                isinstance(session_prior, int)
                                and not isinstance(session_prior, bool)
                            )
                            else value
                        )
                cumulative["quality"] = payload.get(
                    "usage_quality", cumulative.get("quality", "unknown")
                )
                session_usage["quality"] = payload.get(
                    "usage_quality", session_usage.get("quality", "unknown")
                )
                self.state["_usage_cumulative"] = cumulative
                self.state["_usage_session"] = session_usage
                self.state["usage"] = cumulative
        elif event.event_type == "repair_completed":
            if payload.get("retained") is not True:
                add("repair_turns")
        elif event.event_type == "validation_completed":
            add("validation_outcomes")
            if payload.get("parse_outcome") in {"valid", "invalid"}:
                add("parse_outcomes")
            if payload.get("schema_outcome") in {"valid", "invalid"}:
                add("schema_outcomes")
        elif event.event_type == "behavior_probe_completed":
            add("behavior_probe_outcomes")
        elif event.event_type == "candidate_archived":
            status = payload.get("status")
            candidate_id = payload.get("candidate_id")
            seen = self.state.get("_archive_seen")
            duplicate_event = (
                isinstance(candidate_id, str)
                and isinstance(seen, dict)
                and candidate_id in seen
                and seen[candidate_id] == status
            )
            if isinstance(candidate_id, str) and isinstance(seen, dict):
                seen[candidate_id] = status
            if not duplicate_event:
                if status == "accepted":
                    add("accepted_candidates")
                elif status == "duplicate":
                    add("duplicate_candidates")
                elif status == "invalid":
                    add("invalid_candidates")
            archive_size = payload.get("archive_size")
            if isinstance(archive_size, int) and not isinstance(archive_size, bool):
                self.state["archive_size"] = max(integer("archive_size"), archive_size)
            source_lines = payload.get("source_lines")
            if isinstance(source_lines, int) and not isinstance(source_lines, bool):
                self._source_lines += max(0, source_lines)
                self.state["source_lines"] = self._source_lines
        elif event.event_type == "evaluation_started":
            queued = payload.get("evaluations_queued")
            if isinstance(queued, int) and not isinstance(queued, bool):
                self.state["evaluations_queued"] = max(integer("evaluations_queued"), queued)
            else:
                add("evaluations_queued")
            self.state["evaluations_active"] = integer("evaluations_active") + 1
            total = payload.get("evaluation_total")
            if isinstance(total, int) and not isinstance(total, bool):
                self.state["episodes_total"] = total
                self.state["episodes_completed"] = 0
        elif event.event_type == "evaluation_progress":
            active = payload.get("evaluations_active")
            self.state["evaluations_active"] = max(
                integer("evaluations_active"),
                active if isinstance(active, int) and not isinstance(active, bool) else 0,
            )
            for key in (
                "order",
                "graph_seed",
                "policy_seed",
                "evaluations",
                "evaluations_per_second",
                "development_progress",
                "replay_progress",
                "active_workers",
                "worker_count",
                "completed",
                "total",
                "evaluation_total",
                "pass",
                "pass_progress",
            ):
                if key in payload:
                    if key == "evaluations_per_second":
                        self.state["episodes_per_second"] = payload[key]
                    else:
                        self.state[key] = payload[key]
            completed = payload.get("completed")
            if isinstance(completed, int) and not isinstance(completed, bool):
                self.state["episodes_completed"] = max(integer("episodes_completed"), completed)
            total = payload.get("total")
            if isinstance(total, int) and not isinstance(total, bool):
                self.state["episodes_total"] = total
        elif event.event_type in {"evaluation_completed", "evaluation_failed"}:
            self.state["evaluations_active"] = max(0, integer("evaluations_active") - 1)
            completed = payload.get("evaluations_completed")
            if isinstance(completed, int) and not isinstance(completed, bool):
                self.state["evaluations_completed"] = max(
                    integer("evaluations_completed"), completed
                )
            else:
                add("evaluations_completed")
            for key in ("mean_auc", "best_auc", "current_objective", "best_objective"):
                if key in payload:
                    self.state[key] = payload[key]
            if "best_candidate_id" in payload:
                self.state["best_candidate_id"] = payload["best_candidate_id"]
            if "best_score" in payload:
                self.state["best_score"] = payload["best_score"]
            if "baseline_comparison" in payload:
                self.state["baseline_comparison"] = payload["baseline_comparison"]
            if event.event_type == "evaluation_failed":
                self.state["error_summary"] = payload.get("error", "evaluation failed")
        for key in (
            "ir",
            "improvement_rate",
            "acceptance_rate",
            "proposal_evaluations_per_second",
            "timeout_seconds",
            "worker_utilization",
        ):
            if key in payload:
                self.state[key] = payload[key]
        if event.event_type == "provider_turn_failed":
            self.state["error_summary"] = payload.get("error", "provider turn failed")
        elif event.event_type == "validation_completed" and payload.get("valid") is False:
            self.state["error_summary"] = payload.get("error", "validation failed")
        elif event.event_type == "behavior_probe_completed" and payload.get("valid") is False:
            self.state["error_summary"] = payload.get("error", "behavior probe failed")

    def _update_native_slot(self, event: Event) -> None:
        payload = event.payload
        slot = payload.get("slot", payload.get("call_id"))
        if not isinstance(slot, str):
            return
        detail = self._slot_details.setdefault(slot, {})
        for key in ("generation", "parent_id", "phase"):
            value = payload.get(key)
            if value is not None:
                detail[key] = value
        event_type = event.event_type
        phase = payload.get("phase")
        state = payload.get("status")
        now = time.monotonic()
        if event_type == "provider_turn_started":
            state = "repair_running" if phase == "repair" else "model"
            detail.setdefault("_slot_started_at", now)
            detail.pop("error", None)
        elif event_type == "provider_turn_completed":
            state = (
                "repair_running"
                if phase == "repair"
                else "validating"
                if payload.get("accepted") is True
                else "failed"
            )
        elif event_type == "provider_turn_failed":
            state = "repair_failed" if phase == "repair" else "failed"
        elif event_type == "repair_started":
            state = "repair_running"
            detail.setdefault("_slot_started_at", now)
        elif event_type == "repair_completed":
            repair_state = payload.get("repair_state")
            final_state = payload.get("status")
            state = (
                final_state
                if final_state in {"accepted", "repair_pending", "invalid"}
                else repair_state
                if isinstance(repair_state, str)
                else "repair_failed"
            )
        elif event_type == "validation_started":
            state = "validating"
            detail.setdefault("_slot_started_at", now)
        elif event_type == "validation_completed":
            state = "validating" if payload.get("valid") is True else "validation_failed"
        elif event_type == "behavior_probe_started":
            state = "probing"
            detail.setdefault("_slot_started_at", now)
        elif event_type == "behavior_probe_completed":
            state = "probing" if payload.get("valid") is True else "failed"
        elif event_type == "evaluation_started":
            state = "evaluating"
            detail.setdefault("_slot_started_at", now)
        elif event_type == "slot_queued" and payload.get("status") == "recovered":
            recovered_status = payload.get("recovered_status")
            state = (
                recovered_status
                if isinstance(recovered_status, str) and recovered_status
                else "recovered"
            )
        elif event_type == "slot_queued" and payload.get("status") == "retrying":
            state = "retrying"
            detail.pop("error", None)
        elif event_type == "candidate_archived":
            state = payload.get("status")
            source_lines = payload.get("source_lines")
            if isinstance(source_lines, int) and not isinstance(source_lines, bool):
                detail["source_lines"] = source_lines
        if isinstance(state, str) and state:
            detail["state"] = state
            slots = self.state.get("slot_states")
            if not isinstance(slots, dict):
                slots = {}
                self.state["slot_states"] = slots
            slots[slot] = state
        validation_codes = payload.get("validation_codes")
        if isinstance(validation_codes, list):
            codes = [str(code) for code in validation_codes if isinstance(code, str) and code]
            if codes:
                detail["error"] = ",".join(codes)
        if event_type in {
            "provider_turn_completed",
            "provider_turn_failed",
            "repair_completed",
            "validation_completed",
            "behavior_probe_completed",
            "evaluation_completed",
            "evaluation_failed",
            "candidate_archived",
        }:
            started = detail.get("_slot_started_at")
            if isinstance(started, (int, float)):
                detail["elapsed_seconds"] = max(0.0, now - started)
            if event_type in {
                "evaluation_completed",
                "evaluation_failed",
                "candidate_archived",
            }:
                detail.pop("_slot_started_at", None)
        usage = payload.get("usage")
        tokens = payload.get("totalTokens")
        if isinstance(usage, dict):
            tokens = usage.get("totalTokens", tokens)
        if isinstance(tokens, int) and not isinstance(tokens, bool):
            detail["tokens"] = tokens
        candidate = payload.get("candidate_id")
        if isinstance(candidate, str) and candidate:
            detail["candidate"] = candidate
        score = payload.get("best_score", payload.get("best_objective"))
        if isinstance(score, int | float) and not isinstance(score, bool):
            detail["score"] = score
        error = payload.get("error")
        if isinstance(error, str) and error:
            detail["error"] = error

    def _update_native_provider_call_slots(self, event: Event) -> None:
        payload = event.payload
        encoded_slots = payload.get("slot_ids")
        if not isinstance(encoded_slots, str):
            return
        now = time.monotonic()
        event_type = event.event_type
        elapsed_ns = payload.get("operation_elapsed_ns")
        elapsed = (
            float(elapsed_ns) / 1e9
            if isinstance(elapsed_ns, int) and not isinstance(elapsed_ns, bool)
            else payload.get("operation_elapsed_seconds")
        )
        if not isinstance(elapsed, int | float) or isinstance(elapsed, bool):
            latency_ns = payload.get("latency_ns")
            elapsed = (
                float(latency_ns) / 1e9
                if isinstance(latency_ns, int) and not isinstance(latency_ns, bool)
                else None
            )
        error_type = payload.get("error_type")
        error_message = payload.get("error_message")
        diagnostic = ": ".join(
            str(value)
            for value in (error_type, error_message)
            if isinstance(value, str) and value
        )
        slots = self.state.get("slot_states")
        if not isinstance(slots, dict):
            slots = {}
            self.state["slot_states"] = slots
        for slot in (item.strip() for item in encoded_slots.split(",")):
            if not slot:
                continue
            detail = self._slot_details.setdefault(slot, {})
            detail["generation"] = payload.get("generation")
            if event_type == "provider_call_started":
                detail["phase"] = "provider"
                detail["state"] = "model"
                detail["_slot_started_at"] = now
            elif event_type == "provider_call_activity":
                detail["phase"] = "provider"
                detail["state"] = "model"
                if elapsed is not None:
                    detail["elapsed_seconds"] = float(elapsed)
                    detail["_slot_started_at"] = now - float(elapsed)
            elif event_type == "provider_call_failed":
                detail["phase"] = "response"
                detail["state"] = "failed"
                detail["error"] = diagnostic or "provider call failed"
                detail.pop("_slot_started_at", None)
                if elapsed is not None:
                    detail["elapsed_seconds"] = float(elapsed)
            else:
                detail["phase"] = "response"
                detail["state"] = "validating"
                detail.pop("_slot_started_at", None)
                if elapsed is not None:
                    detail["elapsed_seconds"] = float(elapsed)
            slots[slot] = detail["state"]

    def _record_recent_event(self, event: Event) -> None:
        meaningful = {
            "preflight_started",
            "preflight_completed",
            "workspace_initialized",
            "workspace_resumed",
            "session_started",
            "generation_started",
            "slot_queued",
            "provider_turn_started",
            "provider_turn_activity",
            "provider_turn_completed",
            "provider_turn_failed",
            "provider_call_started",
            "provider_call_activity",
            "provider_call_completed",
            "provider_call_failed",
            "candidate_validated",
            "repair_started",
            "repair_activity",
            "repair_completed",
            "validation_completed",
            "behavior_probe_completed",
            "candidate_archived",
            "evaluation_completed",
            "evaluation_failed",
            "selection_completed",
            "checkpoint_written",
            "budget_boundary_reached",
            "experiment_completed",
            "experiment_exhausted",
            "experiment_interrupted",
            "experiment_failed",
            "counterexample_candidate_found",
            "counterexample_primary_verification_started",
            "counterexample_primary_verification_completed",
            "counterexample_independent_verification_started",
            "counterexample_independent_verification_completed",
            "counterexample_verification_conflict",
            "counterexample_verified",
        }
        if event.event_type not in meaningful:
            return
        payload = event.payload
        if event.event_type == "candidate_archived" and payload.get("status") == "accepted":
            pass
        elif event.event_type == "slot_queued" and payload.get("status") != "recovered":
            return
        slot = payload.get("slot")
        timestamp = event.timestamp[11:19] if len(event.timestamp) >= 19 else ""
        slot_label = compact_display_ids(slot) if isinstance(slot, str) else "work"
        if event.event_type in {"provider_turn_started", "provider_call_started"}:
            phase = payload.get("phase")
            phase_label = f" {phase}" if isinstance(phase, str) and phase else ""
            entry = f"{timestamp} prompt sent {slot_label}{phase_label}".strip()
        elif event.event_type in {"provider_turn_completed", "provider_call_completed"}:
            tokens = payload.get("totalTokens")
            token_label = (
                f" {tokens:,} tok"
                if isinstance(tokens, int) and not isinstance(tokens, bool)
                else ""
            )
            entry = f"{timestamp} response received {slot_label}{token_label}".strip()
        elif event.event_type in {"provider_turn_failed", "provider_call_failed"}:
            error = payload.get("error", payload.get("error_message"))
            error_label = f": {error}" if isinstance(error, str) and error else ""
            entry = f"{timestamp} response failed {slot_label}{error_label}".strip()
        else:
            label = event.event_type.removeprefix("experiment_").replace("_", " ")
            if isinstance(slot, str):
                label += f" {compact_display_ids(slot)}"
            status = payload.get("status")
            if isinstance(status, str) and status:
                label += f" {status}"
            error = payload.get("error")
            if isinstance(error, str) and error:
                label += f": {error}"
            entry = f"{timestamp} {label}".strip()
        if event.event_type in {
            "provider_turn_activity",
            "provider_call_activity",
            "repair_activity",
        }:
            # Heartbeats should keep the tail current without consuming all
            # six rows during a long model turn.
            elapsed_ns = payload.get("operation_elapsed_ns")
            elapsed = (
                float(elapsed_ns) / 1e9
                if isinstance(elapsed_ns, int) and not isinstance(elapsed_ns, bool)
                else payload.get("operation_elapsed_seconds")
            )
            elapsed_label = (
                f" {float(elapsed):.0f}s"
                if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                else ""
            )
            waiting = compact_display_ids(
                f"{timestamp} response waiting {slot_label}{elapsed_label}".strip()
            )
            if self._recent_events and "response waiting" in self._recent_events[-1]:
                self._recent_events[-1] = waiting
                return
            entry = waiting
        self._recent_events.append(compact_display_ids(entry))
        del self._recent_events[:-6]

    def _render(self) -> Group | Panel:
        with self._state_lock:
            return self._render_unlocked()

    def _render_unlocked(self) -> Group | Panel:
        if self.state.get("native") is True or self._native_mode:
            return self._render_native()
        return self._render_legacy()

    def _render_native(self) -> Panel:
        width = max(40, self.console.size.width)
        height = max(8, self.console.size.height)
        profile_line = self._native_profile_line()
        metrics = self._native_metrics_line(width - 4)
        progress_lines = self._native_progress_lines(width - 4)
        token_line = self._native_token_line(width - 4)
        heartbeat = self._native_heartbeat_line(width - 4)
        show_metrics = bool(metrics) and height >= 12
        show_tokens = bool(token_line) and height >= 14
        show_heartbeat = bool(heartbeat) and height >= 12
        show_profile = profile_line is not None and height >= 18
        counterexample_line = self._native_counterexample_line(width - 4)
        activity_limit = 3 if height >= 16 else 1
        fixed = (
            1
            + 1
            + (1 if counterexample_line else 0)
            + len(progress_lines)
            + (1 if show_metrics else 0)
            + (1 if show_tokens else 0)
            + (1 if show_heartbeat else 0)
            + 1
            + activity_limit
        )
        if show_profile:
            fixed += 1
        details = self._native_slot_rows()
        max_rows = max(0, min(len(details), height - 2 - fixed - 1))
        parts: list[Text | Table] = [Text(self._native_header(width), style="bold")]
        if counterexample_line:
            verified = self.state.get("counterexample_state") == "verified"
            parts.append(
                Text(
                    counterexample_line,
                    style="bold bright_green" if verified else "bold yellow",
                )
            )
        summary = self._native_summary_line()
        if summary:
            parts.append(Text(summary))
        parts.extend(Text(line) for line in progress_lines)
        if show_metrics:
            parts.append(Text(metrics))
        if show_tokens:
            parts.append(Text(token_line))
        if show_heartbeat:
            parts.append(Text(heartbeat))
        if details and max_rows:
            parts.append(self._native_slot_table(width, details[:max_rows]))
        else:
            parts.append(Text("Slots  waiting for generation"))
        activity = self._recent_events[-activity_limit:]
        if activity:
            parts.extend(
                Text(f"Activity  {item}" if index == 0 else f"          {item}")
                for index, item in enumerate(activity)
            )
        else:
            parts.append(Text("Activity  waiting for native events"))
        if show_profile and profile_line is not None:
            parts.append(Text(profile_line))
        return Panel(
            Group(*parts),
            title="Mutation Forge Lab · Native experiment",
            border_style=(
                "bright_green"
                if self.state.get("counterexample_state") == "verified"
                else "yellow"
                if self.state.get("counterexample_state") not in {None, "none"}
                else "cyan"
            ),
            padding=(0, 1),
            expand=True,
        )

    def _native_counterexample_line(self, width: int) -> str:
        state = self.state.get("counterexample_state")
        if state in {None, "none"}:
            return ""
        candidate = self.state.get("counterexample_candidate", "—")
        order = self.state.get("counterexample_order")
        edges = self.state.get("counterexample_edges")
        minimum_degree = self.state.get("counterexample_minimum_degree")
        lengths = self.state.get("counterexample_lengths")
        length_text = ",".join(str(item) for item in lengths) if isinstance(lengths, list) else "—"
        if state == "verified":
            label = "COUNTEREXAMPLE VERIFIED"
        elif state == "conflict":
            label = "COUNTEREXAMPLE VERIFICATION CONFLICT"
        else:
            label = "COUNTEREXAMPLE CANDIDATE"
        return self._fit(
            f"{label} · {compact_display_ids(candidate)} · order {order or '—'} · "
            f"edges {edges or '—'} · δ {minimum_degree or '—'} · lengths {length_text}",
            width,
        )

    def _native_header(self, width: int) -> str:
        state = self.state
        values: list[str] = []

        def add(label: str, value: object) -> None:
            if value not in (None, "", "-"):
                values.append(f"{label} {value}")

        add("Run", state.get("experiment_id", state.get("run_id")))
        add("session", state.get("session_id"))
        add("state", state.get("state", state.get("stage")))
        generation = state.get("generation")
        generation_limit = state.get("generation_limit")
        if generation is not None:
            add(
                "gen",
                generation if generation_limit is None else f"{generation}/{generation_limit}",
            )
        model = state.get("model")
        effort = state.get("effort")
        if model not in (None, "-"):
            add("model", f"{model}:{effort}" if effort not in (None, "-") else model)
        concurrency = state.get("effective_concurrency")
        if concurrency is not None:
            add("workers", concurrency)
        add("phase", state.get("phase"))
        add("checkpoint", state.get("checkpoint"))
        base_values = list(values)
        optional: list[str] = []
        mode = state.get("run_mode")
        if mode not in (None, "", "-"):
            optional.append(f"mode {mode}")
        elapsed = self._seconds_value(state.get("elapsed_seconds"))
        remaining = self._seconds_value(state.get("remaining_seconds"))
        if elapsed is not None:
            optional.append(f"elapsed {elapsed}")
        if remaining is not None:
            optional.append(f"left {remaining}")
        available = max(1, width - 4)
        while optional and len(" · ".join(base_values + optional)) > available:
            optional.pop()
        values = base_values + optional
        header = " · ".join(values) or "initializing"
        if len(header) <= available:
            return header
        # At narrow terminals add fields in order and drop only the optional
        # identity details that do not fit.  This keeps the stable Run/session
        # identifiers readable for logs and preserves the phase whenever the
        # viewport has enough room.
        compact: list[str] = []
        narrow_values = [
            (
                f"Run …{item[-8:]}"
                if available <= 80 and item.startswith("Run ") and len(item) > 24
                else item
            )
            for item in base_values
        ]
        for item in narrow_values:
            if len(" · ".join([*compact, item])) <= available:
                compact.append(item)
        generation_item = next((item for item in narrow_values if item.startswith("gen ")), None)
        if generation_item is not None and generation_item not in compact:
            compact = [item for item in compact if not item.startswith("state ")]
            if len(" · ".join([*compact, generation_item])) <= available:
                compact.append(generation_item)
        phase_item = next((item for item in narrow_values if item.startswith("phase ")), None)
        if phase_item is not None and phase_item not in compact:
            compact = [
                item for item in compact if not item.startswith(("state ", "model ", "workers "))
            ]
            while len(" · ".join([*compact, phase_item])) > available and len(compact) > 3:
                compact.pop()
            if len(" · ".join([*compact, phase_item])) <= available:
                compact.append(phase_item)
        return self._fit_header(" · ".join(compact), available)

    def _native_progress_lines(self, width: int) -> list[str]:
        """Render bounded progress bars from durable/native counters."""

        state = self.state

        def integer(name: str) -> int | None:
            value = state.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        def pair(current: int | None, total: object) -> tuple[int, int] | None:
            if current is None or not isinstance(total, int) or isinstance(total, bool):
                return None
            return max(0, current), max(0, total)

        generation = pair(integer("generation"), state.get("generation_limit"))
        slots = pair(integer("completed_slots"), state.get("population_size"))
        turns_current = max(
            integer("provider_turns_attempted") or 0,
            integer("provider_turns_completed") or 0,
        )
        turns_total = state.get("max_model_turns")
        turns = pair(turns_current, turns_total)
        episodes = pair(integer("episodes_completed"), state.get("episodes_total"))
        elapsed = self._live_elapsed_seconds()
        wall_total = state.get("configured_wall_seconds")
        wall = (
            (int(elapsed), max(0, int(float(wall_total))))
            if isinstance(wall_total, int | float)
            and not isinstance(wall_total, bool)
            and float(wall_total) > 0
            else None
        )

        def segment(label: str, values: tuple[int, int] | None, *, bar_width: int) -> str:
            if values is None:
                return ""
            current, total = values
            filled = (
                bar_width
                if total <= 0 and current > 0
                else min(bar_width, max(0, round(bar_width * current / max(total, 1))))
            )
            bar = "#" * filled + "-" * (bar_width - filled)
            return f"{label} [{bar}] {current}/{total}"

        available = max(36, width)
        bar_width = 10 if available >= 130 else 7 if available >= 90 else 5
        segments = [
            (
                f"Gen {integer('generation')} · current"
                if integer("generation") is not None and state.get("generation_limit") is None
                else segment("Gen", generation, bar_width=bar_width)
            ),
            segment("Slots", slots, bar_width=bar_width),
            (
                f"Turns {turns_current} cumulative"
                if turns_total is None
                else segment("Turns", turns, bar_width=bar_width)
            ),
            segment("Eval", episodes, bar_width=bar_width),
            segment("Time", wall, bar_width=bar_width),
        ]
        segments = [item for item in segments if item]
        if not segments:
            return []
        if available >= 120:
            return [self._fit(" · ".join(segments), available)]
        # Keep every known bar visible on normal terminals, but split them
        # before Rich can wrap the row unpredictably.
        lines: list[str] = []
        current: list[str] = []
        for item in segments:
            proposed = " · ".join([*current, item])
            if current and len(proposed) > available:
                lines.append(" · ".join(current))
                current = [item]
            else:
                current.append(item)
        if current:
            lines.append(" · ".join(current))
        return [self._fit(line, available) for line in lines]

    def _native_token_line(self, width: int) -> str:
        usage = self.state.get("usage")
        cumulative = usage if isinstance(usage, dict) else self.state.get("_usage_cumulative")
        session = self.state.get("_usage_session")
        cumulative = cumulative if isinstance(cumulative, dict) else {}
        session = session if isinstance(session, dict) else {}
        total = cumulative.get("totalTokens")
        session_total = session.get("totalTokens")
        if not any(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (total, session_total)
        ):
            return ""

        def value(source: dict[str, JsonValue], name: str) -> str:
            current = source.get(name)
            return (
                str(current) if isinstance(current, int) and not isinstance(current, bool) else "?"
            )

        quality = cumulative.get("quality", "unknown")
        text = (
            "tokens "
            f"in {value(cumulative, 'inputTokens')} "
            f"cached {value(cumulative, 'cachedInputTokens')} "
            f"out {value(cumulative, 'outputTokens')} "
            f"reason {value(cumulative, 'reasoningOutputTokens')} "
            f"total {value(cumulative, 'totalTokens')} "
            f"session {value(session, 'totalTokens')} · quality {quality}"
        )
        return self._fit(text, width)

    def _native_heartbeat_line(self, width: int) -> str:
        operation = self._active_operation
        if operation is None:
            return ""
        now = time.monotonic()
        started = operation.get("started")
        elapsed = max(0.0, now - started) if isinstance(started, (int, float)) else 0.0
        timeout = operation.get("timeout")
        timeout_text = ""
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
            timeout_text = f" · timeout in {max(0.0, float(timeout) - elapsed):.1f}s"
        age = self.state.get("last_activity_age_seconds")
        age_text = (
            f" · activity age {float(age):.1f}s"
            if isinstance(age, (int, float)) and not isinstance(age, bool)
            else ""
        )
        warning = ""
        if (
            isinstance(age, (int, float))
            and isinstance(timeout, (int, float))
            and not isinstance(age, bool)
            and not isinstance(timeout, bool)
            and age >= max(5.0, float(timeout) * 0.5)
        ):
            warning = "WARNING stale activity · "
        thread = operation.get("thread") or operation.get("turn")
        thread_text = f" · id {thread}" if isinstance(thread, str) and thread else ""
        text = (
            warning + f"{str(operation.get('phase', 'work')).upper()} "
            f"{compact_display_ids(operation.get('slot', '?'))} · {elapsed:.1f}s elapsed"
            f"{timeout_text}{age_text}{thread_text}"
        )
        return self._fit(text, width)

    def _native_summary_line(self) -> str:
        state = self.state
        values: list[str] = []

        def number(name: str) -> int:
            value = state.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        completed = number("completed_slots")
        total = state.get("population_size", state.get("slot_count"))
        if isinstance(total, int) and not isinstance(total, bool):
            values.append(f"slots {completed}/{total}")
        attempts = number("provider_turns_attempted")
        completed_turns = number("provider_turns_completed")
        if attempts or completed_turns:
            values.append(f"turns {completed_turns}/{attempts}")
        active = number("active_model_turns")
        if active:
            values.append(f"active {active}")
        repairs = number("repair_turns")
        if repairs:
            values.append(f"repairs {repairs}")
        operation = self._active_operation
        if operation is not None:
            slot = operation.get("slot")
            if isinstance(slot, str):
                parent = self._slot_details.get(slot, {}).get("parent_id")
                values.append(
                    f"work {compact_display_ids(slot)}/{self._compact_parent(parent)}"
                )
        accepted = number("accepted_candidates")
        invalid = number("invalid_candidates")
        duplicate = number("duplicate_candidates")
        if accepted or invalid or duplicate:
            values.append(f"candidates {accepted}/{invalid}/{duplicate}")
        evaluations = number("evaluations_completed")
        queued = number("evaluations_queued")
        if evaluations or queued:
            values.append(f"eval {evaluations}/{queued or evaluations}")
        usage = state.get("usage")
        total_tokens = usage.get("totalTokens") if isinstance(usage, dict) else None
        if not isinstance(total_tokens, int):
            total_tokens = state.get("cumulative_tokens")
        if isinstance(total_tokens, int) and total_tokens:
            values.append(f"tokens {total_tokens:,}")
        return " · ".join(values)

    def _native_metrics_line(self, width: int) -> str:
        state = self.state
        values: list[str] = []
        current = state.get("current_objective")
        best = state.get("best_objective")
        if current is not None or best is not None:
            values.append(
                f"objective {current if current is not None else '?'} / "
                f"{best if best is not None else '?'}"
            )
        rate = self._rate_value(state.get("evaluations_per_second"))
        session = state.get("session_id")
        if rate is None:
            completed_evaluations = state.get("evaluations_completed")
            elapsed = self._live_elapsed_seconds()
            if (
                isinstance(completed_evaluations, int)
                and not isinstance(completed_evaluations, bool)
                and elapsed > 0
            ):
                rate = f"{completed_evaluations / elapsed:.2f}"
        if rate is not None or session not in (None, ""):
            values.append(f"eval/s {rate if rate is not None else '0.00'}")
        episode_rate = self._rate_value(state.get("episodes_per_second"))
        if episode_rate is not None:
            values.append(f"eps/s {episode_rate}")
        completed_turns = state.get("provider_turns_completed")
        if (
            isinstance(completed_turns, int)
            and not isinstance(completed_turns, bool)
            and session not in (None, "")
        ):
            elapsed = self._live_elapsed_seconds()
            if elapsed > 0:
                values.append(f"turn/s {completed_turns / elapsed:.2f}")
                values.append(f"turn/min {completed_turns * 60 / elapsed:.1f}")
        lines_rate = self._rate_value(state.get("source_lines_per_second"))
        if lines_rate is None and self._source_lines:
            elapsed = self._live_elapsed_seconds()
            if elapsed > 0:
                lines_rate = f"{self._source_lines / elapsed:.2f}"
        if self._source_lines:
            values.append(
                f"lines {self._source_lines}"
                + (f" · lines/s {lines_rate}" if lines_rate is not None else "")
            )
        ir = state.get("ir", state.get("improvement_rate"))
        if isinstance(ir, int | float) and not isinstance(ir, bool):
            values.append(f"IR {float(ir):.3f}")
        active = state.get("active_model_turns")
        configured = state.get("effective_concurrency", state.get("configured_concurrency"))
        if isinstance(active, int) and isinstance(configured, int) and configured > 0:
            values.append(f"workers {active}/{configured}")
        charged_failed = state.get("charged_failed_turns")
        if isinstance(charged_failed, int) and charged_failed:
            values.append(f"charged-failed {charged_failed}")
        user = state.get("user_seconds")
        system = state.get("system_seconds")
        if isinstance(user, (int, float)) and isinstance(system, (int, float)):
            values.append(f"cpu {float(user):.1f}/{float(system):.1f}s")
        recovered = state.get("recovered_work")
        if isinstance(recovered, int) and recovered:
            values.append(f"recovered {recovered}")
        error = state.get("error_summary")
        if isinstance(error, str) and error:
            values.append(f"ERROR {self._compact_text(error, max(32, width - 12))}")
        return self._fit(" · ".join(values), width)

    def _native_slot_rows(self) -> list[dict[str, JsonValue]]:
        rows = [{**detail, "slot": slot} for slot, detail in self._slot_details.items()]
        if not rows:
            slots = self.state.get("slot_states")
            if isinstance(slots, dict):
                rows = [{"slot": slot, "state": value} for slot, value in slots.items()]
        for row in rows:
            row.setdefault("slot", "?")
            started = row.get("_slot_started_at")
            if isinstance(started, (int, float)) and row.get("state") in {
                "model",
                "repair_running",
                "validating",
                "probing",
                "evaluating",
            }:
                row["elapsed_seconds"] = max(0.0, time.monotonic() - started)
        return sorted(rows, key=lambda row: str(row.get("slot", "")))

    def _native_slot_table(self, width: int, rows: list[dict[str, JsonValue]]) -> Table:
        table = Table(box=None, expand=True, padding=(0, 1), pad_edge=False)
        table.add_column("slot", no_wrap=True, style="cyan")
        if width >= 120:
            table.add_column("parent", no_wrap=True)
            table.add_column("phase", no_wrap=True)
            table.add_column("state", no_wrap=True)
            table.add_column("elapsed", justify="right", no_wrap=True)
            table.add_column("tokens", justify="right", no_wrap=True)
            table.add_column("lines", justify="right", no_wrap=True)
            table.add_column("candidate / error", no_wrap=True, overflow="ellipsis")
        elif width >= 100:
            table.add_column("parent", no_wrap=True)
            table.add_column("state", no_wrap=True)
            table.add_column("elapsed", justify="right", no_wrap=True)
            table.add_column("tokens", justify="right", no_wrap=True)
            table.add_column("result", no_wrap=True, overflow="ellipsis")
        else:
            table.add_column("state", no_wrap=True)
            table.add_column("elapsed", justify="right", no_wrap=True)
            table.add_column("tokens", justify="right", no_wrap=True)
            table.add_column("result", no_wrap=True, overflow="ellipsis")
        for row in rows:
            slot = compact_display_ids(row.get("slot", "?"))
            parent = self._compact_parent(row.get("parent_id"))
            phase = str(row.get("phase", ""))
            state = self._compact_state(row.get("state", "queued"))
            tokens = str(row.get("tokens", ""))
            elapsed_value = row.get("elapsed_seconds")
            elapsed = (
                f"{float(elapsed_value):.1f}s"
                if isinstance(elapsed_value, (int, float)) and not isinstance(elapsed_value, bool)
                else ""
            )
            lines = str(row.get("source_lines", ""))
            result = self._compact_text(
                compact_display_ids(
                    row.get("error", row.get("candidate", row.get("score", "")))
                ),
                96,
            )
            if width >= 120:
                table.add_row(slot, parent, phase, state, elapsed, tokens, lines, result)
            elif width >= 100:
                table.add_row(slot, parent, state, elapsed, tokens, result)
            else:
                table.add_row(slot, state, elapsed, tokens, result)
        return table

    def _native_profile_line(self) -> str | None:
        profile = self.state.get("timing_profile")
        if not isinstance(profile, dict) or profile.get("enabled") is not True:
            if self.state.get("profiling_enabled") is True:
                return "Profile waiting for phase data"
            return None
        phases = profile.get("phase_seconds")
        if not isinstance(phases, dict):
            return None
        top = sorted(
            (
                (name, seconds)
                for name, seconds in phases.items()
                if isinstance(name, str)
                and isinstance(seconds, int | float)
                and not isinstance(seconds, bool)
            ),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:3]
        if not top:
            return None
        calls = profile.get("phase_calls")
        parts = []
        for name, seconds in top:
            call_count = calls.get(name) if isinstance(calls, dict) else None
            suffix = f" x{call_count}" if isinstance(call_count, int) else ""
            parts.append(f"{name} {float(seconds):.2f}s{suffix}")
        unattributed = profile.get("unattributed_fraction")
        if isinstance(unattributed, int | float) and not isinstance(unattributed, bool):
            parts.append(f"unattributed {float(unattributed) * 100:.1f}%")
        throughput = profile.get("throughput", profile.get("evaluations_per_second"))
        if isinstance(throughput, int | float) and not isinstance(throughput, bool):
            parts.append(f"{float(throughput):.2f}/s")
        return "Profile  " + " · ".join(parts)

    @staticmethod
    def _compact_parent(value: JsonValue) -> str:
        if not isinstance(value, str) or not value:
            return "root"
        return compact_display_ids(value)

    @staticmethod
    def _seconds_value(value: JsonValue) -> str | None:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{value:.1f}s"
        return None

    @staticmethod
    def _rate_value(value: JsonValue) -> str | None:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{value:.2f}"
        return None

    @staticmethod
    def _fit(value: str, width: int) -> str:
        value = compact_display_ids(value)
        if len(value) <= width:
            return value
        if width <= 1:
            return value[:width]
        return value[: width - 1] + "…"

    @staticmethod
    def _fit_header(value: str, width: int) -> str:
        value = compact_display_ids(value)
        if len(value) <= width:
            return value
        if width <= 1:
            return value[:width]
        left = max(1, (width - 1) // 2)
        right = max(0, width - left - 1)
        return value[:left] + "…" + value[-right:] if right else value[:left] + "…"

    @staticmethod
    def _compact_text(value: str, limit: int) -> str:
        normalized = " ".join(compact_display_ids(value).split())
        if len(normalized) <= limit:
            return normalized
        if limit <= 1:
            return normalized[:limit]
        return normalized[: limit - 1] + "…"

    @staticmethod
    def _compact_state(value: object) -> str:
        return {
            "probing": "probe",
            "evaluating": "eval",
            "accepted": "accepted",
        }.get(str(value), str(value))

    def _render_legacy(self) -> Group:
        profile_table = self._profile_table()
        deep_profile_table = self._deep_profile_table()
        deep_score_profile_table = self._deep_score_profile_table()
        process_times = tuple(
            self._seconds(self.state.get(key))
            for key in ("real_seconds", "user_seconds", "system_seconds")
        )
        process_time_summary = " / ".join(process_times)
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="cyan")
        overview.add_column()
        rows = [
            ("Run", self.state.get("run_id", "pending")),
            ("Stage", self.state.get("stage", "initializing")),
            ("HEG / backend", self.state.get("heg_commit", self.state.get("backend_id", "-"))),
            ("Dataset / order", self.state.get("split", "-")),
            ("Baseline", self.state.get("baseline", "-")),
            ("Episodes", self.state.get("episodes_completed", 0)),
            ("Evaluations", self.state.get("evaluations", 0)),
            ("Evaluations/s", self._rate(self.state.get("evaluations_per_second"))),
            ("Initial score", self.state.get("initial_total", "-")),
            (
                "Current / best",
                f"{self.state.get('current_total', '-')} / {self.state.get('best_total', '-')}",
            ),
            (
                "Legal / invalid",
                f"{self.state.get('legal_proposals', 0)} / "
                f"{self.state.get('invalid_proposals', 0)}",
            ),
            (
                "Timeouts / crashes",
                f"{self.state.get('timeouts', 0)} / {self.state.get('crashes', 0)}",
            ),
            ("Profile top / unattributed", self._profile_summary()),
            ("Latest event", self.state.get("latest_event", "none")),
        ]
        if self.state.get("native") is True:
            rows.extend(self._native_rows())
        if profile_table is None:
            rows.append(
                (
                    "Time real/user/sys",
                    process_time_summary,
                )
            )
        for label, value in rows:
            overview.add_row(str(label), str(value))
        overview_panel = Panel(
            overview,
            title=(
                "Mutation Forge Lab · Native experiment"
                if self.state.get("native") is True
                else "Mutation Forge Lab · Stage 1"
            ),
        )
        panels: list[Panel] = [overview_panel]
        if profile_table is not None:
            profile = self.state["timing_profile"]
            assert isinstance(profile, dict)
            profiled_episodes = profile.get("profiled_episodes", "-")
            profile_content = Group(profile_table)
            if all(value != "-" for value in process_times):
                profile_content = Group(
                    profile_table,
                    Text.assemble(
                        "\n",
                        ("Run real/user/sys", "cyan"),
                        "  ",
                        process_time_summary,
                    ),
                )
            panels.append(
                Panel(
                    profile_content,
                    title=f"Runtime profile · {profiled_episodes} episodes",
                )
            )
        if deep_profile_table is not None:
            deep_profile = self.state["deep_operator_profile"]
            assert isinstance(deep_profile, dict)
            profiled_episodes = deep_profile.get("profiled_episodes", "-")
            panels.append(
                Panel(
                    deep_profile_table,
                    title=(f"Deep operator profile · {profiled_episodes} episodes"),
                )
            )
        if deep_score_profile_table is not None:
            deep_score_profile = self.state["deep_score_profile"]
            assert isinstance(deep_score_profile, dict)
            profiled_episodes = deep_score_profile.get("profiled_episodes", "-")
            panels.append(
                Panel(
                    deep_score_profile_table,
                    title=(f"Deep score profile · {profiled_episodes} episodes"),
                )
            )
        return Group(*panels)

    def _native_rows(self) -> list[tuple[str, JsonValue]]:
        """Return the stage-independent native operational dashboard rows."""

        state = self.state
        slot_states = state.get("slot_states")
        if isinstance(slot_states, dict):
            slot_summary = (
                ", ".join(
                    f"{compact_display_ids(slot)}={value}"
                    for slot, value in sorted(slot_states.items())
                )
                or "-"
            )
        else:
            slot_summary = "-"
        usage = state.get("usage")
        usage_map = usage if isinstance(usage, dict) else state

        def value(name: str, fallback: JsonValue = "-") -> JsonValue:
            current = state.get(name, fallback)
            return fallback if current is None else current

        def usage_value(name: str) -> JsonValue:
            current = usage_map.get(name, "-") if isinstance(usage_map, dict) else "-"
            return "-" if current is None else current

        return [
            ("Experiment ID", value("experiment_id", state.get("run_id", "-"))),
            ("Workspace", value("workspace")),
            ("Session", value("session_id")),
            ("Mode", value("run_mode", "fresh")),
            ("Experiment state", value("state")),
            ("Elapsed session", self._seconds(value("elapsed_seconds"))),
            ("Invocation budget", self._seconds(value("configured_wall_seconds"))),
            ("Remaining time", self._seconds(value("remaining_seconds"))),
            ("Stop reason", value("stop_reason")),
            ("Latest checkpoint", value("checkpoint")),
            (
                "Generation",
                f"{value('generation', 0)} / {value('generation_limit', value('max_generations'))}",
            ),
            (
                "Slots completed",
                f"{value('completed_slots', 0)} / {value('population_size', value('slot_count'))}",
            ),
            ("Slot states", slot_summary),
            (
                "Parent / root",
                compact_display_ids(value("parent_id", value("parent_status"))),
            ),
            ("Turn phase", value("phase")),
            ("Active model turns", value("active_model_turns", 0)),
            (
                "Concurrency",
                f"{value('effective_concurrency')} / {value('configured_concurrency')}",
            ),
            (
                "Provider turns",
                f"{value('provider_turns_completed', 0)} / {value('provider_turns_attempted', 0)}",
            ),
            (
                "Cumulative provider turns / evaluations",
                f"{value('cumulative_provider_turns', 0)} / {value('cumulative_evaluations', 0)}",
            ),
            ("Repair turns", value("repair_turns", 0)),
            ("Remaining max_model_turns", value("remaining_model_turns")),
            ("Model", f"{value('model')}:{value('effort')}"),
            ("Input tokens", usage_value("inputTokens")),
            ("Cached input tokens", usage_value("cachedInputTokens")),
            ("Output tokens", usage_value("outputTokens")),
            ("Reasoning tokens", usage_value("reasoningOutputTokens")),
            ("Total tokens", usage_value("totalTokens")),
            ("Session tokens", value("token_usage_delta")),
            ("Cumulative tokens", value("cumulative_tokens")),
            ("Usage quality", usage_value("quality")),
            ("Charged failed turns", value("charged_failed_turns", 0)),
            ("Responses received", value("responses_received", 0)),
            ("JSON/schema outcomes", value("parse_outcomes", value("schema_outcomes", 0))),
            ("AST validation outcomes", value("validation_outcomes", 0)),
            ("Behavior probes", value("behavior_probe_outcomes", 0)),
            ("Candidates accepted", value("accepted_candidates", 0)),
            ("Candidates invalid", value("invalid_candidates", 0)),
            ("Candidates duplicate", value("duplicate_candidates", 0)),
            ("Archive size", value("archive_size", 0)),
            ("Selected parents / elite", value("selected_parents")),
            ("Evaluations queued", value("evaluations_queued", 0)),
            ("Evaluations active", value("evaluations_active", 0)),
            ("Evaluations completed", value("evaluations_completed", 0)),
            (
                "Evaluation coordinates",
                f"order={value('order')} graph={value('graph_seed')} policy={value('policy_seed')}",
            ),
            (
                "Development / replay",
                f"{value('development_progress')} / {value('replay_progress')}",
            ),
            ("Evaluation count", value("evaluation_count", value("evaluations", 0))),
            ("Evaluation throughput", self._rate(value("evaluations_per_second"))),
            (
                "Current / best objective",
                f"{value('current_objective')} / {value('best_objective')}",
            ),
            ("Baseline comparison", value("baseline_comparison")),
            (
                "Best candidate",
                f"{compact_display_ids(value('best_candidate_id'))} · "
                f"{value('best_score')}",
            ),
            ("Workers", f"{value('active_workers')} / {value('worker_count')}"),
            ("Provider / validation error", value("error_summary")),
            ("Recovered work", value("recovered_work")),
        ]

    @staticmethod
    def _seconds(value: JsonValue) -> str:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{value:.3f}"
        return "-"

    @staticmethod
    def _rate(value: JsonValue) -> str:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{value:.2f}"
        return "-"

    def _profile_summary(self) -> str:
        profile = self.state.get("timing_profile")
        if not isinstance(profile, dict):
            return "-"
        if profile.get("enabled") is False:
            return "disabled"
        phase = profile.get("dominant_phase")
        seconds = profile.get("dominant_seconds")
        unattributed = profile.get("unattributed_fraction")
        if (
            not isinstance(phase, str)
            or not isinstance(seconds, int | float)
            or isinstance(seconds, bool)
            or not isinstance(unattributed, int | float)
            or isinstance(unattributed, bool)
        ):
            return "-"
        return f"{phase} {seconds:.3f}s / {unattributed * 100:.1f}%"

    def _profile_table(self) -> Table | None:
        profile = self.state.get("timing_profile")
        if not isinstance(profile, dict) or profile.get("enabled") is not True:
            return None
        phases = profile.get("phase_seconds")
        measured = profile.get("measured_total_seconds")
        if (
            not isinstance(phases, dict)
            or not isinstance(measured, int | float)
            or isinstance(measured, bool)
            or measured <= 0
        ):
            return None

        numeric_phases = [
            (phase, seconds)
            for phase, seconds in phases.items()
            if isinstance(phase, str)
            and isinstance(seconds, int | float)
            and not isinstance(seconds, bool)
        ]
        if not numeric_phases:
            return None

        children_by_phase: dict[str, list[tuple[str, float]]] = {}
        raw_children = profile.get("phase_children_seconds")
        if isinstance(raw_children, dict):
            for parent, raw_phase_children in raw_children.items():
                if not isinstance(parent, str) or not isinstance(raw_phase_children, dict):
                    continue
                children_by_phase[parent] = [
                    (child, float(seconds))
                    for child, seconds in raw_phase_children.items()
                    if isinstance(child, str)
                    and isinstance(seconds, int | float)
                    and not isinstance(seconds, bool)
                ]

        calls_by_phase: dict[str, int] = {}
        raw_phase_calls = profile.get("phase_calls")
        if isinstance(raw_phase_calls, dict):
            calls_by_phase = {
                phase: calls
                for phase, calls in raw_phase_calls.items()
                if isinstance(phase, str) and isinstance(calls, int) and not isinstance(calls, bool)
            }

        child_calls_by_phase: dict[str, dict[str, int | None]] = {}
        raw_child_calls = profile.get("phase_children_calls")
        if isinstance(raw_child_calls, dict):
            for parent, raw_calls in raw_child_calls.items():
                if not isinstance(parent, str) or not isinstance(raw_calls, dict):
                    continue
                child_calls_by_phase[parent] = {
                    child: calls
                    for child, calls in raw_calls.items()
                    if isinstance(child, str)
                    and (calls is None or (isinstance(calls, int) and not isinstance(calls, bool)))
                }

        grandchildren_by_phase: dict[tuple[str, str], list[tuple[str, float]]] = {}
        raw_grandchildren = profile.get("phase_grandchildren_seconds")
        if isinstance(raw_grandchildren, dict):
            for phase, raw_phase_children in raw_grandchildren.items():
                if not isinstance(phase, str) or not isinstance(raw_phase_children, dict):
                    continue
                for child, raw_grandchildren_by_child in raw_phase_children.items():
                    if not isinstance(child, str) or not isinstance(
                        raw_grandchildren_by_child, dict
                    ):
                        continue
                    grandchildren_by_phase[(phase, child)] = [
                        (grandchild, float(seconds))
                        for grandchild, seconds in raw_grandchildren_by_child.items()
                        if isinstance(grandchild, str)
                        and isinstance(seconds, int | float)
                        and not isinstance(seconds, bool)
                    ]

        grandchild_calls_by_phase: dict[tuple[str, str], dict[str, int | None]] = {}
        raw_grandchild_calls = profile.get("phase_grandchildren_calls")
        if isinstance(raw_grandchild_calls, dict):
            for phase, raw_phase_children in raw_grandchild_calls.items():
                if not isinstance(phase, str) or not isinstance(raw_phase_children, dict):
                    continue
                for child, raw_calls in raw_phase_children.items():
                    if not isinstance(child, str) or not isinstance(raw_calls, dict):
                        continue
                    grandchild_calls_by_phase[(phase, child)] = {
                        grandchild: calls
                        for grandchild, calls in raw_calls.items()
                        if isinstance(grandchild, str)
                        and (
                            calls is None
                            or (isinstance(calls, int) and not isinstance(calls, bool))
                        )
                    }

        table = Table(box=box.MINIMAL, border_style="grey37", padding=(0, 1))
        table.add_column("Phase", style="cyan", min_width=28)
        table.add_column("Calls", justify="right", no_wrap=True)
        table.add_column(Text("Wall [s]"), justify="right", no_wrap=True)
        table.add_column("Of parent", justify="right", no_wrap=True)
        table.add_column("Of episode", justify="right", no_wrap=True)
        for phase, seconds in sorted(numeric_phases, key=lambda item: item[1], reverse=True):
            phase_children = children_by_phase.get(phase, [])
            table.add_row(
                phase,
                f"{calls_by_phase[phase]:,}" if phase in calls_by_phase else "",
                f"{seconds:.3f}",
                "100.0%" if phase_children else "",
                f"{seconds / measured * 100:.1f}%",
            )
            for index, (child, child_seconds) in enumerate(phase_children):
                child_is_last = index == len(phase_children) - 1
                connector = "└─" if child_is_last else "├─"
                child_calls = child_calls_by_phase.get(phase, {})
                child_call_text = ""
                if child in child_calls:
                    call_count = child_calls[child]
                    child_call_text = "—" if call_count is None else f"{call_count:,}"
                table.add_row(
                    f"  {connector} {child}",
                    child_call_text,
                    f"{child_seconds:.3f}",
                    f"{child_seconds / max(seconds, 1e-9) * 100:.1f}%",
                    f"{child_seconds / measured * 100:.1f}%",
                )
                grandchildren = grandchildren_by_phase.get((phase, child), [])
                for grandchild_index, (
                    grandchild,
                    grandchild_seconds,
                ) in enumerate(grandchildren):
                    grandchild_connector = (
                        "└─" if grandchild_index == len(grandchildren) - 1 else "├─"
                    )
                    parent_branch = "   " if child_is_last else "│  "
                    grandchild_calls = grandchild_calls_by_phase.get((phase, child), {})
                    grandchild_call_text = ""
                    if grandchild in grandchild_calls:
                        call_count = grandchild_calls[grandchild]
                        grandchild_call_text = "—" if call_count is None else f"{call_count:,}"
                    table.add_row(
                        f"  {parent_branch}{grandchild_connector} {grandchild}",
                        grandchild_call_text,
                        f"{grandchild_seconds:.3f}",
                        (f"{grandchild_seconds / max(child_seconds, 1e-9) * 100:.1f}%"),
                        f"{grandchild_seconds / measured * 100:.1f}%",
                    )

        table.add_section()
        for label, key in (
            ("phases subtotal", "accounted_seconds"),
            ("other in episodes", "unattributed_seconds"),
        ):
            value = profile.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                table.add_row(
                    label,
                    "",
                    f"{value:.3f}",
                    "",
                    f"{value / measured * 100:.1f}%",
                )
        table.add_section()
        table.add_row(
            "episode wall total",
            "",
            f"{measured:.3f}",
            "",
            "100.0%",
            style="bright_cyan",
        )
        hotspots = profile.get("hotspots")
        if isinstance(hotspots, list) and hotspots:
            table.add_section()
            table.add_row("Top hotspots", "", "", "", "")
            for item in hotspots[:5]:
                if not isinstance(item, dict):
                    continue
                name = item.get("phase", item.get("name", "-"))
                seconds_value = item.get("seconds", "-")
                percent_value = item.get("percent", "-")
                seconds_text = (
                    f"{seconds_value:.3f}"
                    if isinstance(seconds_value, int | float)
                    else str(seconds_value)
                )
                percent_text = (
                    f"{percent_value:.1f}%"
                    if isinstance(percent_value, int | float)
                    else str(percent_value)
                )
                table.add_row(f"  {name}", "", seconds_text, "", percent_text)
        return table

    def _deep_profile_table(self) -> Table | None:
        profile = self.state.get("deep_operator_profile")
        if not isinstance(profile, dict) or profile.get("enabled") is not True:
            return None
        operators = profile.get("operators")
        if not isinstance(operators, dict) or not operators:
            return None

        table = Table(box=box.MINIMAL, border_style="grey37", padding=(0, 1))
        table.add_column("Operator / phase", style="cyan", min_width=32)
        table.add_column("Calls", justify="right", no_wrap=True)
        table.add_column(Text("Wall [s]"), justify="right", no_wrap=True)
        table.add_column("Of parent", justify="right", no_wrap=True)
        table.add_column("Of operator", justify="right", no_wrap=True)
        table.add_column("Details", no_wrap=True)

        def add_node(
            name: str,
            node: dict[str, JsonValue],
            *,
            parent_seconds: float,
            operator_seconds: float,
            prefix: str,
            connector: str,
        ) -> None:
            seconds = node.get("seconds")
            if not isinstance(seconds, int | float) or isinstance(seconds, bool):
                return
            calls = node.get("calls")
            call_text = (
                f"{calls:,}" if isinstance(calls, int) and not isinstance(calls, bool) else "—"
            )
            details = ""
            counters = node.get("counters")
            if isinstance(counters, dict):
                timing_scope = counters.get("timing_scope")
                if isinstance(timing_scope, str):
                    details = timing_scope
                lookups = counters.get("witness_cache_lookups")
                hits = counters.get("witness_cache_hits")
                misses = counters.get("witness_cache_misses")
                if all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in (lookups, hits, misses)
                ) and any((lookups, hits, misses)):
                    details = f"cache h/m/l {hits:,}/{misses:,}/{lookups:,}"
            label = f"{prefix}{connector} {name}" if connector else name
            table.add_row(
                label,
                call_text,
                f"{seconds:.3f}",
                f"{seconds / max(parent_seconds, 1e-9) * 100:.1f}%",
                f"{seconds / max(operator_seconds, 1e-9) * 100:.1f}%",
                details,
            )
            children = node.get("children")
            if not isinstance(children, dict):
                return
            child_items = [
                (child_name, child_node)
                for child_name, child_node in children.items()
                if isinstance(child_name, str) and isinstance(child_node, dict)
            ]
            child_prefix = prefix
            if connector:
                child_prefix += "   " if connector == "└─" else "│  "
            else:
                child_prefix += "  "
            for index, (child_name, child_node) in enumerate(child_items):
                add_node(
                    child_name,
                    child_node,
                    parent_seconds=float(seconds),
                    operator_seconds=operator_seconds,
                    prefix=child_prefix,
                    connector=("└─" if index == len(child_items) - 1 else "├─"),
                )

        operator_items = [
            (operator, node)
            for operator, node in operators.items()
            if isinstance(operator, str) and isinstance(node, dict)
        ]
        for index, (operator, node) in enumerate(operator_items):
            seconds = node.get("seconds")
            if not isinstance(seconds, int | float) or isinstance(seconds, bool):
                continue
            if index:
                table.add_section()
            add_node(
                operator,
                node,
                parent_seconds=float(seconds),
                operator_seconds=float(seconds),
                prefix="",
                connector="",
            )
        return table

    def _deep_score_profile_table(self) -> Table | None:
        profile = self.state.get("deep_score_profile")
        if not isinstance(profile, dict) or profile.get("enabled") is not True:
            return None
        worker = profile.get("worker")
        prepared = profile.get("prepared_graph")
        counters = profile.get("counters")
        assembly = profile.get("score_assembly")
        if not all(isinstance(value, dict) for value in (worker, prepared, counters, assembly)):
            return None
        assert isinstance(worker, dict)
        assert isinstance(prepared, dict)
        assert isinstance(counters, dict)
        assert isinstance(assembly, dict)

        table = Table(box=box.MINIMAL, border_style="grey37", padding=(0, 1))
        table.add_column("Scoring phase", style="cyan", min_width=28)
        table.add_column("Calls", justify="right", no_wrap=True)
        table.add_column(Text("Wall [s]"), justify="right", no_wrap=True)
        table.add_column("Of parent", justify="right", no_wrap=True)
        table.add_column("Details", no_wrap=True)

        def numeric(node: dict[str, JsonValue], key: str) -> float:
            value = node.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
            return 0.0

        worker_seconds = numeric(worker, "seconds")
        worker_calls = worker.get("calls")
        protocol_overhead = numeric(worker, "protocol_overhead_seconds")
        table.add_row(
            "worker_roundtrip",
            (
                f"{worker_calls:,}"
                if isinstance(worker_calls, int) and not isinstance(worker_calls, bool)
                else "—"
            ),
            f"{worker_seconds:.3f}",
            "100.0%",
            f"protocol overhead {protocol_overhead:.3f}s",
        )
        children = worker.get("children")
        if isinstance(children, dict):
            child_items = [
                (name, node)
                for name, node in children.items()
                if isinstance(name, str) and isinstance(node, dict)
            ]
            for index, (name, node) in enumerate(child_items):
                seconds = numeric(node, "seconds")
                calls = node.get("calls")
                table.add_row(
                    f"  {'└─' if index == len(child_items) - 1 else '├─'} {name}",
                    (
                        f"{calls:,}"
                        if isinstance(calls, int) and not isinstance(calls, bool) and calls
                        else "—"
                    ),
                    f"{seconds:.3f}",
                    f"{seconds / max(worker_seconds, 1e-9) * 100:.1f}%",
                    "",
                )

        table.add_section()
        for label, key in (
            ("graph_materialization", "materialization"),
            ("validation", "validation"),
        ):
            prepared_node = prepared.get(key)
            if not isinstance(prepared_node, dict):
                continue
            seconds = numeric(prepared_node, "seconds")
            calls = prepared_node.get("calls")
            table.add_row(
                label,
                (f"{calls:,}" if isinstance(calls, int) and not isinstance(calls, bool) else "—"),
                f"{seconds:.3f}",
                "",
                "prepared graph work",
            )
        assembly_seconds = numeric(assembly, "seconds")
        assembly_calls = assembly.get("calls")
        table.add_row(
            "score_assembly",
            (
                f"{assembly_calls:,}"
                if isinstance(assembly_calls, int) and not isinstance(assembly_calls, bool)
                else "—"
            ),
            f"{assembly_seconds:.3f}",
            "",
            "",
        )

        table.add_section()
        cache_detail = "{}/{}/{}".format(
            counters.get("score_cache_hits", 0),
            counters.get("score_cache_misses", 0),
            counters.get("score_cache_lookups", 0),
        )
        table.add_row(
            "score results",
            "",
            "",
            "",
            (
                f"full {counters.get('score_result_full_results', 0)} · "
                f"dominated {counters.get('score_result_dominated_results', 0)} · "
                f"failures {counters.get('score_result_failures', 0)}"
            ),
        )
        table.add_row(
            "score cache",
            "",
            "",
            "",
            f"hits / misses / lookups  {cache_detail}",
        )
        table.add_row(
            "worker recovery",
            "",
            "",
            "",
            (
                f"failures {counters.get('worker_failure_calls', 0)} · "
                f"restarts {counters.get('worker_restart_successes', 0)}"
            ),
        )
        return table

    def close(self) -> None:
        self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=1.0)
        self.live.stop()


class ProgressLineSink:
    """Flushed, readable event lines for redirected human output."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def write(self, event: Event) -> None:
        payload = event.payload
        details = []
        for key in (
            "generation",
            "slot",
            "phase",
            "status",
            "completed_slots",
            "evaluations_completed",
            "evaluations_per_second",
            "episodes_per_second",
            "turns_per_minute",
            "source_lines",
            "source_lines_per_second",
            "ir",
            "operation_elapsed_seconds",
            "timeout_seconds",
            "last_activity_age_seconds",
            "inputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
            "best_score",
            "stop_reason",
            "error",
        ):
            value = payload.get(key)
            if value not in (None, ""):
                details.append(f"{key}={value}")
        suffix = f" ({', '.join(details)})" if details else ""
        line = f"[{event.timestamp}] {event.event_type}{suffix}\n"
        self.stream.write(line)
        self.stream.flush()

    def close(self) -> None:
        return None


NonTTYProgressSink = ProgressLineSink
