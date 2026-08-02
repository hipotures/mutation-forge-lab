from __future__ import annotations

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

REFRESH_INTERVAL_SECONDS = 1.0


class RichLiveSink:
    def __init__(self, *, console: Console | None = None) -> None:
        self.console = console or Console()
        self.state: dict[str, JsonValue] = {
            "stage": "initializing",
            "latest_event": "none",
            "evaluations": 0,
            "episodes_completed": 0,
            "native": False,
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
            "_archive_seen": {},
        }
        self.live = Live(
            self._render(),
            console=self.console,
            auto_refresh=False,
            transient=False,
        )
        self.live.start()
        self._last_refresh = time.monotonic()
        self._native_first_refresh = True

    def write(self, event: Event) -> None:
        self.state.update(event.payload)
        self.state["latest_event"] = event.event_type
        self.state["run_id"] = event.run_id
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
            "provider_turn_completed",
            "provider_turn_failed",
            "repair_started",
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
            "selection_completed",
            "budget_boundary_reached",
            "experiment_completed",
            "experiment_interrupted",
            "experiment_failed",
        }
        if native_event:
            self.state["native"] = True
            self._update_native_counters(event)
        if event.event_type == "session_started":
            self.state["state"] = "running"
        elif event.event_type == "experiment_completed":
            self.state["state"] = "completed"
        elif event.event_type == "experiment_interrupted":
            self.state["state"] = "interrupted"
        elif event.event_type == "experiment_failed":
            self.state["state"] = "failed"
        elif event.event_type == "budget_boundary_reached" and event.payload.get(
            "state"
        ) in {"idle", "budget_exhausted"}:
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
            "experiment_interrupted",
            "experiment_failed",
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

    def _update_native_counters(self, event: Event) -> None:
        """Accumulate counters when an event carries only a local delta."""

        payload = event.payload

        def integer(name: str, default: int = 0) -> int:
            value = self.state.get(name, default)
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else default

        def add(name: str, amount: int = 1) -> None:
            self.state[name] = integer(name) + amount

        if event.event_type == "session_started":
            usage = payload.get("usage")
            if isinstance(usage, dict):
                self.state["_usage_cumulative"] = dict(usage)
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
        elif event.event_type in {"provider_turn_completed", "provider_turn_failed"}:
            self.state["active_model_turns"] = max(0, integer("active_model_turns") - 1)
            if event.event_type == "provider_turn_completed":
                add("provider_turns_completed")
                add("responses_received")
            if event.event_type == "provider_turn_failed" and payload.get("charged") is True:
                add("charged_failed_turns")
            usage = payload.get("usage")
            if isinstance(usage, dict):
                current = self.state.get("_usage_cumulative")
                cumulative = dict(current) if isinstance(current, dict) else {}
                for key in (
                    "inputTokens",
                    "cachedInputTokens",
                    "outputTokens",
                    "reasoningOutputTokens",
                    "totalTokens",
                ):
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        prior = cumulative.get(key, 0)
                        cumulative[key] = (
                            prior + value
                            if isinstance(prior, int) and not isinstance(prior, bool)
                            else value
                        )
                cumulative["quality"] = payload.get(
                    "usage_quality", cumulative.get("quality", "unknown")
                )
                self.state["_usage_cumulative"] = cumulative
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
        elif event.event_type == "evaluation_started":
            queued = payload.get("evaluations_queued")
            if isinstance(queued, int) and not isinstance(queued, bool):
                self.state["evaluations_queued"] = max(integer("evaluations_queued"), queued)
            else:
                add("evaluations_queued")
            self.state["evaluations_active"] = integer("evaluations_active") + 1
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
            ):
                if key in payload:
                    self.state[key] = payload[key]
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
        if event.event_type == "provider_turn_failed":
            self.state["error_summary"] = payload.get("error", "provider turn failed")
        elif event.event_type == "validation_completed" and payload.get("valid") is False:
            self.state["error_summary"] = payload.get("error", "validation failed")
        elif event.event_type == "behavior_probe_completed" and payload.get("valid") is False:
            self.state["error_summary"] = payload.get("error", "behavior probe failed")

    def _render(self) -> Group:
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
            ("Current / best", f"{self.state.get('current_total', '-')} / "
             f"{self.state.get('best_total', '-')}"),
            ("Legal / invalid", f"{self.state.get('legal_proposals', 0)} / "
             f"{self.state.get('invalid_proposals', 0)}"),
            ("Timeouts / crashes", f"{self.state.get('timeouts', 0)} / "
             f"{self.state.get('crashes', 0)}"),
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
                    title=(
                        "Deep operator profile · "
                        f"{profiled_episodes} episodes"
                    ),
                )
            )
        if deep_score_profile_table is not None:
            deep_score_profile = self.state["deep_score_profile"]
            assert isinstance(deep_score_profile, dict)
            profiled_episodes = deep_score_profile.get(
                "profiled_episodes", "-"
            )
            panels.append(
                Panel(
                    deep_score_profile_table,
                    title=(
                        "Deep score profile · "
                        f"{profiled_episodes} episodes"
                    ),
                )
            )
        return Group(*panels)

    def _native_rows(self) -> list[tuple[str, JsonValue]]:
        """Return the stage-independent native operational dashboard rows."""

        state = self.state
        slot_states = state.get("slot_states")
        if isinstance(slot_states, dict):
            slot_summary = ", ".join(
                f"{slot}={value}"
                for slot, value in sorted(slot_states.items())
            ) or "-"
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
            ("Parent / root", value("parent_id", value("parent_status"))),
            ("Turn phase", value("phase")),
            ("Active model turns", value("active_model_turns", 0)),
            (
                "Concurrency",
                f"{value('effective_concurrency')} / {value('configured_concurrency')}",
            ),
            (
                "Provider turns",
                f"{value('provider_turns_completed', 0)} / "
                f"{value('provider_turns_attempted', 0)}",
            ),
            (
                "Cumulative provider turns / evaluations",
                f"{value('cumulative_provider_turns', 0)} / "
                f"{value('cumulative_evaluations', 0)}",
            ),
            ("Repair turns", value("repair_turns", 0)),
            ("Remaining max_model_turns", value("remaining_model_turns")),
            ("Model / effort", f"{value('model')} / {value('effort')}"),
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
            ("Best candidate", f"{value('best_candidate_id')} · {value('best_score')}"),
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
                if not isinstance(parent, str) or not isinstance(
                    raw_phase_children, dict
                ):
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
                if isinstance(phase, str)
                and isinstance(calls, int)
                and not isinstance(calls, bool)
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
                    and (
                        calls is None
                        or (
                            isinstance(calls, int)
                            and not isinstance(calls, bool)
                        )
                    )
                }

        grandchildren_by_phase: dict[tuple[str, str], list[tuple[str, float]]] = {}
        raw_grandchildren = profile.get("phase_grandchildren_seconds")
        if isinstance(raw_grandchildren, dict):
            for phase, raw_phase_children in raw_grandchildren.items():
                if not isinstance(phase, str) or not isinstance(
                    raw_phase_children, dict
                ):
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
        for phase, seconds in sorted(
            numeric_phases, key=lambda item: item[1], reverse=True
        ):
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
            if (
                not isinstance(seconds, int | float)
                or isinstance(seconds, bool)
            ):
                return
            calls = node.get("calls")
            call_text = (
                f"{calls:,}"
                if isinstance(calls, int) and not isinstance(calls, bool)
                else "—"
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
                    connector=(
                        "└─" if index == len(child_items) - 1 else "├─"
                    ),
                )

        operator_items = [
            (operator, node)
            for operator, node in operators.items()
            if isinstance(operator, str) and isinstance(node, dict)
        ]
        for index, (operator, node) in enumerate(operator_items):
            seconds = node.get("seconds")
            if (
                not isinstance(seconds, int | float)
                or isinstance(seconds, bool)
            ):
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
        if not all(
            isinstance(value, dict)
            for value in (worker, prepared, counters, assembly)
        ):
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
                if isinstance(worker_calls, int)
                and not isinstance(worker_calls, bool)
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
                        if isinstance(calls, int)
                        and not isinstance(calls, bool)
                        and calls
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
                (
                    f"{calls:,}"
                    if isinstance(calls, int) and not isinstance(calls, bool)
                    else "—"
                ),
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
                if isinstance(assembly_calls, int)
                and not isinstance(assembly_calls, bool)
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
                f"restarts {counters.get('worker_restart_successes', 0)} · "
                f"fallbacks {counters.get('python_fallback_calls', 0)}"
            ),
        )
        return table

    def close(self) -> None:
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
