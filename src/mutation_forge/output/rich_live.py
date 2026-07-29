from __future__ import annotations

import time

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
        }
        self.live = Live(
            self._render(),
            console=self.console,
            auto_refresh=False,
            transient=False,
        )
        self.live.start()
        self._last_refresh = time.monotonic()

    def write(self, event: Event) -> None:
        self.state.update(event.payload)
        self.state["latest_event"] = event.event_type
        self.state["run_id"] = event.run_id
        if event.event_type == "baseline_started":
            self.state["stage"] = "baseline"
        elif event.event_type == "run_completed":
            self.state["stage"] = "completed"
        elif event.event_type == "run_failed":
            self.state["stage"] = "failed"
        now = time.monotonic()
        terminal = event.event_type in {"run_completed", "run_failed"}
        if terminal or now - self._last_refresh >= REFRESH_INTERVAL_SECONDS:
            self.live.update(self._render(), refresh=True)
            self._last_refresh = now

    def _render(self) -> Group:
        profile_table = self._profile_table()
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
        if profile_table is None:
            rows.append(
                (
                    "Time real/user/sys",
                    process_time_summary,
                )
            )
        for label, value in rows:
            overview.add_row(str(label), str(value))
        overview_panel = Panel(overview, title="Mutation Forge Lab · Stage 1")
        if profile_table is None:
            return Group(overview_panel)
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
        return Group(
            overview_panel,
            Panel(
                profile_content,
                title=f"Runtime profile · {profiled_episodes} episodes",
            ),
        )

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
        return table

    def close(self) -> None:
        self.live.stop()
