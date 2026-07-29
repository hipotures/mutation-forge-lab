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
