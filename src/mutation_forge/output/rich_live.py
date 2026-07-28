from __future__ import annotations

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from mutation_forge.events import Event
from mutation_forge.models import JsonValue


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
            refresh_per_second=4,
            transient=False,
        )
        self.live.start()

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
        refresh = event.event_type in {"run_completed", "run_failed"}
        self.live.update(self._render(), refresh=refresh)

    def _render(self) -> Group:
        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="cyan")
        overview.add_column()
        rows = (
            ("Run", self.state.get("run_id", "pending")),
            ("Stage", self.state.get("stage", "initializing")),
            ("HEG / backend", self.state.get("heg_commit", self.state.get("backend_id", "-"))),
            ("Dataset / order", self.state.get("split", "-")),
            ("Baseline", self.state.get("baseline", "-")),
            ("Episodes", self.state.get("episodes_completed", 0)),
            ("Evaluations", self.state.get("evaluations", 0)),
            ("Evaluations/s", self.state.get("evaluations_per_second", 0)),
            (
                "Time real/user/sys",
                f"{self._seconds(self.state.get('real_seconds'))} / "
                f"{self._seconds(self.state.get('user_seconds'))} / "
                f"{self._seconds(self.state.get('system_seconds'))}",
            ),
            ("Initial score", self.state.get("initial_total", "-")),
            ("Current / best", f"{self.state.get('current_total', '-')} / "
             f"{self.state.get('best_total', '-')}"),
            ("Legal / invalid", f"{self.state.get('legal_proposals', 0)} / "
             f"{self.state.get('invalid_proposals', 0)}"),
            ("Timeouts / crashes", f"{self.state.get('timeouts', 0)} / "
             f"{self.state.get('crashes', 0)}"),
            ("Latest event", self.state.get("latest_event", "none")),
        )
        for label, value in rows:
            overview.add_row(str(label), str(value))
        return Group(Panel(overview, title="Mutation Forge Lab · Stage 1"))

    @staticmethod
    def _seconds(value: JsonValue) -> str:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{value:.3f}"
        return "-"

    def close(self) -> None:
        self.live.stop()
