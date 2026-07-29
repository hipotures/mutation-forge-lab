from __future__ import annotations

import io
from unittest.mock import patch

from rich import box
from rich.console import Console

from mutation_forge.events import Event
from mutation_forge.output.rich_live import RichLiveSink


def test_rich_live_formats_evaluations_per_second() -> None:
    assert RichLiveSink._rate(1467.452267324598) == "1467.45"


def test_rich_live_places_timing_on_last_row() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        output = io.StringIO()
        Console(file=output, force_terminal=False, width=160).print(sink._render())
        rendered = output.getvalue()
        assert rendered.index("Profile top / unattributed") < rendered.index(
            "Latest event"
        )
        assert rendered.index("Latest event") < rendered.index("Time real/user/sys")
    finally:
        sink.close()


def test_rich_live_formats_profile_summary() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        sink.state["timing_profile"] = {
            "enabled": True,
            "dominant_phase": "proposal_generation",
            "dominant_seconds": 12.34567,
            "unattributed_fraction": 0.042,
        }
        assert sink._profile_summary() == "proposal_generation 12.346s / 4.2%"
        sink.state["timing_profile"] = {"enabled": False}
        assert sink._profile_summary() == "disabled"
    finally:
        sink.close()


def test_rich_live_renders_full_runtime_profile_table() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        sink.state["timing_profile"] = {
            "enabled": True,
            "profiled_episodes": 2,
            "phase_seconds": {
                "scoring": 2.0,
                "proposal_generation": 5.0,
                "exact_verification": 0.0,
            },
            "phase_children_seconds": {
                "proposal_generation": {
                    "rng_setup": 0.5,
                    "graph_materialization": 1.0,
                    "operator_search": 3.0,
                    "proposal_packaging": 0.4,
                    "other": 0.1,
                }
            },
            "phase_calls": {"proposal_generation": 20},
            "phase_children_calls": {
                "proposal_generation": {
                    "rng_setup": 20,
                    "graph_materialization": 20,
                    "operator_search": 20,
                    "proposal_packaging": 20,
                    "other": None,
                }
            },
            "phase_grandchildren_seconds": {
                "proposal_generation": {
                    "operator_search": {
                        "heg_uniform_two_switch": 1.0,
                        "heg_forbidden_cycle_break": 2.0,
                    }
                }
            },
            "phase_grandchildren_calls": {
                "proposal_generation": {
                    "operator_search": {
                        "heg_uniform_two_switch": 8,
                        "heg_forbidden_cycle_break": 12,
                    }
                }
            },
            "measured_total_seconds": 8.0,
            "accounted_seconds": 7.0,
            "unattributed_seconds": 1.0,
            "unattributed_fraction": 0.125,
            "dominant_phase": "proposal_generation",
            "dominant_seconds": 5.0,
        }
        sink.state["real_seconds"] = 8.5
        sink.state["user_seconds"] = 7.25
        sink.state["system_seconds"] = 0.5
        output = io.StringIO()
        Console(file=output, force_terminal=False, width=80).print(sink._render())
        rendered = output.getvalue()
        assert "Runtime profile · 2 episodes" in rendered
        assert rendered.index("proposal_generation") < rendered.index("scoring")
        assert "Calls" in rendered
        assert "Wall [s]" in rendered
        assert "Of parent" in rendered
        assert "Of episode" in rendered
        assert "├─ rng_setup" in rendered
        assert "└─ other" in rendered
        operator_line = next(
            line for line in rendered.splitlines() if "operator_search" in line
        )
        assert "60.0%" in operator_line
        assert "37.5%" in operator_line
        assert "20" in operator_line
        wide_output = io.StringIO()
        profile_table = sink._profile_table()
        assert profile_table is not None
        Console(file=wide_output, force_terminal=False, width=160).print(profile_table)
        wide_rendered = wide_output.getvalue()
        uniform_line = next(
            line
            for line in wide_rendered.splitlines()
            if "heg_uniform_two_switch" in line
        )
        forbidden_line = next(
            line
            for line in wide_rendered.splitlines()
            if "heg_forbidden_cycle_break" in line
        )
        assert "│  ├─ heg_uniform_two_switch" in uniform_line
        assert "8" in uniform_line
        assert "33.3%" in uniform_line
        assert "12.5%" in uniform_line
        assert "│  └─ heg_forbidden_cycle_break" in forbidden_line
        assert "12" in forbidden_line
        assert "66.7%" in forbidden_line
        assert "25.0%" in forbidden_line
        other_line = next(
            line for line in rendered.splitlines() if "└─ other" in line
        )
        assert "—" in other_line
        assert "phases subtotal" in rendered
        assert "other in episodes" in rendered
        assert "episode wall total" in rendered
        assert "62.5%" in rendered
        assert rendered.index("episode wall total") < rendered.index(
            "Run real/user/sys"
        )
        assert "8.500 / 7.250 / 0.500" in rendered
        assert rendered.count("Run real/user/sys") == 1
        process_line = next(
            line for line in rendered.splitlines() if "Run real/user/sys" in line
        )
        assert process_line.count("│") == 2
        assert "│" in rendered
        assert "┼" in rendered
        assert profile_table.box is box.MINIMAL
        assert profile_table.border_style == "grey37"
        assert profile_table.rows[-1].style == "bright_cyan"

        profile = sink.state["timing_profile"]
        assert isinstance(profile, dict)
        del profile["phase_grandchildren_seconds"]
        del profile["phase_grandchildren_calls"]
        assert sink._profile_table() is not None
    finally:
        sink.close()


def test_rich_live_renders_at_most_once_per_second_and_on_terminal_event() -> None:
    with patch("mutation_forge.output.rich_live.time.monotonic") as monotonic:
        monotonic.return_value = 0.0
        sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
        try:
            with (
                patch.object(sink.live, "update") as update,
                patch.object(sink, "_render", wraps=sink._render) as render,
            ):
                monotonic.return_value = 0.2
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:00+00:00",
                        run_id="run-1",
                        event_type="episode_progress",
                        payload={"evaluations": 50},
                    )
                )
                monotonic.return_value = 1.0
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:01+00:00",
                        run_id="run-1",
                        event_type="episode_progress",
                        payload={"evaluations": 100},
                    )
                )
                monotonic.return_value = 1.2
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:01.2+00:00",
                        run_id="run-1",
                        event_type="episode_progress",
                        payload={"evaluations": 150},
                    )
                )
                monotonic.return_value = 1.3
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:01+00:00",
                        run_id="run-1",
                        event_type="run_completed",
                        payload={
                            "real_seconds": 1.0,
                            "user_seconds": 0.8,
                            "system_seconds": 0.1,
                        },
                    )
                )
            assert update.call_count == 2
            assert render.call_count == 2
            assert all(call.kwargs == {"refresh": True} for call in update.call_args_list)
        finally:
            sink.close()
