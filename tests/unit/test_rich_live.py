from __future__ import annotations

import io
from unittest.mock import patch

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
        Console(file=output, force_terminal=False, width=120).print(sink._render())
        rendered = output.getvalue()
        assert "Runtime profile · 2 episodes" in rendered
        assert rendered.index("proposal_generation") < rendered.index("scoring")
        assert "accounted" in rendered
        assert "unattributed" in rendered
        assert "measured total" in rendered
        assert "62.5%" in rendered
        assert rendered.index("measured total") < rendered.index("Time real/user/sys")
        assert "8.500 / 7.250 / 0.500" in rendered
        assert rendered.count("Time real/user/sys") == 1
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
