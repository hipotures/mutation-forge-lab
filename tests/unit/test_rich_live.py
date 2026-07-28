from __future__ import annotations

import io
from unittest.mock import patch

from rich.console import Console

from mutation_forge.events import Event
from mutation_forge.output.rich_live import RichLiveSink


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
