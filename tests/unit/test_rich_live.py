from __future__ import annotations

import io
from unittest.mock import patch

from rich.console import Console

from mutation_forge.events import Event
from mutation_forge.output.rich_live import RichLiveSink


def test_rich_live_throttles_progress_but_refreshes_terminal_event() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        with patch.object(sink.live, "update") as update:
            sink.write(
                Event(
                    schema_version="1.0",
                    timestamp="2026-07-28T00:00:00+00:00",
                    run_id="run-1",
                    event_type="episode_progress",
                    payload={"evaluations": 50},
                )
            )
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
        assert update.call_args_list[0].kwargs == {"refresh": False}
        assert update.call_args_list[1].kwargs == {"refresh": True}
    finally:
        sink.close()
