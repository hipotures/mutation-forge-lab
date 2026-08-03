from __future__ import annotations

import io
import json

import pytest

from mutation_forge.events import EventBus, JsonlSink


def test_jsonl_event_has_required_envelope_and_no_ansi() -> None:
    stream = io.StringIO()
    bus = EventBus("run-1", [JsonlSink(stream)])
    bus.emit("run_started", stage="stage1")
    bus.close()
    line = stream.getvalue()
    event = json.loads(line)
    assert event["schema_version"] == "mforge.experiment.events.v2"
    assert event["run_id"] == "run-1"
    assert event["event_type"] == "run_started"
    assert event["stage"] == "stage1"
    assert "\x1b" not in line
    assert line.count("\n") == 1


def test_unknown_event_is_rejected() -> None:
    bus = EventBus("run-1", [])
    with pytest.raises(ValueError, match="unknown event"):
        bus.emit("not_registered")
