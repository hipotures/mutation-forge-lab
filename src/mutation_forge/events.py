from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TextIO

from mutation_forge.models import JsonValue

EVENT_SCHEMA_VERSION = "mforge.experiment.events.v2"
EVENT_TYPES = frozenset(
    {
        "run_started",
        "backend_ready",
        "dataset_loaded",
        "baseline_started",
        "episode_started",
        "episode_progress",
        "episode_completed",
        "program_generation_started",
        "program_generated",
        "static_validation_completed",
        "sandbox_validation_completed",
        "program_evaluation_started",
        "program_evaluation_completed",
        "champion_changed",
        "checkpoint_written",
        "run_completed",
        "run_failed",
        # Stage-independent native experiment lifecycle.  These names are
        # deliberately kept in the shared registry so the same Event object
        # can drive Rich, JSONL, and durable session observers.
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
        "model_token_charge_recorded",
        "hourly_token_usage_updated",
        "hourly_token_limit_reached",
        "hourly_token_session_stopped",
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
)


@dataclass(frozen=True, slots=True)
class Event:
    schema_version: str
    timestamp: str
    run_id: str
    event_type: str
    payload: dict[str, JsonValue]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "event_type": self.event_type,
            **self.payload,
        }


class EventSink(Protocol):
    def write(self, event: Event) -> None: ...

    def close(self) -> None: ...


class EventBus:
    def __init__(
        self,
        run_id: str,
        sinks: list[EventSink],
        *,
        schema_version: str = EVENT_SCHEMA_VERSION,
    ) -> None:
        self.run_id = run_id
        self.sinks = sinks
        self.schema_version = schema_version

    def emit(self, event_type: str, **payload: JsonValue) -> Event:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")
        event = Event(
            schema_version=self.schema_version,
            timestamp=datetime.now(UTC).isoformat(),
            run_id=self.run_id,
            event_type=event_type,
            payload=payload,
        )
        for sink in self.sinks:
            sink.write(event)
        return event

    def close(self) -> None:
        for sink in reversed(self.sinks):
            sink.close()


class JsonlSink:
    def __init__(self, stream: TextIO, *, close_stream: bool = False) -> None:
        self.stream = stream
        self.close_stream = close_stream

    def write(self, event: Event) -> None:
        self.stream.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
        self.stream.flush()

    def close(self) -> None:
        if self.close_stream:
            self.stream.close()
