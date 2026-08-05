"""Stage-independent experiment event fan-out and runtime profiling.

The native experiment engine emits events at the execution boundary.  This
module keeps the orchestration layer independent from any particular output
format while allowing the same event to drive durable session evidence, Rich,
JSONL, and a flushed non-TTY fallback.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, cast

from mutation_forge.events import Event, EventBus, EventSink
from mutation_forge.models import JsonValue


def _json_safe(value: Any) -> JsonValue:
    """Project arbitrary callback payloads onto the JSON event contract."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return _json_safe(value.as_dict())
        except Exception:
            pass
    return repr(value)


class ExperimentEventObserver(Protocol):
    """Callback shape accepted by native generation and evaluation code."""

    def __call__(self, event_type: str, payload: Mapping[str, JsonValue]) -> None: ...


class CallbackEventSink:
    """Adapt a two-argument callback to the shared EventSink protocol."""

    def __init__(self, callback: Any) -> None:
        self.callback = callback

    def write(self, event: Event) -> None:
        try:
            self.callback(event.event_type, event.payload)
        except TypeError:
            self.callback(event)

    def close(self) -> None:
        return None


_PHASE_STARTS = {
    "preflight_started": "preflight",
    "generation_started": "generation",
    "provider_turn_started": "provider",
    "repair_started": "repair",
    "validation_started": "validation",
    "behavior_probe_started": "behavior_probe",
    "evaluation_started": "evaluation",
    "checkpoint_started": "checkpoint",
    "selection_started": "selection",
}
_PHASE_ENDS = {
    "preflight_completed": "preflight",
    "generation_completed": "generation",
    "provider_turn_completed": "provider",
    "provider_turn_failed": "provider",
    "repair_completed": "repair",
    "validation_completed": "validation",
    "behavior_probe_completed": "behavior_probe",
    "evaluation_completed": "evaluation",
    "evaluation_failed": "evaluation",
    "checkpoint_written": "checkpoint",
    "selection_completed": "selection",
}


class ExperimentEventHub:
    """Fan out one native event stream to output sinks and session evidence.

    ``SessionManager`` is attached only after a session directory exists.  A
    caller may therefore emit preflight/workspace events before session start;
    those events still reach Rich/JSON sinks and later events are additionally
    persisted to ``sessions/<session>/events.jsonl`` and SQLite.
    """

    def __init__(
        self,
        run_id: str,
        sinks: Iterable[EventSink] = (),
        *,
        profiling_enabled: bool = False,
    ) -> None:
        self.run_id = run_id
        self._sinks = list(sinks)
        self._bus = EventBus(
            run_id,
            self._sinks,
            schema_version="mforge.experiment.events.v2",
        )
        self._lock = threading.RLock()
        self._session_manager: Any | None = None
        self._session: Any | None = None
        self._pending_session_events: list[tuple[str, dict[str, JsonValue]]] = []
        self._closed = False
        self._profiling_enabled = profiling_enabled
        self._profile_started = time.monotonic()
        self._phase_started: dict[tuple[str, str], float] = {}
        self._phase_seconds: dict[str, float] = {}
        self._phase_calls: dict[str, int] = {}
        self._last_evaluations = 0
        self._supplied_profile: dict[str, JsonValue] = {}

    @property
    def profiling_enabled(self) -> bool:
        return self._profiling_enabled

    def attach_session(self, manager: Any, session: Any) -> None:
        """Attach the durable session sink after session creation."""

        with self._lock:
            self._session_manager = manager
            self._session = session
            pending = tuple(self._pending_session_events)
            self._pending_session_events.clear()
            for event_type, payload in pending:
                manager.event(session, event_type, **payload)

    def detach_session(self) -> None:
        with self._lock:
            self._session_manager = None
            self._session = None

    def _profile_key(self, phase: str, payload: Mapping[str, JsonValue]) -> tuple[str, str]:
        identity = ":".join(
            str(payload.get(name, ""))
            for name in ("generation", "slot", "candidate_id", "evaluation_id", "session_id")
        )
        return phase, identity

    def _profile_payload(self, payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        if not self._profiling_enabled:
            result = dict(payload)
            result.pop("timing_profile", None)
            return result
        elapsed = max(0.0, time.monotonic() - self._profile_started)
        accounted = sum(self._phase_seconds.values())
        dominant_phase, dominant_seconds = max(
            self._phase_seconds.items(), key=lambda item: item[1], default=(None, 0.0)
        )
        evaluations = payload.get("evaluations")
        if isinstance(evaluations, int) and not isinstance(evaluations, bool):
            self._last_evaluations = max(self._last_evaluations, evaluations)
        phase_seconds = dict(self._phase_seconds)
        hotspots = [
            {"phase": phase, "seconds": seconds, "percent": seconds / max(elapsed, 1e-9) * 100}
            for phase, seconds in sorted(
                phase_seconds.items(), key=lambda item: item[1], reverse=True
            )[:8]
        ]
        profile: dict[str, JsonValue] = {
            "enabled": True,
            "phase_seconds": cast(JsonValue, phase_seconds),
            "phase_calls": cast(JsonValue, dict(self._phase_calls)),
            "phase_children_seconds": {},
            "phase_children_calls": {},
            "phase_grandchildren_seconds": {},
            "phase_grandchildren_calls": {},
            "measured_total_seconds": elapsed,
            "accounted_seconds": accounted,
            "unattributed_seconds": max(0.0, elapsed - accounted),
            "unattributed_fraction": max(0.0, elapsed - accounted) / max(elapsed, 1e-9),
            "dominant_phase": dominant_phase or "unavailable",
            "dominant_seconds": dominant_seconds,
            "profiled_episodes": self._last_evaluations,
            "throughput": self._last_evaluations / max(elapsed, 1e-9),
            "evaluations_per_second": self._last_evaluations / max(elapsed, 1e-9),
            "hotspots": cast(JsonValue, hotspots),
            "worker_utilization": payload.get("worker_utilization", "unavailable"),
        }
        supplied = payload.get("timing_profile")
        if isinstance(supplied, Mapping):
            # Native evaluators may provide richer inclusive/child profiles;
            # retain every field while keeping the live phase totals current.
            self._supplied_profile.update(
                {str(key): _json_safe(value) for key, value in supplied.items()}
            )
        if self._supplied_profile:
            supplied_profile = dict(self._supplied_profile)
            # Native evaluators may provide richer inclusive/child profiles;
            # retain every field while keeping the live phase totals current.
            for key in ("phase_seconds", "phase_calls"):
                current = profile.get(key)
                incoming = supplied_profile.get(key)
                if isinstance(current, Mapping) and isinstance(incoming, Mapping):
                    profile[key] = {
                        **dict(current),
                        **dict(incoming),
                    }
                    supplied_profile.pop(key, None)
            profile.update(supplied_profile)
            profile["enabled"] = True
        result = dict(payload)
        result["timing_profile"] = cast(JsonValue, profile)
        return result

    def emit(self, event_type: str, **payload: Any) -> Event:
        """Emit an event to all configured sinks and the durable session."""

        with self._lock:
            if self._closed:
                raise RuntimeError("experiment event hub is closed")
            safe_payload = self._profile_payload(
                {str(key): _json_safe(value) for key, value in payload.items()}
            )
            now = time.monotonic()
            phase = _PHASE_STARTS.get(event_type)
            if phase is not None:
                self._phase_started[self._profile_key(phase, safe_payload)] = now
                self._phase_calls[phase] = self._phase_calls.get(phase, 0) + 1
            phase = _PHASE_ENDS.get(event_type)
            if phase is not None:
                key = self._profile_key(phase, safe_payload)
                started = self._phase_started.pop(key, None)
                if started is not None:
                    self._phase_seconds[phase] = self._phase_seconds.get(phase, 0.0) + max(
                        0.0, now - started
                    )
                # The first profile pass may have attached a generated
                # timing_profile.  Recompute it after closing the phase
                # without treating that generated payload as new input.
                safe_payload.pop("timing_profile", None)
                safe_payload = self._profile_payload(safe_payload)
            if "session_id" not in safe_payload and self._session is not None:
                safe_payload["session_id"] = str(self._session.session_id)
            event = self._bus.emit(event_type, **safe_payload)
            manager = self._session_manager
            session = self._session
            if manager is not None and session is not None:
                manager.event(session, event_type, **safe_payload)
            else:
                self._pending_session_events.append((event_type, dict(safe_payload)))
            return event

    def __call__(
        self,
        event_type: str,
        payload: Mapping[str, JsonValue] | None = None,
        **kwargs: Any,
    ) -> Event:
        values: dict[str, Any] = dict(payload or {})
        values.update(kwargs)
        return self.emit(event_type, **values)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._bus.close()


# Friendly aliases for callers/tests that use observer terminology.
ExperimentObserver = ExperimentEventHub
NativeEventBus = ExperimentEventHub


__all__ = [
    "ExperimentEventHub",
    "ExperimentEventObserver",
    "ExperimentObserver",
    "NativeEventBus",
    "CallbackEventSink",
]
