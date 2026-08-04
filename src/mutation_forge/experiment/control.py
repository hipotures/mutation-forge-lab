"""Operator control shared by the dashboard and experiment engine."""

from __future__ import annotations

from threading import Event, Lock


class ExperimentControl:
    """Coordinate a graceful first stop and an immediate second interrupt."""

    def __init__(self) -> None:
        self._graceful_stop = Event()
        self._lock = Lock()

    @property
    def graceful_stop_requested(self) -> bool:
        return self._graceful_stop.is_set()

    def request_graceful_stop(self) -> bool:
        """Return true only for the first stop request."""

        with self._lock:
            if self._graceful_stop.is_set():
                return False
            self._graceful_stop.set()
            return True
