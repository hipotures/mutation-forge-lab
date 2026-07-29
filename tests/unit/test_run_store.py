from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mutation_forge.events import EventBus
from mutation_forge.run_store import RunStore


def test_sqlite_run_store_persists_events(tmp_path: Path) -> None:
    path = tmp_path / "archive.sqlite3"
    store = RunStore(path)
    store.create_run(run_id="r1", config_hash="abc", manifest={"version": 1})
    bus = EventBus("r1", [store])
    bus.emit("run_started", stage="stage1")
    store.finish("r1", "completed", {"status": "completed"})
    bus.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT status FROM runs").fetchone() == ("completed",)
        assert connection.execute("SELECT event_type FROM events").fetchone() == (
            "run_started",
        )
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        connection.close()


def test_sqlite_run_store_commits_progress_at_episode_end(tmp_path: Path) -> None:
    path = tmp_path / "archive.sqlite3"
    store = RunStore(path)
    store.create_run(run_id="r1", config_hash="abc", manifest={"version": 1})
    bus = EventBus("r1", [store])
    observer = sqlite3.connect(path)

    try:
        bus.emit("episode_started", baseline="uniform")
        bus.emit("episode_progress", evaluations=50)
        assert observer.execute("SELECT COUNT(*) FROM events").fetchone() == (0,)

        bus.emit("episode_completed", evaluations=100)
        assert observer.execute(
            "SELECT event_type FROM events ORDER BY sequence"
        ).fetchall() == [
            ("episode_started",),
            ("episode_progress",),
            ("episode_completed",),
        ]
    finally:
        observer.close()
        bus.close()


@pytest.mark.parametrize("terminal_event", ["run_completed", "run_failed"])
def test_sqlite_run_store_terminal_event_commits_pending_events(
    tmp_path: Path,
    terminal_event: str,
) -> None:
    path = tmp_path / "archive.sqlite3"
    store = RunStore(path)
    store.create_run(run_id="r1", config_hash="abc", manifest={"version": 1})
    bus = EventBus("r1", [store])
    observer = sqlite3.connect(path)

    try:
        bus.emit("episode_started", baseline="uniform")
        bus.emit("episode_progress", evaluations=50)
        assert observer.execute("SELECT COUNT(*) FROM events").fetchone() == (0,)

        bus.emit(terminal_event, status=terminal_event.removeprefix("run_"))
        assert observer.execute(
            "SELECT event_type FROM events ORDER BY sequence"
        ).fetchall() == [
            ("episode_started",),
            ("episode_progress",),
            (terminal_event,),
        ]
    finally:
        observer.close()
        bus.close()
