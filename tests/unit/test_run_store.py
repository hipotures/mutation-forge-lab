from __future__ import annotations

import sqlite3
from pathlib import Path

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
