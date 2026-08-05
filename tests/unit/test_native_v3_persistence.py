from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mutation_forge.native_v3.persistence import (
    NativeV3Persistence,
    PersistenceConflict,
    SemanticRecord,
)


def _record(key: str, value: int) -> SemanticRecord:
    return SemanticRecord("episode", key, {"episode_id": key, "value": value})


def test_semantic_commit_is_idempotent_and_conflicts_fail_closed(tmp_path: Path) -> None:
    with NativeV3Persistence(tmp_path / "state.sqlite3") as store:
        first_hash = store.commit_semantic(_record("episode-1", 7))
        assert store.total_wall_time_ns > 0
        assert store.commit_semantic(_record("episode-1", 7)) == first_hash
        with pytest.raises(PersistenceConflict):
            store.commit_semantic(_record("episode-1", 8))


def test_sqlite_contains_semantic_cache_only(tmp_path: Path) -> None:
    checkpoints: list[tuple[str, bytes]] = []
    for index in range(2):
        path = tmp_path / f"state-{index}.sqlite3"
        with NativeV3Persistence(path) as store:
            store.commit_semantic(_record("episode-2", 11))
            checkpoints.append(store.semantic_checkpoint())
        with sqlite3.connect(path) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "native_v3_telemetry" not in tables
    assert checkpoints[0] == checkpoints[1]


def test_reader_connection_is_read_only(tmp_path: Path) -> None:
    with NativeV3Persistence(tmp_path / "state.sqlite3") as store:
        store.commit_semantic(_record("episode-3", 5))
        reader = store.read_connection()
        try:
            row = reader.execute("SELECT semantic_hash FROM native_v3_semantic_records").fetchone()
            assert row is not None
            with pytest.raises(Exception, match="readonly|read-only"):
                reader.execute(
                    "DELETE FROM native_v3_semantic_records WHERE semantic_key='episode-3'"
                )
        finally:
            reader.close()
