from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mutation_forge.events import Event

SCHEMA_VERSION = "mforge.experiment.run.v2"
COMMIT_EVENT_TYPES = frozenset(
    {
        "episode_completed",
        "run_completed",
        "run_failed",
        "counterexample_verified",
        "experiment_completed",
    }
)


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        preexisting = path.is_file() and path.stat().st_size > 0
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                summary_json TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            """
        )
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            if preexisting:
                self.connection.close()
                raise ValueError(
                    "Experiment run database has no v2 schema marker. "
                    "Create a fresh workspace; migration is not supported."
                )
            self.connection.execute(
                "INSERT INTO metadata(key,value) VALUES('schema_version',?)",
                (SCHEMA_VERSION,),
            )
        elif row[0] != SCHEMA_VERSION:
            self.connection.close()
            raise ValueError(
                f"Unsupported experiment run schema: {row[0]!r}. "
                f"This runtime accepts only {SCHEMA_VERSION}. Create a fresh workspace."
            )
        self.connection.commit()

    def create_run(
        self,
        *,
        run_id: str,
        config_hash: str,
        manifest: dict[str, object],
    ) -> None:
        self.connection.execute(
            "INSERT INTO runs(run_id,status,config_hash,manifest_json) VALUES(?,?,?,?)",
            (run_id, "running", config_hash, json.dumps(manifest, sort_keys=True)),
        )
        self.connection.commit()

    def write(self, event: Event) -> None:
        if event.schema_version != "mforge.experiment.events.v2":
            raise ValueError(
                "RunStore accepts only mforge.experiment.events.v2 event records"
            )
        self.connection.execute(
            """
            INSERT INTO events(run_id,event_type,timestamp,payload_json)
            VALUES(?,?,?,?)
            """,
            (
                event.run_id,
                event.event_type,
                event.timestamp,
                json.dumps(
                    {"schema_version": event.schema_version, **event.payload},
                    sort_keys=True,
                ),
            ),
        )
        if event.event_type in COMMIT_EVENT_TYPES:
            self.connection.commit()

    def finish(self, run_id: str, status: str, summary: dict[str, object]) -> None:
        self.connection.execute(
            "UPDATE runs SET status=?, summary_json=? WHERE run_id=?",
            (status, json.dumps(summary, sort_keys=True), run_id),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
