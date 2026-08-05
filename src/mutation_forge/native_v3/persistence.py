"""Single-writer persistence and semantic replay identity for Native v3."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutation_forge.models import JsonValue

from .canonical import canonical_json_bytes, domain_hash, json_value

PERSISTENCE_SCHEMA_VERSION = "mforge.native.persistence.v3"
SEMANTIC_CHECKPOINT_PROTOCOL = "native_v3_semantic_checkpoint_v1"


class PersistenceConflict(RuntimeError):
    """An idempotent key was committed with different semantic content."""


@dataclass(frozen=True, slots=True)
class SemanticRecord:
    record_type: str
    semantic_key: str
    payload: Mapping[str, JsonValue]

    @property
    def canonical_payload(self) -> bytes:
        return canonical_json_bytes(dict(self.payload))

    @property
    def semantic_hash(self) -> str:
        return domain_hash(
            b"mforge-native-v3-semantic-record\0",
            canonical_json_bytes(
                {
                    "record_type": self.record_type,
                    "semantic_key": self.semantic_key,
                    "payload": dict(self.payload),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    event_name: str
    observed_at_ns: int
    fields: Mapping[str, JsonValue]


@dataclass(slots=True)
class _WriteCommand:
    kind: str
    value: SemanticRecord | TelemetryRecord | None
    future: Future[str | None]


class NativeV3Persistence:
    """Own the only writable SQLite connection in a dedicated thread."""

    def __init__(
        self,
        path: Path,
        *,
        queue_capacity: int = 32,
        maximum_batch_latency_seconds: float = 1.0,
    ) -> None:
        if queue_capacity <= 0 or maximum_batch_latency_seconds <= 0:
            raise ValueError("persistence queue and batch latency limits must be positive")
        self.path = path
        self.queue_capacity = queue_capacity
        self.maximum_batch_latency_seconds = maximum_batch_latency_seconds
        self._commands: queue.Queue[_WriteCommand] = queue.Queue(queue_capacity)
        self._started = threading.Event()
        self._closed = False
        self._metric_lock = threading.Lock()
        self._wall_time_ns = 0
        self._thread = threading.Thread(
            target=self._writer_main,
            name="native-v3-persistence-owner",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=10):
            raise RuntimeError("Native v3 persistence writer failed to start")

    def commit_semantic(self, record: SemanticRecord) -> str:
        return_value = self._submit("semantic", record)
        assert isinstance(return_value, str)
        return return_value

    def record_telemetry(self, record: TelemetryRecord) -> None:
        self._submit("telemetry", record)

    def semantic_checkpoint(self) -> tuple[str, bytes]:
        value = self._submit("checkpoint", None)
        assert isinstance(value, str)
        encoded = bytes.fromhex(value)
        hash_bytes, payload = encoded[:32], encoded[32:]
        return hash_bytes.hex(), payload

    def read_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def semantic_records(self, record_type: str) -> tuple[SemanticRecord, ...]:
        connection = self.read_connection()
        try:
            rows = connection.execute(
                "SELECT semantic_key,canonical_payload "
                "FROM native_v3_semantic_records WHERE record_type=? "
                "ORDER BY semantic_key",
                (record_type,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            SemanticRecord(
                record_type,
                str(row["semantic_key"]),
                json.loads(bytes(row["canonical_payload"])),
            )
            for row in rows
        )

    @property
    def total_wall_time_ns(self) -> int:
        with self._metric_lock:
            return self._wall_time_ns

    def close(self) -> None:
        if self._closed:
            return
        self._submit("close", None)
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Native v3 persistence writer did not stop")
        self._closed = True

    def __enter__(self) -> NativeV3Persistence:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _submit(
        self,
        kind: str,
        value: SemanticRecord | TelemetryRecord | None,
    ) -> str | None:
        if self._closed:
            raise RuntimeError("Native v3 persistence is closed")
        started_ns = time.monotonic_ns()
        future: Future[str | None] = Future()
        self._commands.put(_WriteCommand(kind, value, future))
        try:
            return future.result()
        finally:
            with self._metric_lock:
                self._wall_time_ns += time.monotonic_ns() - started_ns

    def _writer_main(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS native_v3_semantic_records (
                    record_type TEXT NOT NULL,
                    semantic_key TEXT NOT NULL,
                    semantic_hash TEXT NOT NULL,
                    canonical_payload BLOB NOT NULL,
                    PRIMARY KEY (record_type, semantic_key)
                ) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS native_v3_telemetry (
                    telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    observed_at_ns INTEGER NOT NULL,
                    fields_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS native_v3_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO native_v3_metadata(key,value) VALUES (?,?)",
                ("schema_version", PERSISTENCE_SCHEMA_VERSION),
            )
            connection.commit()
            self._started.set()
            closing = False
            while not closing:
                first = self._commands.get()
                batch = [first]
                deadline = time.monotonic() + self.maximum_batch_latency_seconds
                while time.monotonic() < deadline:
                    try:
                        batch.append(self._commands.get_nowait())
                    except queue.Empty:
                        break
                try:
                    checkpoint_commands: list[_WriteCommand] = []
                    completed: list[tuple[_WriteCommand, str | None]] = []
                    for command in batch:
                        if command.kind == "semantic":
                            assert isinstance(command.value, SemanticRecord)
                            completed.append(
                                (
                                    command,
                                    self._write_semantic(connection, command.value),
                                )
                            )
                        elif command.kind == "telemetry":
                            assert isinstance(command.value, TelemetryRecord)
                            self._write_telemetry(connection, command.value)
                            completed.append((command, None))
                        elif command.kind == "checkpoint":
                            checkpoint_commands.append(command)
                        elif command.kind == "close":
                            closing = True
                            completed.append((command, None))
                        else:
                            raise RuntimeError(f"unknown persistence command {command.kind!r}")
                    connection.commit()
                    for command in checkpoint_commands:
                        checkpoint_hash, payload = self._checkpoint(connection)
                        command.future.set_result((bytes.fromhex(checkpoint_hash) + payload).hex())
                    for command, result in completed:
                        command.future.set_result(result)
                except BaseException as error:
                    connection.rollback()
                    for command in batch:
                        if not command.future.done():
                            command.future.set_exception(error)
        finally:
            self._started.set()
            connection.close()

    @staticmethod
    def _write_semantic(connection: sqlite3.Connection, record: SemanticRecord) -> str:
        existing = connection.execute(
            "SELECT semantic_hash FROM native_v3_semantic_records "
            "WHERE record_type=? AND semantic_key=?",
            (record.record_type, record.semantic_key),
        ).fetchone()
        if existing is not None:
            existing_hash = str(existing[0])
            if existing_hash != record.semantic_hash:
                raise PersistenceConflict(
                    f"semantic conflict for {record.record_type}:{record.semantic_key}"
                )
            return existing_hash
        connection.execute(
            "INSERT INTO native_v3_semantic_records("
            "record_type,semantic_key,semantic_hash,canonical_payload"
            ") VALUES (?,?,?,?)",
            (
                record.record_type,
                record.semantic_key,
                record.semantic_hash,
                record.canonical_payload,
            ),
        )
        return record.semantic_hash

    @staticmethod
    def _write_telemetry(connection: sqlite3.Connection, record: TelemetryRecord) -> None:
        fields = json.dumps(
            json_value(dict(record.fields)),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        connection.execute(
            "INSERT INTO native_v3_telemetry(event_name,observed_at_ns,fields_json) VALUES (?,?,?)",
            (record.event_name, record.observed_at_ns, fields),
        )

    @staticmethod
    def _checkpoint(connection: sqlite3.Connection) -> tuple[str, bytes]:
        rows = connection.execute(
            "SELECT record_type,semantic_key,semantic_hash,canonical_payload "
            "FROM native_v3_semantic_records ORDER BY record_type,semantic_key"
        ).fetchall()
        records: list[dict[str, Any]] = []
        for record_type, semantic_key, semantic_hash, payload in rows:
            records.append(
                {
                    "record_type": str(record_type),
                    "semantic_key": str(semantic_key),
                    "semantic_hash": str(semantic_hash),
                    "payload": json.loads(bytes(payload)),
                }
            )
        canonical = canonical_json_bytes(
            {
                "protocol": SEMANTIC_CHECKPOINT_PROTOCOL,
                "records": records,
            }
        )
        return domain_hash(b"mforge-native-v3-checkpoint\0", canonical), canonical
