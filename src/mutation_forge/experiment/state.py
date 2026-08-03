"""Durable operational state for experiment continuation.

The filesystem remains the evidence authority.  SQLite stores orchestration
indexes, ownership, idempotency, and the last safe checkpoint so that a
process can stop and resume without repeating completed work.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = 1
TERMINAL_STATES = frozenset({"completed"})
VALID_STATES = frozenset({"created", "running", "idle", "interrupted", "failed", "completed"})
_USAGE_FIELDS = (
    "inputTokens",
    "cachedInputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
)


class StateError(RuntimeError):
    """The durable experiment state is missing or inconsistent."""


class ActiveSessionError(StateError):
    """Another live process currently owns the experiment."""

    def __init__(self, exp_id: str, owner_pid: int | None, owner_started_at: str | None) -> None:
        self.exp_id = exp_id
        self.owner_pid = owner_pid
        self.owner_started_at = owner_started_at
        detail = f"Experiment {exp_id!r} already has an active session" + (
            f" (owner PID {owner_pid}, started {owner_started_at})" if owner_pid else ""
        )
        super().__init__(detail)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _usage_quality(value: Mapping[str, Any]) -> str:
    exact = (
        value.get("final") is True
        and value.get("partial") is False
        and all(
            isinstance(value.get(name), int)
            and not isinstance(value.get(name), bool)
            and int(value[name]) >= 0
            for name in _USAGE_FIELDS
        )
    )
    if exact:
        return "exact"
    if value.get("partial") is True or any(
        isinstance(value.get(name), int)
        and not isinstance(value.get(name), bool)
        and int(value[name]) >= 0
        for name in _USAGE_FIELDS
    ):
        return "partial"
    return "unknown"


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ExperimentStateStore:
    """A small transactional SQLite state store.

    A new connection is opened for each store instance.  Ownership changes use
    ``BEGIN IMMEDIATE`` so two CLI processes cannot both become active owners.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Native generation may invoke provider slots concurrently.  All
        # mutating ledger operations use explicit SQLite transactions; allow
        # those operations to originate from the coordinator's worker threads.
        self.connection = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._verify_schema()

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        *,
        exp_id: str,
        lock_hash: str,
        root: str | Path,
    ) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(state_path, timeout=5.0)
        try:
            connection.executescript(
                """
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment (
                    exp_id TEXT PRIMARY KEY,
                    root TEXT NOT NULL,
                    lock_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    current_session_id TEXT,
                    current_checkpoint TEXT,
                    cumulative_model_turns INTEGER NOT NULL DEFAULT 0,
                    cumulative_tokens INTEGER NOT NULL DEFAULT 0,
                    cumulative_runtime_seconds REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    terminal_stop_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    number INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL UNIQUE,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    wall_seconds REAL NOT NULL,
                    starting_checkpoint TEXT,
                    ending_checkpoint TEXT,
                    starting_state TEXT NOT NULL,
                    ending_state TEXT,
                    status TEXT NOT NULL,
                    provider_turns_attempted INTEGER NOT NULL DEFAULT 0,
                    provider_turns_completed INTEGER NOT NULL DEFAULT 0,
                    candidates_created INTEGER NOT NULL DEFAULT 0,
                    evaluations_completed INTEGER NOT NULL DEFAULT 0,
                    token_usage_delta INTEGER NOT NULL DEFAULT 0,
                    cumulative_tokens INTEGER NOT NULL DEFAULT 0,
                    runtime_seconds REAL NOT NULL DEFAULT 0,
                    stop_reason TEXT,
                    exit_status INTEGER,
                    summary_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS ownership (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    exp_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    started_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_turns (
                    idempotency_key TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    slot TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    state TEXT NOT NULL,
                    provider_thread_id TEXT,
                    provider_turn_id TEXT,
                    artifact_path TEXT,
                    usage_json TEXT NOT NULL DEFAULT '{}',
                    completed_at TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    source_sha256 TEXT,
                    archive_path TEXT,
                    generation INTEGER,
                    slot TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    identity TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    sequence INTEGER PRIMARY KEY,
                    checkpoint_id TEXT NOT NULL UNIQUE,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    completed_slots INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            now = _now()
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                ("schema_version", str(STATE_SCHEMA_VERSION)),
            )
            connection.execute(
                "INSERT INTO experiment(exp_id,root,lock_hash,state,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (exp_id, str(Path(root).resolve()), lock_hash, "created", now, now),
            )
            connection.commit()
        finally:
            connection.close()

    def _verify_schema(self) -> None:
        try:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise StateError(f"cannot read state database: {self.path}") from exc
        if row is None or int(row[0]) != STATE_SCHEMA_VERSION:
            raise StateError(f"unsupported state database schema: {self.path}")
        integrity = self.connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise StateError(f"state database integrity check failed: {self.path}")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ExperimentStateStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def experiment(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM experiment LIMIT 1").fetchone()
        if row is None:
            raise StateError("state database has no experiment row")
        return dict(row)

    def state(self) -> str:
        value = self.experiment()["state"]
        if value not in VALID_STATES:
            raise StateError(f"invalid experiment state: {value!r}")
        return str(value)

    def set_state(
        self,
        state: str,
        *,
        error: str | None = None,
        stop_reason: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"invalid experiment state: {state!r}")
        self.connection.execute(
            "UPDATE experiment SET state=?,updated_at=?,last_error=?,"
            "terminal_stop_reason=?,current_checkpoint=?",
            (state, _now(), error, stop_reason, checkpoint),
        )
        self.connection.commit()

    def next_session_number(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(number),0)+1 FROM sessions").fetchone()
        return int(row[0]) if row is not None else 1

    def create_session(
        self,
        *,
        number: int,
        session_id: str,
        wall_seconds: float,
        starting_checkpoint: str | None,
    ) -> None:
        current = self.state()
        self.connection.execute(
            "INSERT INTO sessions(number,session_id,started_at,wall_seconds,"
            "starting_checkpoint,starting_state,status) "
            "VALUES(?,?,?,?,?,?,?)",
            (number, session_id, _now(), wall_seconds, starting_checkpoint, current, "running"),
        )
        self.connection.execute(
            "UPDATE experiment SET current_session_id=?,state=?,updated_at=?",
            (session_id, "running", _now()),
        )
        self.connection.commit()

    def finish_session(
        self,
        session_id: str,
        *,
        status: str,
        ending_state: str,
        ending_checkpoint: str | None,
        provider_turns_attempted: int = 0,
        provider_turns_completed: int = 0,
        candidates_created: int = 0,
        evaluations_completed: int = 0,
        token_usage_delta: int = 0,
        cumulative_tokens: int = 0,
        runtime_seconds: float = 0.0,
        stop_reason: str | None = None,
        exit_status: int | None = 0,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            "UPDATE sessions SET finished_at=?,ending_checkpoint=?,ending_state=?,status=?,"
            "provider_turns_attempted=MAX(provider_turns_attempted,?),"
            "provider_turns_completed=MAX(provider_turns_completed,?),candidates_created=?,"
            "evaluations_completed=?,token_usage_delta=MAX(token_usage_delta,?),"
            "cumulative_tokens=?,runtime_seconds=?,"
            "stop_reason=?,exit_status=?,summary_json=? WHERE session_id=?",
            (
                _now(),
                ending_checkpoint,
                ending_state,
                status,
                provider_turns_attempted,
                provider_turns_completed,
                candidates_created,
                evaluations_completed,
                token_usage_delta,
                cumulative_tokens,
                runtime_seconds,
                stop_reason,
                exit_status,
                _json(dict(summary or {})),
                session_id,
            ),
        )
        self.connection.execute(
            "UPDATE experiment SET state=?,updated_at=?,"
            "cumulative_runtime_seconds=cumulative_runtime_seconds+?,"
            "last_error=?,terminal_stop_reason=? WHERE current_session_id=?",
            (
                ending_state,
                _now(),
                runtime_seconds,
                (summary or {}).get("last_error") if summary else None,
                stop_reason,
                session_id,
            ),
        )
        self.connection.commit()

    def session(self, session_id: str | None = None) -> dict[str, Any] | None:
        if session_id is None:
            row = self.connection.execute(
                "SELECT * FROM sessions ORDER BY number DESC LIMIT 1"
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def sessions(self) -> list[dict[str, Any]]:
        return [
            dict(row) for row in self.connection.execute("SELECT * FROM sessions ORDER BY number")
        ]

    def acquire_owner(
        self,
        *,
        exp_id: str,
        session_id: str,
        pid: int | None = None,
        started_at: str | None = None,
        alive: Callable[[int], bool] = process_alive,
    ) -> None:
        owner_pid = pid or os.getpid()
        owner_started = started_at or _now()
        connection = self.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM ownership WHERE singleton=1").fetchone()
            if row is not None:
                # Ownership is a durable hint, not a reason to block recovery.
                # A restarted invocation always takes over the workspace and
                # preserves the previous session as interrupted.
                connection.execute(
                    "UPDATE sessions SET status='interrupted',finished_at=?,"
                    "ending_state='interrupted',"
                    "stop_reason=COALESCE(stop_reason,'owner_process_died') "
                    "WHERE session_id=? AND status='running'",
                    (_now(), str(row["session_id"])),
                )
                connection.execute("DELETE FROM ownership WHERE singleton=1")
            connection.execute(
                "INSERT INTO ownership(singleton,exp_id,session_id,pid,started_at) "
                "VALUES(1,?,?,?,?)",
                (exp_id, session_id, owner_pid, owner_started),
            )
            connection.commit()
        except ActiveSessionError:
            raise
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise StateError("could not acquire experiment ownership") from exc
        except Exception:
            connection.rollback()
            raise

    def release_owner(self, session_id: str) -> None:
        self.connection.execute(
            "DELETE FROM ownership WHERE singleton=1 AND session_id=?", (session_id,)
        )
        self.connection.commit()

    def owner(self) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM ownership WHERE singleton=1").fetchone()
        return dict(row) if row is not None else None

    def write_event(
        self, event_type: str, payload: Mapping[str, Any], *, session_id: str | None = None
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO events(session_id,event_type,timestamp,payload_json) VALUES(?,?,?,?)",
            (session_id, event_type, _now(), _json(dict(payload))),
        )
        self.connection.commit()
        return int(cursor.lastrowid or 0)

    def record_provider_turn(
        self,
        *,
        idempotency_key: str,
        generation: int,
        slot: str,
        phase: str,
        state: str,
        artifact_path: str | None = None,
        usage: Mapping[str, Any] | None = None,
        provider_thread_id: str | None = None,
        provider_turn_id: str | None = None,
        error: str | None = None,
    ) -> bool:
        usage_value = dict(usage or {})
        usage_value["quality"] = _usage_quality(usage_value)
        total_tokens = usage_value.get("totalTokens", 0)
        if state == "completed" and (
            not isinstance(total_tokens, int) or isinstance(total_tokens, bool) or total_tokens < 0
        ):
            raise ValueError("completed provider turn requires non-negative totalTokens")
        terminal = state in {"completed", "failed"}
        completed = state == "completed"
        observed_tokens = (
            int(total_tokens)
            if isinstance(total_tokens, int)
            and not isinstance(total_tokens, bool)
            and total_tokens >= 0
            else 0
        )
        try:
            # The turn row, per-session delta, and experiment cumulative totals
            # are one crash-safe accounting unit.  A duplicate completed key
            # exits without touching any ledger.
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT state FROM provider_turns WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if row is not None and row["state"] == "completed":
                self.connection.rollback()
                return False
            if row is not None:
                self.connection.execute(
                    "UPDATE provider_turns SET state=?,provider_thread_id=?,provider_turn_id=?,"
                    "artifact_path=?,usage_json=?,completed_at=?,error=? "
                    "WHERE idempotency_key=?",
                    (
                        state,
                        provider_thread_id,
                        provider_turn_id,
                        artifact_path,
                        _json(usage_value),
                        _now() if terminal else None,
                        error,
                        idempotency_key,
                    ),
                )
            else:
                self.connection.execute(
                    "INSERT INTO provider_turns(idempotency_key,generation,slot,phase,state,"
                    "provider_thread_id,provider_turn_id,artifact_path,usage_json,"
                    "completed_at,error) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        idempotency_key,
                        generation,
                        slot,
                        phase,
                        state,
                        provider_thread_id,
                        provider_turn_id,
                        artifact_path,
                        _json(usage_value),
                        _now() if terminal else None,
                        error,
                    ),
                )
            if terminal:
                self.connection.execute(
                    "UPDATE sessions SET provider_turns_attempted=provider_turns_attempted+1,"
                    "provider_turns_completed=provider_turns_completed+?,"
                    "token_usage_delta=token_usage_delta+? "
                    "WHERE session_id=(SELECT current_session_id FROM experiment LIMIT 1)",
                    (1 if completed else 0, observed_tokens),
                )
                self.connection.execute(
                    "UPDATE experiment SET "
                    "cumulative_model_turns=cumulative_model_turns+?,"
                    "cumulative_tokens=cumulative_tokens+?,updated_at=?",
                    (1 if completed else 0, observed_tokens, _now()),
                )
            self.connection.commit()
            return True
        except BaseException:
            self.connection.rollback()
            raise

    def provider_turn(self, idempotency_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM provider_turns WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        return dict(row) if row is not None else None

    def record_candidate(self, candidate_id: str, **values: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO candidates(candidate_id,source_sha256,archive_path,"
            "generation,slot,status,metadata_json) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                candidate_id,
                values.get("source_sha256"),
                values.get("archive_path"),
                values.get("generation"),
                values.get("slot"),
                values.get("status", "created"),
                _json(values.get("metadata", {})),
            ),
        )
        self.connection.commit()

    def record_evaluation(self, identity: str, **values: Any) -> bool:
        existing = self.evaluation(identity)
        if existing is not None:
            if existing.get("state") == "completed":
                return False
            evaluation_state = values.get("state", "completed")
            self.connection.execute(
                "UPDATE evaluations SET state=?,result_json=?,completed_at=? WHERE identity=?",
                (
                    evaluation_state,
                    _json(values.get("result", {})),
                    _now() if evaluation_state == "completed" else None,
                    identity,
                ),
            )
            self.connection.commit()
            return True
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO evaluations(identity,candidate_id,kind,state,"
            "result_json,completed_at) VALUES(?,?,?,?,?,?)",
            (
                identity,
                values.get("candidate_id"),
                values.get("kind", "development"),
                values.get("state", "completed"),
                _json(values.get("result", {})),
                _now() if values.get("state", "completed") == "completed" else None,
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def evaluation(self, identity: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM evaluations WHERE identity=?", (identity,)
        ).fetchone()
        return dict(row) if row is not None else None

    def record_checkpoint(
        self,
        *,
        sequence: int,
        checkpoint_id: str,
        path: str,
        sha256: str,
        generation: int,
        completed_slots: int,
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO checkpoints(sequence,checkpoint_id,path,sha256,"
            "generation,completed_slots,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (sequence, checkpoint_id, path, sha256, generation, completed_slots, _now()),
        )
        self.connection.execute(
            "UPDATE experiment SET current_checkpoint=?,updated_at=?",
            (checkpoint_id, _now()),
        )
        self.connection.commit()

    def checkpoint(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM checkpoints ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    def counts(self) -> dict[str, int]:
        queries = {
            "candidate_count": "SELECT COUNT(*) FROM candidates",
            "unique_candidate_count": (
                "SELECT COUNT(*) FROM candidates WHERE status NOT IN ('duplicate','invalid')"
            ),
            "evaluation_count": "SELECT COUNT(*) FROM evaluations WHERE state='completed'",
            "provider_turns": "SELECT COUNT(*) FROM provider_turns",
            "provider_turns_completed": (
                "SELECT COUNT(*) FROM provider_turns WHERE state='completed'"
            ),
        }
        return {
            name: int(self.connection.execute(sql).fetchone()[0]) for name, sql in queries.items()
        }

    def latest_error(self) -> str | None:
        value = self.experiment().get("last_error")
        return str(value) if value else None

    def cumulative(self) -> dict[str, int | float]:
        row = self.experiment()
        return {
            "provider_turns": int(row["cumulative_model_turns"]),
            "total_tokens": int(row["cumulative_tokens"]),
            "compute_seconds": float(row["cumulative_runtime_seconds"]),
        }

    def token_usage(self) -> dict[str, int | str]:
        totals = {field: 0 for field in _USAGE_FIELDS}
        qualities: set[str] = set()
        charged_failed_turns = 0
        rows = self.connection.execute(
            "SELECT state,usage_json FROM provider_turns"
        ).fetchall()
        for row in rows:
            try:
                usage = json.loads(str(row["usage_json"]))
            except json.JSONDecodeError:
                usage = {}
            if not isinstance(usage, Mapping):
                continue
            for field in _USAGE_FIELDS:
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    totals[field] += value
            quality = usage.get("quality")
            if isinstance(quality, str):
                qualities.add(quality)
            turn_total = usage.get("totalTokens")
            if (
                row["state"] == "failed"
                and isinstance(turn_total, int)
                and not isinstance(turn_total, bool)
                and turn_total > 0
            ):
                charged_failed_turns += 1
        quality = (
            "unknown"
            if not qualities or qualities == {"unknown"}
            else "exact"
            if qualities == {"exact"}
            else "partial"
        )
        totals["totalTokens"] = int(self.experiment()["cumulative_tokens"])
        return {
            **totals,
            "quality": quality,
            "chargedFailedTurns": charged_failed_turns,
        }

    get_experiment = experiment
    get_session = session
    latest_checkpoint = checkpoint


StateStore = ExperimentStateStore


__all__ = [
    "ActiveSessionError",
    "ExperimentStateStore",
    "StateStore",
    "STATE_SCHEMA_VERSION",
    "StateError",
    "VALID_STATES",
    "process_alive",
]
