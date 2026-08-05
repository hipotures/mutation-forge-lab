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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = "mforge.experiment.state.v2"
TERMINAL_STATES = frozenset({"exhausted", "completed"})
RESUMABLE_STATES = frozenset({"idle", "paused", "interrupted", "failed"})
VALID_STATES = frozenset(
    {
        "running",
        "idle",
        "paused",
        "interrupted",
        "failed",
        "exhausted",
        "completed",
    }
)
_USAGE_FIELDS = (
    "inputTokens",
    "cachedInputTokens",
    "cacheWriteInputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
)
_EXACT_USAGE_FIELDS = tuple(
    field for field in _USAGE_FIELDS if field != "cacheWriteInputTokens"
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
            for name in _EXACT_USAGE_FIELDS
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
                    idempotency_key TEXT UNIQUE,
                    payload_json TEXT NOT NULL
                );
                """
            )
            existing_schema = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing_schema is not None and existing_schema[0] != STATE_SCHEMA_VERSION:
                raise StateError(
                    f"Unsupported experiment state schema: {existing_schema[0]!r}. "
                    f"This runtime accepts only {STATE_SCHEMA_VERSION}. Create a fresh workspace."
                )
            existing_experiment = connection.execute(
                "SELECT 1 FROM experiment LIMIT 1"
            ).fetchone()
            if existing_experiment is not None:
                raise StateError("experiment state database is already initialized")
            now = _now()
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                ("schema_version", STATE_SCHEMA_VERSION),
            )
            connection.execute(
                "INSERT INTO experiment(exp_id,root,lock_hash,state,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (exp_id, str(Path(root).resolve()), lock_hash, "idle", now, now),
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
        if row is None or row[0] != STATE_SCHEMA_VERSION:
            observed = None if row is None else row[0]
            raise StateError(
                f"Unsupported experiment state schema: {observed}. "
                f"This runtime accepts only {STATE_SCHEMA_VERSION}. "
                "Create a fresh workspace."
            )
        integrity = self.connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise StateError(f"state database integrity check failed: {self.path}")
        for row in self.connection.execute(
            "SELECT session_id,summary_json FROM sessions WHERE summary_json != '{}'"
        ):
            try:
                summary = json.loads(str(row["summary_json"]))
            except json.JSONDecodeError as exc:
                raise StateError(
                    f"session {row['session_id']} has an unreadable summary"
                ) from exc
            if not isinstance(summary, Mapping) or summary.get(
                "schema_version"
            ) != "mforge.experiment.session.v2":
                observed = summary.get("schema_version") if isinstance(summary, Mapping) else None
                raise StateError(
                    f"Unsupported session summary schema: {observed!r}. "
                    "This runtime accepts only mforge.experiment.session.v2. "
                    "Create a fresh workspace."
                )
        for row in self.connection.execute("SELECT payload_json FROM events"):
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError as exc:
                raise StateError("experiment event payload is unreadable") from exc
            schema = payload.get("schema_version") if isinstance(payload, Mapping) else None
            if schema is not None and schema != "mforge.experiment.events.v2":
                raise StateError(
                    f"Unsupported experiment event schema: {schema!r}. "
                    "This runtime accepts only mforge.experiment.events.v2. "
                    "Create a fresh workspace."
                )

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
        if state == "completed" and stop_reason not in {
            "counterexample_verified",
            "operator_final_stop",
        }:
            raise StateError(
                "COMPLETED requires counterexample_verified or operator_final_stop"
            )
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
            "UPDATE experiment SET current_session_id=?,state=?,updated_at=?,"
            "last_error=NULL,terminal_stop_reason=NULL",
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

    def latest_meaningful_stop_reason(self) -> str | None:
        row = self.connection.execute(
            "SELECT stop_reason FROM sessions "
            "WHERE stop_reason IS NOT NULL AND stop_reason != 'already_completed' "
            "ORDER BY number DESC LIMIT 1"
        ).fetchone()
        return str(row[0]) if row is not None else None

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
                if alive(int(row["pid"])):
                    raise ActiveSessionError(
                        exp_id,
                        int(row["pid"]),
                        str(row["started_at"]),
                    )
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
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        session_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> int:
        raw_key = idempotency_key if idempotency_key is not None else payload.get("idempotency_key")
        idempotency_key = raw_key if isinstance(raw_key, str) and raw_key else None
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO events("
            "session_id,event_type,timestamp,idempotency_key,payload_json"
            ") VALUES(?,?,?,?,?)",
            (session_id, event_type, _now(), idempotency_key, _json(dict(payload))),
        )
        self.connection.commit()
        if cursor.rowcount:
            return int(cursor.lastrowid or 0)
        row = self.connection.execute(
            "SELECT sequence FROM events WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def event_exists(self, idempotency_key: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM events WHERE idempotency_key=? LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        return row is not None

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
                "SELECT state,usage_json FROM provider_turns WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if row is not None and row["state"] == "completed":
                self.connection.rollback()
                return False
            prior_usage: Mapping[str, Any] = {}
            if row is not None:
                try:
                    decoded = json.loads(str(row["usage_json"]))
                    if isinstance(decoded, Mapping):
                        prior_usage = decoded
                except json.JSONDecodeError:
                    prior_usage = {}
            prior_total = prior_usage.get("totalTokens", 0)
            prior_consumed = row is not None and (
                row["state"] == "completed"
                or (
                    isinstance(prior_total, int)
                    and not isinstance(prior_total, bool)
                    and prior_total > 0
                )
            )
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
                consumed = completed or observed_tokens > 0
                prior_observed_tokens = (
                    int(prior_total)
                    if isinstance(prior_total, int)
                    and not isinstance(prior_total, bool)
                    and prior_total >= 0
                    else 0
                )
                token_delta = max(0, observed_tokens - prior_observed_tokens)
                if token_delta:
                    charged_at = _now()
                    charge_payload = {
                        "idempotency_key": (
                            f"model-token-charge:{idempotency_key}:{observed_tokens}"
                        ),
                        "turn_idempotency_key": idempotency_key,
                        "generation": generation,
                        "slot": slot,
                        "phase": phase,
                        "token_delta": token_delta,
                        "usage": usage_value,
                        "charged_at": charged_at,
                    }
                    self.connection.execute(
                        "INSERT OR IGNORE INTO events("
                        "session_id,event_type,timestamp,idempotency_key,payload_json"
                        ") VALUES((SELECT current_session_id FROM experiment LIMIT 1),?,?,?,?)",
                        (
                            "model_token_charge_recorded",
                            charged_at,
                            charge_payload["idempotency_key"],
                            _json(charge_payload),
                        ),
                    )
                self.connection.execute(
                    "UPDATE sessions SET provider_turns_attempted=provider_turns_attempted+?,"
                    "provider_turns_completed=provider_turns_completed+?,"
                    "token_usage_delta=token_usage_delta+? "
                    "WHERE session_id=(SELECT current_session_id FROM experiment LIMIT 1)",
                    (
                        1 if row is None else 0,
                        1 if completed and (row is None or row["state"] != "completed") else 0,
                        token_delta,
                    ),
                )
                self.connection.execute(
                    "UPDATE experiment SET "
                    "cumulative_model_turns=cumulative_model_turns+?,"
                    "cumulative_tokens=cumulative_tokens+?,updated_at=?",
                    (
                        1 if consumed and not prior_consumed else 0,
                        token_delta,
                        _now(),
                    ),
                )
            self.connection.commit()
            return True
        except BaseException:
            self.connection.rollback()
            raise

    def _backfill_token_charge_events(self) -> None:
        rows = self.connection.execute(
            "SELECT idempotency_key,generation,slot,phase,completed_at,usage_json "
            "FROM provider_turns WHERE completed_at IS NOT NULL"
        ).fetchall()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                try:
                    usage = json.loads(str(row["usage_json"]))
                except json.JSONDecodeError:
                    continue
                total = usage.get("totalTokens") if isinstance(usage, Mapping) else None
                if (
                    not isinstance(total, int)
                    or isinstance(total, bool)
                    or total <= 0
                    or not isinstance(row["completed_at"], str)
                ):
                    continue
                turn_key = str(row["idempotency_key"])
                event_key = f"model-token-charge:{turn_key}:{total}"
                payload = {
                    "idempotency_key": event_key,
                    "turn_idempotency_key": turn_key,
                    "generation": int(row["generation"]),
                    "slot": str(row["slot"]),
                    "phase": str(row["phase"]),
                    "token_delta": total,
                    "usage": usage,
                    "charged_at": str(row["completed_at"]),
                    "backfilled": True,
                }
                self.connection.execute(
                    "INSERT OR IGNORE INTO events("
                    "session_id,event_type,timestamp,idempotency_key,payload_json"
                    ") VALUES(NULL,?,?,?,?)",
                    (
                        "model_token_charge_recorded",
                        str(row["completed_at"]),
                        event_key,
                        _json(payload),
                    ),
                )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def hourly_token_usage(
        self,
        limit: int | None,
        *,
        now: datetime | None = None,
        backfill: bool = False,
    ) -> dict[str, Any]:
        if backfill:
            self._backfill_token_charge_events()
        current = now or datetime.now(UTC)
        current = (
            current.replace(tzinfo=UTC)
            if current.tzinfo is None
            else current.astimezone(UTC)
        )
        cutoff = current - timedelta(hours=1)
        charges: list[tuple[datetime, int]] = []
        rows = self.connection.execute(
            "SELECT timestamp,payload_json FROM events "
            "WHERE event_type='model_token_charge_recorded' "
            "AND timestamp>? AND timestamp<=? ORDER BY timestamp,sequence",
            (cutoff.isoformat(), current.isoformat()),
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
                charged_at = datetime.fromisoformat(
                    str(payload.get("charged_at", row["timestamp"]))
                )
            except (ValueError, json.JSONDecodeError):
                continue
            if charged_at.tzinfo is None:
                charged_at = charged_at.replace(tzinfo=UTC)
            delta = payload.get("token_delta") if isinstance(payload, Mapping) else None
            if (
                cutoff < charged_at <= current
                and isinstance(delta, int)
                and not isinstance(delta, bool)
                and delta > 0
            ):
                charges.append((charged_at, delta))
        used = sum(delta for _, delta in charges)
        reached = limit is not None and used >= limit
        retry_after: str | None = None
        if reached and limit is not None:
            remaining = used
            for charged_at, delta in charges:
                remaining -= delta
                if remaining < limit:
                    retry_after = (charged_at + timedelta(hours=1)).isoformat()
                    break
        return {
            "hourly_token_limit": limit,
            "hourly_tokens_used": used,
            "hourly_tokens_remaining": (
                None if limit is None else max(0, limit - used)
            ),
            "hourly_window_seconds": 3600,
            "hourly_limit_reached": reached,
            "hourly_retry_after": retry_after,
        }

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

    def metadata(self, key: str) -> Any | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(str(row[0]))
        except json.JSONDecodeError as exc:
            raise StateError(f"invalid metadata value for {key!r}") from exc

    def set_metadata(self, key: str, value: object) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (key, _json(value)),
        )
        self.connection.commit()

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
        rows = self.connection.execute("SELECT state,usage_json FROM provider_turns").fetchall()
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
    "RESUMABLE_STATES",
    "StateError",
    "VALID_STATES",
    "process_alive",
]
