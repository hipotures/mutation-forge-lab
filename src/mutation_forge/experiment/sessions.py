"""Numbered execution sessions and append-only session evidence."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .json_io import write_json
from .layout import ExperimentLayout
from .state import ExperimentStateStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_gzip_member(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)
        raw.flush()
        os.fsync(raw.fileno())


@dataclass(slots=True)
class SessionContext:
    number: int
    session_id: str
    directory: Path
    wall_seconds: float
    started_at: str
    starting_checkpoint: str | None
    starting_state: str = "running"
    monotonic_started: float = field(default_factory=time.monotonic)
    provider_turns_attempted: int = 0
    provider_turns_completed: int = 0
    candidates_created: int = 0
    evaluations_completed: int = 0
    token_usage_delta: int = 0
    event_count: int = 0
    ending_checkpoint: str | None = None
    stop_reason: str | None = None

    @property
    def deadline(self) -> float:
        return self.monotonic_started + self.wall_seconds

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.monotonic_started)

    def budget_exhausted(self) -> bool:
        return time.monotonic() >= self.deadline

    def as_dict(
        self,
        *,
        ending_state: str | None = None,
        exit_status: int | None = None,
        cumulative_tokens: int = 0,
    ) -> dict[str, Any]:
        return {
            "schema_version": "mforge.experiment.session.v2",
            "session_number": self.number,
            "session_id": self.session_id,
            "start_time": self.started_at,
            "finish_time": _now(),
            "wall_seconds": self.wall_seconds,
            "starting_checkpoint": self.starting_checkpoint,
            "ending_checkpoint": self.ending_checkpoint,
            "starting_state": self.starting_state,
            "ending_state": ending_state,
            "provider_turns_attempted": self.provider_turns_attempted,
            "provider_turns_completed": self.provider_turns_completed,
            "candidates_created": self.candidates_created,
            "evaluations_completed": self.evaluations_completed,
            "token_usage_delta": self.token_usage_delta,
            "cumulative_tokens": cumulative_tokens,
            "stop_reason": self.stop_reason,
            "exit_status": exit_status,
            "runtime_seconds": self.elapsed_seconds,
        }


class SessionManager:
    def __init__(self, layout: ExperimentLayout, state: ExperimentStateStore) -> None:
        self.layout = layout
        self.state = state

    def start(self, config: ExperimentConfig) -> SessionContext:
        number = self.state.next_session_number()
        session_id = f"session-{number:06d}"
        directory = self.layout.session_dir(number)
        directory.mkdir(parents=True, exist_ok=False)
        starting = self.state.checkpoint()
        starting_checkpoint = str(starting["checkpoint_id"]) if starting else None
        starting_state = self.state.state()
        self.state.create_session(
            number=number,
            session_id=session_id,
            wall_seconds=config.run.wall_seconds,
            starting_checkpoint=starting_checkpoint,
        )
        _atomic_write(directory / "input-config.toml", config.source_bytes)
        write_json(
            directory / "summary.json.gz",
            {
                "schema_version": "mforge.experiment.session.v2",
                "session_number": number,
                "session_id": session_id,
                "start_time": _now(),
                "wall_seconds": config.run.wall_seconds,
                "starting_checkpoint": starting_checkpoint,
                "starting_state": starting_state,
            },
        )
        for filename in ("events.jsonl.gz", "stdout.log", "stderr.log"):
            (directory / filename).touch()
        # Make the append-only session destination visible in the experiment
        # manifest before any adapter/provider work can begin.  This keeps a
        # process interruption immediately after session creation recoverable.
        self.layout.write_artifact_manifest()
        return SessionContext(
            number,
            session_id,
            directory,
            config.run.wall_seconds,
            _now(),
            starting_checkpoint,
            starting_state,
        )

    def event(self, session: SessionContext, event_type: str, **payload: Any) -> None:
        raw_key = payload.get("idempotency_key")
        storage_key = (
            f"{event_type}:{raw_key}"
            if isinstance(raw_key, str) and raw_key
            else None
        )
        if storage_key is None and event_type == "evaluation_progress":
            progress_scope = {
                "session_id": session.session_id,
                "generation": payload.get("generation"),
                "slot": payload.get("slot"),
                "evaluation_id": payload.get("evaluation_id"),
                "candidate_id": payload.get("candidate_id"),
                "phase": payload.get("phase", payload.get("pass")),
                "state": "completed" if payload.get("pass_completed") is True else "running",
            }
            storage_key = f"{event_type}:{_canonical(progress_scope).decode('utf-8')}"
        if storage_key is not None and (
            self.state.event_exists(storage_key)
            or self.state.event_exists(str(raw_key))
        ):
            return
        # Count provider work at the event boundary as well as in the durable
        # provider-turn ledger.  A Ctrl-C can arrive while the provider is
        # blocked, before a terminal turn artifact exists; recording the start
        # here keeps the finished session honest about that attempted work.
        # Retained turns are replayed for resume bookkeeping and must not be
        # charged to the new session.
        if event_type == "provider_turn_started":
            session.provider_turns_attempted += 1
        elif event_type in {"provider_turn_completed", "provider_turn_failed"}:
            if payload.get("retained") is True:
                session.provider_turns_attempted = max(0, session.provider_turns_attempted - 1)
            elif event_type == "provider_turn_completed":
                session.provider_turns_completed += 1
        event = {
            "schema_version": "mforge.experiment.events.v3",
            "event_id": f"{session.session_id}:{session.event_count:08d}",
            "run_id": session.session_id,
            "timestamp": _now(),
            "event_type": event_type,
            **payload,
        }
        session.event_count += 1
        self.state.write_event(
            event_type,
            payload,
            session_id=session.session_id,
            idempotency_key=storage_key,
            timestamp=str(event["timestamp"]),
        )
        _append_gzip_member(
            session.directory / "events.jsonl.gz",
            _canonical(event) + b"\n",
        )
        # The manifest is reconciled at session boundaries.  Rebuilding it for
        # every heartbeat would recursively hash the complete artifact tree and
        # can dominate CPU time during a long provider turn.

    def log(self, session: SessionContext, stream: str, text: str) -> None:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        with (session.directory / f"{stream}.log").open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def finish(
        self,
        session: SessionContext,
        *,
        state: str,
        stop_reason: str,
        exit_status: int = 0,
        summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        session.stop_reason = stop_reason
        latest_checkpoint = self.state.checkpoint()
        session.ending_checkpoint = session.ending_checkpoint or (
            str(latest_checkpoint["checkpoint_id"]) if latest_checkpoint is not None else None
        )
        durable = self.state.session(session.session_id) or {}
        session.provider_turns_attempted = max(
            session.provider_turns_attempted,
            int(durable.get("provider_turns_attempted", 0)),
        )
        session.provider_turns_completed = max(
            session.provider_turns_completed,
            int(durable.get("provider_turns_completed", 0)),
        )
        session.token_usage_delta = max(
            session.token_usage_delta,
            int(durable.get("token_usage_delta", 0)),
        )
        cumulative_tokens = int(self.state.cumulative()["total_tokens"])
        result = session.as_dict(
            ending_state=state,
            exit_status=exit_status,
            cumulative_tokens=cumulative_tokens,
        )
        result["stop_reason"] = stop_reason
        if summary:
            result.update(dict(summary))
        write_json(session.directory / "summary.json.gz", result)
        self.state.finish_session(
            session.session_id,
            status=state,
            ending_state=state,
            ending_checkpoint=session.ending_checkpoint,
            provider_turns_attempted=session.provider_turns_attempted,
            provider_turns_completed=session.provider_turns_completed,
            candidates_created=session.candidates_created,
            evaluations_completed=session.evaluations_completed,
            token_usage_delta=session.token_usage_delta,
            cumulative_tokens=cumulative_tokens,
            runtime_seconds=session.elapsed_seconds,
            stop_reason=stop_reason,
            exit_status=exit_status,
            summary=result,
        )
        self.layout.write_artifact_manifest()
        return result


SessionStore = SessionManager


__all__ = ["SessionContext", "SessionManager", "SessionStore"]
