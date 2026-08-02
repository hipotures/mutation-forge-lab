"""Create-or-continue service for experiment workspaces."""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .checkpoints import CheckpointStore
from .config import ExperimentConfig, load_experiment_config
from .layout import ExperimentLayout, WorkspaceError
from .lock import LockError, build_lock, load_lock, verify_lock
from .sessions import SessionContext, SessionManager
from .state import ExperimentStateStore, StateError


class ExperimentAdapter(Protocol):
    """Boundary implemented by the existing generation/evaluation engine."""

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class SessionBudget:
    wall_seconds: float
    started_monotonic: float

    @property
    def deadline(self) -> float:
        return self.started_monotonic + self.wall_seconds

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.deadline


class NullExperimentAdapter:
    """Safe adapter used until a scientific engine is explicitly configured.

    It writes a durable checkpoint and returns at the session boundary.  This
    keeps workspace and continuation semantics testable without accidentally
    issuing a model call from the generic layer.
    """

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any]:
        del config, layout, state, session
        return {"state": "idle", "stop_reason": "budget_exhausted"}


class LegacyStage4Adapter:
    """Optional adapter for a configured legacy Stage 4 campaign.

    The generic experiment layer never invents a stage path.  A caller may
    provide ``legacy_stage4_config`` in the TOML for an explicit migration
    bridge; normal experiment configurations use ``NullExperimentAdapter`` or
    a real adapter supplied by the application.
    """

    def __init__(self, *, provider: Any | None = None) -> None:
        self.provider = provider

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any]:
        legacy = config.raw.get("legacy_stage4_config")
        if not isinstance(legacy, str) or not legacy:
            return NullExperimentAdapter().run(config, layout, state, session)
        from mutation_forge.stage4.commands import evolve

        def observe(event: Mapping[str, Any]) -> None:
            state.write_event(
                str(event.get("event", "adapter_event")), event, session_id=session.session_id
            )

        result = evolve(
            (config.source_dir / legacy).resolve(),
            provider=self.provider,
            concurrency=config.model.concurrency,
            resume=True,
            observer=observe,
        )
        status = "completed" if result.get("status") == "completed" else "idle"
        return {
            "state": status,
            "stop_reason": str(result.get("stop_reason", "adapter_complete")),
            "result": result,
        }


class ExperimentService:
    def __init__(self, *, adapter: ExperimentAdapter | None = None) -> None:
        self.adapter = adapter or LegacyStage4Adapter()

    def run(self, config_path: str | Path = "experiment.toml") -> dict[str, Any]:
        config = load_experiment_config(config_path)
        layout = ExperimentLayout.from_config(config)
        created = not layout.root.exists()
        if created:
            lock = build_lock(config, layout)
            lock_hash = str(lock["immutable_config_sha256"])
            layout.initialize_atomic(
                config,
                lock_payload=lock,
                state_initializer=lambda state_path: ExperimentStateStore.initialize(
                    state_path,
                    exp_id=config.exp_id,
                    lock_hash=lock_hash,
                    root=layout.root,
                ),
            )
        else:
            layout.verify_root()
            lock = load_lock(layout.lock)
            verify_lock(lock, config, layout)
            self._verify_root_config(layout, config, lock)

        state = ExperimentStateStore(layout.state)
        checkpoints = CheckpointStore(layout.checkpoints)
        try:
            checkpoints.verify()
            self._verify_state_checkpoint(state, checkpoints)
            current_state = state.state()
            if current_state == "completed":
                return self._record_completed_session(config, layout, state)

            number = state.next_session_number()
            session_id = f"session-{number:06d}"
            state.acquire_owner(exp_id=config.exp_id, session_id=session_id)
            manager = SessionManager(layout, state)
            session: SessionContext | None = None
            try:
                session = manager.start(config)
                starting_checkpoint = checkpoints.latest()
                if starting_checkpoint is None:
                    first = checkpoints.save(
                        {
                            "experiment_id": config.exp_id,
                            "state": "running",
                            "generation": 0,
                            "completed_slots": [],
                            "provider_turns": [],
                            "evaluations": [],
                        }
                    )
                    state.record_checkpoint(
                        sequence=int(first["sequence"]),
                        checkpoint_id=str(first["checkpoint_id"]),
                        path=str(layout.checkpoint_path(int(first["sequence"]))),
                        sha256=str(first["checkpoint_sha256"]),
                        generation=0,
                        completed_slots=0,
                    )
                    session.starting_checkpoint = str(first["checkpoint_id"])
                else:
                    session.starting_checkpoint = str(starting_checkpoint["checkpoint_id"])
                manager.event(session, "session_started", checkpoint=session.starting_checkpoint)
                result = self._invoke_adapter(config, layout, state, session)
                outcome = self._normalize_outcome(result, session)
                final_checkpoint = checkpoints.save(
                    {
                        "experiment_id": config.exp_id,
                        "state": outcome["state"],
                        "generation": int(outcome.get("generation", 0)),
                        "completed_slots": list(outcome.get("completed_slots", [])),
                        "provider_turns": list(outcome.get("provider_turns", [])),
                        "evaluations": list(outcome.get("evaluations", [])),
                        "result": dict(outcome),
                    }
                )
                state.record_checkpoint(
                    sequence=int(final_checkpoint["sequence"]),
                    checkpoint_id=str(final_checkpoint["checkpoint_id"]),
                    path=str(layout.checkpoint_path(int(final_checkpoint["sequence"]))),
                    sha256=str(final_checkpoint["checkpoint_sha256"]),
                    generation=int(outcome.get("generation", 0)),
                    completed_slots=len(cast(list[Any], outcome.get("completed_slots", []))),
                )
                session.ending_checkpoint = str(final_checkpoint["checkpoint_id"])
                manager.event(session, "checkpoint_written", checkpoint=session.ending_checkpoint)
                state.set_state(
                    outcome["state"],
                    error=outcome.get("last_error"),
                    stop_reason=outcome.get("stop_reason"),
                    checkpoint=session.ending_checkpoint,
                )
                session_summary = manager.finish(
                    session,
                    state=outcome["state"],
                    stop_reason=str(outcome.get("stop_reason", "budget_exhausted")),
                    summary={**outcome, "result": outcome.get("result")},
                )
                layout.write_artifact_manifest()
                return self._run_result(config, layout, state, session_summary, outcome)
            except BaseException as error:
                if session is not None:
                    session.stop_reason = (
                        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    )
                    with suppress(Exception):
                        manager.finish(
                            session,
                            state="interrupted"
                            if isinstance(error, KeyboardInterrupt)
                            else "failed",
                            stop_reason=session.stop_reason,
                            exit_status=130 if isinstance(error, KeyboardInterrupt) else 1,
                            summary={"last_error": f"{type(error).__name__}: {error}"},
                        )
                    with suppress(Exception):
                        layout.write_artifact_manifest()
                state.set_state(
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=f"{type(error).__name__}: {error}",
                )
                raise
            finally:
                state.release_owner(session_id)
        finally:
            state.close()

    @staticmethod
    def _verify_state_checkpoint(state: ExperimentStateStore, checkpoints: CheckpointStore) -> None:
        file_checkpoint = checkpoints.latest()
        db_checkpoint = state.checkpoint()
        if file_checkpoint is None and db_checkpoint is None:
            return
        if file_checkpoint is None or db_checkpoint is None:
            raise StateError("state/checkpoint indexes disagree")
        if (
            db_checkpoint.get("checkpoint_id") != file_checkpoint.get("checkpoint_id")
            or db_checkpoint.get("sha256") != file_checkpoint.get("checkpoint_sha256")
            or Path(str(db_checkpoint.get("path", ""))).resolve()
            != checkpoints.root / f"checkpoint-{int(file_checkpoint['sequence']):012d}.json"
        ):
            raise StateError("state database checkpoint digest does not match checkpoint chain")

    @staticmethod
    def _verify_root_config(
        layout: ExperimentLayout,
        config: ExperimentConfig,
        lock: Mapping[str, Any],
    ) -> None:
        try:
            stored = layout.experiment_config.read_bytes()
        except OSError as exc:
            raise WorkspaceError(
                f"cannot read immutable experiment.toml: {layout.experiment_config}"
            ) from exc
        import hashlib

        if hashlib.sha256(stored).hexdigest() != lock.get("source_config_sha256"):
            raise LockError("immutable root experiment.toml does not match experiment.lock.json")
        if stored != config.source_bytes and not _same_immutable_config(stored, config):
            raise LockError("immutable root experiment.toml differs from the locked specification")

    def _invoke_adapter(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any] | None:
        # Adapters from early experiments sometimes accepted an explicit
        # budget.  Support that additive form without changing the core API.
        run = self.adapter.run
        parameters: Mapping[str, inspect.Parameter]
        try:
            parameters = inspect.signature(run).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "budget" in parameters:
            return run(
                config,
                layout,
                state,
                session,
                budget=SessionBudget(config.run.wall_seconds, session.monotonic_started),
            )  # type: ignore[call-arg]
        return run(config, layout, state, session)

    @staticmethod
    def _normalize_outcome(
        result: Mapping[str, Any] | None, session: SessionContext
    ) -> dict[str, Any]:
        outcome = dict(result or {})
        requested_state = str(outcome.get("state", "idle"))
        state = (
            requested_state
            if requested_state in {"idle", "interrupted", "failed", "completed"}
            else "idle"
        )
        if session.budget_exhausted() and state == "running":
            state = "idle"
        outcome["state"] = state
        outcome.setdefault("stop_reason", "budget_exhausted" if state == "idle" else "completed")
        return outcome

    @staticmethod
    def _record_completed_session(
        config: ExperimentConfig, layout: ExperimentLayout, state: ExperimentStateStore
    ) -> dict[str, Any]:
        number = state.next_session_number()
        session_id = f"session-{number:06d}"
        state.acquire_owner(exp_id=config.exp_id, session_id=session_id)
        manager = SessionManager(layout, state)
        try:
            session = manager.start(config)
            latest_checkpoint = state.checkpoint()
            session.ending_checkpoint = (
                str(latest_checkpoint["checkpoint_id"]) if latest_checkpoint is not None else None
            )
            summary = manager.finish(
                session,
                state="completed",
                stop_reason="already_completed",
                summary={"provider_calls": 0, "evaluation_calls": 0},
            )
            layout.write_artifact_manifest()
            return {
                "schema_version": "mforge.experiment.run.v1",
                "status": "completed",
                "exp_id": config.exp_id,
                "state": "completed",
                "workspace": str(layout.root),
                "session_id": session_id,
                "stop_reason": "already_completed",
                "checkpoint": session.ending_checkpoint,
                "provider_calls": 0,
                "evaluation_calls": 0,
                "session": summary,
            }
        finally:
            state.release_owner(session_id)

    @staticmethod
    def _run_result(
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "mforge.experiment.run.v1",
            "status": "completed" if outcome.get("state") in {"idle", "completed"} else "failed",
            "exp_id": config.exp_id,
            "state": outcome.get("state"),
            "workspace": str(layout.root),
            "session_id": session.get("session_id"),
            "stop_reason": outcome.get("stop_reason"),
            "checkpoint": session.get("ending_checkpoint"),
            "provider_turns": state.counts().get("provider_turns", 0),
            "evaluation_count": state.counts().get("evaluation_count", 0),
        }
        if "result" in outcome:
            result["result"] = outcome["result"]
        if outcome.get("last_error"):
            result["last_error"] = outcome["last_error"]
        return result


def _same_immutable_config(stored_bytes: bytes, current: ExperimentConfig) -> bool:
    try:
        import tomllib

        raw = tomllib.loads(stored_bytes.decode("utf-8"))
        current_raw = dict(current.raw)
        raw.pop("run", None)
        current_raw.pop("run", None)
        return raw == current_raw
    except (UnicodeError, tomllib.TOMLDecodeError):
        return False


def run_experiment(
    config_path: str | Path = "experiment.toml",
    *,
    adapter: ExperimentAdapter | None = None,
) -> dict[str, Any]:
    return ExperimentService(adapter=adapter).run(config_path)


__all__ = [
    "ExperimentAdapter",
    "ExperimentService",
    "LegacyStage4Adapter",
    "NullExperimentAdapter",
    "SessionBudget",
    "run_experiment",
]
