"""Create-or-continue service for experiment workspaces."""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .checkpoints import CheckpointStore
from .config import ExperimentConfig, load_experiment_config
from .layout import ExperimentLayout, WorkspaceError
from .lock import (
    LockError,
    build_lock,
    load_lock,
    verify_lock,
)
from .native import NativeExperimentAdapter
from .observer import CallbackEventSink, ExperimentEventHub
from .sessions import SessionContext, SessionManager
from .state import ExperimentStateStore, StateError, process_alive


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
    """Explicit test adapter; never selected by the production service."""

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any]:
        del config, layout, state, session
        return {"state": "idle", "stop_reason": "session_wall_seconds"}


class ExperimentService:
    def __init__(
        self,
        *,
        adapter: ExperimentAdapter | None = None,
        event_sinks: Sequence[Any] = (),
        observer: Any | None = None,
        profiling: bool | None = None,
    ) -> None:
        self.adapter = adapter or NativeExperimentAdapter()
        self.event_sinks = list(event_sinks)
        if observer is not None:
            self.event_sinks.append(CallbackEventSink(observer))
        self.profiling = profiling

    def run(self, config_path: str | Path = "experiment.toml") -> dict[str, Any]:
        config = load_experiment_config(config_path)
        layout = ExperimentLayout.from_config(config)
        profiling_enabled = (
            config.run.profiling_enabled if self.profiling is None else bool(self.profiling)
        )
        hub = ExperimentEventHub(
            config.exp_id,
            self.event_sinks,
            profiling_enabled=profiling_enabled,
        )
        hub.emit(
            "preflight_started",
            experiment_id=config.exp_id,
            workspace=str(layout.root),
            run_mode="fresh" if not layout.root.exists() else "continuation",
            state="starting",
            configured_wall_seconds=config.run.wall_seconds,
            profiling_enabled=profiling_enabled,
        )

        def fail_before_session(error: BaseException) -> None:
            with suppress(Exception):
                hub.emit(
                    "experiment_failed",
                    experiment_id=config.exp_id,
                    workspace=str(layout.root),
                    state="failed",
                    stop_reason="preflight_failed",
                    error=f"{type(error).__name__}: {error}",
                )
            with suppress(Exception):
                hub.close()

        created = not layout.root.exists()
        if created:
            try:
                preflight = self._preflight(config)
            except BaseException as error:
                fail_before_session(error)
                raise
            hub.emit(
                "preflight_completed",
                experiment_id=config.exp_id,
                workspace=str(layout.root),
                status="completed",
                preflight=preflight or {},
            )
            try:
                lock = build_lock(config, layout, preflight=preflight)
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
            except BaseException as error:
                fail_before_session(error)
                raise
            hub.emit(
                "workspace_initialized",
                experiment_id=config.exp_id,
                workspace=str(layout.root),
                state="idle",
                run_mode="fresh",
            )
        else:
            try:
                layout.verify_root()
                lock = load_lock(layout.lock)
                verify_lock(lock, config, layout)
                self._verify_root_config(layout, config, lock)
            except BaseException as error:
                fail_before_session(error)
                raise
            hub.emit(
                "preflight_completed",
                experiment_id=config.exp_id,
                workspace=str(layout.root),
                status="completed",
                resumed=True,
            )
            hub.emit(
                "workspace_resumed",
                experiment_id=config.exp_id,
                workspace=str(layout.root),
                state="resumable",
                run_mode="continuation",
            )

        state = ExperimentStateStore(layout.state)
        checkpoints = CheckpointStore(layout.checkpoints)
        try:
            checkpoints.verify()
            self._verify_state_checkpoint(state, checkpoints)
            # A crash may leave new, fsynced files outside the last manifest.
            # Reconciliation accepts only those append-only additions; it
            # first verifies every previously committed digest.
            layout.verify_artifact_manifest(allow_new=True)
            layout.reconcile_artifact_manifest()
            layout.verify_runtime_schemas()
            current_state = state.state()
            experiment = state.experiment()
            last_stop_reason = str(experiment.get("terminal_stop_reason") or "")
            if last_stop_reason == "already_completed":
                last_stop_reason = str(state.latest_meaningful_stop_reason() or "")
            effective_model_turns = config.search.max_model_turns
            if current_state in {"completed", "exhausted", "failed"}:
                result = self._terminal_result(config, layout, state)
                hub.emit(
                    "experiment_completed"
                    if current_state == "completed"
                    else "experiment_failed"
                    if current_state == "failed"
                    else "experiment_exhausted",
                    experiment_id=config.exp_id,
                    workspace=str(layout.root),
                    state=current_state,
                    stop_reason=last_stop_reason,
                    checkpoint=result.get("checkpoint"),
                )
                hub.close()
                return result

            number = state.next_session_number()
            session_id = f"session-{number:06d}"
            state.acquire_owner(exp_id=config.exp_id, session_id=session_id)
            manager = SessionManager(layout, state)
            session: SessionContext | None = None
            try:
                session = manager.start(config)
                hub.attach_session(manager, session)
                starting_checkpoint = checkpoints.latest()
                fresh_session = starting_checkpoint is None
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
                cumulative = state.cumulative()
                counts = state.counts()
                ledger_model_turns = counts.get("provider_turns", 0)
                model_turns_used = ledger_model_turns
                model_turn_counter = getattr(self.adapter, "model_turns_used", None)
                if callable(model_turn_counter):
                    model_turns_used = int(model_turn_counter(layout, state))
                remaining_model_turns = (
                    max(0, effective_model_turns - model_turns_used)
                    if effective_model_turns is not None
                    else None
                )
                hourly_usage = state.hourly_token_usage(
                    config.run.max_total_tokens_per_hour,
                    backfill=True,
                )
                hub.emit(
                    "session_started",
                    experiment_id=config.exp_id,
                    workspace=str(layout.root),
                    session_id=session.session_id,
                    session_number=session.number,
                    run_mode=("fresh" if fresh_session else "continuation"),
                    state="running",
                    checkpoint=session.starting_checkpoint,
                    elapsed_seconds=session.elapsed_seconds,
                    configured_wall_seconds=session.wall_seconds,
                    remaining_seconds=max(0.0, session.deadline - time.monotonic()),
                    model=config.model.name,
                    effort=config.model.effort,
                    configured_concurrency=config.model.concurrency,
                    effective_concurrency=config.model.concurrency,
                    # The native evaluator pool is controlled by
                    # ``resources.thread_count``.  Keep the broader resource
                    # reservation visible separately so the dashboard never
                    # reports the reservation (often 8) as active evaluators.
                    worker_count=max(1, config.resources.thread_count),
                    resource_worker_count=config.resources.workers,
                    active_workers=0,
                    population_size=config.search.population_size,
                    generation_limit=config.search.max_generations,
                    max_model_turns=effective_model_turns,
                    remaining_model_turns=remaining_model_turns,
                    model_turns_used=model_turns_used,
                    reserved_model_turns=max(
                        0,
                        model_turns_used - ledger_model_turns,
                    ),
                    cumulative_provider_turns=ledger_model_turns,
                    cumulative_evaluations=counts.get("evaluation_count", 0),
                    cumulative_candidates=counts.get("candidate_count", 0),
                    archive_size=counts.get("candidate_count", 0),
                    cumulative_tokens=cumulative.get("total_tokens", 0),
                    **hourly_usage,
                    usage=state.token_usage(),
                    session_usage={
                        "inputTokens": 0,
                        "cachedInputTokens": 0,
                        "cacheWriteInputTokens": 0,
                        "outputTokens": 0,
                        "reasoningOutputTokens": 0,
                        "totalTokens": 0,
                        "quality": "unknown",
                    },
                )
                adapter_result = self._invoke_adapter(
                    config,
                    layout,
                    state,
                    session,
                    observer=hub,
                    profiling=profiling_enabled,
                    effective_max_model_turns=effective_model_turns,
                )
                outcome = self._normalize_outcome(adapter_result, session)
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
                session.ending_checkpoint = str(final_checkpoint["checkpoint_id"])
                state.set_state(
                    outcome["state"],
                    error=outcome.get("last_error"),
                    stop_reason=outcome.get("stop_reason"),
                    checkpoint=session.ending_checkpoint,
                )
                state.record_checkpoint(
                    sequence=int(final_checkpoint["sequence"]),
                    checkpoint_id=str(final_checkpoint["checkpoint_id"]),
                    path=str(layout.checkpoint_path(int(final_checkpoint["sequence"]))),
                    sha256=str(final_checkpoint["checkpoint_sha256"]),
                    generation=int(outcome.get("generation", 0)),
                    completed_slots=len(cast(list[Any], outcome.get("completed_slots", []))),
                )
                hub.emit(
                    "checkpoint_written",
                    checkpoint=session.ending_checkpoint,
                    generation=outcome.get("generation", 0),
                    completed_slots=len(cast(list[Any], outcome.get("completed_slots", []))),
                    state=outcome["state"],
                    durable=True,
                )
                session_summary = manager.finish(
                    session,
                    state=outcome["state"],
                    stop_reason=str(outcome.get("stop_reason", "budget_exhausted")),
                    exit_status=(1 if outcome.get("stop_reason") == "infrastructure_failed" else 0),
                    summary={**outcome, "result": outcome.get("result")},
                )
                counterexample = outcome.get("counterexample")
                if (
                    outcome["state"] == "completed"
                    and outcome.get("stop_reason") == "counterexample_verified"
                    and isinstance(counterexample, Mapping)
                ):
                    hub.emit(
                        "counterexample_verified",
                        candidate_id=counterexample.get("candidate_id"),
                        certificate=counterexample.get("certificate_path"),
                        certificate_sha256=counterexample.get("certificate_sha256"),
                        stop_reason="counterexample_verified",
                        checkpoint=session.ending_checkpoint,
                        idempotency_key=(f"{counterexample.get('candidate_id')}:verified"),
                    )
                if outcome.get("stop_reason") == "hourly_token_limit":
                    hub.emit(
                        "hourly_token_session_stopped",
                        state=outcome["state"],
                        stop_reason="hourly_token_limit",
                        checkpoint=session.ending_checkpoint,
                        **state.hourly_token_usage(
                            config.run.max_total_tokens_per_hour
                        ),
                    )
                hub.emit(
                    "budget_boundary_reached"
                    if outcome["state"] in {"idle", "paused", "interrupted"}
                    else "experiment_failed"
                    if outcome["state"] == "failed"
                    else "experiment_exhausted"
                    if outcome["state"] == "exhausted"
                    else "experiment_completed",
                    experiment_id=config.exp_id,
                    workspace=str(layout.root),
                    session_id=session.session_id,
                    state=outcome["state"],
                    stop_reason=outcome.get("stop_reason"),
                    checkpoint=session.ending_checkpoint,
                    elapsed_seconds=session.elapsed_seconds,
                    remaining_seconds=max(0.0, session.deadline - time.monotonic()),
                    provider_turns_attempted=session.provider_turns_attempted,
                    provider_turns_completed=session.provider_turns_completed,
                    evaluations_completed=session.evaluations_completed,
                    token_usage_delta=session_summary.get("token_usage_delta", 0),
                    cumulative_tokens=session_summary.get("cumulative_tokens", 0),
                    **state.hourly_token_usage(
                        config.run.max_total_tokens_per_hour
                    ),
                    recovered_work=outcome.get("recovered_work"),
                )
                layout.reconcile_artifact_manifest()
                final_result = self._run_result(
                    config,
                    layout,
                    state,
                    session_summary,
                    outcome,
                    effective_model_turns=effective_model_turns,
                )
                hub.close()
                return final_result
            except BaseException as error:
                if session is not None:
                    session.stop_reason = (
                        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    )
                    with suppress(Exception):
                        hub.emit(
                            "experiment_interrupted"
                            if isinstance(error, KeyboardInterrupt)
                            else "experiment_failed",
                            experiment_id=config.exp_id,
                            workspace=str(layout.root),
                            session_id=session.session_id,
                            state="interrupted"
                            if isinstance(error, KeyboardInterrupt)
                            else "failed",
                            stop_reason="interrupted"
                            if isinstance(error, KeyboardInterrupt)
                            else "failed",
                            error=f"{type(error).__name__}: {error}",
                            elapsed_seconds=session.elapsed_seconds,
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
                        layout.reconcile_artifact_manifest()
                # Keep the durable stop reason written above.  Calling
                # set_state without it used to clear ``terminal_stop_reason``
                # after Ctrl+C, so status could no longer explain why the
                # resumable session stopped.
                current_checkpoint = state.experiment().get("current_checkpoint")
                state.set_state(
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=f"{type(error).__name__}: {error}",
                    stop_reason=session.stop_reason if session is not None else None,
                    checkpoint=(str(current_checkpoint) if current_checkpoint else None),
                )
                hub.close()
                raise
            finally:
                state.release_owner(session_id)
        finally:
            state.close()
            with suppress(Exception):
                hub.close()

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
        _config: ExperimentConfig,
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

    def _invoke_adapter(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
        *,
        observer: Any | None = None,
        profiling: bool | None = None,
        effective_max_model_turns: int | None = None,
    ) -> Mapping[str, Any] | None:
        # Adapters from early experiments sometimes accepted an explicit
        # budget.  Support that additive form without changing the core API.
        run = self.adapter.run
        parameters: Mapping[str, inspect.Parameter]
        try:
            parameters = inspect.signature(run).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {}
        if "observer" in parameters:
            kwargs["observer"] = observer
        elif "event_callback" in parameters:
            kwargs["event_callback"] = observer
        if "profiling" in parameters:
            kwargs["profiling"] = profiling
        if "effective_max_model_turns" in parameters:
            kwargs["effective_max_model_turns"] = effective_max_model_turns
        if "budget" in parameters:
            kwargs["budget"] = SessionBudget(config.run.wall_seconds, session.monotonic_started)
            return run(
                config,
                layout,
                state,
                session,
                **kwargs,
            )
        return run(config, layout, state, session, **kwargs)

    def _preflight(self, config: ExperimentConfig) -> Mapping[str, Any] | None:
        preflight = getattr(self.adapter, "preflight", None)
        if not callable(preflight):
            return None
        value = preflight(config)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise WorkspaceError("experiment adapter preflight returned a non-object result")
        return value

    @staticmethod
    def _normalize_outcome(
        result: Mapping[str, Any] | None, session: SessionContext
    ) -> dict[str, Any]:
        outcome = dict(result or {})
        requested_state = str(outcome.get("state", "idle"))
        valid_states = {
            "running",
            "idle",
            "paused",
            "interrupted",
            "failed",
            "exhausted",
            "completed",
        }
        if requested_state not in valid_states:
            raise StateError(f"adapter returned invalid experiment state: {requested_state!r}")
        state = requested_state
        if session.budget_exhausted() and state == "running":
            state = "idle"
        outcome["state"] = state
        # Checkpoint fields are append-only identity lists.  Adapters may
        # expose convenient numeric counters for their result envelope; do
        # not let that presentation detail corrupt checkpoint serialization.
        for field in ("completed_slots", "provider_turns", "evaluations"):
            if not isinstance(outcome.get(field), list):
                outcome[field] = []
        if "stop_reason" not in outcome or not isinstance(outcome.get("stop_reason"), str):
            defaults = {
                "idle": "session_wall_seconds",
                "exhausted": "generation_limit",
                "interrupted": "interrupted",
                "paused": "paused",
                "failed": "failed",
            }
            if state == "completed":
                raise StateError(
                    "adapter must provide an explicit terminal stop reason for COMPLETED"
                )
            outcome["stop_reason"] = defaults.get(state, state)
        if state == "completed" and outcome["stop_reason"] not in {
            "counterexample_verified",
            "operator_final_stop",
        }:
            raise StateError(
                "COMPLETED requires counterexample_verified or operator_final_stop"
            )
        return outcome

    @staticmethod
    def _terminal_result(
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
    ) -> dict[str, Any]:
        experiment = state.experiment()
        terminal_state = state.state()
        session = state.session() or {}
        checkpoint = state.checkpoint()
        return {
            "schema_version": "mforge.experiment.run.v2",
            "status": terminal_state,
            "exp_id": config.exp_id,
            "state": terminal_state,
            "workspace": str(layout.root),
            "session_id": session.get("session_id"),
            "stop_reason": experiment.get("terminal_stop_reason"),
            "checkpoint": (checkpoint.get("checkpoint_id") if checkpoint is not None else None),
            "provider_calls": 0,
            "evaluation_calls": 0,
            "session": dict(session),
        }

    @staticmethod
    def _run_result(
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: Mapping[str, Any],
        outcome: Mapping[str, Any],
        *,
        effective_model_turns: int | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "mforge.experiment.run.v2",
            "status": (
                "failed"
                if outcome.get("state") == "failed"
                else "completed"
                if outcome.get("state") == "completed"
                else "exhausted"
                if outcome.get("state") == "exhausted"
                else str(outcome.get("state", "idle"))
            ),
            "exp_id": config.exp_id,
            "state": outcome.get("state"),
            "workspace": str(layout.root),
            "session_id": session.get("session_id"),
            "stop_reason": outcome.get("stop_reason"),
            "checkpoint": session.get("ending_checkpoint"),
            "provider_turns": state.counts().get("provider_turns", 0),
            "evaluation_count": state.counts().get("evaluation_count", 0),
            "effective_model_turns": effective_model_turns,
        }
        if "result" in outcome:
            result["result"] = outcome["result"]
        result["session"] = dict(session)
        for field in (
            "timing_profile",
            "deep_operator_profile",
            "deep_score_profile",
            "recovered_work",
            "ir",
        ):
            if field in outcome:
                result[field] = outcome[field]
        if outcome.get("last_error"):
            result["last_error"] = outcome["last_error"]
        return result


def run_experiment(
    config_path: str | Path = "experiment.toml",
    *,
    adapter: ExperimentAdapter | None = None,
    event_sinks: Sequence[Any] = (),
    observer: Any | None = None,
    profiling: bool | None = None,
) -> dict[str, Any]:
    return ExperimentService(
        adapter=adapter,
        event_sinks=event_sinks,
        observer=observer,
        profiling=profiling,
    ).run(config_path)


def final_stop_experiment(
    config_path: str | Path = "experiment.toml",
) -> dict[str, Any]:
    """Persist an explicit non-resumable operator stop without running work."""

    config = load_experiment_config(config_path)
    layout = ExperimentLayout.from_config(config)
    layout.verify_root()
    lock = load_lock(layout.lock)
    verify_lock(lock, config, layout)
    layout.verify_artifact_manifest(allow_new=True)
    layout.verify_runtime_schemas()
    state = ExperimentStateStore(layout.state)
    try:
        owner = state.owner()
        if owner is not None and process_alive(int(owner["pid"])):
            raise WorkspaceError(
                "experiment is active; interrupt the running session before final stop"
            )
        if state.state() in {"completed", "failed", "exhausted"}:
            return {
                "state": state.state(),
                "stop_reason": state.experiment().get("terminal_stop_reason"),
                "changed": False,
            }
        checkpoints = CheckpointStore(layout.checkpoints)
        latest = checkpoints.latest()
        generation = int(latest.get("generation", 0)) if latest else 0
        checkpoint = checkpoints.save(
            {
                "experiment_id": config.exp_id,
                "state": "completed",
                "generation": generation,
                "completed_slots": [],
                "provider_turns": [],
                "evaluations": [],
                "result": {
                    "state": "completed",
                    "stop_reason": "operator_final_stop",
                },
            }
        )
        checkpoint_id = str(checkpoint["checkpoint_id"])
        state.set_state(
            "completed",
            stop_reason="operator_final_stop",
            checkpoint=checkpoint_id,
        )
        state.record_checkpoint(
            sequence=int(checkpoint["sequence"]),
            checkpoint_id=checkpoint_id,
            path=str(layout.checkpoint_path(int(checkpoint["sequence"]))),
            sha256=str(checkpoint["checkpoint_sha256"]),
            generation=generation,
            completed_slots=0,
        )
        state.write_event(
            "experiment_completed",
            {
                "schema_version": "mforge.experiment.events.v2",
                "state": "completed",
                "stop_reason": "operator_final_stop",
                "checkpoint": checkpoint_id,
                "idempotency_key": f"{config.exp_id}:operator-final-stop",
            },
        )
        layout.reconcile_artifact_manifest()
        return {
            "state": "completed",
            "stop_reason": "operator_final_stop",
            "checkpoint": checkpoint_id,
            "changed": True,
        }
    finally:
        state.close()


__all__ = [
    "ExperimentAdapter",
    "ExperimentService",
    "NullExperimentAdapter",
    "SessionBudget",
    "final_stop_experiment",
    "run_experiment",
]
