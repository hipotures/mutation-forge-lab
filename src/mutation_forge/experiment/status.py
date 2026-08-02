"""Read-only operational status views for experiment workspaces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .checkpoints import CheckpointIntegrityError, CheckpointStore
from .config import ExperimentConfig, load_experiment_config
from .layout import ExperimentLayout, WorkspaceError
from .lock import LockError, load_lock, verify_lock
from .state import ExperimentStateStore, StateError, process_alive

STATUS_SCHEMA_VERSION = "mforge.experiment.status.v1"


def _not_created(config: ExperimentConfig, layout: ExperimentLayout) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "exp_id": config.exp_id,
        "state": "not_created",
        "resumable": True,
        "workspace": str(layout.root),
        "session_id": None,
        "generation": 0,
        "max_generations": config.search.max_generations,
        "completed_slots": 0,
        "slot_count": config.search.population_size,
        "candidate_count": 0,
        "unique_candidate_count": 0,
        "best_program_id": None,
        "best_primary_metric": None,
        "provider_turns": 0,
        "total_tokens": 0,
        "compute_seconds": 0,
        "last_checkpoint": None,
        "last_stop_reason": None,
        "last_error": None,
    }


def experiment_status(config_path: str | Path = "experiment.toml") -> dict[str, Any]:
    """Return status without invoking a provider, scorer, oracle, or evaluator."""

    config = load_experiment_config(config_path)
    layout = ExperimentLayout.from_config(config)
    if not layout.root.exists():
        return _not_created(config, layout)
    try:
        layout.verify_root()
    except WorkspaceError as error:
        return {
            **_not_created(config, layout),
            "state": "failed",
            "resumable": False,
            "last_error": str(error),
        }
    try:
        lock = load_lock(layout.lock)
        verify_lock(lock, config, layout)
    except (LockError, OSError, ValueError) as error:
        return {
            **_not_created(config, layout),
            "state": "failed",
            "resumable": False,
            "last_error": str(error),
        }
    try:
        state = ExperimentStateStore(layout.state)
    except StateError as error:
        return {
            **_not_created(config, layout),
            "state": "failed",
            "resumable": False,
            "last_error": str(error),
        }
    try:
        checkpoints = CheckpointStore(layout.checkpoints)
        try:
            checkpoint = checkpoints.latest()
            checkpoint_error = None
        except CheckpointIntegrityError as error:
            checkpoint = None
            checkpoint_error = str(error)
        experiment = state.experiment()
        db_checkpoint = state.checkpoint()
        if checkpoint_error is None and (
            (checkpoint is None) != (db_checkpoint is None)
            or (
                checkpoint is not None
                and db_checkpoint is not None
                and (
                    db_checkpoint.get("checkpoint_id") != checkpoint.get("checkpoint_id")
                    or db_checkpoint.get("sha256") != checkpoint.get("checkpoint_sha256")
                )
            )
        ):
            checkpoint_error = "state database checkpoint does not match checkpoint chain"
        current_state = state.state()
        owner = state.owner()
        if (
            current_state == "running"
            and owner is not None
            and not process_alive(int(owner["pid"]))
        ):
            current_state = "interrupted"
        if checkpoint_error is not None:
            current_state = "failed"
        counts = state.counts()
        session = state.session()
        current = state.cumulative()
        generation = int(checkpoint.get("generation", 0)) if checkpoint else 0
        completed_slots = checkpoint.get("completed_slots", 0) if checkpoint else 0
        if not isinstance(completed_slots, int):
            completed_slots = len(completed_slots) if isinstance(completed_slots, list) else 0
        best_id, best_metric = _best_candidate(state)
        last_error = checkpoint_error or state.latest_error()
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "exp_id": config.exp_id,
            "state": current_state,
            "resumable": current_state in {"idle", "interrupted"},
            "workspace": str(layout.root),
            "session_id": session.get("session_id") if session else None,
            "generation": generation,
            "max_generations": config.search.max_generations,
            "completed_slots": completed_slots,
            "slot_count": config.search.population_size,
            "candidate_count": counts["candidate_count"],
            "unique_candidate_count": counts["unique_candidate_count"],
            "best_program_id": best_id,
            "best_primary_metric": best_metric,
            "provider_turns": counts["provider_turns"],
            "total_tokens": int(current["total_tokens"]),
            "compute_seconds": float(current["compute_seconds"]),
            "last_checkpoint": checkpoint.get("checkpoint_id")
            if checkpoint
            else experiment.get("current_checkpoint"),
            "last_stop_reason": experiment.get("terminal_stop_reason")
            or (session or {}).get("stop_reason"),
            "last_error": last_error,
        }
    finally:
        state.close()


def _best_candidate(state: ExperimentStateStore) -> tuple[str | None, float | int | None]:
    rows = state.connection.execute(
        "SELECT candidate_id,metadata_json FROM candidates "
        "WHERE status NOT IN ('duplicate','invalid')"
    ).fetchall()
    best_id: str | None = None
    best_value: float | int | None = None
    for row in rows:
        try:
            metadata = json.loads(str(row["metadata_json"]))
        except json.JSONDecodeError:
            continue
        if not isinstance(metadata, Mapping):
            continue
        metric = metadata.get("best_primary_metric", metadata.get("pooled_auc"))
        if not isinstance(metric, int | float) or isinstance(metric, bool):
            continue
        if best_value is None or float(metric) > float(best_value):
            best_id = str(row["candidate_id"])
            best_value = metric
    return best_id, best_value


def render_status(status: Mapping[str, Any], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(dict(status), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    state = str(status.get("state", "unknown"))
    if state == "not_created":
        return (
            f"Experiment: {status['exp_id']}\n"
            "State: not created\n"
            f"Workspace: {status['workspace']}\n"
            "Next action: mforge experiment run"
        )
    state_label = f"{state}, resumable" if status.get("resumable") else state
    lines = [f"Experiment: {status['exp_id']}", f"State: {state_label}"]
    if status.get("generation") is not None:
        lines.append(
            f"Progress: generation {status['generation']}/{status['max_generations']}, "
            f"{status['unique_candidate_count']} unique candidates"
        )
    if status.get("best_program_id"):
        lines.append(
            f"Best: {status['best_program_id']}, primary metric {status['best_primary_metric']}"
        )
    lines.append(f"Usage: {status['provider_turns']} model turns, {status['total_tokens']} tokens")
    if status.get("last_stop_reason"):
        lines.append(f"Last stop: {status['last_stop_reason']}")
    if status.get("last_error"):
        lines.append(f"Last error: {status['last_error']}")
    return "\n".join(lines)


status = experiment_status
get_status = experiment_status

__all__ = [
    "STATUS_SCHEMA_VERSION",
    "experiment_status",
    "get_status",
    "render_status",
    "status",
]
