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
        "evaluation_count": 0,
        "best_program_id": None,
        "best_primary_metric": None,
        "winner_source": None,
        "ranked_candidates": [],
        "artifacts": {},
        "provider_turns": 0,
        "total_tokens": 0,
        "token_usage": {
            "inputTokens": 0,
            "cachedInputTokens": 0,
            "outputTokens": 0,
            "reasoningOutputTokens": 0,
            "totalTokens": 0,
            "quality": "unknown",
            "chargedFailedTurns": 0,
        },
        "ir": None,
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
        layout.verify_artifact_manifest(allow_new=True)
    except WorkspaceError as error:
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
        ranked_candidates = _ranked_candidates(state)
        best = next(
            (candidate for candidate in ranked_candidates if candidate["metric"] is not None),
            None,
        )
        best_id = str(best["candidate_id"]) if best is not None else None
        best_metric = best["metric"] if best is not None else None
        token_usage = state.token_usage()
        last_error = checkpoint_error or state.latest_error()
        session_metrics: dict[str, Any] = {}
        if session is not None:
            try:
                parsed = json.loads(str(session.get("summary_json", "{}")))
            except (TypeError, json.JSONDecodeError):
                parsed = {}
            if isinstance(parsed, Mapping):
                session_metrics = dict(parsed)
        session_id = str(session["session_id"]) if session is not None else None
        artifacts: dict[str, str] = {}
        native_checkpoint = layout.artifacts / "native-generation-checkpoint.json"
        if native_checkpoint.is_file():
            artifacts["generation_checkpoint"] = str(native_checkpoint)
        if session_id is not None:
            session_summary = layout.sessions / session_id / "summary.json"
            if session_summary.is_file():
                artifacts["session_summary"] = str(session_summary)
        if layout.archive.is_dir():
            artifacts["candidate_archive"] = str(layout.archive)
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "exp_id": config.exp_id,
            "state": current_state,
            "resumable": current_state in {"idle", "interrupted"},
            "workspace": str(layout.root),
            "session_id": session_id,
            "generation": generation,
            "max_generations": config.search.max_generations,
            "completed_slots": completed_slots,
            "slot_count": config.search.population_size,
            "candidate_count": counts["candidate_count"],
            "unique_candidate_count": counts["unique_candidate_count"],
            "evaluation_count": counts["evaluation_count"],
            "best_program_id": best_id,
            "best_primary_metric": best_metric,
            "winner_source": best.get("source_path") if best is not None else None,
            "ranked_candidates": ranked_candidates,
            "artifacts": artifacts,
            "provider_turns": counts["provider_turns"],
            "total_tokens": int(current["total_tokens"]),
            "token_usage": token_usage,
            "ir": session_metrics.get("ir"),
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


def _candidate_metric(metadata: Mapping[str, Any]) -> float | int | None:
    search_metrics = metadata.get("search_metrics")
    nested = search_metrics if isinstance(search_metrics, Mapping) else {}
    nested_metric = nested.get("best_primary_metric", nested.get("pooled_auc"))
    if nested_metric is None:
        nested_metric = nested.get("pooled_median_auc")
    metric = metadata.get("best_primary_metric", metadata.get("pooled_auc", nested_metric))
    return (
        metric
        if isinstance(metric, int | float) and not isinstance(metric, bool)
        else None
    )


def _ranked_candidates(state: ExperimentStateStore) -> list[dict[str, Any]]:
    rows = state.connection.execute(
        "SELECT candidate_id,archive_path,generation,slot,status,metadata_json "
        "FROM candidates WHERE status NOT IN ('duplicate','invalid')"
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        try:
            metadata = json.loads(str(row["metadata_json"]))
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        candidates.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "metric": _candidate_metric(metadata),
                "generation": row["generation"],
                "slot": row["slot"],
                "status": str(row["status"]),
                "source_path": str(row["archive_path"]) if row["archive_path"] else None,
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate["metric"] is None,
            -float(candidate["metric"]) if candidate["metric"] is not None else 0.0,
            str(candidate["candidate_id"]),
        ),
    )


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
    if status.get("session_id"):
        lines.append(f"Session: {status['session_id']}")
    if status.get("workspace"):
        lines.append(f"Workspace: {status['workspace']}")
    if status.get("last_checkpoint"):
        lines.append(f"Checkpoint: {status['last_checkpoint']}")
    if status.get("generation") is not None:
        lines.append(
            f"Progress: {status['generation']} completed generations "
            f"(limit {status['max_generations']})"
        )
    lines.append(
        f"Results: {status['unique_candidate_count']} accepted candidates, "
        f"{status.get('evaluation_count', 0)} evaluations"
    )
    ranked = status.get("ranked_candidates")
    if status.get("best_program_id"):
        lines.append(
            f"Winner: {status['best_program_id']}, primary metric "
            f"{status['best_primary_metric']}"
        )
        if status.get("winner_source"):
            lines.append(f"Winner code: {status['winner_source']}")
    elif int(status.get("unique_candidate_count", 0)) == 0:
        lines.append("Winner: none — no candidate was accepted")
    elif int(status.get("evaluation_count", 0)) == 0:
        lines.append("Winner: none — accepted candidates have not been evaluated")
    else:
        lines.append("Winner: none — no evaluated winner is available")
    if isinstance(ranked, list) and ranked:
        lines.append("Best mutations:")
        for index, candidate in enumerate(ranked[:5], start=1):
            metric = candidate.get("metric")
            lines.append(
                f"  {index}. {candidate.get('candidate_id')} "
                f"score={metric if metric is not None else 'not evaluated'} "
                f"generation={candidate.get('generation')} "
                f"code={candidate.get('source_path') or '-'}"
            )
    usage = status.get("token_usage")
    if isinstance(usage, Mapping):
        lines.append(
            "Tokens: "
            f"input {usage.get('inputTokens', 0)}, "
            f"cached {usage.get('cachedInputTokens', 0)}, "
            f"output {usage.get('outputTokens', 0)}, "
            f"reasoning {usage.get('reasoningOutputTokens', 0)}, "
            f"total {usage.get('totalTokens', status['total_tokens'])} "
            f"({usage.get('quality', 'unknown')}); "
            f"charged failed turns {usage.get('chargedFailedTurns', 0)}"
        )
    else:
        lines.append(
            f"Usage: {status['provider_turns']} model turns, "
            f"{status['total_tokens']} tokens"
        )
    lines.append(f"Model turns: {status['provider_turns']}")
    artifacts = status.get("artifacts")
    if isinstance(artifacts, Mapping) and artifacts:
        lines.append("Artifacts:")
        for label, path in artifacts.items():
            lines.append(f"  {str(label).replace('_', ' ')}: {path}")
    if status.get("ir") is not None:
        lines.append(f"IR: {status['ir']}")
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
