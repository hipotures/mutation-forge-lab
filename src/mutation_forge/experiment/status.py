"""Read-only operational status views for experiment workspaces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .checkpoints import CheckpointIntegrityError, CheckpointStore
from .config import ExperimentConfig, load_experiment_config, serialize_search_limit
from .layout import ExperimentLayout, WorkspaceError
from .lock import LockError, load_lock, verify_lock
from .state import RESUMABLE_STATES, ExperimentStateStore, StateError, process_alive

STATUS_SCHEMA_VERSION = "mforge.experiment.status.v2"


def _not_created(config: ExperimentConfig, layout: ExperimentLayout) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "exp_id": config.exp_id,
        "state": "not_created",
        "resumable": True,
        "workspace": str(layout.root),
        "session_id": None,
        "generation": 0,
        "max_generations": serialize_search_limit(config.search.max_generations),
        "search_limits": {
            "max_generations": serialize_search_limit(config.search.max_generations),
            "max_model_turns": serialize_search_limit(config.search.max_model_turns),
        },
        "terminal": False,
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
        "hourly_token_limit": config.run.max_total_tokens_per_hour,
        "hourly_tokens_used": 0,
        "hourly_tokens_remaining": config.run.max_total_tokens_per_hour,
        "hourly_window_seconds": 3600,
        "hourly_limit_reached": False,
        "hourly_retry_after": None,
        "token_usage": {
            "inputTokens": 0,
            "cachedInputTokens": 0,
            "cacheWriteInputTokens": 0,
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
        "counterexample": {"state": "none"},
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
    manifest_error: str | None = None
    try:
        layout.verify_artifact_manifest(allow_new=True)
    except WorkspaceError as error:
        # The active session appends its event log and rewrites the manifest
        # concurrently.  A read racing that append can observe one file hash
        # from before the manifest update; defer the strict verdict until the
        # owner/state row is available instead of reporting a false FAILED.
        manifest_error = str(error)
    runtime_schema_error: str | None = None
    try:
        layout.verify_runtime_schemas()
    except WorkspaceError as error:
        runtime_schema_error = str(error)
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
        owner_active = owner is not None and process_alive(int(owner["pid"]))
        if (manifest_error is not None or runtime_schema_error is not None) and not owner_active:
            return {
                **_not_created(config, layout),
                "state": "failed",
                "resumable": False,
                "last_error": manifest_error or runtime_schema_error,
            }
        if (
            current_state == "running"
            and owner is not None
            and not owner_active
        ):
            current_state = "interrupted"
        if checkpoint_error is not None:
            current_state = "failed"
        counts = state.counts()
        session = state.session()
        current = state.cumulative()
        generation_value = (
            checkpoint.get("next_generation", checkpoint.get("generation", 0))
            if checkpoint
            else 0
        )
        generation = (
            int(generation_value)
            if isinstance(generation_value, int) and not isinstance(generation_value, bool)
            else 0
        )
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
        hourly_usage = state.hourly_token_usage(
            config.run.max_total_tokens_per_hour
        )
        last_error = checkpoint_error or state.latest_error()
        if current_state == "running" and owner_active:
            # A stale error from an earlier resumable session must not be shown
            # as the current failure while the owner is actively progressing.
            last_error = None
        session_metrics: dict[str, Any] = {}
        if session is not None:
            try:
                parsed = json.loads(str(session.get("summary_json", "{}")))
            except (TypeError, json.JSONDecodeError):
                parsed = {}
            if isinstance(parsed, Mapping):
                session_metrics = dict(parsed)
        last_stop_reason = experiment.get("terminal_stop_reason") or (session or {}).get(
            "stop_reason"
        )
        if last_stop_reason == "already_completed":
            last_stop_reason = state.latest_meaningful_stop_reason() or last_stop_reason
        configured_model_turns = _locked_model_turns(lock)
        effective_model_turns = config.search.max_model_turns
        if configured_model_turns is None and config.search.max_model_turns is not None:
            configured_model_turns = config.search.max_model_turns
        model_turns_used = int(current["provider_turns"])
        native_checkpoint = layout.artifacts / "native-generation-checkpoint.json"
        native_state: Mapping[str, Any] = {}
        if native_checkpoint.is_file():
            try:
                raw_native_state = json.loads(native_checkpoint.read_text(encoding="utf-8"))
                if isinstance(raw_native_state, Mapping):
                    native_state = raw_native_state
            except (OSError, UnicodeError, json.JSONDecodeError):
                native_state = {}
            native_generation_value = native_state.get(
                "next_generation", native_state.get("generation")
            )
            if (
                isinstance(native_generation_value, int)
                and not isinstance(native_generation_value, bool)
            ):
                generation = max(generation, native_generation_value)
            native_completed_slots = native_state.get("completed_slots")
            if isinstance(native_completed_slots, int) and not isinstance(
                native_completed_slots, bool
            ):
                completed_slots = max(completed_slots, native_completed_slots)
            checkpoint_turns = (
                native_state.get("model_turns_used")
            )
            if (
                isinstance(checkpoint_turns, int)
                and not isinstance(checkpoint_turns, bool)
                and checkpoint_turns >= 0
            ):
                model_turns_used = max(model_turns_used, checkpoint_turns)
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
        raw_counterexample = session_metrics.get("counterexample")
        counterexample = (
            dict(raw_counterexample)
            if isinstance(raw_counterexample, Mapping)
            else {
                "state": ("verified" if last_stop_reason == "counterexample_verified" else "none")
            }
        )
        if last_stop_reason == "counterexample_verified":
            counterexample["state"] = "verified"
        certificate_path = counterexample.get("certificate_path")
        if isinstance(certificate_path, str):
            artifacts["counterexample_certificate"] = certificate_path
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "exp_id": config.exp_id,
            "state": current_state,
            "resumable": current_state in RESUMABLE_STATES,
            "terminal": current_state in {"exhausted", "failed", "completed"},
            "workspace": str(layout.root),
            "session_id": session_id,
            "generation": generation,
            "max_generations": serialize_search_limit(config.search.max_generations),
            "search_limits": {
                "max_generations": serialize_search_limit(config.search.max_generations),
                "max_model_turns": serialize_search_limit(config.search.max_model_turns),
            },
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
            "model_turns_used": model_turns_used,
            "configured_model_turns": serialize_search_limit(configured_model_turns),
            "effective_model_turns": serialize_search_limit(effective_model_turns),
            "remaining_model_turns": (
                max(0, effective_model_turns - model_turns_used)
                if effective_model_turns is not None
                else None
            ),
            "total_tokens": int(current["total_tokens"]),
            **hourly_usage,
            "token_usage": token_usage,
            "ir": session_metrics.get("ir"),
            "compute_seconds": float(current["compute_seconds"]),
            "last_checkpoint": checkpoint.get("checkpoint_id")
            if checkpoint
            else experiment.get("current_checkpoint"),
            "last_stop_reason": last_stop_reason,
            "last_error": last_error,
            "counterexample": counterexample,
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
    return metric if isinstance(metric, int | float) and not isinstance(metric, bool) else None


def _ranked_candidates(state: ExperimentStateStore) -> list[dict[str, Any]]:
    rows = state.connection.execute(
        "SELECT candidate_id,archive_path,generation,slot,status,metadata_json,"
        "(SELECT result_json FROM evaluations e "
        " WHERE e.identity = candidates.candidate_id || ':development' "
        "   AND e.state='completed' LIMIT 1) AS evaluation_json "
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
        metric = _candidate_metric(metadata)
        if metric is None and row["evaluation_json"]:
            try:
                evaluation = json.loads(str(row["evaluation_json"]))
            except json.JSONDecodeError:
                evaluation = {}
            summary = evaluation.get("summary") if isinstance(evaluation, Mapping) else None
            candidate_metric = summary.get("mean_auc") if isinstance(summary, Mapping) else None
            if (
                isinstance(candidate_metric, (int, float))
                and not isinstance(candidate_metric, bool)
            ):
                metric = float(candidate_metric)
        candidates.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "metric": metric,
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
        if status.get("max_generations") == "unbounded":
            lines.append(f"Progress: generation {status['generation']} · open-ended")
        else:
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
            f"Winner: {status['best_program_id']}, primary metric {status['best_primary_metric']}"
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
            f"Usage: {status['provider_turns']} model turns, {status['total_tokens']} tokens"
        )
    hourly_limit = status.get("hourly_token_limit")
    hourly_used = status.get("hourly_tokens_used", 0)
    lines.append(
        "Hourly tokens: "
        f"{hourly_used} / {hourly_limit if hourly_limit is not None else 'unbounded'}"
    )
    if status.get("hourly_limit_reached"):
        lines.append(f"Retry after: {status.get('hourly_retry_after') or 'pending'}")
    if status.get("effective_model_turns") == "unbounded":
        lines.append(
            f"Model turns: {status.get('model_turns_used', status['provider_turns'])} cumulative"
        )
    elif status.get("effective_model_turns") is not None:
        lines.append(
            "Model turns: "
            f"{status.get('model_turns_used', status['provider_turns'])} "
            f"/ {status['effective_model_turns']} "
            f"(remaining {status.get('remaining_model_turns', 0)})"
        )
    else:
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


def _locked_model_turns(lock: Mapping[str, Any]) -> int | None:
    normalized = lock.get("normalized_immutable_config")
    if isinstance(normalized, Mapping):
        search = normalized.get("search")
        if isinstance(search, Mapping):
            value = search.get("max_model_turns")
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None
