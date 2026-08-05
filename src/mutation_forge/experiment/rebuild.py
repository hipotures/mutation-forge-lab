"""Offline v2-to-v3 state rebuild with fail-closed artifact validation."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import ArtifactIncompleteError, TurnArtifactStore
from .config import load_experiment_config
from .evaluation import SCHEMA_VERSION as EVALUATION_SCHEMA_VERSION
from .evaluation import _remove_redundant_episode_checkpoints
from .json_io import read_json
from .layout import ExperimentLayout
from .state import (
    STATE_SCHEMA_VERSION,
    ExperimentStateStore,
    _usage_quality,
    _usage_value,
)

SOURCE_SCHEMA_VERSION = "mforge.experiment.state.v2"
EVENT_SOURCE_SCHEMA_VERSION = "mforge.experiment.events.v2"
EVENT_SCHEMA_VERSION = "mforge.experiment.events.v3"
_MIN_FREE_MARGIN = 1 << 30


class RebuildError(RuntimeError):
    """The source workspace cannot be rebuilt without risking data loss."""


@contextmanager
def _readonly_connection(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        yield connection
    finally:
        connection.close()


def _schema_version(connection: sqlite3.Connection) -> str | None:
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    return str(row[0]) if row is not None else None


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RebuildError(f"{label} must be a JSON object")
    return dict(value)


def _decode_object(raw: object, label: str) -> dict[str, Any]:
    try:
        return _json_object(json.loads(str(raw)), label)
    except json.JSONDecodeError as exc:
        raise RebuildError(f"{label} is unreadable") from exc


def _evaluation_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    summary = _json_object(value.get("summary"), "evaluation summary")
    runtime = value.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    elapsed = runtime.get("elapsed_seconds")
    return {
        "episode_count": summary.get("episode_count"),
        "mean_auc": summary.get("mean_auc"),
        "best_auc": summary.get("best_auc"),
        "baseline_auc": (
            dict(summary["baseline_auc"])
            if isinstance(summary.get("baseline_auc"), Mapping)
            else {}
        ),
        "improvement_rate": summary.get("improvement_rate"),
        "elapsed_seconds": (
            float(elapsed)
            if isinstance(elapsed, int | float) and not isinstance(elapsed, bool)
            else None
        ),
    }


def _content_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _session_summaries_match(
    database_summary: dict[str, Any],
    artifact_summary: dict[str, Any],
    layout: ExperimentLayout,
) -> bool:
    if database_summary == artifact_summary:
        return True
    database_result = database_summary.get("result")
    artifact_result = artifact_summary.get("result")
    if not isinstance(database_result, Mapping) or not isinstance(
        artifact_result, Mapping
    ):
        return False
    database_result_summary = database_result.get("summary")
    artifact_result_summary = artifact_result.get("summary")
    if not isinstance(database_result_summary, Mapping) or not isinstance(
        artifact_result_summary, Mapping
    ):
        return False
    database_checkpoint = database_result_summary.get("checkpoint")
    artifact_checkpoint = artifact_result_summary.get("checkpoint")
    if (
        not isinstance(database_checkpoint, str)
        or not isinstance(artifact_checkpoint, str)
        or not database_checkpoint.endswith(".json")
        or artifact_checkpoint != f"{database_checkpoint}.gz"
        or Path(artifact_checkpoint).name != "native-generation-checkpoint.json.gz"
    ):
        return False
    checkpoint_path = Path(artifact_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = layout.root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if (
        not checkpoint_path.is_relative_to(layout.root.resolve())
        or not checkpoint_path.is_file()
    ):
        return False
    normalized = deepcopy(database_summary)
    normalized["result"]["summary"]["checkpoint"] = artifact_checkpoint
    return normalized == artifact_summary


def _is_initial_session_record(
    record: Mapping[str, Any],
    row: sqlite3.Row,
) -> bool:
    observed = dict(record)
    start_time = observed.pop("start_time", None)
    try:
        start_delta = (
            datetime.fromisoformat(str(start_time))
            - datetime.fromisoformat(str(row["started_at"]))
        ).total_seconds()
    except ValueError:
        return False
    return 0 <= start_delta <= 1 and observed == {
        "schema_version": "mforge.experiment.session.v2",
        "session_id": row["session_id"],
        "session_number": row["number"],
        "starting_checkpoint": row["starting_checkpoint"],
        "starting_state": row["starting_state"],
        "wall_seconds": row["wall_seconds"],
    }


def _recover_candidate_from_turn(
    layout: ExperimentLayout,
    candidate_id: str,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    match = re.fullmatch(r"g(\d{4})-(slot-\d{2})", candidate_id)
    if match is None:
        raise RebuildError(f"cannot recover candidate identity: {candidate_id}")
    generation = int(match.group(1))
    slot = match.group(2)
    turn_dir = layout.generation_slot_phase(generation, slot)
    try:
        TurnArtifactStore(layout.artifacts).verify_turn(turn_dir)
        manifest = _json_object(
            read_json(turn_dir / "turn-manifest.json.gz"),
            f"turn manifest {candidate_id}",
        )
        identity = _json_object(
            read_json(turn_dir / "identity.json.gz"),
            f"candidate identity {candidate_id}",
        )
        behavior = _json_object(
            read_json(turn_dir / "behavior.json.gz"),
            f"candidate behavior {candidate_id}",
        )
        validation = _json_object(
            read_json(turn_dir / "validation.json.gz"),
            f"candidate validation {candidate_id}",
        )
        metadata_validation = _json_object(
            read_json(turn_dir / "metadata-validation.json.gz"),
            f"candidate metadata validation {candidate_id}",
        )
        request = _json_object(
            read_json(turn_dir / f"{slot}.request.json.gz"),
            f"candidate request {candidate_id}",
        )
    except (
        ArtifactIncompleteError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise RebuildError(
            f"candidate recovery artifacts are invalid: {candidate_id}"
        ) from exc
    source_identity = evaluation.get("source_identity")
    behavior_identity = behavior.get("identity")
    validation_identity = validation.get("identity")
    source_sha256 = (
        source_identity.get("source_sha256")
        if isinstance(source_identity, Mapping)
        else None
    )
    source_path = turn_dir / "source.py"
    parent_id = request.get("parent_id")
    if (
        not isinstance(source_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
        or not source_path.is_file()
        or hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256
        or identity.get("source_sha256") != source_sha256
        or not isinstance(behavior_identity, Mapping)
        or behavior_identity.get("source_sha256") != source_sha256
        or not isinstance(validation_identity, Mapping)
        or validation_identity.get("source_sha256") != source_sha256
        or validation.get("valid") is not True
        or metadata_validation.get("status") != "matched"
        or manifest.get("artifact_complete") is not True
        or manifest.get("terminal_status") != "completed"
        or manifest.get("generation") != generation
        or manifest.get("slot") != slot
        or manifest.get("phase") != "initial"
        or request.get("generation") != generation
        or request.get("slot") != slot
        or request.get("phase") != "initial"
        or not isinstance(parent_id, str)
    ):
        raise RebuildError(
            f"candidate recovery metadata does not match evaluation: {candidate_id}"
        )
    return {
        "candidate_id": candidate_id,
        "source_sha256": source_sha256,
        "archive_path": str(source_path),
        "generation": generation,
        "slot": slot,
        "parent_id": parent_id,
        "status": "created",
        "behavior": behavior,
    }


def _archive_metadata(layout: ExperimentLayout) -> dict[str, dict[str, Any]]:
    path = layout.archive / "index.jsonl"
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RebuildError(f"candidate archive is unreadable: {path}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RebuildError(f"candidate archive line {number} is invalid") from exc
        if not isinstance(value, Mapping) or not isinstance(value.get("program_id"), str):
            raise RebuildError(f"candidate archive line {number} has no program_id")
        records[str(value["program_id"])] = {
            "parent_id": value.get("parent_id"),
            "behavior": (
                dict(value["behavior"])
                if isinstance(value.get("behavior"), Mapping)
                else {}
            ),
        }
    return records


def _iter_session_events(path: Path, session_id: str) -> Iterator[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RebuildError(f"session event stream is unreadable: {path}") from exc
    ordinal = 0
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RebuildError(f"invalid event at {path}:{number}") from exc
        event = _json_object(value, f"event at {path}:{number}")
        if event.get("schema_version") != EVENT_SOURCE_SCHEMA_VERSION:
            raise RebuildError(f"unsupported event schema at {path}:{number}")
        event["schema_version"] = EVENT_SCHEMA_VERSION
        event["event_id"] = f"{session_id}:{ordinal:08d}"
        ordinal += 1
        yield event


def _session_event_payloads(
    layout: ExperimentLayout,
) -> dict[str, tuple[Path, bytes]]:
    payloads: dict[str, tuple[Path, bytes]] = {}
    found_event_stream = False
    for session_dir in sorted(layout.sessions.glob("session-*")):
        if not session_dir.is_dir():
            continue
        source = session_dir / "events.jsonl"
        compressed = session_dir / "events.jsonl.gz"
        if compressed.is_file() and source.is_file():
            raise RebuildError(f"both event stream formats exist: {session_dir}")
        if compressed.is_file():
            try:
                lines = gzip.decompress(compressed.read_bytes()).decode("utf-8").splitlines()
            except (OSError, UnicodeError, gzip.BadGzipFile) as exc:
                raise RebuildError(
                    f"compressed session event stream is unreadable: {compressed}"
                ) from exc
            for number, line in enumerate(lines, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RebuildError(
                        f"invalid compressed event at {compressed}:{number}"
                    ) from exc
                if (
                    not isinstance(event, Mapping)
                    or event.get("schema_version") != EVENT_SCHEMA_VERSION
                    or not isinstance(event.get("event_id"), str)
                ):
                    raise RebuildError(
                        f"unsupported compressed event at {compressed}:{number}"
                    )
            found_event_stream = True
            continue
        if not source.is_file():
            if found_event_stream:
                raise RebuildError(f"session event stream is missing: {source}")
            continue
        found_event_stream = True
        session_id = session_dir.name
        canonical = b"".join(
            (
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            for event in _iter_session_events(source, session_id)
        )
        payloads[session_id] = (source, canonical)
    return payloads


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_session_event_payloads(
    payloads: Mapping[str, tuple[Path, bytes]],
) -> int:
    written = 0
    for source, canonical in payloads.values():
        destination = source.with_suffix(source.suffix + ".gz")
        compressed = gzip.compress(canonical, compresslevel=6, mtime=0)
        _atomic_write(destination, compressed)
        try:
            observed = gzip.decompress(destination.read_bytes())
        except (OSError, gzip.BadGzipFile) as exc:
            raise RebuildError(f"converted event stream is unreadable: {destination}") from exc
        if observed != canonical:
            raise RebuildError(f"converted event stream differs from source: {source}")
        written += source.stat().st_size
        source.unlink()
    return written


def _audit_source(
    connection: sqlite3.Connection,
    layout: ExperimentLayout,
    *,
    remove_checkpoints: bool,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    int,
    int,
    list[Path],
]:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise RebuildError("source state database failed PRAGMA integrity_check")
    owner = connection.execute("SELECT pid FROM ownership WHERE singleton=1").fetchone()
    if owner is not None:
        raise RebuildError(f"experiment still has an owner PID: {owner['pid']}")
    experiment = connection.execute("SELECT state FROM experiment LIMIT 1").fetchone()
    if experiment is None or experiment["state"] == "running":
        raise RebuildError("experiment must be stopped before rebuilding state")

    redundant_session_records: list[Path] = []
    for row in connection.execute(
        "SELECT number,session_id,started_at,finished_at,wall_seconds,"
        "starting_checkpoint,starting_state,ending_state,status,exit_status,summary_json "
        "FROM sessions ORDER BY number"
    ):
        session_id = str(row["session_id"])
        database_summary = _decode_object(
            row["summary_json"],
            f"session {session_id}",
        )
        session_dir = layout.sessions / session_id
        summary_path = session_dir / "summary.json.gz"
        duplicate_path = session_dir / "session.json.gz"
        if not summary_path.is_file():
            if (
                database_summary
                or row["status"] != "running"
                or row["finished_at"] is not None
                or row["ending_state"] is not None
                or row["exit_status"] is not None
                or not duplicate_path.is_file()
            ):
                raise RebuildError(f"session summary artifact is missing: {summary_path}")
            try:
                initial_record = _json_object(
                    read_json(duplicate_path),
                    f"initial session record {session_id}",
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RebuildError(
                    f"initial session record is unreadable: {duplicate_path}"
                ) from exc
            if not _is_initial_session_record(initial_record, row):
                raise RebuildError(
                    f"incomplete session record is invalid: {session_id}"
                )
            continue
        try:
            artifact_summary = _json_object(
                read_json(summary_path),
                f"session summary {session_id}",
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RebuildError(
                f"session summary artifact is unreadable: {summary_path}"
            ) from exc
        if not _session_summaries_match(database_summary, artifact_summary, layout):
            raise RebuildError(
                f"database session summary contains data not matched by artifact: {session_id}"
            )
        if duplicate_path.is_file():
            try:
                duplicate = _json_object(
                    read_json(duplicate_path),
                    f"duplicate session record {session_id}",
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RebuildError(
                    f"duplicate session record is unreadable: {duplicate_path}"
                ) from exc
            if duplicate != artifact_summary:
                raise RebuildError(
                    f"session records differ and cannot be deduplicated: {session_id}"
                )
            redundant_session_records.append(duplicate_path)

    candidate_ids = {
        str(row[0]) for row in connection.execute("SELECT candidate_id FROM candidates")
    }
    recovered_candidates: dict[str, dict[str, Any]] = {}
    projections: dict[str, dict[str, Any]] = {}
    removable_bytes = 0
    rows = connection.execute(
        "SELECT identity,candidate_id,state,result_json,completed_at "
        "FROM evaluations ORDER BY identity"
    )
    for row in rows:
        identity = str(row["identity"])
        candidate_id = str(row["candidate_id"] or "")
        state = str(row["state"])
        if state != "completed":
            projections[identity] = {"_completed_at": row["completed_at"]}
            continue
        artifact_path = (
            layout.evaluations / "development" / f"{candidate_id}.json.gz"
        )
        if not artifact_path.is_file():
            raise RebuildError(f"completed evaluation artifact is missing: {artifact_path}")
        try:
            artifact = _json_object(read_json(artifact_path), f"evaluation {candidate_id}")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RebuildError(f"evaluation artifact is unreadable: {artifact_path}") from exc
        if (
            artifact.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or artifact.get("status") != "completed"
            or artifact.get("candidate_id") != candidate_id
        ):
            raise RebuildError(f"evaluation artifact identity mismatch: {artifact_path}")
        database_result = _decode_object(
            row["result_json"], f"database evaluation {identity}"
        )
        replay_metadata = database_result.get("replay")
        if not isinstance(replay_metadata, Mapping):
            raise RebuildError(f"database evaluation replay metadata is invalid: {identity}")
        primary_result = {
            key: value
            for key, value in database_result.items()
            if key not in {"artifacts", "replay"}
        }
        if primary_result != artifact:
            raise RebuildError(
                f"database evaluation contains data not matched by artifact: {identity}"
            )
        projections[identity] = {
            **_evaluation_projection(artifact),
            "_completed_at": row["completed_at"],
        }
        removable_bytes += _remove_redundant_episode_checkpoints(
            layout.artifacts,
            pass_name="development",
            candidate_id=candidate_id,
            completed=artifact,
            remove=remove_checkpoints,
        )
        replay_path = layout.evaluations / "replay" / f"{candidate_id}.json.gz"
        if replay_metadata.get("enabled") is True and not replay_path.is_file():
            raise RebuildError(f"completed replay artifact is missing: {replay_path}")
        if replay_path.is_file():
            replay = _json_object(read_json(replay_path), f"replay {candidate_id}")
            if (
                replay.get("schema_version") != EVALUATION_SCHEMA_VERSION
                or replay.get("status") != "completed"
                or replay.get("candidate_id") != candidate_id
            ):
                raise RebuildError(f"replay artifact identity mismatch: {replay_path}")
            expected_hash = replay_metadata.get("sha256")
            if isinstance(expected_hash, str) and expected_hash != _content_hash(replay):
                raise RebuildError(f"replay artifact hash mismatch: {replay_path}")
            expected_exact = replay_metadata.get("exact")
            observed_exact = _content_hash(artifact) == _content_hash(replay)
            if isinstance(expected_exact, bool) and expected_exact != observed_exact:
                raise RebuildError(f"replay exactness mismatch: {replay_path}")
            removable_bytes += _remove_redundant_episode_checkpoints(
                layout.artifacts,
                pass_name="replay",
                candidate_id=candidate_id,
                completed=replay,
                remove=remove_checkpoints,
            )

    for artifact_path in sorted((layout.evaluations / "development").glob("*.json.gz")):
        candidate_id = artifact_path.name.removesuffix(".json.gz")
        identity = f"{candidate_id}:development"
        if identity in projections:
            continue
        artifact = _json_object(read_json(artifact_path), f"evaluation {candidate_id}")
        if (
            artifact.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or artifact.get("status") != "completed"
            or artifact.get("candidate_id") != candidate_id
        ):
            raise RebuildError(f"evaluation artifact identity mismatch: {artifact_path}")
        if candidate_id not in candidate_ids:
            recovered_candidates[candidate_id] = _recover_candidate_from_turn(
                layout,
                candidate_id,
                artifact,
            )
        projections[identity] = {
            **_evaluation_projection(artifact),
            "_completed_at": datetime.fromtimestamp(
                artifact_path.stat().st_mtime,
                tz=UTC,
            ).isoformat(),
        }
        removable_bytes += _remove_redundant_episode_checkpoints(
            layout.artifacts,
            pass_name="development",
            candidate_id=candidate_id,
            completed=artifact,
            remove=remove_checkpoints,
        )
    if remove_checkpoints:
        for path in redundant_session_records:
            path.unlink()
    return (
        projections,
        recovered_candidates,
        removable_bytes,
        len(candidate_ids) + len(recovered_candidates),
        redundant_session_records,
    )


def _copy_rows(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    projections: Mapping[str, Mapping[str, Any]],
    archive_metadata: Mapping[str, Mapping[str, Any]],
    recovered_candidates: Mapping[str, Mapping[str, Any]],
) -> None:
    destination.execute("BEGIN IMMEDIATE")
    try:
        for row in source.execute("SELECT key,value FROM metadata WHERE key!='schema_version'"):
            destination.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?)",
                (row["key"], row["value"]),
            )
        experiment = source.execute("SELECT * FROM experiment LIMIT 1").fetchone()
        if experiment is None:
            raise RebuildError("source database has no experiment row")
        destination.execute("DELETE FROM experiment")
        destination.execute(
            "INSERT INTO experiment(exp_id,root,lock_hash,state,created_at,updated_at,"
            "current_session_id,current_checkpoint,cumulative_model_turns,cumulative_tokens,"
            "cumulative_runtime_seconds,last_error,terminal_stop_reason)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(experiment),
        )
        for row in source.execute("SELECT * FROM sessions ORDER BY number"):
            summary = _decode_object(row["summary_json"], f"session {row['session_id']}")
            counterexample = summary.get("counterexample")
            destination.execute(
                "INSERT INTO sessions(number,session_id,started_at,finished_at,wall_seconds,"
                "starting_checkpoint,ending_checkpoint,starting_state,ending_state,status,"
                "provider_turns_attempted,provider_turns_completed,candidates_created,"
                "evaluations_completed,token_usage_delta,cumulative_tokens,runtime_seconds,"
                "stop_reason,exit_status,ir,counterexample_json)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    *tuple(row)[:-1],
                    summary.get("ir"),
                    json.dumps(
                        counterexample if isinstance(counterexample, Mapping) else {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
        for row in source.execute("SELECT * FROM provider_turns ORDER BY idempotency_key"):
            usage = _decode_object(
                row["usage_json"], f"provider turn {row['idempotency_key']}"
            )
            destination.execute(
                "INSERT INTO provider_turns(idempotency_key,generation,slot,phase,state,"
                "provider_thread_id,provider_turn_id,artifact_path,input_tokens,"
                "cached_input_tokens,cache_write_input_tokens,output_tokens,"
                "reasoning_output_tokens,total_tokens,usage_quality,completed_at,error)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    *tuple(row)[:8],
                    *(_usage_value(usage, field) for field in (
                        "inputTokens",
                        "cachedInputTokens",
                        "cacheWriteInputTokens",
                        "outputTokens",
                        "reasoningOutputTokens",
                        "totalTokens",
                    )),
                    str(usage.get("quality") or _usage_quality(usage)),
                    row["completed_at"],
                    row["error"],
                ),
            )
        for row in source.execute("SELECT * FROM candidates ORDER BY candidate_id"):
            metadata = _decode_object(row["metadata_json"], f"candidate {row['candidate_id']}")
            archive = archive_metadata.get(str(row["candidate_id"]), {})
            behavior = metadata.get("behavior")
            if not isinstance(behavior, Mapping):
                behavior = archive.get("behavior", {})
            parent_id = metadata.get("parent_id", archive.get("parent_id"))
            destination.execute(
                "INSERT INTO candidates(candidate_id,source_sha256,archive_path,generation,"
                "slot,parent_id,status,behavior_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    *tuple(row)[:5],
                    parent_id,
                    row["status"],
                    json.dumps(
                        behavior if isinstance(behavior, Mapping) else {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
        for candidate_id, candidate in sorted(recovered_candidates.items()):
            destination.execute(
                "INSERT INTO candidates(candidate_id,source_sha256,archive_path,generation,"
                "slot,parent_id,status,behavior_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    candidate_id,
                    candidate["source_sha256"],
                    candidate["archive_path"],
                    candidate["generation"],
                    candidate["slot"],
                    candidate["parent_id"],
                    candidate["status"],
                    json.dumps(
                        candidate["behavior"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
        source_evaluations = {
            str(row["identity"]): row
            for row in source.execute(
                "SELECT identity,candidate_id,kind,state,completed_at FROM evaluations"
            )
        }
        for identity, projection in sorted(projections.items()):
            row = source_evaluations.get(identity)
            candidate_id = (
                str(row["candidate_id"])
                if row is not None and row["candidate_id"]
                else identity.removesuffix(":development")
            )
            destination.execute(
                "INSERT INTO evaluations(identity,candidate_id,kind,state,episode_count,"
                "mean_auc,best_auc,baseline_auc_json,improvement_rate,elapsed_seconds,"
                "completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    candidate_id,
                    str(row["kind"]) if row is not None else "development",
                    str(row["state"]) if row is not None else "completed",
                    projection.get("episode_count"),
                    projection.get("mean_auc"),
                    projection.get("best_auc"),
                    json.dumps(
                        projection.get("baseline_auc", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    projection.get("improvement_rate"),
                    projection.get("elapsed_seconds"),
                    projection.get("_completed_at"),
                ),
            )
        for row in source.execute("SELECT * FROM checkpoints ORDER BY sequence"):
            destination.execute(
                "INSERT INTO checkpoints(sequence,checkpoint_id,path,sha256,generation,"
                "completed_slots,created_at) VALUES(?,?,?,?,?,?,?)",
                tuple(row),
            )
        for row in source.execute("SELECT * FROM events ORDER BY sequence"):
            payload = _decode_object(row["payload_json"], f"event {row['sequence']}")
            if row["event_type"] == "model_token_charge_recorded":
                token_delta = payload.get("token_delta")
                turn_key = payload.get("turn_idempotency_key")
                charged_at = payload.get("charged_at", row["timestamp"])
                if (
                    not isinstance(token_delta, int)
                    or isinstance(token_delta, bool)
                    or token_delta <= 0
                    or not isinstance(turn_key, str)
                    or not isinstance(charged_at, str)
                ):
                    raise RebuildError(f"invalid token charge event {row['sequence']}")
                destination.execute(
                    "INSERT OR IGNORE INTO token_charges(idempotency_key,"
                    "turn_idempotency_key,charged_at,token_delta) VALUES(?,?,?,?)",
                    (row["idempotency_key"], turn_key, charged_at, token_delta),
                )
                continue
            selected = payload.get("selected_parents")
            destination.execute(
                "INSERT INTO events(sequence,session_id,event_type,timestamp,idempotency_key,"
                "generation,slot,selected_parents_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    row["sequence"],
                    row["session_id"],
                    row["event_type"],
                    row["timestamp"],
                    row["idempotency_key"],
                    payload.get("generation"),
                    payload.get("slot"),
                    (
                        json.dumps(
                            selected,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if isinstance(selected, Mapping)
                        else None
                    ),
                ),
            )
        destination.commit()
    except BaseException:
        destination.rollback()
        raise


def _validate_destination(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    candidate_count: int,
    evaluation_count: int,
) -> None:
    integrity = destination.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise RebuildError("rebuilt database failed PRAGMA integrity_check")
    expected = {
        "sessions": source.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        "provider_turns": source.execute("SELECT COUNT(*) FROM provider_turns").fetchone()[0],
        "candidates": candidate_count,
        "evaluations": evaluation_count,
        "checkpoints": source.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0],
    }
    for table, count in expected.items():
        observed = destination.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if observed != count:
            raise RebuildError(f"rebuilt {table} count mismatch: {observed} != {count}")
    source_experiment = tuple(source.execute("SELECT * FROM experiment LIMIT 1").fetchone())
    destination_experiment = tuple(
        destination.execute("SELECT * FROM experiment LIMIT 1").fetchone()
    )
    if destination_experiment != source_experiment:
        raise RebuildError("rebuilt experiment state differs from source")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rebuild_experiment_state(
    config_path: str | Path = "experiment.toml",
    *,
    apply: bool = False,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    layout = ExperimentLayout.from_config(config)
    state_path = layout.state
    if not state_path.is_file():
        raise RebuildError(f"state database is missing: {state_path}")

    with _readonly_connection(state_path) as source:
        observed_schema = _schema_version(source)
        if observed_schema == STATE_SCHEMA_VERSION:
            return {
                "status": "already_rebuilt",
                "schema_version": observed_schema,
                "state_path": str(state_path),
                "database_bytes": state_path.stat().st_size,
            }
        if observed_schema != SOURCE_SCHEMA_VERSION:
            raise RebuildError(f"unsupported source state schema: {observed_schema!r}")
        (
            projections,
            recovered_candidates,
            removable_bytes,
            candidate_count,
            redundant_sessions,
        ) = _audit_source(source, layout, remove_checkpoints=False)
        event_payloads = _session_event_payloads(layout)
        evaluation_count = len(projections)

    report: dict[str, Any] = {
        "status": "checked",
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "target_schema_version": STATE_SCHEMA_VERSION,
        "state_path": str(state_path),
        "source_database_bytes": state_path.stat().st_size,
        "candidate_count": candidate_count,
        "recovered_candidate_count": len(recovered_candidates),
        "evaluation_count": evaluation_count,
        "redundant_checkpoint_bytes": removable_bytes,
        "redundant_session_records": len(redundant_sessions),
        "session_event_streams": len(event_payloads),
    }
    if not apply:
        return report

    backup_root = Path(work_dir).resolve() if work_dir is not None else layout.root
    backup_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(backup_root).free
    same_filesystem = backup_root.stat().st_dev == layout.root.stat().st_dev
    effective_free = free + (removable_bytes if same_filesystem else 0)
    required = state_path.stat().st_size + _MIN_FREE_MARGIN
    if effective_free < required:
        raise RebuildError(
            f"insufficient free space for SQLite online backup: "
            f"{effective_free} available after cleanup, {required} required"
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_root / f"state.pre-v3-{timestamp}.sqlite3"
    new_path = layout.root / f".state.v3-{os.getpid()}.sqlite3"
    switched = False
    try:
        with _readonly_connection(state_path) as source:
            _audit_source(source, layout, remove_checkpoints=True)
        _write_session_event_payloads(event_payloads)
        layout.reconcile_artifact_manifest()

        before = state_path.stat()
        if backup_path.exists():
            raise RebuildError(f"backup destination already exists: {backup_path}")
        with _readonly_connection(state_path) as source, sqlite3.connect(backup_path) as backup:
            source.backup(backup)
        after = state_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RebuildError("source database changed during online backup")
        with _readonly_connection(backup_path) as snapshot:
            integrity = snapshot.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise RebuildError("SQLite online backup failed integrity_check")

        with _readonly_connection(backup_path) as snapshot:
            lock_row = snapshot.execute(
                "SELECT lock_hash FROM experiment LIMIT 1"
            ).fetchone()
            if lock_row is None:
                raise RebuildError("source database has no experiment lock hash")
            lock_hash = str(lock_row[0])
        ExperimentStateStore.initialize(
            new_path,
            exp_id=config.exp_id,
            lock_hash=lock_hash,
            root=layout.root,
        )
        archive_metadata = _archive_metadata(layout)
        with _readonly_connection(backup_path) as snapshot, sqlite3.connect(new_path) as target:
            snapshot.row_factory = sqlite3.Row
            target.row_factory = sqlite3.Row
            _copy_rows(
                snapshot,
                target,
                projections,
                archive_metadata,
                recovered_candidates,
            )
            _validate_destination(
                snapshot,
                target,
                candidate_count=candidate_count,
                evaluation_count=evaluation_count,
            )
        new_bytes = new_path.stat().st_size
        if state_path.stat().st_size >= 100_000_000 and new_bytes * 20 > state_path.stat().st_size:
            raise RebuildError("rebuilt database is not at least 95% smaller than source")
        with new_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(new_path, state_path)
        _fsync_directory(layout.root)
        switched = True
        report.update(
            {
                "status": "rebuilt",
                "database_bytes": new_bytes,
                "backup_path": str(backup_path),
                "reclaimed_after_backup_removal_bytes": (
                    report["source_database_bytes"] - new_bytes
                ),
            }
        )
        return report
    finally:
        new_path.unlink(missing_ok=True)
        if not switched:
            backup_path.unlink(missing_ok=True)


__all__ = ["RebuildError", "rebuild_experiment_state"]
