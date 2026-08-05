from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from mutation_forge.experiment.json_io import write_json
from mutation_forge.experiment.rebuild import RebuildError, rebuild_experiment_state
from mutation_forge.experiment.state import STATE_SCHEMA_VERSION, ExperimentStateStore


def _config() -> str:
    return '''schema_version = "mforge.experiment.v2"
exp_id = "demo"
workspace = "./workspace"
kind = "heg"
preset = "native"

[run]
wall_seconds = 60

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "high"
concurrency = 1
max_repairs = 0

[search]
population_size = 1
max_generations = 2
max_model_turns = 4
selection = "elite-diversity"

[evaluation]
graph_mode = "unrestricted_min_degree_3"
order_schedule = "static"
orders = [10]
graph_seeds = [401]
policy_seeds = [4001]
horizon = 4
proposal_pool_size = 2
baselines = ["random", "structural"]
replay = false

[resources]
workers = 1
thread_count = 1
'''


def _create_v2_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE experiment (
            exp_id TEXT PRIMARY KEY, root TEXT NOT NULL, lock_hash TEXT NOT NULL,
            state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            current_session_id TEXT, current_checkpoint TEXT,
            cumulative_model_turns INTEGER NOT NULL DEFAULT 0,
            cumulative_tokens INTEGER NOT NULL DEFAULT 0,
            cumulative_runtime_seconds REAL NOT NULL DEFAULT 0,
            last_error TEXT, terminal_stop_reason TEXT
        );
        CREATE TABLE sessions (
            number INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE,
            started_at TEXT NOT NULL, finished_at TEXT, wall_seconds REAL NOT NULL,
            starting_checkpoint TEXT, ending_checkpoint TEXT,
            starting_state TEXT NOT NULL, ending_state TEXT, status TEXT NOT NULL,
            provider_turns_attempted INTEGER NOT NULL DEFAULT 0,
            provider_turns_completed INTEGER NOT NULL DEFAULT 0,
            candidates_created INTEGER NOT NULL DEFAULT 0,
            evaluations_completed INTEGER NOT NULL DEFAULT 0,
            token_usage_delta INTEGER NOT NULL DEFAULT 0,
            cumulative_tokens INTEGER NOT NULL DEFAULT 0,
            runtime_seconds REAL NOT NULL DEFAULT 0, stop_reason TEXT,
            exit_status INTEGER, summary_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE ownership (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1), exp_id TEXT NOT NULL,
            session_id TEXT NOT NULL, pid INTEGER NOT NULL, started_at TEXT NOT NULL
        );
        CREATE TABLE provider_turns (
            idempotency_key TEXT PRIMARY KEY, generation INTEGER NOT NULL,
            slot TEXT NOT NULL, phase TEXT NOT NULL, state TEXT NOT NULL,
            provider_thread_id TEXT, provider_turn_id TEXT, artifact_path TEXT,
            usage_json TEXT NOT NULL DEFAULT '{}', completed_at TEXT, error TEXT
        );
        CREATE TABLE candidates (
            candidate_id TEXT PRIMARY KEY, source_sha256 TEXT, archive_path TEXT,
            generation INTEGER, slot TEXT, status TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE evaluations (
            identity TEXT PRIMARY KEY, candidate_id TEXT, kind TEXT NOT NULL,
            state TEXT NOT NULL, result_json TEXT NOT NULL DEFAULT '{}',
            completed_at TEXT
        );
        CREATE TABLE checkpoints (
            sequence INTEGER PRIMARY KEY, checkpoint_id TEXT NOT NULL UNIQUE,
            path TEXT NOT NULL, sha256 TEXT NOT NULL, generation INTEGER NOT NULL,
            completed_slots INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
        );
        CREATE TABLE events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
            event_type TEXT NOT NULL, timestamp TEXT NOT NULL,
            idempotency_key TEXT UNIQUE, payload_json TEXT NOT NULL
        );
        """
    )
    return connection


def _workspace(
    tmp_path: Path,
    *,
    mismatched_artifact: bool = False,
    active_owner: bool = False,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "experiment.toml"
    config_path.write_text(_config(), encoding="utf-8")
    root = tmp_path / "workspace" / "demo"
    root.mkdir(parents=True)
    state_path = root / "state.sqlite3"
    session_dir = root / "artifacts" / "sessions" / "session-000001"
    evaluation_dir = root / "artifacts" / "evaluations" / "development"
    episode_dir = (
        root
        / "artifacts"
        / "evaluations"
        / "episodes"
        / "development"
        / "g0000-slot-00"
    )
    session_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    (root / "checkpoints").mkdir()

    episode = {
        "episode_id": "episode-0",
        "order": 10,
        "graph_seed": 401,
        "policy_seed": 4001,
        "trace": "x" * 1_000_000,
    }
    result = {
        "schema_version": "mforge.experiment.evaluation.v2",
        "status": "completed",
        "candidate_id": "g0000-slot-00",
        "source_identity": {"source_sha256": "a" * 64},
        "settings": {"orders": [10]},
        "episodes": [episode],
        "summary": {
            "episode_count": 1,
            "mean_auc": 0.75,
            "best_auc": 0.75,
            "baseline_auc": {"random": 0.5},
            "improvement_rate": 0.25,
        },
        "runtime": {"elapsed_seconds": 12.5},
        "provider_calls": 0,
        "model_calls": 0,
        "network_calls": 0,
        "provenance": {"backend_id": "test"},
    }
    artifact = dict(result)
    if mismatched_artifact:
        artifact["summary"] = {**result["summary"], "mean_auc": 0.8}
    write_json(evaluation_dir / "g0000-slot-00.json.gz", artifact, indent=2)
    write_json(
        episode_dir / "episode-000000.json.gz",
        {
            "schema_version": "mforge.experiment.evaluation.episode.v2",
            "identity": "checkpoint-id",
            "index": 0,
            "episode": episode,
        },
    )
    session_event = {
        "schema_version": "mforge.experiment.events.v2",
        "run_id": "session-000001",
        "timestamp": "2026-08-04T12:00:00+00:00",
        "event_type": "selection_completed",
        "generation": 0,
        "selected_parents": {"slot-00": "root"},
    }
    (session_dir / "events.jsonl").write_text(
        json.dumps(session_event, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    connection = _create_v2_state(state_path)
    connection.execute(
        "INSERT INTO metadata(key,value) VALUES('schema_version',?)",
        ("mforge.experiment.state.v2",),
    )
    connection.execute(
        "INSERT INTO experiment VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "demo",
            str(root),
            "lock-hash",
            "interrupted",
            "2026-08-04T11:00:00+00:00",
            "2026-08-04T12:00:00+00:00",
            "session-000001",
            "checkpoint-000000000001",
            1,
            12,
            20.0,
            None,
            "operator_stop",
        ),
    )
    summary = {
        "schema_version": "mforge.experiment.session.v2",
        "ir": 0.25,
        "counterexample": {"state": "none"},
    }
    write_json(session_dir / "summary.json.gz", summary)
    write_json(session_dir / "session.json.gz", summary)
    connection.execute(
        "INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            1,
            "session-000001",
            "2026-08-04T11:00:00+00:00",
            "2026-08-04T12:00:00+00:00",
            60.0,
            None,
            "checkpoint-000000000001",
            "idle",
            "interrupted",
            "interrupted",
            1,
            0,
            1,
            1,
            12,
            12,
            20.0,
            "operator_stop",
            130,
            json.dumps(summary),
        ),
    )
    usage = {
        "inputTokens": 8,
        "cachedInputTokens": 1,
        "outputTokens": 4,
        "reasoningOutputTokens": 2,
        "totalTokens": 12,
        "final": False,
        "partial": True,
        "quality": "partial",
    }
    connection.execute(
        "INSERT INTO provider_turns VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "turn-1",
            0,
            "slot-00",
            "initial",
            "failed",
            "thread-1",
            "turn-provider-1",
            "artifacts/generations/generation-0000/slot-00",
            json.dumps(usage),
            "2026-08-04T11:30:00+00:00",
            "EOF",
        ),
    )
    connection.execute(
        "INSERT INTO candidates VALUES(?,?,?,?,?,?,?)",
        (
            "g0000-slot-00",
            "a" * 64,
            "artifacts/archive/sources/g0000-slot-00.py",
            0,
            "slot-00",
            "created",
            json.dumps({"parent_id": "root", "behavior": {"family": "test"}}),
        ),
    )
    database_result = {
        **result,
        "artifacts": {"development": "artifact.json.gz"},
        "replay": {"enabled": False, "exact": None},
    }
    connection.execute(
        "INSERT INTO evaluations VALUES(?,?,?,?,?,?)",
        (
            "g0000-slot-00:development",
            "g0000-slot-00",
            "development",
            "completed",
            json.dumps(database_result, separators=(",", ":")),
            "2026-08-04T11:45:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?)",
        (
            1,
            "checkpoint-000000000001",
            "checkpoints/checkpoint-000000000001.json.gz",
            "b" * 64,
            0,
            1,
            "2026-08-04T12:00:00+00:00",
        ),
    )
    connection.execute(
        "INSERT INTO events(session_id,event_type,timestamp,idempotency_key,payload_json)"
        " VALUES(?,?,?,?,?)",
        (
            "session-000001",
            "selection_completed",
            "2026-08-04T12:00:00+00:00",
            "selection-0",
            json.dumps(
                {"generation": 0, "selected_parents": {"slot-00": "root"}}
            ),
        ),
    )
    connection.execute(
        "INSERT INTO events(session_id,event_type,timestamp,idempotency_key,payload_json)"
        " VALUES(?,?,?,?,?)",
        (
            "session-000001",
            "model_token_charge_recorded",
            "2026-08-04T11:30:00+00:00",
            "model-token-charge:turn-1:12",
            json.dumps(
                {
                    "turn_idempotency_key": "turn-1",
                    "charged_at": "2026-08-04T11:30:00+00:00",
                    "token_delta": 12,
                }
            ),
        ),
    )
    if active_owner:
        connection.execute(
            "INSERT INTO ownership VALUES(1,'demo','session-000001',12345,?)",
            ("2026-08-04T11:00:00+00:00",),
        )
    connection.commit()
    connection.close()
    return config_path, root


def test_rebuild_is_read_only_until_apply_and_preserves_operational_state(
    tmp_path: Path,
) -> None:
    config_path, root = _workspace(tmp_path)
    state_path = root / "state.sqlite3"
    source_bytes = state_path.stat().st_size

    checked = rebuild_experiment_state(config_path)

    assert checked["status"] == "checked"
    assert checked["evaluation_count"] == 1
    assert checked["redundant_checkpoint_bytes"] > 0
    assert checked["redundant_session_records"] == 1
    assert (root / "artifacts" / "sessions" / "session-000001" / "events.jsonl").is_file()
    assert (
        root
        / "artifacts"
        / "evaluations"
        / "episodes"
        / "development"
        / "g0000-slot-00"
    ).is_dir()

    rebuilt = rebuild_experiment_state(
        config_path,
        apply=True,
        work_dir=tmp_path / "backups",
    )

    assert rebuilt["status"] == "rebuilt"
    assert Path(rebuilt["backup_path"]).is_file()
    assert state_path.stat().st_size < source_bytes
    with sqlite3.connect(Path(rebuilt["backup_path"])) as old:
        assert old.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone() == ("mforge.experiment.state.v2",)
    with ExperimentStateStore(state_path) as state:
        assert state.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == STATE_SCHEMA_VERSION
        assert state.evaluation_summary("g0000-slot-00:development")["mean_auc"] == 0.75
        assert state.token_usage()["totalTokens"] == 12
        assert state.connection.execute("SELECT COUNT(*) FROM token_charges").fetchone()[0] == 1
        columns = {
            row[1] for row in state.connection.execute("PRAGMA table_info(evaluations)")
        }
        assert "result_json" not in columns
    events_path = (
        root / "artifacts" / "sessions" / "session-000001" / "events.jsonl.gz"
    )
    event = json.loads(gzip.decompress(events_path.read_bytes()).decode("utf-8"))
    assert event["schema_version"] == "mforge.experiment.events.v3"
    assert event["event_id"] == "session-000001:00000000"
    assert not events_path.with_suffix("").is_file()
    assert not (
        root / "artifacts" / "sessions" / "session-000001" / "session.json.gz"
    ).exists()
    assert (
        root / "artifacts" / "sessions" / "session-000001" / "summary.json.gz"
    ).is_file()
    assert not (
        root
        / "artifacts"
        / "evaluations"
        / "episodes"
        / "development"
        / "g0000-slot-00"
    ).exists()
    assert rebuild_experiment_state(config_path)["status"] == "already_rebuilt"


def test_rebuild_refuses_active_or_mismatched_source(tmp_path: Path) -> None:
    active_config, _ = _workspace(tmp_path / "active", active_owner=True)
    with pytest.raises(RebuildError, match="owner PID"):
        rebuild_experiment_state(active_config)

    mismatch_config, _ = _workspace(
        tmp_path / "mismatch",
        mismatched_artifact=True,
    )
    with pytest.raises(RebuildError, match="not matched by artifact"):
        rebuild_experiment_state(mismatch_config)


def test_rebuild_accepts_compressed_native_checkpoint_reference(tmp_path: Path) -> None:
    config_path, root = _workspace(tmp_path)
    session_dir = root / "artifacts" / "sessions" / "session-000001"
    artifact_checkpoint = (
        "artifacts/generations/generation-0000/native-generation-checkpoint.json.gz"
    )
    database_summary = {
        "schema_version": "mforge.experiment.session.v2",
        "result": {
            "summary": {
                "checkpoint": artifact_checkpoint.removesuffix(".gz"),
            }
        },
    }
    artifact_summary = {
        **database_summary,
        "result": {
            "summary": {
                "checkpoint": artifact_checkpoint,
            }
        },
    }
    write_json(root / artifact_checkpoint, {"checkpoint": 1})
    write_json(session_dir / "summary.json.gz", artifact_summary)
    write_json(session_dir / "session.json.gz", artifact_summary)
    with sqlite3.connect(root / "state.sqlite3") as connection:
        connection.execute(
            "UPDATE sessions SET summary_json=? WHERE session_id='session-000001'",
            (json.dumps(database_summary),),
        )

    assert rebuild_experiment_state(config_path)["status"] == "checked"


def test_rebuild_accepts_historical_sessions_without_event_streams(
    tmp_path: Path,
) -> None:
    config_path, root = _workspace(tmp_path)
    events_path = (
        root / "artifacts" / "sessions" / "session-000001" / "events.jsonl"
    )
    events_path.unlink()

    checked = rebuild_experiment_state(config_path)

    assert checked["status"] == "checked"
    assert checked["session_event_streams"] == 0


def test_rebuild_refuses_missing_event_stream_after_streams_begin(
    tmp_path: Path,
) -> None:
    config_path, root = _workspace(tmp_path)
    (root / "artifacts" / "sessions" / "session-000002").mkdir()

    with pytest.raises(RebuildError, match="session event stream is missing"):
        rebuild_experiment_state(config_path)


def test_rebuild_accepts_strict_initial_record_for_incomplete_session(
    tmp_path: Path,
) -> None:
    config_path, root = _workspace(tmp_path)
    session_dir = root / "artifacts" / "sessions" / "session-000001"
    (session_dir / "summary.json.gz").unlink()
    write_json(
        session_dir / "session.json.gz",
        {
            "schema_version": "mforge.experiment.session.v2",
            "session_id": "session-000001",
            "session_number": 1,
            "start_time": "2026-08-04T11:00:00+00:00",
            "starting_checkpoint": None,
            "starting_state": "idle",
            "wall_seconds": 60.0,
        },
    )
    with sqlite3.connect(root / "state.sqlite3") as connection:
        connection.execute(
            "UPDATE sessions SET finished_at=NULL,ending_state=NULL,status='running',"
            "exit_status=NULL,summary_json='{}' WHERE session_id='session-000001'"
        )

    checked = rebuild_experiment_state(config_path)

    assert checked["status"] == "checked"
    assert checked["redundant_session_records"] == 0
