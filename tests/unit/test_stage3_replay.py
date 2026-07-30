from __future__ import annotations

import json
from pathlib import Path

from mutation_forge.stage3.replay import canonical_hash, verify_replay


def test_canonical_hash_strips_only_timing_and_paths() -> None:
    first = {
        "schema_version": "stage3.v1",
        "episode_id": "episode-1",
        "started_at": "one",
        "finished_at": "two",
        "elapsed_seconds": 1.0,
        "path": "/tmp/first",
        "nested": {"elapsed_ns": 10, "selected": "stable"},
    }
    second = {**first, "started_at": "different", "path": "/tmp/second"}
    assert canonical_hash(first) == canonical_hash(second)
    changed = {**second, "nested": {"elapsed_ns": 10, "selected": "changed"}}
    assert canonical_hash(first) != canonical_hash(changed)


def test_verify_replay_accepts_json_files_and_never_calls_provider(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    replay = tmp_path / "replay.json"
    payload = {"status": "completed", "decision": "NO_GO", "records": [1, 2]}
    primary.write_text(json.dumps(payload), encoding="utf-8")
    replay.write_text(json.dumps({**payload, "elapsed_seconds": 99}), encoding="utf-8")
    result = verify_replay(primary, replay)
    assert result["status"] == "completed"
    assert result["exact"] is True
    assert result["provider_calls"] == 0


def test_verify_replay_reports_mismatch_and_malformed_input(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    replay = tmp_path / "replay.json"
    primary.write_text(json.dumps({"decision": "GO_TO_STAGE_4", "value": 1}), encoding="utf-8")
    replay.write_text(json.dumps({"decision": "NO_GO", "value": 1}), encoding="utf-8")
    mismatch = verify_replay(primary, replay)
    assert mismatch["status"] == "failed"
    assert mismatch["decision_match"] is False
    replay.write_text("not-json", encoding="utf-8")
    malformed = verify_replay(primary, replay)
    assert malformed["status"] == "failed"
    assert malformed["exact"] is False
    assert malformed["provider_calls"] == 0


def test_verify_replay_rejects_freeze_or_config_identity_mismatch() -> None:
    primary = {
        "decision": "NO_GO",
        "freeze_payload_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "records": [{"episode_id": "e-1", "selected": "p-1"}],
    }
    replay = {
        **primary,
        "freeze_payload_sha256": "c" * 64,
    }
    result = verify_replay(primary, replay)
    assert result["status"] == "failed"
    assert result["decision"] == "mismatch"
    assert result["exact"] is False
    assert result["provider_calls"] == 0


def test_canonical_hash_strips_timing_from_nested_records_but_keeps_hashes() -> None:
    primary = {
        "decision": "NO_GO",
        "records": [{"elapsed_ns": 1, "value": 3, "digest": "a"}],
    }
    replay = {
        "decision": "NO_GO",
        "records": [{"elapsed_ns": 999, "value": 3, "digest": "a"}],
    }
    assert verify_replay(primary, replay)["exact"] is True
    replay["records"][0]["digest"] = "b"
    assert verify_replay(primary, replay)["exact"] is False


def test_canonical_hash_projects_each_record_in_a_record_list() -> None:
    primary = [
        {
            "episode_id": "episode-1",
            "canonical_episode_sha256": "a" * 64,
            "elapsed_seconds": 1.0,
            "value": 3,
        }
    ]
    replay = [
        {
            "episode_id": "episode-1",
            "canonical_episode_sha256": "b" * 64,
            "elapsed_seconds": 99.0,
            "value": 3,
        }
    ]
    assert canonical_hash(primary) == canonical_hash(replay)
