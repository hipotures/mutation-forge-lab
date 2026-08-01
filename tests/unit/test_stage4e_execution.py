from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mutation_forge import stage4e_execution


def _manifest() -> dict[str, Any]:
    return {
        "episodes": [
            {
                "episode_id": f"e-{index:04d}",
                "order": 10,
                "graph_seed": index,
                "policy_seed": index,
                "horizon": 2,
            }
            for index in range(stage4e_execution.EPISODE_COUNT)
        ]
    }


def _fake_episode(
    _config: object, episode: dict[str, Any], policies: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": "stage3.development.episode.v2",
        "terminal_status": "completed",
        "episode_id": episode["episode_id"],
        "horizon": episode["horizon"],
        "policies": {
            name: {
                "auc": float(index),
                "best_total_witnesses": 2,
                "accepted_count": index,
                "failure_count": 0,
                "first_improvement_ns": index,
            }
            for index, name in enumerate(policies)
        },
        "policy_identities": {},
        "steps": [],
        "initial_score_calls": 1,
        "selected_score_calls": 4,
        "evaluation_count": 4,
        "oracle_score_calls": 0,
        "model_calls": 0,
        "app_server_calls": 0,
        "runtime_network_calls": 0,
    }


def test_primary_replay_are_timing_independent_and_use_public_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(stage4e_execution, "run_development_episode", _fake_episode)
    policies = {"champion": "champion-source", "stage3-comparator": "comparator-source"}
    result = stage4e_execution.execute_stage4e_confirmation(
        {}, _manifest(), policies, tmp_path, workers=2
    )
    primary, replay = result["primary"], result["replay"]
    assert result["replay_verification"]["exact"]
    assert primary["record_count"] == stage4e_execution.EPISODE_COUNT
    assert primary["shard_count"] == 24
    assert all(entry["record_count"] == 64 for entry in primary["shards"])
    assert set(primary["records"][0]["policies"]) == set(policies)
    assert "first_improvement_ns" not in primary["records"][0]["policies"]["champion"]
    assert primary["canonical_reduction_sha256"] == replay["canonical_reduction_sha256"]
    summary = next((tmp_path / "primary").glob("*summary.json"))
    assert stage4e_execution.verify_stage4e_pass(summary, _manifest())["exact"]


def test_resume_rejects_an_interrupted_run_after_first_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(stage4e_execution, "run_development_episode", _fake_episode)
    manifest, policies = _manifest(), {"a": "one", "b": "two"}
    rows = stage4e_execution._episodes(manifest)
    identity = stage4e_execution._identity(
        {}, manifest, rows, policies
    )[0]
    state_path, _ = stage4e_execution._paths(tmp_path, identity, "primary")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"outcomes_persisted": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="only before the first"):
        stage4e_execution.execute_stage4e_pass(
            {}, manifest, policies, tmp_path, "primary", workers=1
        )
