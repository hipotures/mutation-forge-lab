from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge.config import load_config
from mutation_forge.evaluation.benchmark import run_benchmark


def _write_config(
    path: Path,
    heg_repo: Path,
    run_root: Path,
    *,
    wall_seconds: float = 60,
) -> None:
    path.write_text(
        f"""
schema_version = "1.0"
[run]
seed = 1
wall_seconds = {wall_seconds}
output = "json"
run_root = {json.dumps(str(run_root))}
[heg]
repo = {json.dumps(str(heg_repo))}
[dataset]
orders = [10]
graph_seeds = [101]
policy_seeds = [1]
split = "test"
[score]
witness_cap = 8
[search]
controller = "fixed_ils_tabu"
evaluations_per_episode = 3
proposal_pool_size = 1
[proposals]
operator_families = ["heg_uniform_two_switch", "heg_forbidden_cycle_break"]
k_values = [2]
""".strip()
        + "\n"
    )


def test_json_and_rich_runs_have_same_canonical_summary(
    tmp_path: Path,
    heg_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, heg_repo, tmp_path / "runs")
    config = load_config(config_path)
    json_result = run_benchmark(config, output="json")
    json_stdout = capsys.readouterr().out
    json_events = [json.loads(line) for line in json_stdout.splitlines()]
    assert json_events
    assert json_events[-1]["event_type"] == "run_completed"
    assert all(
        json_events[-1][field] >= 0
        for field in ("real_seconds", "user_seconds", "system_seconds")
    )
    assert json_events[-1]["timing_profile"]["enabled"] is True
    assert json_events[-1]["timing_profile"]["profiled_episodes"] == 2
    completed_events = [
        event for event in json_events if event["event_type"] == "episode_completed"
    ]
    assert [
        event["timing_profile"]["profiled_episodes"] for event in completed_events
    ] == [1, 2]
    assert all(event["episode_timing_profile"] for event in completed_events)
    assert "\x1b" not in json_stdout
    assert all(
        {"schema_version", "timestamp", "run_id", "event_type"}.issubset(event)
        for event in json_events
    )
    rich_result = run_benchmark(config, output="rich")
    assert json_result.summary["summary_hash"] == rich_result.summary["summary_hash"]
    for result in (json_result, rich_result):
        assert (result.run_path / "events.jsonl").is_file()
        assert (result.run_path / "archive.sqlite3").is_file()
        assert (result.run_path / "dataset_manifest.json").is_file()
        assert result.summary["status"] == "completed"
        assert result.summary["real_seconds"] == result.summary["elapsed_seconds"]
        assert result.summary["real_seconds"] > 0
        assert result.summary["user_seconds"] >= 0
        assert result.summary["system_seconds"] >= 0
        timing_profile = result.summary["timing_profile"]
        assert timing_profile["enabled"] is True
        assert timing_profile["profiled_episodes"] == 2
        assert timing_profile["measured_total_seconds"] > 0
        assert timing_profile["accounted_seconds"] > 0
        assert timing_profile["unattributed_seconds"] >= 0
        assert timing_profile["dominant_phase"] in timing_profile["phase_seconds"]
        proposal_children = timing_profile["phase_children_seconds"][
            "proposal_generation"
        ]
        assert set(proposal_children) == {
            "rng_setup",
            "graph_materialization",
            "operator_search",
            "proposal_packaging",
            "other",
        }
        assert sum(proposal_children.values()) == pytest.approx(
            timing_profile["phase_seconds"]["proposal_generation"]
        )
        assert timing_profile["phase_calls"]["proposal_generation"] == 6
        assert timing_profile["phase_children_calls"]["proposal_generation"] == {
            "rng_setup": 6,
            "graph_materialization": 6,
            "operator_search": 6,
            "proposal_packaging": 6,
            "other": None,
        }
        operator_seconds = timing_profile["phase_grandchildren_seconds"][
            "proposal_generation"
        ]["operator_search"]
        assert set(operator_seconds) == {
            "heg_uniform_two_switch",
            "heg_forbidden_cycle_break",
        }
        assert sum(operator_seconds.values()) == pytest.approx(
            proposal_children["operator_search"]
        )
        operator_calls = timing_profile["phase_grandchildren_calls"][
            "proposal_generation"
        ]["operator_search"]
        assert operator_calls == {
            "heg_uniform_two_switch": 3,
            "heg_forbidden_cycle_break": 3,
        }
        assert sum(operator_calls.values()) == 6
        episodes = result.summary["episodes"]
        assert isinstance(episodes, list)
        assert all(episode["evaluations"] == 3 for episode in episodes)
        assert all("timing_profile" in episode for episode in episodes)


def test_interrupted_run_leaves_readable_failure_artifacts(
    tmp_path: Path, heg_repo: Path
) -> None:
    config_path = tmp_path / "timeout.toml"
    run_root = tmp_path / "runs"
    _write_config(
        config_path,
        heg_repo,
        run_root,
        wall_seconds=0.000001,
    )
    with pytest.raises(TimeoutError):
        run_benchmark(load_config(config_path), output="json")
    run_paths = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(run_paths) == 1
    summary = json.loads((run_paths[0] / "run_summary.json").read_text())
    manifest = json.loads((run_paths[0] / "run_manifest.json").read_text())
    events = [
        json.loads(line)
        for line in (run_paths[0] / "events.jsonl").read_text().splitlines()
    ]
    assert summary["status"] == "failed"
    assert summary["error_type"] == "TimeoutError"
    assert summary["real_seconds"] == summary["elapsed_seconds"]
    assert summary["user_seconds"] >= 0
    assert summary["system_seconds"] >= 0
    assert manifest["status"] == "failed"
    assert events[-1]["event_type"] == "run_failed"
    assert all(
        events[-1][field] >= 0
        for field in ("real_seconds", "user_seconds", "system_seconds")
    )
