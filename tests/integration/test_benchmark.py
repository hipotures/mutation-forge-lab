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
        episodes = result.summary["episodes"]
        assert isinstance(episodes, list)
        assert all(episode["evaluations"] == 3 for episode in episodes)


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
    assert manifest["status"] == "failed"
    assert events[-1]["event_type"] == "run_failed"
