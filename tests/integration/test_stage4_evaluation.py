from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mutation_forge.stage4 import evaluation
from mutation_forge.stage4.replay import verify_replay


class _ProcessConfig:
    limits = SimpleNamespace(reserved_physical_cores=1)

    @staticmethod
    def stable_hash() -> str:
        return "frozen-config"


def _manifest(count: int = 8) -> dict[str, Any]:
    return {
        "episodes": [
            {
                "episode_id": f"e-{index:03d}",
                "order": 10,
                "graph_seed": index,
                "policy_seed": index,
                "horizon": 2,
            }
            for index in range(count)
        ]
    }


def _fake_episode(calls: list[tuple[str, ...]]) -> Any:
    def run(config: object, episode: dict[str, Any], policies: dict[str, Any]) -> dict[str, Any]:
        calls.append(tuple(sorted(policies)))
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
                }
                for index, name in enumerate(policies)
            },
            "policy_identities": {},
            "steps": [],
            "initial_score_calls": 1,
            "selected_score_calls": len(policies),
            "evaluation_count": len(policies),
            "oracle_score_calls": 0,
            "model_calls": 0,
            "app_server_calls": 0,
            "runtime_network_calls": 0,
            "invalid_graphs": 0,
            "timing_ns": {"episode_total": 1},
        }

    return run


def test_one_vs_eight_parity_and_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(evaluation, "run_development_episode", _fake_episode(calls))
    manifest = _manifest()
    primary = evaluation.evaluate_program_manifest(
        {},
        manifest,
        "candidate",
        "source",
        baselines={"random": "r", "structural": "s"},
        output_dir=tmp_path / "one",
        workers=1,
        shard_count=8,
    )
    replay = evaluation.evaluate_program_manifest(
        {},
        manifest,
        "candidate",
        "source",
        baselines={"random": "r", "structural": "s"},
        output_dir=tmp_path / "eight",
        workers=8,
        shard_count=8,
        pass_name="replay",
    )
    assert primary["canonical_reduction_sha256"] == replay["canonical_reduction_sha256"]
    assert primary["metrics_input_sha256"] == replay["metrics_input_sha256"]
    assert verify_replay(primary, replay)["exact"]
    summary = next((tmp_path / "one").glob("*summary.json"))
    assert evaluation.verify_candidate_pass(summary, manifest)["exact"]
    assert set(primary["records"][0]["policies"]) == {"candidate", "random", "structural"}
    assert all(names == ("candidate", "random", "structural") for names in calls)


def test_official_evaluation_uses_an_isolated_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(evaluation, "run_development_episode", _fake_episode([]))
    config = _ProcessConfig()
    result = evaluation.evaluate_policy_roster_manifest(
        config,
        _manifest(1),
        {"random": "r", "structural": "s", "candidate": "c"},
        output_dir=tmp_path,
        workers=1,
        shard_count=1,
    )
    assignment = result["worker_health"]["assignments"][0]
    assert evaluation._uses_process_workers(config)
    assert not evaluation._uses_process_workers({})
    assert assignment["process_id"] != os.getpid()
    assert assignment["observed_affinity"] == [assignment["cpu_id"]]


def test_corrupt_shard_is_recovered_without_rewriting_valid_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(evaluation, "run_development_episode", _fake_episode(calls))
    manifest = _manifest(4)
    first = evaluation.evaluate_program_manifest(
        {},
        manifest,
        "candidate",
        "source",
        baselines={"random": "r", "structural": "s"},
        output_dir=tmp_path,
        workers=2,
        shard_count=2,
    )
    shard_paths = [tmp_path / entry["path"] for entry in first["shards"]]
    preserved = shard_paths[0].read_bytes()
    shard_paths[1].write_bytes(b"corrupt")
    calls.clear()
    recovered = evaluation.evaluate_program_manifest(
        {},
        manifest,
        "candidate",
        "source",
        baselines={"random": "r", "structural": "s"},
        output_dir=tmp_path,
        workers=2,
        shard_count=2,
    )
    assert recovered["canonical_reduction_sha256"] == first["canonical_reduction_sha256"]
    assert shard_paths[0].read_bytes() == preserved
    assert len(calls) == 2


def test_verify_candidate_pass_rejects_duplicate_or_missing(tmp_path: Path) -> None:
    # Empty passes are never admissible evidence.
    evaluation.evaluate_program_manifest(
        {},
        _manifest(0),
        "candidate",
        "source",
        baselines={"random": "r", "structural": "s"},
        output_dir=tmp_path,
        workers=1,
        shard_count=1,
    )
    summary = tmp_path / next(path.name for path in tmp_path.glob("*summary.json"))
    assert not evaluation.verify_candidate_pass(summary)["exact"]
    raw = json.loads(summary.read_text())
    raw["canonical_reduction_sha256"] = "0" * 64
    summary.write_text(json.dumps(raw))
    assert not evaluation.verify_candidate_pass(summary)["exact"]


def test_validation_manifest_has_fixed_allowlist_and_disjoint_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evaluation, "run_development_episode", _fake_episode([]))
    candidates = {
        name: name for name in ("champion", "stage3-candidate-slot-04", "random", "structural")
    }
    result = evaluation.evaluate_validation_manifest(
        {},
        {"episodes": [{"episode_id": "held-out", "horizon": 2}]},
        candidates,
        search_manifest=_manifest(1),
        output_dir=tmp_path,
        workers=1,
        shard_count=1,
    )
    assert set(result) == set(candidates)
    with pytest.raises(ValueError):
        evaluation.evaluate_validation_manifest(
            {}, _manifest(1), candidates, search_manifest=_manifest(1), output_dir=tmp_path
        )
    with pytest.raises(ValueError):
        evaluation.evaluate_validation_manifest(
            {}, {"episodes": [{"episode_id": "held-out"}]}, {"random": "r"}, output_dir=tmp_path
        )


def test_replay_directory_rejects_ambiguous_summaries(tmp_path: Path) -> None:
    (tmp_path / "stage4-a-summary.json").write_text("{}")
    (tmp_path / "stage4-b-summary.json").write_text("{}")
    report = verify_replay(tmp_path, tmp_path)
    assert not report["exact"]
    assert "exactly one summary" in report["error"]


def test_campaign_roster_evaluates_each_episode_once_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(evaluation, "run_development_episode", _fake_episode(calls))
    manifest = _manifest(8)
    policies = {
        "random": "random-source",
        "structural": "structural-source",
        "champion": "champion-source",
    }
    primary = evaluation.evaluate_policy_roster_manifest(
        {}, manifest, policies, output_dir=tmp_path / "primary", workers=1, shard_count=8
    )
    assert len(calls) == len(manifest["episodes"])
    assert set(primary["records"][0]["policies"]) == set(policies)
    calls.clear()
    replay = evaluation.evaluate_policy_roster_manifest(
        {},
        manifest,
        policies,
        output_dir=tmp_path / "replay",
        workers=8,
        shard_count=8,
        pass_name="replay",
    )
    assert len(calls) == len(manifest["episodes"])
    assert primary["canonical_reduction_sha256"] == replay["canonical_reduction_sha256"]
    assert primary["metrics_input_sha256"] == replay["metrics_input_sha256"]
    assert verify_replay(primary, replay)["exact"]
    calls.clear()
    resumed = evaluation.evaluate_policy_roster_manifest(
        {}, manifest, policies, output_dir=tmp_path / "primary", workers=8, shard_count=8
    )
    assert calls == []
    assert resumed["canonical_reduction_sha256"] == primary["canonical_reduction_sha256"]


def test_campaign_roster_rejects_episode_id_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _fake_episode([])

    def wrong_id(
        config: object,
        episode: dict[str, Any],
        policies: dict[str, Any],
    ) -> dict[str, Any]:
        result = run(config, episode, policies)
        if episode["episode_id"] == "e-001":
            result["episode_id"] = "e-extra"
        return result

    monkeypatch.setattr(evaluation, "run_development_episode", wrong_id)
    with pytest.raises(ValueError, match="authoritative episode roster"):
        evaluation.evaluate_policy_roster_manifest(
            {},
            _manifest(2),
            {
                "random": "random-source",
                "structural": "structural-source",
                "candidate": "candidate-source",
            },
            output_dir=tmp_path,
            workers=1,
            shard_count=2,
        )
