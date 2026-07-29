from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from mutation_forge.models import JsonValue
from mutation_forge.stage2b.rankers import SourceRanker
from mutation_forge.stage2d.config import load_stage2d_config
from mutation_forge.stage2d.evaluation import (
    STAGE2D_ARTIFACT_VERSION,
    THREAD_ENVIRONMENT,
    _sha256,
    _write_episode_file,
    reduce_stage2d,
    run_stage2d_shard,
    run_trajectory_episode,
    verify_stage2d_replay,
)
from mutation_forge.stage2d.manifest import load_manifest, validate_manifest
from mutation_forge.stage2d.statistics import (
    hierarchical_bootstrap,
    summarize_episodes,
)


def _config(project_root: Path):
    return load_stage2d_config(
        project_root / "configs" / "stage2d-preregistered.toml"
    )


def _rankers(config):
    return (
        SourceRanker(
            "random",
            config.inputs.random_policy.read_text(),
            config.stage2b.sandbox,
        ),
        SourceRanker(
            "structural",
            config.inputs.structural_policy.read_text(),
            config.stage2b.sandbox,
        ),
    )


def test_stage2d_preregistration_is_exact_and_partitioned(
    project_root: Path,
) -> None:
    config = _config(project_root)
    manifest = load_manifest(config)
    assert config.experiment.orders == (10, 12)
    assert config.experiment.graph_seeds == tuple(range(201, 209))
    assert config.experiment.policy_seeds == tuple(range(1001, 1033))
    assert config.experiment.horizon == 32
    assert config.statistics.bootstrap_samples == 10_000
    assert config.statistics.bootstrap_seed == 2026072902
    assert manifest["episode_count"] == 512
    shards = cast(list[dict[str, JsonValue]], manifest["shards"])
    assert len(shards) == 8
    assert {cast(int, shard["episode_count"]) for shard in shards} == {64}
    ids = [
        episode_id
        for shard in shards
        for episode_id in cast(list[str], shard["episode_ids"])
    ]
    assert len(ids) == len(set(ids)) == 512
    cpu_ids = {
        cast(
            int,
            cast(dict[str, JsonValue], shard["affinity"])["cpu_id"],
        )
        for shard in shards
    }
    assert len(cpu_ids) == 8
    topology = cast(dict[str, JsonValue], manifest["cpu_topology"])
    assert cast(int, topology["reserved_physical_cores"]) >= 8
    assert manifest["thread_environment"] == THREAD_ENVIRONMENT


def test_manifest_rejects_duplicate_assignment(project_root: Path) -> None:
    config = _config(project_root)
    manifest = load_manifest(config)
    tampered = json.loads(json.dumps(manifest))
    tampered["shards"][1]["episode_ids"][0] = tampered["shards"][0]["episode_ids"][0]
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_manifest(config, tampered)


def test_real_trajectory_is_deterministic_strict_and_oracle_free(
    project_root: Path,
) -> None:
    config = _config(project_root)
    episode: dict[str, JsonValue] = {
        "episode_id": "smoke-o10-g0901-p9001",
        "order": 10,
        "graph_seed": 901,
        "policy_seed": 9001,
        "horizon": 4,
        "shard_id": "smoke",
    }
    first_rankers = _rankers(config)
    try:
        first = run_trajectory_episode(
            config,
            episode,
            first_rankers[0],
            first_rankers[1],
        )
    finally:
        first_rankers[0].close()
        first_rankers[1].close()
    second_rankers = _rankers(config)
    try:
        second = run_trajectory_episode(
            config,
            episode,
            second_rankers[0],
            second_rankers[1],
        )
    finally:
        second_rankers[0].close()
        second_rankers[1].close()
    assert first["canonical_episode_sha256"] == second["canonical_episode_sha256"]
    assert first["selected_score_calls"] == 8
    assert first["oracle_score_calls"] == 0
    assert first["network_calls"] == first["model_calls"] == first["app_server_calls"] == 0
    assert first["shared_pool_steps"] + first["independent_pool_steps"] == 4
    steps = cast(list[dict[str, JsonValue]], first["steps"])
    for step in steps:
        if step["states_identical_before_step"]:
            assert step["same_pool"]
        else:
            assert not step["same_pool"]
    for policy in ("random", "structural"):
        summary = cast(dict[str, JsonValue], first[policy])
        previous = tuple(
            cast(
                list[int],
                cast(dict[str, JsonValue], summary["initial_score"])["ordering_key"],
            )
        )
        trace = cast(list[dict[str, JsonValue]], summary["trace"])
        for step in trace:
            current = tuple(
                cast(
                    list[int],
                    cast(dict[str, JsonValue], step["current_score"])[
                        "ordering_key"
                    ],
                )
            )
            if step["accepted"]:
                assert current < previous
            else:
                assert current == previous
            previous = current


def _synthetic_episodes(
    config,
    *,
    structural_primary_auc: float,
) -> list[dict[str, JsonValue]]:
    episodes: list[dict[str, JsonValue]] = []
    for order in config.experiment.orders:
        for graph_seed in config.experiment.graph_seeds:
            for policy_seed in config.experiment.policy_seeds:
                structural_auc = structural_primary_auc if order == 10 else 0.30
                episodes.append(
                    {
                        "episode_id": (
                            f"o{order:02d}-g{graph_seed:04d}-p{policy_seed:04d}"
                        ),
                        "order": order,
                        "graph_seed": graph_seed,
                        "policy_seed": policy_seed,
                        "random": {
                            "auc": 0.20,
                            "best_total_witnesses": 2,
                            "best_score": {
                                "capped_cycle_counts": [[4, 2]],
                                "weighted_penalty": 32,
                                "ordering_key": [0, 2, 32],
                            },
                            "evaluations_to_first_improvement": 1,
                            "accepted_count": 1,
                            "rejected_count": 31,
                            "duplicate_count": 0,
                        },
                        "structural": {
                            "auc": structural_auc,
                            "best_total_witnesses": 1,
                            "best_score": {
                                "capped_cycle_counts": [[4, 1]],
                                "weighted_penalty": 16,
                                "ordering_key": [0, 1, 16],
                            },
                            "evaluations_to_first_improvement": 1,
                            "accepted_count": 1,
                            "rejected_count": 31,
                            "duplicate_count": 0,
                        },
                        "invalid_graphs": 0,
                        "policy_failures": 0,
                        "initial_score_calls": 1,
                        "selected_score_calls": 64,
                        "oracle_score_calls": 0,
                    }
                )
    return episodes


def _write_synthetic_shards(
    root: Path,
    config,
    manifest: dict[str, JsonValue],
) -> None:
    records = {
        cast(str, episode["episode_id"]): episode
        for episode in _synthetic_episodes(
            config,
            structural_primary_auc=0.23,
        )
    }
    for index, shard in enumerate(
        cast(list[dict[str, JsonValue]], manifest["shards"])
    ):
        shard_id = cast(str, shard["shard_id"])
        shard_dir = root / shard_id
        shard_dir.mkdir(parents=True)
        episodes = [
            records[episode_id]
            for episode_id in cast(list[str], shard["episode_ids"])
        ]
        file_manifest = _write_episode_file(
            shard_dir / "episodes.jsonl.gz",
            episodes,
            config,
        )
        base: dict[str, JsonValue] = {
            "schema_version": STAGE2D_ARTIFACT_VERSION,
            "status": "completed",
            "shard_id": shard_id,
            "assignment_sha256": shard["assignment_sha256"],
            "config_sha256": config.stable_hash(),
            "manifest_sha256": manifest["manifest_sha256"],
            "preregistration_commit": "a" * 40,
            "episode_count": 64,
            "canonical_episode_sha256": file_manifest[
                "canonical_episode_sha256"
            ],
        }
        result: dict[str, JsonValue] = {
            **base,
            "shard_hash": _sha256(base),
            "provenance": {
                "runtime_network_calls": 0,
                "model_calls": 0,
                "app_server_calls": 0,
                "diagnostic_oracle_enabled": False,
                "stage3_started": False,
                "heg": {
                    "commit": config.repositories.frozen_heg_commit,
                    "dirty": False,
                },
            },
            "environment": {
                "tmpdir": f"/tmp/stage2d-test-{index}/tmp",
                "uv_cache_dir": f"/tmp/stage2d-test-{index}/uv",
                "xdg_cache_home": f"/tmp/stage2d-test-{index}/xdg",
                "affinity": {
                    "physical_id": f"0:{index}",
                },
            },
        }
        (shard_dir / "result.json").write_text(json.dumps(result))
        (shard_dir / "terminal_status.json").write_text(
            json.dumps({"status": "completed"})
        )


def test_reducer_is_exact_once_and_worker_count_independent(
    tmp_path: Path,
    project_root: Path,
) -> None:
    config = _config(project_root)
    manifest = load_manifest(config)
    input_root = tmp_path / "input"
    _write_synthetic_shards(input_root, config, manifest)
    single = reduce_stage2d(
        config,
        input_root,
        tmp_path / "single",
        bootstrap_workers=1,
    )
    parallel = reduce_stage2d(
        config,
        input_root,
        tmp_path / "parallel",
        bootstrap_workers=8,
    )
    assert single["reduction_sha256"] == parallel["reduction_sha256"]
    assert single["aggregate_sha256"] == parallel["aggregate_sha256"]
    assert single["metrics"] == parallel["metrics"]
    missing = input_root / "shard-07" / "result.json"
    missing.rename(missing.with_suffix(".missing"))
    with pytest.raises(FileNotFoundError):
        reduce_stage2d(
            config,
            input_root,
            tmp_path / "missing",
            bootstrap_workers=1,
        )
    terminal = json.loads(
        (tmp_path / "missing" / "terminal_status.json").read_text()
    )
    assert terminal["status"] == "failed"


def test_bootstrap_worker_parity_and_gate_boundaries(project_root: Path) -> None:
    config = _config(project_root)
    reduced = replace(
        config,
        statistics=replace(config.statistics, bootstrap_samples=64),
    )
    passing = _synthetic_episodes(reduced, structural_primary_auc=0.23)
    assert hierarchical_bootstrap(passing, reduced, workers=1) == (
        hierarchical_bootstrap(passing, reduced, workers=8)
    )
    metrics = summarize_episodes(passing, reduced, bootstrap_workers=8)
    gate = cast(dict[str, JsonValue], metrics["gate_without_replay"])
    assert all(cast(bool, value) for value in gate.values())
    failing = _synthetic_episodes(reduced, structural_primary_auc=0.219)
    failing_metrics = summarize_episodes(
        failing,
        reduced,
        bootstrap_workers=1,
    )
    failing_gate = cast(
        dict[str, JsonValue], failing_metrics["gate_without_replay"]
    )
    assert not failing_gate["primary_relative_median_at_least_10_percent"]


def test_replay_verification_returns_each_allowed_decision(
    tmp_path: Path,
) -> None:
    gate = {
        "primary_relative_median_at_least_10_percent": True,
        "primary_bootstrap_lower_bound_above_zero": True,
        "pooled_stratified_bootstrap_lower_bound_above_zero": True,
        "secondary_median_delta_nonnegative": True,
        "structural_witness_count_no_worse_each_order": True,
        "at_least_six_primary_graph_seeds_nonnegative": True,
        "graph_validity_100_percent": True,
        "policy_failure_rate_zero": True,
        "selected_plan_only_scoring_no_oracle": True,
    }
    base = {
        "config_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "preregistration_commit": "c" * 40,
        "episode_count": 512,
        "canonical_episode_sha256": "d" * 64,
        "aggregate_sha256": "e" * 64,
        "shard_hashes": {"shard-00": "f" * 64},
        "metrics": {"gate_without_replay": gate},
        "reduction_sha256": "1" * 64,
        "validation": {
            "exact_once_coverage": True,
            "provenance_pass": True,
            "unique_run_cache_temp_directories": True,
            "distinct_physical_core_affinity": True,
        },
    }
    primary = tmp_path / "primary.json"
    replay = tmp_path / "replay.json"
    primary.write_text(json.dumps(base))
    replay.write_text(json.dumps(base))
    result = verify_stage2d_replay(primary, replay, tmp_path / "go.json")
    assert result["decision"] == "GO_TO_STAGE_3"
    no_go = json.loads(json.dumps(base))
    no_go["metrics"]["gate_without_replay"][
        "primary_relative_median_at_least_10_percent"
    ] = False
    primary.write_text(json.dumps(no_go))
    replay.write_text(json.dumps(no_go))
    result = verify_stage2d_replay(primary, replay, tmp_path / "no-go.json")
    assert result["decision"] == "NO_GO"
    replay.write_text(json.dumps(base))
    result = verify_stage2d_replay(
        primary,
        replay,
        tmp_path / "inconclusive.json",
    )
    assert result["decision"] == "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"


def test_interrupted_shard_leaves_terminal_artifact(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(project_root)
    monkeypatch.setattr(
        "mutation_forge.stage2d.evaluation._preregistration_provenance",
        lambda _: {
            "preregistration_commit": "a" * 40,
        },
    )
    monkeypatch.setattr(
        "mutation_forge.stage2d.evaluation._verify_affinity",
        lambda _: {"supported": True},
    )
    monkeypatch.setattr(
        "mutation_forge.stage2d.evaluation.run_trajectory_episode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic interruption")
        ),
    )
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_stage2d_shard(config, "shard-00", output)
    terminal = json.loads((output / "terminal_status.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "RuntimeError"
