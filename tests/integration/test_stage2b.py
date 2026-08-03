from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.cli import _emit_policy_result
from mutation_forge.proposals.k_switch import (
    KSwitchPoolGenerator,
    make_scientific_context,
)
from mutation_forge.stage2b.config import load_stage2b_config
from mutation_forge.stage2b.evaluation import (
    inspect_proposals,
    run_heg_pilot,
    run_stage2b_compare,
    run_toy_gate,
)
from mutation_forge.stage2b.rankers import SourceRanker


def _sources(project_root: Path) -> tuple[str, str]:
    root = project_root / "fixtures" / "rankers"
    return (
        (root / "stage2b_random.py").read_text(),
        (root / "stage2b_structural.py").read_text(),
    )


def test_paired_rankers_receive_identical_pool_through_stage2a_worker(
    project_root: Path,
) -> None:
    config = load_stage2b_config(project_root / "configs" / "stage2b-preregistered.toml")
    backend = ToyBackend()
    graph = backend.generate_seed(order=8, seed=101)
    score = backend.score(graph, witness_cap=config.search.witness_cap)
    assert score is not None
    pool = KSwitchPoolGenerator(
        backend,
        pool_limits=config.pool,
        feature_limits=replace(
            config.features,
            forbidden_lengths=backend.target_forbidden_lengths(graph.order),
        ),
    ).generate(graph, policy_seed=1, step=0)
    context = make_scientific_context(
        graph,
        score,
        forbidden_lengths=backend.target_forbidden_lengths(graph.order),
        step=0,
        remaining_steps=7,
    )
    random_source, structural_source = _sources(project_root)
    with (
        SourceRanker("random", random_source, config.sandbox) as random_ranker,
        SourceRanker(
            "structural",
            structural_source,
            config.sandbox,
        ) as structural_ranker,
    ):
        random_result = random_ranker.rank(context, pool)
        structural_result = structural_ranker.rank(context, pool)
    assert random_result.pool_hash == structural_result.pool_hash == pool.pool_hash
    assert len(random_result.ranked) == len(structural_result.ranked) == pool.retained
    assert not any(
        (
            random_result.exception,
            random_result.timeout,
            random_result.crash,
            random_result.protocol,
            structural_result.exception,
            structural_result.timeout,
            structural_result.crash,
            structural_result.protocol,
        )
    )


def test_preregistered_toy_benchmark_records_no_go(project_root: Path) -> None:
    config = load_stage2b_config(project_root / "configs" / "stage2b-preregistered.toml")
    random_source, structural_source = _sources(project_root)
    result = run_toy_gate(config, random_source, structural_source)
    assert result["status"] == "failed"
    assert len(result["paired_runs"]) == 32
    assert result["criteria"] == {
        "paired_policy_seeds_at_least_32": True,
        "relative_auc_improvement_at_least_threshold": False,
        "paired_bootstrap_ci_excludes_zero": False,
        "structural_best_total_no_worse": True,
        "zero_invalid_host_applied_graphs": True,
        "zero_structural_timeouts_or_crashes": True,
    }
    assert result["metrics"]["relative_auc_improvement"] == 0.0
    for run in result["paired_runs"]:
        assert run["pool_generation"]["retained"] == (config.search.steps * config.pool.pool_size)
        maximums = run["pool_generation"]["feature_usage_maximums"]
        assert maximums["cycle_nodes"] <= config.features.cycle_node_budget
        assert maximums["distance_queries"] <= (config.features.distance_query_budget)
        assert maximums["local_risk_operations"] <= (config.features.local_risk_budget)
    assert len(result["behavior_signature"]["signature_sha256"]) == 64


def test_proposal_inspection_is_machine_bounded(project_root: Path) -> None:
    config = load_stage2b_config(project_root / "configs" / "stage2b-preregistered.toml")
    result = inspect_proposals(config)
    assert result["status"] == "completed"
    assert len(json.dumps(result).encode()) < 65536
    assert result["pool"]["telemetry"]["retained"] <= config.pool.pool_size


def test_stage2b_rich_and_json_render_same_canonical_result(
    project_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_stage2b_config(project_root / "configs" / "stage2b-preregistered.toml")
    result = inspect_proposals(config)
    _emit_policy_result(result, json_output=True)
    json_output = capsys.readouterr().out
    _emit_policy_result(result, json_output=False)
    rich_output = capsys.readouterr().out
    assert json.loads(json_output) == json.loads(rich_output) == result


def test_bounded_heg_pilot_replays_without_modifying_heg(
    project_root: Path,
    heg_repo: Path,
) -> None:
    config = load_stage2b_config(project_root / "configs" / "stage2b-preregistered.toml")
    config = replace(
        config,
        repositories=replace(config.repositories, heg_repo=heg_repo),
        heg_pilot=replace(config.heg_pilot, steps=1),
    )
    random_source, structural_source = _sources(project_root)
    result = run_heg_pilot(config, random_source, structural_source)
    assert result["status"] == "completed"
    assert result["deterministic_replay"]
    assert result["all_graphs_valid"]
    assert result["score_calls_match_selected_only"]
    assert not result["hidden_best_of_k_scoring"]
    assert result["rich_json_canonical_equal"]
    assert result["heg_commit"] == "fd97451b0f3d87400d1d955a2c6b1b18303344ff"


def test_interrupted_compare_leaves_terminal_artifact(
    tmp_path: Path,
    project_root: Path,
    heg_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_stage2b_config(project_root / "configs" / "stage2b-preregistered.toml")
    config = replace(
        config,
        run=replace(config.run, run_root=tmp_path),
        repositories=replace(config.repositories, heg_repo=heg_repo),
        heg_pilot=replace(config.heg_pilot, enabled=False),
    )

    def interrupt(*args: object, **kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt("forced interruption")

    monkeypatch.setattr(
        "mutation_forge.stage2b.evaluation.run_toy_gate",
        interrupt,
    )
    root = project_root / "fixtures" / "rankers"
    with pytest.raises(KeyboardInterrupt):
        run_stage2b_compare(
            root / "stage2b_random.py",
            root / "stage2b_structural.py",
            config,
        )
    run_path = next(tmp_path.glob("stage2b-*"))
    terminal = json.loads((run_path / "terminal_status.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "KeyboardInterrupt"
