from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.cli import _stage2c_diagnostic
from mutation_forge.models import GraphState, RewritePlan, normalized_edge
from mutation_forge.proposals.k_switch import (
    FeatureLimits,
    KSwitchPoolGenerator,
    ProposalCandidate,
    _FeatureSnapshot,
)
from mutation_forge.stage2b.evaluation import run_toy_gate
from mutation_forge.stage2c.config import (
    Stage2CMatrixConfig,
    load_stage2c_config,
)
from mutation_forge.stage2c.evaluation import (
    BoundedRecordWriter,
    _canonical_cell_summary,
    _render_parity_proof,
    run_diagnostic_cell,
    run_pool_oracle,
    verify_stage2b_control,
)
from mutation_forge.stage2c.metrics import FeatureAnalyzer, flatten_proposal_features


def _config(project_root: Path):
    return load_stage2c_config(project_root / "configs" / "stage2c-diagnostic.toml")


def test_stage2c_matrix_is_strict_and_preregistered(project_root: Path) -> None:
    config = _config(project_root)
    assert config.schema_version == "stage2c.1"
    assert config.matrix.orders == (8, 10, 12)
    assert config.matrix.graph_seeds == (101, 102, 103, 104)
    assert config.matrix.policy_seeds == tuple(range(1, 33))
    assert config.matrix.horizons == (8, 16, 32)
    assert config.stage2b.stable_hash() == config.control.expected_config_hash
    assert config.repositories.frozen_project_commit == (
        "5b949f84dca77474d242665152300521fbe8dd31"
    )


def test_documented_control_cli_accepts_frozen_stage2b_config(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(project_root)
    loaded: list[Path] = []

    def load(path: Path):
        loaded.append(path.resolve())
        return config

    monkeypatch.setattr("mutation_forge.cli.load_stage2c_config", load)
    monkeypatch.setattr(
        "mutation_forge.cli.run_stage2c_control",
        lambda _: {"status": "completed"},
    )
    assert (
        _stage2c_diagnostic(
            "stage2c-control",
            project_root / "configs" / "stage2b-preregistered.toml",
            json_output=True,
        )
        == 0
    )
    assert loaded == [
        (project_root / "configs" / "stage2c-diagnostic.toml").resolve()
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_stage2c_rich_json_parity_is_measured() -> None:
    proof = _render_parity_proof(
        {
            "status": "completed",
            "nested": {"values": [1, 2, 3]},
        }
    )
    assert proof["equal"]
    assert proof["canonical_payload_sha256"] == proof["json_roundtrip_sha256"]
    assert proof["json_roundtrip_sha256"] == proof["rich_roundtrip_sha256"]


def test_replay_hashes_exclude_timing_and_artifact_paths(
    tmp_path: Path,
    project_root: Path,
) -> None:
    config = _config(project_root)
    first = BoundedRecordWriter(tmp_path / "first", config)
    second = BoundedRecordWriter(tmp_path / "second", config)
    first.write({"pool_hash": "a" * 64, "scoring_ns": 10})
    second.write({"pool_hash": "a" * 64, "scoring_ns": 20})
    first_manifest = first.close()
    second_manifest = second.close()
    assert first_manifest["raw_records_sha256"] != (
        second_manifest["raw_records_sha256"]
    )
    assert first_manifest["canonical_records_sha256"] == (
        second_manifest["canonical_records_sha256"]
    )
    first_cell = {
        "canonical_hash": "b" * 64,
        "timing_ns": {"ranker": 10},
        "feature_diagnostics_artifact": "/first/run/cell.json",
    }
    second_cell = {
        "canonical_hash": "b" * 64,
        "timing_ns": {"ranker": 20},
        "feature_diagnostics_artifact": "/second/run/cell.json",
    }
    assert _canonical_cell_summary(first_cell) == _canonical_cell_summary(
        second_cell
    )


def test_diagnostic_oracle_is_opt_in_and_trajectory_isolated(
    project_root: Path,
) -> None:
    config = _config(project_root)
    without_oracle = run_diagnostic_cell(
        config,
        order=8,
        graph_seed=101,
        policy_seeds=(1,),
        horizon=2,
    )
    with_oracle = run_diagnostic_cell(
        config,
        order=8,
        graph_seed=101,
        policy_seeds=(1,),
        horizon=2,
        oracle_enabled=True,
    )
    without_episode = without_oracle["episodes"][0]
    with_episode = with_oracle["episodes"][0]
    assert without_episode["trajectory_hash"] == with_episode["trajectory_hash"]
    assert without_episode["stage2b_compatible_trace"] == (
        with_episode["stage2b_compatible_trace"]
    )
    assert without_oracle["feature_diagnostics"]["status"] == "disabled"
    assert with_oracle["aggregates"]["accounting"] == {
        "selected_score_calls": 4,
        "oracle_score_calls": 24,
        "exact_verify_calls": 0,
        "hidden_best_of_pool_scoring_in_normal_search": False,
    }
    assert with_oracle["aggregates"]["invalid_host_applied_graphs"] == 0


def test_stage2b_selected_only_path_remains_unchanged(project_root: Path) -> None:
    config = _config(project_root)
    root = project_root / "fixtures" / "rankers"
    result = run_toy_gate(
        config.stage2b,
        (root / "stage2b_random.py").read_text(),
        (root / "stage2b_structural.py").read_text(),
    )
    assert result["status"] == "failed"
    assert all(run["score_calls"] == 17 for run in result["paired_runs"])
    assert result["metrics"]["median_random_best_so_far_auc"] == 0.5
    assert result["metrics"]["median_structural_best_so_far_auc"] == 0.5


def test_stage2b_durable_control_is_reproduced_exactly(project_root: Path) -> None:
    result = verify_stage2b_control(_config(project_root))
    assert result["status"] == "completed"
    assert all(result["checks"].values())
    assert result["observed_metrics"] == result["expected_metrics"]


def test_diagnostic_features_and_polarity_are_relabeling_invariant() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=8, seed=101)
    limits = FeatureLimits(
        forbidden_lengths=(4,),
        witness_sample_cap=256,
        cycle_node_budget=100_000,
        distance_query_budget=1_000,
        local_risk_budget=10_000,
    )
    candidate = KSwitchPoolGenerator(
        backend,
        feature_limits=limits,
    ).generate(graph, policy_seed=2, step=0).candidates[0]
    mapping = {vertex: (vertex * 3 + 1) % graph.order for vertex in range(graph.order)}
    relabeled_graph = GraphState(
        graph.order,
        tuple(sorted(normalized_edge((mapping[u], mapping[v])) for u, v in graph.edges)),
    )
    removed = tuple(
        sorted(
            normalized_edge((mapping[u], mapping[v]))
            for u, v in candidate.rewrite.removed_edges
        )
    )
    added = tuple(
        sorted(
            normalized_edge((mapping[u], mapping[v]))
            for u, v in candidate.rewrite.added_edges
        )
    )
    payload = _FeatureSnapshot(relabeled_graph, limits).proposal_payload(
        proposal_id="f" * 64,
        removed=removed,
        added=added,
        selector=candidate.payload["selector_tags"][0],
        k=candidate.payload["k"],
        anchor_length=candidate.payload["anchor_forbidden_length"],
    )
    relabeled = ProposalCandidate(
        RewritePlan(removed, added, candidate.rewrite.operator_family),
        payload,
    )
    original_features = flatten_proposal_features(candidate, (4,))
    relabeled_features = flatten_proposal_features(relabeled, (4,))
    assert original_features == relabeled_features
    first = FeatureAnalyzer(
        forbidden_lengths=(4,),
        sample_cap=32,
        distinct_cap=32,
        near_constant_epsilon=0.01,
    )
    second = FeatureAnalyzer(
        forbidden_lengths=(4,),
        sample_cap=32,
        distinct_cap=32,
        near_constant_epsilon=0.01,
    )
    first.add_pool((candidate,), {candidate.proposal_id: 1}, {candidate.proposal_id: 2.0})
    second.add_pool((relabeled,), {relabeled.proposal_id: 1}, {relabeled.proposal_id: 2.0})
    assert first.as_dict() == second.as_dict()


def test_reduced_matrix_cell_replays_deterministically(project_root: Path) -> None:
    config = _config(project_root)
    config = replace(
        config,
        matrix=Stage2CMatrixConfig(
            orders=(8,),
            graph_seeds=(101,),
            policy_seeds=(1, 2),
            horizons=(2,),
        ),
    )
    first = run_diagnostic_cell(
        config,
        order=8,
        graph_seed=101,
        policy_seeds=(1, 2),
        horizon=2,
        oracle_enabled=True,
    )
    second = run_diagnostic_cell(
        config,
        order=8,
        graph_seed=101,
        policy_seeds=(1, 2),
        horizon=2,
        oracle_enabled=True,
    )
    assert first["canonical_hash"] == second["canonical_hash"]


def test_interrupted_oracle_run_leaves_bounded_terminal_artifact(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(project_root)
    config = replace(config, run=replace(config.run, run_root=tmp_path))
    monkeypatch.setattr(
        "mutation_forge.stage2c.evaluation._repository_provenance",
        lambda _: {"status": "test"},
    )
    monkeypatch.setattr(
        "mutation_forge.stage2c.evaluation.verify_stage2b_control",
        lambda _: {"control_identity": {}, "observed_metrics": {}},
    )

    def interrupt(*args: object, **kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt("forced interruption")

    monkeypatch.setattr(
        "mutation_forge.stage2c.evaluation.run_diagnostic_cell",
        interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        run_pool_oracle(config)
    run_path = next(tmp_path.glob("stage2c-pool-oracle-*"))
    terminal = json.loads((run_path / "terminal_status.json").read_text())
    assert terminal["status"] == "failed"
    assert terminal["error_type"] == "KeyboardInterrupt"
    assert len((run_path / "terminal_status.json").read_bytes()) < (
        config.run.max_artifact_bytes
    )
