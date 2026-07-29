from mutation_forge.stage3.statistics import (
    GO_TO_STAGE_4,
    INCONCLUSIVE_INFRASTRUCTURE_FAILURE,
    NO_GO,
    evaluate_gate,
    gate_report,
    select_champion,
    summarize_development,
)


def _episode(i: int, candidate: float, random: float = 1.0, structural: float = 1.0) -> dict:
    return {
        "episode_id": f"o10-g0301-p{3001 + i:04d}",
        "order": 10,
        "graph_seed": 301,
        "policies": {
            "random": {"auc": random, "best_total_witnesses": 4},
            "structural": {"auc": structural, "best_total_witnesses": 3},
            "candidate": {"auc": candidate, "best_total_witnesses": 2},
        },
    }


def test_nested_summary_pairs_by_episode_id() -> None:
    episodes = [_episode(1, 2.0), _episode(0, 1.5)]
    summary = summarize_development(
        episodes, ["random", "structural", "candidate"], bootstrap_samples=8
    )
    assert summary["policies"]["candidate"]["pooled_median_auc"] == 1.75
    assert summary["policies"]["candidate"]["paired_deltas"]["random"]["median_auc_delta"] == 0.75


def test_champion_tie_uses_ast_hash() -> None:
    summary = {
        "policies": {
            "random": {"pooled_median_auc": 1.0},
            "structural": {"pooled_median_auc": 1.0},
            "a": {
                "pooled_median_auc": 2.0,
                "pooled_median_best_witnesses": 2,
                "by_order": {"10": {"median_auc": 2.0}},
            },
            "b": {
                "pooled_median_auc": 2.0,
                "pooled_median_best_witnesses": 2,
                "by_order": {"10": {"median_auc": 2.0}},
            },
        }
    }
    assert (
        select_champion(
            summary, {"a": {"normalized_ast_sha256": "b"}, "b": {"normalized_ast_sha256": "a"}}
        )
        == "b"
    )


def test_gate_boundaries_and_infrastructure() -> None:
    base = {
        "dependency_provenance": True,
        "protocol_safety": True,
        "campaign_authority": True,
        "exact_usage": True,
        "champion_random_relative": 0.05,
        "champion_structural_relative": 0.90,
        "minimum_unique": 4,
        "baseline_ast_distinct": True,
        "primary_replay_exact": True,
        "invalid_records": 0,
        "worker_failures": 0,
        "selected_only_equal_bounded_parity": True,
        "repository_and_heg_validation": True,
    }
    assert evaluate_gate(base) == GO_TO_STAGE_4
    assert len(gate_report(base)["checks"]) == 12
    failed = dict(base)
    failed["champion_random_relative"] = 0.049
    assert evaluate_gate(failed) == NO_GO
    infra = dict(base)
    infra["primary_replay_exact"] = False
    assert evaluate_gate(infra) == INCONCLUSIVE_INFRASTRUCTURE_FAILURE
    assert (
        evaluate_gate(base, infrastructure={"auth_failure": True})
        == INCONCLUSIVE_INFRASTRUCTURE_FAILURE
    )
