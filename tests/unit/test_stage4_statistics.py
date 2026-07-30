from __future__ import annotations

from typing import Any

import pytest

from mutation_forge.stage4.statistics import (
    DEFAULT_BOOTSTRAP_SEED,
    GO_TO_STAGE_5,
    INCONCLUSIVE_INFRASTRUCTURE_FAILURE,
    NO_GO,
    evaluate_gate,
    gate_report,
    hierarchical_bootstrap,
    paired_bootstrap,
    select_champion,
    summarize_development,
)


def _episodes() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for order in (10, 12):
        for graph in range(4):
            for policy_seed in range(16):
                curve = [0.1, 0.2, 0.3]
                base = {"normalized_best_so_far_curve": curve, "best_total_witnesses": 2}
                result.append(
                    {
                        "episode_id": f"{order}-{graph}-{policy_seed}",
                        "order": order,
                        "graph_seed": graph,
                        "policy_seed": policy_seed,
                        "policies": {
                            "random": {**base, "auc": 0.2},
                            "structural": {**base, "auc": 0.3},
                            "candidate": {**base, "auc": 0.4},
                        },
                    }
                )
    return result


def _passing() -> dict[str, Any]:
    return {
        "policies": {
            "random": {"pooled_median_auc": 1.0},
            "structural": {"pooled_median_auc": 1.0},
            "candidate": {
                "pooled_median_auc": 1.2,
                "pooled_median_best_total_witness": 1,
            },
        },
        "dependency_import_provenance_heg": True,
        "four_generations_exact_usage": True,
        "four_generations": True,
        "exact_usage": True,
        "initial_turns_exact_32": True,
        "new_unique_valid_offspring": 16,
        "champion_distinct": True,
        "pooled_relative_improvement": 0.02,
        "pooled_bootstrap_lower_bound": 0.0001,
        "order_deltas": {"10": 0.0, "12": 0.0},
        "graph_seed_nonnegative_counts": {"10": 3, "12": 4},
        "structural_retention": 0.99,
        "primary_replay_exact": True,
        "graph_validity_100_percent": True,
        "worker_failures_zero": True,
        "selected_plan_only": True,
        "oracle_score_calls_zero": True,
        "equal_budgets": True,
        "archive_lineage_repository": True,
    }


def test_bootstrap_is_deterministic_and_balanced() -> None:
    first = hierarchical_bootstrap(_episodes(), samples=16)
    assert first == hierarchical_bootstrap(_episodes(), samples=16)
    assert first["pooled"]["seed"] == DEFAULT_BOOTSTRAP_SEED
    assert paired_bootstrap([1.0, 2.0], samples=16) == paired_bootstrap([1.0, 2.0], samples=16)


def test_bootstrap_rejects_incomplete_matrix() -> None:
    with pytest.raises(ValueError, match="sixteen"):
        hierarchical_bootstrap(_episodes()[:-1], samples=2)


def test_summary_and_champion_tie_breaking() -> None:
    summary = summarize_development(
        _episodes(), policies=["random", "structural", "candidate"], bootstrap_samples=2
    )
    assert summary["policies"]["candidate"]["pooled_median_normalized_best_so_far_auc"] == 0.4
    champion = {
        "policies": {
            "z": {
                "pooled_median_auc": 1.0,
                "pooled_median_best_total_witness": 2,
                "normalized_ast_sha256": "b",
            },
            "a": {
                "pooled_median_auc": 1.0,
                "pooled_median_best_total_witness": 2,
                "normalized_ast_sha256": "a",
            },
        }
    }
    assert select_champion(champion) == "a"


def test_champion_campaign_completion_allows_early_stage4_offspring() -> None:
    summary = {
        "policies": {
            "early": {
                "pooled_median_auc": 2.0,
                "pooled_median_best_total_witness": 1,
            },
            "later": {
                "pooled_median_auc": 1.0,
                "pooled_median_best_total_witness": 1,
            },
        },
        "policy_identities": {
            "early": {"generation": 1, "origin": "stage4", "normalized_ast_sha256": "a"},
            "later": {"generation": 4, "origin": "stage4", "normalized_ast_sha256": "b"},
        },
    }
    assert select_champion(summary, generation=4) == "early"
    assert select_champion(summary, generation=3) is None


def test_gate_boundaries_and_infrastructure_terminal() -> None:
    passing = _passing()
    assert evaluate_gate(passing, champion="candidate") == GO_TO_STAGE_5
    assert (
        evaluate_gate({**passing, "pooled_relative_improvement": 0.0199}, champion="candidate")
        == NO_GO
    )
    incomplete = {**passing, "incomplete": True, "exhausted": True}
    assert (
        evaluate_gate(incomplete, champion="candidate", infrastructure={"timeout": True})
        == INCONCLUSIVE_INFRASTRUCTURE_FAILURE
    )
    report = gate_report(passing, champion="candidate")
    assert len(report["checks"]) == 12
    assert len(report["canonical_sha256"]) == 64


def test_nonfinite_metric_fails_closed() -> None:
    invalid = {**_passing(), "pooled_relative_improvement": float("nan")}
    assert evaluate_gate(invalid, champion="candidate") == NO_GO
