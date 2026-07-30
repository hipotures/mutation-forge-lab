from __future__ import annotations

from typing import Any

import pytest

from mutation_forge.stage3.statistics import (
    GO_TO_STAGE_4,
    INCONCLUSIVE_INFRASTRUCTURE_FAILURE,
    NO_GO,
    evaluate_gate,
    gate_report,
    hierarchical_bootstrap,
    paired_bootstrap,
    select_champion,
    summarize_development,
)


def _episode(index: int, *, candidate_auc: float = 0.30) -> dict[str, Any]:
    base = {
        "auc": 0.20,
        "best_total_witnesses": 4,
        "accepted_count": 1,
        "rejected_count": 2,
        "duplicate_count": 0,
        "normalized_best_so_far_curve": [0.1, 0.2, 0.3],
    }
    return {
        "episode_id": f"episode-{index:03d}",
        "order": 10 if index % 2 == 0 else 12,
        "graph_seed": 300 + index % 2,
        "policy_seed": 3000 + index,
        "policies": {
            "random": base,
            "structural": {**base, "auc": 0.24, "best_total_witnesses": 3},
            "candidate": {
                **base,
                "auc": candidate_auc,
                "best_total_witnesses": 2,
                "normalized_best_so_far_curve": [0.2, 0.3, candidate_auc],
            },
        },
    }


def _passing_summary() -> dict[str, Any]:
    return {
        "policies": {
            "random": {"pooled_median_auc": 1.0},
            "structural": {"pooled_median_auc": 1.0},
            "candidate": {
                "pooled_median_auc": 1.2,
                "pooled_median_best_witnesses": 1,
            },
        },
        "minimum_unique": 4,
        "baseline_ast_distinct": True,
        "champion_random_relative": 0.10,
        "champion_structural_relative": 1.0,
        "dependency_provenance": True,
        "protocol_safety": True,
        "campaign_authority": True,
        "exact_usage": True,
        "primary_replay_exact": True,
        "invalid_records": 0,
        "worker_failures": 0,
        "selected_only_equal_bounded_parity": True,
        "repository_and_heg_validation": True,
    }


def test_bootstrap_and_hierarchical_bootstrap_are_fixed_and_bounded() -> None:
    values = [0.1, 0.2, 0.4, 0.8]
    first = paired_bootstrap(values, samples=64, seed=7)
    second = paired_bootstrap(values, samples=64, seed=7)
    assert first == second
    assert first["samples"] == 64
    episodes = [_episode(i) for i in range(8)]
    assert hierarchical_bootstrap(episodes, samples=32, seed=7) == hierarchical_bootstrap(
        episodes, samples=32, seed=7
    )


def test_summary_is_order_stratified_and_requires_paired_baselines() -> None:
    summary = summarize_development([_episode(i) for i in range(8)], bootstrap_samples=32)
    assert summary["episode_count"] == 8
    assert set(summary["policies"]) == {"random", "structural", "candidate"}
    assert set(summary["policies"]["candidate"]["by_order"]) == {"10", "12"}
    assert "paired_deltas" in summary["policies"]["candidate"]


def test_champion_tie_breaking_ends_at_normalized_ast_identity() -> None:
    summary = {
        "policies": {
            "zeta": {"pooled_median_auc": 1.0, "pooled_median_best_witnesses": 2},
            "alpha": {"pooled_median_auc": 1.0, "pooled_median_best_witnesses": 2},
            "random": {},
            "structural": {},
        }
    }
    identities = {
        "zeta": {"normalized_ast_sha256": "b" * 64, "source_sha256": "a" * 64},
        "alpha": {"normalized_ast_sha256": "a" * 64, "source_sha256": "b" * 64},
    }
    assert select_champion(summary, identities) == "alpha"


def test_champion_does_not_use_source_or_policy_name_after_ast_tie() -> None:
    summary = {
        "policies": {
            "zeta": {"pooled_median_auc": 1.0, "pooled_median_best_witnesses": 2},
            "alpha": {"pooled_median_auc": 1.0, "pooled_median_best_witnesses": 2},
            "random": {},
            "structural": {},
        }
    }
    same_ast = "a" * 64
    identities = {
        "zeta": {"normalized_ast_sha256": same_ast, "source_sha256": "z" * 64},
        "alpha": {"normalized_ast_sha256": same_ast, "source_sha256": "a" * 64},
    }
    # The four-key rule ends at normalized AST identity.  With all keys tied,
    # the stable roster order wins; source hash and policy name are irrelevant.
    assert select_champion(summary, identities) == "zeta"


def test_gate_has_exact_decision_boundaries() -> None:
    passing = _passing_summary()
    assert evaluate_gate(passing, champion="candidate") == GO_TO_STAGE_4
    report = gate_report(passing, champion="candidate")
    assert len(report["checks"]) == 12
    no_go = {**passing, "champion_random_relative": 0.049}
    assert evaluate_gate(no_go, champion="candidate") == NO_GO
    inconclusive = {**passing, "protocol_safety": False}
    assert evaluate_gate(inconclusive, champion="candidate") == INCONCLUSIVE_INFRASTRUCTURE_FAILURE
    infra_report = gate_report(passing, champion="candidate", infrastructure={"failure": True})
    assert infra_report["decision"] == INCONCLUSIVE_INFRASTRUCTURE_FAILURE


def test_statistics_reject_negative_and_nonfinite_metrics() -> None:
    with pytest.raises(ValueError):
        summarize_development(
            [{**_episode(0), "policies": {**_episode(0)["policies"], "candidate": {"auc": -1}}}]
        )
    with pytest.raises(ValueError):
        summarize_development(
            [
                {
                    **_episode(0),
                    "policies": {
                        **_episode(0)["policies"],
                        "candidate": {"auc": float("nan")},
                    },
                }
            ]
        )
    invalid = _passing_summary()
    invalid["policies"]["candidate"]["pooled_median_auc"] = -1
    assert evaluate_gate(invalid, champion="candidate") == NO_GO
