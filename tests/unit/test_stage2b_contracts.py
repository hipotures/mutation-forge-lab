from __future__ import annotations

from pathlib import Path

import pytest

from mutation_forge.sandbox.contracts import (
    SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
    SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
    ContractError,
    validate_ranker_inputs,
)
from mutation_forge.stage2b.config import load_stage2b_config


def _context() -> dict[str, object]:
    return {
        "schema_version": SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
        "order": 8,
        "forbidden_lengths": [4, 5, 6],
        "capped_cycle_counts": [2, 0, 0],
        "weighted_penalty": 32,
        "step": 0,
        "remaining_steps": 7,
        "stagnation": 0,
        "recent_best_improvement": 0.0,
        "recent_acceptance_rate": 0.0,
        "recent_duplicate_rate": 0.0,
    }


def _proposal() -> dict[str, object]:
    return {
        "schema_version": SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": "a" * 64,
        "k": 2,
        "operator_family": "legal_2_switch",
        "selector_tags": ["uniform_random"],
        "anchor_forbidden_length": None,
        "broken_sampled_witnesses_by_length": [1, 0, 0],
        "removed_edge_load_sum_by_length": [2, 0, 0],
        "removed_edge_load_max_by_length": [1, 0, 0],
        "minimum_distance_between_removed_edges": 1,
        "mean_distance_between_removed_edges": 1.0,
        "minimum_preexisting_distance_for_new_edges": 2,
        "mean_preexisting_distance_for_new_edges": 2.0,
        "local_triangle_risk": 0,
        "local_c4_risk": 0,
        "reconnection_span": 2.0,
    }


def test_frozen_scientific_contract_accepts_exact_bounded_payload() -> None:
    context, proposal = validate_ranker_inputs(
        _context(),
        _proposal(),
        max_request_bytes=65536,
    )
    assert context["schema_version"] == SCIENTIFIC_CONTEXT_SCHEMA_VERSION
    assert proposal["schema_version"] == SCIENTIFIC_PROPOSAL_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("target", "key", "value"),
    [
        ("context", "graph", {"edges": []}),
        ("context", "recent_acceptance_rate", 1.1),
        ("proposal", "removed_edges", [[0, 1]]),
        ("proposal", "k", 5),
        ("proposal", "operator_family", "legal_3_switch"),
        ("proposal", "proposal_id", "not-a-sha"),
        ("proposal", "selector_tags", ["unreviewed"]),
        ("proposal", "local_c4_risk", -1),
        ("proposal", "mean_preexisting_distance_for_new_edges", -0.5),
    ],
)
def test_frozen_scientific_contract_rejects_authority_and_invalid_values(
    target: str,
    key: str,
    value: object,
) -> None:
    context = _context()
    proposal = _proposal()
    selected = context if target == "context" else proposal
    selected[key] = value
    with pytest.raises(ContractError):
        validate_ranker_inputs(context, proposal, max_request_bytes=65536)


def test_frozen_scientific_vectors_must_align_with_context() -> None:
    proposal = _proposal()
    proposal["broken_sampled_witnesses_by_length"] = [1]
    with pytest.raises(ContractError, match="must align"):
        validate_ranker_inputs(_context(), proposal, max_request_bytes=65536)


def test_stage2b_preregistered_config_is_strict_and_frozen(
    project_root: Path,
) -> None:
    config = load_stage2b_config(project_root / "configs" / "stage2b-preregistered.toml")
    assert config.schema_version == "stage2b.1"
    assert config.pool.k_values == (2, 3, 4)
    assert len(config.pool.selectors) == 6
    assert len(config.toy_gate.policy_seeds) == 32
    assert config.toy_gate.auc_relative_improvement_threshold == 0.10
    assert config.repositories.frozen_project_commit == ("e2d11bb86b4fa5dbc7ebfb441923e0f02e9799a9")
    assert len(config.stable_hash()) == 64


def test_stage2b_config_rejects_less_than_32_preregistered_seeds(
    tmp_path: Path,
    project_root: Path,
) -> None:
    source = (project_root / "configs" / "stage2b-preregistered.toml").read_text()
    start = source.index("policy_seeds = [")
    end = source.index("]\nbootstrap_samples", start)
    path = tmp_path / "invalid.toml"
    path.write_text(source[:start] + "policy_seeds = [1]\n" + source[end + 2 :])
    with pytest.raises(ValueError, match="at least 32"):
        load_stage2b_config(path)
