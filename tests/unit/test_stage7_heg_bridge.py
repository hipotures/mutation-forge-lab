from __future__ import annotations

from pathlib import Path

import pytest

from mutation_forge.stage7_heg_bridge.bridge import BridgeError, HegPolicyBridge
from mutation_forge.stage7_heg_bridge.contract import (
    CATALOG_ID,
    FROZEN_CYCLE_NODE_BUDGET,
    FROZEN_DISTANCE_QUERY_BUDGET,
    FROZEN_FORBIDDEN_LENGTHS,
    FROZEN_LOCAL_RISK_BUDGET,
    FROZEN_WITNESS_SAMPLE_CAP,
    HEG_COMMIT,
    POLICY_AST_SHA256,
    POLICY_SOURCE_SHA256,
    ContractViolation,
    verify_frozen_policy,
)
from mutation_forge.stage7_heg_bridge.fixtures import build_fixture, fixture_specs


@pytest.fixture()
def bridge() -> HegPolicyBridge:
    instance = HegPolicyBridge(Path("../heg"))
    try:
        yield instance
    finally:
        instance.close()


def test_frozen_catalog_identity() -> None:
    result = verify_frozen_policy()
    assert result["source_sha256"] == POLICY_SOURCE_SHA256
    assert result["normalized_ast_sha256"] == POLICY_AST_SHA256
    assert result["catalog_id"] == CATALOG_ID


def test_bridge_rejects_unreviewed_catalog() -> None:
    with pytest.raises(ContractViolation):
        HegPolicyBridge(Path("../heg"), catalog_id="unreviewed")


def test_bridge_is_pinned_and_deterministic(bridge: HegPolicyBridge) -> None:
    assert bridge.heg_identity["commit"] == HEG_COMMIT
    assert bridge.feature_limits.forbidden_lengths == FROZEN_FORBIDDEN_LENGTHS
    assert bridge.feature_limits.witness_sample_cap == FROZEN_WITNESS_SAMPLE_CAP
    assert bridge.feature_limits.cycle_node_budget == FROZEN_CYCLE_NODE_BUDGET
    assert bridge.feature_limits.distance_query_budget == FROZEN_DISTANCE_QUERY_BUDGET
    assert bridge.feature_limits.local_risk_budget == FROZEN_LOCAL_RISK_BUDGET
    fixture = build_fixture(bridge, fixture_specs()[0])
    left = bridge.select(fixture.context, fixture.pool, graph=fixture.graph)
    right = bridge.select(fixture.context, fixture.pool, graph=fixture.graph)
    assert left.selected_proposal_id == right.selected_proposal_id
    assert left.rank_order == right.rank_order
    assert left.telemetry["m4_calls"] == 0


def test_bridge_rejects_empty_pool(bridge: HegPolicyBridge) -> None:
    fixture = build_fixture(bridge, fixture_specs()[0])
    from dataclasses import replace

    empty = replace(fixture.pool, candidates=(), pool_hash="0" * 64)
    with pytest.raises(BridgeError):
        bridge.validate_pool(fixture.graph, empty)
