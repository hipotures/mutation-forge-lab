from __future__ import annotations

import pytest

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import GraphScore, GraphState, RewritePlan, normalized_edge
from mutation_forge.proposals.k_switch import (
    EvaluationContractError,
    FeatureLimits,
    KSwitchPoolGenerator,
    PoolLimits,
    _FeatureSnapshot,
    _perfect_matchings,
    make_scientific_context,
)


def test_exact_perfect_matching_counts_are_bounded() -> None:
    assert len(_perfect_matchings((0, 1, 2, 3))) == 3
    assert len(_perfect_matchings((0, 1, 2, 3, 4, 5))) == 15
    assert len(_perfect_matchings(tuple(range(8)))) == 105


def test_context_requires_exact_ordered_backend_score_lengths() -> None:
    graph = ToyBackend().generate_seed(order=8, seed=101)
    score = GraphScore(
        valid=True,
        capped_cycle_counts=((4, 1), (8, 0)),
        total_capped_witnesses=1,
        weighted_penalty=16,
        complete=True,
        ordering_key=(0, 1, 16),
    )
    context = make_scientific_context(
        graph,
        score,
        forbidden_lengths=(4, 8),
        step=0,
        remaining_steps=1,
    )
    assert context["forbidden_lengths"] == [4, 8]
    assert context["capped_cycle_counts"] == [1, 0]

    for observed in (((8, 0), (4, 1)), ((4, 1),), ((4, 1), (4, 0))):
        invalid = GraphScore(
            valid=True,
            capped_cycle_counts=observed,
            total_capped_witnesses=sum(count for _, count in observed),
            weighted_penalty=16,
            complete=True,
            ordering_key=(0, 1, 16),
        )
        with pytest.raises(EvaluationContractError):
            make_scientific_context(
                graph,
                invalid,
                forbidden_lengths=(4, 8),
                step=0,
                remaining_steps=1,
            )


def test_context_rejects_inconsistent_score_total() -> None:
    graph = ToyBackend().generate_seed(order=8, seed=101)
    score = GraphScore(
        valid=True,
        capped_cycle_counts=((4, 1),),
        total_capped_witnesses=0,
        weighted_penalty=16,
        complete=True,
        ordering_key=(0, 1, 16),
    )
    with pytest.raises(EvaluationContractError, match="count vector"):
        make_scientific_context(
            graph,
            score,
            forbidden_lengths=(4,),
            step=0,
            remaining_steps=1,
        )


def test_pool_is_deterministic_bounded_deduplicated_and_host_validated() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=8, seed=101)
    generator = KSwitchPoolGenerator(
        backend,
        feature_limits=FeatureLimits(forbidden_lengths=(4,)),
    )
    first = generator.generate(graph, policy_seed=7, step=3)
    second = generator.generate(graph, policy_seed=7, step=3)
    assert first.pool_hash == second.pool_hash
    assert first.candidates == second.candidates
    assert first.rejected == second.rejected
    assert first.selector_counts == second.selector_counts
    assert first.k_counts == second.k_counts
    assert first.feature_usage == second.feature_usage
    assert 0 < first.retained <= generator.pool_limits.pool_size
    assert len({candidate.proposal_id for candidate in first.candidates}) == (first.retained)
    assert {"2", "3", "4"}.issubset(first.k_counts)
    for candidate in first.candidates:
        rewrite = candidate.rewrite
        k = candidate.payload["k"]
        assert len(rewrite.removed_edges) == len(rewrite.added_edges) == k
        assert len({vertex for edge in rewrite.removed_edges for vertex in edge}) == 2 * k
        assert backend.validate(backend.apply_rewrite(graph, rewrite)).valid
        assert "removed_edges" not in candidate.payload
        assert "added_edges" not in candidate.payload
        assert "graph" not in candidate.payload


def test_retry_exhaustion_returns_auditable_empty_pool() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=4, seed=1)
    generator = KSwitchPoolGenerator(
        backend,
        feature_limits=FeatureLimits(forbidden_lengths=(4,)),
        pool_limits=PoolLimits(
            pool_size=4,
            k_values=(4,),
            selectors=("uniform_random",),
            selector_weights=(1,),
            retry_limit=3,
            matching_limit=4,
        ),
    )
    pool = generator.generate(graph, policy_seed=1, step=0)
    assert pool.retained == 0
    assert pool.rejected["disjoint_selection"] == 3


def test_feature_budgets_fail_bounded_and_are_reported() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=8, seed=101)
    generator = KSwitchPoolGenerator(
        backend,
        feature_limits=FeatureLimits(
            forbidden_lengths=(4, 8, 16),
            witness_sample_cap=32,
            cycle_node_budget=1,
            distance_query_budget=1,
            local_risk_budget=1,
        ),
    )
    pool = generator.generate(graph, policy_seed=1, step=0)
    usage = pool.feature_usage
    assert usage["cycle_budget_exhausted"]
    assert usage["distance_queries"] <= 1
    assert usage["local_risk_operations"] <= 1


def test_semantic_features_are_invariant_under_consistent_relabeling() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=8, seed=101)
    limits = FeatureLimits(
        forbidden_lengths=(4,),
        witness_sample_cap=256,
        cycle_node_budget=100_000,
        distance_query_budget=1_000,
        local_risk_budget=10_000,
    )
    pool = KSwitchPoolGenerator(
        backend,
        feature_limits=limits,
    ).generate(graph, policy_seed=2, step=0)
    candidate = pool.candidates[0]
    mapping = {vertex: (vertex * 3 + 1) % graph.order for vertex in range(graph.order)}
    relabeled_graph = GraphState(
        graph.order,
        tuple(sorted(normalized_edge((mapping[u], mapping[v])) for u, v in graph.edges)),
    )
    relabeled_removed = tuple(
        sorted(
            normalized_edge((mapping[u], mapping[v])) for u, v in candidate.rewrite.removed_edges
        )
    )
    relabeled_added = tuple(
        sorted(normalized_edge((mapping[u], mapping[v])) for u, v in candidate.rewrite.added_edges)
    )
    relabeled_payload = _FeatureSnapshot(relabeled_graph, limits).proposal_payload(
        proposal_id="different-opaque-id",
        removed=relabeled_removed,
        added=relabeled_added,
        selector=candidate.payload["selector_tags"][0],
        k=candidate.payload["k"],
        anchor_length=candidate.payload["anchor_forbidden_length"],
    )
    original = dict(candidate.payload)
    relabeled = dict(relabeled_payload)
    original.pop("proposal_id")
    relabeled.pop("proposal_id")
    assert original == relabeled


def test_invalid_k_switch_plan_is_rejected_before_scoring() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=8, seed=101)
    invalid = RewritePlan(
        removed_edges=(graph.edges[0], graph.edges[1]),
        added_edges=((0, 0), graph.edges[2]),
        operator_family="legal_2_switch",
    )
    try:
        backend.apply_rewrite(graph, invalid)
    except ValueError as error:
        assert "loop" in str(error) or "existing" in str(error)
    else:
        raise AssertionError("invalid plan was accepted")
