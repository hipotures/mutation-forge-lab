from __future__ import annotations

import pytest

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import RewritePlan


def test_toy_seed_is_deterministic_connected_cubic() -> None:
    backend = ToyBackend()
    first = backend.generate_seed(order=30, seed=101)
    second = backend.generate_seed(order=30, seed=101)
    assert first == second
    assert backend.validate(first).valid
    assert len(first.edges) == 45
    assert backend.score(first, witness_cap=64).valid


def test_host_rejects_invalid_rewrite() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=10, seed=1)
    with pytest.raises(ValueError, match="missing"):
        backend.apply_rewrite(
            graph,
            RewritePlan(
                removed_edges=((0, 99),),
                added_edges=(),
                operator_family="test",
            ),
        )


def test_toy_proposal_preserves_invariants() -> None:
    backend = ToyBackend()
    graph = backend.generate_seed(order=30, seed=101)
    rewrite = backend.propose_rewrite(
        graph,
        operator_family="heg_uniform_two_switch",
        policy_seed=1,
        evaluation=1,
    )
    candidate = backend.apply_rewrite(graph, rewrite)
    assert backend.validate(candidate).valid
