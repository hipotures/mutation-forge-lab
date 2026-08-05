from __future__ import annotations

import json
from itertools import combinations

from mutation_forge.models import GraphState
from mutation_forge.native_v3.contracts import ProgramLimits, validate_program
from mutation_forge.native_v3.interpreter import (
    GraphFeatureInput,
    NoPlanReason,
    ProgramContext,
    invoke_program,
)

PRISM = GraphState(
    6,
    (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
    ),
)


def _validated(entry: object):
    raw = json.dumps(
        {"schema_version": "mforge.native.program.v3", "entry": entry},
        separators=(",", ":"),
    )
    validation = validate_program(raw)
    assert validation.valid, validation.diagnostics
    assert validation.program is not None
    return validation.program


def _context() -> ProgramContext:
    return ProgramContext(
        protocol_id="test",
        step_index=0,
        horizon=4,
        acceptance_profile_id="strict_only",
        stagnation_steps=0,
        exploration_window_index=None,
        accepted_rewrites=0,
        accepted_non_improving_rewrites=0,
        consecutive_non_improving_rewrites=0,
        target_forbidden_lengths=(4,),
        witness_cap=64,
    )


def _add_non_edge_program(*, with_failed_branch: bool = False):
    add_branch = {
        "op": "block",
        "children": [
            {
                "op": "apply",
                "action_id": "add_edge",
                "arguments": {"edge": {"op": "ref", "name": "non_edge"}},
            },
            {"op": "emit"},
        ],
    }
    body: object = add_branch
    if with_failed_branch:
        body = {
            "op": "try",
            "branches": [
                {
                    "op": "block",
                    "children": [
                        {
                            "op": "apply",
                            "action_id": "remove_edge",
                            "arguments": {"edge": {"op": "ref", "name": "existing"}},
                        },
                        {"op": "emit"},
                    ],
                },
                add_branch,
            ],
        }
    return _validated(
        {
            "op": "let",
            "name": "vertices",
            "value": {
                "op": "selector",
                "selector_id": "vertices_degree_extreme",
                "arguments": {"mode": "min"},
            },
            "body": {
                "op": "let",
                "name": "vertex",
                "value": {
                    "op": "pick",
                    "source": {"op": "ref", "name": "vertices"},
                    "mode": "seeded_uniform",
                },
                "body": {
                    "op": "let",
                    "name": "non_edges",
                    "value": {
                        "op": "selector",
                        "selector_id": "non_edges_from_vertex",
                        "arguments": {"vertex": {"op": "ref", "name": "vertex"}},
                    },
                    "body": {
                        "op": "let",
                        "name": "non_edge",
                        "value": {
                            "op": "pick",
                            "source": {"op": "ref", "name": "non_edges"},
                            "mode": "seeded_uniform",
                        },
                        "body": {
                            "op": "let",
                            "name": "edges",
                            "value": {
                                "op": "selector",
                                "selector_id": "edges_witness_load_extreme",
                                "arguments": {"length": 4, "mode": "min"},
                            },
                            "body": {
                                "op": "let",
                                "name": "existing",
                                "value": {
                                    "op": "pick",
                                    "source": {"op": "ref", "name": "edges"},
                                    "mode": "seeded_uniform",
                                },
                                "body": body,
                            },
                        },
                    },
                },
            },
        }
    )


def test_add_edge_changes_edge_count_and_degree_vector() -> None:
    program = _add_non_edge_program()
    result = invoke_program(
        program,
        PRISM,
        context=_context(),
        features=GraphFeatureInput(),
        policy_seed=17,
        episode_id="episode",
    )
    assert result.failure is None
    assert result.rewrite is not None
    assert result.rewrite.removed_edges == ()
    assert len(result.rewrite.added_edges) == 1

    degrees = [3] * PRISM.order
    u, v = result.rewrite.added_edges[0]
    degrees[u] += 1
    degrees[v] += 1
    assert sorted(degrees) == [3, 3, 3, 3, 4, 4]


def test_witness_load_sampling_is_lazy_and_shared_within_one_propose() -> None:
    witness_program = _validated(
        {
            "op": "let",
            "name": "vertices",
            "value": {
                "op": "selector",
                "selector_id": "vertices_witness_load_extreme",
                "arguments": {"length": 4, "mode": "max"},
            },
            "body": {
                "op": "let",
                "name": "edges",
                "value": {
                    "op": "selector",
                    "selector_id": "edges_witness_load_extreme",
                    "arguments": {"length": 4, "mode": "max"},
                },
                "body": {"op": "no_plan", "reason": "EXPLICIT"},
            },
        }
    )
    calls = 0

    def sample_loads(_graph: GraphState):
        nonlocal calls
        calls += 1
        return ({(4, 0): 1}, {(4, (0, 1)): 1})

    result = invoke_program(
        witness_program,
        PRISM,
        context=_context(),
        features=GraphFeatureInput(witness_load_provider=sample_loads),
        policy_seed=17,
        episode_id="witness-loads",
    )
    assert result.failure is None
    assert calls == 1

    result = invoke_program(
        _validated(
            {
                "op": "let",
                "name": "vertices",
                "value": {
                    "op": "selector",
                    "selector_id": "vertices_degree_extreme",
                    "arguments": {"mode": "max"},
                },
                "body": {"op": "no_plan", "reason": "EXPLICIT"},
            }
        ),
        PRISM,
        context=_context(),
        features=GraphFeatureInput(witness_load_provider=sample_loads),
        policy_seed=17,
        episode_id="no-witness-loads",
    )
    assert result.failure is None
    assert calls == 1


def test_failed_branch_rolls_back_overlay_before_fallback() -> None:
    program = _add_non_edge_program(with_failed_branch=True)
    result = invoke_program(
        program,
        PRISM,
        context=_context(),
        policy_seed=9,
        episode_id="rollback",
    )
    assert result.rewrite is not None
    assert result.rewrite.removed_edges == ()
    assert len(result.rewrite.added_edges) == 1
    assert result.gross_actions == 2


def test_final_invalid_graph_becomes_no_plan() -> None:
    program = _validated(
        {
            "op": "let",
            "name": "edges",
            "value": {
                "op": "selector",
                "selector_id": "edges_witness_load_extreme",
                "arguments": {"length": 4, "mode": "min"},
            },
            "body": {
                "op": "let",
                "name": "edge",
                "value": {
                    "op": "pick",
                    "source": {"op": "ref", "name": "edges"},
                    "mode": "seeded_uniform",
                },
                "body": {
                    "op": "block",
                    "children": [
                        {
                            "op": "apply",
                            "action_id": "remove_edge",
                            "arguments": {"edge": {"op": "ref", "name": "edge"}},
                        },
                        {"op": "emit"},
                    ],
                },
            },
        }
    )
    result = invoke_program(
        program,
        PRISM,
        context=_context(),
        policy_seed=1,
        episode_id="invalid",
    )
    assert result.rewrite is None
    assert result.no_plan is not None
    assert result.no_plan.reason == NoPlanReason.ILLEGAL_FINAL_STATE


def test_runtime_selector_budget_is_uncatchable_program_failure() -> None:
    program = _add_non_edge_program()
    result = invoke_program(
        program,
        PRISM,
        context=_context(),
        policy_seed=1,
        episode_id="budget",
        limits=ProgramLimits(maximum_selector_cost_units=1),
    )
    assert result.failure is not None
    assert result.failure.code == "SELECTOR_COST_LIMIT"
    assert result.no_plan is None


def test_large_tie_set_uses_seeded_reservoir_without_raw_id_prefix() -> None:
    program = _validated(
        {
            "op": "let",
            "name": "vertices",
            "value": {
                "op": "selector",
                "selector_id": "vertices_degree_extreme",
                "arguments": {"mode": "max"},
            },
            "body": {
                "op": "let",
                "name": "selected",
                "value": {
                    "op": "pick",
                    "source": {"op": "ref", "name": "vertices"},
                    "mode": "seeded_uniform",
                },
                "body": {"op": "no_plan", "reason": "EXPLICIT"},
            },
        }
    )
    result = invoke_program(
        program,
        GraphState(100, ()),
        context=_context(),
        policy_seed=17,
        episode_id="large-tie",
    )
    record = result.selections[0]
    assert record.population_size == 100
    assert record.sample_size == 64
    assert record.sampling_seed64 > 0
    assert record.selected_reference_sha256 is not None


def test_same_labeled_graph_and_seed_replay_exactly() -> None:
    program = _add_non_edge_program()
    kwargs = {
        "context": _context(),
        "policy_seed": 99,
        "episode_id": "replay",
    }
    first = invoke_program(program, PRISM, **kwargs)
    second = invoke_program(program, PRISM, **kwargs)
    assert first == second


def _scientifically_valid(graph: GraphState) -> bool:
    adjacency = [set() for _ in range(graph.order)]
    for u, v in graph.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    seen = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return len(seen) == graph.order and min(map(len, adjacency)) >= 3


def _via_complete(source: GraphState, target: GraphState) -> tuple[GraphState, ...]:
    current = set(source.edges)
    states: list[GraphState] = []
    complete = set(combinations(range(source.order), 2))
    for edge in sorted(complete - current):
        current.add(edge)
        states.append(GraphState(source.order, tuple(sorted(current))))
    for edge in sorted(complete - set(target.edges)):
        current.remove(edge)
        states.append(GraphState(source.order, tuple(sorted(current))))
    return tuple(states)


def test_add_remove_reachability_lemma_is_exhaustive_for_order_five() -> None:
    all_edges = tuple(combinations(range(5), 2))
    valid_graphs = []
    for mask in range(1 << len(all_edges)):
        graph = GraphState(
            5,
            tuple(edge for index, edge in enumerate(all_edges) if mask & (1 << index)),
        )
        if _scientifically_valid(graph):
            valid_graphs.append(graph)
    assert valid_graphs
    for source in valid_graphs:
        for target in valid_graphs:
            states = _via_complete(source, target)
            assert all(_scientifically_valid(state) for state in states)
            if states:
                assert states[-1] == target


def test_remove_edge_can_change_edge_count_while_preserving_validity() -> None:
    complete = GraphState(5, tuple(combinations(range(5), 2)))
    program = _validated(
        {
            "op": "let",
            "name": "edges",
            "value": {
                "op": "selector",
                "selector_id": "edges_bridge_risk",
                "arguments": {"mode": "min"},
            },
            "body": {
                "op": "let",
                "name": "edge",
                "value": {
                    "op": "pick",
                    "source": {"op": "ref", "name": "edges"},
                    "mode": "seeded_uniform",
                },
                "body": {
                    "op": "block",
                    "children": [
                        {
                            "op": "apply",
                            "action_id": "remove_edge",
                            "arguments": {"edge": {"op": "ref", "name": "edge"}},
                        },
                        {"op": "emit"},
                    ],
                },
            },
        }
    )
    result = invoke_program(
        program,
        complete,
        context=_context(),
        policy_seed=5,
        episode_id="remove-degree",
    )
    assert result.rewrite is not None
    assert len(result.rewrite.removed_edges) == 1
    assert result.rewrite.added_edges == ()
