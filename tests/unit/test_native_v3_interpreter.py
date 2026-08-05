from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from itertools import permutations

import pytest

from mutation_forge.models import Edge, GraphState, GraphValidation, RewritePlan, normalized_edge
from mutation_forge.native_v3.contracts import (
    PROGRAM_SCHEMA_VERSION,
    ValidatedProgram,
    validate_program,
)
from mutation_forge.native_v3.graph_runtime import (
    EdgeRef,
    GraphFeatureInput,
    GraphPreconditionError,
    GraphRuntime,
    MatchingRef,
    NonEdgeRef,
    PathRef,
    RewriteHost,
    VertexRef,
)
from mutation_forge.native_v3.interpreter import (
    InterpreterLimits,
    InvocationResult,
    NoPlan,
    ProgramContext,
    invoke_program,
)
from mutation_forge.native_v3.randomness import derive_seed64, splitmix64


def _validated(entry: object) -> ValidatedProgram:
    validation = validate_program(
        json.dumps(
            {"schema_version": PROGRAM_SCHEMA_VERSION, "entry": entry},
            separators=(",", ":"),
        )
    )
    assert validation.valid, validation.diagnostics
    assert validation.program is not None
    return validation.program


def _selector(selector_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
    return {"op": "selector", "selector_id": selector_id, "arguments": dict(arguments)}


def _pick(selector_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
    return {
        "op": "pick",
        "source": _selector(selector_id, arguments),
        "mode": "seeded_uniform",
    }


def _apply(action_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
    return {"op": "apply", "action_id": action_id, "arguments": dict(arguments)}


def _cubic_graph(order: int) -> GraphState:
    assert order >= 6 and order % 2 == 0
    edges = {normalized_edge((vertex, (vertex + 1) % order)) for vertex in range(order)}
    edges.update((vertex, vertex + order // 2) for vertex in range(order // 2))
    return GraphState(order, tuple(sorted(edges)))


def _degrees(graph: GraphState) -> tuple[int, ...]:
    result = [0] * graph.order
    for u, v in graph.edges:
        result[u] += 1
        result[v] += 1
    return tuple(result)


def _connected(graph: GraphState) -> bool:
    adjacency = [set[int]() for _ in range(graph.order)]
    for u, v in graph.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    seen = {0}
    pending = [0]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return len(seen) == graph.order


class _MinDegreeHost(RewriteHost):
    def validate(self, graph: GraphState) -> GraphValidation:
        errors: list[str] = []
        if graph.order < 1:
            errors.append("order")
        if len(set(graph.edges)) != len(graph.edges):
            errors.append("duplicate")
        if any(
            u == v or not (0 <= u < graph.order and 0 <= v < graph.order) for u, v in graph.edges
        ):
            errors.append("invalid edge")
        if graph.order and min(_degrees(graph)) < 3:
            errors.append("minimum degree")
        if graph.order and not _connected(graph):
            errors.append("connectivity")
        return GraphValidation(not errors, tuple(errors))

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: object | None = None,
    ) -> GraphState:
        del record_score_profile
        removed = tuple(normalized_edge(edge) for edge in rewrite.removed_edges)
        added = tuple(normalized_edge(edge) for edge in rewrite.added_edges)
        if len(set(removed)) != len(removed) or len(set(added)) != len(added):
            raise ValueError("duplicate rewrite edge")
        current = set(graph.edges)
        if not set(removed).issubset(current):
            raise ValueError("missing removed edge")
        remaining = current.difference(removed)
        if any(edge in remaining for edge in added):
            raise ValueError("existing added edge")
        candidate = GraphState(graph.order, tuple(sorted(remaining.union(added))))
        validation = self.validate(candidate)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return candidate


HOST = _MinDegreeHost()
CONTEXT = ProgramContext(
    step_index=7,
    horizon=100,
    acceptance_profile_id="native-v3-test",
    invocation_ordinal=0,
)


def _run(
    program: ValidatedProgram,
    graph: GraphState,
    *,
    context: ProgramContext = CONTEXT,
    features: GraphFeatureInput | None = None,
    limits: InterpreterLimits | None = None,
) -> InvocationResult:
    return invoke_program(
        program,
        graph,
        rewrite_host=HOST,
        context=context,
        episode_id="fixture-cubic",
        features=features,
        limits=limits,
    )


def _add_one_program() -> ValidatedProgram:
    return _validated(
        {
            "op": "let",
            "name": "edge",
            "value": _pick("non_edges_legal", {}),
            "body": {
                "op": "block",
                "children": [
                    _apply("add_edge", {"edge": {"op": "ref", "name": "edge"}}),
                    {"op": "emit"},
                ],
            },
        }
    )


def test_random_protocol_vectors_and_replay_are_frozen() -> None:
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert derive_seed64("program", "episode", 7) == 0x32083FE73F9CFA5B
    program = _add_one_program()
    first = _run(program, _cubic_graph(6))
    second = _run(program, _cubic_graph(6))
    assert first == second


def test_even_cubic_seed_reaches_valid_non_cubic_fixed_order_graph() -> None:
    graph = _cubic_graph(6)
    result = _run(_add_one_program(), graph)
    assert result.successful
    assert result.rewrite is not None
    assert result.no_plan is None
    candidate = HOST.apply_rewrite(graph, result.rewrite)
    assert candidate.order == graph.order
    assert len(candidate.edges) == len(graph.edges) + 1
    assert sorted(_degrees(candidate)) == [3, 3, 3, 3, 4, 4]
    assert HOST.validate(candidate).valid


def test_remove_edge_can_reduce_edge_count_when_final_graph_remains_legal() -> None:
    cubic = _cubic_graph(6)
    extra = next(
        edge
        for edge in ((u, v) for u in range(cubic.order) for v in range(u + 1, cubic.order))
        if edge not in cubic.edges
    )
    graph = GraphState(cubic.order, tuple(sorted((*cubic.edges, extra))))
    program = _validated(
        {
            "op": "let",
            "name": "edge",
            "value": _pick("edges_witness_load_extreme", {"length": 4, "mode": "max"}),
            "body": {
                "op": "block",
                "children": [
                    _apply("remove_edge", {"edge": {"op": "ref", "name": "edge"}}),
                    {"op": "emit"},
                ],
            },
        }
    )
    features = GraphFeatureInput(edge_witness_load={(4, extra): 100})
    result = _run(program, graph, features=features)
    assert result.rewrite is not None
    assert result.rewrite.removed_edges == (extra,)
    assert not result.rewrite.added_edges
    assert HOST.apply_rewrite(graph, result.rewrite) == cubic


@pytest.mark.parametrize(("order", "k"), ((8, 2), (10, 3), (12, 4)))
def test_legal_k_switch_preserves_order_edge_count_and_degree_vector(
    order: int,
    k: int,
) -> None:
    graph = _cubic_graph(order)
    program = _validated(
        {
            "op": "let",
            "name": "matching",
            "value": _pick("matching_k_switch_reconnections", {"k": k}),
            "body": {
                "op": "block",
                "children": [
                    _apply("k_switch", {"matching": {"op": "ref", "name": "matching"}}),
                    {"op": "emit"},
                ],
            },
        }
    )
    result = _run(program, graph)
    assert result.rewrite is not None
    candidate = HOST.apply_rewrite(graph, result.rewrite)
    assert candidate.order == order
    assert len(candidate.edges) == len(graph.edges)
    assert _degrees(candidate) == _degrees(graph)


def test_failed_graph_branch_restores_overlay_and_outer_binding() -> None:
    program = _validated(
        {
            "op": "let",
            "name": "edge",
            "value": _pick("non_edges_legal", {}),
            "body": {
                "op": "try",
                "branches": [
                    {
                        "op": "block",
                        "children": [
                            _apply("add_edge", {"edge": {"op": "ref", "name": "edge"}}),
                            _apply("add_edge", {"edge": {"op": "ref", "name": "edge"}}),
                            {"op": "emit"},
                        ],
                    },
                    {
                        "op": "block",
                        "children": [
                            _apply("add_edge", {"edge": {"op": "ref", "name": "edge"}}),
                            {"op": "emit"},
                        ],
                    },
                ],
            },
        }
    )
    result = _run(program, _cubic_graph(6))
    assert result.rewrite is not None
    assert len(result.rewrite.added_edges) == 1
    assert result.counters.actions == 3
    assert result.counters.bindings == 1


def test_witness_selector_observes_current_private_overlay_once() -> None:
    observed: list[GraphState] = []

    def loads(
        graph: GraphState,
    ) -> tuple[Mapping[tuple[int, int], int], Mapping[tuple[int, Edge], int]]:
        observed.append(graph)
        return {}, {}

    program = _validated(
        {
            "op": "let",
            "name": "edge",
            "value": _pick("non_edges_legal", {}),
            "body": {
                "op": "block",
                "children": [
                    _apply("add_edge", {"edge": {"op": "ref", "name": "edge"}}),
                    {
                        "op": "let",
                        "name": "vertices",
                        "value": _selector(
                            "vertices_witness_load_extreme",
                            {"length": 4, "mode": "max"},
                        ),
                        "body": {"op": "emit"},
                    },
                ],
            },
        }
    )
    graph = _cubic_graph(6)
    result = _run(
        program,
        graph,
        features=GraphFeatureInput(witness_load_provider=loads),
    )
    assert result.rewrite is not None
    assert len(observed) == 1
    assert len(observed[0].edges) == len(graph.edges) + 1


def test_gross_and_net_edge_limits_are_independent_program_failures() -> None:
    graph = _cubic_graph(6)
    program = _add_one_program()
    gross = _run(program, graph, limits=InterpreterLimits(maximum_actions=0))
    assert gross.failure is not None
    assert gross.failure.code == "BUDGET_EXHAUSTED"
    net = _run(program, graph, limits=InterpreterLimits(maximum_net_added_edges=0))
    assert net.failure is not None
    assert net.failure.code == "BUDGET_EXHAUSTED"


def test_noop_and_illegal_final_graph_are_no_plan_not_rewrites() -> None:
    noop = _run(_validated({"op": "emit"}), _cubic_graph(6))
    assert noop.no_plan == NoPlan("NO_EFFECT")
    remove = _validated(
        {
            "op": "let",
            "name": "edge",
            "value": _pick("edges_removable", {}),
            "body": {
                "op": "block",
                "children": [
                    _apply("remove_edge", {"edge": {"op": "ref", "name": "edge"}}),
                    {"op": "emit"},
                ],
            },
        }
    )
    illegal = _run(remove, _cubic_graph(6))
    assert illegal.no_plan == NoPlan("ILLEGAL_FINAL_STATE")
    assert illegal.rewrite is None


def test_reference_and_action_preconditions_forbid_invalid_graph_resources() -> None:
    with pytest.raises(ValueError, match="loop"):
        EdgeRef((1, 1))
    with pytest.raises(ValueError, match="distinct"):
        PathRef(1, 1, 2)
    with pytest.raises(ValueError, match="2/3/4"):
        MatchingRef(((0, 1),), ((0, 2),))

    runtime = GraphRuntime(_cubic_graph(8), GraphFeatureInput())
    runtime.apply_action("edge_fanout", {"edge": EdgeRef((0, 1)), "w": VertexRef(3)})
    assert (0, 1) not in runtime.overlay.edges
    assert {(0, 3), (1, 3)}.issubset(runtime.overlay.edges)
    runtime.apply_action("edge_fold", {"path": PathRef(0, 3, 1)})
    assert (0, 1) in runtime.overlay.edges
    runtime.apply_action(
        "relocate_endpoint",
        {"edge": EdgeRef((0, 1)), "keep": VertexRef(0), "new": VertexRef(2)},
    )
    assert (0, 1) not in runtime.overlay.edges
    assert (0, 2) in runtime.overlay.edges
    with pytest.raises(GraphPreconditionError, match="cannot be added"):
        runtime.apply_action("add_edge", {"edge": NonEdgeRef((0, 2))})


def _canonical_class(graph: GraphState) -> str:
    best: str | None = None
    for permutation in permutations(range(graph.order)):
        edges = sorted(normalized_edge((permutation[u], permutation[v])) for u, v in graph.edges)
        text = ";".join(f"{u}-{v}" for u, v in edges)
        if best is None or text < best:
            best = text
    assert best is not None
    return best


def _relabel(graph: GraphState, mapping: tuple[int, ...]) -> GraphState:
    return GraphState(
        graph.order,
        tuple(sorted(normalized_edge((mapping[u], mapping[v])) for u, v in graph.edges)),
    )


def test_strategy_is_label_oblivious_over_frozen_seed_distribution() -> None:
    graph = _cubic_graph(6)
    relabeled = _relabel(graph, (3, 5, 1, 4, 0, 2))
    program = _add_one_program()

    def distribution(candidate: GraphState) -> Counter[str]:
        classes: Counter[str] = Counter()
        for ordinal in range(96):
            context = ProgramContext(
                step_index=7,
                horizon=100,
                acceptance_profile_id="native-v3-test",
                invocation_ordinal=ordinal,
            )
            result = _run(program, candidate, context=context)
            assert result.rewrite is not None
            classes[_canonical_class(HOST.apply_rewrite(candidate, result.rewrite))] += 1
        return classes

    assert set(distribution(graph)) == set(distribution(relabeled))


def test_invalid_input_graph_fails_closed_without_escaping() -> None:
    graph = GraphState(4, ((0, 0),))
    result = _run(_validated({"op": "no_plan", "reason": "EXPLICIT"}), graph)
    assert not result.successful
    assert result.failure is not None
    assert result.failure.code == "INTERPRETER_FAULT"


def test_graph_resource_error_is_not_catchable() -> None:
    program = _validated(
        {
            "op": "try",
            "branches": [
                _add_one_program().ast["entry"],
                {"op": "no_plan", "reason": "EXPLICIT"},
            ],
        }
    )
    result = _run(
        program,
        _cubic_graph(6),
        limits=InterpreterLimits(maximum_net_added_edges=0),
    )
    assert result.failure is not None
    assert result.failure.code == "BUDGET_EXHAUSTED"
    assert result.no_plan is None
