from __future__ import annotations

from collections.abc import Mapping

import pytest

from mutation_forge.models import Edge, GraphState, GraphValidation, RewritePlan, normalized_edge
from mutation_forge.native_v3_python import (
    GraphFeatureInputV1,
    IllegalRewriteError,
    NoPlan,
    PolicyContextV1,
    PolicyRuntimeLimitsV1,
    SafeAPIProgramError,
    SafeGraphSessionV1,
    graph_view_v1,
)


def _cubic_graph(order: int) -> GraphState:
    edges = {normalized_edge((vertex, (vertex + 1) % order)) for vertex in range(order)}
    edges.update((vertex, vertex + order // 2) for vertex in range(order // 2))
    return GraphState(order, tuple(sorted(edges)))


def _degrees(graph: GraphState) -> tuple[int, ...]:
    values = [0] * graph.order
    for u, v in graph.edges:
        values[u] += 1
        values[v] += 1
    return tuple(values)


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


class _MinDegreeHost:
    def validate(self, graph: GraphState) -> GraphValidation:
        errors: list[str] = []
        if len(set(graph.edges)) != len(graph.edges):
            errors.append("duplicate")
        if any(
            u == v or not (0 <= u < graph.order and 0 <= v < graph.order)
            for u, v in graph.edges
        ):
            errors.append("invalid edge")
        if graph.order and min(_degrees(graph)) < 3:
            errors.append("minimum degree")
        if graph.order and not _connected(graph):
            errors.append("connectivity")
        return GraphValidation(not errors, tuple(errors))

    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState:
        current = set(graph.edges)
        removed = set(rewrite.removed_edges)
        added = set(rewrite.added_edges)
        if not removed.issubset(current) or added & (current - removed):
            raise IllegalRewriteError("invalid rewrite delta")
        candidate = GraphState(
            graph.order,
            tuple(sorted((current - removed) | added)),
        )
        validation = self.validate(candidate)
        if not validation.valid:
            raise IllegalRewriteError("; ".join(validation.errors))
        return candidate


HOST = _MinDegreeHost()


def _context(invocation_ordinal: int = 0) -> PolicyContextV1:
    return PolicyContextV1(
        step_index=2,
        horizon=10,
        acceptance_profile_id="m2-fixture",
        stagnation_steps=1,
        exploration_window_index=0,
        accepted_rewrites=1,
        accepted_non_improving_rewrites=0,
        consecutive_non_improving_rewrites=0,
        witness_cap=100,
        invocation_ordinal=invocation_ordinal,
        forbidden_lengths=(4, 6),
    )


def _session(
    graph: GraphState | None = None,
    *,
    context: PolicyContextV1 | None = None,
    features: GraphFeatureInputV1 | None = None,
    limits: PolicyRuntimeLimitsV1 | None = None,
) -> SafeGraphSessionV1:
    return SafeGraphSessionV1(
        graph=graph or _cubic_graph(6),
        context=context or _context(),
        seed=17,
        program_hash="a" * 64,
        rewrite_host=HOST,
        limits=limits or PolicyRuntimeLimitsV1(),
        features=features or GraphFeatureInputV1(),
    )


def _references(
    session: SafeGraphSessionV1,
    method: str,
    arguments: Mapping[str, object] | None = None,
) -> tuple[tuple[str, object], ...]:
    encoded = session.handle_call(method, arguments or {})
    assert isinstance(encoded, list)
    result: list[tuple[str, object]] = []
    for item in encoded:
        assert isinstance(item, dict)
        token = item["$ref"]
        assert isinstance(token, str)
        minted = session._references[token]  # noqa: SLF001 - host-only parity evidence
        result.append((minted.reference.kind, minted.reference.payload))
    return tuple(result)


def test_graph_view_is_exactly_label_opaque_scalars() -> None:
    graph = _cubic_graph(8)
    view = graph_view_v1(graph)
    assert view.order == 8
    assert view.edge_count == 12
    assert view.minimum_degree == view.maximum_degree == 3
    assert {field for field in view.__slots__} == {
        "order",
        "edge_count",
        "minimum_degree",
        "maximum_degree",
    }


def test_witness_selectors_match_donor_and_observe_private_overlay() -> None:
    graph = _cubic_graph(6)
    extra = next(
        edge
        for edge in (
            (u, v)
            for u in range(graph.order)
            for v in range(u + 1, graph.order)
        )
        if edge not in graph.edges
    )
    observed: list[GraphState] = []

    def loads(
        current: GraphState,
    ) -> tuple[Mapping[tuple[int, int], int], Mapping[tuple[int, Edge], int]]:
        observed.append(current)
        return {(4, 0): 10}, {(4, extra): 20}

    session = _session(features=GraphFeatureInputV1(witness_load_provider=loads))
    non_edge = session.handle_call("non_edges_legal", {})[0]  # type: ignore[index]
    session.handle_call("add_edge", {"edge": non_edge})
    vertices = _references(
        session,
        "vertices_witness_load_extreme",
        {"length": 4, "mode": "max"},
    )
    edges = _references(
        session,
        "edges_witness_load_extreme",
        {"length": 4, "mode": "max"},
    )
    assert vertices == (("vertex", 0),)
    assert edges == (("edge", extra),)
    assert len(observed) == 1
    assert len(observed[0].edges) == len(graph.edges) + 1


def test_distance_selector_requires_current_invocation_vertex_reference() -> None:
    session = _session()
    source = session.handle_call("vertices_degree_extreme", {"mode": "max"})[0]  # type: ignore[index]
    actual = _references(
        session,
        "vertices_distance_band",
        {"source": source, "minimum": 1, "maximum": 1},
    )
    source_token = source["$ref"]  # type: ignore[index]
    source_vertex = session._references[source_token].reference.payload  # noqa: SLF001
    assert actual
    assert all(kind == "vertex" for kind, _payload in actual)
    assert source_vertex not in {payload for _kind, payload in actual}

    foreign = _session(context=_context(1))
    with pytest.raises(SafeAPIProgramError, match="current invocation") as error:
        foreign.handle_call(
            "vertices_distance_band",
            {"source": source, "minimum": 1, "maximum": 1},
        )
    assert error.value.code == "STALE_OR_FOREIGN_REFERENCE"


def test_k_switch_and_edge_fold_preserve_donor_action_semantics() -> None:
    graph = _cubic_graph(8)
    session = _session(graph)
    matchings = session.handle_call("matching_k_switch_reconnections", {"k": 2})
    assert isinstance(matchings, list) and matchings
    session.handle_call("k_switch", {"matching": matchings[0]})
    switched = session.overlay.graph()
    assert len(switched.edges) == len(graph.edges)
    assert _degrees(switched) == _degrees(graph)

    fold_session = _session(graph)
    paths = fold_session.handle_call("paths_length_two", {})
    assert isinstance(paths, list)
    foldable = next(
        item
        for item in paths
        if normalized_edge(
            (
                fold_session._references[item["$ref"]].reference.payload[0],  # noqa: SLF001
                fold_session._references[item["$ref"]].reference.payload[2],  # noqa: SLF001
            )
        )
        not in graph.edges
    )
    fold_session.handle_call("edge_fold", {"path": foldable})
    assert len(fold_session.overlay.edges) == len(graph.edges) - 1


def test_emit_mints_plan_and_no_plan_with_fail_closed_final_checks() -> None:
    session = _session()
    non_edges = session.handle_call("non_edges_legal", {})
    assert isinstance(non_edges, list)
    session.handle_call("add_edge", {"edge": non_edges[0]})
    encoded = session.handle_call("emit", {})
    assert isinstance(encoded, dict)
    result = session.resolve_result(encoded["$host_result"], encoded["kind"])  # type: ignore[index]
    assert isinstance(result, RewritePlan)
    assert result.operator_family == "native_v3_python_policy"
    assert len(result.added_edges) == 1

    noop = _session()
    encoded_noop = noop.handle_call("emit", {})
    assert isinstance(encoded_noop, dict)
    assert noop.resolve_result(
        encoded_noop["$host_result"],  # type: ignore[index]
        encoded_noop["kind"],  # type: ignore[index]
    ) == NoPlan("NO_EFFECT")

    illegal = _session()
    edge = illegal.handle_call("edges_removable", {})[0]  # type: ignore[index]
    illegal.handle_call("remove_edge", {"edge": edge})
    encoded_illegal = illegal.handle_call("emit", {})
    assert isinstance(encoded_illegal, dict)
    assert illegal.resolve_result(
        encoded_illegal["$host_result"],  # type: ignore[index]
        encoded_illegal["kind"],  # type: ignore[index]
    ) == NoPlan("ILLEGAL_FINAL_STATE")


def test_pick_and_semantic_trace_replay_without_opaque_token_values() -> None:
    traces = []
    selected_ordinals = []
    for _ in range(2):
        session = _session()
        items = session.handle_call("non_edges_legal", {})
        chosen = session.handle_call(
            "pick",
            {"items": items, "seed": 17, "salt": "fixture", "feature": "uniform"},
        )
        assert isinstance(chosen, dict)
        session.handle_call("add_edge", {"edge": chosen})
        session.handle_call("emit", {})
        traces.append(tuple(event.as_dict() for event in session.semantic_trace))
        selected_ordinals.append(
            session._references[chosen["$ref"]].ordinal  # noqa: SLF001
        )
    assert traces[0] == traces[1]
    assert selected_ordinals == [selected_ordinals[0], selected_ordinals[0]]
    assert "$ref" not in repr(traces[0])
    assert all("wall" not in repr(event) for event in traces[0])


def test_reference_action_and_random_budgets_fail_closed() -> None:
    limited = _session(limits=PolicyRuntimeLimitsV1(total_api_calls=1))
    limited.handle_call("non_edges_legal", {})
    with pytest.raises(SafeAPIProgramError) as error:
        limited.handle_call("no_plan", {})
    assert error.value.code == "API_CALL_BUDGET_EXCEEDED"

    wrong_seed = _session()
    items = wrong_seed.handle_call("non_edges_legal", {})
    with pytest.raises(SafeAPIProgramError) as error:
        wrong_seed.handle_call(
            "pick",
            {"items": items, "seed": 18, "salt": 0, "feature": "uniform"},
        )
    assert error.value.code == "INVALID_API_ARGUMENT"

    unknown = _session()
    with pytest.raises(SafeAPIProgramError) as error:
        unknown.handle_call("future_unrecognized_method", {})
    assert error.value.code == "UNKNOWN_API_METHOD"

    oversized_pick = _session()
    one = oversized_pick.handle_call("non_edges_legal", {})[0]  # type: ignore[index]
    with pytest.raises(SafeAPIProgramError) as error:
        oversized_pick.handle_call(
            "pick",
            {
                "items": [one] * 65,
                "seed": 17,
                "salt": 0,
                "feature": "uniform",
            },
        )
    assert error.value.code == "INVALID_API_ARGUMENT"
