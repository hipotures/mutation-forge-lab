"""Private graph overlay, selectors, and rewrite actions for Native v3."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Protocol

from mutation_forge.backends.base import ScoreProfileRecorder
from mutation_forge.models import Edge, GraphState, RewritePlan, normalized_edge

from .contracts import ValueType

GRAPH_RUNTIME_PROTOCOL_ID = "native_v3_graph_runtime_v1"
MAXIMUM_TIE_SET = 64


class GraphPreconditionError(Exception):
    """A local graph precondition failure that ordered fallback may catch."""


class GraphFinalStateError(Exception):
    """The private overlay cannot be emitted as a legal rewrite."""


class GraphResourceError(Exception):
    """A graph-specific dynamic resource limit was exceeded."""


class RewriteHost(Protocol):
    """The existing backend rewrite-validation boundary used at emit."""

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: ScoreProfileRecorder | None = None,
    ) -> GraphState: ...


@dataclass(frozen=True, slots=True, order=True)
class VertexRef:
    vertex: int

    def __post_init__(self) -> None:
        if self.vertex < 0:
            raise ValueError("VertexRef must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class EdgeRef:
    edge: Edge

    def __post_init__(self) -> None:
        edge = normalized_edge(self.edge)
        if edge[0] == edge[1]:
            raise ValueError("EdgeRef cannot contain a loop")
        object.__setattr__(self, "edge", edge)


@dataclass(frozen=True, slots=True, order=True)
class NonEdgeRef:
    edge: Edge

    def __post_init__(self) -> None:
        edge = normalized_edge(self.edge)
        if edge[0] == edge[1]:
            raise ValueError("NonEdgeRef cannot contain a loop")
        object.__setattr__(self, "edge", edge)


@dataclass(frozen=True, slots=True, order=True)
class PathRef:
    u: int
    w: int
    v: int

    def __post_init__(self) -> None:
        if min(self.u, self.w, self.v) < 0 or len({self.u, self.w, self.v}) != 3:
            raise ValueError("PathRef requires three distinct non-negative vertices")


@dataclass(frozen=True, slots=True)
class MatchingRef:
    removed_edges: tuple[Edge, ...]
    added_edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        removed = tuple(sorted(normalized_edge(edge) for edge in self.removed_edges))
        added = tuple(sorted(normalized_edge(edge) for edge in self.added_edges))
        if len(removed) not in {2, 3, 4} or len(added) != len(removed):
            raise ValueError("MatchingRef requires a legal 2/3/4-switch shape")
        if len(set(removed)) != len(removed) or len(set(added)) != len(added):
            raise ValueError("MatchingRef edges must be unique")
        if any(u == v for u, v in (*removed, *added)):
            raise ValueError("MatchingRef cannot contain loops")
        removed_endpoints = sorted(vertex for edge in removed for vertex in edge)
        added_endpoints = sorted(vertex for edge in added for vertex in edge)
        if len(set(removed_endpoints)) != 2 * len(removed):
            raise ValueError("MatchingRef source edges must be vertex-disjoint")
        if removed_endpoints != added_endpoints or set(removed) == set(added):
            raise ValueError("MatchingRef must preserve endpoints and change edges")
        object.__setattr__(self, "removed_edges", removed)
        object.__setattr__(self, "added_edges", added)


type ReferenceValue = VertexRef | EdgeRef | NonEdgeRef | PathRef | MatchingRef


@dataclass(frozen=True, slots=True)
class VertexSetRef:
    items: tuple[VertexRef, ...]

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(self.items)))
        if canonical != self.items:
            object.__setattr__(self, "items", canonical)


@dataclass(frozen=True, slots=True)
class EdgeSetRef:
    items: tuple[EdgeRef, ...]

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(self.items)))
        if canonical != self.items:
            object.__setattr__(self, "items", canonical)


@dataclass(frozen=True, slots=True)
class SelectionPopulation:
    value_type: ValueType
    items: tuple[NonEdgeRef | PathRef | MatchingRef, ...]
    population_size: int


type Population = VertexSetRef | EdgeSetRef | SelectionPopulation
type GraphScalar = bool | int | Fraction | str | ReferenceValue
type WitnessLoadProvider = Callable[
    [GraphState],
    tuple[
        Mapping[tuple[int, int], int],
        Mapping[tuple[int, Edge], int],
    ],
]
type RandomIndex = Callable[[str, tuple[int, ...]], int]


@dataclass(frozen=True, slots=True)
class GraphFeatureInput:
    vertex_witness_load: Mapping[tuple[int, int], int] = field(default_factory=dict)
    edge_witness_load: Mapping[tuple[int, Edge], int] = field(default_factory=dict)
    witness_load_provider: WitnessLoadProvider | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(slots=True)
class GraphOverlay:
    order: int
    initial_edges: frozenset[Edge]
    edges: set[Edge]

    @classmethod
    def from_graph(cls, graph: GraphState) -> GraphOverlay:
        if graph.order < 1:
            raise ValueError("input graph order must be positive")
        normalized = tuple(normalized_edge(edge) for edge in graph.edges)
        if len(set(normalized)) != len(normalized):
            raise ValueError("input graph contains duplicate edges")
        if any(
            u == v or not (0 <= u < graph.order and 0 <= v < graph.order) for u, v in normalized
        ):
            raise ValueError("input graph contains an invalid edge")
        initial = frozenset(normalized)
        return cls(graph.order, initial, set(initial))

    def snapshot(self) -> frozenset[Edge]:
        return frozenset(self.edges)

    def restore(self, snapshot: frozenset[Edge]) -> None:
        self.edges = set(snapshot)

    def graph(self) -> GraphState:
        return GraphState(self.order, tuple(sorted(self.edges)))

    def adjacency(self) -> tuple[set[int], ...]:
        result = tuple(set[int]() for _ in range(self.order))
        for u, v in self.edges:
            result[u].add(v)
            result[v].add(u)
        return result


def population_items(population: Population) -> tuple[ReferenceValue, ...]:
    return population.items


def population_type(population: Population) -> ValueType:
    if isinstance(population, VertexSetRef):
        return ValueType.VERTEX_SET
    if isinstance(population, EdgeSetRef):
        return ValueType.EDGE_SET
    return population.value_type


def reference_type(value: ReferenceValue) -> ValueType:
    if isinstance(value, VertexRef):
        return ValueType.VERTEX
    if isinstance(value, EdgeRef):
        return ValueType.EDGE
    if isinstance(value, NonEdgeRef):
        return ValueType.NON_EDGE
    if isinstance(value, PathRef):
        return ValueType.PATH
    return ValueType.MATCHING


def _reference_key(value: ReferenceValue) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _extreme(values: Mapping[ReferenceValue, int], mode: str) -> tuple[ReferenceValue, ...]:
    if not values:
        return ()
    target = min(values.values()) if mode == "min" else max(values.values())
    return tuple(
        sorted(
            (item for item, value in values.items() if value == target),
            key=_reference_key,
        )
    )


def _connected(overlay: GraphOverlay) -> bool:
    adjacency = overlay.adjacency()
    visited = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == overlay.order


def _tarjan(overlay: GraphOverlay) -> tuple[frozenset[int], frozenset[Edge]]:
    adjacency = overlay.adjacency()
    discovery = [-1] * overlay.order
    low = [0] * overlay.order
    parent = [-1] * overlay.order
    articulations: set[int] = set()
    bridges: set[Edge] = set()
    clock = 0

    def visit(vertex: int) -> None:
        nonlocal clock
        discovery[vertex] = low[vertex] = clock
        clock += 1
        children = 0
        for neighbor in sorted(adjacency[vertex]):
            if discovery[neighbor] == -1:
                parent[neighbor] = vertex
                children += 1
                visit(neighbor)
                low[vertex] = min(low[vertex], low[neighbor])
                if parent[vertex] == -1 and children > 1:
                    articulations.add(vertex)
                if parent[vertex] != -1 and low[neighbor] >= discovery[vertex]:
                    articulations.add(vertex)
                if low[neighbor] > discovery[vertex]:
                    bridges.add(normalized_edge((vertex, neighbor)))
            elif neighbor != parent[vertex]:
                low[vertex] = min(low[vertex], discovery[neighbor])

    for vertex in range(overlay.order):
        if discovery[vertex] == -1:
            visit(vertex)
    return frozenset(articulations), frozenset(bridges)


def _distances(overlay: GraphOverlay, source: int) -> tuple[int, ...]:
    adjacency = overlay.adjacency()
    distances = [-1] * overlay.order
    distances[source] = 0
    pending: deque[int] = deque([source])
    while pending:
        vertex = pending.popleft()
        for neighbor in sorted(adjacency[vertex]):
            if distances[neighbor] == -1:
                distances[neighbor] = distances[vertex] + 1
                pending.append(neighbor)
    return tuple(distances)


@dataclass(slots=True)
class GraphRuntime:
    graph: GraphState
    features: GraphFeatureInput
    overlay: GraphOverlay = field(init=False)
    _witness_cache: dict[
        tuple[Edge, ...],
        tuple[Mapping[tuple[int, int], int], Mapping[tuple[int, Edge], int]],
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.overlay = GraphOverlay.from_graph(self.graph)

    def feature_values(self) -> dict[str, int]:
        degrees = tuple(len(neighbors) for neighbors in self.overlay.adjacency())
        return {
            "order": self.overlay.order,
            "edge_count": len(self.overlay.edges),
            "minimum_degree": min(degrees),
            "maximum_degree": max(degrees),
        }

    def _witness_loads(
        self,
    ) -> tuple[Mapping[tuple[int, int], int], Mapping[tuple[int, Edge], int]]:
        key = tuple(sorted(self.overlay.edges))
        cached = self._witness_cache.get(key)
        if cached is not None:
            return cached
        vertices = dict(self.features.vertex_witness_load)
        edges = dict(self.features.edge_witness_load)
        if self.features.witness_load_provider is not None:
            sampled_vertices, sampled_edges = self.features.witness_load_provider(
                self.overlay.graph()
            )
            vertices.update(sampled_vertices)
            edges.update(sampled_edges)
        result = vertices, edges
        self._witness_cache[key] = result
        return result

    def _bounded(
        self,
        items: tuple[ReferenceValue, ...],
        *,
        value_type: ValueType,
        path: str,
        random_index: RandomIndex,
    ) -> Population:
        population_size = len(items)
        if population_size > MAXIMUM_TIE_SET:
            reservoir = list(items[:MAXIMUM_TIE_SET])
            for index in range(MAXIMUM_TIE_SET, population_size):
                selected = random_index(
                    f"{path}/reservoir/{index}",
                    tuple(1 for _ in range(index + 1)),
                )
                if selected < MAXIMUM_TIE_SET:
                    reservoir[selected] = items[index]
            items = tuple(sorted(reservoir, key=_reference_key))
        if value_type is ValueType.VERTEX_SET:
            return VertexSetRef(tuple(item for item in items if isinstance(item, VertexRef)))
        if value_type is ValueType.EDGE_SET:
            return EdgeSetRef(tuple(item for item in items if isinstance(item, EdgeRef)))
        return SelectionPopulation(
            value_type,
            tuple(item for item in items if isinstance(item, NonEdgeRef | PathRef | MatchingRef)),
            population_size,
        )

    def _matching_candidates(
        self,
        k: int,
        *,
        path: str,
        random_index: RandomIndex,
    ) -> tuple[ReferenceValue, ...]:
        if k not in {2, 3, 4}:
            raise GraphPreconditionError("k-switch requires k in {2,3,4}")
        edges = tuple(sorted(self.overlay.edges))
        if len(edges) < k:
            return ()
        candidates: set[MatchingRef] = set()
        for attempt in range(64):
            available = list(edges)
            removed: list[Edge] = []
            for index in range(k):
                selected = random_index(
                    f"{path}/attempt/{attempt}/edge/{index}",
                    tuple(1 for _ in available),
                )
                removed.append(available.pop(selected))
            endpoints = [vertex for edge in removed for vertex in edge]
            if len(set(endpoints)) != 2 * k:
                continue
            shuffled = list(endpoints)
            for index in range(len(shuffled) - 1, 0, -1):
                selected = random_index(
                    f"{path}/attempt/{attempt}/shuffle/{index}",
                    tuple(1 for _ in range(index + 1)),
                )
                shuffled[index], shuffled[selected] = shuffled[selected], shuffled[index]
            added = tuple(
                sorted(
                    normalized_edge((shuffled[index], shuffled[index + 1]))
                    for index in range(0, len(shuffled), 2)
                )
            )
            try:
                candidate = MatchingRef(tuple(removed), added)
            except ValueError:
                continue
            remaining = self.overlay.edges.difference(candidate.removed_edges)
            if any(edge in remaining for edge in candidate.added_edges):
                continue
            candidates.add(candidate)
        return tuple(
            sorted(
                candidates,
                key=lambda item: (item.removed_edges, item.added_edges),
            )
        )

    def select(
        self,
        selector_id: str,
        arguments: Mapping[str, GraphScalar],
        *,
        path: str,
        random_index: RandomIndex,
    ) -> Population:
        adjacency = self.overlay.adjacency()
        degrees = tuple(len(neighbors) for neighbors in adjacency)
        mode = arguments.get("mode", "max")
        if mode not in {"min", "max"}:
            raise GraphPreconditionError("selector mode must be min or max")
        items: tuple[ReferenceValue, ...]
        value_type: ValueType
        if selector_id == "vertices_degree_extreme":
            items = _extreme(
                {VertexRef(vertex): degree for vertex, degree in enumerate(degrees)},
                str(mode),
            )
            value_type = ValueType.VERTEX_SET
        elif selector_id == "vertices_degree_class":
            degree = arguments.get("degree")
            if isinstance(degree, bool) or not isinstance(degree, int):
                raise TypeError("degree must be an integer")
            items = tuple(
                VertexRef(vertex) for vertex, actual in enumerate(degrees) if actual == degree
            )
            value_type = ValueType.VERTEX_SET
        elif selector_id == "vertices_witness_load_extreme":
            length = arguments.get("length")
            if isinstance(length, bool) or not isinstance(length, int):
                raise TypeError("length must be an integer")
            vertex_loads, _ = self._witness_loads()
            items = _extreme(
                {
                    VertexRef(vertex): vertex_loads.get((length, vertex), 0)
                    for vertex in range(self.overlay.order)
                },
                str(mode),
            )
            value_type = ValueType.VERTEX_SET
        elif selector_id == "edges_witness_load_extreme":
            length = arguments.get("length")
            if isinstance(length, bool) or not isinstance(length, int):
                raise TypeError("length must be an integer")
            _, edge_loads = self._witness_loads()
            items = _extreme(
                {
                    EdgeRef(edge): edge_loads.get((length, edge), 0)
                    for edge in sorted(self.overlay.edges)
                },
                str(mode),
            )
            value_type = ValueType.EDGE_SET
        elif selector_id == "vertices_articulation_risk":
            articulations, _ = _tarjan(self.overlay)
            items = _extreme(
                {
                    VertexRef(vertex): int(vertex in articulations)
                    for vertex in range(self.overlay.order)
                },
                str(mode),
            )
            value_type = ValueType.VERTEX_SET
        elif selector_id == "edges_bridge_risk":
            _, bridges = _tarjan(self.overlay)
            items = _extreme(
                {EdgeRef(edge): int(edge in bridges) for edge in sorted(self.overlay.edges)},
                str(mode),
            )
            value_type = ValueType.EDGE_SET
        elif selector_id == "edges_removable":
            items = tuple(EdgeRef(edge) for edge in sorted(self.overlay.edges))
            value_type = ValueType.EDGE_SET
        elif selector_id == "vertices_distance_band":
            source = arguments.get("source")
            minimum = arguments.get("minimum")
            maximum = arguments.get("maximum")
            if not isinstance(source, VertexRef):
                raise TypeError("source must be VertexRef")
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or minimum < 0
                or maximum < minimum
            ):
                raise GraphPreconditionError("distance band must satisfy 0 <= min <= max")
            if source.vertex >= self.overlay.order:
                raise GraphPreconditionError("source vertex is outside the graph")
            distances = _distances(self.overlay, source.vertex)
            items = tuple(
                VertexRef(vertex)
                for vertex, distance in enumerate(distances)
                if minimum <= distance <= maximum
            )
            value_type = ValueType.VERTEX_SET
        elif selector_id == "non_edges_from_vertex":
            source = arguments.get("vertex")
            if not isinstance(source, VertexRef):
                raise TypeError("vertex must be VertexRef")
            if source.vertex >= self.overlay.order:
                raise GraphPreconditionError("vertex is outside the graph")
            items = tuple(
                NonEdgeRef((source.vertex, vertex))
                for vertex in range(self.overlay.order)
                if vertex != source.vertex and vertex not in adjacency[source.vertex]
            )
            value_type = ValueType.NON_EDGE
        elif selector_id in {"non_edges_legal", "non_edges_local_cycle_risk"}:
            values: dict[ReferenceValue, int] = {}
            for u in range(self.overlay.order):
                for v in range(u + 1, self.overlay.order):
                    edge = (u, v)
                    if edge not in self.overlay.edges:
                        values[NonEdgeRef(edge)] = len(adjacency[u] & adjacency[v])
            items = (
                tuple(sorted(values, key=_reference_key))
                if selector_id == "non_edges_legal"
                else _extreme(values, str(mode))
            )
            value_type = ValueType.NON_EDGE
        elif selector_id == "paths_length_two":
            paths: list[ReferenceValue] = []
            for center in range(self.overlay.order):
                for u, v in combinations(sorted(adjacency[center]), 2):
                    paths.append(PathRef(u, center, v))
            items = tuple(sorted(paths, key=_reference_key))
            value_type = ValueType.PATH
        elif selector_id == "matching_k_switch_reconnections":
            k = arguments.get("k")
            if isinstance(k, bool) or not isinstance(k, int):
                raise TypeError("k must be an integer")
            items = self._matching_candidates(k, path=path, random_index=random_index)
            value_type = ValueType.MATCHING
        else:
            raise TypeError(f"unknown selector: {selector_id}")
        return self._bounded(
            items,
            value_type=value_type,
            path=path,
            random_index=random_index,
        )

    def weight(self, item: ReferenceValue, feature: str) -> int:
        if feature == "uniform":
            return 1
        if isinstance(item, VertexRef):
            degree = len(self.overlay.adjacency()[item.vertex])
            if feature == "degree":
                return max(1, degree)
            if feature == "inverse_degree":
                return max(1, self.overlay.order - degree)
        raise TypeError(f"unsupported weight feature: {feature}")

    def apply_action(
        self,
        action_id: str,
        arguments: Mapping[str, GraphScalar],
    ) -> None:
        edges = self.overlay.edges
        if action_id == "add_edge":
            reference = arguments.get("edge")
            if not isinstance(reference, NonEdgeRef):
                raise TypeError("add_edge requires NonEdgeRef")
            edge = reference.edge
            if max(edge) >= self.overlay.order or edge in edges:
                raise GraphPreconditionError("edge cannot be added")
            edges.add(edge)
            return
        if action_id == "remove_edge":
            reference = arguments.get("edge")
            if not isinstance(reference, EdgeRef):
                raise TypeError("remove_edge requires EdgeRef")
            if reference.edge not in edges:
                raise GraphPreconditionError("edge is absent")
            edges.remove(reference.edge)
            return
        if action_id == "relocate_endpoint":
            reference = arguments.get("edge")
            keep = arguments.get("keep")
            new = arguments.get("new")
            if (
                not isinstance(reference, EdgeRef)
                or not isinstance(keep, VertexRef)
                or not isinstance(new, VertexRef)
            ):
                raise TypeError("relocate_endpoint received invalid reference types")
            edge = reference.edge
            if (
                edge not in edges
                or keep.vertex not in edge
                or new.vertex >= self.overlay.order
                or new.vertex in edge
            ):
                raise GraphPreconditionError("invalid relocation references")
            replacement = normalized_edge((keep.vertex, new.vertex))
            if replacement in edges:
                raise GraphPreconditionError("replacement edge exists")
            edges.remove(edge)
            edges.add(replacement)
            return
        if action_id == "k_switch":
            matching = arguments.get("matching")
            if not isinstance(matching, MatchingRef):
                raise TypeError("k_switch requires MatchingRef")
            if not set(matching.removed_edges).issubset(edges):
                raise GraphPreconditionError("matching source edge is absent")
            remaining = edges.difference(matching.removed_edges)
            if any(edge in remaining for edge in matching.added_edges):
                raise GraphPreconditionError("matching target edge exists")
            edges.difference_update(matching.removed_edges)
            edges.update(matching.added_edges)
            return
        if action_id == "edge_fanout":
            reference = arguments.get("edge")
            vertex = arguments.get("w")
            if not isinstance(reference, EdgeRef) or not isinstance(vertex, VertexRef):
                raise TypeError("edge_fanout received invalid reference types")
            edge = reference.edge
            u, v = edge
            additions = {
                normalized_edge((u, vertex.vertex)),
                normalized_edge((v, vertex.vertex)),
            }
            if (
                edge not in edges
                or vertex.vertex >= self.overlay.order
                or vertex.vertex in edge
                or any(addition in edges for addition in additions)
            ):
                raise GraphPreconditionError("fanout precondition failed")
            edges.remove(edge)
            edges.update(additions)
            return
        if action_id == "edge_fold":
            reference = arguments.get("path")
            if not isinstance(reference, PathRef):
                raise TypeError("edge_fold requires PathRef")
            first = normalized_edge((reference.u, reference.w))
            second = normalized_edge((reference.w, reference.v))
            replacement = normalized_edge((reference.u, reference.v))
            if first not in edges or second not in edges or replacement in edges:
                raise GraphPreconditionError("fold precondition failed")
            edges.remove(first)
            edges.remove(second)
            edges.add(replacement)
            return
        raise TypeError(f"unknown action: {action_id}")

    def emit(
        self,
        *,
        host: RewriteHost,
        program_hash: str,
        gross_actions: int,
        selector_cost_units: int,
        maximum_net_added_edges: int,
        maximum_net_removed_edges: int,
    ) -> RewritePlan:
        adjacency = self.overlay.adjacency()
        if not _connected(self.overlay) or min(map(len, adjacency)) < 3:
            raise GraphFinalStateError(
                "final graph must be connected with minimum degree at least three"
            )
        removed = tuple(sorted(self.overlay.initial_edges.difference(self.overlay.edges)))
        added = tuple(sorted(self.overlay.edges.difference(self.overlay.initial_edges)))
        if not removed and not added:
            raise GraphPreconditionError("final rewrite has no effect")
        if len(removed) > maximum_net_removed_edges or len(added) > maximum_net_added_edges:
            raise GraphResourceError("net rewrite exceeds edge limits")
        rewrite = RewritePlan(
            removed_edges=removed,
            added_edges=added,
            operator_family="native_v3_program",
            metadata={
                "program_hash": program_hash,
                "graph_runtime_protocol_id": GRAPH_RUNTIME_PROTOCOL_ID,
                "gross_actions": gross_actions,
                "selector_cost_units": selector_cost_units,
            },
        )
        try:
            candidate = host.apply_rewrite(self.graph, rewrite)
        except (RuntimeError, ValueError) as exc:
            raise GraphFinalStateError(str(exc)) from exc
        expected = self.overlay.graph()
        if candidate.order != self.graph.order or candidate != expected:
            raise GraphFinalStateError("rewrite host returned a different graph")
        return rewrite
