"""Bounded interpreter for Native v3 mutation programs."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from itertools import combinations
from typing import cast

from mutation_forge.models import Edge, GraphState, JsonValue, RewritePlan, normalized_edge

from .contracts import SELECTOR_COSTS, ProgramLimits, ValidatedProgram
from .randomness import derive_seed64, uniform_below, weighted_index

INTERPRETER_PROTOCOL_ID = "native_v3_interpreter_v1"


class NoPlanReason(StrEnum):
    EXPLICIT = "EXPLICIT"
    NO_MATCH = "NO_MATCH"
    ILLEGAL_FINAL_STATE = "ILLEGAL_FINAL_STATE"
    NO_EFFECT = "NO_EFFECT"


class BranchFailureCode(StrEnum):
    NO_MATCH = "NO_MATCH"
    LOCAL_PRECONDITION_FAILED = "LOCAL_PRECONDITION_FAILED"
    ILLEGAL_FINAL_STATE = "ILLEGAL_FINAL_STATE"
    NO_EFFECT = "NO_EFFECT"


@dataclass(frozen=True, slots=True)
class VertexRef:
    vertex: int


@dataclass(frozen=True, slots=True)
class EdgeRef:
    edge: Edge


@dataclass(frozen=True, slots=True)
class NonEdgeRef:
    edge: Edge


@dataclass(frozen=True, slots=True)
class Path2Ref:
    u: int
    w: int
    v: int


@dataclass(frozen=True, slots=True)
class MatchingRef:
    removed_edges: tuple[Edge, ...]
    added_edges: tuple[Edge, ...]


type ReferenceValue = VertexRef | EdgeRef | NonEdgeRef | Path2Ref | MatchingRef
type RuntimeValue = bool | int | str | Fraction | ReferenceValue | "SelectionPopulation"


@dataclass(frozen=True, slots=True)
class SelectionPopulation:
    selector_id: str
    items: tuple[ReferenceValue, ...]
    population_size: int


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    selector_id: str
    population_size: int
    sample_size: int
    selected_index: int | None
    path: str
    sampling_seed64: int
    selected_reference_sha256: str | None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "selector_id": self.selector_id,
            "population_size": self.population_size,
            "sample_size": self.sample_size,
            "selected_index": self.selected_index,
            "path": self.path,
            "sampling_seed64": self.sampling_seed64,
            "selected_reference_sha256": self.selected_reference_sha256,
        }


@dataclass(frozen=True, slots=True)
class NoPlan:
    reason: NoPlanReason


@dataclass(frozen=True, slots=True)
class ProgramFailure:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class InvocationResult:
    rewrite: RewritePlan | None
    no_plan: NoPlan | None
    failure: ProgramFailure | None
    selector_calls: int
    selector_cost_units: int
    gross_actions: int
    selections: tuple[SelectionRecord, ...]

    @property
    def successful(self) -> bool:
        return self.rewrite is not None or self.no_plan is not None


@dataclass(frozen=True, slots=True)
class ProgramContext:
    protocol_id: str
    step_index: int
    horizon: int
    acceptance_profile_id: str
    stagnation_steps: int
    exploration_window_index: int | None
    accepted_rewrites: int
    accepted_non_improving_rewrites: int
    consecutive_non_improving_rewrites: int
    target_forbidden_lengths: tuple[int, ...]
    witness_cap: int
    current_score_component_bounds: tuple[tuple[int, int, int, str], ...] = ()

    def value(self, field_name: str) -> RuntimeValue:
        values: dict[str, RuntimeValue] = {
            "step_index": self.step_index,
            "horizon": self.horizon,
            "acceptance_profile_id": self.acceptance_profile_id,
            "stagnation_steps": self.stagnation_steps,
            "exploration_window_index": (
                self.exploration_window_index if self.exploration_window_index is not None else -1
            ),
            "accepted_rewrites": self.accepted_rewrites,
            "accepted_non_improving_rewrites": self.accepted_non_improving_rewrites,
            "consecutive_non_improving_rewrites": self.consecutive_non_improving_rewrites,
            "witness_cap": self.witness_cap,
        }
        try:
            return values[field_name]
        except KeyError as exc:
            raise _ProgramRuntimeError(
                "UNKNOWN_CONTEXT_FIELD", "/", f"unsupported ctx field: {field_name}"
            ) from exc


@dataclass(frozen=True, slots=True)
class GraphFeatureInput:
    total_witness_interval: tuple[int, int] = (0, 0)
    weighted_penalty_interval: tuple[int, int] = (0, 0)
    energy_interval: tuple[int, int] = (0, 0)
    vertex_witness_load: Mapping[tuple[int, int], int] = field(default_factory=dict)
    edge_witness_load: Mapping[tuple[int, Edge], int] = field(default_factory=dict)
    witness_load_provider: (
        Callable[
            [GraphState],
            tuple[
                Mapping[tuple[int, int], int],
                Mapping[tuple[int, Edge], int],
            ],
        ]
        | None
    ) = field(default=None, repr=False, compare=False)


class _BranchFailure(Exception):
    def __init__(self, code: BranchFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class _ProgramRuntimeError(Exception):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


class _TerminalResult(Exception):
    def __init__(self, result: RewritePlan | NoPlan) -> None:
        super().__init__()
        self.result = result


@dataclass(slots=True)
class _Overlay:
    order: int
    initial_edges: frozenset[Edge]
    edges: set[Edge]

    @classmethod
    def from_graph(cls, graph: GraphState) -> _Overlay:
        initial = frozenset(graph.edges)
        if any(u == v or not (0 <= u < graph.order and 0 <= v < graph.order) for u, v in initial):
            raise ValueError("input graph has invalid edges")
        if len(initial) != len(graph.edges):
            raise ValueError("input graph has duplicate edges")
        return cls(graph.order, initial, set(initial))

    def clone_edges(self) -> set[Edge]:
        return set(self.edges)

    def graph(self) -> GraphState:
        return GraphState(self.order, tuple(sorted(self.edges)))

    def adjacency(self) -> tuple[set[int], ...]:
        result: tuple[set[int], ...] = tuple(set() for _ in range(self.order))
        for u, v in self.edges:
            result[u].add(v)
            result[v].add(u)
        return result


@dataclass(slots=True)
class _Runtime:
    program: ValidatedProgram
    graph: GraphState
    context: ProgramContext
    features: GraphFeatureInput
    policy_seed: int
    episode_id: str
    limits: ProgramLimits
    overlay: _Overlay
    selector_calls: int = 0
    selector_cost: int = 0
    gross_actions: int = 0
    draw_ordinal: int = 0
    selections: list[SelectionRecord] = field(default_factory=list)
    query_cache: dict[tuple[str, str], SelectionPopulation] = field(default_factory=dict)
    witness_load_cache: dict[
        tuple[Edge, ...],
        tuple[
            Mapping[tuple[int, int], int],
            Mapping[tuple[int, Edge], int],
        ],
    ] = field(default_factory=dict)

    @property
    def random_seed(self) -> int:
        return derive_seed64(
            INTERPRETER_PROTOCOL_ID,
            self.program.program_hash,
            self.episode_id,
            self.policy_seed,
            self.context.step_index,
        )

    def feature_value(self, field_name: str) -> RuntimeValue:
        adjacency = self.overlay.adjacency()
        degrees = tuple(len(neighbors) for neighbors in adjacency)
        values: dict[str, RuntimeValue] = {
            "order": self.overlay.order,
            "edge_count": len(self.overlay.edges),
            "minimum_degree": min(degrees, default=0),
            "maximum_degree": max(degrees, default=0),
        }
        try:
            return values[field_name]
        except KeyError as exc:
            raise _ProgramRuntimeError(
                "UNKNOWN_FEATURE_FIELD", "/", f"unsupported feature field: {field_name}"
            ) from exc

    def charge_selector(self, selector_id: str, cache_hit: bool, path: str) -> None:
        self.selector_calls += 1
        self.selector_cost += 1 if cache_hit else SELECTOR_COSTS[selector_id]
        if self.selector_calls > self.limits.maximum_selector_calls:
            raise _ProgramRuntimeError("SELECTOR_CALL_LIMIT", path, "selector call limit exceeded")
        if self.selector_cost > self.limits.maximum_selector_cost_units:
            raise _ProgramRuntimeError("SELECTOR_COST_LIMIT", path, "selector cost limit exceeded")

    def charge_action(self, path: str) -> None:
        self.gross_actions += 1
        if self.gross_actions > self.limits.maximum_gross_actions:
            raise _ProgramRuntimeError("ACTION_LIMIT", path, "gross action limit exceeded")

    def witness_loads(
        self,
    ) -> tuple[
        Mapping[tuple[int, int], int],
        Mapping[tuple[int, Edge], int],
    ]:
        overlay_key = tuple(sorted(self.overlay.edges))
        cached = self.witness_load_cache.get(overlay_key)
        if cached is not None:
            return cached
        vertex_loads = dict(self.features.vertex_witness_load)
        edge_loads = dict(self.features.edge_witness_load)
        if self.features.witness_load_provider is not None:
            sampled_vertices, sampled_edges = self.features.witness_load_provider(
                self.overlay.graph()
            )
            vertex_loads.update(sampled_vertices)
            edge_loads.update(sampled_edges)
        loads = vertex_loads, edge_loads
        self.witness_load_cache[overlay_key] = loads
        return loads


def _reference_key(value: ReferenceValue) -> tuple[object, ...]:
    if isinstance(value, VertexRef):
        return ("v", value.vertex)
    if isinstance(value, EdgeRef):
        return ("e", *value.edge)
    if isinstance(value, NonEdgeRef):
        return ("n", *value.edge)
    if isinstance(value, Path2Ref):
        return ("p", value.u, value.w, value.v)
    return ("m", value.removed_edges, value.added_edges)


def _query_cache_key(
    selector_id: str,
    arguments: Mapping[str, RuntimeValue],
    edges: set[Edge],
) -> tuple[str, str]:
    pieces: list[str] = []
    for name in sorted(arguments):
        value = arguments[name]
        if isinstance(value, (VertexRef, EdgeRef, NonEdgeRef, Path2Ref, MatchingRef)):
            rendered = repr(_reference_key(value))
        else:
            rendered = repr(value)
        pieces.append(f"{name}={rendered}")
    pieces.append(f"overlay={tuple(sorted(edges))!r}")
    return selector_id, "|".join(pieces)


def _extreme(
    values: Mapping[ReferenceValue, int],
    mode: str,
) -> tuple[ReferenceValue, ...]:
    if not values:
        return ()
    target = min(values.values()) if mode == "min" else max(values.values())
    return tuple(item for item, value in values.items() if value == target)


def _tarjan(overlay: _Overlay) -> tuple[frozenset[int], frozenset[Edge]]:
    adjacency = overlay.adjacency()
    discovery = [-1] * overlay.order
    low = [0] * overlay.order
    parent = [-1] * overlay.order
    articulations: set[int] = set()
    bridges: set[Edge] = set()
    time = 0

    def visit(vertex: int) -> None:
        nonlocal time
        discovery[vertex] = low[vertex] = time
        time += 1
        children = 0
        for neighbor in adjacency[vertex]:
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


def _distances(overlay: _Overlay, source: int) -> tuple[int, ...]:
    adjacency = overlay.adjacency()
    distances = [-1] * overlay.order
    distances[source] = 0
    pending: deque[int] = deque([source])
    while pending:
        vertex = pending.popleft()
        for neighbor in adjacency[vertex]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[vertex] + 1
                pending.append(neighbor)
    return tuple(distances)


def _matching_candidates(runtime: _Runtime, k: int, path: str) -> tuple[ReferenceValue, ...]:
    if k not in {2, 3, 4}:
        raise _BranchFailure(
            BranchFailureCode.LOCAL_PRECONDITION_FAILED,
            "k-switch requires k in {2,3,4}",
        )
    edges = tuple(sorted(runtime.overlay.edges))
    if len(edges) < k:
        return ()
    candidates: dict[tuple[tuple[Edge, ...], tuple[Edge, ...]], MatchingRef] = {}
    ordinal = runtime.draw_ordinal
    for _ in range(64):
        available = list(edges)
        chosen: list[Edge] = []
        for _index in range(k):
            selected, ordinal = uniform_below(runtime.random_seed, ordinal, len(available))
            chosen.append(available.pop(selected))
        endpoints = [vertex for edge in chosen for vertex in edge]
        if len(set(endpoints)) != 2 * k:
            continue
        shuffled = list(endpoints)
        for index in range(len(shuffled) - 1, 0, -1):
            selected, ordinal = uniform_below(runtime.random_seed, ordinal, index + 1)
            shuffled[index], shuffled[selected] = shuffled[selected], shuffled[index]
        added = tuple(
            sorted(
                normalized_edge((shuffled[index], shuffled[index + 1]))
                for index in range(0, len(shuffled), 2)
            )
        )
        removed = tuple(sorted(chosen))
        if (
            len(set(added)) != k
            or any(u == v for u, v in added)
            or set(added) == set(removed)
            or any(edge in runtime.overlay.edges and edge not in removed for edge in added)
        ):
            continue
        candidate = MatchingRef(removed, added)
        candidates[(removed, added)] = candidate
    runtime.draw_ordinal = ordinal
    if not candidates:
        return ()
    return tuple(candidates[key] for key in sorted(candidates))


def _selector(
    runtime: _Runtime,
    selector_id: str,
    arguments: Mapping[str, RuntimeValue],
    path: str,
) -> SelectionPopulation:
    key = _query_cache_key(selector_id, arguments, runtime.overlay.edges)
    cached = runtime.query_cache.get(key)
    runtime.charge_selector(selector_id, cached is not None, path)
    if cached is not None:
        return cached
    adjacency = runtime.overlay.adjacency()
    degrees = tuple(len(neighbors) for neighbors in adjacency)
    items: tuple[ReferenceValue, ...]
    mode = str(arguments.get("mode", "max"))
    if mode not in {"min", "max"}:
        raise _BranchFailure(BranchFailureCode.LOCAL_PRECONDITION_FAILED, "mode must be min or max")
    if selector_id == "vertices_degree_extreme":
        degree_values: dict[ReferenceValue, int] = {
            VertexRef(vertex): degrees[vertex] for vertex in range(runtime.overlay.order)
        }
        items = _extreme(degree_values, mode)
    elif selector_id == "vertices_degree_class":
        degree = int(cast(int, arguments.get("degree")))
        items = tuple(
            VertexRef(vertex)
            for vertex in range(runtime.overlay.order)
            if degrees[vertex] == degree
        )
    elif selector_id == "vertices_witness_load_extreme":
        length = int(cast(int, arguments.get("length")))
        vertex_witness_load, _edge_witness_load = runtime.witness_loads()
        vertex_load_values: dict[ReferenceValue, int] = {
            VertexRef(vertex): vertex_witness_load.get((length, vertex), 0)
            for vertex in range(runtime.overlay.order)
        }
        items = _extreme(vertex_load_values, mode)
    elif selector_id == "edges_witness_load_extreme":
        length = int(cast(int, arguments.get("length")))
        _vertex_witness_load, edge_witness_load = runtime.witness_loads()
        edge_load_values: dict[ReferenceValue, int] = {
            EdgeRef(edge): edge_witness_load.get((length, edge), 0)
            for edge in runtime.overlay.edges
        }
        items = _extreme(edge_load_values, mode)
    elif selector_id == "vertices_articulation_risk":
        articulations, _ = _tarjan(runtime.overlay)
        articulation_values: dict[ReferenceValue, int] = {
            VertexRef(vertex): int(vertex in articulations)
            for vertex in range(runtime.overlay.order)
        }
        items = _extreme(articulation_values, mode)
    elif selector_id == "edges_bridge_risk":
        _, bridges = _tarjan(runtime.overlay)
        bridge_values: dict[ReferenceValue, int] = {
            EdgeRef(edge): int(edge in bridges) for edge in runtime.overlay.edges
        }
        items = _extreme(bridge_values, mode)
    elif selector_id == "vertices_distance_band":
        source = arguments.get("source")
        if not isinstance(source, VertexRef):
            raise _ProgramRuntimeError("SELECTOR_ARGUMENT", path, "source must be VertexRef")
        minimum = int(cast(int, arguments.get("minimum")))
        maximum = int(cast(int, arguments.get("maximum")))
        distances = _distances(runtime.overlay, source.vertex)
        items = tuple(
            VertexRef(vertex)
            for vertex, distance in enumerate(distances)
            if minimum <= distance <= maximum
        )
    elif selector_id == "non_edges_from_vertex":
        source = arguments.get("vertex")
        if not isinstance(source, VertexRef):
            raise _ProgramRuntimeError("SELECTOR_ARGUMENT", path, "vertex must be VertexRef")
        items = tuple(
            NonEdgeRef(normalized_edge((source.vertex, vertex)))
            for vertex in range(runtime.overlay.order)
            if vertex != source.vertex and vertex not in adjacency[source.vertex]
        )
    elif selector_id == "non_edges_local_cycle_risk":
        risk_values: dict[ReferenceValue, int] = {}
        for u in range(runtime.overlay.order):
            for v in range(u + 1, runtime.overlay.order):
                edge = (u, v)
                if edge not in runtime.overlay.edges:
                    risk_values[NonEdgeRef(edge)] = len(adjacency[u] & adjacency[v])
        items = _extreme(risk_values, mode)
    elif selector_id == "paths_length_two":
        paths: list[ReferenceValue] = []
        for center in range(runtime.overlay.order):
            for u, v in combinations(sorted(adjacency[center]), 2):
                paths.append(Path2Ref(u, center, v))
        items = tuple(paths)
    elif selector_id == "matching_k_switch_reconnections":
        k = int(cast(int, arguments.get("k")))
        items = _matching_candidates(runtime, k, path)
    else:
        raise _ProgramRuntimeError("UNKNOWN_SELECTOR", path, f"unknown selector: {selector_id}")
    population_size = len(items)
    if population_size > 64:
        reservoir = list(items[:64])
        ordinal = runtime.draw_ordinal
        for index in range(64, population_size):
            selected, ordinal = uniform_below(
                runtime.random_seed,
                ordinal,
                index + 1,
            )
            if selected < 64:
                reservoir[selected] = items[index]
        runtime.draw_ordinal = ordinal
        items = tuple(reservoir)
    population = SelectionPopulation(selector_id, tuple(items), population_size)
    runtime.query_cache[key] = population
    return population


def _weight(runtime: _Runtime, value: ReferenceValue, feature: str) -> int:
    adjacency = runtime.overlay.adjacency()
    if feature == "degree" and isinstance(value, VertexRef):
        return max(1, len(adjacency[value.vertex]))
    if feature == "inverse_degree" and isinstance(value, VertexRef):
        return max(1, runtime.overlay.order - len(adjacency[value.vertex]))
    if feature == "uniform":
        return 1
    raise _ProgramRuntimeError("WEIGHT_FEATURE", "/", f"unsupported weight feature: {feature}")


def _number(value: RuntimeValue, path: str) -> int | Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise _ProgramRuntimeError("EXPRESSION_TYPE", path, "numeric value required")
    return value


def _evaluate_expression(
    value: object,
    *,
    runtime: _Runtime,
    environment: Mapping[str, RuntimeValue],
    path: str,
) -> RuntimeValue:
    if isinstance(value, bool | int | str):
        return value
    expression = cast(dict[str, object], value)
    operation = expression["op"]
    if operation == "ref":
        name = cast(str, expression["name"])
        try:
            return environment[name]
        except KeyError as exc:
            raise _ProgramRuntimeError("UNBOUND_REFERENCE", path, name) from exc
    if operation == "ctx":
        return runtime.context.value(cast(str, expression["field"]))
    if operation == "feature":
        return runtime.feature_value(cast(str, expression["field"]))
    if operation == "rational":
        return Fraction(
            cast(int, expression["numerator"]),
            cast(int, expression["denominator"]),
        )
    if operation == "selector":
        arguments = {
            name: _evaluate_expression(
                item,
                runtime=runtime,
                environment=environment,
                path=f"{path}/arguments/{name}",
            )
            for name, item in cast(dict[str, object], expression["arguments"]).items()
        }
        return _selector(
            runtime,
            cast(str, expression["selector_id"]),
            arguments,
            path,
        )
    if operation == "pick":
        source = _evaluate_expression(
            expression["source"],
            runtime=runtime,
            environment=environment,
            path=f"{path}/source",
        )
        if not isinstance(source, SelectionPopulation):
            raise _ProgramRuntimeError("PICK_TYPE", path, "pick source is not a population")
        if not source.items:
            raise _BranchFailure(BranchFailureCode.NO_MATCH, "selector population is empty")
        mode = expression["mode"]
        selected_index: int
        if mode == "require_singleton":
            if len(source.items) != 1:
                raise _BranchFailure(
                    BranchFailureCode.NO_MATCH, "selector result is not a singleton"
                )
            selected_index = 0
        elif mode == "seeded_uniform":
            selected_index, runtime.draw_ordinal = uniform_below(
                runtime.random_seed,
                runtime.draw_ordinal,
                len(source.items),
            )
        else:
            feature = cast(str, expression["weight_feature"])
            selected_index, runtime.draw_ordinal = weighted_index(
                runtime.random_seed,
                runtime.draw_ordinal,
                (_weight(runtime, item, feature) for item in source.items),
            )
        runtime.selections.append(
            SelectionRecord(
                source.selector_id,
                source.population_size,
                len(source.items),
                selected_index,
                path,
                runtime.random_seed,
                hashlib.sha256(
                    repr(_reference_key(source.items[selected_index])).encode("ascii")
                ).hexdigest(),
            )
        )
        return source.items[selected_index]
    if operation in {
        "add",
        "subtract",
        "multiply",
        "minimum",
        "maximum",
        "equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "and",
        "or",
    }:
        left = _evaluate_expression(
            expression["left"],
            runtime=runtime,
            environment=environment,
            path=f"{path}/left",
        )
        right = _evaluate_expression(
            expression["right"],
            runtime=runtime,
            environment=environment,
            path=f"{path}/right",
        )
        if operation == "add":
            return _number(left, path) + _number(right, path)
        if operation == "subtract":
            return _number(left, path) - _number(right, path)
        if operation == "multiply":
            return _number(left, path) * _number(right, path)
        if operation == "minimum":
            return min(_number(left, path), _number(right, path))
        if operation == "maximum":
            return max(_number(left, path), _number(right, path))
        if operation == "equal":
            return left == right
        if operation == "less":
            return _number(left, path) < _number(right, path)
        if operation == "less_equal":
            return _number(left, path) <= _number(right, path)
        if operation == "greater":
            return _number(left, path) > _number(right, path)
        if operation == "greater_equal":
            return _number(left, path) >= _number(right, path)
        if operation == "and":
            return bool(left) and bool(right)
        return bool(left) or bool(right)
    if operation == "not":
        return not bool(
            _evaluate_expression(
                expression["value"],
                runtime=runtime,
                environment=environment,
                path=f"{path}/value",
            )
        )
    if operation == "exists":
        try:
            result = _evaluate_expression(
                expression["value"],
                runtime=runtime,
                environment=environment,
                path=f"{path}/value",
            )
        except _BranchFailure:
            return False
        return not isinstance(result, SelectionPopulation) or bool(result.items)
    raise _ProgramRuntimeError("UNKNOWN_EXPRESSION", path, str(operation))


def _vertex(value: RuntimeValue, path: str) -> int:
    if not isinstance(value, VertexRef):
        raise _ProgramRuntimeError("REFERENCE_TYPE", path, "VertexRef required")
    return value.vertex


def _edge(value: RuntimeValue, path: str) -> Edge:
    if not isinstance(value, EdgeRef):
        raise _ProgramRuntimeError("REFERENCE_TYPE", path, "EdgeRef required")
    return value.edge


def _apply_action(
    runtime: _Runtime,
    action_id: str,
    arguments: Mapping[str, RuntimeValue],
    path: str,
) -> None:
    runtime.charge_action(path)
    edges = runtime.overlay.edges
    if action_id == "add_edge":
        reference = arguments["edge"]
        if not isinstance(reference, NonEdgeRef):
            raise _ProgramRuntimeError("REFERENCE_TYPE", path, "NonEdgeRef required")
        edge = reference.edge
        if edge[0] == edge[1] or edge in edges:
            raise _BranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED, "edge cannot be added"
            )
        edges.add(edge)
        return
    if action_id == "remove_edge":
        edge = _edge(arguments["edge"], path)
        if edge not in edges:
            raise _BranchFailure(BranchFailureCode.LOCAL_PRECONDITION_FAILED, "edge is absent")
        edges.remove(edge)
        return
    if action_id == "relocate_endpoint":
        edge = _edge(arguments["edge"], path)
        keep = _vertex(arguments["keep"], path)
        new = _vertex(arguments["new"], path)
        if edge not in edges or keep not in edge or new in edge:
            raise _BranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED, "invalid relocation references"
            )
        replacement = normalized_edge((keep, new))
        if replacement in edges:
            raise _BranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED, "replacement edge exists"
            )
        edges.remove(edge)
        edges.add(replacement)
        return
    if action_id == "k_switch":
        matching = arguments["matching"]
        if not isinstance(matching, MatchingRef):
            raise _ProgramRuntimeError("REFERENCE_TYPE", path, "MatchingRef required")
        if not all(edge in edges for edge in matching.removed_edges):
            raise _BranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED, "matching source edge is absent"
            )
        remaining = edges.difference(matching.removed_edges)
        if any(edge in remaining for edge in matching.added_edges):
            raise _BranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED, "matching target edge exists"
            )
        edges.difference_update(matching.removed_edges)
        edges.update(matching.added_edges)
        return
    if action_id == "edge_fanout":
        edge = _edge(arguments["edge"], path)
        w = _vertex(arguments["w"], path)
        u, v = edge
        additions = {normalized_edge((u, w)), normalized_edge((v, w))}
        if edge not in edges or w in edge or any(addition in edges for addition in additions):
            raise _BranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED, "fanout precondition failed"
            )
        edges.remove(edge)
        edges.update(additions)
        return
    if action_id == "edge_fold":
        path_ref = arguments["path"]
        if not isinstance(path_ref, Path2Ref):
            raise _ProgramRuntimeError("REFERENCE_TYPE", path, "Path2Ref required")
        first = normalized_edge((path_ref.u, path_ref.w))
        second = normalized_edge((path_ref.w, path_ref.v))
        replacement = normalized_edge((path_ref.u, path_ref.v))
        if first not in edges or second not in edges or replacement in edges:
            raise _BranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED, "fold precondition failed"
            )
        edges.remove(first)
        edges.remove(second)
        edges.add(replacement)
        return
    raise _ProgramRuntimeError("UNKNOWN_ACTION", path, action_id)


def _connected(overlay: _Overlay) -> bool:
    if overlay.order == 0:
        return False
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


def _emit(runtime: _Runtime) -> RewritePlan:
    adjacency = runtime.overlay.adjacency()
    if not _connected(runtime.overlay) or min(map(len, adjacency), default=0) < 3:
        raise _BranchFailure(
            BranchFailureCode.ILLEGAL_FINAL_STATE,
            "final graph must be connected with minimum degree at least three",
        )
    removed = tuple(sorted(runtime.overlay.initial_edges.difference(runtime.overlay.edges)))
    added = tuple(sorted(runtime.overlay.edges.difference(runtime.overlay.initial_edges)))
    if not removed and not added:
        raise _BranchFailure(BranchFailureCode.NO_EFFECT, "final rewrite has no effect")
    if (
        len(removed) > runtime.limits.maximum_net_removed_edges
        or len(added) > runtime.limits.maximum_net_added_edges
    ):
        raise _ProgramRuntimeError("NET_EDGE_LIMIT", "/emit", "net rewrite exceeds edge limits")
    return RewritePlan(
        removed_edges=removed,
        added_edges=added,
        operator_family="native_v3_program",
        metadata={
            "program_hash": runtime.program.program_hash,
            "interpreter_protocol_id": INTERPRETER_PROTOCOL_ID,
            "gross_actions": runtime.gross_actions,
            "selector_cost_units": runtime.selector_cost,
        },
    )


def _execute_node(
    node: Mapping[str, object],
    *,
    runtime: _Runtime,
    environment: dict[str, RuntimeValue],
    path: str,
    repeat_indices: tuple[int, ...] = (),
) -> None:
    operation = node["op"]
    if operation == "block":
        for index, child in enumerate(cast(list[dict[str, object]], node["children"])):
            _execute_node(
                child,
                runtime=runtime,
                environment=environment,
                path=f"{path}/children/{index}",
                repeat_indices=repeat_indices,
            )
        return
    if operation == "let":
        value = _evaluate_expression(
            node["value"],
            runtime=runtime,
            environment=environment,
            path=f"{path}/value",
        )
        nested = environment.copy()
        nested[cast(str, node["name"])] = value
        _execute_node(
            cast(dict[str, object], node["body"]),
            runtime=runtime,
            environment=nested,
            path=f"{path}/body",
            repeat_indices=repeat_indices,
        )
        return
    if operation == "if":
        condition = _evaluate_expression(
            node["condition"],
            runtime=runtime,
            environment=environment,
            path=f"{path}/condition",
        )
        branch_name = "then" if bool(condition) else "else"
        _execute_node(
            cast(dict[str, object], node[branch_name]),
            runtime=runtime,
            environment=environment.copy(),
            path=f"{path}/{branch_name}",
            repeat_indices=repeat_indices,
        )
        return
    if operation == "try":
        entry_edges = runtime.overlay.clone_edges()
        for index, branch in enumerate(cast(list[dict[str, object]], node["branches"])):
            runtime.overlay.edges = set(entry_edges)
            try:
                _execute_node(
                    branch,
                    runtime=runtime,
                    environment=environment.copy(),
                    path=f"{path}/branches/{index}",
                    repeat_indices=repeat_indices,
                )
                return
            except _BranchFailure:
                continue
        runtime.overlay.edges = entry_edges
        raise _BranchFailure(BranchFailureCode.NO_MATCH, "all try branches failed")
    if operation == "repeat":
        for index in range(cast(int, node["count"])):
            _execute_node(
                cast(dict[str, object], node["body"]),
                runtime=runtime,
                environment=environment.copy(),
                path=f"{path}/body",
                repeat_indices=(*repeat_indices, index),
            )
        return
    if operation == "choose":
        branches = cast(list[dict[str, object]], node["branches"])
        selected, runtime.draw_ordinal = weighted_index(
            runtime.random_seed,
            runtime.draw_ordinal,
            (cast(int, branch["weight"]) for branch in branches),
        )
        _execute_node(
            cast(dict[str, object], branches[selected]["body"]),
            runtime=runtime,
            environment=environment.copy(),
            path=f"{path}/branches/{selected}/body",
            repeat_indices=repeat_indices,
        )
        return
    if operation == "apply":
        arguments = {
            name: _evaluate_expression(
                value,
                runtime=runtime,
                environment=environment,
                path=f"{path}/arguments/{name}",
            )
            for name, value in cast(dict[str, object], node["arguments"]).items()
        }
        _apply_action(runtime, cast(str, node["action_id"]), arguments, path)
        return
    if operation == "emit":
        raise _TerminalResult(_emit(runtime))
    if operation == "no_plan":
        raise _TerminalResult(NoPlan(NoPlanReason(cast(str, node["reason"]))))
    raise _ProgramRuntimeError("UNKNOWN_NODE", path, str(operation))


def invoke_program(
    program: ValidatedProgram,
    graph: GraphState,
    *,
    context: ProgramContext,
    features: GraphFeatureInput | None = None,
    policy_seed: int,
    episode_id: str,
    limits: ProgramLimits | None = None,
) -> InvocationResult:
    effective_limits = limits or ProgramLimits()
    runtime = _Runtime(
        program=program,
        graph=graph,
        context=context,
        features=features or GraphFeatureInput(),
        policy_seed=policy_seed,
        episode_id=episode_id,
        limits=effective_limits,
        overlay=_Overlay.from_graph(graph),
    )
    try:
        _execute_node(
            cast(dict[str, object], program.ast["entry"]),
            runtime=runtime,
            environment={},
            path="/entry",
        )
        failure = ProgramFailure(
            "UNTERMINATED_PATH", "/entry", "program returned without emit or NoPlan"
        )
        return InvocationResult(
            None,
            None,
            failure,
            runtime.selector_calls,
            runtime.selector_cost,
            runtime.gross_actions,
            tuple(runtime.selections),
        )
    except _TerminalResult as terminal:
        rewrite = terminal.result if isinstance(terminal.result, RewritePlan) else None
        no_plan = terminal.result if isinstance(terminal.result, NoPlan) else None
        return InvocationResult(
            rewrite,
            no_plan,
            None,
            runtime.selector_calls,
            runtime.selector_cost,
            runtime.gross_actions,
            tuple(runtime.selections),
        )
    except _BranchFailure as failure:
        reason = {
            BranchFailureCode.NO_MATCH: NoPlanReason.NO_MATCH,
            BranchFailureCode.LOCAL_PRECONDITION_FAILED: NoPlanReason.NO_MATCH,
            BranchFailureCode.ILLEGAL_FINAL_STATE: NoPlanReason.ILLEGAL_FINAL_STATE,
            BranchFailureCode.NO_EFFECT: NoPlanReason.NO_EFFECT,
        }[failure.code]
        return InvocationResult(
            None,
            NoPlan(reason),
            None,
            runtime.selector_calls,
            runtime.selector_cost,
            runtime.gross_actions,
            tuple(runtime.selections),
        )
    except _ProgramRuntimeError as failure:
        return InvocationResult(
            None,
            None,
            ProgramFailure(failure.code, failure.path, str(failure)),
            runtime.selector_calls,
            runtime.selector_cost,
            runtime.gross_actions,
            tuple(runtime.selections),
        )
