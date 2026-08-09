"""Host-owned label-opaque graph capability for ordinary-Python policies."""

from __future__ import annotations

import hashlib
import secrets
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import cast

from mutation_forge.models import Edge, GraphState, JsonValue, RewritePlan, normalized_edge
from mutation_forge.native_v3.randomness import derive_seed64, weighted_index

from .contracts import GraphViewV1, NoPlan, PolicyContextV1
from .runtime_contracts import (
    RANDOM_PROTOCOL_ID,
    SAFE_GRAPH_API_PROTOCOL_ID,
    GraphFeatureInputV1,
    IllegalRewriteError,
    PolicyRuntimeLimitsV1,
    RewriteHostV1,
    SemanticAPIEventV1,
)
from .validation import ACTION_METHODS, SELECTOR_METHODS


class SafeAPIProgramError(RuntimeError):
    """A deterministic candidate-caused capability failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SafeAPIInfrastructureError(RuntimeError):
    """A trusted host dependency violated the safe-API contract."""


@dataclass(frozen=True, slots=True)
class _Reference:
    kind: str
    payload: object


@dataclass(frozen=True, slots=True)
class _MintedReference:
    token: str
    ordinal: int
    reference: _Reference


@dataclass(slots=True)
class _Overlay:
    order: int
    initial_edges: frozenset[Edge]
    edges: set[Edge]

    @classmethod
    def from_graph(cls, graph: GraphState) -> _Overlay:
        if graph.order < 1:
            raise ValueError("runtime graph order must be positive")
        normalized = tuple(normalized_edge(edge) for edge in graph.edges)
        if len(normalized) != len(set(normalized)):
            raise ValueError("runtime graph contains duplicate edges")
        if any(
            u == v or not (0 <= u < graph.order and 0 <= v < graph.order)
            for u, v in normalized
        ):
            raise ValueError("runtime graph contains an invalid edge")
        return cls(graph.order, frozenset(normalized), set(normalized))

    def graph(self) -> GraphState:
        return GraphState(self.order, tuple(sorted(self.edges)))

    def adjacency(self) -> tuple[set[int], ...]:
        result = tuple(set[int]() for _ in range(self.order))
        for u, v in self.edges:
            result[u].add(v)
            result[v].add(u)
        return result


def graph_view_v1(graph: GraphState) -> GraphViewV1:
    """Project a graph to the complete accepted label-opaque scalar view."""

    overlay = _Overlay.from_graph(graph)
    degrees = tuple(len(neighbors) for neighbors in overlay.adjacency())
    return GraphViewV1(
        order=graph.order,
        edge_count=len(graph.edges),
        minimum_degree=min(degrees),
        maximum_degree=max(degrees),
    )


def _connected(overlay: _Overlay) -> bool:
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


def _tarjan(overlay: _Overlay) -> tuple[frozenset[int], frozenset[Edge]]:
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


def _distances(overlay: _Overlay, source: int) -> tuple[int, ...]:
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


def _reference_key(reference: _Reference) -> tuple[str, str]:
    return reference.kind, repr(reference.payload)


def _extreme(
    values: Mapping[_Reference, int],
    mode: str,
) -> tuple[_Reference, ...]:
    if not values:
        return ()
    target = min(values.values()) if mode == "min" else max(values.values())
    return tuple(
        sorted(
            (reference for reference, value in values.items() if value == target),
            key=_reference_key,
        )
    )


@dataclass(slots=True)
class SafeGraphSessionV1:
    """One invocation-scoped, host-owned graph overlay and capability registry."""

    graph: GraphState
    context: PolicyContextV1
    seed: int
    program_hash: str
    rewrite_host: RewriteHostV1
    limits: PolicyRuntimeLimitsV1
    features: GraphFeatureInputV1 = field(default_factory=GraphFeatureInputV1)
    overlay: _Overlay = field(init=False)
    _references: dict[str, _MintedReference] = field(default_factory=dict, init=False)
    _tokens_by_reference: dict[_Reference, str] = field(default_factory=dict, init=False)
    _results: dict[str, RewritePlan | NoPlan] = field(default_factory=dict, init=False)
    _events: list[SemanticAPIEventV1] = field(default_factory=list, init=False)
    _secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), init=False)
    _api_calls: int = field(default=0, init=False)
    _selector_calls: int = field(default=0, init=False)
    _action_calls: int = field(default=0, init=False)
    _random_draws: int = field(default=0, init=False)
    _gross_actions: int = field(default=0, init=False)
    _terminal_token: str | None = field(default=None, init=False)
    _witness_cache: dict[
        tuple[Edge, ...],
        tuple[Mapping[tuple[int, int], int], Mapping[tuple[int, Edge], int]],
    ] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.overlay = _Overlay.from_graph(self.graph)
        if self.graph.order > self.limits.graph_order:
            raise SafeAPIInfrastructureError(
                f"graph order {self.graph.order} exceeds runtime cap "
                f"{self.limits.graph_order}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not 0 <= self.seed < 1 << 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if len(self.program_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.program_hash
        ):
            raise ValueError("program_hash must be a lowercase SHA-256 digest")

    @property
    def semantic_trace(self) -> tuple[SemanticAPIEventV1, ...]:
        return tuple(self._events)

    @property
    def counters(self) -> dict[str, int]:
        return {
            "api_calls": self._api_calls,
            "selector_calls": self._selector_calls,
            "action_calls": self._action_calls,
            "random_draws": self._random_draws,
            "gross_actions": self._gross_actions,
        }

    def _consume(self, counter_name: str, maximum: int, code: str) -> None:
        value = getattr(self, counter_name) + 1
        setattr(self, counter_name, value)
        if value > maximum:
            raise SafeAPIProgramError(code, f"{counter_name[1:]} budget exceeded")

    def _consume_draws(self, draws: int) -> None:
        self._random_draws += draws
        if self._random_draws > self.limits.random_draws:
            raise SafeAPIProgramError(
                "RANDOM_DRAW_BUDGET_EXCEEDED",
                "deterministic random-draw budget exceeded",
            )

    def _random_index(self, path: str, weights: Sequence[int]) -> int:
        random_seed = derive_seed64(
            RANDOM_PROTOCOL_ID,
            self.seed,
            self.context.invocation_ordinal,
            path,
        )
        index, draws = weighted_index(random_seed, weights)
        self._consume_draws(draws)
        return index

    def _mint_reference(self, reference: _Reference) -> dict[str, JsonValue]:
        existing = self._tokens_by_reference.get(reference)
        if existing is not None:
            minted = self._references[existing]
            return {"$ref": minted.token, "kind": reference.kind}
        ordinal = len(self._references)
        token = hashlib.sha256(
            self._secret + ordinal.to_bytes(8, "big") + reference.kind.encode("ascii")
        ).hexdigest()[:32]
        minted = _MintedReference(token, ordinal, reference)
        self._references[token] = minted
        self._tokens_by_reference[reference] = token
        return {"$ref": token, "kind": reference.kind}

    def _mint_result(self, result: RewritePlan | NoPlan, kind: str) -> dict[str, JsonValue]:
        if self._terminal_token is not None:
            raise SafeAPIProgramError("MULTIPLE_TERMINAL_RESULTS", "terminal result already minted")
        token = hashlib.sha256(
            self._secret + b"result" + len(self._results).to_bytes(8, "big")
        ).hexdigest()
        self._results[token] = result
        self._terminal_token = token
        return {"$host_result": token, "kind": kind}

    def resolve_result(self, token: str, kind: str) -> RewritePlan | NoPlan:
        result = self._results.get(token)
        if token != self._terminal_token or result is None:
            raise SafeAPIProgramError(
                "STALE_OR_FOREIGN_RESULT",
                "returned result was not minted for this invocation",
            )
        if kind == "rewrite_plan" and isinstance(result, RewritePlan):
            return result
        if kind == "no_plan" and isinstance(result, NoPlan):
            return result
        raise SafeAPIProgramError("INVALID_RETURN", "returned result kind does not match token")

    def _decode_reference(self, value: object, expected_kind: str | None = None) -> _Reference:
        if not isinstance(value, dict) or set(value) != {"$ref", "kind"}:
            raise SafeAPIProgramError("INVALID_REFERENCE", "expected an opaque reference")
        token = value.get("$ref")
        kind = value.get("kind")
        if not isinstance(token, str) or not isinstance(kind, str):
            raise SafeAPIProgramError("INVALID_REFERENCE", "malformed opaque reference")
        minted = self._references.get(token)
        if minted is None or minted.reference.kind != kind:
            raise SafeAPIProgramError(
                "STALE_OR_FOREIGN_REFERENCE",
                "reference does not belong to the current invocation",
            )
        if expected_kind is not None and kind != expected_kind:
            raise SafeAPIProgramError(
                "INVALID_REFERENCE_TYPE",
                f"expected {expected_kind}, received {kind}",
            )
        return minted.reference

    def _trace_value(self, value: object) -> JsonValue:
        if isinstance(value, dict) and set(value) == {"$ref", "kind"}:
            token = value.get("$ref")
            kind = value.get("kind")
            if isinstance(token, str) and isinstance(kind, str) and token in self._references:
                return {
                    "kind": kind,
                    "ordinal": self._references[token].ordinal,
                }
        if isinstance(value, dict) and set(value) == {"$host_result", "kind"}:
            return {"kind": cast(str, value["kind"])}
        if isinstance(value, dict):
            return {
                str(key): self._trace_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [self._trace_value(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise SafeAPIInfrastructureError(f"cannot trace value of type {type(value).__name__}")

    def _record(self, method: str, arguments: Mapping[str, object], result: object) -> None:
        self._events.append(
            SemanticAPIEventV1(
                ordinal=len(self._events),
                method=method,
                arguments={
                    key: self._trace_value(value)
                    for key, value in sorted(arguments.items())
                },
                result=self._trace_value(result),
            )
        )

    @staticmethod
    def _require_keys(
        arguments: Mapping[str, object],
        *,
        required: frozenset[str] = frozenset(),
        optional: frozenset[str] = frozenset(),
    ) -> None:
        keys = frozenset(arguments)
        if not required.issubset(keys) or not keys.issubset(required | optional):
            raise SafeAPIProgramError(
                "INVALID_API_ARGUMENTS",
                f"required keys are {sorted(required)}; optional keys are {sorted(optional)}",
            )

    @staticmethod
    def _mode(arguments: Mapping[str, object]) -> str:
        mode = arguments.get("mode", "max")
        if mode not in {"min", "max"}:
            raise SafeAPIProgramError("INVALID_API_ARGUMENT", "mode must be min or max")
        return mode

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
            try:
                sampled_vertices, sampled_edges = self.features.witness_load_provider(
                    self.overlay.graph()
                )
            except Exception as error:
                raise SafeAPIInfrastructureError("witness-load provider failed") from error
            vertices.update(sampled_vertices)
            edges.update(sampled_edges)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*vertices.values(), *edges.values())
        ):
            raise SafeAPIInfrastructureError(
                "witness-load provider returned a non-negative-integer violation"
            )
        result = vertices, edges
        self._witness_cache[key] = result
        return result

    def _bounded(
        self,
        references: tuple[_Reference, ...],
        *,
        kind: str,
        path: str,
    ) -> tuple[_Reference, ...]:
        maximum = self.limits.selector_result_size
        population_size = len(references)
        if population_size <= maximum:
            return references
        if kind in {"relocation", "fanout"}:
            start = self._random_index(path + "/bounded-window", (1,) * population_size)
            return tuple(
                sorted(
                    (
                        references[(start + offset) % population_size]
                        for offset in range(maximum)
                    ),
                    key=_reference_key,
                )
            )
        reservoir = list(references[:maximum])
        for index in range(maximum, population_size):
            selected = self._random_index(
                f"{path}/reservoir/{index}",
                (1,) * (index + 1),
            )
            if selected < maximum:
                reservoir[selected] = references[index]
        return tuple(sorted(reservoir, key=_reference_key))

    def _matching_candidates(self, k: int, *, path: str) -> tuple[_Reference, ...]:
        if k not in {2, 3, 4}:
            raise SafeAPIProgramError("INVALID_API_ARGUMENT", "k must be one of 2, 3, or 4")
        edges = tuple(sorted(self.overlay.edges))
        if len(edges) < k:
            return ()
        candidates: set[tuple[tuple[Edge, ...], tuple[Edge, ...]]] = set()
        for attempt in range(64):
            available = list(edges)
            removed: list[Edge] = []
            for index in range(k):
                selected = self._random_index(
                    f"{path}/attempt/{attempt}/edge/{index}",
                    (1,) * len(available),
                )
                removed.append(available.pop(selected))
            endpoints = [vertex for edge in removed for vertex in edge]
            if len(set(endpoints)) != 2 * k:
                continue
            shuffled = list(endpoints)
            for index in range(len(shuffled) - 1, 0, -1):
                selected = self._random_index(
                    f"{path}/attempt/{attempt}/shuffle/{index}",
                    (1,) * (index + 1),
                )
                shuffled[index], shuffled[selected] = shuffled[selected], shuffled[index]
            added = tuple(
                sorted(
                    normalized_edge((shuffled[index], shuffled[index + 1]))
                    for index in range(0, len(shuffled), 2)
                )
            )
            removed_tuple = tuple(sorted(removed))
            if len(set(added)) != k or any(u == v for u, v in added):
                continue
            if sorted(vertex for edge in removed_tuple for vertex in edge) != sorted(
                vertex for edge in added for vertex in edge
            ):
                continue
            if set(removed_tuple) == set(added):
                continue
            remaining = self.overlay.edges.difference(removed_tuple)
            if any(edge in remaining for edge in added):
                continue
            candidates.add((removed_tuple, added))
        return tuple(
            _Reference("matching", candidate)
            for candidate in sorted(candidates)
        )

    def _select(self, method: str, arguments: Mapping[str, object]) -> tuple[_Reference, ...]:
        adjacency = self.overlay.adjacency()
        degrees = tuple(len(neighbors) for neighbors in adjacency)
        mode = self._mode(arguments)
        path = f"selector/{self._selector_calls - 1}/{method}"
        kind: str
        references: tuple[_Reference, ...]
        if method == "vertices_degree_extreme":
            self._require_keys(arguments, optional=frozenset({"mode"}))
            references = _extreme(
                {
                    _Reference("vertex", vertex): degree
                    for vertex, degree in enumerate(degrees)
                },
                mode,
            )
            kind = "vertex"
        elif method == "vertices_degree_class":
            self._require_keys(arguments, required=frozenset({"degree"}))
            degree = arguments["degree"]
            if isinstance(degree, bool) or not isinstance(degree, int):
                raise SafeAPIProgramError("INVALID_API_ARGUMENT", "degree must be an integer")
            references = tuple(
                _Reference("vertex", vertex)
                for vertex, actual in enumerate(degrees)
                if actual == degree
            )
            kind = "vertex"
        elif method in {"vertices_witness_load_extreme", "edges_witness_load_extreme"}:
            self._require_keys(
                arguments,
                required=frozenset({"length"}),
                optional=frozenset({"mode"}),
            )
            length = arguments["length"]
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or length not in self.context.forbidden_lengths
            ):
                raise SafeAPIProgramError(
                    "INVALID_API_ARGUMENT",
                    "length must be one of ctx.forbidden_lengths",
                )
            vertex_loads, edge_loads = self._witness_loads()
            if method.startswith("vertices_"):
                references = _extreme(
                    {
                        _Reference("vertex", vertex): vertex_loads.get((length, vertex), 0)
                        for vertex in range(self.overlay.order)
                    },
                    mode,
                )
                kind = "vertex"
            else:
                references = _extreme(
                    {
                        _Reference("edge", edge): edge_loads.get((length, edge), 0)
                        for edge in sorted(self.overlay.edges)
                    },
                    mode,
                )
                kind = "edge"
        elif method == "vertices_articulation_risk":
            self._require_keys(arguments, optional=frozenset({"mode"}))
            articulations, _ = _tarjan(self.overlay)
            references = _extreme(
                {
                    _Reference("vertex", vertex): int(vertex in articulations)
                    for vertex in range(self.overlay.order)
                },
                mode,
            )
            kind = "vertex"
        elif method == "edges_bridge_risk":
            self._require_keys(arguments, optional=frozenset({"mode"}))
            _, bridges = _tarjan(self.overlay)
            references = _extreme(
                {
                    _Reference("edge", edge): int(edge in bridges)
                    for edge in sorted(self.overlay.edges)
                },
                mode,
            )
            kind = "edge"
        elif method == "edges_removable":
            self._require_keys(arguments)
            references = tuple(
                _Reference("edge", edge) for edge in sorted(self.overlay.edges)
            )
            kind = "edge"
        elif method == "vertices_distance_band":
            self._require_keys(
                arguments,
                required=frozenset({"source", "minimum", "maximum"}),
            )
            source = self._decode_reference(arguments["source"], "vertex")
            minimum = arguments["minimum"]
            maximum = arguments["maximum"]
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or minimum < 0
                or maximum < minimum
            ):
                raise SafeAPIProgramError(
                    "INVALID_API_ARGUMENT",
                    "distance band must satisfy 0 <= minimum <= maximum",
                )
            source_vertex = cast(int, source.payload)
            distances = _distances(self.overlay, source_vertex)
            references = tuple(
                _Reference("vertex", vertex)
                for vertex, distance in enumerate(distances)
                if minimum <= distance <= maximum
            )
            kind = "vertex"
        elif method == "non_edges_from_vertex":
            self._require_keys(arguments, required=frozenset({"vertex"}))
            source = self._decode_reference(arguments["vertex"], "vertex")
            source_vertex = cast(int, source.payload)
            references = tuple(
                _Reference("non_edge", normalized_edge((source_vertex, vertex)))
                for vertex in range(self.overlay.order)
                if vertex != source_vertex and vertex not in adjacency[source_vertex]
            )
            kind = "non_edge"
        elif method in {"non_edges_legal", "non_edges_local_cycle_risk"}:
            self._require_keys(
                arguments,
                optional=frozenset({"mode"})
                if method == "non_edges_local_cycle_risk"
                else frozenset(),
            )
            values: dict[_Reference, int] = {}
            for u in range(self.overlay.order):
                for v in range(u + 1, self.overlay.order):
                    edge = (u, v)
                    if edge not in self.overlay.edges:
                        values[_Reference("non_edge", edge)] = len(
                            adjacency[u] & adjacency[v]
                        )
            references = (
                tuple(sorted(values, key=_reference_key))
                if method == "non_edges_legal"
                else _extreme(values, mode)
            )
            kind = "non_edge"
        elif method == "paths_length_two":
            self._require_keys(arguments)
            references = tuple(
                sorted(
                    (
                        _Reference("path", (u, center, v))
                        for center in range(self.overlay.order)
                        for u, v in combinations(sorted(adjacency[center]), 2)
                    ),
                    key=_reference_key,
                )
            )
            kind = "path"
        elif method == "matching_k_switch_reconnections":
            self._require_keys(arguments, required=frozenset({"k"}))
            k = arguments["k"]
            if isinstance(k, bool) or not isinstance(k, int):
                raise SafeAPIProgramError("INVALID_API_ARGUMENT", "k must be an integer")
            references = self._matching_candidates(k, path=path)
            kind = "matching"
        elif method == "relocations_legal":
            self._require_keys(arguments)
            references = tuple(
                _Reference("relocation", (edge, keep, new))
                for edge in sorted(self.overlay.edges)
                for keep in edge
                for new in range(self.overlay.order)
                if new not in edge
                and normalized_edge((keep, new)) not in self.overlay.edges
            )
            kind = "relocation"
        elif method == "edge_fanouts_legal":
            self._require_keys(arguments)
            references = tuple(
                _Reference("fanout", (edge, vertex))
                for edge in sorted(self.overlay.edges)
                for vertex in range(self.overlay.order)
                if vertex not in edge
                and {
                    normalized_edge((edge[0], vertex)),
                    normalized_edge((edge[1], vertex)),
                }.isdisjoint(self.overlay.edges)
            )
            kind = "fanout"
        else:
            raise SafeAPIProgramError("UNKNOWN_API_METHOD", f"unknown selector {method}")
        return self._bounded(references, kind=kind, path=path)

    def _pick(self, arguments: Mapping[str, object]) -> dict[str, JsonValue] | None:
        self._require_keys(
            arguments,
            required=frozenset({"items", "seed", "salt"}),
            optional=frozenset({"feature"}),
        )
        items = arguments["items"]
        supplied_seed = arguments["seed"]
        salt = arguments["salt"]
        feature = arguments.get("feature", "uniform")
        if not isinstance(items, list):
            raise SafeAPIProgramError("INVALID_API_ARGUMENT", "items must be a tuple of refs")
        if len(items) > self.limits.selector_result_size:
            raise SafeAPIProgramError(
                "INVALID_API_ARGUMENT",
                "pick items exceed the selector-result cap",
            )
        if supplied_seed != self.seed or isinstance(supplied_seed, bool):
            raise SafeAPIProgramError(
                "INVALID_API_ARGUMENT",
                "pick seed must equal the invocation seed",
            )
        if isinstance(salt, bool) or not isinstance(salt, (int, str)):
            raise SafeAPIProgramError("INVALID_API_ARGUMENT", "salt must be an int or string")
        if isinstance(salt, int) and not -(1 << 63) <= salt <= (1 << 63) - 1:
            raise SafeAPIProgramError("INVALID_API_ARGUMENT", "integer salt is out of range")
        if isinstance(salt, str) and (
            not salt.isascii()
            or not salt.isprintable()
            or len(salt.encode("ascii")) > 128
        ):
            raise SafeAPIProgramError(
                "INVALID_API_ARGUMENT",
                "string salt must be printable ASCII of at most 128 bytes",
            )
        if feature not in {"uniform", "degree", "inverse_degree"}:
            raise SafeAPIProgramError("INVALID_API_ARGUMENT", "unsupported pick feature")
        references = tuple(self._decode_reference(item) for item in items)
        if not references:
            return None
        kinds = {reference.kind for reference in references}
        if len(kinds) != 1:
            raise SafeAPIProgramError(
                "INVALID_REFERENCE_TYPE",
                "pick items must have one reference type",
            )
        adjacency = self.overlay.adjacency()
        weights: list[int] = []
        for reference in references:
            if feature == "uniform":
                weights.append(1)
            elif reference.kind == "vertex":
                vertex = cast(int, reference.payload)
                degree = len(adjacency[vertex])
                weights.append(
                    max(1, degree)
                    if feature == "degree"
                    else max(1, self.overlay.order - degree)
                )
            else:
                raise SafeAPIProgramError(
                    "INVALID_API_ARGUMENT",
                    "degree features require vertex references",
                )
        random_seed = derive_seed64(
            RANDOM_PROTOCOL_ID,
            self.seed,
            self.context.invocation_ordinal,
            salt,
            feature,
        )
        selected, draws = weighted_index(random_seed, weights)
        self._consume_draws(draws)
        return self._mint_reference(references[selected])

    def _apply_action(self, method: str, arguments: Mapping[str, object]) -> None:
        edges = self.overlay.edges
        if method == "add_edge":
            self._require_keys(arguments, required=frozenset({"edge"}))
            reference = self._decode_reference(arguments["edge"], "non_edge")
            edge = cast(Edge, reference.payload)
            if edge in edges:
                raise SafeAPIProgramError("ACTION_PRECONDITION", "edge cannot be added")
            edges.add(edge)
        elif method == "remove_edge":
            self._require_keys(arguments, required=frozenset({"edge"}))
            reference = self._decode_reference(arguments["edge"], "edge")
            edge = cast(Edge, reference.payload)
            if edge not in edges:
                raise SafeAPIProgramError("ACTION_PRECONDITION", "edge is absent")
            edges.remove(edge)
        elif method == "relocate_endpoint":
            self._require_keys(arguments, required=frozenset({"relocation"}))
            reference = self._decode_reference(arguments["relocation"], "relocation")
            edge, keep, new = cast(tuple[Edge, int, int], reference.payload)
            replacement = normalized_edge((keep, new))
            if (
                edge not in edges
                or keep not in edge
                or new in edge
                or replacement in edges
            ):
                raise SafeAPIProgramError(
                    "ACTION_PRECONDITION",
                    "relocation precondition failed",
                )
            edges.remove(edge)
            edges.add(replacement)
        elif method == "k_switch":
            self._require_keys(arguments, required=frozenset({"matching"}))
            reference = self._decode_reference(arguments["matching"], "matching")
            removed, added = cast(tuple[tuple[Edge, ...], tuple[Edge, ...]], reference.payload)
            if not set(removed).issubset(edges):
                raise SafeAPIProgramError(
                    "ACTION_PRECONDITION",
                    "matching source edge is absent",
                )
            remaining = edges.difference(removed)
            if any(edge in remaining for edge in added):
                raise SafeAPIProgramError(
                    "ACTION_PRECONDITION",
                    "matching target edge exists",
                )
            edges.difference_update(removed)
            edges.update(added)
        elif method == "edge_fanout":
            self._require_keys(arguments, required=frozenset({"fanout"}))
            reference = self._decode_reference(arguments["fanout"], "fanout")
            edge, vertex = cast(tuple[Edge, int], reference.payload)
            additions = {
                normalized_edge((edge[0], vertex)),
                normalized_edge((edge[1], vertex)),
            }
            if (
                edge not in edges
                or vertex in edge
                or any(addition in edges for addition in additions)
            ):
                raise SafeAPIProgramError("ACTION_PRECONDITION", "fanout precondition failed")
            edges.remove(edge)
            edges.update(additions)
        elif method == "edge_fold":
            self._require_keys(arguments, required=frozenset({"path"}))
            reference = self._decode_reference(arguments["path"], "path")
            u, center, v = cast(tuple[int, int, int], reference.payload)
            first = normalized_edge((u, center))
            second = normalized_edge((center, v))
            replacement = normalized_edge((u, v))
            if first not in edges or second not in edges or replacement in edges:
                raise SafeAPIProgramError("ACTION_PRECONDITION", "fold precondition failed")
            edges.remove(first)
            edges.remove(second)
            edges.add(replacement)
        else:
            raise SafeAPIProgramError("UNKNOWN_API_METHOD", f"unknown action {method}")
        self._gross_actions += 1

    def _emit(self) -> dict[str, JsonValue]:
        adjacency = self.overlay.adjacency()
        if not _connected(self.overlay) or min(map(len, adjacency)) < 3:
            return self._mint_result(NoPlan("ILLEGAL_FINAL_STATE"), "no_plan")
        removed = tuple(sorted(self.overlay.initial_edges.difference(self.overlay.edges)))
        added = tuple(sorted(self.overlay.edges.difference(self.overlay.initial_edges)))
        if not removed and not added:
            return self._mint_result(NoPlan("NO_EFFECT"), "no_plan")
        if (
            len(removed) > self.limits.net_removed_edges
            or len(added) > self.limits.net_added_edges
        ):
            raise SafeAPIProgramError(
                "EDGE_BUDGET_EXCEEDED",
                "net rewrite exceeds edge limits",
            )
        rewrite = RewritePlan(
            removed_edges=removed,
            added_edges=added,
            operator_family="native_v3_python_policy",
            metadata={
                "program_hash": self.program_hash,
                "safe_graph_api_protocol_id": SAFE_GRAPH_API_PROTOCOL_ID,
                "gross_actions": self._gross_actions,
                "selector_calls": self._selector_calls,
            },
        )
        try:
            candidate = self.rewrite_host.apply_rewrite(self.graph, rewrite)
        except IllegalRewriteError:
            return self._mint_result(NoPlan("ILLEGAL_FINAL_STATE"), "no_plan")
        except Exception as error:
            raise SafeAPIInfrastructureError("rewrite host failed unexpectedly") from error
        if candidate.order != self.graph.order or candidate != self.overlay.graph():
            return self._mint_result(NoPlan("ILLEGAL_FINAL_STATE"), "no_plan")
        return self._mint_result(rewrite, "rewrite_plan")

    def handle_call(self, method: str, arguments: Mapping[str, object]) -> JsonValue:
        """Validate and execute one worker-originated safe-API request."""

        self._consume("_api_calls", self.limits.total_api_calls, "API_CALL_BUDGET_EXCEEDED")
        if self._terminal_token is not None:
            raise SafeAPIProgramError(
                "CALL_AFTER_TERMINAL_RESULT",
                "safe API cannot be used after emit/no_plan",
            )
        try:
            result: JsonValue
            if method in SELECTOR_METHODS:
                self._consume(
                    "_selector_calls",
                    self.limits.selector_calls,
                    "SELECTOR_CALL_BUDGET_EXCEEDED",
                )
                references = self._select(method, arguments)
                result = [self._mint_reference(reference) for reference in references]
            elif method == "pick":
                result = self._pick(arguments)
            elif method in ACTION_METHODS - {"emit", "no_plan"}:
                self._consume(
                    "_action_calls",
                    self.limits.action_calls,
                    "ACTION_CALL_BUDGET_EXCEEDED",
                )
                self._apply_action(method, arguments)
                result = None
            elif method == "emit":
                self._require_keys(arguments)
                result = self._emit()
            elif method == "no_plan":
                self._require_keys(arguments, optional=frozenset({"reason"}))
                reason = arguments.get("reason", "EXPLICIT")
                if not isinstance(reason, str):
                    raise SafeAPIProgramError(
                        "INVALID_API_ARGUMENT",
                        "NoPlan reason must be a string",
                    )
                try:
                    no_plan = NoPlan(reason)
                except ValueError as error:
                    raise SafeAPIProgramError(
                        "INVALID_API_ARGUMENT",
                        str(error),
                    ) from error
                result = self._mint_result(no_plan, "no_plan")
            else:
                raise SafeAPIProgramError(
                    "UNKNOWN_API_METHOD",
                    f"unknown safe API method {method!r}",
                )
        except SafeAPIProgramError as error:
            self._record(method, arguments, {"failure": error.code})
            raise
        self._record(method, arguments, result)
        return result
