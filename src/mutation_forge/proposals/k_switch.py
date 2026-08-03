from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from typing import cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.models import Edge, GraphScore, GraphState, JsonValue, RewritePlan
from mutation_forge.sandbox.contracts import (
    SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
    SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
    ScientificContext,
    ScientificProposal,
)

SUPPORTED_K_VALUES = (2, 3, 4)
SUPPORTED_SELECTORS = (
    "uniform_random",
    "sampled_forbidden_cycle_anchored",
    "high_sampled_witness_load",
    "remote_from_anchor",
    "pairwise_distant_disjoint",
    "mixed_exploit_explore",
)
POOL_SCHEMA_VERSION = "stage2b.pool.v1"
FEATURE_SCHEMA_VERSION = "stage2b.features.v1"


class EvaluationContractError(RuntimeError):
    """The authoritative score and target-length contracts disagree."""


@dataclass(frozen=True, slots=True)
class FeatureLimits:
    forbidden_lengths: tuple[int, ...]
    witness_sample_cap: int = 32
    cycle_node_budget: int = 20_000
    distance_query_budget: int = 256
    local_risk_budget: int = 2_048

    def __post_init__(self) -> None:
        if (
            not self.forbidden_lengths
            or len(set(self.forbidden_lengths)) != len(self.forbidden_lengths)
            or any(length < 1 for length in self.forbidden_lengths)
        ):
            raise ValueError("forbidden_lengths must be unique positive integers")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "forbidden_lengths": list(self.forbidden_lengths),
            "witness_sample_cap": self.witness_sample_cap,
            "cycle_node_budget": self.cycle_node_budget,
            "distance_query_budget": self.distance_query_budget,
            "local_risk_budget": self.local_risk_budget,
        }


@dataclass(frozen=True, slots=True)
class PoolLimits:
    pool_size: int = 12
    k_values: tuple[int, ...] = SUPPORTED_K_VALUES
    selectors: tuple[str, ...] = SUPPORTED_SELECTORS
    selector_weights: tuple[int, ...] = (2, 2, 2, 1, 2, 3)
    retry_limit: int = 96
    matching_limit: int = 105

    def __post_init__(self) -> None:
        if not 1 <= self.pool_size <= 64:
            raise ValueError("pool_size must be in [1, 64]")
        if (
            not self.k_values
            or any(k not in SUPPORTED_K_VALUES for k in self.k_values)
            or len(set(self.k_values)) != len(self.k_values)
        ):
            raise ValueError("k_values must be unique values from 2, 3, 4")
        if (
            not self.selectors
            or any(selector not in SUPPORTED_SELECTORS for selector in self.selectors)
            or len(set(self.selectors)) != len(self.selectors)
        ):
            raise ValueError("selectors contain unsupported or duplicate values")
        if len(self.selector_weights) != len(self.selectors) or any(
            weight <= 0 for weight in self.selector_weights
        ):
            raise ValueError("selector_weights must be aligned positive integers")
        if not 1 <= self.retry_limit <= 1_024:
            raise ValueError("retry_limit must be in [1, 1024]")
        if not 1 <= self.matching_limit <= 105:
            raise ValueError("matching_limit must be in [1, 105]")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "pool_size": self.pool_size,
            "k_values": list(self.k_values),
            "selectors": list(self.selectors),
            "selector_weights": list(self.selector_weights),
            "retry_limit": self.retry_limit,
            "matching_limit": self.matching_limit,
        }


@dataclass(slots=True)
class FeatureUsage:
    cycle_nodes: int = 0
    sampled_witnesses: int = 0
    distance_queries: int = 0
    distance_cache_hits: int = 0
    local_risk_operations: int = 0
    cycle_budget_exhausted: bool = False
    distance_budget_exhausted: bool = False
    local_risk_budget_exhausted: bool = False

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "cycle_nodes": self.cycle_nodes,
            "sampled_witnesses": self.sampled_witnesses,
            "distance_queries": self.distance_queries,
            "distance_cache_hits": self.distance_cache_hits,
            "local_risk_operations": self.local_risk_operations,
            "cycle_budget_exhausted": self.cycle_budget_exhausted,
            "distance_budget_exhausted": self.distance_budget_exhausted,
            "local_risk_budget_exhausted": self.local_risk_budget_exhausted,
        }


@dataclass(frozen=True, slots=True)
class ProposalCandidate:
    rewrite: RewritePlan
    payload: ScientificProposal

    @property
    def proposal_id(self) -> str:
        return self.payload["proposal_id"]

    def as_dict(self, *, include_plan: bool = False) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"proposal": cast(dict[str, JsonValue], self.payload)}
        if include_plan:
            result["rewrite"] = {
                "removed_edges": [list(edge) for edge in self.rewrite.removed_edges],
                "added_edges": [list(edge) for edge in self.rewrite.added_edges],
                "operator_family": self.rewrite.operator_family,
                "metadata": dict(self.rewrite.metadata),
            }
        return result


@dataclass(frozen=True, slots=True)
class ProposalPool:
    schema_version: str
    candidates: tuple[ProposalCandidate, ...]
    pool_hash: str
    attempted: int
    rejected: dict[str, int]
    deduplicated: int
    retained: int
    selector_counts: dict[str, int]
    k_counts: dict[str, int]
    feature_usage: dict[str, JsonValue]
    legality_elapsed_ns: int
    feature_elapsed_ns: int

    def as_dict(self, *, include_plans: bool = False) -> dict[str, JsonValue]:
        telemetry = cast(
            dict[str, JsonValue],
            {
                "attempted": self.attempted,
                "rejected": cast(dict[str, JsonValue], self.rejected),
                "deduplicated": self.deduplicated,
                "retained": self.retained,
                "selector_counts": cast(dict[str, JsonValue], self.selector_counts),
                "k_counts": cast(dict[str, JsonValue], self.k_counts),
                "feature_usage": self.feature_usage,
                "legality_elapsed_ns": self.legality_elapsed_ns,
                "feature_elapsed_ns": self.feature_elapsed_ns,
            },
        )
        return {
            "schema_version": self.schema_version,
            "pool_hash": self.pool_hash,
            "candidates": [
                candidate.as_dict(include_plan=include_plans) for candidate in self.candidates
            ],
            "telemetry": telemetry,
        }


def make_scientific_context(
    graph: GraphState,
    score: GraphScore,
    *,
    forbidden_lengths: tuple[int, ...],
    step: int,
    remaining_steps: int,
    stagnation: int = 0,
    recent_best_improvement: float = 0.0,
    recent_acceptance_rate: float = 0.0,
    recent_duplicate_rate: float = 0.0,
) -> ScientificContext:
    observed = tuple(length for length, _ in score.capped_cycle_counts)
    counts = tuple(count for _, count in score.capped_cycle_counts)
    if observed != forbidden_lengths:
        raise EvaluationContractError(
            "score lengths do not match backend target lengths: "
            f"expected {forbidden_lengths!r}, observed {observed!r}"
        )
    if len(observed) != len(set(observed)):
        raise EvaluationContractError("score contains duplicate target lengths")
    if any(count < 0 for count in counts):
        raise EvaluationContractError("score contains a negative witness count")
    if sum(counts) != score.total_capped_witnesses:
        raise EvaluationContractError(
            "score total_capped_witnesses does not equal its count vector"
        )
    if not score.valid and score.total_capped_witnesses == 0:
        raise EvaluationContractError("invalid score cannot report zero witnesses")
    return {
        "schema_version": SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
        "order": graph.order,
        "forbidden_lengths": list(forbidden_lengths),
        "capped_cycle_counts": list(counts),
        "weighted_penalty": score.weighted_penalty,
        "step": step,
        "remaining_steps": remaining_steps,
        "stagnation": stagnation,
        "recent_best_improvement": recent_best_improvement,
        "recent_acceptance_rate": recent_acceptance_rate,
        "recent_duplicate_rate": recent_duplicate_rate,
    }


class _FeatureSnapshot:
    def __init__(self, graph: GraphState, limits: FeatureLimits) -> None:
        self.graph = graph
        self.limits = limits
        self.usage = FeatureUsage()
        self.adjacency = [set[int]() for _ in range(graph.order)]
        for u, v in graph.edges:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
        self.witnesses: dict[int, tuple[tuple[int, ...], ...]] = {}
        self.edge_loads: dict[int, Counter[Edge]] = {}
        self._distance_cache: dict[tuple[int, int], int] = {}
        for length in limits.forbidden_lengths:
            cycles = self._sample_cycles(length)
            self.witnesses[length] = cycles
            loads: Counter[Edge] = Counter()
            for cycle in cycles:
                for index, vertex in enumerate(cycle):
                    edge = tuple(sorted((vertex, cycle[(index + 1) % len(cycle)])))
                    loads[cast(Edge, edge)] += 1
            self.edge_loads[length] = loads

    @staticmethod
    def _canonical_cycle(cycle: tuple[int, ...]) -> tuple[int, ...]:
        rotations: list[tuple[int, ...]] = []
        for oriented in (cycle, tuple(reversed(cycle))):
            rotations.extend(oriented[index:] + oriented[:index] for index in range(len(oriented)))
        return min(rotations)

    def _sample_cycles(self, length: int) -> tuple[tuple[int, ...], ...]:
        found: set[tuple[int, ...]] = set()
        stop = False

        def visit(start: int, path: tuple[int, ...]) -> None:
            nonlocal stop
            if stop:
                return
            if self.usage.cycle_nodes >= self.limits.cycle_node_budget:
                self.usage.cycle_budget_exhausted = True
                stop = True
                return
            self.usage.cycle_nodes += 1
            if len(path) == length:
                if start in self.adjacency[path[-1]]:
                    found.add(self._canonical_cycle(path))
                    if len(found) >= self.limits.witness_sample_cap:
                        stop = True
                return
            for neighbor in sorted(self.adjacency[path[-1]]):
                if neighbor == start or neighbor in path:
                    continue
                visit(start, path + (neighbor,))
                if stop:
                    return

        for start in range(self.graph.order):
            visit(start, (start,))
            if stop:
                break
        cycles = tuple(sorted(found))[: self.limits.witness_sample_cap]
        self.usage.sampled_witnesses += len(cycles)
        return cycles

    def edge_total_load(self, edge: Edge) -> int:
        return sum(loads[edge] for loads in self.edge_loads.values())

    def distance(self, left: int, right: int) -> int:
        key = (min(left, right), max(left, right))
        if key in self._distance_cache:
            self.usage.distance_cache_hits += 1
            return self._distance_cache[key]
        if self.usage.distance_queries >= self.limits.distance_query_budget:
            self.usage.distance_budget_exhausted = True
            return self.graph.order
        self.usage.distance_queries += 1
        queue: deque[tuple[int, int]] = deque([(left, 0)])
        seen = {left}
        result = self.graph.order
        while queue:
            vertex, distance = queue.popleft()
            if vertex == right:
                result = distance
                break
            for neighbor in sorted(self.adjacency[vertex]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        self._distance_cache[key] = result
        return result

    def edge_distance(self, left: Edge, right: Edge) -> int:
        return min(self.distance(u, v) for u in left for v in right)

    def local_risks(
        self,
        removed: tuple[Edge, ...],
        added: tuple[Edge, ...],
    ) -> tuple[int, int]:
        if self.usage.local_risk_budget_exhausted:
            return 0, 0
        adjacency = [set(neighbors) for neighbors in self.adjacency]
        for u, v in removed:
            adjacency[u].remove(v)
            adjacency[v].remove(u)
        for u, v in added:
            adjacency[u].add(v)
            adjacency[v].add(u)
        triangles: set[tuple[int, ...]] = set()
        squares: set[tuple[int, ...]] = set()
        for u, v in added:
            for middle in adjacency[u].intersection(adjacency[v]):
                triangles.add(self._canonical_cycle((u, middle, v)))
            for first in adjacency[u].difference({v}):
                for second in adjacency[first].difference({u}):
                    if self.usage.local_risk_operations >= self.limits.local_risk_budget:
                        self.usage.local_risk_budget_exhausted = True
                        return len(triangles), len(squares)
                    self.usage.local_risk_operations += 1
                    if second != v and v in adjacency[second]:
                        squares.add(self._canonical_cycle((u, first, second, v)))
        return len(triangles), len(squares)

    def proposal_payload(
        self,
        *,
        proposal_id: str,
        removed: tuple[Edge, ...],
        added: tuple[Edge, ...],
        selector: str,
        k: int,
        anchor_length: int | None,
    ) -> ScientificProposal:
        removed_pairs = tuple(combinations(removed, 2))
        removed_distances = [self.edge_distance(left, right) for left, right in removed_pairs]
        new_distances = [self.distance(u, v) for u, v in added]
        triangle_risk, c4_risk = self.local_risks(removed, added)
        broken: list[int] = []
        load_sums: list[int] = []
        load_maxima: list[int] = []
        removed_set = set(removed)
        for length in self.limits.forbidden_lengths:
            witnesses = self.witnesses[length]
            broken.append(
                sum(
                    any(
                        tuple(sorted((vertex, cycle[(index + 1) % len(cycle)]))) in removed_set
                        for index, vertex in enumerate(cycle)
                    )
                    for cycle in witnesses
                )
            )
            loads = [self.edge_loads[length][edge] for edge in removed]
            load_sums.append(sum(loads))
            load_maxima.append(max(loads, default=0))
        return {
            "schema_version": SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": proposal_id,
            "k": k,
            "operator_family": f"legal_{k}_switch",
            "selector_tags": [selector],
            "anchor_forbidden_length": anchor_length,
            "broken_sampled_witnesses_by_length": broken,
            "removed_edge_load_sum_by_length": load_sums,
            "removed_edge_load_max_by_length": load_maxima,
            "minimum_distance_between_removed_edges": min(
                removed_distances,
                default=0,
            ),
            "mean_distance_between_removed_edges": (
                sum(removed_distances) / len(removed_distances) if removed_distances else 0.0
            ),
            "minimum_preexisting_distance_for_new_edges": min(
                new_distances,
                default=0,
            ),
            "mean_preexisting_distance_for_new_edges": (
                sum(new_distances) / len(new_distances) if new_distances else 0.0
            ),
            "local_triangle_risk": triangle_risk,
            "local_c4_risk": c4_risk,
            "reconnection_span": (
                sum(new_distances) / len(new_distances) if new_distances else 0.0
            ),
        }


def _perfect_matchings(vertices: tuple[int, ...]) -> tuple[tuple[Edge, ...], ...]:
    if not vertices:
        return ((),)
    first = vertices[0]
    result: list[tuple[Edge, ...]] = []
    for index in range(1, len(vertices)):
        second = vertices[index]
        remaining = vertices[1:index] + vertices[index + 1 :]
        for suffix in _perfect_matchings(remaining):
            edge = cast(Edge, tuple(sorted((first, second))))
            result.append(tuple(sorted((edge, *suffix))))
    return tuple(sorted(set(result)))


class KSwitchPoolGenerator:
    def __init__(
        self,
        backend: GraphBackend,
        *,
        pool_limits: PoolLimits | None = None,
        feature_limits: FeatureLimits,
    ) -> None:
        self.backend = backend
        self.pool_limits = pool_limits or PoolLimits()
        self.feature_limits = feature_limits

    @staticmethod
    def _seed(
        graph: GraphState,
        policy_seed: int,
        step: int,
        attempt: int,
        selector: str,
    ) -> int:
        payload = json.dumps(
            [graph.order, graph.edges, policy_seed, step, attempt, selector],
            separators=(",", ":"),
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _selector_for_attempt(self, attempt: int, policy_seed: int) -> str:
        weighted = [
            selector
            for selector, weight in zip(
                self.pool_limits.selectors,
                self.pool_limits.selector_weights,
                strict=True,
            )
            for _ in range(weight)
        ]
        return weighted[(attempt + policy_seed) % len(weighted)]

    @staticmethod
    def _disjoint_greedy(edges: list[Edge], k: int) -> tuple[Edge, ...] | None:
        selected: list[Edge] = []
        used: set[int] = set()
        for edge in edges:
            if not used.intersection(edge):
                selected.append(edge)
                used.update(edge)
                if len(selected) == k:
                    return tuple(sorted(selected))
        return None

    def _select_edges(
        self,
        snapshot: _FeatureSnapshot,
        *,
        k: int,
        selector: str,
        seed: int,
    ) -> tuple[tuple[Edge, ...] | None, int | None]:
        rng = random.Random(seed)
        edges = list(snapshot.graph.edges)
        rng.shuffle(edges)
        anchor_length: int | None = None
        if selector == "uniform_random":
            return self._disjoint_greedy(edges, k), None
        loads = {edge: snapshot.edge_total_load(edge) for edge in edges}
        loaded = sorted(edges, key=lambda edge: (-loads[edge], edge))
        if selector == "sampled_forbidden_cycle_anchored":
            available = [
                length for length in snapshot.limits.forbidden_lengths if snapshot.witnesses[length]
            ]
            anchor_length = available[0] if available else None
            if anchor_length is not None:
                cycle = snapshot.witnesses[anchor_length][
                    seed % len(snapshot.witnesses[anchor_length])
                ]
                cycle_edges = {
                    cast(
                        Edge,
                        tuple(
                            sorted(
                                (
                                    vertex,
                                    cycle[(index + 1) % len(cycle)],
                                )
                            )
                        ),
                    )
                    for index, vertex in enumerate(cycle)
                }
                ordered = sorted(
                    edges,
                    key=lambda edge: (
                        edge not in cycle_edges,
                        -loads[edge],
                        edge,
                    ),
                )
                return self._disjoint_greedy(ordered, k), anchor_length
            return self._disjoint_greedy(loaded, k), None
        if selector == "high_sampled_witness_load":
            return self._disjoint_greedy(loaded, k), None
        anchor = loaded[0] if loaded else None
        if anchor is None:
            return None, None
        if selector == "remote_from_anchor":
            ordered = [
                anchor,
                *sorted(
                    (edge for edge in edges if edge != anchor),
                    key=lambda edge: (-snapshot.edge_distance(anchor, edge), edge),
                ),
            ]
            return self._disjoint_greedy(ordered, k), None
        if selector == "pairwise_distant_disjoint":
            selected = [anchor]
            used = set(anchor)
            while len(selected) < k:
                candidates = [edge for edge in edges if not used.intersection(edge)]
                if not candidates:
                    return None, None
                candidate = max(
                    candidates,
                    key=lambda edge: (
                        min(snapshot.edge_distance(edge, prior) for prior in selected),
                        loads[edge],
                        tuple(-item for item in edge),
                    ),
                )
                selected.append(candidate)
                used.update(candidate)
            return tuple(sorted(selected)), None
        exploit = loaded if seed % 2 == 0 else edges
        return self._disjoint_greedy(exploit, k), None

    def generate(
        self,
        graph: GraphState,
        *,
        policy_seed: int,
        step: int,
    ) -> ProposalPool:
        feature_started = time.perf_counter_ns()
        snapshot = _FeatureSnapshot(graph, self.feature_limits)
        feature_elapsed = time.perf_counter_ns() - feature_started
        retained: list[ProposalCandidate] = []
        seen: set[tuple[tuple[Edge, ...], tuple[Edge, ...]]] = set()
        rejected: Counter[str] = Counter()
        selector_counts: Counter[str] = Counter()
        k_counts: Counter[str] = Counter()
        attempted = 0
        deduplicated = 0
        legality_elapsed = 0
        current_edges = set(graph.edges)
        for attempt in range(self.pool_limits.retry_limit):
            if len(retained) >= self.pool_limits.pool_size:
                break
            selector = self._selector_for_attempt(attempt, policy_seed)
            k = self.pool_limits.k_values[
                (attempt + step + policy_seed) % len(self.pool_limits.k_values)
            ]
            seed = self._seed(graph, policy_seed, step, attempt, selector)
            removed, anchor_length = self._select_edges(
                snapshot,
                k=k,
                selector=selector,
                seed=seed,
            )
            if removed is None:
                rejected["disjoint_selection"] += 1
                continue
            vertices = tuple(vertex for edge in removed for vertex in edge)
            original = frozenset(removed)
            matchings = list(_perfect_matchings(vertices))
            random.Random(seed ^ 0x9E3779B97F4A7C15).shuffle(matchings)
            for matching_index, matching in enumerate(matchings[: self.pool_limits.matching_limit]):
                if len(retained) >= self.pool_limits.pool_size:
                    break
                attempted += 1
                added = tuple(sorted(matching))
                if frozenset(added) == original:
                    rejected["original_pairing"] += 1
                    continue
                if original.intersection(added):
                    rejected["original_edge_reused"] += 1
                    continue
                if len(set(added)) != k or any(u == v for u, v in added):
                    rejected["loop_or_duplicate"] += 1
                    continue
                remaining = current_edges.difference(removed)
                if remaining.intersection(added):
                    rejected["preexisting_edge"] += 1
                    continue
                key = (removed, added)
                if key in seen:
                    deduplicated += 1
                    continue
                seen.add(key)
                proposal_id = hashlib.sha256(
                    json.dumps(
                        [
                            POOL_SCHEMA_VERSION,
                            seed,
                            matching_index,
                            removed,
                            added,
                        ],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                rewrite = RewritePlan(
                    removed_edges=removed,
                    added_edges=added,
                    operator_family=f"legal_{k}_switch",
                    metadata={
                        "k": k,
                        "selector": selector,
                        "proposal_id": proposal_id,
                    },
                )
                legality_started = time.perf_counter_ns()
                try:
                    self.backend.apply_rewrite(graph, rewrite)
                except ValueError:
                    rejected["host_validation"] += 1
                    legality_elapsed += time.perf_counter_ns() - legality_started
                    continue
                legality_elapsed += time.perf_counter_ns() - legality_started
                payload_started = time.perf_counter_ns()
                payload = snapshot.proposal_payload(
                    proposal_id=proposal_id,
                    removed=removed,
                    added=added,
                    selector=selector,
                    k=k,
                    anchor_length=anchor_length,
                )
                feature_elapsed += time.perf_counter_ns() - payload_started
                retained.append(ProposalCandidate(rewrite, payload))
                selector_counts[selector] += 1
                k_counts[str(k)] += 1
                break
        canonical_candidates = [
            {
                "proposal": candidate.payload,
                "removed_edges": candidate.rewrite.removed_edges,
                "added_edges": candidate.rewrite.added_edges,
            }
            for candidate in retained
        ]
        pool_hash = hashlib.sha256(
            json.dumps(
                canonical_candidates,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return ProposalPool(
            schema_version=POOL_SCHEMA_VERSION,
            candidates=tuple(retained),
            pool_hash=pool_hash,
            attempted=attempted,
            rejected=dict(sorted(rejected.items())),
            deduplicated=deduplicated,
            retained=len(retained),
            selector_counts=dict(sorted(selector_counts.items())),
            k_counts=dict(sorted(k_counts.items())),
            feature_usage=snapshot.usage.as_dict(),
            legality_elapsed_ns=legality_elapsed,
            feature_elapsed_ns=feature_elapsed,
        )
