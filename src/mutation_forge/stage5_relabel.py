"""Deterministic vertex relabeling for Stage 5 held-out graphs."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from mutation_forge.models import GraphState, JsonValue, normalized_edge
from mutation_forge.stage3.manifest import canonical_bytes

RELABEL_ALGORITHM = "fisher-yates-sha256-v1"


def _digest_int(domain: str, order: int, graph_seed: int, relabeling_seed: int, index: int) -> int:
    payload = [domain, order, graph_seed, relabeling_seed, index]
    return int.from_bytes(hashlib.sha256(canonical_bytes(payload)).digest()[:8], "big")


def deterministic_permutation(
    order: int,
    graph_seed: int,
    relabeling_seed: int,
) -> tuple[int, ...]:
    """Return an old-vertex -> new-vertex permutation with no global RNG state."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    if isinstance(graph_seed, bool) or not isinstance(graph_seed, int):
        raise ValueError("graph_seed must be an integer")
    if isinstance(relabeling_seed, bool) or not isinstance(relabeling_seed, int):
        raise ValueError("relabeling_seed must be an integer")
    values = list(range(order))
    for index in range(order - 1, 0, -1):
        swap = _digest_int(
            "stage5.relabel.permutation.v1", order, graph_seed, relabeling_seed, index
        ) % (index + 1)
        values[index], values[swap] = values[swap], values[index]
    return tuple(values)


def apply_permutation(graph: GraphState, permutation: tuple[int, ...]) -> GraphState:
    if len(permutation) != graph.order or set(permutation) != set(range(graph.order)):
        raise ValueError("permutation must contain each graph vertex exactly once")
    edges = tuple(
        sorted(normalized_edge((permutation[left], permutation[right])) for left, right in graph.edges)
    )
    return GraphState(order=graph.order, edges=edges)


def relabel_graph(
    graph: GraphState,
    *,
    graph_seed: int,
    relabeling_seed: int,
) -> tuple[GraphState, tuple[int, ...]]:
    permutation = deterministic_permutation(
        graph.order, graph_seed, relabeling_seed
    )
    relabeled = apply_permutation(graph, permutation)
    return relabeled, permutation


def graph_label_hash(graph: GraphState) -> str:
    return hashlib.sha256(
        canonical_bytes({"order": graph.order, "edges": [list(edge) for edge in graph.edges]})
    ).hexdigest()


def canonical_unlabeled_identity(graph: GraphState) -> str:
    """Return a label-independent graph identity without relabeling execution inputs.

    The certificate is a deterministic Weisfeiler--Leman refinement plus
    all-pairs distance profiles and edge-colour multiset.  Every component is
    constructed from unordered neighbourhood data, so it is invariant under a
    vertex permutation while remaining cheap for the frozen cubic graphs.
    """
    adjacency = [set[int]() for _ in range(graph.order)]
    for left, right in graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    colors = [str(len(neighbors)) for neighbors in adjacency]
    for _ in range(graph.order):
        signatures = [
            (colors[vertex], tuple(sorted(colors[neighbor] for neighbor in adjacency[vertex])))
            for vertex in range(graph.order)
        ]
        palette = {signature: str(index) for index, signature in enumerate(sorted(set(signatures), key=repr))}
        refined = [palette[signature] for signature in signatures]
        if refined == colors:
            break
        colors = refined
    distance_profiles: list[tuple[int, ...]] = []
    for start in range(graph.order):
        distances = [-1] * graph.order
        distances[start] = 0
        frontier = [start]
        while frontier:
            vertex = frontier.pop(0)
            for neighbor in sorted(adjacency[vertex]):
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[vertex] + 1
                    frontier.append(neighbor)
        distance_profiles.append(tuple(sorted(distances)))
    certificate = {
        "algorithm": "wl-distance-edge-v1",
        "order": graph.order,
        "vertex_colors": sorted(colors),
        "distance_profiles": sorted(distance_profiles),
        "edge_colors": sorted((min(colors[left], colors[right]), max(colors[left], colors[right])) for left, right in graph.edges),
    }
    return hashlib.sha256(canonical_bytes(certificate)).hexdigest()


def relabel_contract_digest(
    orders: Iterable[int], graph_seeds: Iterable[int], relabeling_seeds: Iterable[int]
) -> str:
    """Prove the frozen relabeling matrix before any policy trajectory runs."""
    from mutation_forge.backends.toy import ToyBackend

    backend = ToyBackend()
    proofs: list[dict[str, JsonValue]] = []
    for order in orders:
        for graph_seed in graph_seeds:
            base = backend.generate_seed(order=order, seed=graph_seed)
            identities: list[str] = []
            labeled_hashes: list[str] = []
            for relabeling_seed in relabeling_seeds:
                relabeled, permutation = relabel_graph(
                    base, graph_seed=graph_seed, relabeling_seed=relabeling_seed
                )
                identities.append(canonical_unlabeled_identity(relabeled))
                labeled_hashes.append(graph_label_hash(relabeled))
                proofs.append(
                    {
                        "order": order,
                        "graph_seed": graph_seed,
                        "relabeling_seed": relabeling_seed,
                        "permutation_sha256": hashlib.sha256(canonical_bytes(list(permutation))).hexdigest(),
                        "canonical_unlabeled_hash": identities[-1],
                        "relabeled_graph_hash": labeled_hashes[-1],
                    }
                )
            if len(set(identities)) != 1:
                raise ValueError("relabelings of one base graph have different unlabeled identities")
    return hashlib.sha256(canonical_bytes(proofs)).hexdigest()


@dataclass(frozen=True, slots=True)
class RelabelProof:
    algorithm: str
    order: int
    graph_seed: int
    relabeling_seed: int
    permutation: tuple[int, ...]
    base_graph_hash: str
    relabeled_graph_hash: str
    canonical_unlabeled_hash: str
    labels_changed: bool

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "algorithm": self.algorithm,
            "order": self.order,
            "graph_seed": self.graph_seed,
            "relabeling_seed": self.relabeling_seed,
            "permutation": list(self.permutation),
            "permutation_sha256": hashlib.sha256(canonical_bytes(list(self.permutation))).hexdigest(),
            "base_graph_hash": self.base_graph_hash,
            "relabeled_graph_hash": self.relabeled_graph_hash,
            "canonical_unlabeled_hash": self.canonical_unlabeled_hash,
            "labels_changed": self.labels_changed,
        }


def make_relabel_proof(
    base: GraphState,
    relabeled: GraphState,
    *,
    graph_seed: int,
    relabeling_seed: int,
    permutation: tuple[int, ...],
) -> RelabelProof:
    if base.order != relabeled.order:
        raise ValueError("base and relabeled graph orders differ")
    return RelabelProof(
        algorithm=RELABEL_ALGORITHM,
        order=base.order,
        graph_seed=graph_seed,
        relabeling_seed=relabeling_seed,
        permutation=permutation,
        base_graph_hash=graph_label_hash(base),
        relabeled_graph_hash=graph_label_hash(relabeled),
        canonical_unlabeled_hash=canonical_unlabeled_identity(base),
        labels_changed=base.edges != relabeled.edges,
    )


__all__ = [
    "RELABEL_ALGORITHM",
    "RelabelProof",
    "apply_permutation",
    "canonical_unlabeled_identity",
    "deterministic_permutation",
    "graph_label_hash",
    "make_relabel_proof",
    "relabel_contract_digest",
    "relabel_graph",
]
