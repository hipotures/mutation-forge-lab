from __future__ import annotations

import hashlib
import random

from mutation_forge.backends.base import (
    DeepProposalProfileRecorder,
    ProposalTimingRecorder,
    ScoreProfileRecorder,
)
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
    normalized_edge,
)


class ToyBackend:
    """Deterministic connected-cubic backend for isolated harness tests."""

    backend_id = "toy-connected-cubic-v1"

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        if order < 4:
            return ()
        return (4,)

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        if order < 4 or order % 2:
            raise ValueError("toy connected-cubic seeds require an even order >= 4")
        rng = random.Random(seed)
        labels = list(range(order))
        rng.shuffle(labels)
        edges = {
            normalized_edge((labels[index], labels[(index + 1) % order])) for index in range(order)
        }
        half = order // 2
        edges.update(
            normalized_edge((labels[index], labels[index + half])) for index in range(half)
        )
        graph = GraphState(order=order, edges=tuple(sorted(edges)))
        validation = self.validate(graph)
        if not validation.valid:
            raise RuntimeError(f"toy seed generation failed: {validation.errors}")
        return graph

    def validate(self, graph: GraphState) -> GraphValidation:
        errors: list[str] = []
        if graph.order < 1:
            errors.append("graph order must be positive")
        edge_set = set(graph.edges)
        if len(edge_set) != len(graph.edges):
            errors.append("duplicate edges")
        adjacency = [set[int]() for _ in range(max(0, graph.order))]
        for u, v in graph.edges:
            if not (0 <= u < graph.order and 0 <= v < graph.order):
                errors.append("edge endpoint out of range")
                continue
            if u == v:
                errors.append("loops are forbidden")
                continue
            adjacency[u].add(v)
            adjacency[v].add(u)
        if adjacency and any(len(neighbors) != 3 for neighbors in adjacency):
            errors.append("every vertex must have degree 3")
        if adjacency:
            seen = {0}
            frontier = [0]
            while frontier:
                vertex = frontier.pop()
                for neighbor in adjacency[vertex]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        frontier.append(neighbor)
            if len(seen) != graph.order:
                errors.append("graph must be connected")
        return GraphValidation(not errors, tuple(errors))

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: ScoreProfileRecorder | None = None,
    ) -> GraphScore | None:
        validation = self.validate(graph)
        if not validation.valid:
            return GraphScore(False, (), 10**9, 10**9, True, (1, 10**9, 10**9))
        adjacency = [set[int]() for _ in range(graph.order)]
        for u, v in graph.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        twice_squares = 0
        for u in range(graph.order):
            for v in range(u + 1, graph.order):
                common = len(adjacency[u].intersection(adjacency[v]))
                twice_squares += common * (common - 1) // 2
        squares = twice_squares // 2
        capped = min(squares, witness_cap)
        complete = squares <= witness_cap
        result = GraphScore(
            True,
            ((4, capped),),
            capped,
            capped * 16,
            complete,
            (0, capped, capped * 16, 0, len(graph.edges)),
        )
        if (
            cutoff is not None
            and cutoff.total_capped_witnesses > 0
            and result.ordering_key >= cutoff.ordering_key
        ):
            return None
        return result

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        score = self.score(graph, witness_cap=2**31 - 1)
        assert score is not None
        status = "VERIFIED" if score.total_capped_witnesses == 0 else "REJECTED"
        return ExactVerification(
            status=status,
            complete=True,
            message="toy exact four-cycle enumeration",
            implementation="toy-python",
        )

    def canonical_hash(self, graph: GraphState) -> str:
        return hashlib.sha256(self.serialize_graph6(graph).encode()).hexdigest()

    def state_hash(self, graph: GraphState) -> str:
        return hashlib.sha256(self.serialize_graph6(graph).encode()).hexdigest()

    def serialize_graph6(self, graph: GraphState) -> str:
        edge_text = ";".join(f"{u}-{v}" for u, v in graph.edges)
        return f"toy:{graph.order}:{edge_text}"

    def deserialize_graph6(self, value: str) -> GraphState:
        prefix, order_text, edge_text = value.split(":", 2)
        if prefix != "toy":
            raise ValueError("not a toy graph serialization")
        edges = tuple(
            (int(item.split("-", 1)[0]), int(item.split("-", 1)[1]))
            for item in edge_text.split(";")
            if item
        )
        return GraphState(int(order_text), edges)

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: ScoreProfileRecorder | None = None,
    ) -> GraphState:
        removed = tuple(normalized_edge(edge) for edge in rewrite.removed_edges)
        added = tuple(normalized_edge(edge) for edge in rewrite.added_edges)
        if len(removed) > 4 or len(added) > 4:
            raise ValueError("rewrites are limited to four removed and added edges")
        if len(set(removed)) != len(removed) or len(set(added)) != len(added):
            raise ValueError("rewrite contains duplicate edges")
        current = set(graph.edges)
        if not set(removed).issubset(current):
            raise ValueError("rewrite removes a missing edge")
        remaining = current.difference(removed)
        if any(u == v for u, v in added):
            raise ValueError("rewrite adds a loop")
        if set(added).intersection(remaining):
            raise ValueError("rewrite adds an existing edge")
        candidate = GraphState(graph.order, tuple(sorted(remaining.union(added))))
        validation = self.validate(candidate)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return candidate

    def propose_rewrite(
        self,
        graph: GraphState,
        *,
        operator_family: str,
        policy_seed: int,
        evaluation: int,
        record_timing: ProposalTimingRecorder | None = None,
        record_deep_profile: DeepProposalProfileRecorder | None = None,
    ) -> RewritePlan:
        rng = random.Random((policy_seed << 32) ^ evaluation)
        edges = list(graph.edges)
        for _ in range(64):
            first, second = rng.sample(edges, 2)
            if len(set(first + second)) != 4:
                continue
            a, b = first
            c, d = second
            if rng.randrange(2):
                added = (normalized_edge((a, c)), normalized_edge((b, d)))
            else:
                added = (normalized_edge((a, d)), normalized_edge((b, c)))
            rewrite = RewritePlan(
                removed_edges=(first, second),
                added_edges=added,
                operator_family=operator_family,
                metadata={"evaluation": evaluation},
            )
            try:
                self.apply_rewrite(graph, rewrite)
            except ValueError:
                continue
            return rewrite
        return RewritePlan((), (), operator_family, {"evaluation": evaluation, "noop": True})

    def close(self) -> None:
        return None
