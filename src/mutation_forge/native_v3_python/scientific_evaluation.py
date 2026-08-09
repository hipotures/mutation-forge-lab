"""Deterministic scientific-evaluation workload for ordinary-Python search."""

from __future__ import annotations

import random
from dataclasses import dataclass

from mutation_forge.backends.base import GraphBackend
from mutation_forge.models import JsonValue

from .search import DevelopmentCaseV1, panel_hash

SUPPORTED_GRAPH_MODES = frozenset(
    {
        "cubic_first",
        "minimal_structure_mixed_degree",
        "unrestricted_min_degree_3",
    }
)
SUPPORTED_BASELINES = ("random", "structural")
BASELINE_OPERATOR_FAMILIES = {
    "random": "heg_uniform_two_switch",
    "structural": "heg_forbidden_cycle_break",
}


@dataclass(frozen=True, slots=True)
class ScientificEvaluationOptionsV1:
    """Immutable Native-v2-equivalent evaluation controls."""

    graph_mode: str
    order_schedule: str
    min_order: int
    max_order: int
    orders_per_generation: int
    graph_seeds: tuple[int, ...]
    policy_seeds: tuple[int, ...]
    horizon: int
    witness_cap: int
    baselines: tuple[str, ...]
    replay: bool

    def __post_init__(self) -> None:
        if self.graph_mode not in SUPPORTED_GRAPH_MODES:
            raise ValueError(f"unsupported graph_mode: {self.graph_mode}")
        if self.order_schedule != "adaptive":
            raise ValueError("order_schedule must be 'adaptive'")
        if not 4 <= self.min_order <= self.max_order <= 128:
            raise ValueError("evaluation orders must satisfy 4 <= min <= max <= 128")
        if self.graph_mode == "minimal_structure_mixed_degree" and self.min_order < 5:
            raise ValueError("minimal_structure_mixed_degree requires min_order >= 5")
        if self.graph_mode == "cubic_first" and (
            self.min_order % 2 != 0 or self.max_order % 2 != 0
        ):
            raise ValueError("cubic_first requires even min_order and max_order")
        domain = self._order_domain()
        if not 1 <= self.orders_per_generation <= len(domain):
            raise ValueError("orders_per_generation exceeds the configured order domain")
        if (
            not self.graph_seeds
            or len(set(self.graph_seeds)) != len(self.graph_seeds)
            or any(seed < 0 for seed in self.graph_seeds)
        ):
            raise ValueError("graph_seeds must be nonempty, unique, and nonnegative")
        if (
            not self.policy_seeds
            or len(set(self.policy_seeds)) != len(self.policy_seeds)
            or any(seed < 0 for seed in self.policy_seeds)
        ):
            raise ValueError("policy_seeds must be nonempty, unique, and nonnegative")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.witness_cap < 1:
            raise ValueError("witness_cap must be positive")
        if self.baselines != SUPPORTED_BASELINES:
            raise ValueError("baselines must be exactly ['random', 'structural']")
        if self.replay:
            raise ValueError(
                "replay is not applicable to immutable ordinary-Python generation artifacts"
            )

    def _order_domain(self) -> tuple[int, ...]:
        orders = tuple(range(self.min_order, self.max_order + 1))
        if self.graph_mode == "cubic_first":
            return tuple(order for order in orders if order % 2 == 0)
        return orders

    def orders_for_generation(self, generation: int) -> tuple[int, ...]:
        if generation < 0:
            raise ValueError("generation must be nonnegative")
        seed = f"{','.join(map(str, self.graph_seeds))}:{generation}"
        sampled = random.Random(seed).sample(
            self._order_domain(),
            self.orders_per_generation,
        )
        return tuple(sorted(sampled))

    def panel_for_generation(
        self,
        *,
        generation: int,
        backend: GraphBackend,
    ) -> tuple[DevelopmentCaseV1, ...]:
        return tuple(
            DevelopmentCaseV1(
                case_id=(f"g{generation:04d}-o{order:04d}-g{graph_seed:04d}-p{policy_seed:04d}"),
                order=order,
                graph_seed=graph_seed,
                policy_seed=policy_seed,
                horizon=self.horizon,
                witness_cap=self.witness_cap,
                forbidden_lengths=backend.target_forbidden_lengths(order),
                graph_mode=self.graph_mode,
            )
            for order in self.orders_for_generation(generation)
            for graph_seed in self.graph_seeds
            for policy_seed in self.policy_seeds
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "graph_mode": self.graph_mode,
            "order_schedule": self.order_schedule,
            "min_order": self.min_order,
            "max_order": self.max_order,
            "orders_per_generation": self.orders_per_generation,
            "graph_seeds": list(self.graph_seeds),
            "policy_seeds": list(self.policy_seeds),
            "horizon": self.horizon,
            "witness_cap": self.witness_cap,
            "baselines": list(self.baselines),
            "replay": self.replay,
        }


def workload_projection(
    *,
    generation: int,
    panel: tuple[DevelopmentCaseV1, ...],
    options: ScientificEvaluationOptionsV1,
) -> dict[str, JsonValue]:
    """Return canonical status/report metadata for one frozen generation."""

    return {
        "generation": generation,
        "panel_hash": panel_hash(panel),
        "case_count": len(panel),
        "orders": list(options.orders_for_generation(generation)),
        "graph_mode": options.graph_mode,
        "order_schedule": options.order_schedule,
        "graph_seed_count": len(options.graph_seeds),
        "policy_seed_count": len(options.policy_seeds),
        "horizon": options.horizon,
        "witness_cap": options.witness_cap,
        "baselines": list(options.baselines),
        "replay": options.replay,
    }


__all__ = [
    "BASELINE_OPERATOR_FAMILIES",
    "SUPPORTED_BASELINES",
    "ScientificEvaluationOptionsV1",
    "workload_projection",
]
