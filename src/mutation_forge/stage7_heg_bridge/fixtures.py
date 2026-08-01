# ruff: noqa: E501
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from mutation_forge.models import GraphState, JsonValue
from mutation_forge.proposals.k_switch import ProposalPool
from mutation_forge.sandbox.contracts import ScientificContext
from mutation_forge.stage7_heg_bridge.bridge import HegPolicyBridge
from mutation_forge.stage7_heg_bridge.contract import FIXTURE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FixtureSpec:
    fixture_id: str
    family: str
    order: int
    graph_seed: int
    relabeling_seed: int
    policy_seed: int
    selectors: tuple[str, ...]
    k_values: tuple[int, ...]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "fixture_id": self.fixture_id,
            "family": self.family,
            "order": self.order,
            "graph_seed": self.graph_seed,
            "relabeling_seed": self.relabeling_seed,
            "policy_seed": self.policy_seed,
            "selectors": list(self.selectors),
            "k_values": list(self.k_values),
        }


@dataclass(frozen=True, slots=True)
class HEGFixture:
    spec: FixtureSpec
    graph: GraphState
    pool: ProposalPool
    context: ScientificContext

    def as_dict(
        self,
        *,
        include_plans: bool = True,
        include_timings: bool = True,
    ) -> dict[str, JsonValue]:
        pool = self.pool.as_dict(include_plans=include_plans)
        if not include_timings:
            telemetry = cast(dict[str, JsonValue], pool["telemetry"])
            telemetry.pop("legality_elapsed_ns", None)
            telemetry.pop("feature_elapsed_ns", None)
        return {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "spec": self.spec.as_dict(),
            "graph": {
                "order": self.graph.order,
                "edges": [[u, v] for u, v in self.graph.edges],
            },
            "context": cast(JsonValue, self.context),
            "pool": pool,
        }


ALL_SELECTORS = (
    "uniform_random",
    "sampled_forbidden_cycle_anchored",
    "high_sampled_witness_load",
    "remote_from_anchor",
    "pairwise_distant_disjoint",
    "mixed_exploit_explore",
)
ALL_K_VALUES = (2, 3, 4)


def fixture_specs() -> tuple[FixtureSpec, ...]:
    """Frozen held-out/fresh strata plus an order-30 HEG representative."""
    return (
        FixtureSpec("stage5-o14", "stage5_held_out", 14, 601, 6101, 6001, ALL_SELECTORS, ALL_K_VALUES),
        FixtureSpec("stage5-o18", "stage5_held_out", 18, 606, 6102, 6006, ALL_SELECTORS, ALL_K_VALUES),
        FixtureSpec("stage5-o22", "stage5_held_out", 22, 611, 6101, 6011, ALL_SELECTORS, ALL_K_VALUES),
        FixtureSpec("stage6-o20", "stage6_fresh", 20, 701, 7101, 7001, ALL_SELECTORS, ALL_K_VALUES),
        FixtureSpec("stage6-o24", "stage6_fresh", 24, 704, 7102, 7004, ALL_SELECTORS, ALL_K_VALUES),
        FixtureSpec("stage6-o28", "stage6_fresh", 28, 708, 7101, 7008, ALL_SELECTORS, ALL_K_VALUES),
        FixtureSpec("heg-order-30", "heg_representative", 30, 730, 7301, 7300, ALL_SELECTORS, ALL_K_VALUES),
    )


def build_fixture(bridge: HegPolicyBridge, spec: FixtureSpec) -> HEGFixture:
    graph = bridge.backend.generate_seed(order=spec.order, seed=spec.graph_seed)
    pool = bridge.generate_pool(graph, policy_seed=spec.policy_seed, step=0)
    score = bridge.backend.score(graph, witness_cap=32)
    bridge.telemetry.scorer_calls += 1
    if score is None or bridge.backend.score_implementation != "heg-cpp-score-worker":
        raise RuntimeError("fixture score did not use the mandatory HEG scorer")
    context = bridge.context_for_graph(
        graph,
        step=0,
        remaining_steps=31,
        score=score,
    )
    return HEGFixture(spec, graph, pool, context)


def build_fixtures(bridge: HegPolicyBridge) -> tuple[HEGFixture, ...]:
    return tuple(build_fixture(bridge, spec) for spec in fixture_specs())
