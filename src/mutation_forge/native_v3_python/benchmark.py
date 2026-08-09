"""Benchmark the provisional M2 policy-worker limits."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from typing import cast

from mutation_forge.models import GraphState, RewritePlan, normalized_edge
from mutation_forge.native_v3_python import (
    IllegalRewriteError,
    IsolatedPolicyWorkerV1,
    PolicyContextV1,
    PolicyRuntimeLimitsV1,
)

REPORT_SCHEMA_VERSION = "mforge.native.python_m2_benchmark.v1"

NO_PLAN_SOURCE = """\
def propose(ctx, graph, api, seed):
    return api.no_plan()
"""

ADD_EDGE_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.non_edges_legal()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    edge = api.pick(candidates, seed, "benchmark")
    if edge == None:
        return api.no_plan(reason="NO_MATCH")
    api.add_edge(edge)
    return api.emit()
"""

RELOCATION_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.relocations_legal()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    relocation = api.pick(candidates, seed, "benchmark-relocation")
    if relocation == None:
        return api.no_plan(reason="NO_MATCH")
    api.relocate_endpoint(relocation)
    return api.emit()
"""

FANOUT_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.edge_fanouts_legal()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    fanout = api.pick(candidates, seed, "benchmark-fanout")
    if fanout == None:
        return api.no_plan(reason="NO_MATCH")
    api.edge_fanout(fanout)
    return api.emit()
"""

K_SWITCH_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.matching_k_switch_reconnections(2)
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    matching = api.pick(candidates, seed, "benchmark-k-switch")
    if matching == None:
        return api.no_plan(reason="NO_MATCH")
    api.k_switch(matching)
    return api.emit()
"""

EDGE_FOLD_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.paths_length_two()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    path = candidates[0]
    api.edge_fold(path)
    return api.emit()
"""

RISK_SELECTOR_SOURCE = """\
def propose(ctx, graph, api, seed):
    vertices = api.vertices_articulation_risk()
    edges = api.edges_bridge_risk()
    candidates = api.non_edges_local_cycle_risk()
    return api.no_plan()
"""

WITNESS_SELECTOR_SOURCE = """\
def propose(ctx, graph, api, seed):
    vertices = api.vertices_witness_load_extreme(4)
    edges = api.edges_witness_load_extreme(4)
    return api.no_plan()
"""

FAILURE_SOURCE = """\
def propose(ctx, graph, api, seed):
    return 1
"""


def _cubic_graph(order: int = 30) -> GraphState:
    edges = {normalized_edge((vertex, (vertex + 1) % order)) for vertex in range(order)}
    edges.update((vertex, vertex + order // 2) for vertex in range(order // 2))
    return GraphState(order, tuple(sorted(edges)))


def _fold_graph() -> GraphState:
    graph = _cubic_graph(30)
    return GraphState(graph.order, tuple(sorted((*graph.edges, (0, 3), (0, 5)))))


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
    visited = {0}
    pending = [0]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == graph.order


class _Host:
    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState:
        current = set(graph.edges)
        removed = set(rewrite.removed_edges)
        added = set(rewrite.added_edges)
        if not removed.issubset(current) or added & (current - removed):
            raise IllegalRewriteError("invalid delta")
        candidate = GraphState(graph.order, tuple(sorted((current - removed) | added)))
        if min(_degrees(candidate)) < 3 or not _connected(candidate):
            raise IllegalRewriteError("illegal final graph")
        return candidate


HOST = _Host()
GRAPH = _cubic_graph()
SELECTOR_ACTION_CASES = (
    ("add-edge-order-30", ADD_EDGE_SOURCE, _cubic_graph(30)),
    ("relocation-order-30", RELOCATION_SOURCE, _cubic_graph(30)),
    ("fanout-order-30", FANOUT_SOURCE, _cubic_graph(30)),
    ("k-switch-order-30", K_SWITCH_SOURCE, _cubic_graph(30)),
    ("edge-fold-order-30", EDGE_FOLD_SOURCE, _fold_graph()),
    ("risk-selectors-order-128", RISK_SELECTOR_SOURCE, _cubic_graph(128)),
    ("witness-selectors-order-64", WITNESS_SELECTOR_SOURCE, _cubic_graph(64)),
)


def _context(ordinal: int) -> PolicyContextV1:
    return PolicyContextV1(
        step_index=ordinal,
        horizon=10_000,
        acceptance_profile_id="m2-benchmark",
        stagnation_steps=0,
        exploration_window_index=0,
        accepted_rewrites=0,
        accepted_non_improving_rewrites=0,
        consecutive_non_improving_rewrites=0,
        witness_cap=100,
        invocation_ordinal=ordinal,
        forbidden_lengths=(4, 6),
    )


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "p50_ms": percentile(0.50) * 1_000,
        "p95_ms": percentile(0.95) * 1_000,
        "max_ms": ordered[-1] * 1_000,
    }


@dataclass(frozen=True, slots=True)
class _Series:
    wall_seconds: list[float]
    rss_kib: list[int]
    worker_age_seconds: list[float]
    api_calls: list[int]


def _call_series(source: str, calls: int, *, graph: GraphState = GRAPH) -> _Series:
    wall: list[float] = []
    rss: list[int] = []
    ages: list[float] = []
    api_calls: list[int] = []
    with IsolatedPolicyWorkerV1(source) as worker:
        for ordinal in range(calls):
            result = worker.invoke(
                context=_context(ordinal),
                graph=graph,
                rewrite_host=HOST,
                seed=20260807,
            )
            if result.failure is not None:
                raise RuntimeError(result.failure.as_dict())
            wall.append(result.wall_seconds)
            rss.append(result.worker_rss_kib)
            ages.append(cast(float, worker.telemetry()["worker_age_seconds"]))
            api_calls.append(len(result.semantic_trace))
    return _Series(wall, rss, ages, api_calls)


def run_benchmark(
    *,
    startup_runs: int,
    call_runs: int,
    failure_runs: int,
) -> dict[str, object]:
    limits = PolicyRuntimeLimitsV1()
    startup: list[float] = []
    startup_rss: list[int] = []
    for _ in range(startup_runs):
        with IsolatedPolicyWorkerV1(NO_PLAN_SOURCE) as worker:
            telemetry = worker.telemetry()
            startup.append(cast(float, telemetry["startup_seconds"]))
            startup_rss.append(cast(int, telemetry["worker_rss_kib"]))

    normal = _call_series(NO_PLAN_SOURCE, call_runs)
    case_counts = [call_runs // len(SELECTOR_ACTION_CASES)] * len(SELECTOR_ACTION_CASES)
    for index in range(call_runs % len(SELECTOR_ACTION_CASES)):
        case_counts[index] += 1
    selector_action_cases: dict[str, dict[str, float | int]] = {}
    selector_action_series: list[_Series] = []
    maximum_graph_order = 0
    for (name, source, graph), count in zip(
        SELECTOR_ACTION_CASES,
        case_counts,
        strict=True,
    ):
        if count == 0:
            continue
        series = _call_series(source, count, graph=graph)
        selector_action_series.append(series)
        selector_action_cases[name] = {
            "calls": count,
            "graph_order": graph.order,
            **_distribution(series.wall_seconds),
        }
        maximum_graph_order = max(maximum_graph_order, graph.order)
    selector_action = _Series(
        wall_seconds=[
            value for series in selector_action_series for value in series.wall_seconds
        ],
        rss_kib=[value for series in selector_action_series for value in series.rss_kib],
        worker_age_seconds=[
            value
            for series in selector_action_series
            for value in series.worker_age_seconds
        ],
        api_calls=[
            value for series in selector_action_series for value in series.api_calls
        ],
    )

    failure_wall: list[float] = []
    failure_rss: list[int] = []
    for ordinal in range(failure_runs):
        started = time.monotonic()
        with IsolatedPolicyWorkerV1(FAILURE_SOURCE) as worker:
            result = worker.invoke(
                context=_context(ordinal),
                graph=GRAPH,
                rewrite_host=HOST,
                seed=20260807,
            )
            if result.failure is None or result.failure.code != "INVALID_RETURN":
                raise RuntimeError("failure benchmark did not produce INVALID_RETURN")
            failure_rss.append(result.worker_rss_kib)
        failure_wall.append(time.monotonic() - started)

    maximum_call_seconds = max((*normal.wall_seconds, *selector_action.wall_seconds))
    maximum_rss_kib = max(
        (*startup_rss, *normal.rss_kib, *selector_action.rss_kib, *failure_rss)
    )
    maximum_worker_age = max((*normal.worker_age_seconds, *selector_action.worker_age_seconds))
    maximum_api_calls = max((*normal.api_calls, *selector_action.api_calls))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "sample_counts": {
            "startup": startup_runs,
            "normal_calls": call_runs,
            "selector_action_calls": call_runs,
            "failure_paths": failure_runs,
        },
        "provisional_limits": limits.as_dict(),
        "measurements": {
            "startup": _distribution(startup),
            "normal_call": _distribution(normal.wall_seconds),
            "selector_action_call": _distribution(selector_action.wall_seconds),
            "selector_action_matrix": selector_action_cases,
            "invalid_return_failure_total": _distribution(failure_wall),
            "maximum_worker_rss_kib": maximum_rss_kib,
            "maximum_worker_age_seconds": maximum_worker_age,
            "maximum_api_calls": maximum_api_calls,
        },
        "headroom": {
            "propose_wall_seconds": limits.propose_wall_seconds - maximum_call_seconds,
            "address_space_bytes": limits.address_space_bytes - maximum_rss_kib * 1_024,
            "worker_lifetime_seconds": limits.worker_lifetime_seconds - maximum_worker_age,
            "api_calls": limits.total_api_calls - maximum_api_calls,
            "graph_order": limits.graph_order - maximum_graph_order,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--startup-runs", type=int, default=20)
    parser.add_argument("--call-runs", type=int, default=200)
    parser.add_argument("--failure-runs", type=int, default=20)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if min(arguments.startup_runs, arguments.call_runs, arguments.failure_runs) < 1:
        raise SystemExit("all run counts must be positive")
    report = run_benchmark(
        startup_runs=arguments.startup_runs,
        call_runs=arguments.call_runs,
        failure_runs=arguments.failure_runs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
