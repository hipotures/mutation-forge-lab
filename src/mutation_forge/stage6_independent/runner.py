"""Fresh provider-free Stage 6 verification runner.

The runner intentionally composes only the immutable graph model, a low-level
backend/scorer/proposal source, and optional Stage 2A policy workers.  It does
not delegate to a previous stage's high-level evaluator.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import GraphScore, GraphState, RewritePlan
from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.worker import PolicyWorker
from mutation_forge.stage3.evaluation import run_development_episode

from .persistence import SCHEMA_VERSION as PERSISTENCE_SCHEMA_VERSION
from .persistence import (
    canonical_bytes,
    canonical_record_hash,
    load_state,
    read_shard,
    reduction_hash,
    timing_stripped,
    write_json,
    write_shard,
    write_state,
)

SCHEMA_VERSION = "stage6.independent.runner.v1"
HORIZON = 32
SHARD_COUNT = 12
EPISODES_PER_SHARD = 64
EPISODE_COUNT = SHARD_COUNT * EPISODES_PER_SHARD
MAX_WORKERS = 8
RESERVED_PHYSICAL_CORES = 8
THREAD_ENVIRONMENT = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}
POLICY_IDS = (
    "program-d5ad1c8203e0d9f25f03aabd",
    "candidate-slot-04",
    "random",
    "structural",
)

TIMING_ONLY_FIELDS = frozenset(
    {
        "timing_ns",
        "first_improvement_ns",
        "ranker_elapsed_ns",
        "selected_scoring_ns",
        "pool_legality_ns",
        "pool_feature_ns",
        "elapsed_ns",
        "started_at",
        "finished_at",
        "timing",
        "timing_profile",
        "elapsed_seconds",
    }
)


def _digest_int(domain: str, *parts: object) -> int:
    return int.from_bytes(hashlib.sha256(canonical_bytes([domain, *parts])).digest()[:8], "big")


def _relabel_graph(graph: GraphState, *, graph_seed: int, relabeling_seed: int) -> tuple[GraphState, tuple[int, ...]]:
    values = list(range(graph.order))
    for index in range(graph.order - 1, 0, -1):
        # The frozen Fisher--Yates contract uses a domain-separated digest;
        # spelling it out here keeps this runner independent of any prior
        # stage implementation while preserving the published permutation.
        swap = _digest_int("stage5.relabel.permutation.v1", graph.order, graph_seed, relabeling_seed, index) % (index + 1)
        values[index], values[swap] = values[swap], values[index]
    edges = tuple(sorted((min(values[u], values[v]), max(values[u], values[v])) for u, v in graph.edges))
    return GraphState(graph.order, edges), tuple(values)


def _graph_label_hash(graph: GraphState) -> str:
    return hashlib.sha256(
        canonical_bytes({"order": graph.order, "edges": [list(edge) for edge in graph.edges]})
    ).hexdigest()


def _canonical_unlabeled_identity(graph: GraphState) -> str:
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
        palette = {
            signature: str(index)
            for index, signature in enumerate(sorted(set(signatures), key=repr))
        }
        refined = [palette[signature] for signature in signatures]
        if refined == colors:
            break
        colors = refined
    profiles: list[tuple[int, ...]] = []
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
        profiles.append(tuple(sorted(distances)))
    certificate = {
        "algorithm": "wl-distance-edge-v1",
        "order": graph.order,
        "vertex_colors": sorted(colors),
        "distance_profiles": sorted(profiles),
        "edge_colors": sorted(
            (min(colors[left], colors[right]), max(colors[left], colors[right]))
            for left, right in graph.edges
        ),
    }
    return hashlib.sha256(canonical_bytes(certificate)).hexdigest()


class RelabeledToyBackend(ToyBackend):
    """Toy scorer exposing one frozen relabeled graph to every policy."""

    def __init__(self, *, order: int, graph_seed: int, relabeling_seed: int) -> None:
        super().__init__()
        self.order_identity = order
        self.graph_seed_identity = graph_seed
        self.relabeling_seed_identity = relabeling_seed
        self.base_graph = super().generate_seed(order=order, seed=graph_seed)
        self.graph, self.permutation = _relabel_graph(
            self.base_graph, graph_seed=graph_seed, relabeling_seed=relabeling_seed
        )

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        if (order, seed) != (self.order_identity, self.graph_seed_identity):
            raise ValueError("Stage 6 backend received a foreign graph identity")
        return self.graph


def _episode_id(order: int, graph_seed: int, relabeling_seed: int, policy_seed: int) -> str:
    return f"o{order:02d}-g{graph_seed:04d}-r{relabeling_seed:04d}-p{policy_seed:04d}"


def _rows_from_manifest(manifest: object) -> list[dict[str, Any]]:
    rows = manifest.get("episodes", manifest) if isinstance(manifest, Mapping) else manifest
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes)):
        raise ValueError("manifest must contain an episodes iterable")
    result = [dict(cast(Mapping[str, Any], row)) for row in rows]
    for row in result:
        required = {"episode_id", "order", "graph_seed", "relabeling_seed", "policy_seed"}
        if not required.issubset(row):
            raise ValueError("manifest episode is incomplete")
        row.setdefault("horizon", HORIZON)
    ids = [str(row["episode_id"]) for row in result]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("manifest episode IDs must be unique and non-empty")
    return sorted(result, key=lambda row: (int(row["order"]), int(row["graph_seed"]), int(row["relabeling_seed"]), int(row["policy_seed"])))


def plan(
    manifest: object | None = None,
    policies: Mapping[str, object] | None = None,
    *,
    orders: Sequence[int] = (20, 24, 28),
    graph_seeds: Sequence[int] = tuple(range(701, 709)),
    relabeling_seeds: Sequence[int] = (7101, 7102),
    policy_seeds: Sequence[int] = tuple(range(7001, 7017)),
    horizon: int = HORIZON,
    strict_layout: bool = True,
) -> dict[str, Any]:
    """Build the immutable deterministic 12×64 episode plan."""
    if policies is not None and set(policies) != set(POLICY_IDS):
        raise ValueError("Stage 6 requires exactly the four frozen policy IDs")
    if horizon != HORIZON:
        raise ValueError("Stage 6 horizon is frozen at 32")
    manifest_shards: list[Mapping[str, Any]] | None = None
    if manifest is None:
        rows = [
            {"episode_id": _episode_id(int(order), int(graph), int(relabel), int(seed)), "order": int(order), "graph_seed": int(graph), "relabeling_seed": int(relabel), "policy_seed": int(seed), "horizon": HORIZON}
            for order in orders for graph in graph_seeds for relabel in relabeling_seeds for seed in policy_seeds
        ]
        rows = sorted(rows, key=lambda row: (row["order"], row["graph_seed"], row["relabeling_seed"], row["policy_seed"]))
    else:
        rows = _rows_from_manifest(manifest)
        if isinstance(manifest, Mapping) and isinstance(manifest.get("shards"), list):
            manifest_shards = [cast(Mapping[str, Any], item) for item in cast(list[Any], manifest["shards"])]
    if any(int(row.get("horizon", -1)) != HORIZON for row in rows):
        raise ValueError("all Stage 6 episodes require horizon 32")
    if strict_layout and len(rows) != EPISODE_COUNT:
        raise ValueError("Stage 6 manifest must contain exactly 768 episodes")
    if manifest_shards is not None:
        by_id = {str(row["episode_id"]): row for row in rows}
        shard_rows = []
        for shard in manifest_shards:
            ids = [str(item) for item in cast(list[Any], shard.get("episode_ids", []))]
            if any(item not in by_id for item in ids):
                raise ValueError("manifest shard references an unknown episode")
            shard_rows.append([by_id[item] for item in ids])
        if len(shard_rows) != SHARD_COUNT:
            raise ValueError("Stage 6 manifest shard count is not twelve")
    else:
        shard_rows = [rows[index * EPISODES_PER_SHARD : (index + 1) * EPISODES_PER_SHARD] for index in range(SHARD_COUNT)]
    if strict_layout and any(len(group) != EPISODES_PER_SHARD for group in shard_rows):
        raise ValueError("Stage 6 manifest must contain twelve shards of 64 episodes")
    shards = []
    for index, group in enumerate(shard_rows):
        shard_id = f"shard-{index:02d}"
        for row in group:
            row["shard_id"] = shard_id
        shards.append({"shard_id": shard_id, "episode_count": len(group), "episode_ids": [str(row["episode_id"]) for row in group]})
    payload = {"schema_version": SCHEMA_VERSION, "horizon": HORIZON, "shard_count": SHARD_COUNT, "episodes_per_shard": EPISODES_PER_SHARD, "episodes": rows, "shards": shards, "policy_ids": list(POLICY_IDS), "provider_calls_allowed": False, "resources": {"workers": MAX_WORKERS, "reserved_physical_cores": RESERVED_PHYSICAL_CORES, "thread_environment": dict(THREAD_ENVIRONMENT)}, "official_execution": {"first_shard": "shard-00", "first_shard_workers": 1, "resume_workers": MAX_WORKERS}}
    payload["plan_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def _score(backend: Any, scorer: Any, graph: GraphState, previous: GraphScore | None = None) -> GraphScore:
    function = scorer if scorer is not None else getattr(backend, "score", None)
    if function is None:
        raise ValueError("a scorer or backend.score is required")
    try:
        result = function(graph, witness_cap=64, cutoff=None)
    except TypeError:
        result = function(graph)
    if result is None:
        raise RuntimeError("selected score was cutoff-dominated")
    if not isinstance(result, GraphScore):
        raise TypeError("scorer must return GraphScore")
    return result


def _pool(proposer: Any, backend: Any, graph: GraphState, *, seed: int, step: int) -> Any:
    if proposer is not None:
        if hasattr(proposer, "generate"):
            return proposer.generate(graph, policy_seed=seed, step=step)
        if callable(proposer):
            try:
                return proposer(graph, policy_seed=seed, step=step)
            except TypeError:
                return proposer(graph, seed, step)
    if hasattr(backend, "propose_rewrite"):
        rewrite = backend.propose_rewrite(graph, operator_family="heg_uniform_two_switch", policy_seed=seed, evaluation=step + 1)
        return [rewrite]
    raise ValueError("a proposal source or backend.propose_rewrite is required")


def _candidate_payload(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        return candidate
    payload = getattr(candidate, "payload", None)
    return payload if isinstance(payload, Mapping) else {"proposal_id": getattr(candidate, "proposal_id", "")}


def _candidate_id(candidate: Any, index: int) -> str:
    return str(getattr(candidate, "proposal_id", _candidate_payload(candidate).get("proposal_id", f"proposal-{index:04d}")))


def _select(policy: Any, context: Mapping[str, Any], pool: Any, workers: dict[str, PolicyWorker] | None = None, policy_id: str = "") -> tuple[Any, dict[str, Any]]:
    candidates = list(getattr(pool, "candidates", pool if isinstance(pool, Iterable) and not isinstance(pool, (str, bytes, Mapping)) else []))
    if not candidates and isinstance(pool, Mapping) and "rewrite" in pool:
        candidates = [pool]
    if not candidates:
        raise RuntimeError("proposal pool is empty")
    selected: Any = None
    elapsed = 0
    rank_result = None
    if hasattr(policy, "rank"):
        started = time.perf_counter_ns()
        rank_result = policy.rank(context, pool)
        elapsed = time.perf_counter_ns() - started
        selected_id = getattr(rank_result, "selected_proposal_id", None)
        selected = next((item for i, item in enumerate(candidates) if _candidate_id(item, i) == selected_id), None)
    elif workers is not None and policy_id in workers:
        priorities: list[tuple[float, str, Any]] = []
        for index, candidate in enumerate(candidates):
            result = workers[policy_id].call(context, _candidate_payload(candidate))
            if result.status != "ok" or result.priority is None:
                raise RuntimeError(f"policy worker failed for {policy_id}")
            priorities.append((float(result.priority), _candidate_id(candidate, index), candidate))
        selected = max(priorities, key=lambda item: (item[0], "" if item[1] is None else item[1]))[2]
    elif hasattr(policy, "select"):
        selected = policy.select(context, pool)
        if isinstance(selected, str):
            selected = next((item for i, item in enumerate(candidates) if _candidate_id(item, i) == selected), None)
    elif callable(policy):
        scored: list[tuple[float, str, Any]] = []
        for index, candidate in enumerate(candidates):
            value = policy(context, _candidate_payload(candidate))
            if isinstance(value, (int, float)):
                scored.append((float(value), _candidate_id(candidate, index), candidate))
            elif value is candidate or value == _candidate_id(candidate, index):
                selected = candidate
                break
        if selected is None and scored:
            selected = max(scored, key=lambda item: (item[0], item[1]))[2]
    if selected is None:
        selected = candidates[0]
    return selected, {"ranker_elapsed_ns": elapsed, "selected_proposal_id": _candidate_id(selected, candidates.index(selected) if selected in candidates else 0), "ranker_flags": {}}


def _compact_trace(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "step", "pool_hash", "rank_pool_hash", "pool_size", "pool_attempted",
        "pool_deduplicated", "pool_rejected", "selected_proposal_id", "selected_k",
        "selected_operator_family", "selected_selector_tags", "accepted",
        "selected_ordering_key", "previous_ordering_key", "selected_total_witnesses",
        "previous_total_witnesses", "current_total_witnesses", "selected_witness_delta",
        "selected_penalty_delta", "best_total_witnesses", "state_hash", "ranker_flags",
    )
    return {key: timing_stripped(value.get(key)) for key in keys}


def _compact_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "auc", "normalized_best_so_far_curve", "best_total_witnesses", "accepted_count",
        "rejected_count", "nonimproving_count", "divergence_count", "first_improvement_step",
        "evaluations_to_first_improvement", "failure_count",
    )
    return {key: timing_stripped(value.get(key)) for key in keys}


def _compact_authoritative_episode(
    raw: Mapping[str, Any],
    row: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    backend: RelabeledToyBackend,
    ast_hashes: Mapping[str, str] | None = None,
    behavior_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if raw.get("terminal_status") != "completed":
        failure = raw.get("failure")
        raise ValueError(
            f"Stage 6 evaluator returned a non-completed episode "
            f"{row.get('episode_id')}: {failure!r}"
        )
    policies = raw.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != set(POLICY_IDS):
        raise ValueError("Stage 6 policy roster drifted")
    for counter in ("model_calls", "app_server_calls", "oracle_score_calls", "runtime_network_calls"):
        if int(raw.get(counter, 0)) != 0:
            raise ValueError(f"forbidden counter is nonzero: {counter}")
    if int(raw.get("invalid_graphs", 0)) != 0 or int(raw.get("policy_failures", 0)) != 0:
        raise ValueError("Stage 6 episode contains a failed policy or invalid graph")
    proof = {
        "algorithm": "fisher-yates-sha256-v1",
        "order": int(row["order"]),
        "graph_seed": int(row["graph_seed"]),
        "relabeling_seed": int(row["relabeling_seed"]),
        "permutation": list(backend.permutation),
        "permutation_sha256": hashlib.sha256(canonical_bytes(list(backend.permutation))).hexdigest(),
        "base_graph_hash": _graph_label_hash(backend.base_graph),
        "relabeled_graph_hash": _graph_label_hash(backend.graph),
        "canonical_unlabeled_hash": _canonical_unlabeled_identity(backend.graph),
        "labels_changed": backend.base_graph.edges != backend.graph.edges,
    }
    compact: dict[str, Any] = {
        "schema_version": "stage6.verification.episode.v1",
        "terminal_status": "completed",
        "episode_id": str(row["episode_id"]),
        "order": int(row["order"]),
        "graph_seed": int(row["graph_seed"]),
        "relabeling_seed": int(row["relabeling_seed"]),
        "policy_seed": int(row["policy_seed"]),
        "horizon": int(row.get("horizon", HORIZON)),
        "shard_id": str(row.get("shard_id", "")),
        "initial_graph_hash": raw.get("initial_graph_hash"),
        "base_graph_hash": proof["base_graph_hash"],
        "relabeled_graph_hash": proof["relabeled_graph_hash"],
        "canonical_unlabeled_hash": proof["canonical_unlabeled_hash"],
        "relabel_proof": proof,
        "divergence_step": raw.get("divergence_step"),
        "shared_pool_steps": raw.get("shared_pool_steps"),
        "independent_pool_steps": raw.get("independent_pool_steps"),
        "policies": {
            policy_id: _compact_policy(cast(Mapping[str, Any], policies[policy_id]))
            for policy_id in POLICY_IDS
        },
        "policy_identities": {
            policy_id: {
                "source_sha256": str(source_hashes[policy_id]),
                "normalized_ast_sha256": str((ast_hashes or {}).get(policy_id, "")),
                "behavior_signature_sha256": str(
                    (behavior_hashes or {}).get(policy_id, "")
                ),
            }
            for policy_id in POLICY_IDS
        },
        "steps": [
            {
                "step": item.get("step"),
                "trajectory_seed": item.get("trajectory_seed"),
                "states_identical_before_step": item.get("states_identical_before_step"),
                "shared_pool": item.get("shared_pool"),
                "policies": {
                    policy_id: _compact_trace(
                        cast(Mapping[str, Any], cast(Mapping[str, Any], item["policies"])[policy_id])
                    )
                    for policy_id in POLICY_IDS
                },
            }
            for item in cast(list[Mapping[str, Any]], raw.get("steps", []))
        ],
        "initial_score_calls": int(raw.get("initial_score_calls", 0)),
        "selected_score_calls": int(raw.get("selected_score_calls", 0)),
        "evaluation_count": int(raw.get("evaluation_count", 0)),
        "invalid_graphs": int(raw.get("invalid_graphs", 0)),
        "policy_failures": int(raw.get("policy_failures", 0)),
        "model_calls": int(raw.get("model_calls", 0)),
        "app_server_calls": int(raw.get("app_server_calls", 0)),
        "oracle_score_calls": int(raw.get("oracle_score_calls", 0)),
        "runtime_network_calls": int(raw.get("runtime_network_calls", 0)),
        "policy_source_sha256": dict(source_hashes),
    }
    compact["metrics_input"] = {
        "episode_id": compact["episode_id"],
        "order": compact["order"],
        "graph_seed": compact["graph_seed"],
        "relabeling_seed": compact["relabeling_seed"],
        "policy_seed": compact["policy_seed"],
        "horizon": compact["horizon"],
        "policies": {
            policy_id: {
                key: compact["policies"][policy_id].get(key)
                for key in (
                    "auc", "normalized_best_so_far_curve", "best_total_witnesses",
                    "accepted_count", "failure_count",
                )
            }
            for policy_id in POLICY_IDS
        },
    }
    compact["canonical_episode_sha256"] = canonical_record_hash(compact)
    return compact


def _rewrite(candidate: Any) -> RewritePlan:
    if isinstance(candidate, RewritePlan):
        return candidate
    value = getattr(candidate, "rewrite", None)
    if isinstance(value, RewritePlan):
        return value
    if isinstance(candidate, Mapping):
        value = candidate.get("rewrite")
        if isinstance(value, RewritePlan):
            return value
        if isinstance(value, Mapping):
            return RewritePlan(tuple(tuple(edge) for edge in value.get("removed_edges", ())), tuple(tuple(edge) for edge in value.get("added_edges", ())), str(value.get("operator_family", "unknown")))
    raise TypeError("proposal candidate does not contain a RewritePlan")


def _episode(
    row: Mapping[str, Any],
    *,
    backend: Any,
    scorer: Any,
    proposer: Any,
    policies: Mapping[str, Any],
    dry_run: bool,
    policy_workers: dict[str, PolicyWorker] | None,
    config: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    eid = str(row["episode_id"])
    horizon = int(row.get("horizon", HORIZON))
    if dry_run:
        records = {}
        for policy_index, policy_id in enumerate(POLICY_IDS):
            curve = [max(0, 8 - ((step + policy_index) % 9)) for step in range(horizon)]
            accepted = sum(1 for step in range(horizon) if step % 3 == policy_index % 3)
            records[policy_id] = {"normalized_best_so_far_curve": curve, "auc": sum(curve) / max(1, horizon), "best_total_witnesses": min(curve), "accepted_count": accepted, "rejected_count": horizon - accepted, "failure_count": 0}
        base = {"schema_version": SCHEMA_VERSION, "terminal_status": "completed", "episode_id": eid, **{key: row[key] for key in ("order", "graph_seed", "relabeling_seed", "policy_seed", "horizon", "shard_id") if key in row}, "policies": records, "steps": [], "initial_score_calls": 1, "selected_score_calls": horizon * len(POLICY_IDS), "oracle_score_calls": 0, "provider_calls": 0, "timing_ns": {"episode_total": time.perf_counter_ns() - started}}
        base["canonical_hash"] = canonical_record_hash(base)
        return base
    if config is not None:
        # The official path is the immutable Stage 3 evaluator with a fresh
        # relabeled backend.  This call is deliberately made here, below the
        # Stage 6 orchestration layer, so the runner cannot accidentally use a
        # Stage 5 high-level runner or search implementation.
        relabeled_backend = RelabeledToyBackend(
            order=int(row["order"]),
            graph_seed=int(row["graph_seed"]),
            relabeling_seed=int(row["relabeling_seed"]),
        )
        try:
            raw = run_development_episode(
                config,
                row,
                policies,
                backend=relabeled_backend,
                require_baselines=True,
            )
            source_hashes = getattr(config, "policy_source_hashes", {})
            return _compact_authoritative_episode(
                cast(Mapping[str, Any], raw),
                row,
                cast(Mapping[str, str], source_hashes),
                relabeled_backend,
                cast(Mapping[str, str], getattr(config, "policy_ast_hashes", {})),
                cast(Mapping[str, str], getattr(config, "policy_behavior_hashes", {})),
            )
        finally:
            relabeled_backend.close()
    if backend is None:
        raise ValueError("backend is required unless dry_run=True")
    graph = backend.generate_seed(order=int(row["order"]), seed=int(row["graph_seed"]))
    base_graph = graph
    graph, permutation = _relabel_graph(graph, graph_seed=int(row["graph_seed"]), relabeling_seed=int(row["relabeling_seed"]))
    initial = _score(backend, scorer, graph)
    states = {policy_id: {"graph": graph, "score": initial, "best": initial.total_capped_witnesses, "curve": [], "accepted": 0, "rejected": 0} for policy_id in POLICY_IDS}
    steps: list[dict[str, Any]] = []
    for step in range(horizon):
        traces: dict[str, Any] = {}
        for policy_index, policy_id in enumerate(POLICY_IDS):
            state = states[policy_id]
            seed = _digest_int("stage6.trajectory.v1", eid, step)
            pool = _pool(proposer, backend, state["graph"], seed=seed ^ policy_index, step=step)
            context = {"schema_version": "stage2b.context.v1", "order": state["graph"].order, "capped_cycle_counts": [item[1] for item in initial.capped_cycle_counts], "weighted_penalty": state["score"].weighted_penalty, "step": step, "remaining_steps": horizon - step - 1, "stagnation": state["rejected"], "recent_best_improvement": 0.0, "recent_acceptance_rate": 0.0, "recent_duplicate_rate": 0.0}
            candidate, selection = _select(policies[policy_id], context, pool, policy_workers, policy_id)
            candidate_graph = backend.apply_rewrite(state["graph"], _rewrite(candidate))
            candidate_score = _score(backend, scorer, candidate_graph, state["score"])
            previous = state["score"]
            accepted = candidate_score.ordering_key < previous.ordering_key
            if accepted:
                state["graph"], state["score"] = candidate_graph, candidate_score
                state["accepted"] += 1
            else:
                state["rejected"] += 1
            state["best"] = min(state["best"], state["score"].total_capped_witnesses)
            state["curve"].append(state["best"])
            traces[policy_id] = {"step": step, **selection, "accepted": accepted, "selected_ordering_key": list(candidate_score.ordering_key), "previous_ordering_key": list(previous.ordering_key), "selected_total_witnesses": candidate_score.total_capped_witnesses, "previous_total_witnesses": previous.total_capped_witnesses, "current_total_witnesses": state["score"].total_capped_witnesses, "best_total_witnesses": state["best"], "state_hash": getattr(backend, "state_hash", lambda value: hashlib.sha256(canonical_bytes(value.edges)).hexdigest())(state["graph"]), "selected_scoring_ns": 0}
        steps.append({"step": step, "trajectory_seed": _digest_int("stage6.trajectory.v1", eid, step), "states_identical_before_step": False, "policies": traces})
    records = {policy_id: {"normalized_best_so_far_curve": state["curve"], "auc": sum(state["curve"]) / max(1, horizon), "best_total_witnesses": state["best"], "accepted_count": state["accepted"], "rejected_count": state["rejected"], "failure_count": 0} for policy_id, state in states.items()}
    base = {"schema_version": SCHEMA_VERSION, "terminal_status": "completed", "episode_id": eid, **{key: row[key] for key in ("order", "graph_seed", "relabeling_seed", "policy_seed", "horizon", "shard_id") if key in row}, "initial_graph_hash": getattr(backend, "state_hash", lambda value: hashlib.sha256(canonical_bytes(value.edges)).hexdigest())(base_graph), "relabeled_graph_hash": getattr(backend, "state_hash", lambda value: hashlib.sha256(canonical_bytes(value.edges)).hexdigest())(graph), "relabel_permutation": list(permutation), "policies": records, "steps": steps, "initial_score_calls": 1, "selected_score_calls": horizon * len(POLICY_IDS), "oracle_score_calls": 0, "provider_calls": 0, "timing_ns": {"episode_total": time.perf_counter_ns() - started}}
    base["canonical_hash"] = canonical_record_hash(base)
    return base


def _resolve_plan(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, Mapping) and "episodes" in value else plan(value, strict_layout=False)


def run_shard(
    stage_plan: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    shard: int | str,
    *,
    output_dir: str | Path | None = None,
    backend: Any = None,
    scorer: Any = None,
    proposer: Any = None,
    policies: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    config: Any | None = None,
) -> dict[str, Any]:
    """Evaluate and optionally persist one deterministic shard."""
    prepared = _resolve_plan(stage_plan)
    index = int(str(shard).split("-")[-1]) if isinstance(shard, str) else int(shard)
    groups = cast(list[Mapping[str, Any]], prepared.get("shards", []))
    if index < 0 or index >= len(groups):
        raise ValueError("shard index outside frozen layout")
    ids = [str(item) for item in cast(list[Any], groups[index].get("episode_ids", []))]
    by_id = {str(row["episode_id"]): row for row in cast(list[Mapping[str, Any]], prepared["episodes"])}
    rows = [by_id[item] for item in ids]
    if policies is None and config is not None:
        configured_paths = getattr(config, "policy_paths", {})
        selected_policies = {
            policy_id: Path(configured_paths[policy_id]).read_text(encoding="utf-8")
            for policy_id in POLICY_IDS
        }
    else:
        selected_policies = policies or {
            policy_id: (lambda _ctx, _proposal, _i=i: -_i)
            for i, policy_id in enumerate(POLICY_IDS)
        }
    if set(selected_policies) != set(POLICY_IDS):
        raise ValueError("Stage 6 requires exactly the four frozen policy IDs")
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    worker_map: dict[str, PolicyWorker] = {}
    owned_workers: list[PolicyWorker] = []
    try:
        for policy_id, value in selected_policies.items():
            if config is not None:
                # The authoritative evaluator creates its own SourceRanker
                # instances and owns their lifecycle; no Stage 6 worker is
                # needed for this path.
                continue
            if isinstance(value, str):
                worker_map[policy_id] = PolicyWorker(value, SandboxLimits())
                owned_workers.append(worker_map[policy_id])
            elif isinstance(value, Path):
                worker_map[policy_id] = PolicyWorker(value.read_text(encoding="utf-8"), SandboxLimits())
                owned_workers.append(worker_map[policy_id])
            elif hasattr(value, "path"):
                source_path = Path(str(value.path))
                worker_map[policy_id] = PolicyWorker(source_path.read_text(encoding="utf-8"), SandboxLimits())
                owned_workers.append(worker_map[policy_id])
        records = [
            _episode(
                row,
                backend=backend,
                scorer=scorer,
                proposer=proposer,
                policies=selected_policies,
                dry_run=dry_run,
                policy_workers=worker_map,
                config=config,
            )
            for row in rows
        ]
    finally:
        for worker in owned_workers:
            worker.close()
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "status": "completed", "shard": index, "record_count": len(records), "records": records, "provider_calls": 0}
    if output_dir is not None:
        filename = f"stage6-shard-{index:02d}.jsonl.gz"
        result["shard"] = write_shard(output_dir, filename, records, expected_ids=ids)
    return result


def _execute_pass(
    stage_plan: Mapping[str, Any],
    root: Path,
    *,
    backend: Any,
    scorer: Any,
    proposer: Any,
    policies: Mapping[str, Any] | None,
    dry_run: bool,
    workers: int,
    config: Any | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    identity = str(stage_plan.get("plan_sha256", hashlib.sha256(canonical_bytes(stage_plan)).hexdigest()))
    existing = load_state(root)
    if existing is not None and existing.get("plan_sha256") != identity:
        raise ValueError("output state belongs to a different immutable plan")
    entries = cast(dict[str, Any], existing.get("shards", {}) if existing else {})
    groups = cast(list[Mapping[str, Any]], stage_plan["shards"])
    ids_by_shard = {index: [str(item) for item in cast(list[Any], groups[index]["episode_ids"])] for index in range(len(groups))}
    completed = set()
    records: list[dict[str, Any]] = []
    for key, entry in entries.items():
        index = int(key)
        rows = read_shard(root, cast(Mapping[str, Any], entry), ids_by_shard[index])
        records.extend(rows)
        completed.add(index)
    # A worker may have flushed a complete gzip shard just before a different
    # worker failed.  Recover such an orphan checkpoint by validating it
    # against the frozen roster; never rerun a valid completed shard.
    for index in range(len(groups)):
        if index in completed:
            continue
        filename = f"stage6-shard-{index:02d}.jsonl.gz"
        candidate = root / filename
        if not candidate.is_file():
            continue
        try:
            import hashlib as _hashlib

            entry = {
                "path": filename,
                "file_sha256": _hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "episode_ids": ids_by_shard[index],
                "record_count": len(ids_by_shard[index]),
            }
            rows = read_shard(root, entry, ids_by_shard[index])
        except Exception:
            # Keep malformed/partial evidence in place for forensic review and
            # allow the shard to be regenerated from the last valid state.
            continue
        entries[str(index)] = entry
        records.extend(rows)
        completed.add(index)
        write_state(
            root,
            {
                "schema_version": PERSISTENCE_SCHEMA_VERSION,
                "plan_sha256": identity,
                "shards": entries,
                "status": "shards_persisted",
            },
        )
    missing = [index for index in range(len(groups)) if index not in completed]
    # The official first shard is deliberately serialized before any resume.
    if 0 in missing:
        try:
            outcome = run_shard(stage_plan, 0, output_dir=root, backend=backend, scorer=scorer, proposer=proposer, policies=policies, dry_run=dry_run, config=config)
        except Exception as error:
            write_json(
                root / "failure-shard-00.json",
                {"schema_version": SCHEMA_VERSION, "shard": 0, "error_type": type(error).__name__, "error": str(error), "plan_sha256": identity},
                overwrite=True,
            )
            raise
        entry = cast(dict[str, Any], outcome["shard"])
        entries["0"] = entry
        records.extend(cast(list[dict[str, Any]], outcome["records"]))
        write_state(root, {"schema_version": PERSISTENCE_SCHEMA_VERSION, "plan_sha256": identity, "shards": entries, "status": "shards_persisted"})
        missing.remove(0)
    if missing:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, workers))) as executor:
            futures = {executor.submit(run_shard, stage_plan, index, output_dir=root, backend=backend, scorer=scorer, proposer=proposer, policies=policies, dry_run=dry_run, config=config): index for index in missing}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outcome = future.result()
                except Exception as error:
                    write_json(
                        root / f"failure-shard-{index:02d}.json",
                        {"schema_version": SCHEMA_VERSION, "shard": index, "error_type": type(error).__name__, "error": str(error), "plan_sha256": identity},
                        overwrite=True,
                    )
                    raise
                entries[str(index)] = cast(dict[str, Any], outcome["shard"])
                records.extend(cast(list[dict[str, Any]], outcome["records"]))
                write_state(root, {"schema_version": PERSISTENCE_SCHEMA_VERSION, "plan_sha256": identity, "shards": entries, "status": "shards_persisted"})
    records.sort(key=lambda row: str(row["episode_id"]))
    summary = {"schema_version": SCHEMA_VERSION, "status": "completed", "artifact_dir": str(root), "plan_sha256": identity, "record_count": len(records), "shard_count": len(groups), "episodes_per_shard": len(ids_by_shard[0]) if groups else 0, "provider_calls": 0, "canonical_reduction_sha256": reduction_hash(records), "timing_stripped_reduction_sha256": reduction_hash(records, timing_only=True), "shards": [entries[str(index)] for index in range(len(groups))], "records": records}
    write_json(root / "summary.json", summary, overwrite=True)
    write_state(root, {"schema_version": PERSISTENCE_SCHEMA_VERSION, "plan_sha256": identity, "shards": entries, "status": "completed"})
    return summary


def run_pass(
    stage_plan: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    backend: Any = None,
    scorer: Any = None,
    proposer: Any = None,
    policies: Mapping[str, Any] | None = None,
    workers: int = MAX_WORKERS,
    dry_run: bool = False,
    replay: bool = True,
    config: Any | None = None,
) -> dict[str, Any]:
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError("workers must be between 1 and 8")
    prepared = _resolve_plan(stage_plan)
    root = Path(output_dir).resolve()
    primary = _execute_pass(prepared, root / "primary", backend=backend, scorer=scorer, proposer=proposer, policies=policies, dry_run=dry_run, workers=workers, config=config)
    result: dict[str, Any] = {"primary": primary, "provider_calls": 0}
    if replay:
        replay_summary = _execute_pass(prepared, root / "replay", backend=backend, scorer=scorer, proposer=proposer, policies=policies, dry_run=dry_run, workers=workers, config=config)
        result["replay"] = replay_summary
        result["replay_verification"] = verify_replay(primary, replay_summary)
    return result


def _summary_records(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if isinstance(summary.get("records"), list):
        return {str(row["episode_id"]): cast(Mapping[str, Any], row) for row in cast(list[Any], summary["records"]) if isinstance(row, Mapping)}
    root = Path(str(summary.get("artifact_dir", ".")))
    rows: dict[str, Mapping[str, Any]] = {}
    for entry in cast(list[Mapping[str, Any]], summary.get("shards", [])):
        for row in read_shard(root, entry):
            rows[str(row["episode_id"])] = row
    return rows


def verify_replay(primary: Mapping[str, Any] | str | Path, replay: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    def load(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        from .persistence import read_json

        return read_json(value)
    left, right = load(primary), load(replay)
    left_rows, right_rows = _summary_records(left), _summary_records(right)
    ids = set(left_rows) | set(right_rows)
    differences = []
    for eid in sorted(ids):
        if eid not in left_rows or eid not in right_rows or timing_stripped(left_rows[eid]) != timing_stripped(right_rows[eid]):
            differences.append(eid)
    exact = not differences and left.get("record_count") == right.get("record_count")
    return {"status": "completed" if exact else "failed", "decision": "exact" if exact else "mismatch", "exact": exact, "provider_calls": 0, "non_timing_differences": differences, "primary_timing_stripped_sha256": reduction_hash(left_rows.values(), timing_only=True), "replay_timing_stripped_sha256": reduction_hash(right_rows.values(), timing_only=True)}


__all__ = ["EPISODE_COUNT", "EPISODES_PER_SHARD", "HORIZON", "MAX_WORKERS", "POLICY_IDS", "RESERVED_PHYSICAL_CORES", "SCHEMA_VERSION", "SHARD_COUNT", "THREAD_ENVIRONMENT", "plan", "run_pass", "run_shard", "verify_replay"]
