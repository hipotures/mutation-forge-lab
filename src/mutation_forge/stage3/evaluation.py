"""Stage 3 deterministic development trajectories and reduction helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import GraphScore, GraphState, JsonValue
from mutation_forge.proposals.k_switch import (
    KSwitchPoolGenerator,
    ProposalCandidate,
    ProposalPool,
    make_scientific_context,
)
from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage2b.rankers import SourceRanker
from mutation_forge.stage3.manifest import canonical_bytes, sha256

STAGE3_EPISODE_VERSION = "stage3.development.episode.v1"
STAGE3_REDUCTION_VERSION = "stage3.development.reduction.v1"
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


def trajectory_seed(episode: Mapping[str, Any], step: int) -> int:
    payload = [
        "stage3.trajectory-seed.v1",
        episode["order"],
        episode["graph_seed"],
        episode["policy_seed"],
        step,
    ]
    return int.from_bytes(hashlib.sha256(canonical_bytes(payload)).digest()[:4], "big")


def _strip_timing(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    if isinstance(value, dict):
        return {
            k: _strip_timing(v)
            for k, v in value.items()
            if k not in {"started_at", "finished_at", "elapsed_seconds", "path"}
            and not k.endswith("_ns")
        }
    return value


def canonical_projection(record: Mapping[str, Any]) -> dict[str, JsonValue]:
    value = {
        k: v for k, v in record.items() if k not in {"canonical_episode_sha256", "canonical_hash"}
    }
    return cast(dict[str, JsonValue], _strip_timing(value))


@dataclass(slots=True)
class _State:
    graph: GraphState
    score: GraphScore
    initial_total: int
    best_total: int
    curve: list[float] = field(default_factory=list)
    raw_curve: list[int] = field(default_factory=list)
    trace: list[dict[str, JsonValue]] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    nonimproving: int = 0
    stagnation: int = 0
    divergences: int = 0
    first_improvement_step: int | None = None
    first_improvement_ns: int | None = None
    failures: int = 0
    recent_accepts: list[int] = field(default_factory=list)
    recent_improvements: list[float] = field(default_factory=list)
    recent_duplicate_rates: list[float] = field(default_factory=list)

    def context_rates(self) -> tuple[float, float, float]:
        """Return the same bounded controller telemetry used by Stage 2D."""
        window = 8
        accepts = self.recent_accepts[-window:]
        improvements = self.recent_improvements[-window:]
        duplicates = self.recent_duplicate_rates[-window:]
        return (
            max(improvements, default=0.0),
            sum(accepts) / len(accepts) if accepts else 0.0,
            sum(duplicates) / len(duplicates) if duplicates else 0.0,
        )


def _get(config: object, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _stage2b(config: object) -> Any:
    return _get(config, "stage2b", config)


def _candidate(pool: ProposalPool, selected: str | None) -> ProposalCandidate:
    if selected is None:
        raise RuntimeError("ranker did not select a proposal")
    for candidate in pool.candidates:
        if candidate.proposal_id == selected:
            return candidate
    raise RuntimeError("ranker selected a proposal outside its immutable pool")


def _pool_generator(backend: GraphBackend, config: object) -> KSwitchPoolGenerator:
    c = _stage2b(config)
    return KSwitchPoolGenerator(
        backend, pool_limits=_get(c, "pool"), feature_limits=_get(c, "features")
    )


def _apply(
    backend: GraphBackend,
    config: object,
    state: _State,
    ranker: Any,
    pool: ProposalPool,
    *,
    step: int,
    horizon: int,
    started_ns: int,
) -> dict[str, JsonValue]:
    c = _stage2b(config)
    improvement, acceptance_rate, duplicate_rate = state.context_rates()
    context = make_scientific_context(
        state.graph,
        state.score,
        forbidden_lengths=_get(_get(c, "features"), "forbidden_lengths", ()),
        step=step,
        remaining_steps=horizon - step - 1,
        stagnation=state.stagnation,
        recent_best_improvement=improvement,
        recent_acceptance_rate=acceptance_rate,
        recent_duplicate_rate=duplicate_rate,
    )
    rank = ranker.rank(context, pool)
    if rank.pool_hash != pool.pool_hash:
        state.failures += 1
        raise RuntimeError("ranker returned a foreign pool hash")
    if (
        getattr(rank, "exception", False)
        or getattr(rank, "timeout", False)
        or getattr(rank, "crash", False)
        or getattr(rank, "protocol", False)
    ):
        state.failures += 1
        raise RuntimeError(f"ranker failed: {getattr(rank, 'error', None)}")
    candidate = _candidate(pool, getattr(rank, "selected_proposal_id", None))
    candidate_graph = backend.apply_rewrite(state.graph, candidate.rewrite)
    validation = backend.validate(candidate_graph)
    if not validation.valid:
        state.failures += 1
        raise RuntimeError(f"invalid selected graph: {validation.errors}")
    scoring_started_ns = time.perf_counter_ns()
    score = backend.score(
        candidate_graph, witness_cap=int(_get(_get(c, "search"), "witness_cap", 64)), cutoff=None
    )
    selected_scoring_ns = time.perf_counter_ns() - scoring_started_ns
    if score is None:
        raise RuntimeError("selected score unexpectedly cutoff-dominated")
    previous = state.score
    accepted = score.ordering_key < state.score.ordering_key
    if accepted:
        state.graph, state.score = candidate_graph, score
        state.accepted += 1
        state.stagnation = 0
        if state.first_improvement_step is None:
            state.first_improvement_step = step
            state.first_improvement_ns = time.perf_counter_ns() - started_ns
    else:
        state.rejected += 1
        state.nonimproving += 1
        state.stagnation += 1
    state.best_total = min(state.best_total, state.score.total_capped_witnesses)
    state.raw_curve.append(state.best_total)
    state.curve.append((state.initial_total - state.best_total) / max(1, state.initial_total))
    state.duplicates += pool.deduplicated
    state.recent_accepts.append(int(accepted))
    state.recent_improvements.append(
        max(
            0.0,
            (previous.total_capped_witnesses - state.score.total_capped_witnesses)
            / max(1, state.initial_total),
        )
    )
    state.recent_duplicate_rates.append(pool.deduplicated / max(1, pool.attempted))
    trace: dict[str, JsonValue] = {
        "step": step,
        "pool_hash": pool.pool_hash,
        "rank_pool_hash": rank.pool_hash,
        "pool_size": pool.retained,
        "pool_attempted": pool.attempted,
        "pool_deduplicated": pool.deduplicated,
        "pool_rejected": cast(dict[str, JsonValue], pool.rejected),
        "selected_proposal_id": candidate.proposal_id,
        "selected_k": candidate.payload["k"],
        "selected_operator_family": candidate.payload["operator_family"],
        "selected_selector_tags": cast(list[JsonValue], candidate.payload["selector_tags"]),
        "accepted": accepted,
        "selected_score": score.as_dict(),
        "current_score": state.score.as_dict(),
        "previous_score": previous.as_dict(),
        "best_total_witnesses": state.best_total,
        "state_hash": backend.state_hash(state.graph),
        "ranker_elapsed_ns": getattr(rank, "elapsed_ns", 0),
        "ranker_flags": {
            "exception": bool(getattr(rank, "exception", False)),
            "timeout": bool(getattr(rank, "timeout", False)),
            "crash": bool(getattr(rank, "crash", False)),
            "protocol": bool(getattr(rank, "protocol", False)),
        },
        "selected_scoring_ns": selected_scoring_ns,
        "pool_legality_ns": pool.legality_elapsed_ns,
        "pool_feature_ns": pool.feature_elapsed_ns,
    }
    state.trace.append(trace)
    return trace


def _rankers(
    config: object, policies: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], list[SourceRanker]]:
    result: dict[str, Any] = {}
    owned: list[SourceRanker] = []
    for name, value in policies.items():
        if isinstance(value, SourceRanker):
            result[name] = value
        elif isinstance(value, str):
            sandbox = _get(_stage2b(config), "sandbox")
            if sandbox is None:
                raise ValueError("source policies require sandbox limits")
            validation = validate_policy(value, sandbox)
            if not validation.valid:
                raise ValueError(f"invalid policy source {name}")
            result[name] = SourceRanker(name, value, sandbox)
            owned.append(result[name])
        else:
            raise TypeError(f"policy {name} must be SourceRanker or source text")
    names = list(result)
    if "random" not in names or "structural" not in names:
        raise ValueError("frozen random and structural policies are required")
    if len(names) != len(set(names)):
        raise ValueError("policy IDs must be unique")
    identities = [getattr(result[name], "identity", None) for name in names]
    ast_hashes = [
        getattr(identity, "normalized_ast_sha256", None)
        for identity in identities
        if identity is not None
    ]
    if len(ast_hashes) != len(set(ast_hashes)):
        raise ValueError("duplicate normalized policy AST")
    return result, names, owned


def run_development_episode(
    config: object,
    episode: Mapping[str, Any],
    policies: Mapping[str, Any],
    *,
    backend: GraphBackend | None = None,
) -> dict[str, JsonValue]:
    owned = backend is None
    be = backend or ToyBackend()
    started = time.perf_counter_ns()
    rankers, names, owned_rankers = _rankers(config, policies)
    try:
        graph = be.generate_seed(order=int(episode["order"]), seed=int(episode["graph_seed"]))
        initial_validation = be.validate(graph)
        if not initial_validation.valid:
            failure: dict[str, JsonValue] = {
                "schema_version": STAGE3_EPISODE_VERSION,
                "episode_id": episode["episode_id"],
                "order": episode["order"],
                "graph_seed": episode["graph_seed"],
                "policy_seed": episode["policy_seed"],
                "horizon": int(episode.get("horizon", 32)),
                "terminal_status": "failure",
                "failure": {
                    "code": "invalid_initial_graph",
                    "message": str(initial_validation.errors),
                },
                "policies": {},
                "initial_score_calls": 0,
                "selected_score_calls": 0,
                "oracle_score_calls": 0,
                "evaluation_count": 0,
                "policy_failures": len(names),
                "invalid_graphs": 1,
                "model_calls": 0,
                "app_server_calls": 0,
                "runtime_network_calls": 0,
                "timing_ns": {"episode_total": time.perf_counter_ns() - started},
            }
            return {**failure, "canonical_episode_sha256": sha256(canonical_projection(failure))}
        initial = be.score(
            graph,
            witness_cap=int(_get(_get(_stage2b(config), "search"), "witness_cap", 64)),
            cutoff=None,
        )
        if initial is None:
            raise RuntimeError("initial score unavailable")
        states = {
            name: _State(
                graph, initial, initial.total_capped_witnesses, initial.total_capped_witnesses
            )
            for name in names
        }
        generator = _pool_generator(be, config)
        horizon = int(episode.get("horizon", 32))
        steps: list[dict[str, JsonValue]] = []
        divergence_step: int | None = None
        ever_diverged = False
        shared_steps = independent_steps = 0
        for step in range(horizon):
            seed = trajectory_seed(episode, step)
            pools: dict[str, ProposalPool] = {}
            by_state: dict[GraphState, ProposalPool] = {}
            # Once one policy accepts a different rewrite, each policy owns an
            # independent proposal pool thereafter.  Sharing is only allowed
            # while all current graphs are identical, matching Stage 2D.
            states_identical = len({state.graph for state in states.values()}) == 1
            share_pool = states_identical and not ever_diverged
            for name in names:
                state = states[name]
                pool = None if not share_pool else by_state.get(state.graph)
                if pool is None:
                    pool = generator.generate(state.graph, policy_seed=seed, step=step)
                    if share_pool:
                        by_state[state.graph] = pool
                pools[name] = pool
            if share_pool:
                shared_steps += 1
            else:
                independent_steps += len(pools)
                if divergence_step is None:
                    divergence_step = step
            traces: dict[str, JsonValue] = {}
            for name in names:
                traces[name] = _apply(
                    be,
                    config,
                    states[name],
                    rankers[name],
                    pools[name],
                    step=step,
                    horizon=horizon,
                    started_ns=started,
                )
            steps.append(
                {
                    "step": step,
                    "trajectory_seed": seed,
                    "states_identical_before_step": states_identical,
                    "shared_pool": share_pool,
                    "policies": traces,
                }
            )
            if len({state.graph for state in states.values()}) != 1:
                ever_diverged = True
        policy_records: dict[str, JsonValue] = {}
        for name, state in states.items():
            policy_records[name] = {
                "auc": sum(state.curve) / len(state.curve) if state.curve else 0.0,
                "raw_best_so_far_curve": cast(list[JsonValue], state.raw_curve),
                "normalized_best_so_far_curve": cast(list[JsonValue], state.curve),
                "best_total_witnesses": state.best_total,
                "accepted_count": state.accepted,
                "rejected_count": state.rejected,
                "nonimproving_count": state.nonimproving,
                "duplicate_count": state.duplicates,
                "divergence_count": 0
                if divergence_step is None
                else max(0, horizon - divergence_step),
                "first_improvement_step": state.first_improvement_step,
                "evaluations_to_first_improvement": None
                if state.first_improvement_step is None
                else state.first_improvement_step + 1,
                "first_improvement_ns": state.first_improvement_ns,
                "failure_count": state.failures,
                "initial_score": initial.as_dict(),
                "final_score": state.score.as_dict(),
                "best_score": state.score.as_dict(),
                "trace": cast(list[JsonValue], state.trace),
                "timings_ns": {"first_improvement": state.first_improvement_ns or 0},
                "resources": {"network_calls": 0, "model_calls": 0, "app_server_calls": 0},
            }
        identities: dict[str, JsonValue] = {}
        for name, ranker in rankers.items():
            identity = getattr(ranker, "identity", None)
            if identity is not None:
                identities[name] = cast(dict[str, JsonValue], identity.as_dict())
        base: dict[str, JsonValue] = {
            "schema_version": STAGE3_EPISODE_VERSION,
            "terminal_status": "completed",
            "episode_id": episode["episode_id"],
            "order": episode["order"],
            "graph_seed": episode["graph_seed"],
            "policy_seed": episode["policy_seed"],
            "horizon": horizon,
            "initial_graph_hash": be.state_hash(graph),
            "divergence_step": divergence_step,
            "shared_pool_steps": shared_steps,
            "independent_pool_steps": independent_steps,
            "policies": policy_records,
            "policy_identities": identities,
            "steps": cast(list[JsonValue], steps),
            "initial_score_calls": 1,
            "selected_score_calls": horizon * len(names),
            "oracle_score_calls": 0,
            "evaluation_count": horizon * len(names),
            "policy_failures": sum(s.failures for s in states.values()),
            "invalid_graphs": 0,
            "model_calls": 0,
            "app_server_calls": 0,
            "runtime_network_calls": 0,
            "timing_ns": {"episode_total": time.perf_counter_ns() - started},
        }
        return {**base, "canonical_episode_sha256": sha256(canonical_projection(base))}
    except Exception as error:
        failure = {
            "schema_version": STAGE3_EPISODE_VERSION,
            "episode_id": episode["episode_id"],
            "order": episode["order"],
            "graph_seed": episode["graph_seed"],
            "policy_seed": episode["policy_seed"],
            "horizon": int(episode.get("horizon", 32)),
            "terminal_status": "failure",
            "failure": {"code": "evaluation_failure", "message": str(error)},
            "policies": {},
            "initial_score_calls": 1,
            "selected_score_calls": 0,
            "oracle_score_calls": 0,
            "evaluation_count": 0,
            "policy_failures": len(names),
            "invalid_graphs": 0,
            "model_calls": 0,
            "app_server_calls": 0,
            "runtime_network_calls": 0,
            "timing_ns": {"episode_total": time.perf_counter_ns() - started},
        }
        return {**failure, "canonical_episode_sha256": sha256(canonical_projection(failure))}
    finally:
        for ranker in owned_rankers:
            ranker.close()
        if owned:
            be.close()


def validate_record(
    record: Mapping[str, Any],
    expected_policy_ids: set[str] | None = None,
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
    expected_ast_hashes: Mapping[str, str] | None = None,
) -> None:
    if record.get("schema_version") != STAGE3_EPISODE_VERSION:
        raise ValueError("unexpected episode schema")
    policies = record.get("policies")
    if not isinstance(policies, Mapping):
        raise ValueError("episode policies missing")
    terminal = record.get("terminal_status") == "failure"
    if terminal:
        raise ValueError("failed episode cannot enter the confirmatory reduction")
    if record.get("terminal_status") != "completed":
        raise ValueError("episode terminal status is not completed")
    if expected_policy_ids is not None and not terminal and set(policies) != expected_policy_ids:
        raise ValueError("policy roster mismatch")
    identities = record.get("policy_identities", {})
    if expected_source_hashes is not None:
        for policy, digest in expected_source_hashes.items():
            if cast(Mapping[str, Any], identities).get(policy, {}).get("source_sha256") != digest:
                raise ValueError("source identity hash mismatch")
    if expected_ast_hashes is not None:
        for policy, digest in expected_ast_hashes.items():
            if (
                cast(Mapping[str, Any], identities).get(policy, {}).get("normalized_ast_sha256")
                != digest
            ):
                raise ValueError("AST identity hash mismatch")
    expected = sha256(canonical_projection(record))
    if record.get("canonical_episode_sha256") != expected:
        raise ValueError("episode canonical hash mismatch")
    if int(record.get("selected_score_calls", -1)) != int(record.get("evaluation_count", -2)):
        raise ValueError("selected scoring accounting mismatch")
    policy_count = len(policies)
    horizon = int(record.get("horizon", -1))
    if int(record.get("initial_score_calls", -1)) != 1:
        raise ValueError("initial scoring accounting mismatch")
    if int(record.get("oracle_score_calls", -1)) != 0:
        raise ValueError("oracle scoring is forbidden")
    if int(record.get("selected_score_calls", -1)) != horizon * policy_count:
        raise ValueError("selected scoring budget mismatch")
    if int(record.get("invalid_graphs", -1)) != 0:
        raise ValueError("invalid graph reached an episode")
    for counter in ("model_calls", "app_server_calls", "runtime_network_calls"):
        if int(record.get(counter, -1)) != 0:
            raise ValueError(f"unexpected {counter}")
    steps = record.get("steps")
    if not isinstance(steps, list) or len(steps) != horizon:
        raise ValueError("trajectory step count mismatch")
    for step in steps:
        traces = step.get("policies") if isinstance(step, Mapping) else None
        if not isinstance(traces, Mapping) or set(traces) != set(policies):
            raise ValueError("trajectory policy roster mismatch")
        for trace in traces.values():
            if not isinstance(trace, Mapping) or trace.get("pool_hash") != trace.get(
                "rank_pool_hash"
            ):
                raise ValueError("ranker did not receive the recorded pool")


def reduce_records(
    records: list[Mapping[str, Any]],
    expected_episode_ids: set[str] | None = None,
    expected_policy_ids: set[str] | None = None,
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
    expected_ast_hashes: Mapping[str, str] | None = None,
) -> list[dict[str, JsonValue]]:
    seen: set[str] = set()
    reduced: list[dict[str, JsonValue]] = []
    for value in records:
        validate_record(
            value,
            expected_policy_ids,
            expected_source_hashes=expected_source_hashes,
            expected_ast_hashes=expected_ast_hashes,
        )
        eid = str(value.get("episode_id"))
        if eid in seen:
            raise ValueError("duplicate episode record")
        seen.add(eid)
        reduced.append(cast(dict[str, JsonValue], dict(value)))
    if expected_episode_ids is not None and seen != expected_episode_ids:
        raise ValueError("missing or extra episode records")
    return sorted(reduced, key=lambda value: str(value["episode_id"]))


def write_records(
    path: str | os.PathLike[str],
    records: list[Mapping[str, Any]],
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> dict[str, JsonValue]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_bytes(dict(r)) + b"\n" for r in records)
    if len(payload) > maximum_bytes:
        raise ValueError("shard exceeds bounded output size")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as out,
    ):
        out.write(payload)
    temporary.replace(target)
    return {
        "path": target.name,
        "record_count": len(records),
        "uncompressed_bytes": len(payload),
        "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def read_records(
    path: str | os.PathLike[str], *, max_records: int = 128, max_record_bytes: int = 4 * 1024 * 1024
) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if len(line.encode()) > max_record_bytes:
                raise ValueError("record exceeds bounded size")
            if len(records) >= max_records:
                raise ValueError("too many records")
            records.append(cast(dict[str, JsonValue], json.loads(line)))
    return records


# Names used by the command-line runner and tests.
run_episode = run_development_episode


def canonical_hash(record: Mapping[str, Any]) -> str:
    return sha256(canonical_projection(record))


_trajectory_seed = trajectory_seed


def run_trajectory_episode(
    config: object,
    episode: Mapping[str, Any],
    *rankers: Any,
    policies: Mapping[str, Any] | None = None,
    backend: GraphBackend | None = None,
) -> dict[str, JsonValue]:
    """Compatibility entry point mirroring the Stage 2D runner.

    A mapping is preferred; two positional rankers are interpreted as the
    frozen random and structural baselines.
    """
    roster = policies
    if roster is None:
        if len(rankers) == 1 and isinstance(rankers[0], Mapping):
            roster = cast(Mapping[str, Any], rankers[0])
        elif rankers:
            roster = {
                name: value for name, value in zip(("random", "structural"), rankers, strict=False)
            }
        else:
            raise TypeError("at least one policy ranker is required")
    return run_development_episode(config, episode, roster, backend=backend)


def replay_exact(primary: Mapping[str, Any], replay: Mapping[str, Any]) -> bool:
    """Compare replay after removing only declared timing/path fields."""
    return canonical_hash(primary) == canonical_hash(replay)


def verify_replay(primary: Any, replay: Any) -> dict[str, Any]:
    """Compatibility wrapper for the side-effect-free replay verifier."""
    from mutation_forge.stage3.replay import verify_replay as _verify_replay

    return _verify_replay(primary, replay)
