"""Native, provider-free evaluation for experiment ranker candidates.

This module intentionally lives outside the historical stage runners.  It uses
only the generic :class:`~mutation_forge.backends.base.GraphBackend` contract and
is therefore useful for both the real HEG backend and small deterministic test
backends.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.counterexamples import (
    CandidateProvenance,
    CounterexampleDecision,
    CounterexampleHalt,
    CounterexamplePipeline,
    CounterexampleVerified,
)
from mutation_forge.models import GraphScore, GraphState, JsonValue
from mutation_forge.policies.baselines import (
    BASELINES,
    HEG_FORBIDDEN_CYCLE_BREAK,
    HEG_UNIFORM_TWO_SWITCH,
)
from mutation_forge.proposals.k_switch import (
    EvaluationContractError,
    FeatureLimits,
    KSwitchPoolGenerator,
    PoolLimits,
    make_scientific_context,
)
from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.validation import ProgramIdentity
from mutation_forge.stage2b.rankers import SourceRanker

SCHEMA_VERSION = "mforge.experiment.evaluation.v2"
DEVELOPMENT_ARTIFACT_VERSION = "mforge.experiment.evaluation.development.v2"
REPLAY_ARTIFACT_VERSION = "mforge.experiment.evaluation.replay.v2"
EPISODE_CHECKPOINT_VERSION = "mforge.experiment.evaluation.episode.v2"
DEFAULT_WITNESS_CAP = 64
_THREAD_ENVIRONMENT = {
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
_BASELINE_OPERATORS = {
    "random": HEG_UNIFORM_TWO_SWITCH.operator_family,
    "structural": HEG_FORBIDDEN_CYCLE_BREAK.operator_family,
    **{name: policy.operator_family for name, policy in BASELINES.items()},
}


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _evaluation_config(config: object) -> object:
    return _get(config, "evaluation", config)


def _resources_config(config: object) -> object:
    return _get(config, "resources", config)


def _ints(config: object, name: str) -> tuple[int, ...]:
    value = _get(_evaluation_config(config), name)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"evaluation.{name} must be a non-empty integer sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"evaluation.{name} must contain positive integers")
        result.append(item)
    return tuple(result)


def _positive(config: object, name: str) -> int:
    value = _get(_evaluation_config(config), name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"evaluation.{name} must be a positive integer")
    return cast(int, value)


def _settings(config: object) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    baselines = _get(evaluation, "baselines", ())
    if not isinstance(baselines, (list, tuple)) or any(not isinstance(x, str) for x in baselines):
        raise ValueError("evaluation.baselines must be a string sequence")
    unknown = sorted(set(baselines).difference(BASELINES, {"random", "structural"}))
    if unknown:
        raise ValueError(f"unsupported baseline policy: {unknown}")
    resources = _resources_config(config)
    workers = _get(resources, "workers", 1)
    thread_count = _get(resources, "thread_count", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("resources.workers must be positive")
    if isinstance(thread_count, bool) or not isinstance(thread_count, int) or thread_count <= 0:
        raise ValueError("resources.thread_count must be positive")
    return {
        "orders": _ints(config, "orders"),
        "graph_seeds": _ints(config, "graph_seeds"),
        "policy_seeds": _ints(config, "policy_seeds"),
        "horizon": _positive(config, "horizon"),
        "proposal_pool_size": _positive(config, "proposal_pool_size"),
        "baselines": tuple(baselines),
        "replay": bool(_get(evaluation, "replay", False)),
        "workers": workers,
        "thread_count": thread_count,
        "witness_cap": int(_get(_get(config, "score", {}), "witness_cap", DEFAULT_WITNESS_CAP)),
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()
        + b"\n"
    )
    temporary.replace(path)


def _load_completed_pass(
    path: Path,
    *,
    candidate_id: str,
    source: Any,
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file() or not isinstance(source, str):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    source_identity = value.get("source_identity")
    observed_settings = value.get("settings")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "completed"
        or value.get("candidate_id") != candidate_id
        or not isinstance(source_identity, Mapping)
        or source_identity.get("source_sha256")
        != hashlib.sha256(source.encode("utf-8")).hexdigest()
        or not isinstance(observed_settings, Mapping)
    ):
        return None
    scientific_settings = {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in settings.items()
        if key not in {"workers", "thread_count"}
    }
    if any(
        observed_settings.get(key) != item
        for key, item in scientific_settings.items()
    ):
        return None
    return value


def _episode_checkpoint_identity(
    candidate_id: str,
    source_identity: Mapping[str, Any],
    settings: Mapping[str, Any],
    pass_name: str,
) -> str:
    scientific_settings = {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in settings.items()
        if key not in {"workers", "thread_count"}
    }
    return _hash(
        {
            "candidate_id": candidate_id,
            "source_identity": dict(source_identity),
            "settings": scientific_settings,
            "pass": pass_name,
        }
    )


def _load_episode_checkpoint(
    path: Path,
    *,
    identity: str,
    index: int,
    order: int,
    graph_seed: int,
    policy_seed: int,
) -> dict[str, JsonValue] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"evaluation episode checkpoint is unreadable: {path}") from exc
    episode = value.get("episode") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != EPISODE_CHECKPOINT_VERSION
        or value.get("identity") != identity
        or value.get("index") != index
        or not isinstance(episode, dict)
        or episode.get("order") != order
        or episode.get("graph_seed") != graph_seed
        or episode.get("policy_seed") != policy_seed
    ):
        raise ValueError(f"evaluation episode checkpoint does not match request: {path}")
    return cast(dict[str, JsonValue], episode)


def _artifact_root(config: object, artifact_root: str | Path | None) -> Path:
    if artifact_root is not None:
        root = Path(artifact_root)
        if root.name != "artifacts":
            root = root / "artifacts"
        return root
    workspace = _get(config, "workspace")
    exp_id = _get(config, "exp_id")
    if workspace is None or exp_id is None:
        raise ValueError("artifact_root is required when config has no workspace/exp_id")
    return Path(workspace) / str(exp_id) / "artifacts"


def _repo_path(config: object, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    raw = _get(config, "raw", config)
    repositories = _get(raw, "repositories", {})
    configured = _get(repositories, "heg_repo")
    if configured is None:
        configured = _get(raw, "heg_repo")
    if configured:
        source_dir = _get(config, "source_dir", Path.cwd())
        path = Path(str(configured))
        return ((Path(source_dir) / path) if not path.is_absolute() else path).resolve()
    # The sibling repository is the canonical local fallback used by the lab.
    return (Path(__file__).resolve().parents[4] / "heg").resolve()


def _git_state(repo: Path) -> dict[str, JsonValue]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"repo": str(repo), "commit": commit, "dirty": dirty}


def _identity(source: str, ranker: Any) -> dict[str, JsonValue]:
    identity = getattr(ranker, "identity", None)
    if isinstance(identity, ProgramIdentity):
        return identity.as_dict()
    if identity is not None and callable(getattr(identity, "as_dict", None)):
        return cast(dict[str, JsonValue], identity.as_dict())
    return {"source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def _source_and_ranker(
    candidate_id: str, source: Any, limits: SandboxLimits
) -> tuple[str, Any, bool]:
    if isinstance(source, str):
        return source, SourceRanker(candidate_id, source, limits), True
    value = _get(source, "source")
    if isinstance(value, str) and not callable(getattr(source, "rank", None)):
        return value, SourceRanker(candidate_id, value, limits), True
    if isinstance(value, str) and callable(getattr(source, "rank", None)):
        return value, source, False
    if callable(getattr(source, "rank", None)):
        return str(getattr(source, "source", "")), source, False
    raise TypeError("candidate source must be source text or a ranker object")


def _score(backend: GraphBackend, graph: GraphState, witness_cap: int) -> GraphScore:
    result = backend.score(graph, witness_cap=witness_cap, cutoff=None)
    if result is None:
        raise RuntimeError("backend returned a cutoff-dominated score without a cutoff")
    expected = backend.target_forbidden_lengths(graph.order)
    observed = tuple(length for length, _ in result.capped_cycle_counts)
    counts = tuple(count for _, count in result.capped_cycle_counts)
    if observed != expected:
        raise EvaluationContractError(
            f"score lengths {observed!r} do not match target lengths {expected!r}"
        )
    if len(observed) != len(set(observed)):
        raise EvaluationContractError("score contains duplicate target lengths")
    if any(count < 0 or count > witness_cap for count in counts):
        raise EvaluationContractError(
            f"score counts must be within [0, {witness_cap}]"
        )
    if sum(counts) != result.total_capped_witnesses:
        raise EvaluationContractError("score total does not equal its count vector")
    if result.total_capped_witnesses < 0:
        raise EvaluationContractError("score total cannot be negative")
    if not result.valid and result.total_capped_witnesses == 0:
        raise EvaluationContractError("invalid score cannot report zero witnesses")
    return result


def _summary(
    curve: list[int], initial: int, final: GraphScore, accepted: int, failures: int
) -> dict[str, JsonValue]:
    normalized = [(initial - value) / max(1, initial) for value in curve]
    return {
        "initial_total_witnesses": initial,
        "best_total_witnesses": min(curve, default=initial),
        "final_score": final.as_dict(),
        "accepted_count": accepted,
        "rejected_count": max(0, len(curve) - accepted - failures),
        "failure_count": failures,
        "auc": sum(normalized) / len(normalized) if normalized else 0.0,
        "raw_best_so_far_curve": cast(list[JsonValue], curve),
        "normalized_best_so_far_curve": cast(list[JsonValue], normalized),
    }


def _rank_projection(rank: Any) -> dict[str, JsonValue]:
    """Keep rank decisions while excluding process timing from replay identity."""
    raw = (
        rank.as_dict()
        if callable(getattr(rank, "as_dict", None))
        else {
            "selected_proposal_id": getattr(rank, "selected_proposal_id", None),
            "exception": bool(getattr(rank, "exception", False)),
            "timeout": bool(getattr(rank, "timeout", False)),
            "crash": bool(getattr(rank, "crash", False)),
            "protocol": bool(getattr(rank, "protocol", False)),
        }
    )
    return cast(
        dict[str, JsonValue],
        {str(key): value for key, value in raw.items() if not str(key).endswith("_ns")},
    )


def _trajectory(
    backend: GraphBackend,
    ranker: Any,
    settings: Mapping[str, Any],
    *,
    order: int,
    graph_seed: int,
    policy_seed: int,
    candidate_id: str,
    counterexample_pipeline: CounterexamplePipeline | None = None,
) -> dict[str, JsonValue]:
    graph = backend.generate_seed(order=order, seed=graph_seed)
    validation = backend.validate(graph)
    if not validation.valid:
        raise ValueError(f"invalid generated graph: {validation.errors}")
    initial = _score(backend, graph, settings["witness_cap"])
    episode_id = f"o{order}-g{graph_seed}-p{policy_seed}"
    candidate_generation: int | None = None
    candidate_slot: str | None = None
    candidate_parts = candidate_id.split("-", 1)
    if (
        len(candidate_parts) == 2
        and candidate_parts[0].startswith("g")
        and candidate_parts[0][1:].isdigit()
    ):
        candidate_generation = int(candidate_parts[0][1:])
        candidate_slot = candidate_parts[1]
    if counterexample_pipeline is not None:
        outcome = counterexample_pipeline.inspect(
            graph=graph,
            score=initial,
            witness_cap=settings["witness_cap"],
            provenance=CandidateProvenance(
                source_kind="seed",
                source_id=candidate_id,
                generation=candidate_generation,
                slot=candidate_slot,
                episode_id=episode_id,
                graph_seed=graph_seed,
                policy_seed=policy_seed,
                evaluation_step=0,
            ),
        )
        if outcome.decision is CounterexampleDecision.STOP_VERIFIED:
            raise CounterexampleVerified(outcome)
        if outcome.decision in {
            CounterexampleDecision.PAUSE_INCONCLUSIVE,
            CounterexampleDecision.FAIL,
        }:
            raise CounterexampleHalt(outcome)
    states: dict[str, dict[str, Any]] = {}
    names = [candidate_id, *settings["baselines"]]
    for name in names:
        states[name] = {
            "graph": graph,
            "score": initial,
            "curve": [],
            "accepted": 0,
            "failures": 0,
            "trace": [],
            "initial_total": initial.total_capped_witnesses,
            "stagnation": 0,
            "recent_accepts": [],
            "recent_improvements": [],
            "recent_duplicate_rates": [],
        }
    pool_generator = KSwitchPoolGenerator(
        backend,
        pool_limits=PoolLimits(pool_size=settings["proposal_pool_size"]),
        feature_limits=FeatureLimits(
            forbidden_lengths=backend.target_forbidden_lengths(graph.order)
        ),
    )
    for step in range(settings["horizon"]):
        state = states[candidate_id]
        pool = pool_generator.generate(state["graph"], policy_seed=policy_seed, step=step)
        if pool.retained:
            window = 8
            recent_accepts = state["recent_accepts"][-window:]
            recent_improvements = state["recent_improvements"][-window:]
            recent_duplicates = state["recent_duplicate_rates"][-window:]
            context = make_scientific_context(
                state["graph"],
                state["score"],
                forbidden_lengths=backend.target_forbidden_lengths(state["graph"].order),
                step=step,
                remaining_steps=settings["horizon"] - step - 1,
                stagnation=state["stagnation"],
                recent_best_improvement=max(recent_improvements, default=0.0),
                recent_acceptance_rate=(
                    sum(recent_accepts) / len(recent_accepts) if recent_accepts else 0.0
                ),
                recent_duplicate_rate=(
                    sum(recent_duplicates) / len(recent_duplicates) if recent_duplicates else 0.0
                ),
            )
            rank = ranker.rank(context, pool)
            candidate = (
                ranker.candidate(pool, rank.selected_proposal_id)
                if callable(getattr(ranker, "candidate", None))
                else next(
                    (
                        item
                        for item in pool.candidates
                        if item.proposal_id == rank.selected_proposal_id
                    ),
                    None,
                )
            )
            if candidate is None:
                state["failures"] += 1
                state["curve"].append(state["score"].total_capped_witnesses)
                state["recent_accepts"].append(0)
                state["recent_improvements"].append(0.0)
                state["stagnation"] += 1
                state["trace"].append(
                    {
                        "step": step,
                        "accepted": False,
                        "error": "ranker did not select a pool proposal",
                        "rank": _rank_projection(rank),
                    }
                )
            else:
                _advance(
                    backend,
                    state,
                    candidate.rewrite,
                    settings,
                    step,
                    _rank_projection(rank),
                    counterexample_pipeline=counterexample_pipeline,
                    source_kind="generated_ranker",
                    source_id=candidate_id,
                    generation=candidate_generation,
                    slot=candidate_slot,
                    episode_id=episode_id,
                    graph_seed=graph_seed,
                    policy_seed=policy_seed,
                )
            state["recent_duplicate_rates"].append(pool.deduplicated / max(1, pool.attempted))
        else:
            state["failures"] += 1
            state["curve"].append(state["score"].total_capped_witnesses)
            state["recent_accepts"].append(0)
            state["recent_improvements"].append(0.0)
            state["recent_duplicate_rates"].append(pool.deduplicated / max(1, pool.attempted))
            state["stagnation"] += 1
            state["trace"].append(
                {
                    "step": step,
                    "accepted": False,
                    "error": "empty proposal pool",
                    "pool_hash": pool.pool_hash,
                }
            )
        for baseline in settings["baselines"]:
            baseline_state = states[baseline]
            operator_family = _BASELINE_OPERATORS[baseline]
            rewrite = backend.propose_rewrite(
                baseline_state["graph"],
                operator_family=operator_family,
                policy_seed=policy_seed,
                evaluation=step,
            )
            _advance(
                backend,
                baseline_state,
                rewrite,
                settings,
                step,
                {"operator_family": operator_family},
                counterexample_pipeline=counterexample_pipeline,
                source_kind="baseline",
                source_id=baseline,
                generation=candidate_generation,
                slot=candidate_slot,
                episode_id=episode_id,
                graph_seed=graph_seed,
                policy_seed=policy_seed,
            )
            baseline_state["recent_duplicate_rates"].append(0.0)
    policies: dict[str, JsonValue] = {}
    for name, state in states.items():
        policies[name] = _summary(
            state["curve"],
            initial.total_capped_witnesses,
            state["score"],
            state["accepted"],
            state["failures"],
        )
        cast(dict[str, JsonValue], policies[name])["trace"] = state["trace"]
    return {
        "episode_id": episode_id,
        "order": order,
        "graph_seed": graph_seed,
        "policy_seed": policy_seed,
        "horizon": settings["horizon"],
        "initial_score": initial.as_dict(),
        "policies": policies,
    }


def _advance(
    backend: GraphBackend,
    state: dict[str, Any],
    rewrite: Any,
    settings: Mapping[str, Any],
    step: int,
    rank: Mapping[str, Any],
    *,
    counterexample_pipeline: CounterexamplePipeline | None = None,
    source_kind: str,
    source_id: str,
    generation: int | None,
    slot: str | None,
    episode_id: str,
    graph_seed: int,
    policy_seed: int,
) -> None:
    previous = state["score"]
    try:
        proposed = backend.apply_rewrite(state["graph"], rewrite)
        if not backend.validate(proposed).valid:
            raise ValueError("rewrite produced an invalid graph")
        score = _score(backend, proposed, settings["witness_cap"])
    except Exception as error:
        state["failures"] += 1
        state["curve"].append(state["score"].total_capped_witnesses)
        state["recent_accepts"].append(0)
        state["recent_improvements"].append(0.0)
        state["stagnation"] += 1
        state["trace"].append(
            {"step": step, "accepted": False, "error": str(error), "rank": dict(rank)}
        )
        return
    if counterexample_pipeline is not None:
        outcome = counterexample_pipeline.inspect(
            graph=proposed,
            score=score,
            witness_cap=settings["witness_cap"],
            provenance=CandidateProvenance(
                source_kind=source_kind,
                source_id=source_id,
                generation=generation,
                slot=slot,
                episode_id=episode_id,
                graph_seed=graph_seed,
                policy_seed=policy_seed,
                evaluation_step=step,
            ),
        )
        if outcome.decision is CounterexampleDecision.STOP_VERIFIED:
            raise CounterexampleVerified(outcome)
        if outcome.decision in {
            CounterexampleDecision.PAUSE_INCONCLUSIVE,
            CounterexampleDecision.FAIL,
        }:
            raise CounterexampleHalt(outcome)
    accepted = score.ordering_key < state["score"].ordering_key
    if accepted:
        state["graph"], state["score"] = proposed, score
        state["accepted"] += 1
        state["stagnation"] = 0
    else:
        state["stagnation"] += 1
    state["recent_accepts"].append(int(accepted))
    state["recent_improvements"].append(
        max(
            0.0,
            (previous.total_capped_witnesses - state["score"].total_capped_witnesses)
            / max(1, state["initial_total"]),
        )
    )
    state["curve"].append(
        min(
            state["score"].total_capped_witnesses,
            min(state["curve"], default=state["score"].total_capped_witnesses),
        )
    )
    state["trace"].append(
        {"step": step, "accepted": accepted, "score": score.as_dict(), "rank": dict(rank)}
    )


def _run_once(
    config: object,
    candidate_id: str,
    source: Any,
    settings: Mapping[str, Any],
    *,
    backend: GraphBackend,
    limits: SandboxLimits,
    progress: Callable[[Mapping[str, JsonValue]], None] | None = None,
    profiling_enabled: bool = False,
    pass_name: str = "development",
    counterexample_pipeline: CounterexamplePipeline | None = None,
    checkpoint_root: Path | None = None,
) -> dict[str, JsonValue]:
    source_text, ranker, owned_ranker = _source_and_ranker(candidate_id, source, limits)
    source_identity = _identity(source_text, ranker)
    checkpoint_identity = _episode_checkpoint_identity(
        candidate_id,
        source_identity,
        settings,
        pass_name,
    )
    started = time.monotonic()
    phase_seconds: dict[str, float] = {}
    phase_calls: dict[str, int] = {}
    try:
        episodes: list[dict[str, JsonValue]] = []
        executed_episodes = 0
        restored_episodes = 0
        execution_seconds = 0.0
        total = (
            len(settings["orders"]) * len(settings["graph_seeds"]) * len(settings["policy_seeds"])
        )
        completed = 0
        for order in settings["orders"]:
            order_started = time.monotonic()
            for graph_seed in settings["graph_seeds"]:
                for policy_seed in settings["policy_seeds"]:
                    episode_path = (
                        checkpoint_root
                        / pass_name
                        / candidate_id
                        / f"episode-{completed:06d}.json"
                        if checkpoint_root is not None
                        else None
                    )
                    episode = (
                        _load_episode_checkpoint(
                            episode_path,
                            identity=checkpoint_identity,
                            index=completed,
                            order=order,
                            graph_seed=graph_seed,
                            policy_seed=policy_seed,
                        )
                        if episode_path is not None
                        else None
                    )
                    restored = episode is not None
                    if episode is None:
                        episode_started = time.monotonic()
                        episode = _trajectory(
                            backend,
                            ranker,
                            settings,
                            order=order,
                            graph_seed=graph_seed,
                            policy_seed=policy_seed,
                            candidate_id=candidate_id,
                            counterexample_pipeline=counterexample_pipeline,
                        )
                        execution_seconds += time.monotonic() - episode_started
                        executed_episodes += 1
                        if episode_path is not None:
                            _write_json(
                                episode_path,
                                {
                                    "schema_version": EPISODE_CHECKPOINT_VERSION,
                                    "identity": checkpoint_identity,
                                    "index": completed,
                                    "episode": episode,
                                },
                            )
                    else:
                        restored_episodes += 1
                    episodes.append(episode)
                    completed += 1
                    if progress is not None:
                        progress_fields: dict[str, JsonValue] = {
                            "candidate_id": candidate_id,
                            "order": order,
                            "graph_seed": graph_seed,
                            "policy_seed": policy_seed,
                            "completed": completed,
                            "total": total,
                            "evaluations": completed,
                            "restored": restored,
                            "executed": executed_episodes,
                            "restored_count": restored_episodes,
                            "pass": pass_name,
                            "pass_progress": completed / max(total, 1),
                            "development_progress": (
                                completed / max(total, 1) if pass_name == "development" else 0.0
                            ),
                            "replay_progress": (
                                completed / max(total, 1) if pass_name == "replay" else 0.0
                            ),
                        }
                        if executed_episodes:
                            progress_fields["evaluations_per_second"] = (
                                executed_episodes / max(execution_seconds, 1e-9)
                            )
                        progress(progress_fields)
            phase_seconds[f"order_{order}"] = time.monotonic() - order_started
            phase_calls[f"order_{order}"] = len(settings["graph_seeds"]) * len(
                settings["policy_seeds"]
            )
    finally:
        if owned_ranker:
            ranker.close()
    candidate_rows = [cast(Mapping[str, Any], row["policies"])[candidate_id] for row in episodes]
    aucs = [float(_get(row, "auc", 0.0)) for row in candidate_rows]
    baseline_names = tuple(str(name) for name in settings["baselines"])
    baseline_auc: dict[str, JsonValue] = {}
    for baseline in baseline_names:
        values = [
            float(_get(cast(Mapping[str, Any], row["policies"]).get(baseline), "auc", 0.0))
            for row in episodes
        ]
        baseline_auc[baseline] = sum(values) / len(values) if values else 0.0
    result: dict[str, JsonValue] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "candidate_id": candidate_id,
        "source_identity": source_identity,
        "settings": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in settings.items()
        },
        "episodes": cast(list[JsonValue], episodes),
        "summary": {
            "episode_count": len(episodes),
            "mean_auc": sum(aucs) / len(aucs) if aucs else 0.0,
            "best_auc": max(aucs, default=0.0),
            "baseline_auc": baseline_auc,
        },
        "runtime": {
            "elapsed_seconds": max(time.monotonic() - started, 0.0),
            "execution_seconds": execution_seconds,
            "executed_episodes": executed_episodes,
            "restored_episodes": restored_episodes,
        },
        "provider_calls": 0,
        "model_calls": 0,
        "network_calls": 0,
    }
    if profiling_enabled:
        measured = max(time.monotonic() - started, 1e-9)
        accounted = sum(phase_seconds.values())
        result["timing_profile"] = {
            "enabled": True,
            "phase_seconds": cast(JsonValue, phase_seconds),
            "phase_calls": cast(JsonValue, phase_calls),
            "phase_children_seconds": {},
            "phase_children_calls": {},
            "measured_total_seconds": measured,
            "accounted_seconds": accounted,
            "unattributed_seconds": max(0.0, measured - accounted),
            "unattributed_fraction": max(0.0, measured - accounted) / measured,
            "dominant_phase": max(
                phase_seconds.items(), key=lambda item: item[1], default=("evaluation", measured)
            )[0],
            "dominant_seconds": max(phase_seconds.values(), default=measured),
            "profiled_episodes": len(episodes),
            "throughput": len(episodes) / measured,
            "evaluations_per_second": len(episodes) / measured,
            "hotspots": [
                {
                    "phase": name,
                    "seconds": seconds,
                    "percent": seconds / measured * 100,
                }
                for name, seconds in sorted(
                    phase_seconds.items(), key=lambda item: item[1], reverse=True
                )
            ],
        }
    return result


def evaluate_candidate(
    config: object,
    candidate_id: str,
    source: Any | None = None,
    *,
    ranker_source: Any | None = None,
    artifact_root: str | Path | None = None,
    output_root: str | Path | None = None,
    layout: Any | None = None,
    backend: GraphBackend | None = None,
    backend_factory: Callable[[Path], GraphBackend] | None = None,
    heg_repo: str | Path | None = None,
    sandbox_limits: SandboxLimits | None = None,
    progress: Callable[[Mapping[str, JsonValue]], None] | None = None,
    profiling_enabled: bool = False,
    counterexample_pipeline: CounterexamplePipeline | None = None,
    counterexample_event_callback: (Callable[[str, Mapping[str, Any]], None] | None) = None,
) -> dict[str, JsonValue]:
    """Evaluate one validated ranker source and persist development/replay evidence."""
    if (
        not candidate_id
        or candidate_id in {".", ".."}
        or "/" in candidate_id
        or "\\" in candidate_id
        or "\x00" in candidate_id
        or Path(candidate_id).name != candidate_id
    ):
        raise ValueError("candidate_id must be one safe directory/file name")
    if source is None:
        source = ranker_source
    if source is None:
        raise ValueError("ranker source is required")
    if artifact_root is None:
        artifact_root = output_root or _get(layout, "artifacts")
    settings = _settings(config)
    root = _artifact_root(config, artifact_root)
    limits = sandbox_limits or SandboxLimits()
    injected = backend is not None or backend_factory is not None

    def make_backend() -> GraphBackend:
        if backend_factory is not None:
            try:
                return backend_factory(_repo_path(config, heg_repo))
            except TypeError:
                # A no-argument factory is convenient for tiny deterministic
                # test backends; production factories receive the resolved HEG
                # repository path.
                no_arg_factory = cast(Callable[[], GraphBackend], backend_factory)
                return no_arg_factory()
        if backend is not None:
            return backend
        from mutation_forge.backends.heg import HegBackend

        return HegBackend(_repo_path(config, heg_repo))

    primary_backend = make_backend()
    pipeline = counterexample_pipeline or CounterexamplePipeline(
        backend=primary_backend,
        artifact_root=root,
        event_callback=counterexample_event_callback,
    )
    try:
        development_path = root / "evaluations" / "development" / f"{candidate_id}.json"
        primary = _load_completed_pass(
            development_path,
            candidate_id=candidate_id,
            source=source,
            settings=settings,
        )
        primary_recovered = primary is not None
        if primary is None:
            primary = _run_once(
                config,
                candidate_id,
                source,
                settings,
                backend=primary_backend,
                limits=limits,
                progress=progress,
                profiling_enabled=profiling_enabled,
                pass_name="development",
                counterexample_pipeline=pipeline,
                checkpoint_root=root / "evaluations" / "episodes",
            )
        backend_repo = getattr(primary_backend, "repo", None)
        observed_repo = (
            Path(backend_repo) if backend_repo is not None else _repo_path(config, heg_repo)
        )
        git = (
            _git_state(observed_repo)
            if getattr(primary_backend, "commit", None) is None
            else {
                "repo": str(observed_repo),
                "commit": getattr(primary_backend, "commit", None),
                "dirty": getattr(primary_backend, "dirty", None),
            }
        )
        provenance = {
            "backend_id": getattr(primary_backend, "backend_id", "unknown"),
            "heg": git,
            "workers": settings["workers"],
            "thread_count": settings["thread_count"],
            "provider_calls": 0,
            "model_calls": 0,
            "network_calls": 0,
            "backend_injected": injected,
        }
        if not primary_recovered:
            primary["provenance"] = provenance
            _write_json(development_path, primary)
        result = dict(primary)
        result["artifacts"] = {"development": str(development_path)}
        if progress is not None:
            summary = primary.get("summary")
            if isinstance(summary, Mapping):
                progress(
                    {
                        "candidate_id": candidate_id,
                        "pass": "development",
                        "pass_completed": True,
                        "development_progress": 1.0,
                        "replay_progress": 0.0,
                        "current_objective": cast(JsonValue, summary.get("mean_auc")),
                        "best_auc": cast(JsonValue, summary.get("best_auc")),
                    }
                )
        if settings["replay"]:
            replay_path = root / "evaluations" / "replay" / f"{candidate_id}.json"
            replay = _load_completed_pass(
                replay_path,
                candidate_id=candidate_id,
                source=source,
                settings=settings,
            )
            if replay is None:
                replay_backend = make_backend()
                try:
                    replay = _run_once(
                        config,
                        candidate_id,
                        source,
                        settings,
                        backend=replay_backend,
                        limits=limits,
                        progress=progress,
                        profiling_enabled=profiling_enabled,
                        pass_name="replay",
                        counterexample_pipeline=None,
                        checkpoint_root=root / "evaluations" / "episodes",
                    )
                finally:
                    if replay_backend is not primary_backend and backend is None:
                        replay_backend.close()
                replay["provenance"] = provenance
                _write_json(replay_path, replay)
            result["replay"] = {
                "enabled": True,
                "exact": _hash(primary) == _hash(replay),
                "artifact": str(replay_path),
                "sha256": _hash(replay),
            }
        else:
            result["replay"] = {"enabled": False, "exact": None}
        return result
    finally:
        if backend is None:
            primary_backend.close()


def evaluate_population(
    config: object,
    candidates: Mapping[str, Any] | None = None,
    *,
    population: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None = None,
    layout: Any | None = None,
    backend: GraphBackend | None = None,
    backend_factory: Callable[[Path], GraphBackend] | None = None,
    heg_repo: str | Path | None = None,
    sandbox_limits: SandboxLimits | None = None,
    progress: Callable[[Mapping[str, JsonValue]], None] | None = None,
    profiling_enabled: bool = False,
) -> dict[str, JsonValue]:
    """Evaluate a deterministic candidate roster using the native evaluator."""
    if candidates is None:
        candidates = population
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a non-empty mapping")
    rows: dict[str, JsonValue] = {}
    for candidate_id in sorted(candidates):
        rows[candidate_id] = evaluate_candidate(
            config,
            candidate_id,
            candidates[candidate_id],
            artifact_root=artifact_root,
            layout=layout,
            backend=backend,
            backend_factory=backend_factory,
            heg_repo=heg_repo,
            sandbox_limits=sandbox_limits,
            progress=progress,
            profiling_enabled=profiling_enabled,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "candidate_ids": list(sorted(candidates)),
        "candidates": rows,
        "provider_calls": 0,
        "model_calls": 0,
        "network_calls": 0,
    }


__all__ = ["evaluate_candidate", "evaluate_population", "SCHEMA_VERSION"]
