from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from mutation_forge.artifacts import git_state
from mutation_forge.backends.base import GraphBackend
from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import GraphScore, GraphState, JsonValue
from mutation_forge.proposals.k_switch import (
    KSwitchPoolGenerator,
    ProposalCandidate,
    ProposalPool,
    make_scientific_context,
)
from mutation_forge.sandbox.validation import ProgramIdentity, validate_policy
from mutation_forge.stage2b.rankers import RankResult, SourceRanker
from mutation_forge.stage2d.config import Stage2DConfig
from mutation_forge.stage2d.manifest import (
    load_manifest,
    write_manifest,
)
from mutation_forge.stage2d.statistics import summarize_episodes

STAGE2D_ARTIFACT_VERSION = "stage2d.artifact.v1"
STAGE2D_EPISODE_VERSION = "stage2d.episode.v1"
STAGE2D_REDUCTION_VERSION = "stage2d.reduction.v1"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strip_timing(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_strip_timing(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_timing(item)
            for key, item in value.items()
            if not key.endswith("_ns")
            and key not in {"started_at", "finished_at", "elapsed_seconds"}
        }
    return value


def _write_json(path: Path, value: object, *, maximum: int) -> None:
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    if len(encoded) > maximum:
        raise ValueError(f"artifact {path.name} exceeds {maximum} bytes")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _identity(source: str, config: Stage2DConfig, policy: str) -> ProgramIdentity:
    validation = validate_policy(source, config.stage2b.sandbox)
    if not validation.valid:
        raise ValueError(f"frozen {policy} ranker is invalid")
    expected_source = getattr(config.identity, f"{policy}_source_sha256")
    expected_ast = getattr(config.identity, f"{policy}_ast_sha256")
    if (
        validation.identity.source_sha256 != expected_source
        or validation.identity.normalized_ast_sha256 != expected_ast
    ):
        raise ValueError(f"frozen {policy} ranker identity mismatch")
    return validation.identity


def _candidate(pool: ProposalPool, rank: RankResult) -> ProposalCandidate:
    if (
        rank.exception
        or rank.timeout
        or rank.crash
        or rank.protocol
        or rank.selected_proposal_id is None
    ):
        raise RuntimeError(f"{rank.policy_id} ranker failed: {rank.error}")
    for candidate in pool.candidates:
        if candidate.proposal_id == rank.selected_proposal_id:
            return candidate
    raise RuntimeError("ranker selected a proposal outside its immutable pool")


def _trajectory_seed(episode: dict[str, JsonValue], step: int) -> int:
    payload = [
        "stage2d.trajectory-seed.v1",
        episode["order"],
        episode["graph_seed"],
        episode["policy_seed"],
        step,
    ]
    return int.from_bytes(hashlib.sha256(_canonical_bytes(payload)).digest()[:4], "big")


@dataclass(slots=True)
class _PolicyState:
    graph: GraphState
    score: GraphScore
    best_total: int
    curve_raw: list[int] = field(default_factory=list)
    curve_normalized: list[float] = field(default_factory=list)
    trace: list[dict[str, JsonValue]] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    first_improvement_step: int | None = None
    first_improvement_ns: int | None = None
    stagnation: int = 0
    recent_accepts: list[int] = field(default_factory=list)
    recent_improvements: list[float] = field(default_factory=list)
    recent_duplicate_rates: list[float] = field(default_factory=list)

    def context_rates(self) -> tuple[float, float, float]:
        window = 8
        acceptance = self.recent_accepts[-window:]
        improvements = self.recent_improvements[-window:]
        duplicate_rates = self.recent_duplicate_rates[-window:]
        return (
            max(improvements, default=0.0),
            sum(acceptance) / len(acceptance) if acceptance else 0.0,
            sum(duplicate_rates) / len(duplicate_rates) if duplicate_rates else 0.0,
        )


def _score_selected(
    backend: GraphBackend,
    graph: GraphState,
    candidate: ProposalCandidate,
    witness_cap: int,
) -> tuple[GraphState, GraphScore, int]:
    started = time.perf_counter_ns()
    candidate_graph = backend.apply_rewrite(graph, candidate.rewrite)
    validation = backend.validate(candidate_graph)
    if not validation.valid:
        raise ValueError(f"host-applied graph is invalid: {validation.errors}")
    score = backend.score(candidate_graph, witness_cap=witness_cap, cutoff=None)
    if score is None:
        raise RuntimeError("selected-plan score cannot be cutoff-dominated")
    return candidate_graph, score, time.perf_counter_ns() - started


def _apply_policy_step(
    backend: GraphBackend,
    config: Stage2DConfig,
    state: _PolicyState,
    ranker: SourceRanker,
    pool: ProposalPool,
    *,
    initial_total: int,
    step: int,
    horizon: int,
    trajectory_started_ns: int,
) -> tuple[dict[str, JsonValue], int]:
    improvement, acceptance_rate, duplicate_rate = state.context_rates()
    context = make_scientific_context(
        state.graph,
        state.score,
        forbidden_lengths=config.stage2b.features.forbidden_lengths,
        step=step,
        remaining_steps=horizon - step - 1,
        stagnation=state.stagnation,
        recent_best_improvement=improvement,
        recent_acceptance_rate=acceptance_rate,
        recent_duplicate_rate=duplicate_rate,
    )
    rank = ranker.rank(context, pool)
    candidate = _candidate(pool, rank)
    candidate_graph, candidate_score, scoring_ns = _score_selected(
        backend,
        state.graph,
        candidate,
        config.stage2b.search.witness_cap,
    )
    accepted = candidate_score.ordering_key < state.score.ordering_key
    previous_total = state.score.total_capped_witnesses
    if accepted:
        state.graph = candidate_graph
        state.score = candidate_score
        state.accepted += 1
        state.stagnation = 0
        if state.first_improvement_step is None:
            state.first_improvement_step = step
            state.first_improvement_ns = time.perf_counter_ns() - trajectory_started_ns
    else:
        state.rejected += 1
        state.stagnation += 1
    state.best_total = min(state.best_total, state.score.total_capped_witnesses)
    quality = (initial_total - state.best_total) / max(1, initial_total)
    state.curve_raw.append(state.best_total)
    state.curve_normalized.append(quality)
    state.duplicates += pool.deduplicated
    state.recent_accepts.append(int(accepted))
    state.recent_improvements.append(
        max(0.0, (previous_total - state.score.total_capped_witnesses) / max(1, initial_total))
    )
    state.recent_duplicate_rates.append(
        pool.deduplicated / max(1, pool.attempted)
    )
    trace: dict[str, JsonValue] = {
        "step": step,
        "pool_hash": pool.pool_hash,
        "pool_size": pool.retained,
        "pool_attempted": pool.attempted,
        "pool_rejected": cast(dict[str, JsonValue], pool.rejected),
        "pool_deduplicated": pool.deduplicated,
        "selected_proposal_id": candidate.proposal_id,
        "selected_k": candidate.payload["k"],
        "selected_operator_family": candidate.payload["operator_family"],
        "selected_selector_tags": cast(
            list[JsonValue], candidate.payload["selector_tags"]
        ),
        "accepted": accepted,
        "selected_score": candidate_score.as_dict(),
        "current_score": state.score.as_dict(),
        "best_total_witnesses": state.best_total,
        "state_hash": backend.state_hash(state.graph),
        "ranker_flags": {
            "exception": rank.exception,
            "timeout": rank.timeout,
            "crash": rank.crash,
            "protocol": rank.protocol,
        },
        "ranker_elapsed_ns": rank.elapsed_ns,
        "selected_scoring_ns": scoring_ns,
        "pool_legality_ns": pool.legality_elapsed_ns,
        "pool_feature_ns": pool.feature_elapsed_ns,
    }
    state.trace.append(trace)
    return trace, scoring_ns


def _policy_summary(
    state: _PolicyState,
    initial_score: GraphScore,
) -> dict[str, JsonValue]:
    return {
        "auc": sum(state.curve_normalized) / len(state.curve_normalized),
        "raw_best_so_far_curve": cast(list[JsonValue], state.curve_raw),
        "normalized_best_so_far_curve": cast(
            list[JsonValue], state.curve_normalized
        ),
        "initial_score": initial_score.as_dict(),
        "final_score": state.score.as_dict(),
        "best_score": state.score.as_dict(),
        "best_total_witnesses": state.best_total,
        "accepted_count": state.accepted,
        "rejected_count": state.rejected,
        "duplicate_count": state.duplicates,
        "evaluations_to_first_improvement": (
            state.first_improvement_step + 1
            if state.first_improvement_step is not None
            else None
        ),
        "first_improvement_ns": state.first_improvement_ns,
        "trace": cast(list[JsonValue], state.trace),
    }


def run_trajectory_episode(
    config: Stage2DConfig,
    episode: dict[str, JsonValue],
    random_ranker: SourceRanker,
    structural_ranker: SourceRanker,
    *,
    backend: GraphBackend | None = None,
) -> dict[str, JsonValue]:
    owned_backend = backend is None
    applied_backend = backend or ToyBackend()
    started_ns = time.perf_counter_ns()
    try:
        order = cast(int, episode["order"])
        graph_seed = cast(int, episode["graph_seed"])
        horizon = cast(int, episode["horizon"])
        graph = applied_backend.generate_seed(order=order, seed=graph_seed)
        validation = applied_backend.validate(graph)
        if not validation.valid:
            raise RuntimeError(f"initial graph is invalid: {validation.errors}")
        initial_score = applied_backend.score(
            graph,
            witness_cap=config.stage2b.search.witness_cap,
            cutoff=None,
        )
        if initial_score is None:
            raise RuntimeError("initial score unavailable")
        random_state = _PolicyState(
            graph=graph,
            score=initial_score,
            best_total=initial_score.total_capped_witnesses,
        )
        structural_state = _PolicyState(
            graph=graph,
            score=initial_score,
            best_total=initial_score.total_capped_witnesses,
        )
        generator = KSwitchPoolGenerator(
            applied_backend,
            pool_limits=config.stage2b.pool,
            feature_limits=config.stage2b.features,
        )
        divergence_step: int | None = None
        shared_pool_steps = 0
        independent_pool_steps = 0
        selected_scoring_ns = 0
        step_records: list[dict[str, JsonValue]] = []
        for step in range(horizon):
            step_seed = _trajectory_seed(episode, step)
            states_identical = random_state.graph == structural_state.graph
            random_pool = generator.generate(
                random_state.graph,
                policy_seed=step_seed,
                step=step,
            )
            if not random_pool.candidates:
                raise RuntimeError("random trajectory produced an empty legal pool")
            if states_identical:
                structural_pool = random_pool
                shared_pool_steps += 1
            else:
                structural_pool = generator.generate(
                    structural_state.graph,
                    policy_seed=step_seed,
                    step=step,
                )
                independent_pool_steps += 1
            if not structural_pool.candidates:
                raise RuntimeError("structural trajectory produced an empty legal pool")
            random_trace, random_scoring_ns = _apply_policy_step(
                applied_backend,
                config,
                random_state,
                random_ranker,
                random_pool,
                initial_total=initial_score.total_capped_witnesses,
                step=step,
                horizon=horizon,
                trajectory_started_ns=started_ns,
            )
            structural_trace, structural_scoring_ns = _apply_policy_step(
                applied_backend,
                config,
                structural_state,
                structural_ranker,
                structural_pool,
                initial_total=initial_score.total_capped_witnesses,
                step=step,
                horizon=horizon,
                trajectory_started_ns=started_ns,
            )
            selected_scoring_ns += random_scoring_ns + structural_scoring_ns
            states_diverged = random_state.graph != structural_state.graph
            if states_diverged and divergence_step is None:
                divergence_step = step
            step_records.append(
                {
                    "step": step,
                    "trajectory_seed": step_seed,
                    "states_identical_before_step": states_identical,
                    "same_pool": random_pool.pool_hash == structural_pool.pool_hash,
                    "random": random_trace,
                    "structural": structural_trace,
                    "states_diverged_after_step": states_diverged,
                }
            )
        invalid_graphs = int(
            not applied_backend.validate(random_state.graph).valid
        ) + int(not applied_backend.validate(structural_state.graph).valid)
        base: dict[str, JsonValue] = {
            "schema_version": STAGE2D_EPISODE_VERSION,
            "episode_id": episode["episode_id"],
            "order": order,
            "graph_seed": graph_seed,
            "policy_seed": episode["policy_seed"],
            "horizon": horizon,
            "initial_graph_hash": applied_backend.state_hash(graph),
            "divergence_step": divergence_step,
            "shared_pool_steps": shared_pool_steps,
            "independent_pool_steps": independent_pool_steps,
            "random": _policy_summary(random_state, initial_score),
            "structural": _policy_summary(structural_state, initial_score),
            "steps": cast(list[JsonValue], step_records),
            "initial_score_calls": 1,
            "selected_score_calls": 2 * horizon,
            "oracle_score_calls": 0,
            "exact_verify_calls": 0,
            "invalid_graphs": invalid_graphs,
            "policy_failures": 0,
            "network_calls": 0,
            "model_calls": 0,
            "app_server_calls": 0,
            "timing_ns": {
                "selected_scoring": selected_scoring_ns,
                "episode_total": time.perf_counter_ns() - started_ns,
            },
        }
        canonical = cast(dict[str, JsonValue], _strip_timing(base))
        return {**base, "canonical_episode_sha256": _sha256(canonical)}
    finally:
        if owned_backend:
            applied_backend.close()


def plan_stage2d(config: Stage2DConfig) -> dict[str, JsonValue]:
    manifest = write_manifest(config)
    return {
        "status": "completed",
        "schema_version": STAGE2D_ARTIFACT_VERSION,
        "config_sha256": config.stable_hash(),
        "manifest": str(config.inputs.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "episode_count": manifest["episode_count"],
        "shard_count": manifest["shard_count"],
        "cpu_topology": manifest["cpu_topology"],
        "confirmatory_results_observed": False,
    }


def _preregistration_provenance(
    config: Stage2DConfig,
) -> dict[str, JsonValue]:
    project_state = git_state(config.repositories.project_repo)
    heg_state = git_state(config.repositories.heg_repo)
    tag_type = _git(
        config.repositories.project_repo,
        "cat-file",
        "-t",
        config.repositories.preregistration_tag,
    )
    tag_commit = _git(
        config.repositories.project_repo,
        "rev-list",
        "-n",
        "1",
        config.repositories.preregistration_tag,
    )
    if tag_type != "tag":
        raise RuntimeError("Stage 2D preregistration tag is not annotated")
    if project_state["commit"] != tag_commit or project_state["dirty"]:
        raise RuntimeError("shard must run from a clean detached preregistration checkout")
    if (
        heg_state["commit"] != config.repositories.frozen_heg_commit
        or heg_state["dirty"]
    ):
        raise RuntimeError("HEG pin or clean-state check failed")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must be frozen to {expected}")
    for name in ("TMPDIR", "UV_CACHE_DIR", "XDG_CACHE_HOME"):
        value = os.environ.get(name)
        if value is None or not Path(value).is_absolute() or not Path(value).is_dir():
            raise RuntimeError(f"{name} must name an existing isolated directory")
    return {
        "project": project_state,
        "heg": heg_state,
        "preregistration_tag": config.repositories.preregistration_tag,
        "preregistration_commit": tag_commit,
        "tag_type": tag_type,
        "runtime_network_calls": 0,
        "model_calls": 0,
        "app_server_calls": 0,
        "diagnostic_oracle_enabled": False,
        "stage3_started": False,
    }


def _verify_affinity(shard: dict[str, JsonValue]) -> dict[str, JsonValue]:
    affinity = cast(dict[str, JsonValue], shard["affinity"])
    expected_cpu = cast(int, affinity["cpu_id"])
    if not hasattr(os, "sched_getaffinity"):
        return {"supported": False, "expected_cpu": expected_cpu, "observed_cpus": []}
    observed = sorted(os.sched_getaffinity(0))
    if observed != [expected_cpu]:
        raise RuntimeError(
            f"shard affinity mismatch: expected {[expected_cpu]}, observed {observed}"
        )
    return {
        "supported": True,
        "expected_cpu": expected_cpu,
        "observed_cpus": cast(list[JsonValue], observed),
        "physical_id": affinity["physical_id"],
    }


def _write_episode_file(
    path: Path,
    episodes: list[dict[str, JsonValue]],
    config: Stage2DConfig,
) -> dict[str, JsonValue]:
    raw_hash = hashlib.sha256()
    canonical_hash = hashlib.sha256()
    uncompressed = 0
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle,
    ):
            for episode in episodes:
                encoded = _canonical_bytes(episode) + b"\n"
                if len(encoded) > config.run.max_episode_bytes:
                    raise ValueError("Stage 2D episode exceeds bounded record size")
                if uncompressed + len(encoded) > config.run.max_shard_bytes:
                    raise ValueError("Stage 2D shard exceeds bounded output size")
                canonical = _canonical_bytes(
                    _strip_timing(cast(JsonValue, episode))
                ) + b"\n"
                handle.write(encoded)
                raw_hash.update(encoded)
                canonical_hash.update(canonical)
                uncompressed += len(encoded)
    return {
        "path": path.name,
        "episode_count": len(episodes),
        "uncompressed_bytes": uncompressed,
        "compressed_bytes": path.stat().st_size,
        "raw_episode_sha256": raw_hash.hexdigest(),
        "canonical_episode_sha256": canonical_hash.hexdigest(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _read_episode_file(path: Path) -> list[dict[str, JsonValue]]:
    episodes: list[dict[str, JsonValue]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Stage 2D episode record must be an object")
            episodes.append(cast(dict[str, JsonValue], value))
    return episodes


def run_stage2d_shard(
    config: Stage2DConfig,
    shard_id: str,
    output_dir: Path,
) -> dict[str, JsonValue]:
    manifest = load_manifest(config)
    shards = cast(list[dict[str, JsonValue]], manifest["shards"])
    shard = next((item for item in shards if item["shard_id"] == shard_id), None)
    if shard is None:
        raise ValueError(f"unknown Stage 2D shard {shard_id!r}")
    output_dir.mkdir(parents=True, exist_ok=False)
    terminal_path = output_dir / "terminal_status.json"
    _write_json(
        terminal_path,
        {"status": "failed", "error": "shard interrupted before completion"},
        maximum=config.run.max_episode_bytes,
    )
    try:
        provenance = _preregistration_provenance(config)
        affinity = _verify_affinity(shard)
        random_source = config.inputs.random_policy.read_text()
        structural_source = config.inputs.structural_policy.read_text()
        random_identity = _identity(random_source, config, "random")
        structural_identity = _identity(structural_source, config, "structural")
        episode_by_id = {
            cast(str, episode["episode_id"]): episode
            for episode in cast(list[dict[str, JsonValue]], manifest["episodes"])
        }
        assigned_ids = cast(list[str], shard["episode_ids"])
        episodes: list[dict[str, JsonValue]] = []
        shard_started = time.perf_counter_ns()
        with (
            SourceRanker(
                "random",
                random_source,
                config.stage2b.sandbox,
            ) as random_ranker,
            SourceRanker(
                "structural",
                structural_source,
                config.stage2b.sandbox,
            ) as structural_ranker,
        ):
            for episode_id in assigned_ids:
                episodes.append(
                    run_trajectory_episode(
                        config,
                        episode_by_id[episode_id],
                        random_ranker,
                        structural_ranker,
                    )
                )
            worker_telemetry: dict[str, JsonValue] = {
                "random": random_ranker.telemetry(),
                "structural": structural_ranker.telemetry(),
            }
        file_manifest = _write_episode_file(
            output_dir / "episodes.jsonl.gz",
            episodes,
            config,
        )
        shard_base: dict[str, JsonValue] = {
            "schema_version": STAGE2D_ARTIFACT_VERSION,
            "status": "completed",
            "shard_id": shard_id,
            "assignment_sha256": shard["assignment_sha256"],
            "config_sha256": config.stable_hash(),
            "manifest_sha256": manifest["manifest_sha256"],
            "preregistration_commit": provenance["preregistration_commit"],
            "episode_count": len(episodes),
            "canonical_episode_sha256": file_manifest[
                "canonical_episode_sha256"
            ],
        }
        shard_hash = _sha256(shard_base)
        result: dict[str, JsonValue] = {
            **shard_base,
            "shard_hash": shard_hash,
            "episode_file": file_manifest,
            "ranker_identities": {
                "random": random_identity.as_dict(),
                "structural": structural_identity.as_dict(),
            },
            "provenance": provenance,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "affinity": affinity,
                "thread_environment": cast(
                    dict[str, JsonValue], THREAD_ENVIRONMENT
                ),
                "tmpdir": os.environ.get("TMPDIR"),
                "uv_cache_dir": os.environ.get("UV_CACHE_DIR"),
                "xdg_cache_home": os.environ.get("XDG_CACHE_HOME"),
            },
            "worker_telemetry": worker_telemetry,
            "timing_ns": {
                "shard_total": time.perf_counter_ns() - shard_started,
            },
        }
        _write_json(
            output_dir / "shard_manifest.json",
            shard,
            maximum=config.run.max_episode_bytes,
        )
        _write_json(
            output_dir / "result.json",
            result,
            maximum=config.run.max_episode_bytes,
        )
        _write_json(
            terminal_path,
            {"status": "completed", "shard_hash": shard_hash},
            maximum=config.run.max_episode_bytes,
        )
        return result
    except BaseException as error:
        _write_json(
            terminal_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
            maximum=config.run.max_episode_bytes,
        )
        raise


def _load_shard_result(
    config: Stage2DConfig,
    manifest: dict[str, JsonValue],
    shard: dict[str, JsonValue],
    input_root: Path,
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    shard_id = cast(str, shard["shard_id"])
    shard_dir = input_root / shard_id
    result = cast(
        dict[str, JsonValue],
        json.loads((shard_dir / "result.json").read_text()),
    )
    terminal = cast(
        dict[str, JsonValue],
        json.loads((shard_dir / "terminal_status.json").read_text()),
    )
    if result.get("status") != "completed" or terminal.get("status") != "completed":
        raise ValueError(f"Stage 2D shard {shard_id} is not complete")
    episodes = _read_episode_file(shard_dir / "episodes.jsonl.gz")
    expected_ids = cast(list[str], shard["episode_ids"])
    observed_ids = [cast(str, episode["episode_id"]) for episode in episodes]
    if observed_ids != expected_ids:
        raise ValueError(f"Stage 2D shard {shard_id} episode order mismatch")
    canonical_hash = hashlib.sha256()
    for episode in episodes:
        canonical_hash.update(
            _canonical_bytes(_strip_timing(cast(JsonValue, episode))) + b"\n"
        )
    if result.get("canonical_episode_sha256") != canonical_hash.hexdigest():
        raise ValueError(f"Stage 2D shard {shard_id} canonical record hash mismatch")
    base = {
        key: result[key]
        for key in (
            "schema_version",
            "status",
            "shard_id",
            "assignment_sha256",
            "config_sha256",
            "manifest_sha256",
            "preregistration_commit",
            "episode_count",
            "canonical_episode_sha256",
        )
    }
    if result.get("shard_hash") != _sha256(base):
        raise ValueError(f"Stage 2D shard {shard_id} hash mismatch")
    if (
        result.get("assignment_sha256") != shard["assignment_sha256"]
        or result.get("config_sha256") != config.stable_hash()
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("episode_count") != len(expected_ids)
    ):
        raise ValueError(f"Stage 2D shard {shard_id} provenance mismatch")
    return result, episodes


def reduce_stage2d(
    config: Stage2DConfig,
    input_root: Path,
    output_dir: Path,
    *,
    bootstrap_workers: int,
) -> dict[str, JsonValue]:
    manifest = load_manifest(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    terminal_path = output_dir / "terminal_status.json"
    _write_json(
        terminal_path,
        {"status": "failed", "error": "reduction interrupted before completion"},
        maximum=config.run.max_episode_bytes,
    )
    try:
        episodes: list[dict[str, JsonValue]] = []
        shard_hashes: dict[str, JsonValue] = {}
        preregistration_commits: set[str] = set()
        provenance_pass = True
        isolated_directories: dict[str, set[str]] = {
            "tmpdir": set(),
            "uv_cache_dir": set(),
            "xdg_cache_home": set(),
        }
        affinity_ids: set[str] = set()
        for shard in cast(list[dict[str, JsonValue]], manifest["shards"]):
            result, records = _load_shard_result(
                config, manifest, shard, input_root
            )
            shard_id = cast(str, shard["shard_id"])
            shard_hashes[shard_id] = result["shard_hash"]
            preregistration_commits.add(
                cast(str, result["preregistration_commit"])
            )
            provenance = cast(dict[str, JsonValue], result["provenance"])
            provenance_pass = provenance_pass and bool(
                provenance["runtime_network_calls"] == 0
                and provenance["model_calls"] == 0
                and provenance["app_server_calls"] == 0
                and provenance["diagnostic_oracle_enabled"] is False
                and provenance["stage3_started"] is False
                and cast(dict[str, JsonValue], provenance["heg"])["commit"]
                == config.repositories.frozen_heg_commit
                and cast(dict[str, JsonValue], provenance["heg"])["dirty"] is False
            )
            environment = cast(dict[str, JsonValue], result["environment"])
            for name in isolated_directories:
                value = environment.get(name)
                if isinstance(value, str) and value:
                    isolated_directories[name].add(value)
            affinity = cast(dict[str, JsonValue], environment["affinity"])
            physical_id = affinity.get("physical_id")
            if isinstance(physical_id, str):
                affinity_ids.add(physical_id)
            episodes.extend(records)
        episodes.sort(key=lambda item: cast(str, item["episode_id"]))
        episode_ids = [cast(str, item["episode_id"]) for item in episodes]
        manifest_ids = sorted(
            cast(str, item["episode_id"])
            for item in cast(list[dict[str, JsonValue]], manifest["episodes"])
        )
        exact_coverage = (
            episode_ids == manifest_ids
            and len(episode_ids) == len(set(episode_ids))
        )
        isolated_shards = all(
            len(values) == config.experiment.shard_count
            for values in isolated_directories.values()
        )
        distinct_affinity = len(affinity_ids) == config.experiment.shard_count
        if not exact_coverage:
            raise ValueError("Stage 2D reduction failed exactly-once coverage")
        if len(preregistration_commits) != 1:
            raise ValueError("Stage 2D shards used different preregistration commits")
        canonical_episode_hash = hashlib.sha256()
        for episode in episodes:
            canonical_episode_hash.update(
                _canonical_bytes(_strip_timing(cast(JsonValue, episode))) + b"\n"
            )
        aggregate_sha256 = _sha256(
            [
                _strip_timing(cast(JsonValue, episode))
                for episode in episodes
            ]
        )
        metrics = summarize_episodes(
            episodes,
            config,
            bootstrap_workers=bootstrap_workers,
        )
        reduction_base: dict[str, JsonValue] = {
            "schema_version": STAGE2D_REDUCTION_VERSION,
            "config_sha256": config.stable_hash(),
            "manifest_sha256": manifest["manifest_sha256"],
            "preregistration_commit": next(iter(preregistration_commits)),
            "episode_count": len(episodes),
            "canonical_episode_sha256": canonical_episode_hash.hexdigest(),
            "aggregate_sha256": aggregate_sha256,
            "shard_hashes": shard_hashes,
            "metrics": metrics,
            "validation": {
                "exact_once_coverage": exact_coverage,
                "provenance_pass": provenance_pass,
                "completion_order_independent": True,
                "unique_run_cache_temp_directories": isolated_shards,
                "distinct_physical_core_affinity": distinct_affinity,
            },
        }
        reduction_sha256 = _sha256(reduction_base)
        summary: dict[str, JsonValue] = {
            **reduction_base,
            "status": "completed",
            "reduction_sha256": reduction_sha256,
            "decision": "AWAITING_DETERMINISTIC_REPLAY",
            "execution": {"bootstrap_workers": bootstrap_workers},
        }
        _write_json(
            output_dir / "summary.json",
            summary,
            maximum=config.run.max_shard_bytes,
        )
        _write_json(
            output_dir / "metrics.json",
            metrics,
            maximum=config.run.max_shard_bytes,
        )
        _write_json(
            terminal_path,
            {"status": "completed", "reduction_sha256": reduction_sha256},
            maximum=config.run.max_episode_bytes,
        )
        return summary
    except BaseException as error:
        _write_json(
            terminal_path,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
            maximum=config.run.max_episode_bytes,
        )
        raise


def verify_stage2d_replay(
    primary_summary_path: Path,
    replay_summary_path: Path,
    output_path: Path,
) -> dict[str, JsonValue]:
    primary = cast(
        dict[str, JsonValue], json.loads(primary_summary_path.read_text())
    )
    replay = cast(
        dict[str, JsonValue], json.loads(replay_summary_path.read_text())
    )
    identity_fields = (
        "config_sha256",
        "manifest_sha256",
        "preregistration_commit",
        "episode_count",
        "canonical_episode_sha256",
        "aggregate_sha256",
        "shard_hashes",
        "metrics",
        "reduction_sha256",
    )
    replay_checks: dict[str, JsonValue] = {
        field: primary.get(field) == replay.get(field)
        for field in identity_fields
    }
    validation = cast(dict[str, JsonValue], primary["validation"])
    metrics = cast(dict[str, JsonValue], primary["metrics"])
    gate_without_replay = cast(
        dict[str, JsonValue], metrics["gate_without_replay"]
    )
    full_gate: dict[str, JsonValue] = {
        **gate_without_replay,
        "primary_replay_identity": all(
            cast(bool, value) for value in replay_checks.values()
        ),
        "all_validation_and_provenance_pass": bool(
            validation["exact_once_coverage"]
            and validation["provenance_pass"]
            and validation["unique_run_cache_temp_directories"]
            and validation["distinct_physical_core_affinity"]
            and replay.get("validation") == primary.get("validation")
        ),
    }
    infrastructure_checks = (
        "graph_validity_100_percent",
        "policy_failure_rate_zero",
        "selected_plan_only_scoring_no_oracle",
        "primary_replay_identity",
        "all_validation_and_provenance_pass",
    )
    infrastructure_pass = all(
        cast(bool, full_gate[name]) for name in infrastructure_checks
    )
    if not infrastructure_pass:
        decision = "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"
    elif all(cast(bool, value) for value in full_gate.values()):
        decision = "GO_TO_STAGE_3"
    else:
        decision = "NO_GO"
    result: dict[str, JsonValue] = {
        "schema_version": "stage2d.replay.v1",
        "status": "completed",
        "replay_checks": replay_checks,
        "full_gate": full_gate,
        "decision": decision,
        "primary_reduction_sha256": primary["reduction_sha256"],
        "replay_reduction_sha256": replay["reduction_sha256"],
        "verification_sha256": _sha256(
            {
                "replay_checks": replay_checks,
                "full_gate": full_gate,
                "decision": decision,
                "primary_reduction_sha256": primary["reduction_sha256"],
                "replay_reduction_sha256": replay["reduction_sha256"],
            }
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result, maximum=4 * 1024 * 1024)
    return result
