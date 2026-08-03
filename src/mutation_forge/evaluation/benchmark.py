from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from mutation_forge.artifacts import (
    RunArtifacts,
    canonical_json_hash,
    environment_record,
    git_state,
)
from mutation_forge.backends.heg import HegBackend
from mutation_forge.config import LabConfig
from mutation_forge.counterexamples import (
    CounterexampleDecision,
    CounterexampleHalt,
    CounterexamplePipeline,
    CounterexampleVerified,
)
from mutation_forge.evaluation.dataset import build_dataset
from mutation_forge.evaluation.episode import run_episode
from mutation_forge.evaluation.fitness import aggregate_fitness
from mutation_forge.evaluation.profiling import (
    aggregate_deep_operator_profiles,
    aggregate_deep_score_profiles,
    aggregate_timing_profiles,
)
from mutation_forge.events import EventBus, EventSink, JsonlSink
from mutation_forge.models import EpisodeResult, JsonValue
from mutation_forge.output.rich_live import RichLiveSink
from mutation_forge.policies.baselines import BASELINES
from mutation_forge.run_store import RunStore

RUN_SCHEMA_VERSION = "mforge.experiment.run.v2"


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    run_path: Path
    summary: dict[str, JsonValue]


def _run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"stage1-{timestamp}-{uuid.uuid4().hex[:8]}"


def _canonical_summary_payload(
    *,
    dataset_hash: str,
    episodes: list[EpisodeResult],
    fitness: dict[str, dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    normalized_fitness: dict[str, JsonValue] = {}
    for baseline, metrics in fitness.items():
        normalized = dict(metrics)
        normalized["median_policy_call_ms"] = 0
        ordering_key = cast(list[int | float], normalized["ordering_key"])
        normalized["ordering_key"] = [*ordering_key[:6], 0, *ordering_key[7:]]
        normalized_fitness[baseline] = cast(JsonValue, normalized)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "dataset_manifest_hash": dataset_hash,
        "episodes": [episode.as_dict(include_timing=False) for episode in episodes],
        "fitness": normalized_fitness,
    }


def run_benchmark(config: LabConfig, *, output: str | None = None) -> BenchmarkResult:
    selected_output = output or config.run.output
    if selected_output not in {"rich", "json"}:
        raise ValueError("output must be 'rich' or 'json'")
    run_id = _run_id()
    artifacts = RunArtifacts(config, run_id)
    backend = HegBackend(
        config.heg.repo,
        score_cutoff_enabled=config.search.score_cutoff_enabled,
        prepared_graph_cache_enabled=(config.search.prepared_graph_cache_enabled),
        prepared_proposal_handoff_enabled=(config.search.prepared_proposal_handoff_enabled),
        score_longest_first_enabled=(config.search.score_longest_first_enabled),
        score_compact_dominated_enabled=(config.search.score_compact_dominated_enabled),
        score_prepared_request_cache_enabled=(config.search.score_prepared_request_cache_enabled),
    )
    project_root = Path(__file__).resolve().parents[3]
    project_state = git_state(project_root)
    heg_state: dict[str, JsonValue] = {
        "commit": backend.commit,
        "dirty": backend.dirty,
        "repo": str(backend.repo),
    }
    environment = environment_record(project_root / "uv.lock")
    manifest = artifacts.write_manifest(
        config,
        project_state=project_state,
        heg_state=heg_state,
        environment=environment,
        status="running",
    )
    artifacts.write_json("environment.json", environment)
    store = RunStore(artifacts.path / "archive.sqlite3")
    store.create_run(
        run_id=run_id,
        config_hash=config.stable_hash(),
        manifest=manifest,
    )
    event_file = (artifacts.path / "events.jsonl").open("w")
    sinks: list[EventSink] = [JsonlSink(event_file, close_stream=True), store]
    if selected_output == "json":
        sinks.append(JsonlSink(sys.stdout))
    else:
        sinks.append(RichLiveSink())
    bus = EventBus(run_id, sinks)

    def emit_counterexample(
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        bus.emit(event_type, **dict(payload))

    counterexample_pipeline = CounterexamplePipeline(
        backend=backend,
        artifact_root=artifacts.path,
        event_callback=emit_counterexample,
    )
    started = time.monotonic()
    cpu_started = os.times()
    deadline = started + config.run.wall_seconds
    episodes: list[EpisodeResult] = []
    backend_closed = False

    try:
        bus.emit(
            "run_started",
            stage="stage1",
            config_hash=config.stable_hash(),
            split=config.dataset.split,
        )
        bus.emit(
            "backend_ready",
            backend_id=backend.backend_id,
            heg_commit=backend.commit,
            heg_dirty=backend.dirty,
        )
        dataset = build_dataset(config, backend, heg_commit=backend.commit)
        artifacts.write_json("dataset_manifest.json", dataset)
        entries = cast(list[dict[str, JsonValue]], dataset["entries"])
        bus.emit(
            "dataset_loaded",
            split=config.dataset.split,
            entries=len(entries),
            dataset_manifest_hash=cast(str, dataset["manifest_hash"]),
        )
        bus.emit(
            "checkpoint_written",
            checkpoint="dataset_manifest",
            path=str(artifacts.path / "dataset_manifest.json"),
        )
        for operator_family in config.proposals.operator_families:
            baseline = BASELINES[operator_family]
            bus.emit("baseline_started", baseline=baseline.policy_id)
            baseline_episodes = 0
            for entry in entries:
                for policy_seed in config.dataset.policy_seeds:
                    initial_graph = backend.deserialize_graph6(cast(str, entry["graph6"]))
                    if time.monotonic() >= deadline:
                        raise TimeoutError("run wall-time budget exhausted")
                    bus.emit(
                        "episode_started",
                        baseline=baseline.policy_id,
                        entry_id=cast(str, entry["entry_id"]),
                        graph_seed=cast(int, entry["graph_seed"]),
                        policy_seed=policy_seed,
                        run_seed=config.run.seed,
                    )

                    def emit_progress(payload: dict[str, JsonValue]) -> None:
                        bus.emit("episode_progress", **payload)

                    episode = run_episode(
                        backend=backend,
                        initial_graph=initial_graph,
                        entry_id=cast(str, entry["entry_id"]),
                        graph_seed=cast(int, entry["graph_seed"]),
                        policy_seed=policy_seed,
                        run_seed=config.run.seed,
                        baseline=baseline,
                        evaluations=config.search.evaluations_per_episode,
                        witness_cap=config.score.witness_cap,
                        deadline=deadline,
                        progress=emit_progress,
                        profiling_enabled=config.search.profiling_enabled,
                        deep_profiling_enabled=(config.search.deep_profiling_enabled),
                        score_cache_enabled=config.search.score_cache_enabled,
                        counterexample_pipeline=counterexample_pipeline,
                    )
                    if episode.timed_out:
                        raise TimeoutError("episode wall-time budget exhausted")
                    if episode.score_failures:
                        raise RuntimeError(
                            f"episode encountered {episode.score_failures} score failures"
                        )
                    validation = backend.validate(backend.deserialize_graph6(episode.final_graph6))
                    if not validation.valid:
                        raise RuntimeError(
                            "episode produced an HEG-invalid result graph: "
                            + "; ".join(validation.errors)
                        )
                    episodes.append(episode)
                    baseline_episodes += 1
                    cumulative_timing_profile = aggregate_timing_profiles(
                        ((item.baseline, item.timing_profile) for item in episodes),
                        enabled=config.search.profiling_enabled,
                    )
                    cumulative_deep_profile = aggregate_deep_operator_profiles(
                        (item.deep_operator_profile for item in episodes),
                        enabled=config.search.deep_profiling_enabled,
                    )
                    cumulative_deep_score_profile = aggregate_deep_score_profiles(
                        item.deep_score_profile for item in episodes
                    )
                    episode_payload: dict[str, JsonValue] = {
                        "baseline": baseline.policy_id,
                        "entry_id": episode.entry_id,
                        "graph_seed": episode.graph_seed,
                        "policy_seed": episode.policy_seed,
                        "evaluations": episode.evaluations,
                        "initial_total": (episode.initial_score.total_capped_witnesses),
                        "best_total": episode.best_score.total_capped_witnesses,
                        "legal_proposals": episode.legal_proposals,
                        "invalid_proposals": episode.invalid_proposals,
                        "episodes_completed": baseline_episodes,
                        "episode_timing_profile": (
                            episode.timing_profile.as_dict()
                            if episode.timing_profile is not None
                            else None
                        ),
                        "timing_profile": cumulative_timing_profile,
                        "episode_deep_operator_profile": (
                            episode.deep_operator_profile.as_dict()
                            if episode.deep_operator_profile is not None
                            else None
                        ),
                        "deep_operator_profile": cumulative_deep_profile,
                    }
                    if episode.deep_score_profile is not None:
                        episode_payload["episode_deep_score_profile"] = (
                            episode.deep_score_profile.as_dict()
                        )
                    if cumulative_deep_score_profile is not None:
                        episode_payload["deep_score_profile"] = cumulative_deep_score_profile
                    bus.emit("episode_completed", **episode_payload)

        fitness = {
            baseline: aggregate_fitness(
                [episode for episode in episodes if episode.baseline == baseline]
            )
            for baseline in config.proposals.operator_families
        }
        timing_profile = aggregate_timing_profiles(
            ((episode.baseline, episode.timing_profile) for episode in episodes),
            enabled=config.search.profiling_enabled,
        )
        deep_operator_profile = aggregate_deep_operator_profiles(
            (episode.deep_operator_profile for episode in episodes),
            enabled=config.search.deep_profiling_enabled,
        )
        deep_score_profile = aggregate_deep_score_profiles(
            episode.deep_score_profile for episode in episodes
        )
        canonical_payload = _canonical_summary_payload(
            dataset_hash=cast(str, dataset["manifest_hash"]),
            episodes=episodes,
            fitness=fitness,
        )
        summary_hash = canonical_json_hash(canonical_payload)
        backend.close()
        backend_closed = True
        cpu_finished = os.times()
        elapsed = time.monotonic() - started
        user_seconds = max(
            0.0,
            cpu_finished.user
            + cpu_finished.children_user
            - cpu_started.user
            - cpu_started.children_user,
        )
        system_seconds = max(
            0.0,
            cpu_finished.system
            + cpu_finished.children_system
            - cpu_started.system
            - cpu_started.children_system,
        )
        summary: dict[str, JsonValue] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "dataset_manifest_hash": cast(str, dataset["manifest_hash"]),
            "episodes": [episode.as_dict() for episode in episodes],
            "fitness": cast(dict[str, JsonValue], fitness),
            "timing_profile": timing_profile,
            "deep_operator_profile": deep_operator_profile,
            "run_id": run_id,
            "status": "completed",
            "summary_hash": summary_hash,
            "elapsed_seconds": elapsed,
            "real_seconds": elapsed,
            "user_seconds": user_seconds,
            "system_seconds": system_seconds,
            "evaluations": sum(episode.evaluations for episode in episodes),
            "evaluations_per_second": sum(episode.evaluations for episode in episodes)
            / max(elapsed, 1e-9),
            "score_implementation": backend.score_implementation,
        }
        if deep_score_profile is not None:
            summary["deep_score_profile"] = deep_score_profile
        artifacts.write_json("run_summary.json", summary)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        artifacts.write_json("run_manifest.json", manifest)
        store.finish(run_id, "completed", cast(dict[str, object], summary))
        completed_payload: dict[str, JsonValue] = {
            "status": "completed",
            "run_path": str(artifacts.path),
            "summary_hash": summary_hash,
            "evaluations": cast(int, summary["evaluations"]),
            "evaluations_per_second": cast(float, summary["evaluations_per_second"]),
            "real_seconds": elapsed,
            "user_seconds": user_seconds,
            "system_seconds": system_seconds,
            "timing_profile": timing_profile,
            "deep_operator_profile": deep_operator_profile,
        }
        if deep_score_profile is not None:
            completed_payload["deep_score_profile"] = deep_score_profile
        bus.emit("run_completed", **completed_payload)
        return BenchmarkResult(artifacts.path, summary)
    except CounterexampleHalt as halted:
        if not backend_closed:
            backend.close()
            backend_closed = True
        elapsed = time.monotonic() - started
        candidate = halted.outcome.candidate
        status = (
            "failed"
            if halted.outcome.decision is CounterexampleDecision.FAIL
            else "paused"
        )
        summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "status": status,
            "stop_reason": halted.outcome.stop_reason,
            "candidate_id": candidate.candidate_id if candidate is not None else None,
            "elapsed_seconds": elapsed,
        }
        terminal_checkpoint = {
            "schema_version": "mforge.experiment.checkpoint.v2",
            "state": status,
            "stop_reason": halted.outcome.stop_reason,
            "candidate_id": summary["candidate_id"],
        }
        terminal_checkpoint_path = artifacts.write_json(
            "terminal-checkpoint.json",
            terminal_checkpoint,
        )
        summary["terminal_checkpoint"] = str(terminal_checkpoint_path)
        artifacts.write_json("run_summary.json", summary)
        manifest["status"] = status
        manifest["stop_reason"] = halted.outcome.stop_reason
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        artifacts.write_json("run_manifest.json", manifest)
        store.finish(run_id, status, cast(dict[str, object], summary))
        bus.emit(
            "run_failed" if status == "failed" else "budget_boundary_reached",
            status=status,
            stop_reason=halted.outcome.stop_reason,
            candidate_id=summary["candidate_id"],
            run_path=str(artifacts.path),
            real_seconds=elapsed,
        )
        return BenchmarkResult(artifacts.path, summary)
    except CounterexampleVerified as verified:
        if not backend_closed:
            backend.close()
            backend_closed = True
        certificate = verified.outcome.certificate
        candidate = verified.outcome.candidate
        elapsed = time.monotonic() - started
        summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "completed",
            "stop_reason": "counterexample_verified",
            "candidate_id": candidate.candidate_id if candidate else None,
            "certificate": str(certificate.artifact_path) if certificate else None,
            "certificate_sha256": certificate.sha256 if certificate else None,
            "elapsed_seconds": elapsed,
        }
        terminal_checkpoint = {
            "schema_version": "mforge.experiment.checkpoint.v2",
            "state": "completed",
            "stop_reason": "counterexample_verified",
            "candidate_id": summary["candidate_id"],
            "certificate_sha256": summary["certificate_sha256"],
        }
        terminal_checkpoint_path = artifacts.write_json(
            "terminal-checkpoint.json",
            terminal_checkpoint,
        )
        summary["terminal_checkpoint"] = str(terminal_checkpoint_path)
        artifacts.write_json("run_summary.json", summary)
        manifest["status"] = "completed"
        manifest["stop_reason"] = "counterexample_verified"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        artifacts.write_json("run_manifest.json", manifest)
        store.finish(run_id, "completed", cast(dict[str, object], summary))
        bus.emit(
            "counterexample_verified",
            candidate_id=summary["candidate_id"],
            certificate=summary["certificate"],
            certificate_sha256=summary["certificate_sha256"],
            stop_reason="counterexample_verified",
            checkpoint=summary["terminal_checkpoint"],
            idempotency_key=f"{summary['candidate_id']}:verified",
        )
        bus.emit(
            "experiment_completed",
            status="completed",
            stop_reason="counterexample_verified",
            checkpoint=summary["terminal_checkpoint"],
            run_path=str(artifacts.path),
            idempotency_key=f"{summary['candidate_id']}:experiment-completed",
        )
        bus.emit(
            "run_completed",
            status="completed",
            stop_reason="counterexample_verified",
            run_path=str(artifacts.path),
            real_seconds=elapsed,
        )
        return BenchmarkResult(artifacts.path, summary)
    except Exception as error:
        if not backend_closed:
            backend.close()
            backend_closed = True
        cpu_finished = os.times()
        elapsed = time.monotonic() - started
        user_seconds = max(
            0.0,
            cpu_finished.user
            + cpu_finished.children_user
            - cpu_started.user
            - cpu_started.children_user,
        )
        system_seconds = max(
            0.0,
            cpu_finished.system
            + cpu_finished.children_system
            - cpu_started.system
            - cpu_started.children_system,
        )
        failure: dict[str, JsonValue] = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": elapsed,
            "real_seconds": elapsed,
            "user_seconds": user_seconds,
            "system_seconds": system_seconds,
        }
        artifacts.write_json("run_summary.json", failure)
        manifest["status"] = "failed"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        artifacts.write_json("run_manifest.json", manifest)
        store.finish(run_id, "failed", cast(dict[str, object], failure))
        bus.emit(
            "run_failed",
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            run_path=str(artifacts.path),
            real_seconds=elapsed,
            user_seconds=user_seconds,
            system_seconds=system_seconds,
        )
        raise
    finally:
        if not backend_closed:
            backend.close()
        bus.close()


def load_summary(path: Path) -> dict[str, JsonValue]:
    summary_path = path / "run_summary.json" if path.is_dir() else path
    return cast(dict[str, JsonValue], json.loads(summary_path.read_text()))
