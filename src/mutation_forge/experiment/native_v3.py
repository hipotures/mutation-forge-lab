"""Production Native v3 AST search adapter."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import GraphState, JsonValue
from mutation_forge.native_v3.baselines import load_baseline_programs
from mutation_forge.native_v3.canonical import canonical_json_bytes, domain_hash
from mutation_forge.native_v3.contracts import ValidatedProgram, validate_program
from mutation_forge.native_v3.evaluation import (
    ApparentZero,
    EpisodeResult,
    EpisodeStatus,
    ShardInfrastructureFailure,
    StepRecord,
    make_heg_shard_evaluator,
)
from mutation_forge.native_v3.heg_scoring import backend_identity
from mutation_forge.native_v3.interpreter import INTERPRETER_PROTOCOL_ID
from mutation_forge.native_v3.persistence import (
    NativeV3Persistence,
    SemanticRecord,
    TelemetryRecord,
)
from mutation_forge.native_v3.provider import (
    NativeV3Provider,
    ProviderArtifact,
    ProviderInputProfile,
    ProviderOutputProfile,
    ProviderRawArtifact,
    ProviderSlotSpec,
    build_provider_request,
)
from mutation_forge.native_v3.scheduler import (
    EpisodeShard,
    EpisodeSpec,
    EpisodeTask,
    EpochSnapshot,
    EpochStatus,
    GeneratedEntry,
    ProviderCall,
    SchedulerConfig,
    StreamingEpochScheduler,
    TelemetryEvent,
    build_episode_shards,
    split_residual_shard,
)
from mutation_forge.native_v3.scoring import (
    ACCEPTANCE_PROTOCOL_ID,
    FITNESS_PROTOCOL_ID,
    SCORE_PROTOCOL_ID,
    RationalInterval,
    aggregate_order_balanced,
)
from mutation_forge.native_v3.selection import (
    ProgramFitness,
    freeze_promotion_shortlist,
    missing_current_manifest_evaluations,
    validated_global_best,
)
from mutation_forge.native_v3.telemetry import summarize_scheduler_telemetry
from mutation_forge.native_v3.verification import (
    VerificationDecision,
    VerificationJob,
    VerificationOutcome,
    VerificationSupervisor,
    graph_content_hash,
    verify_heg_primary,
    verify_independent_python,
)

from .config import ExperimentConfig, orders_for_generation
from .control import ExperimentControl
from .layout import ExperimentLayout, WorkspaceError
from .provider import LocalCodexAppServerProvider
from .sessions import SessionContext
from .state import ExperimentStateStore

NATIVE_V3_PROTOCOL_BUNDLE = "native_v3_protocol_bundle_v1"
NATIVE_V3_RUN_SCHEMA = "mforge.native.run.v3"
_BASELINE_IDS = frozenset(
    {
        "add-low-local-cycle-risk",
        "remove-low-bridge-risk",
        "random-valid",
        "degree-fanout",
    }
)
_BRIEFS = (
    "Explore a degree-changing add-edge strategy with low local cycle risk.",
    "Explore a safe remove-edge strategy that avoids structural bridges.",
    "Explore endpoint relocation while preserving a useful dense core.",
    "Explore fanout or fold operations that change the degree vector.",
    "Explore 2-, 3-, or 4-switches as one part of a broader strategy.",
    "Adapt operator choice to stagnation and the exploration-window context.",
    "Combine witness-load selectors with spatial or articulation-risk selectors.",
    "Favor a compact, diverse mechanism unlike the supplied parents.",
)


@dataclass(frozen=True, slots=True)
class _Panel:
    name: str
    manifest_hash: str
    episodes: tuple[EpisodeSpec, ...]
    initial_hashes: Mapping[tuple[int, int], str]


def _fraction(value: Fraction) -> dict[str, JsonValue]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _interval(value: RationalInterval) -> dict[str, JsonValue]:
    return {"lower": _fraction(value.lower), "upper": _fraction(value.upper)}


def _parse_fraction(value: object) -> Fraction:
    if not isinstance(value, Mapping):
        raise ValueError("fraction must be an object")
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _parse_interval(value: object) -> RationalInterval:
    if not isinstance(value, Mapping):
        raise ValueError("interval must be an object")
    return RationalInterval(
        _parse_fraction(value["lower"]),
        _parse_fraction(value["upper"]),
    )


def _episode_payload(result: EpisodeResult) -> dict[str, JsonValue]:
    return {
        "episode_id": result.episode_id,
        "program_hash": result.program_hash,
        "status": result.status.value,
        "order": result.order,
        "graph_seed": result.graph_seed,
        "policy_seed": result.policy_seed,
        "trajectory": [_interval(value) for value in result.trajectory],
        "auc": _interval(result.auc) if result.auc is not None else None,
        "failure": result.failure,
        "steps": [
            {
                "step_index": step.step_index,
                "proposed": step.proposed,
                "accepted": step.accepted,
                "strict_improvement": step.strict_improvement,
                "exploration_window_index": step.exploration_window_index,
                "incumbent_graph_hash": step.incumbent_graph_hash,
                "utility_interval": _interval(step.utility_interval),
                "no_plan_reason": step.no_plan_reason,
            }
            for step in result.steps
        ],
        "apparent_zero_graph_hashes": [candidate.graph_hash for candidate in result.apparent_zeros],
        "raw_graph_score_calls": result.raw_graph_score_calls,
        "unique_graph_scores": result.unique_graph_scores,
        "score_cache_hits": result.score_cache_hits,
        "accepted_rewrites": result.accepted_rewrites,
        "scorer_restarts": result.scorer_restarts,
    }


def _parse_episode(payload: Mapping[str, JsonValue]) -> EpisodeResult:
    order = int(cast(int, payload["order"]))
    trajectory_value = payload.get("trajectory")
    steps_value = payload.get("steps")
    zero_hashes = payload.get("apparent_zero_graph_hashes")
    if (
        not isinstance(trajectory_value, list)
        or not isinstance(steps_value, list)
        or not isinstance(zero_hashes, list)
    ):
        raise ValueError("persisted episode arrays are invalid")
    steps: list[StepRecord] = []
    for value in steps_value:
        if not isinstance(value, Mapping):
            raise ValueError("persisted episode step is invalid")
        window = value.get("exploration_window_index")
        steps.append(
            StepRecord(
                int(cast(int, value["step_index"])),
                value.get("proposed") is True,
                value.get("accepted") is True,
                value.get("strict_improvement") is True,
                int(window) if isinstance(window, int) and not isinstance(window, bool) else None,
                str(value["incumbent_graph_hash"]),
                _parse_interval(value["utility_interval"]),
                (
                    str(value["no_plan_reason"])
                    if isinstance(value.get("no_plan_reason"), str)
                    else None
                ),
            )
        )
    auc_value = payload.get("auc")
    return EpisodeResult(
        str(payload["episode_id"]),
        str(payload["program_hash"]),
        EpisodeStatus(str(payload["status"])),
        order,
        int(cast(int, payload["graph_seed"])),
        int(cast(int, payload["policy_seed"])),
        tuple(_parse_interval(value) for value in trajectory_value),
        _parse_interval(auc_value) if isinstance(auc_value, Mapping) else None,
        tuple(steps),
        tuple(
            ApparentZero(
                str(graph_hash),
                GraphState(order, ()),
                {"source": "durably_committed_episode"},
            )
            for graph_hash in zero_hashes
        ),
        str(payload["failure"]) if isinstance(payload.get("failure"), str) else None,
        int(cast(int, payload.get("raw_graph_score_calls", 0))),
        int(cast(int, payload.get("unique_graph_scores", 0))),
        int(cast(int, payload.get("score_cache_hits", 0))),
        int(cast(int, payload.get("accepted_rewrites", 0))),
        int(cast(int, payload.get("scorer_restarts", 0))),
    )


def _fitness_payload(value: ProgramFitness) -> dict[str, JsonValue]:
    return {
        "program_hash": value.program_hash,
        "manifest_hash": value.manifest_hash,
        "protocol_bundle_hash": value.protocol_bundle_hash,
        "interval": _interval(value.interval),
        "exact_episode_count": value.exact_episode_count,
        "total_episode_count": value.total_episode_count,
        "behavior_signature": value.behavior_signature,
        "archive_eligible": value.archive_eligible,
    }


def _parse_fitness(payload: Mapping[str, JsonValue]) -> ProgramFitness:
    return ProgramFitness(
        str(payload["program_hash"]),
        str(payload["manifest_hash"]),
        str(payload["protocol_bundle_hash"]),
        _parse_interval(payload["interval"]),
        int(cast(int, payload["exact_episode_count"])),
        int(cast(int, payload["total_episode_count"])),
        str(payload["behavior_signature"]),
        payload.get("archive_eligible") is not False,
    )


def _program_fitness(
    *,
    program_hash: str,
    manifest_hash: str,
    protocol_bundle_hash: str,
    results: Sequence[EpisodeResult],
    expected_episodes: int,
) -> ProgramFitness | None:
    if len(results) != expected_episodes or any(
        result.status
        in {
            EpisodeStatus.PROGRAM_FAILURE,
            EpisodeStatus.INFRASTRUCTURE_INCONCLUSIVE,
        }
        or result.auc is None
        for result in results
    ):
        return None
    by_order: dict[int, list[RationalInterval]] = defaultdict(list)
    for result in results:
        assert result.auc is not None
        by_order[result.order].append(result.auc)
    fitness = aggregate_order_balanced(by_order)
    exact_count = sum(result.auc is not None and result.auc.exact for result in results)
    signature = domain_hash(
        b"mforge-native-v3-behavior\0",
        canonical_json_bytes(
            [
                [
                    result.order,
                    sum(step.accepted for step in result.steps),
                    sum(step.strict_improvement for step in result.steps),
                    len(result.apparent_zeros),
                ]
                for result in sorted(
                    results,
                    key=lambda item: (
                        item.order,
                        item.graph_seed,
                        item.policy_seed,
                    ),
                )
            ]
        ),
    )
    return ProgramFitness(
        program_hash,
        manifest_hash,
        protocol_bundle_hash,
        fitness,
        exact_count,
        len(results),
        signature,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class NativeV3ExperimentAdapter:
    """Run the active Native v3 streaming FunSearch-like workflow."""

    def __init__(
        self,
        *,
        provider: Any | None = None,
    ) -> None:
        self.provider = provider

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _assets(cls) -> dict[str, Path]:
        root = cls._project_root()
        return {
            "system": root / "prompts/native/system.md",
            "request": root / "prompts/native/request.md",
            "repair": root / "prompts/native/repair.md",
            "output_schema": root / "configs/native/generated-program-batch.schema.json",
            "program_schema": root / "configs/native/native-v3-program.schema.json",
            "context_schema": root / "configs/native/native-v3-context.schema.json",
            "selectors": root / "configs/native/native-v3-selector-registry.json",
            "actions": root / "configs/native/native-v3-action-registry.json",
            "semantics": root / "configs/native/native-v3-semantics.md",
            "baselines": root / "configs/native/native-v3-baseline-programs.json",
        }

    @staticmethod
    def _default_provider(config: ExperimentConfig) -> LocalCodexAppServerProvider:
        return LocalCodexAppServerProvider(
            model=config.model.name,
            effort=config.model.effort,
            concurrency=config.model.concurrency,
            max_repairs=config.model.max_repairs,
            turn_timeout_base_seconds=config.run.turn_timeout_base_seconds,
            auth_json=config.model.auth_json,
            persist_artifacts=False,
        )

    def preflight(self, config: ExperimentConfig) -> Mapping[str, Any]:
        if config.kind != "heg" or config.preset != "native":
            raise WorkspaceError("Native v3 supports only kind='heg', preset='native'")
        if config.model.provider != "codex":
            raise WorkspaceError("Native v3 requires the local Codex provider")
        if config.search.population_size != 8:
            raise WorkspaceError("Native v3 epoch cohorts require population_size=8")
        if config.model.max_repairs not in {0, 1}:
            raise WorkspaceError("Native v3 permits at most one frozen full-batch repair")
        if self.provider is None:
            provider = self._default_provider(config)
            try:
                provider.preflight()
            except Exception as error:
                raise WorkspaceError(f"Codex authentication preflight failed: {error}") from error
            finally:
                provider.close()
        configured_baselines = set(config.evaluation.baselines)
        if configured_baselines != _BASELINE_IDS:
            raise WorkspaceError(
                f"Native v3 requires the four locked DSL baselines: {sorted(_BASELINE_IDS)}"
            )
        assets = self._assets()
        missing = [str(path) for path in assets.values() if not path.is_file()]
        if missing:
            raise WorkspaceError(f"Native v3 assets are missing: {missing}")
        try:
            output_schema = json.loads(assets["output_schema"].read_text(encoding="utf-8"))
            for name in ("program_schema", "context_schema", "selectors", "actions"):
                json.loads(assets[name].read_text(encoding="utf-8"))
            load_baseline_programs(assets["baselines"])
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise WorkspaceError(f"invalid Native v3 assets: {error}") from error
        heg_repo = self._project_root().parent / "heg"
        from mutation_forge.backends.heg import HegBackend

        try:
            backend = HegBackend(
                heg_repo,
                graph_mode=config.evaluation.graph_mode,
                score_timeout_seconds=20.0,
            )
            identity = backend_identity(backend)
            backend.close()
        except Exception as error:
            raise WorkspaceError(f"HEG C++ scorer preflight failed: {error}") from error
        slots = tuple(f"slot-{index:02d}" for index in range(8))
        snapshot = EpochSnapshot(
            "preflight",
            0,
            (),
            "preflight-archive",
            "preflight-development",
            "preflight-protocol",
            slots,
        )
        call_slots = slots[: config.native_v3.provider_batch_size]
        request = build_provider_request(
            call=ProviderCall("preflight-call", call_slots, snapshot),
            slots=tuple(
                ProviderSlotSpec(slot_id, (), _BRIEFS[index])
                for index, slot_id in enumerate(call_slots)
            ),
            parent_programs={},
            archive_summary={},
            system_prompt=assets["system"].read_text(encoding="utf-8"),
            output_schema=cast(Mapping[str, Any], output_schema),
            contract_bundle={
                name: json.loads(assets[name].read_text(encoding="utf-8"))
                for name in ("program_schema", "context_schema", "selectors", "actions")
            },
            request_prompt=assets["request"].read_text(encoding="utf-8"),
            repair_prompt=assets["repair"].read_text(encoding="utf-8"),
            input_profile=ProviderInputProfile(),
            output_profile=ProviderOutputProfile(),
        )
        return {
            "native_v3": {
                "protocol_bundle": NATIVE_V3_PROTOCOL_BUNDLE,
                "actual_preflight_request_bytes": len(request.encoded_bytes),
                "conservative_preflight_input_tokens": request.conservative_token_bound,
                "provider_input_profile": "4 AST / 128 KiB / 32k tokens",
                "verification_profile": "1 / 16 / 10m / 4GiB",
                "selector_cost_units_per_propose": 128,
            },
            "assets": {
                name: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in assets.items()
            },
            "heg": identity.as_dict(),
        }

    @staticmethod
    def model_turns_used(
        _layout: ExperimentLayout,
        state: ExperimentStateStore,
    ) -> int:
        return int(state.cumulative().get("provider_turns", 0))

    def _panel(
        self,
        *,
        name: str,
        orders: Sequence[int],
        graph_seeds: Sequence[int],
        policy_seeds: Sequence[int],
        backend: Any,
        protocol_bundle_hash: str,
    ) -> _Panel:
        initial_hashes = {
            (order, graph_seed): graph_content_hash(
                backend.generate_seed(order=order, seed=graph_seed)
            )
            for order in orders
            for graph_seed in graph_seeds
        }
        episodes = tuple(
            EpisodeSpec(order, graph_seed, policy_seed, name)
            for order in orders
            for graph_seed in graph_seeds
            for policy_seed in policy_seeds
        )
        payload = {
            "name": name,
            "orders": list(orders),
            "graph_seeds": list(graph_seeds),
            "policy_seeds": list(policy_seeds),
            "initial_graph_hashes": [
                [order, seed, initial_hashes[(order, seed)]]
                for order, seed in sorted(initial_hashes)
            ],
            "protocol_bundle_hash": protocol_bundle_hash,
        }
        return _Panel(
            name,
            domain_hash(
                b"mforge-native-v3-panel\0",
                canonical_json_bytes(payload),
            ),
            episodes,
            initial_hashes,
        )

    @staticmethod
    def _tasks_for(
        *,
        program_hash: str,
        panel: _Panel,
        horizon: int,
        protocol_bundle_hash: str,
    ) -> tuple[EpisodeTask, ...]:
        return tuple(
            EpisodeTask(
                program_hash,
                panel.manifest_hash,
                protocol_bundle_hash,
                panel.initial_hashes[(episode.order, episode.graph_seed)],
                horizon,
                INTERPRETER_PROTOCOL_ID,
                "native_v3_selectors_v1",
                SCORE_PROTOCOL_ID,
                ACCEPTANCE_PROTOCOL_ID,
                episode,
            )
            for episode in panel.episodes
        )

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
        *,
        observer: Any | None = None,
        effective_max_model_turns: int | None = None,
        control: ExperimentControl | None = None,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        assets = self._assets()
        output_schema = cast(
            Mapping[str, Any],
            json.loads(assets["output_schema"].read_text(encoding="utf-8")),
        )
        provider_contract = {
            name: json.loads(assets[name].read_text(encoding="utf-8"))
            for name in ("program_schema", "context_schema", "selectors", "actions")
        }
        system_prompt = assets["system"].read_text(encoding="utf-8")
        request_prompt = assets["request"].read_text(encoding="utf-8")
        repair_prompt = assets["repair"].read_text(encoding="utf-8")
        baseline_programs = load_baseline_programs(assets["baselines"])
        heg_repo = self._project_root().parent / "heg"
        provider = self.provider or self._default_provider(config)
        if self.provider is None:
            try:
                provider.preflight()
            except Exception as error:
                provider.close()
                raise RuntimeError(f"Codex authentication preflight failed: {error}") from error
        artifact_lock = threading.Lock()
        provider_artifacts: list[tuple[ProviderArtifact, Path]] = []
        generated_programs: dict[str, ValidatedProgram] = {}
        verification_futures: list[Future[VerificationOutcome]] = []
        verified_outcome: VerificationOutcome | None = None
        semantic_path = layout.artifacts / "native-v3-state.sqlite3"

        def emit(event_type: str, **payload: JsonValue) -> None:
            if observer is not None:
                observer.emit(event_type, **payload)

        with NativeV3Persistence(
            semantic_path,
            queue_capacity=2 * config.resources.thread_count,
        ) as persistence:
            programs: dict[str, ValidatedProgram] = {}
            fitness_cache: dict[tuple[str, str, str], ProgramFitness] = {}
            for record in persistence.semantic_records("program"):
                canonical = record.payload.get("program_json_canonical")
                if not isinstance(canonical, str):
                    continue
                validation = validate_program(canonical)
                if (
                    validation.program is not None
                    and validation.program.program_hash == record.semantic_key
                ):
                    programs[record.semantic_key] = validation.program
            for record in persistence.semantic_records("fitness"):
                try:
                    fitness = _parse_fitness(record.payload)
                except (KeyError, TypeError, ValueError):
                    continue
                fitness_cache[fitness.cache_key] = fitness
            committed_episodes: dict[str, dict[str, EpisodeResult]] = {
                "development": {},
                "validation": {},
            }
            for phase in committed_episodes:
                for record in persistence.semantic_records(f"episode:{phase}"):
                    result = _parse_episode(record.payload)
                    if result.episode_id != record.semantic_key:
                        raise RuntimeError(f"persisted {phase} episode identity is inconsistent")
                    committed_episodes[phase][result.episode_id] = result

            from mutation_forge.backends.heg import HegBackend

            identity_backend = HegBackend(
                heg_repo,
                graph_mode=config.evaluation.graph_mode,
                score_timeout_seconds=20.0,
            )
            identity = backend_identity(identity_backend)
            protocol_bundle_hash = domain_hash(
                b"mforge-native-v3-protocol-bundle\0",
                canonical_json_bytes(
                    {
                        "protocol": NATIVE_V3_PROTOCOL_BUNDLE,
                        "interpreter": INTERPRETER_PROTOCOL_ID,
                        "score": SCORE_PROTOCOL_ID,
                        "fitness": FITNESS_PROTOCOL_ID,
                        "acceptance": ACCEPTANCE_PROTOCOL_ID,
                        "backend": identity.as_dict(),
                        "witness_cap": config.native_v3.witness_cap,
                    }
                ),
            )
            identity_backend.close()
            primary = partial(
                verify_heg_primary,
                heg_repo=heg_repo,
                graph_mode=config.evaluation.graph_mode,
            )

            def verification_telemetry(
                name: str,
                fields: Mapping[str, JsonValue],
            ) -> None:
                emit(name, **dict(fields))
                persistence.record_telemetry(TelemetryRecord(name, time.time_ns(), fields))

            verification = VerificationSupervisor(
                artifact_root=layout.artifacts / "counterexamples-v3",
                primary_verifier=primary,
                independent_verifier=verify_independent_python,
                telemetry_sink=verification_telemetry,
            )
            verification_futures.extend(verification.recover_pending())

            def observe_verification(outcome: VerificationOutcome) -> None:
                nonlocal verified_outcome
                if outcome.decision is not VerificationDecision.VERIFIED:
                    return
                if verified_outcome is None or (
                    outcome.graph_hash,
                    outcome.verification_protocol_id,
                ) < (
                    verified_outcome.graph_hash,
                    verified_outcome.verification_protocol_id,
                ):
                    verified_outcome = outcome

            def submit_result(result: EpisodeResult, phase: str) -> None:
                started = time.monotonic_ns()
                # Every apparent zero crosses the supervisor's durable
                # candidate boundary before the episode itself becomes
                # terminal. Recovery may therefore skip a committed episode
                # without losing a verification obligation.
                for candidate in result.apparent_zeros:
                    future = verification.submit(
                        VerificationJob(
                            candidate.graph_hash,
                            candidate.graph,
                            candidate.provenance,
                        )
                    )
                    verification_futures.append(future)
                persistence.commit_semantic(
                    SemanticRecord(
                        f"episode:{phase}",
                        result.episode_id,
                        _episode_payload(result),
                    )
                )
                persistence.record_telemetry(
                    TelemetryRecord(
                        "episode_committed",
                        time.time_ns(),
                        {
                            "episode_id": result.episode_id,
                            "phase": phase,
                            "persistence_wall_ns": time.monotonic_ns() - started,
                        },
                    )
                )
                for future in verification_futures:
                    if not future.done():
                        continue
                    observe_verification(future.result())

            completed_epoch_records = persistence.semantic_records("epoch_terminal")
            epoch_number = len(completed_epoch_records)
            model_turns_used = max(
                self.model_turns_used(layout, state),
                len(persistence.semantic_records("provider_batch")),
            )
            last_epoch_status: EpochStatus | None = None
            global_best: ProgramFitness | None = None
            sealed_validation_orders = orders_for_generation(config.evaluation, 0)

            try:
                while (
                    config.search.max_generations is None
                    or epoch_number < config.search.max_generations
                ):
                    if session.budget_exhausted():
                        break
                    if control is not None and control.graceful_stop_requested:
                        break
                    if (
                        effective_max_model_turns is not None
                        and model_turns_used >= effective_max_model_turns
                    ):
                        break
                    epoch_number += 1
                    epoch_persistence_wall_start_ns = persistence.total_wall_time_ns
                    orders = orders_for_generation(config.evaluation, epoch_number - 1)
                    panel_backend = HegBackend(
                        heg_repo,
                        graph_mode=config.evaluation.graph_mode,
                    )
                    try:
                        development = self._panel(
                            name="development",
                            orders=orders,
                            graph_seeds=config.evaluation.graph_seeds,
                            policy_seeds=config.evaluation.policy_seeds,
                            backend=panel_backend,
                            protocol_bundle_hash=protocol_bundle_hash,
                        )
                        validation_panel = self._panel(
                            name="validation",
                            orders=sealed_validation_orders,
                            graph_seeds=config.evaluation.validation_graph_seeds,
                            policy_seeds=config.evaluation.validation_policy_seeds,
                            backend=panel_backend,
                            protocol_bundle_hash=protocol_bundle_hash,
                        )
                    finally:
                        panel_backend.close()
                    retained_hashes = tuple(
                        value.program_hash
                        for value in sorted(
                            (
                                item
                                for item in fitness_cache.values()
                                if item.manifest_hash == validation_panel.manifest_hash
                                and item.protocol_bundle_hash == protocol_bundle_hash
                            ),
                            key=lambda item: (
                                -item.interval.lower,
                                item.program_hash,
                            ),
                        )[:4]
                    )
                    if not retained_hashes:
                        retained_hashes = tuple(sorted(programs)[:4])
                    slots = tuple(f"slot-{index:02d}" for index in range(8))
                    archive_snapshot_hash = domain_hash(
                        b"mforge-native-v3-archive\0",
                        canonical_json_bytes(list(sorted(programs))),
                    )
                    epoch_id = f"epoch-{epoch_number:04d}"
                    existing_epoch = next(
                        (
                            record
                            for record in persistence.semantic_records("epoch")
                            if record.semantic_key == epoch_id
                        ),
                        None,
                    )
                    if existing_epoch is not None:
                        payload = existing_epoch.payload
                        retained_hashes = tuple(
                            str(value)
                            for value in cast(
                                Sequence[JsonValue],
                                payload["parent_program_hashes"],
                            )
                        )
                        archive_snapshot_hash = str(payload["archive_snapshot_hash"])
                        planned_slot_ids = payload.get("planned_slot_ids")
                        if not isinstance(planned_slot_ids, list) or not all(
                            isinstance(value, str) for value in planned_slot_ids
                        ):
                            raise RuntimeError("resumed epoch has invalid planned slot identities")
                        if (
                            payload.get("development_manifest_hash") != development.manifest_hash
                            or payload.get("validation_manifest_hash")
                            != validation_panel.manifest_hash
                            or payload.get("protocol_bundle_hash") != protocol_bundle_hash
                            or tuple(planned_slot_ids) != slots
                        ):
                            raise RuntimeError(
                                "resumed epoch does not match its frozen semantic manifest"
                            )
                    snapshot = EpochSnapshot(
                        epoch_id,
                        epoch_number,
                        retained_hashes,
                        archive_snapshot_hash,
                        development.manifest_hash,
                        protocol_bundle_hash,
                        slots,
                    )
                    persistence.commit_semantic(
                        SemanticRecord(
                            "epoch",
                            epoch_id,
                            {
                                "state": "PLANNED",
                                "epoch_number": epoch_number,
                                "parent_program_hashes": list(retained_hashes),
                                "archive_snapshot_hash": archive_snapshot_hash,
                                "development_manifest_hash": development.manifest_hash,
                                "validation_manifest_hash": validation_panel.manifest_hash,
                                "protocol_bundle_hash": protocol_bundle_hash,
                                "planned_slot_ids": list(slots),
                            },
                        )
                    )
                    # A PLANNED record is immutable. Its terminal state uses a
                    # distinct semantic key so idempotent replay never mutates it.
                    slot_specs = tuple(
                        ProviderSlotSpec(
                            slot,
                            (
                                tuple(
                                    retained_hashes[(index + offset) % len(retained_hashes)]
                                    for offset in range(min(2, len(retained_hashes)))
                                )
                                if retained_hashes
                                else ()
                            ),
                            _BRIEFS[index],
                        )
                        for index, slot in enumerate(slots)
                    )
                    slot_by_id = {slot.slot_id: slot for slot in slot_specs}
                    persistence.commit_semantic(
                        SemanticRecord(
                            "epoch_slots",
                            epoch_id,
                            {
                                "slots": [
                                    {
                                        "slot_id": slot.slot_id,
                                        "parent_program_hashes": list(slot.parent_program_hashes),
                                        "brief": slot.brief,
                                    }
                                    for slot in slot_specs
                                ]
                            },
                        )
                    )
                    epoch_archive_record = next(
                        (
                            record
                            for record in persistence.semantic_records("epoch_archive")
                            if record.semantic_key == epoch_id
                        ),
                        None,
                    )
                    if epoch_archive_record is not None:
                        stored_summary = epoch_archive_record.payload.get("archive_summary")
                        if not isinstance(stored_summary, Mapping) or any(
                            not isinstance(value, Mapping) for value in stored_summary.values()
                        ):
                            raise RuntimeError("persisted epoch archive summary is invalid")
                        archive_summary = dict(stored_summary)
                    else:
                        archive_fitness: dict[str, ProgramFitness] = {}
                        for item in sorted(
                            fitness_cache.values(),
                            key=lambda value: (
                                value.program_hash,
                                0
                                if value.manifest_hash == validation_panel.manifest_hash
                                else 1
                                if value.manifest_hash == development.manifest_hash
                                else 2,
                                value.manifest_hash,
                                value.protocol_bundle_hash,
                            ),
                        ):
                            archive_fitness.setdefault(item.program_hash, item)
                        archive_summary = {
                            item.program_hash: {
                                "fitness_manifest_hash": item.manifest_hash,
                                "fitness_lower": _fraction(item.interval.lower),
                                "fitness_upper": _fraction(item.interval.upper),
                                "behavior_signature": item.behavior_signature,
                            }
                            for item in archive_fitness.values()
                            if item.program_hash in programs
                        }
                        persistence.commit_semantic(
                            SemanticRecord(
                                "epoch_archive",
                                epoch_id,
                                {
                                    "archive_snapshot_hash": archive_snapshot_hash,
                                    "archive_summary": archive_summary,
                                },
                            )
                        )
                    frozen_archive_summary = cast(
                        Mapping[str, Mapping[str, JsonValue]],
                        archive_summary,
                    )

                    def request_factory(
                        call: Any,
                        *,
                        frozen_slots: Mapping[str, ProviderSlotSpec] = slot_by_id,
                        frozen_archive: Mapping[
                            str, Mapping[str, JsonValue]
                        ] = frozen_archive_summary,
                    ) -> Any:
                        return build_provider_request(
                            call=call,
                            slots=tuple(frozen_slots[slot] for slot in call.slot_ids),
                            parent_programs=programs,
                            archive_summary=frozen_archive,
                            system_prompt=system_prompt,
                            output_schema=output_schema,
                            contract_bundle=provider_contract,
                            request_prompt=request_prompt,
                            repair_prompt=repair_prompt,
                            artifact_dir=str(
                                layout.artifacts
                                / "provider-v3"
                                / call.snapshot.epoch_id
                                / call.call_id.replace(":", "_")
                                / "transport"
                            ),
                            artifact_prefix="initial",
                        )

                    def artifact_sink(
                        artifact: ProviderArtifact,
                        *,
                        frozen_epoch_id: str = epoch_id,
                    ) -> None:
                        call_dir = (
                            layout.artifacts
                            / "provider-v3"
                            / frozen_epoch_id
                            / artifact.call_id.replace(":", "_")
                            / ("repair" if artifact.repair else "initial")
                        )
                        _atomic_write(
                            call_dir / "response.raw.json",
                            artifact.raw_response.encode("utf-8"),
                        )
                        phase = "repair" if artifact.repair else "initial"
                        # The batch is the atomic recovery boundary. Embed each
                        # canonical validated AST so a crash before archive
                        # materialization cannot force a provider replay.
                        persistence.commit_semantic(
                            SemanticRecord(
                                "provider_batch",
                                f"{artifact.call_id}:{phase}",
                                {
                                    "epoch_id": frozen_epoch_id,
                                    "call_id": artifact.call_id,
                                    "phase": phase,
                                    "entries": [
                                        {
                                            "slot_id": entry.slot_id,
                                            "program_hash": entry.program_hash,
                                            "program_json_canonical": (
                                                entry.program.canonical_json
                                                if entry.program is not None
                                                else None
                                            ),
                                            "error": entry.error,
                                        }
                                        for entry in artifact.entries
                                    ],
                                },
                            )
                        )
                        for entry in artifact.entries:
                            if entry.program is not None:
                                assert entry.program_hash is not None
                                with artifact_lock:
                                    generated_programs[entry.program_hash] = entry.program
                                _atomic_write(
                                    call_dir / f"{entry.slot_id}.program_json_raw.json",
                                    entry.program.raw.encode("utf-8"),
                                )
                                persistence.commit_semantic(
                                    SemanticRecord(
                                        "program",
                                        entry.program_hash,
                                        {
                                            "program_ast": cast(
                                                dict[str, JsonValue],
                                                entry.program.ast,
                                            ),
                                            "program_json_canonical": (
                                                entry.program.canonical_json
                                            ),
                                        },
                                    )
                                )
                        with artifact_lock:
                            provider_artifacts.append((artifact, call_dir))

                    def raw_artifact_sink(
                        artifact: ProviderRawArtifact,
                        *,
                        frozen_epoch_id: str = epoch_id,
                    ) -> None:
                        phase = "repair" if artifact.repair else "initial"
                        call_dir = (
                            layout.artifacts
                            / "provider-v3"
                            / frozen_epoch_id
                            / artifact.call_id.replace(":", "_")
                            / phase
                        )
                        raw_path = call_dir / "response.raw.json"
                        raw_bytes = artifact.raw_response.encode("utf-8")
                        _atomic_write(raw_path, raw_bytes)
                        persistence.commit_semantic(
                            SemanticRecord(
                                "provider_response",
                                f"{artifact.call_id}:{phase}",
                                {
                                    "epoch_id": frozen_epoch_id,
                                    "call_id": artifact.call_id,
                                    "phase": phase,
                                    "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                                    "raw_response_path": str(
                                        raw_path.relative_to(layout.artifacts)
                                    ),
                                },
                            )
                        )

                    provider_adapter = NativeV3Provider(
                        provider,
                        request_factory=request_factory,
                        artifact_sink=artifact_sink,
                        raw_artifact_sink=raw_artifact_sink,
                        allow_one_full_batch_repair=config.model.max_repairs == 1,
                    )
                    baseline_map = {
                        baseline.program.program_hash: baseline.program
                        for baseline in baseline_programs
                    }
                    current_auxiliary_hashes = (
                        *retained_hashes,
                        *(baseline_map),
                    )
                    missing_auxiliary = missing_current_manifest_evaluations(
                        program_hashes=current_auxiliary_hashes,
                        manifest_hash=development.manifest_hash,
                        protocol_bundle_hash=protocol_bundle_hash,
                        cache=fitness_cache,
                    )
                    auxiliary_programs = {
                        program_hash: (programs.get(program_hash) or baseline_map[program_hash])
                        for program_hash in missing_auxiliary
                    }
                    auxiliary_hash_set = set(auxiliary_programs)
                    persisted_batches: dict[str, dict[str, SemanticRecord]] = defaultdict(dict)
                    for record in persistence.semantic_records("provider_batch"):
                        if record.payload.get("epoch_id") != epoch_id:
                            continue
                        call_id = str(record.payload.get("call_id", ""))
                        phase = str(record.payload.get("phase", ""))
                        if phase not in {"initial", "repair"}:
                            raise RuntimeError("persisted provider batch has invalid phase")
                        persisted_batches[call_id][phase] = record
                    provider_calls_by_id = {
                        f"{epoch_id}:provider:{ordinal:04d}": ProviderCall(
                            f"{epoch_id}:provider:{ordinal:04d}",
                            tuple(slots[offset : offset + config.native_v3.provider_batch_size]),
                            snapshot,
                        )
                        for ordinal, offset in enumerate(
                            range(0, len(slots), config.native_v3.provider_batch_size)
                        )
                    }
                    for response_record in persistence.semantic_records("provider_response"):
                        if response_record.payload.get("epoch_id") != epoch_id:
                            continue
                        call_id = str(response_record.payload.get("call_id", ""))
                        phase = str(response_record.payload.get("phase", ""))
                        if phase not in {"initial", "repair"}:
                            raise RuntimeError("persisted provider response has invalid phase")
                        if phase in persisted_batches.get(call_id, {}):
                            continue
                        call = provider_calls_by_id.get(call_id)
                        if call is None:
                            raise RuntimeError("persisted provider response has an unknown call ID")
                        relative_path = response_record.payload.get("raw_response_path")
                        expected_sha256 = response_record.payload.get("raw_response_sha256")
                        if not isinstance(relative_path, str) or not isinstance(
                            expected_sha256, str
                        ):
                            raise RuntimeError("persisted provider response provenance is invalid")
                        raw_path = layout.artifacts / relative_path
                        raw_bytes = raw_path.read_bytes()
                        if hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
                            raise RuntimeError(
                                "persisted provider response content hash is inconsistent"
                            )
                        provider_adapter.parse_persisted_response(
                            call,
                            raw_response=raw_bytes.decode("utf-8"),
                            repair=phase == "repair",
                        )
                    persisted_batches = defaultdict(dict)
                    for record in persistence.semantic_records("provider_batch"):
                        if record.payload.get("epoch_id") != epoch_id:
                            continue
                        call_id = str(record.payload.get("call_id", ""))
                        phase = str(record.payload.get("phase", ""))
                        persisted_batches[call_id][phase] = record

                    def restore_batch(
                        record: SemanticRecord,
                    ) -> tuple[GeneratedEntry[ValidatedProgram], ...]:
                        restored: list[GeneratedEntry[ValidatedProgram]] = []
                        entries_value = record.payload.get("entries")
                        if not isinstance(entries_value, list):
                            raise RuntimeError("persisted provider batch has invalid entries")
                        for value in entries_value:
                            if not isinstance(value, Mapping):
                                raise RuntimeError("persisted provider batch entry is invalid")
                            slot_id = str(value.get("slot_id", ""))
                            program_hash_value = value.get("program_hash")
                            if not isinstance(program_hash_value, str):
                                restored.append(
                                    GeneratedEntry(
                                        slot_id,
                                        None,
                                        None,
                                        str(value.get("error") or "invalid program"),
                                    )
                                )
                                continue
                            recovered_program = programs.get(program_hash_value)
                            if recovered_program is None:
                                canonical = value.get("program_json_canonical")
                                if not isinstance(canonical, str):
                                    raise RuntimeError(
                                        "persisted provider batch references a missing program"
                                    )
                                validation = validate_program(canonical)
                                recovered_program = validation.program
                                if (
                                    recovered_program is None
                                    or recovered_program.program_hash != program_hash_value
                                ):
                                    raise RuntimeError(
                                        "persisted provider AST fails canonical validation"
                                    )
                                programs[program_hash_value] = recovered_program
                                persistence.commit_semantic(
                                    SemanticRecord(
                                        "program",
                                        program_hash_value,
                                        {
                                            "program_ast": cast(
                                                dict[str, JsonValue],
                                                recovered_program.ast,
                                            ),
                                            "program_json_canonical": canonical,
                                        },
                                    )
                                )
                            restored.append(
                                GeneratedEntry(
                                    slot_id,
                                    recovered_program,
                                    program_hash_value,
                                )
                            )
                        return tuple(restored)

                    recovered_entries: list[GeneratedEntry[ValidatedProgram]] = []
                    pending_repairs: dict[str, tuple[GeneratedEntry[ValidatedProgram], ...]] = {}
                    for call_id, phases in sorted(persisted_batches.items()):
                        if "repair" in phases:
                            recovered_entries.extend(restore_batch(phases["repair"]))
                            continue
                        initial = restore_batch(phases["initial"])
                        if config.model.max_repairs == 1 and all(
                            entry.program is None for entry in initial
                        ):
                            pending_repairs[call_id] = initial
                        else:
                            recovered_entries.extend(initial)

                    def provider_call(
                        call: ProviderCall,
                        *,
                        frozen_pending_repairs: Mapping[
                            str, tuple[GeneratedEntry[ValidatedProgram], ...]
                        ] = pending_repairs,
                        frozen_provider_adapter: NativeV3Provider = provider_adapter,
                    ) -> Any:
                        persisted_initial = frozen_pending_repairs.get(call.call_id)
                        if persisted_initial is not None:
                            return frozen_provider_adapter.repair_persisted_batch(
                                call,
                                persisted_initial,
                            )
                        return frozen_provider_adapter(call)

                    def streaming_provider_call(
                        call: ProviderCall,
                        entry_sink: Callable[
                            [GeneratedEntry[ValidatedProgram]],
                            None,
                        ],
                        *,
                        frozen_pending_repairs: Mapping[
                            str, tuple[GeneratedEntry[ValidatedProgram], ...]
                        ] = pending_repairs,
                        frozen_provider_adapter: NativeV3Provider = provider_adapter,
                    ) -> Any:
                        persisted_initial = frozen_pending_repairs.get(call.call_id)
                        if persisted_initial is not None:
                            return frozen_provider_adapter.repair_persisted_batch(
                                call,
                                persisted_initial,
                                entry_sink=entry_sink,
                            )
                        return frozen_provider_adapter.call_streaming(call, entry_sink)

                    development_tasks: dict[str, tuple[EpisodeTask, ...]] = {}

                    def build_shards(
                        program_hash: str,
                        _program: ValidatedProgram,
                        *,
                        task_cache: dict[str, tuple[EpisodeTask, ...]] = development_tasks,
                        frozen_panel: _Panel = development,
                        frozen_auxiliary: set[str] = auxiliary_hash_set,
                    ) -> Sequence[EpisodeShard]:
                        tasks = task_cache.setdefault(
                            program_hash,
                            self._tasks_for(
                                program_hash=program_hash,
                                panel=frozen_panel,
                                horizon=config.evaluation.horizon,
                                protocol_bundle_hash=protocol_bundle_hash,
                            ),
                        )
                        size = (
                            config.native_v3.auxiliary_shard_size
                            if program_hash in frozen_auxiliary
                            else config.native_v3.candidate_shard_size
                        )
                        pending_tasks = tuple(
                            task
                            for task in tasks
                            if task.episode_id not in committed_episodes["development"]
                        )
                        return build_episode_shards(
                            program_hash=program_hash,
                            task_factory=lambda episode: next(
                                task for task in pending_tasks if task.episode == episode
                            ),
                            episodes=(task.episode for task in pending_tasks),
                            shard_size=size,
                        )

                    evaluator = make_heg_shard_evaluator(
                        heg_repo=heg_repo,
                        graph_mode=config.evaluation.graph_mode,
                        witness_cap=config.native_v3.witness_cap,
                    )

                    def result_sink(
                        _shard: EpisodeShard,
                        results: tuple[EpisodeResult, ...],
                    ) -> None:
                        for result in results:
                            submit_result(result, "development")

                    def telemetry_sink(event: TelemetryEvent) -> None:
                        emit(
                            event.name,
                            **{
                                str(key): cast(JsonValue, value)
                                for key, value in event.fields.items()
                                if value is None or isinstance(value, bool | int | float | str)
                            },
                        )
                        persistence.record_telemetry(
                            TelemetryRecord(
                                event.name,
                                time.time_ns(),
                                {
                                    str(key): cast(JsonValue, value)
                                    for key, value in event.fields.items()
                                    if value is None or isinstance(value, bool | int | float | str)
                                },
                            )
                        )

                    with ProcessPoolExecutor(
                        max_workers=config.resources.thread_count,
                        mp_context=multiprocessing.get_context("spawn"),
                    ) as evaluation_executor:
                        scheduler = StreamingEpochScheduler[
                            ValidatedProgram,
                            EpisodeResult,
                        ](
                            config=SchedulerConfig(
                                provider_concurrency=config.model.concurrency,
                                evaluator_workers=config.resources.thread_count,
                                provider_batch_size=config.native_v3.provider_batch_size,
                                candidate_queue_capacity=(
                                    config.native_v3.candidate_queue_capacity
                                ),
                                evaluation_queue_capacity=(
                                    config.native_v3.evaluation_queue_capacity
                                ),
                                target_evaluation_backlog=(
                                    config.native_v3.target_evaluation_backlog
                                ),
                                candidate_shard_size=(config.native_v3.candidate_shard_size),
                                auxiliary_shard_size=(config.native_v3.auxiliary_shard_size),
                                provider_call_timeout_seconds=(
                                    config.turn_timeout_seconds
                                ),
                            ),
                            provider_call=provider_call,
                            streaming_provider_call=streaming_provider_call,
                            build_shards=build_shards,
                            evaluate_shard=evaluator,
                            evaluator_executor=evaluation_executor,
                            telemetry_sink=telemetry_sink,
                            result_sink=result_sink,
                        )
                        epoch_result = scheduler.run(
                            snapshot,
                            auxiliary_programs=auxiliary_programs,
                            recovered_entries=tuple(recovered_entries),
                        )
                    provider_failures = tuple(
                        event
                        for event in epoch_result.telemetry
                        if event.name == "provider_call_failed"
                    )
                    if provider_failures and not epoch_result.program_aliases:
                        first_failure = provider_failures[0]
                        error_type = str(first_failure.fields.get("error_type", "ProviderError"))
                        error_message = str(
                            first_failure.fields.get(
                                "error_message",
                                "provider call failed without diagnostics",
                            )
                        )
                        raise RuntimeError(
                            "all provider calls failed before producing a valid program: "
                            f"{error_type}: {error_message}"
                        )
                    with artifact_lock:
                        new_artifacts = [
                            item
                            for item in provider_artifacts
                            if item[0].call_id.startswith(f"{epoch_id}:")
                        ]
                    model_turns_used = max(
                        model_turns_used,
                        len(persistence.semantic_records("provider_batch")),
                    )
                    for artifact, call_dir in new_artifacts:
                        state.record_provider_turn(
                            idempotency_key=(
                                f"{artifact.call_id}:{'repair' if artifact.repair else 'initial'}"
                            ),
                            generation=epoch_number,
                            slot=artifact.entries[0].slot_id if artifact.entries else "slot-00",
                            phase="repair" if artifact.repair else "initial",
                            state="completed",
                            artifact_path=str(call_dir),
                            usage=artifact.usage,
                        )
                    for program_hash, aliases in epoch_result.program_aliases.items():
                        with artifact_lock:
                            program = generated_programs.get(program_hash)
                        if program is None:
                            program = programs[program_hash]
                        programs[program_hash] = program
                        parent_sets = {
                            alias: list(slot_by_id[alias].parent_program_hashes)
                            for alias in aliases
                        }
                        persistence.commit_semantic(
                            SemanticRecord(
                                "program",
                                program_hash,
                                {
                                    "program_ast": cast(
                                        dict[str, JsonValue],
                                        program.ast,
                                    ),
                                    "program_json_canonical": program.canonical_json,
                                },
                            )
                        )
                        for alias in aliases:
                            persistence.commit_semantic(
                                SemanticRecord(
                                    "lineage",
                                    f"{epoch_id}:{alias}",
                                    {
                                        "epoch_id": epoch_id,
                                        "slot_id": alias,
                                        "program_hash": program_hash,
                                        "parent_program_hashes": cast(
                                            list[JsonValue],
                                            parent_sets[alias],
                                        ),
                                    },
                                )
                            )
                        archive_path = layout.archive / "programs" / f"{program_hash}.json"
                        _atomic_write(
                            archive_path,
                            program.canonical_json.encode("utf-8") + b"\n",
                        )
                        state.record_candidate(
                            program_hash,
                            source_sha256=program_hash,
                            archive_path=str(archive_path),
                            generation=epoch_number,
                            slot=min(aliases),
                            parent_id=",".join(parent_sets[min(aliases)]),
                            status="evaluated",
                        )
                    combined_development_results: dict[str, tuple[EpisodeResult, ...]] = {}
                    evaluated_program_hashes = (
                        set(epoch_result.program_results)
                        | set(epoch_result.program_aliases)
                        | set(auxiliary_programs)
                    )
                    for program_hash in sorted(evaluated_program_hashes):
                        tasks = development_tasks.setdefault(
                            program_hash,
                            self._tasks_for(
                                program_hash=program_hash,
                                panel=development,
                                horizon=config.evaluation.horizon,
                                protocol_bundle_hash=protocol_bundle_hash,
                            ),
                        )
                        new_by_id = {
                            result.episode_id: result
                            for result in epoch_result.program_results.get(program_hash, ())
                        }
                        combined_development_results[program_hash] = tuple(
                            new_by_id.get(task.episode_id)
                            or committed_episodes["development"][task.episode_id]
                            for task in tasks
                            if task.episode_id in new_by_id
                            or task.episode_id in committed_episodes["development"]
                        )
                    epoch_fitness: list[ProgramFitness] = []
                    for program_hash, results in combined_development_results.items():
                        candidate_fitness = _program_fitness(
                            program_hash=program_hash,
                            manifest_hash=development.manifest_hash,
                            protocol_bundle_hash=protocol_bundle_hash,
                            results=cast(Sequence[EpisodeResult], results),
                            expected_episodes=len(development.episodes),
                        )
                        if candidate_fitness is None:
                            continue
                        fitness_cache[candidate_fitness.cache_key] = candidate_fitness
                        persistence.commit_semantic(
                            SemanticRecord(
                                "fitness",
                                ":".join(candidate_fitness.cache_key),
                                _fitness_payload(candidate_fitness),
                            )
                        )
                        if program_hash in epoch_result.program_aliases:
                            epoch_fitness.append(candidate_fitness)
                    unique_valid = len(epoch_fitness)
                    last_epoch_status = (
                        EpochStatus.COMPLETE
                        if unique_valid >= 8
                        else EpochStatus.DEGRADED
                        if unique_valid >= 4
                        else EpochStatus.INCONCLUSIVE
                    )
                    development_scientific_results = [
                        result
                        for results in epoch_result.program_results.values()
                        for result in cast(Sequence[EpisodeResult], results)
                    ]

                    def publish_metrics(
                        telemetry_events: tuple[TelemetryEvent, ...],
                        scientific_results: Sequence[EpisodeResult],
                        *,
                        validation_wall_ns: int = 0,
                        frozen_epoch_number: int = epoch_number,
                        frozen_persistence_wall_start_ns: int = (epoch_persistence_wall_start_ns),
                    ) -> None:
                        telemetry = summarize_scheduler_telemetry(
                            telemetry_events,
                            provider_concurrency=config.model.concurrency,
                            evaluator_workers=config.resources.thread_count,
                            validation_wall_ns=validation_wall_ns,
                            persistence_wall_ns=(
                                persistence.total_wall_time_ns - frozen_persistence_wall_start_ns
                            ),
                        )
                        raw_score_calls = sum(
                            result.raw_graph_score_calls for result in scientific_results
                        )
                        unique_scores = sum(
                            result.unique_graph_scores for result in scientific_results
                        )
                        accepted_rewrites = sum(
                            result.accepted_rewrites for result in scientific_results
                        )
                        cache_hits = sum(result.score_cache_hits for result in scientific_results)
                        scorer_restarts = sum(
                            result.scorer_restarts for result in scientific_results
                        )
                        epoch_wall_ns = max(1, telemetry.epoch_wall_ns)
                        fields: dict[str, JsonValue] = {
                            **telemetry.as_dict(),
                            "raw_graph_score_calls": raw_score_calls,
                            "unique_graph_scores": unique_scores,
                            "episodes": len(scientific_results),
                            "accepted_rewrites": accepted_rewrites,
                            "score_cache_hits": cache_hits,
                            "score_cache_hit_rate": _fraction(
                                Fraction(cache_hits, max(1, raw_score_calls))
                            ),
                            "raw_graph_score_calls_per_second": _fraction(
                                Fraction(
                                    raw_score_calls * 1_000_000_000,
                                    epoch_wall_ns,
                                )
                            ),
                            "unique_graph_scores_per_second": _fraction(
                                Fraction(
                                    unique_scores * 1_000_000_000,
                                    epoch_wall_ns,
                                )
                            ),
                            "episodes_per_second": _fraction(
                                Fraction(
                                    len(scientific_results) * 1_000_000_000,
                                    epoch_wall_ns,
                                )
                            ),
                            "accepted_rewrites_per_second": _fraction(
                                Fraction(
                                    accepted_rewrites * 1_000_000_000,
                                    epoch_wall_ns,
                                )
                            ),
                            "active_cpp_scorers": config.resources.thread_count,
                            "scorer_restarts": scorer_restarts,
                            "forbidden_fallback_count": 0,
                        }
                        emit(
                            "native_v3_metrics",
                            generation=frozen_epoch_number,
                            **fields,
                        )
                        persistence.record_telemetry(
                            TelemetryRecord(
                                "native_v3_metrics",
                                time.time_ns(),
                                fields,
                            )
                        )

                    if last_epoch_status is EpochStatus.INCONCLUSIVE:
                        publish_metrics(
                            epoch_result.telemetry,
                            development_scientific_results,
                        )
                        persistence.commit_semantic(
                            SemanticRecord(
                                "epoch_terminal",
                                epoch_id,
                                {
                                    "state": "TERMINAL",
                                    "status": last_epoch_status.value,
                                    "unique_valid_programs": unique_valid,
                                },
                            )
                        )
                        emit(
                            "epoch_terminal",
                            generation=epoch_number,
                            status=last_epoch_status.value,
                            unique_valid_programs=unique_valid,
                        )
                        break
                    shortlist = freeze_promotion_shortlist(
                        epoch_id=epoch_id,
                        values=epoch_fitness,
                    )
                    persistence.commit_semantic(
                        SemanticRecord(
                            "promotion_shortlist",
                            epoch_id,
                            {
                                "protocol_id": shortlist.protocol_id,
                                "development_manifest_hash": (shortlist.development_manifest_hash),
                                "protocol_bundle_hash": shortlist.protocol_bundle_hash,
                                "program_hashes": list(shortlist.program_hashes),
                            },
                        )
                    )
                    emit(
                        "development_cohort_terminal",
                        generation=epoch_number,
                        status=last_epoch_status.value,
                        unique_valid_programs=unique_valid,
                        promotion_shortlist=list(shortlist.program_hashes),
                    )
                    emit(
                        "promotion_shortlist_frozen",
                        generation=epoch_number,
                        program_hashes=list(shortlist.program_hashes),
                    )
                    validation_started_ns = time.monotonic_ns()

                    def validation_result_sink(result: EpisodeResult) -> None:
                        submit_result(result, "validation")

                    validation_results = self._evaluate_validation(
                        program_hashes=shortlist.program_hashes,
                        programs=programs,
                        panel=validation_panel,
                        horizon=config.evaluation.horizon,
                        protocol_bundle_hash=protocol_bundle_hash,
                        evaluator=evaluator,
                        evaluator_workers=config.resources.thread_count,
                        result_sink=validation_result_sink,
                        committed_results=committed_episodes["validation"],
                    )
                    validation_wall_ns = time.monotonic_ns() - validation_started_ns
                    validation_fitness: list[ProgramFitness] = []
                    for program_hash in shortlist.program_hashes:
                        validated_fitness = _program_fitness(
                            program_hash=program_hash,
                            manifest_hash=validation_panel.manifest_hash,
                            protocol_bundle_hash=protocol_bundle_hash,
                            results=validation_results.get(program_hash, ()),
                            expected_episodes=len(validation_panel.episodes),
                        )
                        if validated_fitness is None:
                            continue
                        fitness_cache[validated_fitness.cache_key] = validated_fitness
                        validation_fitness.append(validated_fitness)
                        persistence.commit_semantic(
                            SemanticRecord(
                                "fitness",
                                ":".join(validated_fitness.cache_key),
                                _fitness_payload(validated_fitness),
                            )
                        )
                    if len(validation_fitness) != len(shortlist.program_hashes):
                        last_epoch_status = EpochStatus.INCONCLUSIVE
                    else:
                        global_best = validated_global_best(
                            validation_fitness,
                            validation_manifest_hash=validation_panel.manifest_hash,
                            protocol_bundle_hash=protocol_bundle_hash,
                        )
                    full_telemetry = (
                        *epoch_result.telemetry,
                        TelemetryEvent(
                            "epoch_observation_completed",
                            time.monotonic_ns(),
                            {},
                        ),
                    )
                    publish_metrics(
                        full_telemetry,
                        (
                            *development_scientific_results,
                            *(
                                result
                                for values in validation_results.values()
                                for result in values
                            ),
                        ),
                        validation_wall_ns=validation_wall_ns,
                    )
                    persistence.commit_semantic(
                        SemanticRecord(
                            "epoch_terminal",
                            epoch_id,
                            {
                                "state": "TERMINAL",
                                "status": last_epoch_status.value,
                                "unique_valid_programs": unique_valid,
                                "promotion_shortlist": list(shortlist.program_hashes),
                                "validated_global_best": (
                                    global_best.program_hash if global_best else None
                                ),
                            },
                        )
                    )
                    emit(
                        "epoch_terminal",
                        generation=epoch_number,
                        status=last_epoch_status.value,
                        unique_valid_programs=unique_valid,
                        validated_global_best=(global_best.program_hash if global_best else None),
                    )
                    for future in verification_futures:
                        if future.done():
                            observe_verification(future.result())
                    if verified_outcome is not None:
                        break
            finally:
                verification.close()
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
            for future in verification_futures:
                if not future.done():
                    continue
                observe_verification(future.result())
            checkpoint_hash, _checkpoint = persistence.semantic_checkpoint()
            if verified_outcome is not None:
                return {
                    "schema_version": NATIVE_V3_RUN_SCHEMA,
                    "state": "completed",
                    "stop_reason": "counterexample_verified",
                    "generation": epoch_number,
                    "counterexample": {
                        "candidate_id": verified_outcome.graph_hash,
                        "certificate_path": str(
                            verified_outcome.artifact_directory / "certificate.json"
                        ),
                    },
                    "semantic_checkpoint_hash": checkpoint_hash,
                }
            if last_epoch_status is EpochStatus.INCONCLUSIVE:
                return {
                    "schema_version": NATIVE_V3_RUN_SCHEMA,
                    "state": "exhausted",
                    "stop_reason": "epoch_inconclusive",
                    "generation": epoch_number,
                    "semantic_checkpoint_hash": checkpoint_hash,
                }
            stopped_by_wall = session.budget_exhausted()
            return {
                "schema_version": NATIVE_V3_RUN_SCHEMA,
                "state": "idle" if stopped_by_wall else "exhausted",
                "stop_reason": ("session_wall_seconds" if stopped_by_wall else "generation_limit"),
                "generation": epoch_number,
                "validated_global_best": (
                    global_best.program_hash if global_best is not None else None
                ),
                "semantic_checkpoint_hash": checkpoint_hash,
            }

    @staticmethod
    def _evaluate_validation(
        *,
        program_hashes: Sequence[str],
        programs: Mapping[str, ValidatedProgram],
        panel: _Panel,
        horizon: int,
        protocol_bundle_hash: str,
        evaluator: Any,
        evaluator_workers: int,
        result_sink: Any,
        committed_results: Mapping[str, EpisodeResult],
    ) -> dict[str, tuple[EpisodeResult, ...]]:
        pending: dict[Future[Sequence[EpisodeResult]], tuple[EpisodeShard, bool]] = {}
        collected: dict[str, dict[str, tuple[EpisodeResult, ...]]] = defaultdict(dict)
        expected_tasks: dict[str, tuple[EpisodeTask, ...]] = {}
        shard_sources: deque[Iterator[EpisodeShard]] = deque()
        retry_queue: deque[tuple[EpisodeShard, bool]] = deque()
        with ProcessPoolExecutor(
            max_workers=evaluator_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            for program_hash in program_hashes:
                tasks = NativeV3ExperimentAdapter._tasks_for(
                    program_hash=program_hash,
                    panel=panel,
                    horizon=horizon,
                    protocol_bundle_hash=protocol_bundle_hash,
                )
                expected_tasks[program_hash] = tasks
                pending_tasks = tuple(
                    task for task in tasks if task.episode_id not in committed_results
                )
                task_by_episode = {task.episode: task for task in tasks}
                shards = build_episode_shards(
                    program_hash=program_hash,
                    task_factory=task_by_episode.__getitem__,
                    episodes=(task.episode for task in pending_tasks),
                    shard_size=1,
                )
                if shards:
                    shard_sources.append(iter(shards))

            def next_shard() -> tuple[EpisodeShard, bool] | None:
                if retry_queue:
                    return retry_queue.popleft()
                while shard_sources:
                    source = shard_sources.popleft()
                    try:
                        shard = next(source)
                    except StopIteration:
                        continue
                    shard_sources.append(source)
                    return shard, False
                return None

            def fill_pending() -> None:
                # Do not place hidden work behind the process pool. If exact
                # verification applies backpressure, only already-running
                # evaluator tasks may finish; no queued scoring starts.
                capacity = max(1, evaluator_workers)
                while len(pending) < capacity:
                    item = next_shard()
                    if item is None:
                        return
                    shard, retried = item
                    pending[executor.submit(evaluator, shard, programs[shard.program_hash])] = (
                        shard,
                        retried,
                    )

            fill_pending()
            while pending or shard_sources or retry_queue:
                if not pending:
                    fill_pending()
                    continue
                future = next(as_completed(tuple(pending)))
                shard, retried = pending.pop(future)
                try:
                    results = tuple(future.result())
                except ShardInfrastructureFailure:
                    if retried:
                        fill_pending()
                        continue
                    for residual in split_residual_shard(shard):
                        retry_queue.append((residual, True))
                    fill_pending()
                    continue
                for result in results:
                    result_sink(result)
                collected[shard.program_hash][shard.shard_id] = results
                fill_pending()
        output: dict[str, tuple[EpisodeResult, ...]] = {}
        for program_hash, tasks in sorted(expected_tasks.items()):
            new_by_id = {
                result.episode_id: result
                for shard_id in sorted(collected.get(program_hash, {}))
                for result in collected[program_hash][shard_id]
            }
            output[program_hash] = tuple(
                new_by_id.get(task.episode_id) or committed_results[task.episode_id]
                for task in tasks
                if task.episode_id in new_by_id or task.episode_id in committed_results
            )
        return output
