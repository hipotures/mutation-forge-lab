"""Deterministic episode evaluation for Native v3 programs."""

from __future__ import annotations

import signal
import threading
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from mutation_forge.backends.base import GraphBackend, ScoringBackendError
from mutation_forge.backends.heg import HegBackend
from mutation_forge.models import Edge, GraphState, JsonValue

from .heg_scoring import HegScoreEvidenceAdapter, merge_score_evidence
from .interpreter import (
    GraphFeatureInput,
    ProgramContext,
    invoke_program,
)
from .randomness import derive_seed64, uniform_below
from .scheduler import EpisodeShard, EpisodeTask, RetryableShardFailure
from .scoring import (
    ACCEPTANCE_PROTOCOL_ID,
    AttemptKind,
    EnergyScale,
    RationalInterval,
    ScoreEvidence,
    ScoreTimeoutWithoutPartial,
    acceptance_seed64,
    episode_auc,
    metropolis_accepts,
)
from .verification import graph_content_hash

EPISODE_PROTOCOL_ID = "native_v3_episode_v1"
ACCEPTANCE_PROFILE_ID = "native_v3_stagnation_8_window_4_v1"
EXPLORATION_TEMPERATURES = (
    Fraction(1, 16),
    Fraction(1, 32),
    Fraction(1, 64),
    Fraction(1, 128),
)
TABU_CAPACITY = 32
SHARD_WALL_TIMEOUT_SECONDS = 900


def _seeded_vertex_permutation(order: int, seed64: int) -> tuple[int, ...]:
    permutation = list(range(order))
    ordinal = 0
    for index in range(order - 1, 0, -1):
        selected, ordinal = uniform_below(seed64, ordinal, index + 1)
        permutation[index], permutation[selected] = (
            permutation[selected],
            permutation[index],
        )
    return tuple(permutation)


class EpisodeStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PROGRAM_FAILURE = "PROGRAM_FAILURE"
    INFRASTRUCTURE_INCONCLUSIVE = "INFRASTRUCTURE_INCONCLUSIVE"
    INCONCLUSIVE_TIMEOUT = "INCONCLUSIVE_TIMEOUT"


class ShardInfrastructureFailure(RetryableShardFailure):
    """A shard must be retried once on a different evaluator process."""


class ScoreEvidenceScorer(Protocol):
    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence: ...


@dataclass(frozen=True, slots=True)
class ApparentZero:
    graph_hash: str
    graph: GraphState
    provenance: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    proposed: bool
    accepted: bool
    strict_improvement: bool
    exploration_window_index: int | None
    incumbent_graph_hash: str
    utility_interval: RationalInterval
    no_plan_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_id: str
    program_hash: str
    status: EpisodeStatus
    order: int
    graph_seed: int
    policy_seed: int
    trajectory: tuple[RationalInterval, ...]
    auc: RationalInterval | None
    steps: tuple[StepRecord, ...]
    apparent_zeros: tuple[ApparentZero, ...]
    failure: str | None = None
    raw_graph_score_calls: int = 0
    unique_graph_scores: int = 0
    score_cache_hits: int = 0
    accepted_rewrites: int = 0
    scorer_restarts: int = 0


def _is_apparent_zero(evidence: ScoreEvidence) -> bool:
    return bool(evidence.components) and all(
        component.observed_count == 0 for component in evidence.components
    )


def _zero(
    *,
    graph: GraphState,
    task: EpisodeTask,
    source: str,
    step_index: int,
) -> ApparentZero:
    return ApparentZero(
        graph_content_hash(graph),
        graph,
        {
            "source": source,
            "episode_id": task.episode_id,
            "program_hash": task.program_hash,
            "step_index": step_index,
            "order": task.episode.order,
            "graph_seed": task.episode.graph_seed,
            "policy_seed": task.episode.policy_seed,
        },
    )


def _utility(scale: EnergyScale, evidence: ScoreEvidence) -> RationalInterval:
    return scale.utility(scale.interval(evidence))


def _expanded_if_relevant(
    adapter: ScoreEvidenceScorer,
    graph: GraphState,
    evidence: ScoreEvidence,
) -> ScoreEvidence:
    relevant = tuple(
        component.forbidden_length
        for component in evidence.components
        if not component.interval.exact
    )
    if not relevant:
        return evidence
    try:
        expanded = adapter.score(
            graph,
            witness_cap=evidence.witness_cap,
            forbidden_lengths=relevant,
            attempt_kind=AttemptKind.EXPANDED,
        )
    except ScoreTimeoutWithoutPartial:
        return evidence
    return merge_score_evidence(evidence, expanded)


def evaluate_episode(
    *,
    task: EpisodeTask,
    program: object,
    backend: GraphBackend,
    scorer: ScoreEvidenceScorer,
    witness_cap: int,
) -> EpisodeResult:
    from .contracts import ValidatedProgram

    if not isinstance(program, ValidatedProgram):
        raise TypeError("episode program must be a ValidatedProgram")
    episode = task.episode
    graph = backend.generate_seed(order=episode.order, seed=episode.graph_seed)
    if task.initial_graph_hash and graph_content_hash(graph) != task.initial_graph_hash:
        raise RuntimeError("initial graph hash violates the frozen episode manifest")
    try:
        incumbent = scorer.score(graph, witness_cap=witness_cap)
    except ScoreTimeoutWithoutPartial:
        full = RationalInterval(Fraction(), Fraction(1))
        timeout_trajectory = tuple(full for _ in range(task.horizon + 1))
        return EpisodeResult(
            task.episode_id,
            task.program_hash,
            EpisodeStatus.INCONCLUSIVE_TIMEOUT,
            episode.order,
            episode.graph_seed,
            episode.policy_seed,
            timeout_trajectory,
            episode_auc(timeout_trajectory, horizon=task.horizon),
            (),
            (),
            "initial scoring timed out without safe partial evidence",
        )
    except ScoringBackendError as error:
        raise ShardInfrastructureFailure(str(error)) from error
    scale = EnergyScale.build(
        order=episode.order,
        forbidden_lengths=tuple(component.forbidden_length for component in incumbent.components),
        witness_cap=witness_cap,
    )
    trajectory = [_utility(scale, incumbent)]
    apparent_zeros: list[ApparentZero] = []
    if _is_apparent_zero(incumbent):
        apparent_zeros.append(_zero(graph=graph, task=task, source="initial_graph", step_index=0))
    steps: list[StepRecord] = []
    accepted_rewrites = 0
    accepted_non_improving = 0
    consecutive_non_improving = 0
    exploration_index: int | None = None
    tabu = deque([graph_content_hash(graph)], maxlen=TABU_CAPACITY)

    for step_index in range(task.horizon):
        if exploration_index is None and consecutive_non_improving >= 8:
            exploration_index = 0
        context = ProgramContext(
            protocol_id=EPISODE_PROTOCOL_ID,
            step_index=step_index,
            horizon=task.horizon,
            acceptance_profile_id=ACCEPTANCE_PROFILE_ID,
            stagnation_steps=consecutive_non_improving,
            exploration_window_index=exploration_index,
            accepted_rewrites=accepted_rewrites,
            accepted_non_improving_rewrites=accepted_non_improving,
            consecutive_non_improving_rewrites=consecutive_non_improving,
            target_forbidden_lengths=scale.forbidden_lengths,
            witness_cap=witness_cap,
            current_score_component_bounds=tuple(
                (
                    component.forbidden_length,
                    component.lower_bound,
                    component.upper_bound,
                    component.status.value,
                )
                for component in incumbent.components
            ),
        )

        def witness_load_provider(
            overlay_graph: GraphState,
            *,
            frozen_backend: GraphBackend = backend,
            frozen_sampling_seed: int = derive_seed64(
                "native-v3-witness-feature",
                task.program_hash,
                task.episode_id,
                episode.policy_seed,
                step_index,
            ),
        ) -> tuple[
            dict[tuple[int, int], int],
            dict[tuple[int, Edge], int],
        ]:
            if isinstance(frozen_backend, HegBackend):
                return frozen_backend.sampled_forbidden_witness_loads(
                    overlay_graph,
                    relabeling=_seeded_vertex_permutation(
                        overlay_graph.order,
                        frozen_sampling_seed,
                    ),
                )
            return {}, {}

        invocation = invoke_program(
            program,
            graph,
            context=context,
            features=GraphFeatureInput(
                total_witness_interval=(
                    incumbent.total_witness_interval.lower,
                    incumbent.total_witness_interval.upper,
                ),
                weighted_penalty_interval=(
                    incumbent.weighted_penalty_interval.lower,
                    incumbent.weighted_penalty_interval.upper,
                ),
                energy_interval=(
                    scale.interval(incumbent).lower,
                    scale.interval(incumbent).upper,
                ),
                witness_load_provider=witness_load_provider,
            ),
            policy_seed=episode.policy_seed,
            episode_id=task.episode_id,
        )
        if invocation.failure is not None:
            return EpisodeResult(
                task.episode_id,
                task.program_hash,
                EpisodeStatus.PROGRAM_FAILURE,
                episode.order,
                episode.graph_seed,
                episode.policy_seed,
                tuple(trajectory),
                None,
                tuple(steps),
                tuple(apparent_zeros),
                (
                    f"{invocation.failure.code}@{invocation.failure.path}: "
                    f"{invocation.failure.message}"
                ),
            )
        if invocation.rewrite is None:
            consecutive_non_improving += 1
            if exploration_index is not None:
                exploration_index += 1
                if exploration_index == len(EXPLORATION_TEMPERATURES):
                    exploration_index = None
                    consecutive_non_improving = 0
            utility = _utility(scale, incumbent)
            trajectory.append(utility)
            steps.append(
                StepRecord(
                    step_index,
                    False,
                    False,
                    False,
                    context.exploration_window_index,
                    graph_content_hash(graph),
                    utility,
                    invocation.no_plan.reason.value if invocation.no_plan else "NO_PLAN",
                )
            )
            continue
        proposal_graph = backend.apply_rewrite(graph, invocation.rewrite)
        validation = backend.validate(proposal_graph)
        if not validation.valid:
            return EpisodeResult(
                task.episode_id,
                task.program_hash,
                EpisodeStatus.PROGRAM_FAILURE,
                episode.order,
                episode.graph_seed,
                episode.policy_seed,
                tuple(trajectory),
                None,
                tuple(steps),
                tuple(apparent_zeros),
                f"interpreter emitted invalid graph: {validation.errors}",
            )
        try:
            proposal = scorer.score(proposal_graph, witness_cap=witness_cap)
        except ScoreTimeoutWithoutPartial:
            consecutive_non_improving += 1
            used_window = exploration_index
            if exploration_index is not None:
                exploration_index += 1
                if exploration_index == len(EXPLORATION_TEMPERATURES):
                    exploration_index = None
                    consecutive_non_improving = 0
            utility = _utility(scale, incumbent)
            trajectory.append(utility)
            steps.append(
                StepRecord(
                    step_index,
                    True,
                    False,
                    False,
                    used_window,
                    graph_content_hash(graph),
                    utility,
                    "PROPOSAL_SCORE_TIMEOUT",
                )
            )
            continue
        except ScoringBackendError as error:
            raise ShardInfrastructureFailure(str(error)) from error
        if _is_apparent_zero(proposal):
            apparent_zeros.append(
                _zero(
                    graph=proposal_graph,
                    task=task,
                    source="program_proposal",
                    step_index=step_index + 1,
                )
            )
        incumbent_energy = scale.interval(incumbent)
        proposal_energy = scale.interval(proposal)
        if (not incumbent_energy.exact or not proposal_energy.exact) and not (
            proposal_energy.upper < incumbent_energy.lower
            or proposal_energy.lower > incumbent_energy.upper
        ):
            try:
                incumbent = _expanded_if_relevant(scorer, graph, incumbent)
                proposal = _expanded_if_relevant(scorer, proposal_graph, proposal)
            except ScoringBackendError as error:
                raise ShardInfrastructureFailure(str(error)) from error
            incumbent_energy = scale.interval(incumbent)
            proposal_energy = scale.interval(proposal)
        strict_improvement = proposal_energy.upper < incumbent_energy.lower
        accepted = strict_improvement
        proposal_hash = graph_content_hash(proposal_graph)
        used_window = exploration_index
        if (
            not accepted
            and incumbent_energy.exact
            and proposal_energy.exact
            and exploration_index is not None
            and proposal_hash not in tabu
        ):
            raw_delta = proposal_energy.lower - incumbent_energy.lower
            if raw_delta >= 0:
                normalized_delta = Fraction(
                    raw_delta,
                    scale.energy_max - scale.energy_min,
                )
                seed64 = acceptance_seed64(
                    protocol_id=ACCEPTANCE_PROTOCOL_ID,
                    program_hash=task.program_hash,
                    episode_id=task.episode_id,
                    policy_seed=episode.policy_seed,
                    step_index=step_index,
                    ast_path="/acceptance",
                    repeat_indices=(),
                    invocation_ordinal=step_index,
                    draw_ordinal=0,
                )
                accepted, _threshold, _draw = metropolis_accepts(
                    delta=normalized_delta,
                    temperature=EXPLORATION_TEMPERATURES[exploration_index],
                    seed64=seed64,
                )
        if accepted:
            graph = proposal_graph
            incumbent = proposal
            accepted_rewrites += 1
            tabu.append(proposal_hash)
            if strict_improvement:
                consecutive_non_improving = 0
                exploration_index = None
            else:
                accepted_non_improving += 1
                consecutive_non_improving += 1
        else:
            consecutive_non_improving += 1
        if exploration_index is not None:
            exploration_index += 1
            if exploration_index == len(EXPLORATION_TEMPERATURES):
                exploration_index = None
                consecutive_non_improving = 0
        utility = _utility(scale, incumbent)
        trajectory.append(utility)
        steps.append(
            StepRecord(
                step_index,
                True,
                accepted,
                strict_improvement,
                used_window,
                graph_content_hash(graph),
                utility,
            )
        )
    return EpisodeResult(
        task.episode_id,
        task.program_hash,
        EpisodeStatus.COMPLETE,
        episode.order,
        episode.graph_seed,
        episode.policy_seed,
        tuple(trajectory),
        episode_auc(trajectory, horizon=task.horizon),
        tuple(steps),
        tuple(apparent_zeros),
        raw_graph_score_calls=int(getattr(scorer, "raw_graph_score_calls", 0)),
        unique_graph_scores=int(getattr(scorer, "unique_graph_scores", 0)),
        score_cache_hits=(
            int(getattr(scorer, "raw_graph_score_calls", 0))
            - int(getattr(scorer, "unique_graph_scores", 0))
        ),
        accepted_rewrites=accepted_rewrites,
        scorer_restarts=int(getattr(backend, "score_worker_restarts", 0)),
    )


def evaluate_heg_shard(
    shard: EpisodeShard,
    program: object,
    *,
    heg_repo: Path,
    graph_mode: str,
    witness_cap: int,
) -> Sequence[EpisodeResult]:
    previous_handler: object | None = None

    def shard_timeout(_signal_number: int, _frame: object) -> None:
        raise ShardInfrastructureFailure(
            f"evaluation shard exceeded {SHARD_WALL_TIMEOUT_SECONDS} seconds"
        )

    watchdog_enabled = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    if watchdog_enabled:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, shard_timeout)
        signal.setitimer(signal.ITIMER_REAL, SHARD_WALL_TIMEOUT_SECONDS)
    backend: HegBackend | None = None
    try:
        backend = HegBackend(
            heg_repo,
            graph_mode=graph_mode,
            score_timeout_seconds=20.0,
        )
        scorer = HegScoreEvidenceAdapter(backend)
        return tuple(
            evaluate_episode(
                task=task,
                program=program,
                backend=backend,
                scorer=scorer,
                witness_cap=witness_cap,
            )
            for task in shard.tasks
        )
    finally:
        if backend is not None:
            backend.close()
        if watchdog_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            assert previous_handler is not None
            signal.signal(signal.SIGALRM, cast(Any, previous_handler))


def make_heg_shard_evaluator(
    *,
    heg_repo: Path,
    graph_mode: str,
    witness_cap: int,
) -> Callable[[EpisodeShard, object], Sequence[EpisodeResult]]:
    return partial(
        evaluate_heg_shard,
        heg_repo=heg_repo,
        graph_mode=graph_mode,
        witness_cap=witness_cap,
    )
