"""Deterministic serial evaluation of one Native v3 program."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING, Protocol

from mutation_forge.backends.base import GraphBackend, ScoreProfileRecorder
from mutation_forge.counterexamples import (
    CandidateProvenance,
    CounterexampleOutcome,
)
from mutation_forge.models import GraphScore, GraphState, JsonValue, RewritePlan

from .canonical import canonical_json_bytes, domain_hash
from .contracts import ProgramContract, ValidatedProgram
from .execution import ProgramFailure, SemanticEvent
from .heg_scoring import (
    ScoreEvidenceScorer,
    merge_score_evidence,
)
from .scoring import (
    AttemptKind,
    EnergyScale,
    IntegerInterval,
    RationalInterval,
    ScoreEvidence,
    ScoreTimeoutWithoutPartial,
    best_so_far_curve,
    candidate_fitness,
    episode_auc,
    proved_strict_energy_improvement,
)

if TYPE_CHECKING:
    from .interpreter import InterpreterLimits

SERIAL_EVALUATOR_PROTOCOL_ID = "native_v3_serial_interval_evaluator_v2"
_TRACE_HASH_DOMAIN = b"mforge-native-v3-serial-trace\0"


class CounterexampleInspector(Protocol):
    def inspect(
        self,
        *,
        graph: GraphState,
        score: GraphScore,
        provenance: CandidateProvenance,
        witness_cap: int,
    ) -> CounterexampleOutcome: ...


class SerialEvaluationStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PROGRAM_FAILURE = "PROGRAM_FAILURE"
    INCONCLUSIVE_UNSAFE_TIMEOUT = "INCONCLUSIVE_UNSAFE_TIMEOUT"


@dataclass(frozen=True, slots=True)
class SerialEpisodeConfig:
    order: int
    graph_seed: int
    policy_seed: int
    horizon: int
    witness_cap: int
    episode_id: str

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be positive")
        if self.graph_seed < 0 or self.policy_seed < 0:
            raise ValueError("seeds must be non-negative")
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")
        if self.witness_cap < 1:
            raise ValueError("witness_cap must be positive")
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")


@dataclass(frozen=True, slots=True)
class GraphIdentity:
    state_hash: str
    canonical_hash: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "state_hash": self.state_hash,
            "canonical_hash": self.canonical_hash,
        }


@dataclass(frozen=True, slots=True)
class CounterexampleTrace:
    graph_identity: GraphIdentity
    decision: str
    candidate_id: str | None
    primary_status: str | None
    independent_status: str | None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "graph_identity": self.graph_identity.as_dict(),
            "decision": self.decision,
            "candidate_id": self.candidate_id,
            "primary_status": self.primary_status,
            "independent_status": self.independent_status,
        }


@dataclass(frozen=True, slots=True)
class SerialStepTrace:
    step_index: int
    incumbent_before: GraphIdentity
    evidence_before: ScoreEvidence
    energy_before: IntegerInterval
    utility_before: RationalInterval
    interpreter_trace: tuple[SemanticEvent, ...]
    outcome: str
    no_plan_reason: str | None
    failure: ProgramFailure | None
    rewrite: RewritePlan | None
    candidate_identity: GraphIdentity | None
    candidate_evidence: ScoreEvidence | None
    candidate_energy: IntegerInterval | None
    accepted: bool
    acceptance_proved: bool
    expanded_retry_timeouts: tuple[str, ...]
    incumbent_after: GraphIdentity
    evidence_after: ScoreEvidence
    utility_after: RationalInterval
    counterexample: CounterexampleTrace | None

    def as_dict(
        self,
        *,
        include_telemetry: bool = True,
    ) -> dict[str, JsonValue]:
        return {
            "step_index": self.step_index,
            "incumbent_before": self.incumbent_before.as_dict(),
            "evidence_before": self.evidence_before.as_dict(
                include_telemetry=include_telemetry
            ),
            "energy_before": self.energy_before.as_dict(),
            "utility_before": self.utility_before.as_dict(),
            "interpreter_trace": [event.as_dict() for event in self.interpreter_trace],
            "outcome": self.outcome,
            "no_plan_reason": self.no_plan_reason,
            "failure": (
                {
                    "code": self.failure.code,
                    "path": self.failure.path,
                    "message": self.failure.message,
                }
                if self.failure is not None
                else None
            ),
            "rewrite": _rewrite_dict(self.rewrite),
            "candidate_identity": (
                self.candidate_identity.as_dict()
                if self.candidate_identity is not None
                else None
            ),
            "candidate_evidence": (
                self.candidate_evidence.as_dict(
                    include_telemetry=include_telemetry
                )
                if self.candidate_evidence is not None
                else None
            ),
            "candidate_energy": (
                self.candidate_energy.as_dict()
                if self.candidate_energy is not None
                else None
            ),
            "accepted": self.accepted,
            "acceptance_proved": self.acceptance_proved,
            "expanded_retry_timeouts": list(self.expanded_retry_timeouts),
            "incumbent_after": self.incumbent_after.as_dict(),
            "evidence_after": self.evidence_after.as_dict(
                include_telemetry=include_telemetry
            ),
            "utility_after": self.utility_after.as_dict(),
            "counterexample": (
                self.counterexample.as_dict() if self.counterexample is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SerialEpisodeResult:
    protocol_id: str
    program_hash: str
    status: SerialEvaluationStatus
    config: SerialEpisodeConfig
    initial_identity: GraphIdentity
    initial_evidence: ScoreEvidence | None
    initial_counterexample: CounterexampleTrace | None
    steps: tuple[SerialStepTrace, ...]
    terminal_identity: GraphIdentity
    terminal_evidence: ScoreEvidence | None
    utility_trajectory: tuple[RationalInterval, ...]
    best_so_far_utility: tuple[RationalInterval, ...]
    auc_interval: RationalInterval
    fitness_interval: RationalInterval
    accepted_rewrites: int
    score_attempts: int
    unique_graph_scores: int
    failure: ProgramFailure | None
    scientific_error: str | None
    semantic_trace_hash: str
    execution_trace_protocol_id: str | None = None

    def as_dict(
        self,
        *,
        include_hash: bool = True,
        include_telemetry: bool = True,
    ) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "protocol_id": self.protocol_id,
            "program_hash": self.program_hash,
            "status": self.status.value,
            "config": {
                "order": self.config.order,
                "graph_seed": self.config.graph_seed,
                "policy_seed": self.config.policy_seed,
                "horizon": self.config.horizon,
                "witness_cap": self.config.witness_cap,
                "episode_id": self.config.episode_id,
            },
            "initial_identity": self.initial_identity.as_dict(),
            "initial_evidence": (
                self.initial_evidence.as_dict(
                    include_telemetry=include_telemetry
                )
                if self.initial_evidence is not None
                else None
            ),
            "initial_counterexample": (
                self.initial_counterexample.as_dict()
                if self.initial_counterexample is not None
                else None
            ),
            "steps": [
                step.as_dict(include_telemetry=include_telemetry)
                for step in self.steps
            ],
            "terminal_identity": self.terminal_identity.as_dict(),
            "terminal_evidence": (
                self.terminal_evidence.as_dict(
                    include_telemetry=include_telemetry
                )
                if self.terminal_evidence is not None
                else None
            ),
            "utility_trajectory": [
                interval.as_dict() for interval in self.utility_trajectory
            ],
            "best_so_far_utility": [
                interval.as_dict() for interval in self.best_so_far_utility
            ],
            "auc_interval": self.auc_interval.as_dict(),
            "fitness_interval": self.fitness_interval.as_dict(),
            "accepted_rewrites": self.accepted_rewrites,
            "score_attempts": self.score_attempts,
            "unique_graph_scores": self.unique_graph_scores,
            "failure": (
                {
                    "code": self.failure.code,
                    "path": self.failure.path,
                    "message": self.failure.message,
                }
                if self.failure is not None
                else None
            ),
            "scientific_error": self.scientific_error,
        }
        if self.execution_trace_protocol_id is not None:
            result["execution_trace_protocol_id"] = self.execution_trace_protocol_id
        if include_hash:
            result["semantic_trace_hash"] = self.semantic_trace_hash
        return result


@dataclass(slots=True)
class _CapturingRewriteHost:
    backend: GraphBackend
    candidate: GraphState | None = None

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: ScoreProfileRecorder | None = None,
    ) -> GraphState:
        candidate = self.backend.apply_rewrite(
            graph,
            rewrite,
            record_score_profile=record_score_profile,
        )
        self.candidate = candidate
        return candidate


@dataclass(frozen=True, slots=True)
class _SerialInvocation:
    semantic_trace: tuple[SemanticEvent, ...]
    no_plan_reason: str | None = None
    failure: ProgramFailure | None = None
    rewrite: RewritePlan | None = None
    candidate: GraphState | None = None

    def __post_init__(self) -> None:
        outcomes = (
            self.no_plan_reason is not None,
            self.failure is not None,
            self.rewrite is not None or self.candidate is not None,
        )
        if sum(outcomes) != 1:
            raise ValueError("serial invocation must contain exactly one outcome")


type _StepInvoker = Callable[
    [GraphState, int, int],
    _SerialInvocation,
]


def _identity(backend: GraphBackend, graph: GraphState) -> GraphIdentity:
    return GraphIdentity(
        state_hash=backend.state_hash(graph),
        canonical_hash=backend.canonical_hash(graph),
    )


def _rewrite_dict(rewrite: RewritePlan | None) -> dict[str, JsonValue] | None:
    if rewrite is None:
        return None
    return {
        "removed_edges": [list(edge) for edge in rewrite.removed_edges],
        "added_edges": [list(edge) for edge in rewrite.added_edges],
        "operator_family": rewrite.operator_family,
        "metadata": dict(rewrite.metadata),
    }


def _inspect_apparent_zero(
    *,
    pipeline: CounterexampleInspector | None,
    backend: GraphBackend,
    graph: GraphState,
    evidence: ScoreEvidence,
    config: SerialEpisodeConfig,
    program_hash: str,
    step_index: int,
    provenance_source_kind: str,
) -> CounterexampleTrace | None:
    total = evidence.total_witness_interval
    if pipeline is None or not total.exact or total.upper != 0:
        return None
    score = GraphScore(
        valid=True,
        capped_cycle_counts=tuple(
            (component.forbidden_length, component.observed_count)
            for component in evidence.components
        ),
        total_capped_witnesses=0,
        weighted_penalty=0,
        complete=evidence.complete_under_cap,
        ordering_key=(0, 0, 0, 0, evidence.edge_count),
    )
    outcome: CounterexampleOutcome = pipeline.inspect(
        graph=graph,
        score=score,
        witness_cap=config.witness_cap,
        provenance=CandidateProvenance(
            source_kind=provenance_source_kind,
            source_id=program_hash,
            episode_id=config.episode_id,
            graph_seed=config.graph_seed,
            policy_seed=config.policy_seed,
            evaluation_step=step_index,
        ),
    )
    return CounterexampleTrace(
        graph_identity=_identity(backend, graph),
        decision=outcome.decision.value,
        candidate_id=(
            outcome.candidate.candidate_id if outcome.candidate is not None else None
        ),
        primary_status=(
            outcome.primary.status if outcome.primary is not None else None
        ),
        independent_status=(
            outcome.independent.status if outcome.independent is not None else None
        ),
    )


def _trace_hash(payload: dict[str, JsonValue]) -> str:
    return domain_hash(_TRACE_HASH_DOMAIN, canonical_json_bytes(payload))


def _utility(scale: EnergyScale, evidence: ScoreEvidence) -> RationalInterval:
    return scale.utility(scale.interval(evidence))


def _expand_nonpoint_components(
    *,
    scorer: ScoreEvidenceScorer,
    graph: GraphState,
    evidence: ScoreEvidence,
) -> tuple[ScoreEvidence, bool]:
    selected_lengths = tuple(
        component.forbidden_length
        for component in evidence.components
        if not component.interval.exact
    )
    if not selected_lengths:
        return evidence, False
    try:
        expanded = scorer.score_evidence(
            graph,
            witness_cap=evidence.witness_cap,
            forbidden_lengths=selected_lengths,
            attempt_kind=AttemptKind.EXPANDED,
        )
    except ScoreTimeoutWithoutPartial:
        return evidence, True
    return merge_score_evidence(evidence, expanded), False


def _full_uncertainty() -> RationalInterval:
    return RationalInterval(Fraction(), Fraction(1))


def _program_failure_fitness() -> RationalInterval:
    return RationalInterval(Fraction(), Fraction())


def _score_initial_evidence(
    *,
    scorer: ScoreEvidenceScorer,
    graph: GraphState,
    witness_cap: int,
    forbidden_lengths: tuple[int, ...] | None,
) -> ScoreEvidence:
    if forbidden_lengths is None:
        return scorer.score_evidence(graph, witness_cap=witness_cap)
    return scorer.score_evidence(
        graph,
        witness_cap=witness_cap,
        forbidden_lengths=forbidden_lengths,
    )


def _evaluate_serial(
    *,
    backend: GraphBackend,
    scorer: ScoreEvidenceScorer,
    program_hash: str,
    config: SerialEpisodeConfig,
    invoke_step: _StepInvoker,
    counterexample_pipeline: CounterexampleInspector | None = None,
    provenance_source_kind: str,
    protocol_id: str,
    execution_trace_protocol_id: str | None = None,
    forbidden_lengths: tuple[int, ...] | None = None,
) -> SerialEpisodeResult:
    """Run one representation-independent trajectory with proved improvements."""

    attempts_before = scorer.raw_graph_score_calls
    unique_before = scorer.unique_graph_scores
    initial = backend.generate_seed(order=config.order, seed=config.graph_seed)
    validation = backend.validate(initial)
    if not validation.valid:
        raise ValueError(f"backend generated an invalid seed: {validation.errors}")
    initial_identity = _identity(backend, initial)
    try:
        initial_evidence = _score_initial_evidence(
            scorer=scorer,
            graph=initial,
            witness_cap=config.witness_cap,
            forbidden_lengths=forbidden_lengths,
        )
    except ScoreTimeoutWithoutPartial:
        uncertainty = _full_uncertainty()
        timeout_trajectory = tuple(
            uncertainty for _ in range(config.horizon + 1)
        )
        provisional = SerialEpisodeResult(
            protocol_id=protocol_id,
            program_hash=program_hash,
            status=SerialEvaluationStatus.INCONCLUSIVE_UNSAFE_TIMEOUT,
            config=config,
            initial_identity=initial_identity,
            initial_evidence=None,
            initial_counterexample=None,
            steps=(),
            terminal_identity=initial_identity,
            terminal_evidence=None,
            utility_trajectory=timeout_trajectory,
            best_so_far_utility=best_so_far_curve(timeout_trajectory),
            auc_interval=episode_auc(
                timeout_trajectory,
                horizon=config.horizon,
            ),
            fitness_interval=uncertainty,
            accepted_rewrites=0,
            score_attempts=scorer.raw_graph_score_calls - attempts_before,
            unique_graph_scores=scorer.unique_graph_scores - unique_before,
            failure=None,
            scientific_error=(
                "initial scoring timed out without safe partial evidence"
            ),
            semantic_trace_hash="",
            execution_trace_protocol_id=execution_trace_protocol_id,
        )
        payload = provisional.as_dict(
            include_hash=False,
            include_telemetry=False,
        )
        return replace(
            provisional,
            semantic_trace_hash=_trace_hash(payload),
        )
    scale = EnergyScale.build(
        order=config.order,
        forbidden_lengths=tuple(
            component.forbidden_length
            for component in initial_evidence.components
        ),
        witness_cap=config.witness_cap,
    )
    initial_utility = _utility(scale, initial_evidence)
    initial_counterexample = _inspect_apparent_zero(
        pipeline=counterexample_pipeline,
        backend=backend,
        graph=initial,
        evidence=initial_evidence,
        config=config,
        program_hash=program_hash,
        step_index=0,
        provenance_source_kind=provenance_source_kind,
    )

    current = initial
    current_evidence = initial_evidence
    accepted_rewrites = 0
    failure: ProgramFailure | None = None
    steps: list[SerialStepTrace] = []
    trajectory = [initial_utility]
    for step_index in range(config.horizon):
        before_identity = _identity(backend, current)
        evidence_before = current_evidence
        invocation = invoke_step(current, step_index, accepted_rewrites)
        outcome = "failure"
        no_plan_reason: str | None = None
        rewrite: RewritePlan | None = None
        candidate_identity: GraphIdentity | None = None
        candidate_evidence: ScoreEvidence | None = None
        candidate_energy: IntegerInterval | None = None
        accepted = False
        acceptance_proved = False
        expanded_retry_timeouts: list[str] = []
        counterexample: CounterexampleTrace | None = None

        if invocation.failure is not None:
            failure = invocation.failure
        elif invocation.no_plan_reason is not None:
            outcome = "no_plan"
            no_plan_reason = invocation.no_plan_reason
        else:
            rewrite = invocation.rewrite
            candidate = invocation.candidate
            if rewrite is None or candidate is None:
                failure = ProgramFailure(
                    "INTERPRETER_FAULT",
                    "/entry",
                    "rewrite outcome did not retain its host-validated candidate",
                )
            else:
                outcome = "rewrite"
                candidate_identity = _identity(backend, candidate)
                try:
                    candidate_evidence = scorer.score_evidence(
                        candidate,
                        witness_cap=config.witness_cap,
                        forbidden_lengths=forbidden_lengths,
                    )
                except ScoreTimeoutWithoutPartial:
                    outcome = "score_timeout_without_partial"
                if candidate_evidence is not None:
                    incumbent_energy = scale.interval(current_evidence)
                    candidate_energy = scale.interval(candidate_evidence)
                    intervals_overlap = not (
                        candidate_energy.upper < incumbent_energy.lower
                        or incumbent_energy.upper < candidate_energy.lower
                    )
                    if intervals_overlap and (
                        not incumbent_energy.exact
                        or not candidate_energy.exact
                    ):
                        current_evidence, incumbent_timeout = (
                            _expand_nonpoint_components(
                                scorer=scorer,
                                graph=current,
                                evidence=current_evidence,
                            )
                        )
                        if incumbent_timeout:
                            expanded_retry_timeouts.append("incumbent")
                        evidence_before = current_evidence
                        candidate_evidence, candidate_timeout = (
                            _expand_nonpoint_components(
                                scorer=scorer,
                                graph=candidate,
                                evidence=candidate_evidence,
                            )
                        )
                        if candidate_timeout:
                            expanded_retry_timeouts.append("candidate")
                        incumbent_energy = scale.interval(current_evidence)
                        candidate_energy = scale.interval(candidate_evidence)
                    counterexample = _inspect_apparent_zero(
                        pipeline=counterexample_pipeline,
                        backend=backend,
                        graph=candidate,
                        evidence=candidate_evidence,
                        config=config,
                        program_hash=program_hash,
                        step_index=step_index + 1,
                        provenance_source_kind=provenance_source_kind,
                    )
                    acceptance_proved = proved_strict_energy_improvement(
                        candidate_energy,
                        incumbent_energy,
                    )
                    accepted = acceptance_proved
                    if accepted:
                        current = candidate
                        current_evidence = candidate_evidence
                        accepted_rewrites += 1

        energy_before = scale.interval(evidence_before)
        utility_before = _utility(scale, evidence_before)
        utility_after = _utility(scale, current_evidence)
        trajectory.append(utility_after)
        steps.append(
            SerialStepTrace(
                step_index=step_index,
                incumbent_before=before_identity,
                evidence_before=evidence_before,
                energy_before=energy_before,
                utility_before=utility_before,
                interpreter_trace=invocation.semantic_trace,
                outcome=outcome,
                no_plan_reason=no_plan_reason,
                failure=failure,
                rewrite=rewrite,
                candidate_identity=candidate_identity,
                candidate_evidence=candidate_evidence,
                candidate_energy=candidate_energy,
                accepted=accepted,
                acceptance_proved=acceptance_proved,
                expanded_retry_timeouts=tuple(expanded_retry_timeouts),
                incumbent_after=_identity(backend, current),
                evidence_after=current_evidence,
                utility_after=utility_after,
                counterexample=counterexample,
            )
        )
        if failure is not None:
            break

    while len(trajectory) < config.horizon + 1:
        trajectory.append(_utility(scale, current_evidence))
    terminal_identity = _identity(backend, current)
    status = (
        SerialEvaluationStatus.PROGRAM_FAILURE
        if failure is not None
        else SerialEvaluationStatus.COMPLETE
    )
    auc = episode_auc(trajectory, horizon=config.horizon)
    fitness = (
        _program_failure_fitness()
        if failure is not None
        else candidate_fitness({config.order: (auc,)})
    )
    provisional = SerialEpisodeResult(
        protocol_id=protocol_id,
        program_hash=program_hash,
        status=status,
        config=config,
        initial_identity=initial_identity,
        initial_evidence=initial_evidence,
        initial_counterexample=initial_counterexample,
        steps=tuple(steps),
        terminal_identity=terminal_identity,
        terminal_evidence=current_evidence,
        utility_trajectory=tuple(trajectory),
        best_so_far_utility=best_so_far_curve(trajectory),
        auc_interval=auc,
        fitness_interval=fitness,
        accepted_rewrites=accepted_rewrites,
        score_attempts=scorer.raw_graph_score_calls - attempts_before,
        unique_graph_scores=scorer.unique_graph_scores - unique_before,
        failure=failure,
        scientific_error=(
            f"{failure.code}@{failure.path}: {failure.message}"
            if failure is not None
            else None
        ),
        semantic_trace_hash="",
        execution_trace_protocol_id=execution_trace_protocol_id,
    )
    payload = provisional.as_dict(
        include_hash=False,
        include_telemetry=False,
    )
    return replace(
        provisional,
        semantic_trace_hash=_trace_hash(payload),
    )


def evaluate_serial_program(
    *,
    backend: GraphBackend,
    scorer: ScoreEvidenceScorer,
    program: ValidatedProgram,
    config: SerialEpisodeConfig,
    interpreter_limits: InterpreterLimits | None = None,
    program_contract: ProgramContract | None = None,
    counterexample_pipeline: CounterexampleInspector | None = None,
    provenance_source_kind: str = "native_v3_fixture",
) -> SerialEpisodeResult:
    """Run the existing JSON-DSL trajectory without changing its artifact shape."""

    from .interpreter import ProgramContext, invoke_program

    invocation_episode_id = f"{config.episode_id}/policy-{config.policy_seed}"

    def invoke_step(
        graph: GraphState,
        step_index: int,
        accepted_rewrites: int,
    ) -> _SerialInvocation:
        host = _CapturingRewriteHost(backend)
        invocation = invoke_program(
            program,
            graph,
            rewrite_host=host,
            context=ProgramContext(
                step_index=step_index,
                horizon=config.horizon,
                acceptance_profile_id="strict_improvement",
                stagnation_steps=step_index - accepted_rewrites,
                accepted_rewrites=accepted_rewrites,
                witness_cap=config.witness_cap,
                invocation_ordinal=step_index,
            ),
            episode_id=invocation_episode_id,
            limits=interpreter_limits,
            contract=program_contract,
        )
        return _SerialInvocation(
            semantic_trace=invocation.semantic_trace,
            no_plan_reason=(
                invocation.no_plan.reason if invocation.no_plan is not None else None
            ),
            failure=invocation.failure,
            rewrite=invocation.rewrite,
            candidate=host.candidate if invocation.rewrite is not None else None,
        )

    return _evaluate_serial(
        backend=backend,
        scorer=scorer,
        program_hash=program.program_hash,
        config=config,
        invoke_step=invoke_step,
        counterexample_pipeline=counterexample_pipeline,
        provenance_source_kind=provenance_source_kind,
        protocol_id=SERIAL_EVALUATOR_PROTOCOL_ID,
    )
