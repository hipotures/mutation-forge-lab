"""Deterministic serial evaluation of one Native v3 program."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from mutation_forge.backends.base import GraphBackend, ScoreProfileRecorder
from mutation_forge.counterexamples import (
    CandidateProvenance,
    CounterexampleOutcome,
)
from mutation_forge.models import GraphScore, GraphState, JsonValue, RewritePlan

from .canonical import canonical_json_bytes, domain_hash
from .contracts import ValidatedProgram
from .interpreter import (
    InterpreterLimits,
    ProgramContext,
    ProgramFailure,
    SemanticEvent,
    invoke_program,
)

SERIAL_EVALUATOR_PROTOCOL_ID = "native_v3_serial_evaluator_v1"
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
    score_before: GraphScore
    interpreter_trace: tuple[SemanticEvent, ...]
    outcome: str
    no_plan_reason: str | None
    failure: ProgramFailure | None
    rewrite: RewritePlan | None
    candidate_identity: GraphIdentity | None
    score_after: GraphScore | None
    accepted: bool
    incumbent_after: GraphIdentity
    counterexample: CounterexampleTrace | None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "step_index": self.step_index,
            "incumbent_before": self.incumbent_before.as_dict(),
            "score_before": self.score_before.as_dict(),
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
            "score_after": (
                self.score_after.as_dict() if self.score_after is not None else None
            ),
            "accepted": self.accepted,
            "incumbent_after": self.incumbent_after.as_dict(),
            "counterexample": (
                self.counterexample.as_dict() if self.counterexample is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SerialEpisodeResult:
    protocol_id: str
    program_hash: str
    config: SerialEpisodeConfig
    initial_identity: GraphIdentity
    initial_score: GraphScore
    initial_counterexample: CounterexampleTrace | None
    steps: tuple[SerialStepTrace, ...]
    terminal_identity: GraphIdentity
    terminal_score: GraphScore
    accepted_rewrites: int
    failure: ProgramFailure | None
    semantic_trace_hash: str

    def as_dict(self, *, include_hash: bool = True) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "protocol_id": self.protocol_id,
            "program_hash": self.program_hash,
            "config": {
                "order": self.config.order,
                "graph_seed": self.config.graph_seed,
                "policy_seed": self.config.policy_seed,
                "horizon": self.config.horizon,
                "witness_cap": self.config.witness_cap,
                "episode_id": self.config.episode_id,
            },
            "initial_identity": self.initial_identity.as_dict(),
            "initial_score": self.initial_score.as_dict(),
            "initial_counterexample": (
                self.initial_counterexample.as_dict()
                if self.initial_counterexample is not None
                else None
            ),
            "steps": [step.as_dict() for step in self.steps],
            "terminal_identity": self.terminal_identity.as_dict(),
            "terminal_score": self.terminal_score.as_dict(),
            "accepted_rewrites": self.accepted_rewrites,
            "failure": (
                {
                    "code": self.failure.code,
                    "path": self.failure.path,
                    "message": self.failure.message,
                }
                if self.failure is not None
                else None
            ),
        }
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
    score: GraphScore,
    config: SerialEpisodeConfig,
    program: ValidatedProgram,
    step_index: int,
) -> CounterexampleTrace | None:
    if pipeline is None or score.total_capped_witnesses != 0:
        return None
    outcome: CounterexampleOutcome = pipeline.inspect(
        graph=graph,
        score=score,
        witness_cap=config.witness_cap,
        provenance=CandidateProvenance(
            source_kind="native_v3_fixture",
            source_id=program.program_hash,
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


def evaluate_serial_program(
    *,
    backend: GraphBackend,
    program: ValidatedProgram,
    config: SerialEpisodeConfig,
    interpreter_limits: InterpreterLimits | None = None,
    counterexample_pipeline: CounterexampleInspector | None = None,
) -> SerialEpisodeResult:
    """Run one deterministic strict-improvement trajectory without a provider."""

    initial = backend.generate_seed(order=config.order, seed=config.graph_seed)
    validation = backend.validate(initial)
    if not validation.valid:
        raise ValueError(f"backend generated an invalid seed: {validation.errors}")
    initial_score = backend.score(initial, witness_cap=config.witness_cap)
    if initial_score is None:
        raise RuntimeError("initial score cannot be cutoff-dominated")
    initial_identity = _identity(backend, initial)
    initial_counterexample = _inspect_apparent_zero(
        pipeline=counterexample_pipeline,
        backend=backend,
        graph=initial,
        score=initial_score,
        config=config,
        program=program,
        step_index=0,
    )

    current = initial
    current_score = initial_score
    accepted_rewrites = 0
    failure: ProgramFailure | None = None
    steps: list[SerialStepTrace] = []
    invocation_episode_id = f"{config.episode_id}/policy-{config.policy_seed}"

    for step_index in range(config.horizon):
        before_identity = _identity(backend, current)
        score_before = current_score
        host = _CapturingRewriteHost(backend)
        invocation = invoke_program(
            program,
            current,
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
        )
        outcome = "failure"
        no_plan_reason: str | None = None
        rewrite: RewritePlan | None = None
        candidate_identity: GraphIdentity | None = None
        candidate_score: GraphScore | None = None
        accepted = False
        counterexample: CounterexampleTrace | None = None

        if invocation.failure is not None:
            failure = invocation.failure
        elif invocation.no_plan is not None:
            outcome = "no_plan"
            no_plan_reason = invocation.no_plan.reason
        else:
            rewrite = invocation.rewrite
            candidate = host.candidate
            if rewrite is None or candidate is None:
                failure = ProgramFailure(
                    "INTERPRETER_FAULT",
                    "/entry",
                    "rewrite outcome did not retain its host-validated candidate",
                )
            else:
                outcome = "rewrite"
                candidate_identity = _identity(backend, candidate)
                candidate_score = backend.score(
                    candidate,
                    witness_cap=config.witness_cap,
                )
                if candidate_score is None:
                    failure = ProgramFailure(
                        "SCORING_FAILURE",
                        f"/steps/{step_index}",
                        "authoritative score unexpectedly returned no result",
                    )
                else:
                    counterexample = _inspect_apparent_zero(
                        pipeline=counterexample_pipeline,
                        backend=backend,
                        graph=candidate,
                        score=candidate_score,
                        config=config,
                        program=program,
                        step_index=step_index + 1,
                    )
                    accepted = candidate_score.ordering_key < current_score.ordering_key
                    if accepted:
                        current = candidate
                        current_score = candidate_score
                        accepted_rewrites += 1

        steps.append(
            SerialStepTrace(
                step_index=step_index,
                incumbent_before=before_identity,
                score_before=score_before,
                interpreter_trace=invocation.semantic_trace,
                outcome=outcome,
                no_plan_reason=no_plan_reason,
                failure=failure,
                rewrite=rewrite,
                candidate_identity=candidate_identity,
                score_after=candidate_score,
                accepted=accepted,
                incumbent_after=_identity(backend, current),
                counterexample=counterexample,
            )
        )
        if failure is not None:
            break

    terminal_identity = _identity(backend, current)
    provisional = SerialEpisodeResult(
        protocol_id=SERIAL_EVALUATOR_PROTOCOL_ID,
        program_hash=program.program_hash,
        config=config,
        initial_identity=initial_identity,
        initial_score=initial_score,
        initial_counterexample=initial_counterexample,
        steps=tuple(steps),
        terminal_identity=terminal_identity,
        terminal_score=current_score,
        accepted_rewrites=accepted_rewrites,
        failure=failure,
        semantic_trace_hash="",
    )
    payload = provisional.as_dict(include_hash=False)
    return replace(
        provisional,
        semantic_trace_hash=_trace_hash(payload),
    )
