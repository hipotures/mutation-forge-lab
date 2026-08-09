"""Fixture-only ordinary-Python adapter for the Native v3 serial evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from mutation_forge.backends.base import GraphBackend, InvalidRewriteError
from mutation_forge.models import GraphState, JsonValue, RewritePlan
from mutation_forge.native_v3.canonical import canonical_json_bytes, domain_hash
from mutation_forge.native_v3.execution import ProgramFailure, SemanticEvent
from mutation_forge.native_v3.heg_scoring import ScoreEvidenceScorer
from mutation_forge.native_v3.serial_evaluator import (
    CounterexampleInspector,
    SerialEpisodeConfig,
    SerialEpisodeResult,
    _evaluate_serial,
    _SerialInvocation,
)

from .contracts import BehaviorIdentityV1, PolicyContextV1
from .runner import IsolatedPolicyWorkerV1
from .runtime_contracts import (
    SEMANTIC_TRACE_PROTOCOL_ID,
    GraphFeatureInputV1,
    IllegalRewriteError,
    PolicyRuntimeLimitsV1,
)
from .validation import PythonProgramIdentityV1, validate_python_policy_source

PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID = "native_v3_python_serial_interval_evaluator_v1"
PYTHON_SERIAL_RESULT_PROTOCOL_ID = "mforge.native.python_serial_evaluation.v1"
PYTHON_FIXTURE_PROVENANCE_SOURCE_KIND = "native_v3_python_fixture"

_BEHAVIOR_MANIFEST_DOMAIN = b"mforge-native-v3-python-behavior-manifest-v1\0"
_BEHAVIOR_SIGNATURE_DOMAIN = b"mforge-native-v3-python-behavior-signature-v1\0"


@dataclass(frozen=True, slots=True)
class PythonSerialEpisodeConfigV1:
    """Immutable scientific episode inputs exposed to an ordinary-Python policy."""

    order: int
    graph_seed: int
    policy_seed: int
    horizon: int
    witness_cap: int
    episode_id: str
    forbidden_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        SerialEpisodeConfig(
            order=self.order,
            graph_seed=self.graph_seed,
            policy_seed=self.policy_seed,
            horizon=self.horizon,
            witness_cap=self.witness_cap,
            episode_id=self.episode_id,
        )
        PolicyContextV1(
            step_index=0,
            horizon=self.horizon,
            acceptance_profile_id="strict_improvement",
            stagnation_steps=0,
            exploration_window_index=0,
            accepted_rewrites=0,
            accepted_non_improving_rewrites=0,
            consecutive_non_improving_rewrites=0,
            witness_cap=self.witness_cap,
            invocation_ordinal=0,
            forbidden_lengths=self.forbidden_lengths,
        )

    def serial_config(self) -> SerialEpisodeConfig:
        return SerialEpisodeConfig(
            order=self.order,
            graph_seed=self.graph_seed,
            policy_seed=self.policy_seed,
            horizon=self.horizon,
            witness_cap=self.witness_cap,
            episode_id=self.episode_id,
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "order": self.order,
            "graph_seed": self.graph_seed,
            "policy_seed": self.policy_seed,
            "horizon": self.horizon,
            "witness_cap": self.witness_cap,
            "episode_id": self.episode_id,
            "forbidden_lengths": list(self.forbidden_lengths),
        }


@dataclass(slots=True)
class _PythonRewriteHost:
    """Translate only documented invalid rewrites into candidate NoPlan."""

    backend: GraphBackend
    candidate: GraphState | None = None

    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState:
        try:
            candidate = self.backend.apply_rewrite(graph, rewrite)
        except InvalidRewriteError as error:
            raise IllegalRewriteError(str(error)) from error
        self.candidate = candidate
        return candidate


@dataclass(frozen=True, slots=True)
class PythonSerialEpisodeResultV1:
    """M3 fixture evaluation with separate program and behavior identities."""

    program_identity: PythonProgramIdentityV1
    behavior_identity: BehaviorIdentityV1
    config: PythonSerialEpisodeConfigV1
    scientific_result: SerialEpisodeResult
    worker_telemetry: dict[str, JsonValue]
    runtime_profile: dict[str, JsonValue]
    protocol_id: str = PYTHON_SERIAL_RESULT_PROTOCOL_ID

    def as_dict(
        self,
        *,
        include_telemetry: bool = True,
        include_external_activity: bool = True,
    ) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "protocol_id": self.protocol_id,
            "program_identity": self.program_identity.as_dict(),
            "behavior_identity": {
                "protocol_id": self.behavior_identity.protocol_id,
                "probe_manifest_sha256": (
                    self.behavior_identity.probe_manifest_sha256
                ),
                "behavior_signature": self.behavior_identity.behavior_signature,
            },
            "config": self.config.as_dict(),
            "scientific_result": self.scientific_result.as_dict(
                include_telemetry=include_telemetry
            ),
        }
        if include_external_activity:
            result["external_activity"] = {
                "provider_turns": 0,
                "model_turns": 0,
                "app_server_calls": 0,
            }
        if include_telemetry:
            result["worker_telemetry"] = self.worker_telemetry
            result["runtime_profile"] = self.runtime_profile
        return result


def _behavior_rewrite_payload(
    rewrite: RewritePlan | None,
) -> dict[str, JsonValue] | None:
    if rewrite is None:
        return None
    return {
        "removed_edges": [list(edge) for edge in rewrite.removed_edges],
        "added_edges": [list(edge) for edge in rewrite.added_edges],
        "operator_family": rewrite.operator_family,
        "metadata": {
            key: value
            for key, value in rewrite.metadata.items()
            if key != "program_hash"
        },
    }


def _behavior_identity(
    *,
    config: PythonSerialEpisodeConfigV1,
    result: SerialEpisodeResult,
) -> BehaviorIdentityV1:
    manifest = {
        "protocol_id": PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
        "semantic_trace_protocol_id": SEMANTIC_TRACE_PROTOCOL_ID,
        "config": config.as_dict(),
        "initial_identity": result.initial_identity.as_dict(),
    }
    behavior = {
        "manifest": manifest,
        "steps": [
            {
                "step_index": step.step_index,
                "incumbent_before": step.incumbent_before.as_dict(),
                "api_events": [event.as_dict() for event in step.interpreter_trace],
                "outcome": step.outcome,
                "no_plan_reason": step.no_plan_reason,
                "failure": (
                    {
                        "code": step.failure.code,
                        "path": step.failure.path,
                        "message": step.failure.message,
                    }
                    if step.failure is not None
                    else None
                ),
                "rewrite": _behavior_rewrite_payload(step.rewrite),
                "candidate_identity": (
                    step.candidate_identity.as_dict()
                    if step.candidate_identity is not None
                    else None
                ),
            }
            for step in result.steps
        ],
    }
    return BehaviorIdentityV1(
        probe_manifest_sha256=domain_hash(
            _BEHAVIOR_MANIFEST_DOMAIN,
            canonical_json_bytes(manifest),
        ),
        behavior_signature=domain_hash(
            _BEHAVIOR_SIGNATURE_DOMAIN,
            canonical_json_bytes(behavior),
        ),
    )


def evaluate_serial_python_policy(
    *,
    backend: GraphBackend,
    scorer: ScoreEvidenceScorer,
    source: str,
    config: PythonSerialEpisodeConfigV1,
    runtime_limits: PolicyRuntimeLimitsV1 | None = None,
    counterexample_pipeline: CounterexampleInspector | None = None,
    features: GraphFeatureInputV1 | None = None,
    provenance_source_kind: str = PYTHON_FIXTURE_PROVENANCE_SOURCE_KIND,
) -> PythonSerialEpisodeResultV1:
    """Evaluate one checked-in Python fixture through the M2 worker and M3 core."""

    validation = validate_python_policy_source(source)
    if (
        not validation.valid
        or validation.identity is None
        or validation.identity.program_hash is None
    ):
        raise ValueError(
            "ordinary-Python policy failed M1 validation: "
            + ",".join(item.code for item in validation.diagnostics)
        )
    identity = validation.identity
    program_hash = identity.program_hash
    if program_hash is None:
        raise ValueError("valid ordinary-Python policy identity is missing program_hash")
    backend_lengths = backend.target_forbidden_lengths(config.order)
    if backend_lengths != config.forbidden_lengths:
        raise ValueError(
            "episode forbidden_lengths do not match the authoritative backend target"
        )

    worker = IsolatedPolicyWorkerV1(source, limits=runtime_limits)
    invocation_wall_seconds = 0.0
    selector_wall_seconds = 0.0
    action_wall_seconds = 0.0
    try:

        def invoke_step(
            graph: GraphState,
            step_index: int,
            accepted_rewrites: int,
        ) -> _SerialInvocation:
            nonlocal action_wall_seconds
            nonlocal invocation_wall_seconds
            nonlocal selector_wall_seconds
            host = _PythonRewriteHost(backend)
            invocation = worker.invoke(
                context=PolicyContextV1(
                    step_index=step_index,
                    horizon=config.horizon,
                    acceptance_profile_id="strict_improvement",
                    stagnation_steps=step_index - accepted_rewrites,
                    exploration_window_index=0,
                    accepted_rewrites=accepted_rewrites,
                    accepted_non_improving_rewrites=0,
                    consecutive_non_improving_rewrites=0,
                    witness_cap=config.witness_cap,
                    invocation_ordinal=step_index,
                    forbidden_lengths=config.forbidden_lengths,
                ),
                graph=graph,
                rewrite_host=host,
                seed=config.policy_seed,
                features=features,
            )
            invocation_wall_seconds += invocation.wall_seconds
            selector_wall_seconds += invocation.selector_wall_seconds
            action_wall_seconds += invocation.action_wall_seconds
            failure = (
                ProgramFailure(
                    code=invocation.failure.code,
                    path="/propose",
                    message=invocation.failure.message,
                )
                if invocation.failure is not None
                else None
            )
            return _SerialInvocation(
                semantic_trace=cast(
                    tuple[SemanticEvent, ...],
                    invocation.semantic_trace,
                ),
                no_plan_reason=(
                    invocation.no_plan.reason
                    if invocation.no_plan is not None
                    else None
                ),
                failure=failure,
                rewrite=invocation.rewrite_plan,
                candidate=(
                    host.candidate if invocation.rewrite_plan is not None else None
                ),
            )

        result = _evaluate_serial(
            backend=backend,
            scorer=scorer,
            program_hash=program_hash,
            config=config.serial_config(),
            invoke_step=invoke_step,
            counterexample_pipeline=counterexample_pipeline,
            provenance_source_kind=provenance_source_kind,
            protocol_id=PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
            execution_trace_protocol_id=SEMANTIC_TRACE_PROTOCOL_ID,
            forbidden_lengths=config.forbidden_lengths,
        )
        telemetry = worker.telemetry()
    finally:
        worker.close()

    return PythonSerialEpisodeResultV1(
        program_identity=identity,
        behavior_identity=_behavior_identity(
            config=config,
            result=result,
        ),
        config=config,
        scientific_result=result,
        worker_telemetry=telemetry,
        runtime_profile={
            "sandbox_wall_seconds": invocation_wall_seconds,
            "selector_wall_seconds": selector_wall_seconds,
            "action_wall_seconds": action_wall_seconds,
        },
    )
