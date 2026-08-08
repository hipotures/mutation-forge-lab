"""Isolated contracts for Native v3 declarative mutation programs."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .canonical import (
    CANONICAL_PROTOCOL_ID,
    CanonicalJsonError,
    canonical_json_bytes,
    domain_hash,
    parse_strict_json,
    program_hash,
)
from .contracts import (
    PROGRAM_SCHEMA_VERSION,
    VALIDATOR_PROTOCOL_ID,
    ProgramDiagnostic,
    ProgramLimits,
    ProgramValidation,
    ValidatedProgram,
    validate_program,
    validated_program_artifact,
)
from .execution import ProgramFailure, SemanticEvent
from .graph_runtime import (
    GRAPH_RUNTIME_PROTOCOL_ID,
    EdgeRef,
    EdgeSetRef,
    GraphFeatureInput,
    MatchingRef,
    NonEdgeRef,
    PathRef,
    RewriteHost,
    SelectionPopulation,
    VertexRef,
    VertexSetRef,
)
from .randomness import (
    RANDOM_PROTOCOL_ID,
    derive_seed64,
    draw64,
    splitmix64,
    uniform_below,
    weighted_index,
)
from .scoring import (
    FITNESS_PROTOCOL_ID,
    SCORE_EVIDENCE_SCHEMA_VERSION,
    SCORE_PROTOCOL_ID,
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EnergyScale,
    EvidenceStatus,
    IntegerInterval,
    RationalInterval,
    ScoreEvidence,
    ScoreTimeoutWithoutPartial,
    best_so_far_curve,
    candidate_fitness,
    conservative_fitness_key,
    episode_auc,
    proved_strict_energy_improvement,
)
from .serial_evaluator import (
    SERIAL_EVALUATOR_PROTOCOL_ID,
    CounterexampleInspector,
    CounterexampleTrace,
    GraphIdentity,
    SerialEpisodeConfig,
    SerialEpisodeResult,
    SerialEvaluationStatus,
    SerialStepTrace,
    evaluate_serial_program,
)
from .single_program_contract import (
    SINGLE_PROGRAM_RESPONSE_SCHEMA_VERSION,
    SingleProgramContractError,
    SingleProgramRequest,
    SingleProgramResponse,
    build_single_program_contract,
    build_single_program_output_schema,
    build_single_program_request,
    model_facing_contract,
    single_program_request_size_bytes,
    validate_single_program_response,
)

if TYPE_CHECKING:
    from .interpreter import (
        INTERPRETER_PROTOCOL_ID,
        BranchFailureCode,
        CatchableBranchFailure,
        InterpreterLimits,
        InvocationCounters,
        InvocationResult,
        NoPlan,
        ProgramContext,
        invoke_program,
    )

_INTERPRETER_EXPORTS = frozenset(
    {
        "INTERPRETER_PROTOCOL_ID",
        "BranchFailureCode",
        "CatchableBranchFailure",
        "InterpreterLimits",
        "InvocationCounters",
        "InvocationResult",
        "NoPlan",
        "ProgramContext",
        "invoke_program",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _INTERPRETER_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.interpreter"), name)
    globals()[name] = value
    return value

__all__ = [
    "CANONICAL_PROTOCOL_ID",
    "FITNESS_PROTOCOL_ID",
    "GRAPH_RUNTIME_PROTOCOL_ID",
    "INTERPRETER_PROTOCOL_ID",
    "PROGRAM_SCHEMA_VERSION",
    "RANDOM_PROTOCOL_ID",
    "SCORE_EVIDENCE_SCHEMA_VERSION",
    "SCORE_PROTOCOL_ID",
    "SERIAL_EVALUATOR_PROTOCOL_ID",
    "SINGLE_PROGRAM_RESPONSE_SCHEMA_VERSION",
    "VALIDATOR_PROTOCOL_ID",
    "BranchFailureCode",
    "AttemptKind",
    "BackendIdentity",
    "CanonicalJsonError",
    "CatchableBranchFailure",
    "CounterexampleInspector",
    "CounterexampleTrace",
    "CycleComponentEvidence",
    "EdgeRef",
    "EdgeSetRef",
    "EnergyScale",
    "EvidenceStatus",
    "GraphFeatureInput",
    "GraphIdentity",
    "InterpreterLimits",
    "IntegerInterval",
    "InvocationCounters",
    "InvocationResult",
    "MatchingRef",
    "NonEdgeRef",
    "NoPlan",
    "PathRef",
    "ProgramContext",
    "ProgramDiagnostic",
    "ProgramFailure",
    "ProgramLimits",
    "ProgramValidation",
    "RationalInterval",
    "RewriteHost",
    "SemanticEvent",
    "SelectionPopulation",
    "SerialEpisodeConfig",
    "SerialEpisodeResult",
    "SerialEvaluationStatus",
    "SerialStepTrace",
    "ScoreEvidence",
    "ScoreTimeoutWithoutPartial",
    "SingleProgramContractError",
    "SingleProgramRequest",
    "SingleProgramResponse",
    "ValidatedProgram",
    "VertexRef",
    "VertexSetRef",
    "canonical_json_bytes",
    "best_so_far_curve",
    "build_single_program_contract",
    "build_single_program_output_schema",
    "build_single_program_request",
    "candidate_fitness",
    "conservative_fitness_key",
    "derive_seed64",
    "domain_hash",
    "draw64",
    "episode_auc",
    "evaluate_serial_program",
    "invoke_program",
    "model_facing_contract",
    "parse_strict_json",
    "program_hash",
    "proved_strict_energy_improvement",
    "splitmix64",
    "single_program_request_size_bytes",
    "uniform_below",
    "validate_program",
    "validate_single_program_response",
    "validated_program_artifact",
    "weighted_index",
]
