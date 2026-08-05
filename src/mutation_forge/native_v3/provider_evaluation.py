"""One-slot Native v3 provider-to-evaluation smoke orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mutation_forge.artifacts import git_state
from mutation_forge.backends.base import GraphBackend
from mutation_forge.counterexamples import CounterexamplePipeline
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider

from .canonical import CANONICAL_PROTOCOL_ID
from .contracts import (
    VALIDATOR_PROTOCOL_ID,
    ValidatedProgram,
    validate_program,
    validated_program_artifact,
)
from .graph_runtime import GRAPH_RUNTIME_PROTOCOL_ID
from .heg_scoring import scorer_for_backend
from .interpreter import INTERPRETER_PROTOCOL_ID
from .provider_smoke import NativeV3ProviderSmokeError, run_provider_smoke
from .scoring import FITNESS_PROTOCOL_ID, SCORE_PROTOCOL_ID
from .serial_evaluator import (
    SERIAL_EVALUATOR_PROTOCOL_ID,
    SerialEpisodeConfig,
    SerialEpisodeResult,
    SerialEvaluationStatus,
    evaluate_serial_program,
)

PROVIDER_EVALUATION_SCHEMA_VERSION = "mforge.native-v3-provider-evaluation-smoke.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_validated_program(path: Path) -> ValidatedProgram:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise NativeV3ProviderSmokeError("validated program artifact is not an object")
    raw = payload.get("program_json_raw")
    if not isinstance(raw, str):
        raise NativeV3ProviderSmokeError("validated program artifact has no raw program")
    validation = validate_program(raw)
    if validation.program is None:
        raise NativeV3ProviderSmokeError("validated program artifact no longer validates")
    program = validation.program
    expected = {
        "program_json_canonical": program.canonical_json,
        "program_hash": program.program_hash,
        "validator_protocol_id": VALIDATOR_PROTOCOL_ID,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise NativeV3ProviderSmokeError(
                f"validated program artifact {key} does not match its raw program"
            )
    return program


def _provider_provenance(turn: Path) -> dict[str, Any]:
    provenance = read_json(turn / "provenance.json.gz")
    identity = read_json(turn / "identity.json.gz")
    if not isinstance(provenance, Mapping) or not isinstance(identity, Mapping):
        raise NativeV3ProviderSmokeError("provider provenance artifacts are invalid")
    return {
        "identity": dict(identity),
        "provenance": dict(provenance),
        "turn_directory": str(turn),
    }


def _backend_provenance(backend: GraphBackend) -> dict[str, Any]:
    repo = getattr(backend, "repo", None)
    return {
        "graph_backend_id": backend.backend_id,
        "score_implementation": getattr(backend, "score_implementation", None),
        "repo": str(repo) if repo is not None else None,
        "commit": getattr(backend, "commit", None),
        "dirty": getattr(backend, "dirty", None),
    }


def _graph_evaluations(result: SerialEpisodeResult) -> int:
    return result.score_attempts


def run_provider_evaluation_smoke(
    provider: LocalCodexAppServerProvider,
    experiment_root: str | Path,
    *,
    backend_factory: Callable[[], GraphBackend],
    config: SerialEpisodeConfig,
) -> dict[str, Any]:
    """Run one provider turn, then one serial scientific evaluation."""

    root = Path(experiment_root)
    output_root = root / "native-v3-output"
    report_path = output_root / "provider-evaluation-smoke-report.json.gz"
    evaluation_path = output_root / "evaluation-result.json.gz"
    provider_report = run_provider_smoke(provider, root)
    if provider_report["status"] != "completed":
        report = {
            "schema_version": PROVIDER_EVALUATION_SCHEMA_VERSION,
            "status": "provider_error",
            "valid_ast": False,
            "model_turns": provider_report["model_turns"],
            "graph_evaluations": 0,
            "scientific_terminal_result": False,
            "provider_report": str(
                output_root / "provider-smoke-report.json.gz"
            ),
            "error_classification": provider_report.get("error_classification"),
            "error_type": provider_report.get("error_type"),
            "error": provider_report.get("error"),
            "usage": provider_report.get("usage"),
            "resumable": bool(provider_report.get("resumable")),
        }
        write_json(report_path, report)
        return report

    backend: GraphBackend | None = None
    try:
        program_path = Path(str(provider_report["validated_program"]))
        program = _load_validated_program(program_path)
        turn = Path(str(provider_report["turn_directory"]))
        provider_provenance = _provider_provenance(turn)
        backend = backend_factory()
        scorer = scorer_for_backend(backend)
        pipeline = CounterexamplePipeline(
            backend=backend,
            artifact_root=output_root,
        )
        evaluation = evaluate_serial_program(
            backend=backend,
            scorer=scorer,
            program=program,
            config=config,
            counterexample_pipeline=pipeline,
            provenance_source_kind="native_v3_provider",
        )
        backend_provenance = _backend_provenance(backend)
    except Exception as error:
        report = {
            "schema_version": PROVIDER_EVALUATION_SCHEMA_VERSION,
            "status": "evaluation_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "valid_ast": True,
            "model_turns": 1,
            "graph_evaluations": 0,
            "scientific_terminal_result": False,
            "provider_report": str(
                output_root / "provider-smoke-report.json.gz"
            ),
            "resumable": True,
        }
        write_json(report_path, report)
        return report
    finally:
        if backend is not None:
            backend.close()

    semantic_payload = {
        "schema_version": PROVIDER_EVALUATION_SCHEMA_VERSION,
        "slot": "slot-00",
        "program": validated_program_artifact(program),
        "protocols": {
            "canonical_json": CANONICAL_PROTOCOL_ID,
            "validator": VALIDATOR_PROTOCOL_ID,
            "interpreter": INTERPRETER_PROTOCOL_ID,
            "graph_runtime": GRAPH_RUNTIME_PROTOCOL_ID,
            "serial_evaluator": SERIAL_EVALUATOR_PROTOCOL_ID,
            "score_evidence": SCORE_PROTOCOL_ID,
            "fitness": FITNESS_PROTOCOL_ID,
        },
        "provider": provider_provenance,
        "backend": backend_provenance,
        "mutation_forge": git_state(PROJECT_ROOT),
        "evaluation": evaluation.as_dict(),
    }
    write_json(evaluation_path, semantic_payload, exclusive=True)
    graph_evaluations = _graph_evaluations(evaluation)
    completed = evaluation.status is SerialEvaluationStatus.COMPLETE
    report = {
        "schema_version": PROVIDER_EVALUATION_SCHEMA_VERSION,
        "status": "completed" if completed else "evaluation_failed",
        "valid_ast": True,
        "program_hash": program.program_hash,
        "model_turns": 1,
        "graph_evaluations": graph_evaluations,
        "semantic_trace_hash": evaluation.semantic_trace_hash,
        "scientific_terminal_result": completed,
        "usage": provider_report.get("usage"),
        "provider_turn_directory": str(provider_report["turn_directory"]),
        "evaluation_result": str(evaluation_path),
        "resumable": True,
    }
    write_json(report_path, report)
    return report


__all__ = [
    "PROVIDER_EVALUATION_SCHEMA_VERSION",
    "run_provider_evaluation_smoke",
]
