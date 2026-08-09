"""One-root model-to-Python scientific evaluation for Native v3 M4."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mutation_forge.artifacts import git_state
from mutation_forge.backends.base import GraphBackend
from mutation_forge.counterexamples import (
    CandidateProvenance,
    CounterexampleOutcome,
    CounterexamplePipeline,
)
from mutation_forge.experiment.artifacts import (
    NATIVE_V3_PYTHON_POLICY_PROJECTION,
    TurnArtifactStore,
    copy_canonical_source,
)
from mutation_forge.experiment.json_io import write_json
from mutation_forge.experiment.provider import (
    AuthenticationError,
    LocalCodexAppServerProvider,
    NativeProviderError,
)
from mutation_forge.models import GraphScore, GraphState, JsonValue
from mutation_forge.native_v3.heg_scoring import (
    ScoreEvidenceScorer,
    scorer_for_backend,
)
from mutation_forge.native_v3.scoring import (
    AttemptKind,
    ScoreEvidence,
    ScoreTimeoutWithoutPartial,
)
from mutation_forge.native_v3.serial_evaluator import SerialEvaluationStatus

from .runtime_contracts import PolicyInfrastructureError, PolicyRuntimeLimitsV1
from .serial_evaluator import (
    PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
    PythonSerialEpisodeConfigV1,
    PythonSerialEpisodeResultV1,
    evaluate_serial_python_policy,
)
from .validation import (
    PythonPolicyValidation,
    normalize_source_newlines,
    validate_python_policy_response,
)

M4_REPORT_SCHEMA_VERSION = "mforge.native.python_m4_root_evaluation.v1"
M4_EVALUATION_SCHEMA_VERSION = "mforge.native.python_m4_scientific_result.v1"
M4_CAMPAIGN_ID = "native-v3-python-m4-single-root"
M4_PROVENANCE_SOURCE_KIND = "native_v3_python_model_root"
MAX_REPAIR_DIAGNOSTICS = 32
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_HEX_ID = re.compile(r"(?i)\b[0-9a-f]{64}\b")
_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_PRIVATE_PATH = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)")
_FORBIDDEN_PROMPT_TERMS = (
    "thread_id",
    "thread id",
    "turn_id",
    "turn id",
    "program_hash",
    "source_sha256",
    "canonical_ast_sha256",
    "workspace path",
    "provider state",
    "held-out",
)


class M4RootEvaluationError(RuntimeError):
    """The single-root M4 boundary could not produce a scientific result."""


class M4ScoringError(RuntimeError):
    """The authoritative component scorer failed outside a safe timeout."""


class M4VerificationError(RuntimeError):
    """The exact-verification seam failed to return a valid outcome."""


class _TrackingScorer:
    def __init__(self, scorer: ScoreEvidenceScorer) -> None:
        self.scorer = scorer
        self.raw_graph_score_calls = scorer.raw_graph_score_calls
        self.unique_graph_scores = scorer.unique_graph_scores

    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        try:
            return self.scorer.score_evidence(
                graph,
                witness_cap=witness_cap,
                forbidden_lengths=forbidden_lengths,
                attempt_kind=attempt_kind,
            )
        except ScoreTimeoutWithoutPartial:
            raise
        except Exception as error:
            raise M4ScoringError(str(error)) from error
        finally:
            self.raw_graph_score_calls = self.scorer.raw_graph_score_calls
            self.unique_graph_scores = self.scorer.unique_graph_scores


class _TrackingCounterexamplePipeline:
    def __init__(self, pipeline: CounterexamplePipeline) -> None:
        self.pipeline = pipeline

    def inspect(
        self,
        *,
        graph: GraphState,
        score: GraphScore,
        provenance: CandidateProvenance,
        witness_cap: int,
    ) -> CounterexampleOutcome:
        try:
            return self.pipeline.inspect(
                graph=graph,
                score=score,
                provenance=provenance,
                witness_cap=witness_cap,
            )
        except Exception as error:
            raise M4VerificationError(str(error)) from error


def _load_text(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _load_schema() -> dict[str, Any]:
    value = json.loads(
        _load_text("configs/native/native-v3-python-policy-response.schema.json")
    )
    if not isinstance(value, dict):
        raise M4RootEvaluationError("ordinary-Python response schema is not an object")
    return value


def _prompt_hygiene(system_prompt: str, prompt: str) -> dict[str, JsonValue]:
    combined = f"{system_prompt}\n{prompt}"
    findings: list[str] = []
    if _HEX_ID.search(combined):
        findings.append("cryptographic_hash")
    if _UUID.search(combined):
        findings.append("uuid")
    if _PRIVATE_PATH.search(combined):
        findings.append("private_path")
    lowered = combined.lower()
    findings.extend(term for term in _FORBIDDEN_PROMPT_TERMS if term in lowered)
    return {
        "valid": not findings,
        "findings": cast(JsonValue, sorted(set(findings))),
        "system_prompt_bytes": len(system_prompt.encode("utf-8")),
        "request_prompt_bytes": len(prompt.encode("utf-8")),
    }


def _request_identity(
    *, prompt: str, model: str, effort: str, phase: str
) -> tuple[str, str]:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "campaign": M4_CAMPAIGN_ID,
            "generation": 0,
            "slot": "slot-00",
            "phase": phase,
            "prompt_hash": prompt_hash,
            "model": model,
            "effort": effort,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return prompt_hash, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_m4_request(
    experiment_root: str | Path,
    *,
    model: str,
    effort: str,
    phase: str = "initial",
    repair_attempt: int = 0,
    diagnostics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the exact single-program provider request without host identities."""

    if phase not in {"initial", "repair-01"}:
        raise ValueError("M4 permits only initial or repair-01")
    prompt = _load_text("prompts/native-v3-python/m4-request.md")
    if diagnostics:
        bounded = [
            {
                "code": str(item.get("code", "INVALID"))[:128],
                "path": str(item.get("path", "/"))[:512],
                "message": str(item.get("message", "invalid response"))[:512],
                "line": (
                    item.get("line")
                    if isinstance(item.get("line"), int)
                    and not isinstance(item.get("line"), bool)
                    else None
                ),
                "column": (
                    item.get("column")
                    if isinstance(item.get("column"), int)
                    and not isinstance(item.get("column"), bool)
                    else None
                ),
            }
            for item in diagnostics[:MAX_REPAIR_DIAGNOSTICS]
        ]
        prompt = (
            prompt
            + "\n\n# Repair this response\n\n"
            + "Return a complete replacement program that fixes these validator "
            + "diagnostics:\n\n"
            + json.dumps(bounded, ensure_ascii=False, sort_keys=True, indent=2)
        )
    system_prompt = _load_text("prompts/native-v3-python/m4-system.md").strip()
    hygiene = _prompt_hygiene(system_prompt, prompt)
    if hygiene["valid"] is not True:
        raise M4RootEvaluationError(
            f"model-facing prompt failed hygiene checks: {hygiene['findings']}"
        )
    schema = _load_schema()
    prompt_hash, idempotency_key = _request_identity(
        prompt=prompt,
        model=model,
        effort=effort,
        phase=phase,
    )
    root = Path(experiment_root)
    turn = TurnArtifactStore(root / "artifacts").turn_directory(
        0, "slot-00", phase
    )
    return {
        "campaign_id": M4_CAMPAIGN_ID,
        "generation": 0,
        "slot": "slot-00",
        "phase": phase,
        "parent_id": None,
        "brief_id": "native-v3-python-m4-root",
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "idempotency_key": idempotency_key,
        "model": model,
        "effort": effort,
        "reasoning_effort": effort,
        "system_prompt": system_prompt,
        "output_schema": schema,
        "response_projection": NATIVE_V3_PYTHON_POLICY_PROJECTION,
        "max_repairs": 1,
        "remaining_repairs": 1 - repair_attempt,
        "repair_attempt": repair_attempt,
        "artifact_dir": str(turn),
        "artifact_root": str(root / "artifacts"),
        "artifact_prefix": "slot-00",
    }


def _usage(result: Mapping[str, Any]) -> dict[str, JsonValue]:
    value = result.get("usage")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): cast(JsonValue, item)
        for key, item in value.items()
        if isinstance(item, str | int | float | bool) or item is None
    }


def _failure_classification(error: BaseException) -> str:
    if isinstance(error, AuthenticationError):
        return "authentication"
    if isinstance(error, NativeProviderError | M4RootEvaluationError):
        return "provider"
    if isinstance(error, PolicyInfrastructureError):
        return "infrastructure"
    if isinstance(error, M4ScoringError):
        return "scoring"
    if isinstance(error, M4VerificationError):
        return "verification"
    return "infrastructure"


def _validation_parts(
    validation: PythonPolicyValidation,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue] | None]:
    identity = (
        None if validation.identity is None else validation.identity.as_dict()
    )
    return validation.as_dict(), identity


def _provider_provenance(
    result: Mapping[str, Any],
) -> dict[str, JsonValue]:
    return {
        "model": str(result.get("model", "")),
        "effort": str(result.get("effort", "")),
        "provider_request_id": str(result.get("provider_request_id", "")),
        "provider_thread_id": str(result.get("provider_thread_id", "")),
        "provider_turn_id": str(result.get("provider_turn_id", "")),
        "transport_sha256": str(result.get("transport_sha256", "")),
    }


def _metadata_validation() -> dict[str, JsonValue]:
    return {
        "valid": True,
        "one_root": True,
        "one_program_per_turn": True,
        "parent": None,
        "lineage": [],
        "dsl_runtime_used": False,
    }


def _turn_warnings(turn: Path) -> int:
    count = 0
    for path in turn.glob("*.events.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping) and value.get("method") == "warning":
                count += 1
    return count


def _write_or_index_turn(
    *,
    store: TurnArtifactStore,
    request: Mapping[str, Any],
    raw_result: Mapping[str, Any],
    validation: PythonPolicyValidation,
    phase: str,
    evaluation: PythonSerialEpisodeResultV1 | None,
) -> dict[str, Any]:
    turn = Path(str(request["artifact_dir"]))
    validation_payload, identity = _validation_parts(validation)
    response = (
        validation.response.as_dict()
        if validation.response is not None
        else raw_result.get("response")
    )
    result: dict[str, Any] = {
        **dict(raw_result),
        "response": response,
        "response_projection_valid": validation.valid,
        "response_diagnostics": [
            item.as_dict() for item in validation.diagnostics
        ],
        "validation": validation_payload,
        "identity": identity or {"status": "unavailable"},
        "behavior": (
            {
                "protocol_id": evaluation.behavior_identity.protocol_id,
                "probe_manifest_sha256": (
                    evaluation.behavior_identity.probe_manifest_sha256
                ),
                "behavior_signature": (
                    evaluation.behavior_identity.behavior_signature
                ),
            }
            if evaluation is not None
            else {"status": "unavailable"}
        ),
        "worker_telemetry": (
            evaluation.worker_telemetry
            if evaluation is not None
            else {"status": "not_run"}
        ),
        "provenance": _provider_provenance(raw_result),
        "canonical_response": (
            validation.response.as_dict()
            if validation.response is not None
            else {"status": "invalid"}
        ),
        "metadata_validation": _metadata_validation(),
        "validation_completed": True,
        "uncharged": not bool(raw_result.get("charged")),
    }
    if turn.is_dir():
        usage = raw_result.get("usage")
        artifact_prefix = store.artifact_prefix(turn, raw_result, "slot-00")
        usage_path = turn / f"{artifact_prefix}.usage.json.gz"
        if isinstance(usage, Mapping) and not usage_path.exists():
            write_json(usage_path, usage, exclusive=True)
        manifest = store.record_existing_turn(
            turn,
            generation=0,
            slot="slot-00",
            phase=phase,
            request=request,
            result=result,
        )
    else:
        manifest = store.write_turn(
            generation=0,
            slot="slot-00",
            phase=phase,
            request=request,
            request_text=str(request["prompt"]),
            response=response,
            response_text=str(raw_result.get("response_text", "")),
            source=(
                validation.response.source
                if validation.response is not None
                else None
            ),
            usage=cast(Mapping[str, Any], raw_result.get("usage", {})),
            identity=identity or {"status": "unavailable"},
            behavior=cast(
                Mapping[str, Any],
                result["behavior"],
            ),
            provenance=cast(Mapping[str, Any], result["provenance"]),
            validation=validation_payload,
            worker_telemetry=cast(
                Mapping[str, Any],
                result["worker_telemetry"],
            ),
            canonical_response=cast(
                Mapping[str, Any],
                result["canonical_response"],
            ),
            provider_raw=raw_result,
            system_prompt=str(request["system_prompt"]),
            output_schema=cast(Mapping[str, Any], request["output_schema"]),
            response_projection_valid=validation.valid,
            response_diagnostics=[
                item.as_dict() for item in validation.diagnostics
            ],
            metadata_validation=_metadata_validation(),
            request_text_redact=False,
            codex_profile={
                "model": request["model"],
                "effort": request["effort"],
                "concurrency": 1,
            },
            rpc=raw_result.get("rpc", []),
            events=raw_result.get("events", []),
            wire=raw_result.get(
                "wire",
                [
                    {"direction": "client_to_server", "method": "turn/start"},
                    {"direction": "server_to_client", "method": "turn/completed"},
                ],
            ),
            stdout=raw_result.get("stdout", []),
            stderr=raw_result.get("stderr", ""),
            request_idempotency_key=str(request["idempotency_key"]),
            provider_thread_id=(
                str(raw_result["provider_thread_id"])
                if raw_result.get("provider_thread_id") is not None
                else None
            ),
            provider_turn_id=(
                str(raw_result["provider_turn_id"])
                if raw_result.get("provider_turn_id") is not None
                else None
            ),
            terminal_status=str(raw_result.get("status", "completed")),
            request_accepted=bool(raw_result.get("accepted", True)),
            charged=bool(raw_result.get("charged")),
            uncharged=not bool(raw_result.get("charged")),
            content_received=bool(raw_result.get("response_text")),
            validation_completed=True,
            error=(
                str(raw_result["error"])
                if raw_result.get("error") is not None
                else None
            ),
        )
    store.verify_turn(turn)
    return manifest


def _verification_activity(
    evaluation: PythonSerialEpisodeResultV1,
) -> dict[str, JsonValue]:
    result = evaluation.scientific_result
    traces = [
        item
        for item in (
            result.initial_counterexample,
            *(step.counterexample for step in result.steps),
        )
        if item is not None
    ]
    payloads = [trace.as_dict() for trace in traces]
    return {
        "submissions": len(payloads),
        "records": cast(JsonValue, payloads),
        "verified": any(
            str(payload.get("decision", "")).lower() == "stop_verified"
            for payload in payloads
        ),
        "authority": "exact_verifier_only",
    }


def _outcome(evaluation: PythonSerialEpisodeResultV1) -> dict[str, JsonValue]:
    scientific = evaluation.scientific_result
    if scientific.failure is not None:
        return {
            "kind": "PROGRAM_FAILURE",
            "code": scientific.failure.code,
            "message": scientific.failure.message,
        }
    if not scientific.steps:
        return {"kind": "NO_INVOCATION"}
    step = scientific.steps[0]
    if step.rewrite is not None:
        return {
            "kind": "REWRITE_PLAN",
            "rewrite": {
                "removed_edges": [list(edge) for edge in step.rewrite.removed_edges],
                "added_edges": [list(edge) for edge in step.rewrite.added_edges],
                "operator_family": step.rewrite.operator_family,
                "metadata": dict(step.rewrite.metadata),
            },
            "accepted": step.accepted,
            "acceptance_proved": step.acceptance_proved,
        }
    return {
        "kind": "NO_PLAN",
        "reason": step.no_plan_reason,
        "accepted": False,
    }


def _report_failure(
    *,
    report_path: Path,
    status: str,
    attempts: Sequence[Mapping[str, Any]],
    provider_attempts: int,
    error: BaseException | None = None,
    artifact_recording_error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": M4_REPORT_SCHEMA_VERSION,
        "status": status,
        "provider_completed": status != "provider_error"
        and bool(attempts)
        and attempts[-1].get("status") == "completed"
        and attempts[-1].get("accepted") is True
        and attempts[-1].get("content") is True,
        "contract_valid": False,
        "sandbox_completed": False,
        "evaluation_completed": False,
        "scientific_result": False,
        "verification_completed": False,
        "model_turns": len(attempts),
        "provider_attempts": provider_attempts,
        "graph_score_attempts": 0,
        "dsl_runtime_used": False,
        "m5_features_used": False,
        "usage": [_usage(item) for item in attempts],
    }
    if error is not None:
        report.update(
            {
                "error_classification": _failure_classification(error),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    if artifact_recording_error is not None:
        report["artifact_recording_error"] = artifact_recording_error
    write_json(report_path, report, exclusive=True)
    return report


def _retain_provider_failure(
    *,
    store: TurnArtifactStore,
    request: Mapping[str, Any],
    error: BaseException,
) -> str | None:
    turn = Path(str(request["artifact_dir"]))
    if not turn.is_dir():
        return None
    try:
        store.record_existing_turn(
            turn,
            generation=0,
            slot="slot-00",
            phase=str(request["phase"]),
            request=request,
            result={
                "status": "failed",
                "accepted": False,
                "charged": False,
                "uncharged": True,
                "content": False,
                "error": f"{type(error).__name__}: {error}",
                "validation_completed": False,
            },
        )
    except Exception as artifact_error:
        return f"{type(artifact_error).__name__}: {artifact_error}"
    return None


def run_m4_single_root(
    provider: LocalCodexAppServerProvider,
    experiment_root: str | Path,
    *,
    backend_factory: Callable[[], GraphBackend],
    config: PythonSerialEpisodeConfigV1,
    runtime_limits: PolicyRuntimeLimitsV1 | None = None,
) -> dict[str, Any]:
    """Generate and evaluate exactly one ordinary-Python root program."""

    root = Path(experiment_root)
    output_root = root / "native-v3-python-output"
    root_output = output_root / "root-0000"
    report_path = root_output / "m4-report.json.gz"
    evaluation_path = root_output / "evaluation-result.json.gz"
    store = TurnArtifactStore(root / "artifacts")
    attempts: list[Mapping[str, Any]] = []
    provider_calls = 0
    request = build_m4_request(
        root,
        model=provider.model,
        effort=provider.effort,
    )
    started = time.monotonic()
    try:
        provider_calls += 1
        raw = dict(provider.generate(request))
        attempts.append(raw)
    except Exception as error:
        artifact_recording_error = _retain_provider_failure(
            store=store,
            request=request,
            error=error,
        )
        return _report_failure(
            report_path=report_path,
            status="provider_error",
            attempts=attempts,
            provider_attempts=provider_calls,
            error=error,
            artifact_recording_error=artifact_recording_error,
        )
    if (
        raw.get("status") != "completed"
        or raw.get("accepted") is not True
        or raw.get("content") is not True
    ):
        boundary_error = M4RootEvaluationError(
            "provider turn did not complete successfully"
        )
        artifact_recording_error = _retain_provider_failure(
            store=store,
            request=request,
            error=boundary_error,
        )
        return _report_failure(
            report_path=report_path,
            status="provider_error",
            attempts=attempts,
            provider_attempts=provider_calls,
            error=boundary_error,
            artifact_recording_error=artifact_recording_error,
        )

    response_text = raw.get("response_text")
    validation = validate_python_policy_response(
        response_text if isinstance(response_text, str) else ""
    )
    if not validation.valid:
        try:
            _write_or_index_turn(
                store=store,
                request=request,
                raw_result=raw,
                validation=validation,
                phase="initial",
                evaluation=None,
            )
        except Exception as artifact_error:
            return _report_failure(
                report_path=report_path,
                status="artifact_error",
                attempts=attempts,
                provider_attempts=provider_calls,
                error=artifact_error,
            )
        diagnostics = [
            item.as_dict()
            for item in validation.diagnostics[:MAX_REPAIR_DIAGNOSTICS]
        ]
        repair_request = build_m4_request(
            root,
            model=provider.model,
            effort=provider.effort,
            phase="repair-01",
            repair_attempt=1,
            diagnostics=diagnostics,
        )
        try:
            provider_calls += 1
            raw = dict(provider.repair(repair_request, diagnostics))
            attempts.append(raw)
        except Exception as error:
            artifact_recording_error = _retain_provider_failure(
                store=store,
                request=repair_request,
                error=error,
            )
            return _report_failure(
                report_path=report_path,
                status="provider_error",
                attempts=attempts,
                provider_attempts=provider_calls,
                error=error,
                artifact_recording_error=artifact_recording_error,
            )
        if (
            raw.get("status") != "completed"
            or raw.get("accepted") is not True
            or raw.get("content") is not True
        ):
            boundary_error = M4RootEvaluationError(
                "provider repair turn did not complete successfully"
            )
            artifact_recording_error = _retain_provider_failure(
                store=store,
                request=repair_request,
                error=boundary_error,
            )
            return _report_failure(
                report_path=report_path,
                status="provider_error",
                attempts=attempts,
                provider_attempts=provider_calls,
                error=boundary_error,
                artifact_recording_error=artifact_recording_error,
            )
        request = repair_request
        response_text = raw.get("response_text")
        validation = validate_python_policy_response(
            response_text if isinstance(response_text, str) else ""
        )
    if (
        not validation.valid
        or validation.response is None
        or validation.identity is None
        or validation.identity.program_hash is None
    ):
        try:
            _write_or_index_turn(
                store=store,
                request=request,
                raw_result=raw,
                validation=validation,
                phase=str(request["phase"]),
                evaluation=None,
            )
        except Exception as artifact_error:
            return _report_failure(
                report_path=report_path,
                status="artifact_error",
                attempts=attempts,
                provider_attempts=provider_calls,
                error=artifact_error,
            )
        return _report_failure(
            report_path=report_path,
            status="contract_invalid",
            attempts=attempts,
            provider_attempts=provider_calls,
        )

    source = normalize_source_newlines(validation.response.source)
    backend: GraphBackend | None = None
    evaluation: PythonSerialEpisodeResultV1 | None = None
    try:
        backend = backend_factory()
        scorer = _TrackingScorer(scorer_for_backend(backend))
        pipeline = CounterexamplePipeline(
            backend=backend,
            artifact_root=root_output,
        )
        evaluation = evaluate_serial_python_policy(
            backend=backend,
            scorer=scorer,
            source=source,
            config=config,
            runtime_limits=runtime_limits,
            counterexample_pipeline=_TrackingCounterexamplePipeline(pipeline),
            provenance_source_kind=M4_PROVENANCE_SOURCE_KIND,
        )
    except Exception as error:
        try:
            _write_or_index_turn(
                store=store,
                request=request,
                raw_result=raw,
                validation=validation,
                phase=str(request["phase"]),
                evaluation=None,
            )
        except Exception as artifact_error:
            return _report_failure(
                report_path=report_path,
                status="artifact_error",
                attempts=attempts,
                provider_attempts=provider_calls,
                error=artifact_error,
            )
        return _report_failure(
            report_path=report_path,
            status="evaluation_error",
            attempts=attempts,
            provider_attempts=provider_calls,
            error=error,
        )
    finally:
        if backend is not None:
            backend.close()

    turn = Path(str(request["artifact_dir"]))
    try:
        manifest = _write_or_index_turn(
            store=store,
            request=request,
            raw_result=raw,
            validation=validation,
            phase=str(request["phase"]),
            evaluation=evaluation,
        )
    except Exception as artifact_error:
        return _report_failure(
            report_path=report_path,
            status="artifact_error",
            attempts=attempts,
            provider_attempts=provider_calls,
            error=artifact_error,
        )
    source_sha256 = copy_canonical_source(
        turn,
        output_root,
        validation.identity.program_hash,
    )
    scientific = evaluation.scientific_result
    verifier = _verification_activity(evaluation)
    evaluation_payload: dict[str, Any] = {
        "schema_version": M4_EVALUATION_SCHEMA_VERSION,
        "slot": "slot-00",
        "root_index": 0,
        "source": source,
        "source_sha256": source_sha256,
        "program_identity": validation.identity.as_dict(),
        "behavior_identity": {
            "protocol_id": evaluation.behavior_identity.protocol_id,
            "probe_manifest_sha256": (
                evaluation.behavior_identity.probe_manifest_sha256
            ),
            "behavior_signature": evaluation.behavior_identity.behavior_signature,
        },
        "evaluation": evaluation.as_dict(include_external_activity=False),
        "outcome": _outcome(evaluation),
        "verification": verifier,
        "protocols": {
            "provider_projection": NATIVE_V3_PYTHON_POLICY_PROJECTION,
            "serial_evaluator": PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
            "dsl_runtime_used": False,
        },
        "external_activity": {
            "provider_turns": len(attempts),
            "model_turns": len(attempts),
            "app_server_calls": len(attempts),
        },
        "mutation_forge": git_state(PROJECT_ROOT),
    }
    write_json(evaluation_path, evaluation_payload, exclusive=True)
    hygiene = _prompt_hygiene(
        str(request["system_prompt"]),
        str(request["prompt"]),
    )
    turn_warnings = sum(
        _turn_warnings(Path(str(item_request["artifact_dir"])))
        for item_request in (
            build_m4_request(
                root,
                model=provider.model,
                effort=provider.effort,
                phase="initial",
            ),
            *(
                [request]
                if request["phase"] == "repair-01"
                else []
            ),
        )
    )
    if scientific.status is SerialEvaluationStatus.COMPLETE:
        report_status = "completed"
        scientific_result = True
        verification_completed = True
    elif scientific.status is SerialEvaluationStatus.PROGRAM_FAILURE:
        report_status = "program_failure"
        scientific_result = True
        verification_completed = True
    else:
        report_status = "scoring_inconclusive"
        scientific_result = False
        verification_completed = False
    report = {
        "schema_version": M4_REPORT_SCHEMA_VERSION,
        "status": report_status,
        "provider_completed": True,
        "contract_valid": True,
        "sandbox_completed": True,
        "evaluation_completed": True,
        "scientific_result": scientific_result,
        "verification_completed": verification_completed,
        "root_count": 1,
        "programs_per_turn": 1,
        "parent_count": 0,
        "lineage_count": 0,
        "generation_count": 0,
        "model_turns": len(attempts),
        "provider_attempts": provider_calls,
        "repair_attempts": len(attempts) - 1,
        "provider_duration_ms": sum(
            int(item.get("provider_duration_ms", 0) or 0) for item in attempts
        ),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "warnings": turn_warnings,
        "usage": [_usage(item) for item in attempts],
        "model": provider.model,
        "reasoning_effort": provider.effort,
        "prompt": {
            **hygiene,
            "output_schema_bytes": len(
                json.dumps(
                    request["output_schema"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        },
        "program_hash": validation.identity.program_hash,
        "source": source,
        "source_archive": str(
            output_root / "sources" / f"{validation.identity.program_hash}.py"
        ),
        "validation": validation.as_dict(),
        "worker_telemetry": evaluation.worker_telemetry,
        "outcome": _outcome(evaluation),
        "graph_score_attempts": scientific.score_attempts,
        "unique_graph_scores": scientific.unique_graph_scores,
        "fitness_interval": scientific.fitness_interval.as_dict(),
        "acceptance_reason": (
            "strict_interval_improvement"
            if scientific.accepted_rewrites
            else "no_proved_strict_improvement"
        ),
        "verification": verifier,
        "evaluation_result": str(evaluation_path),
        "provider_turn_directory": str(turn),
        "turn_artifact_complete": manifest["artifact_complete"],
        "dsl_runtime_used": False,
        "m5_features_used": False,
        "resumable": False,
    }
    write_json(report_path, report, exclusive=True)
    return report


__all__ = [
    "M4_EVALUATION_SCHEMA_VERSION",
    "M4_REPORT_SCHEMA_VERSION",
    "M4RootEvaluationError",
    "M4ScoringError",
    "M4VerificationError",
    "build_m4_request",
    "run_m4_single_root",
]
