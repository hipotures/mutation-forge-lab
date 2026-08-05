"""One-turn Native v3 smoke harness over the unchanged Native v2 provider."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutation_forge.experiment.artifacts import (
    TurnArtifactStore,
    generated_policy_diagnostics,
)
from mutation_forge.experiment.generation import GenerationRequest
from mutation_forge.experiment.json_io import write_json
from mutation_forge.experiment.provider import (
    AuthenticationError,
    LocalCodexAppServerProvider,
)

from .canonical import CanonicalJsonError, parse_strict_json
from .contracts import (
    VALIDATOR_PROTOCOL_ID,
    ValidatedProgram,
    validate_program,
    validated_program_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAXIMUM_RESPONSE_BYTES = 65_536
CAMPAIGN_ID = "native-v3-provider-smoke"


class NativeV3ProviderSmokeError(RuntimeError):
    """The provider returned no usable Native v3 program."""


def _load_text(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _load_schema() -> dict[str, Any]:
    value = json.loads(
        _load_text("configs/native/native-v3-provider-envelope.schema.json")
    )
    if not isinstance(value, dict):
        raise NativeV3ProviderSmokeError("Native v3 provider schema is not an object")
    return value


def build_request(
    experiment_root: str | Path,
    *,
    model: str,
    effort: str,
) -> dict[str, Any]:
    """Build the dedicated prompt/schema request accepted by the v2 provider."""

    root = Path(experiment_root)
    prompt = _load_text("prompts/native-v3/request.md")
    system_prompt = _load_text("prompts/native-v3/system.md")
    output_schema = _load_schema()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    identity = json.dumps(
        {
            "campaign_id": CAMPAIGN_ID,
            "generation": 0,
            "slot": "slot-00",
            "prompt_hash": prompt_hash,
            "model": model,
            "effort": effort,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    idempotency_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    request = GenerationRequest(
        campaign_id=CAMPAIGN_ID,
        generation=0,
        slot="slot-00",
        parent_id="native-v3-empty-parent",
        brief_id="native-v3-provider-smoke",
        prompt=prompt,
        prompt_hash=prompt_hash,
        idempotency_key=idempotency_key,
        model=model,
        effort=effort,
        system_prompt=system_prompt,
        output_schema=output_schema,
        max_repairs=0,
        remaining_repairs=0,
    ).as_dict()
    artifacts = root / "artifacts"
    turn = TurnArtifactStore(artifacts).turn_directory(0, "slot-00")
    request.update(
        {
            "artifact_dir": str(turn),
            "artifact_root": str(artifacts),
            "artifact_prefix": "slot-00",
            "reasoning_effort": effort,
        }
    )
    return request


def parse_provider_response(response_text: str) -> tuple[dict[str, Any], ValidatedProgram]:
    """Strictly project the v2 envelope and validate its Native v3 source."""

    try:
        value = parse_strict_json(response_text, maximum_bytes=MAXIMUM_RESPONSE_BYTES)
    except CanonicalJsonError as exc:
        raise NativeV3ProviderSmokeError(f"invalid provider JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise NativeV3ProviderSmokeError("provider response is not a JSON object")
    envelope = {str(key): item for key, item in value.items()}
    diagnostics = generated_policy_diagnostics(envelope)
    if diagnostics:
        summary = "; ".join(str(item["message"]) for item in diagnostics)
        raise NativeV3ProviderSmokeError(f"invalid v2 provider envelope: {summary}")
    source = envelope.get("source")
    if not isinstance(source, str):
        raise NativeV3ProviderSmokeError("provider envelope source is not a string")
    validation = validate_program(source)
    if validation.program is None:
        summary = "; ".join(
            f"{item.code} at {item.path}: {item.message}" for item in validation.diagnostics
        )
        raise NativeV3ProviderSmokeError(f"invalid Native v3 AST: {summary}")
    return envelope, validation.program


def _failure_classification(error: Exception) -> str:
    message = str(error).lower()
    if (
        isinstance(error, AuthenticationError)
        or "auth" in type(error).__name__.lower()
        or "auth" in message
        or "login" in message
    ):
        return "authentication"
    return "provider"


def _record_failure(
    store: TurnArtifactStore,
    turn: Path,
    request: Mapping[str, Any],
    error: Exception,
) -> str | None:
    if not turn.is_dir():
        return None
    result = {
        "status": "failed",
        "accepted": False,
        "charged": False,
        "uncharged": True,
        "content": False,
        "error": f"{type(error).__name__}: {error}",
        "validation_completed": False,
    }
    try:
        store.record_existing_turn(
            turn,
            generation=0,
            slot="slot-00",
            phase="initial",
            request=request,
            result=result,
        )
    except Exception as artifact_error:
        return f"{type(artifact_error).__name__}: {artifact_error}"
    return None


def run_provider_smoke(
    provider: LocalCodexAppServerProvider,
    experiment_root: str | Path,
) -> dict[str, Any]:
    """Run exactly one provider call and no graph evaluation."""

    root = Path(experiment_root)
    store = TurnArtifactStore(root / "artifacts")
    request = build_request(root, model=provider.model, effort=provider.effort)
    turn = Path(str(request["artifact_dir"]))
    report_path = root / "native-v3-output" / "provider-smoke-report.json.gz"
    try:
        raw_result = provider.generate(request)
        response_text = raw_result.get("response_text")
        if not isinstance(response_text, str):
            raise NativeV3ProviderSmokeError("provider result has no response_text")
        envelope, program = parse_provider_response(response_text)
        output_path = root / "native-v3-output" / "validated-program.json.gz"
        source = str(envelope["source"])
        identity = {
            "ast_node_count": program.node_count,
            "normalized_ast_sha256": program.program_hash,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "validator_version": VALIDATOR_PROTOCOL_ID,
        }
        result = {
            **dict(raw_result),
            "canonical_response": envelope,
            "identity": identity,
            "provenance": {
                "effort": raw_result.get("effort", provider.effort),
                "model": raw_result.get("model", provider.model),
                "provider_request_id": raw_result.get("provider_request_id"),
                "provider_thread_id": raw_result.get("provider_thread_id"),
                "provider_turn_id": raw_result.get("provider_turn_id"),
                "transport_sha256": raw_result.get("transport_sha256"),
            },
            "validation": {
                "errors": ["Native v2 Python validation is not applicable to a Native v3 AST"],
                "identity": identity,
                "valid": False,
            },
            "validation_completed": True,
            "uncharged": not bool(raw_result.get("charged")),
        }
        artifact_prefix = store.artifact_prefix(turn, result, "slot-00")
        usage_path = turn / f"{artifact_prefix}.usage.json.gz"
        usage = result.get("usage")
        if isinstance(usage, Mapping) and not usage_path.exists():
            write_json(usage_path, usage, exclusive=True)
        manifest = store.record_existing_turn(
            turn,
            generation=0,
            slot="slot-00",
            phase="initial",
            request=request,
            result=result,
        )
        store.verify_turn(turn)
        write_json(output_path, validated_program_artifact(program), exclusive=True)
        report: dict[str, Any] = {
            "schema_version": "mforge.native-v3-provider-smoke.v1",
            "status": "completed",
            "valid_ast": True,
            "program_hash": program.program_hash,
            "model_turns": 1,
            "graph_evaluations": 0,
            "provider_model": provider.model,
            "provider_effort": provider.effort,
            "provider_thread_id": raw_result.get("provider_thread_id"),
            "provider_turn_id": raw_result.get("provider_turn_id"),
            "usage": raw_result.get("usage"),
            "turn_artifact_complete": manifest["artifact_complete"],
            "turn_directory": str(turn),
            "validated_program": str(output_path),
            "resumable": True,
        }
    except Exception as error:
        artifact_recording_error = _record_failure(store, turn, request, error)
        report = {
            "schema_version": "mforge.native-v3-provider-smoke.v1",
            "status": "provider_error",
            "error_classification": _failure_classification(error),
            "error_type": type(error).__name__,
            "error": str(error),
            "valid_ast": False,
            "model_turns": 0,
            "graph_evaluations": 0,
            "provider_model": provider.model,
            "provider_effort": provider.effort,
            "turn_directory": str(turn),
            "resumable": True,
        }
        if artifact_recording_error is not None:
            report["artifact_recording_error"] = artifact_recording_error
    write_json(report_path, report)
    return report


__all__ = [
    "NativeV3ProviderSmokeError",
    "build_request",
    "parse_provider_response",
    "run_provider_smoke",
]
