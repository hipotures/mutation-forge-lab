"""Native experiment adapter and durable experiment-side provider boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from mutation_forge.proposals.k_switch import FeatureLimits
from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.policy import probe_scientific_policy
from mutation_forge.sandbox.validation import accessed_policy_fields, validate_policy

from .artifacts import (
    ArtifactIncompleteError,
    TurnArtifactStore,
    generated_policy_diagnostics,
    is_generated_policy,
)
from .config import ExperimentConfig
from .evaluation import DEFAULT_FORBIDDEN_LENGTHS, DEFAULT_WITNESS_CAP, evaluate_candidate
from .generation import Candidate, GenerationConfig, GenerationCoordinator, SlotResult
from .layout import ExperimentLayout, WorkspaceError
from .provider import LocalCodexAppServerProvider
from .sessions import SessionContext
from .state import ExperimentStateStore


class NativeExperimentError(RuntimeError):
    """A native experiment could not complete its current safe boundary."""


class _BehaviorProbeError(ValueError):
    def __init__(
        self,
        error_type: str,
        message: str,
        telemetry: Mapping[str, Any],
    ) -> None:
        self.error_type = error_type
        self.telemetry = dict(telemetry)
        super().__init__(f"{error_type}: {message}" if message else error_type)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _asset_path(name: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    assets = {
        "system": root / "prompts" / "native" / "system.md",
        "request": root / "prompts" / "native" / "request.md",
        "repair": root / "prompts" / "native" / "repair.md",
        "schema": root / "configs" / "native" / "generated-policy.schema.json",
        "context": root / "configs" / "schemas" / "stage2b-context.schema.json",
        "proposal": root / "configs" / "schemas" / "stage2b-proposal.schema.json",
        "semantic": root / "configs" / "stage3-field-semantics.v1.json",
        "baseline": root / "configs" / "native" / "baseline-rankers.json",
    }
    try:
        return assets[name]
    except KeyError as exc:
        raise NativeExperimentError(f"unknown native asset {name!r}") from exc


def _load_assets() -> tuple[
    str,
    dict[str, Any],
    str,
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    system = _asset_path("system").read_text(encoding="utf-8")
    request = _asset_path("request").read_text(encoding="utf-8")
    repair = _asset_path("repair").read_text(encoding="utf-8")
    parsed: dict[str, dict[str, Any]] = {}
    for name in ("schema", "context", "proposal", "semantic", "baseline"):
        value = json.loads(_asset_path(name).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise NativeExperimentError(f"native {name} asset must be an object")
        parsed[name] = dict(value)
    if not request.strip() or not repair.strip():
        raise NativeExperimentError("native request and repair prompts must be non-empty")
    return (
        system,
        parsed["schema"],
        request,
        repair,
        parsed["context"],
        parsed["proposal"],
        parsed["semantic"],
        parsed["baseline"],
    )


def _load_slot_briefs() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3] / "configs" / "stage3-slots"
    briefs: dict[str, str] = {}
    for index in range(8):
        slot = f"slot-{index:02d}"
        path = root / f"{slot}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise NativeExperimentError(f"cannot load native mutation brief {path}") from exc
        brief = value.get("brief") if isinstance(value, Mapping) else None
        if (
            not isinstance(brief, str)
            or not brief.strip()
            or value.get("slot_id") != slot
        ):
            raise NativeExperimentError(f"invalid native mutation brief {path}")
        briefs[slot] = brief.strip()
    return briefs


def _native_behavior(
    source: str, limits: SandboxLimits
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result = probe_scientific_policy(source, limits)
    if result.get("status") != "completed":
        telemetry = result.get("worker_telemetry")
        retained_telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
        failure = result.get("error")
        if isinstance(failure, Mapping):
            error_type = str(failure.get("error_type", failure.get("code", "BehaviorProbeError")))
            message = str(failure.get("message", ""))
        else:
            error_type = "BehaviorProbeError"
            message = str(result.get("status", "behavior probe failed"))
        raise _BehaviorProbeError(
            error_type,
            message,
            retained_telemetry,
        )
    signature = result.get("behavior_signature")
    telemetry = result.get("worker_telemetry")
    if not isinstance(signature, Mapping):
        raise ValueError("behavior probe did not produce a signature")
    return signature, telemetry if isinstance(telemetry, Mapping) else {}


class _NativeArchive:
    """Small append-only native archive used by the coordinator."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.programs = root / "programs"
        self.sources = root / "sources"
        self.programs.mkdir(parents=True, exist_ok=True)
        self.sources.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def records(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.programs.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(value, Mapping):
                rows.append(dict(value))
        return tuple(rows)

    def append(self, value: Mapping[str, Any]) -> dict[str, Any]:
        program_id = str(value.get("program_id", ""))
        if not program_id or any(char in program_id for char in "/\\\x00"):
            raise NativeExperimentError("native archive program_id is unsafe")
        source = value.get("source")
        if not isinstance(source, str):
            raise NativeExperimentError("native archive record has no source")
        source_sha = str(value.get("source_sha256") or hashlib.sha256(source.encode()).hexdigest())
        record = {
            "schema_version": "mforge.native.program.v1",
            **{str(key): item for key, item in value.items() if key != "source"},
            "program_id": program_id,
            "source_sha256": source_sha,
            "source_path": f"sources/{program_id}.py",
        }
        with self._lock:
            source_path = self.sources / f"{program_id}.py"
            record_path = self.programs / f"{program_id}.json"
            if source_path.exists() and source_path.read_text(encoding="utf-8") != source:
                raise NativeExperimentError(f"native archive source collision: {program_id}")
            if not source_path.exists():
                temporary = source_path.with_suffix(".py.tmp")
                temporary.write_text(source, encoding="utf-8")
                os.replace(temporary, source_path)
            if not record_path.exists():
                temporary = record_path.with_suffix(".json.tmp")
                temporary.write_bytes(_canonical(record) + b"\n")
                os.replace(temporary, record_path)
        return record

    def existing_sources(self) -> tuple[str, ...]:
        result: list[str] = []
        for path in sorted(self.sources.glob("*.py")):
            try:
                result.append(path.read_text(encoding="utf-8"))
            except OSError:
                continue
        return tuple(result)


class _NativeProvider:
    """Route provider evidence into the native workspace before ledger commit."""

    def __init__(
        self,
        provider: Any,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
        *,
        sandbox_limits: SandboxLimits,
    ) -> None:
        self.provider = provider
        self.layout = layout
        self.state = state
        self.session = session
        self.sandbox_limits = sandbox_limits
        self.turns = TurnArtifactStore(layout.artifacts)
        self._lock = threading.RLock()

    @staticmethod
    def _phase(request: Mapping[str, Any]) -> str:
        if str(request.get("phase", "initial")) == "repair":
            attempt = int(request.get("repair_attempt", 1))
            return f"repair-{attempt:02d}"
        return str(request.get("phase", "initial"))

    def _payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(request)
        generation = int(request.get("generation", 0))
        slot = str(request.get("slot", "slot-00"))
        phase = self._phase(request)
        directory = self.layout.generation_slot_phase(generation, slot, phase)
        value.update(
            {
                "artifact_dir": str(directory),
                "artifact_root": str(self.layout.artifacts),
                "artifact_prefix": slot,
            }
        )
        return value

    @staticmethod
    def _key(request: Mapping[str, Any]) -> str:
        value = request.get("idempotency_key", request.get("request_idempotency_key"))
        if isinstance(value, str) and value:
            return value
        return _sha256(
            {
                "generation": request.get("generation", 0),
                "slot": request.get("slot", "slot-00"),
                "phase": request.get("phase", "initial"),
            }
        )

    def _retained(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        key = self._key(request)
        row = self.state.provider_turn(key)
        directory = self.layout.generation_slot_phase(
            int(request.get("generation", 0)),
            str(request.get("slot", "slot-00")),
            self._phase(request),
        )
        manifest_path = directory / "turn-manifest.json"
        if row is None and not manifest_path.is_file():
            return None
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            slot = str(request.get("slot", "slot-00"))
            response_path = directory / f"{slot}.response.json"
            if not response_path.is_file():
                responses = sorted(directory.glob("*.response.json"))
                response_path = responses[-1] if responses else response_path
            raw_response_path = directory / f"{slot}.response.raw.txt"
            if not raw_response_path.is_file():
                raw_responses = sorted(directory.glob("*.response.raw.txt"))
                raw_response_path = raw_responses[-1] if raw_responses else raw_response_path
            usage_path = directory / "usage.json"
            if not usage_path.is_file():
                usages = sorted(directory.glob("*.usage.json"))
                usage_path = usages[-1] if usages else usage_path
            response = (
                json.loads(response_path.read_text(encoding="utf-8"))
                if response_path.is_file()
                else (
                    raw_response_path.read_text(encoding="utf-8")
                    if raw_response_path.is_file()
                    else {}
                )
            )
            usage = (
                json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.is_file() else {}
            )
            retained_evidence: dict[str, Any] = {}
            for key_name, filename in (
                ("validation", "validation.json"),
                ("identity", "identity.json"),
                ("behavior", "behavior.json"),
                ("worker_telemetry", "worker_telemetry.json"),
                ("canonical_response", "canonical_response.json"),
                ("metadata_validation", "metadata-validation.json"),
            ):
                evidence_path = directory / filename
                if evidence_path.is_file():
                    retained_evidence[key_name] = json.loads(
                        evidence_path.read_text(encoding="utf-8")
                    )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIncompleteError(
                "retained native provider evidence is unreadable"
            ) from exc
        if not isinstance(manifest, Mapping):
            return None
        usage_total = (
            usage.get("totalTokens") if isinstance(usage, Mapping) else None
        )
        retryable = manifest.get("artifact_complete") is not True or (
            str(manifest.get("terminal_status", "completed")) != "completed"
            and manifest.get("charged") is not True
            and not (
                isinstance(usage_total, int)
                and not isinstance(usage_total, bool)
                and usage_total > 0
            )
        )
        if retryable:
            self.turns.archive_retryable_manifest(directory)
            return None
        self.turns.verify_turn(directory)
        status = str(manifest.get("terminal_status", "completed"))
        return {
            "status": status,
            "accepted": bool(manifest.get("request_accepted", False)),
            "content": bool(manifest.get("content_received", False)),
            "charged": manifest.get("charged"),
            "uncharged": manifest.get("uncharged"),
            "response": response,
            "response_text": (
                raw_response_path.read_text(encoding="utf-8")
                if raw_response_path.is_file()
                else None
            ),
            "response_projection_valid": isinstance(response, Mapping)
            and is_generated_policy(response),
            "usage": usage if isinstance(usage, Mapping) else {},
            "provider_thread_id": manifest.get("provider_thread_id"),
            "provider_turn_id": manifest.get("provider_turn_id"),
            "retained": True,
            "error": manifest.get("error"),
            **retained_evidence,
        }

    def _evidence(self, result: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(result)
        response = value.get("response")
        response_text = value.get("response_text")
        if isinstance(response, (str, bytes, bytearray)):
            try:
                decoded = json.loads(
                    response.decode("utf-8")
                    if isinstance(response, (bytes, bytearray))
                    else response
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, Mapping):
                response = dict(decoded)
                value["response"] = response
        projection_value = response if response is not None else response_text
        projection_diagnostics = generated_policy_diagnostics(projection_value)
        value["response_projection_valid"] = is_generated_policy(projection_value)
        existing_diagnostics = value.get("response_diagnostics")
        if isinstance(existing_diagnostics, Sequence) and not isinstance(
            existing_diagnostics, (str, bytes, bytearray)
        ):
            projection_diagnostics = tuple(
                [dict(item) for item in existing_diagnostics if isinstance(item, Mapping)]
                or projection_diagnostics
            )
        value["response_diagnostics"] = [dict(item) for item in projection_diagnostics]
        source = response.get("source") if isinstance(response, Mapping) else None
        if not isinstance(value.get("response_text"), str) and response is not None:
            value["response_text"] = (
                response
                if isinstance(response, str)
                else json.dumps(response, sort_keys=True, separators=(",", ":"))
            )
        if isinstance(source, str):
            validation = validate_policy(source, self.sandbox_limits)
            value["validation"] = validation.as_dict()
            value["identity"] = validation.identity.as_dict()
            value["validation_completed"] = True
            canonical_response = dict(cast(Mapping[str, Any], response))
            declared_fields = cast(Mapping[str, Any], response).get("used_fields")
            declared = (
                [str(item) for item in declared_fields if isinstance(item, str)]
                if isinstance(declared_fields, list)
                else []
            )
            extracted = list(accessed_policy_fields(source)) if validation.valid else []
            matches = (
                validation.valid
                and len(declared) == len(set(declared))
                and sorted(declared) == extracted
            )
            value["metadata_validation"] = {
                "schema_version": "mforge.native.metadata-validation.v1",
                "status": (
                    "matched"
                    if matches
                    else "mismatch"
                    if validation.valid
                    else "not_validated"
                ),
                "declared_used_fields": declared,
                "extracted_used_fields": extracted,
                "errors": (
                    []
                    if matches or not validation.valid
                    else [
                        {
                            "code": "used_fields_mismatch",
                            "message": (
                                "declared used_fields do not match fields extracted "
                                "from validated source"
                            ),
                        }
                    ]
                ),
            }
            if validation.valid:
                canonical_response["used_fields"] = extracted
            if validation.valid:
                try:
                    behavior, telemetry = _native_behavior(source, self.sandbox_limits)
                except Exception as error:
                    value["behavior"] = {
                        "status": "failed",
                        "error_type": getattr(error, "error_type", type(error).__name__),
                        "error": str(error),
                    }
                    retained_telemetry = getattr(error, "telemetry", {})
                    value["worker_telemetry"] = (
                        dict(retained_telemetry)
                        if isinstance(retained_telemetry, Mapping)
                        else {}
                    )
                else:
                    value["behavior"] = behavior
                    value["worker_telemetry"] = telemetry
            value["canonical_response"] = canonical_response
        provenance = value.get("provenance")
        value["provenance"] = {
            **(dict(provenance) if isinstance(provenance, Mapping) else {}),
            "provider_request_id": value.get("provider_request_id", value.get("request_id")),
            "provider_thread_id": value.get("provider_thread_id", value.get("thread_id")),
            "provider_turn_id": value.get("provider_turn_id", value.get("turn_id")),
            "model": value.get("model"),
            "effort": value.get("effort"),
            "transport_sha256": value.get("transport_sha256"),
        }
        return value

    def _record(self, request: Mapping[str, Any], result: Mapping[str, Any]) -> Mapping[str, Any]:
        value = self._evidence(result)
        generation = int(request.get("generation", 0))
        slot = str(request.get("slot", "slot-00"))
        phase = self._phase(request)
        directory = self.layout.generation_slot_phase(generation, slot, phase)
        usage = value.get("usage") if isinstance(value.get("usage"), Mapping) else {}
        status = str(value.get("status", "completed"))
        manifest_path = directory / "turn-manifest.json"
        retained_manifest: Any = None
        if manifest_path.is_file():
            retained_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if directory.exists() and (
            not isinstance(retained_manifest, Mapping)
            or retained_manifest.get("artifact_complete") is not True
        ):
            artifact_prefix = self.turns.artifact_prefix(directory, value, slot)
            usage_path = directory / f"{artifact_prefix}.usage.json"
            if not usage_path.is_file() and usage:
                descriptor = os.open(
                    usage_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    payload = _canonical(usage) + b"\n"
                    os.write(descriptor, payload)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        if not directory.exists():
            self.turns.write_turn(
                generation=generation,
                slot=slot,
                phase=phase,
                request=request,
                request_text=str(request.get("prompt", "")),
                request_text_redact=False,
                response=value.get("response"),
                response_text=value.get("response_text"),
                source=(value.get("response") or {}).get("source")
                if isinstance(value.get("response"), Mapping)
                else None,
                usage=cast(Mapping[str, Any], usage),
                identity=cast(Mapping[str, Any] | None, value.get("identity")),
                behavior=cast(Mapping[str, Any] | None, value.get("behavior")),
                provenance=cast(Mapping[str, Any] | None, value.get("provenance")),
                validation=cast(Mapping[str, Any] | None, value.get("validation")),
                worker_telemetry=cast(Mapping[str, Any] | None, value.get("worker_telemetry")),
                canonical_response=cast(Mapping[str, Any] | None, value.get("canonical_response")),
                provider_raw=value,
                system_prompt=str(request.get("system_prompt", ""))
                if request.get("system_prompt") is not None
                else None,
                output_schema=cast(Mapping[str, Any] | None, request.get("output_schema")),
                response_projection_valid=(
                    bool(value.get("response_projection_valid"))
                    if isinstance(value.get("response_projection_valid"), bool)
                    else None
                ),
                response_diagnostics=cast(
                    Sequence[Mapping[str, Any]] | None, value.get("response_diagnostics")
                ),
                transport_diagnostics=cast(
                    Sequence[Mapping[str, Any]] | None,
                    value.get("transport_diagnostics"),
                ),
                metadata_validation=cast(
                    Mapping[str, Any] | None,
                    value.get("metadata_validation"),
                ),
                codex_profile={"model": request.get("model"), "effort": request.get("effort")},
                rpc=value.get("rpc", []),
                events=value.get("events", []),
                wire=value.get("wire", []),
                stdout=value.get("stdout", []),
                stderr=value.get("stderr", ""),
                request_idempotency_key=self._key(request),
                provider_thread_id=str(value.get("provider_thread_id", value.get("thread_id")))
                if value.get("provider_thread_id", value.get("thread_id")) is not None
                else None,
                provider_turn_id=str(value.get("provider_turn_id", value.get("turn_id")))
                if value.get("provider_turn_id", value.get("turn_id")) is not None
                else None,
                terminal_status=status,
                request_accepted=bool(value.get("accepted", value.get("accepted_turn", False))),
                charged=value.get("charged") if isinstance(value.get("charged"), bool) else None,
                uncharged=value.get("uncharged")
                if isinstance(value.get("uncharged"), bool)
                else None,
                content_received=bool(value.get("content", value.get("response_text"))),
                validation_completed=bool(value.get("validation_completed", False)),
                error=str(value.get("error")) if value.get("error") else None,
            )
        elif isinstance(retained_manifest, Mapping):
            if (
                retained_manifest.get("artifact_complete") is True
            ):
                self.turns.verify_turn(directory)
            else:
                self.turns.record_existing_turn(
                    directory,
                    generation=generation,
                    slot=slot,
                    phase=phase,
                    request=request,
                    result=value,
                )
                self.turns.verify_turn(directory)
        else:
            self.turns.record_existing_turn(
                directory,
                generation=generation,
                slot=slot,
                phase=phase,
                request=request,
                result=value,
            )
            self.turns.verify_turn(directory)
        # The transport logger may flush stream files after the session
        # observer has indexed the workspace. Refresh the experiment manifest
        # after the complete turn artifact is durable so continuation does not
        # reject a valid turn for a stale stream digest.
        self.layout.write_artifact_manifest()
        with self._lock:
            self.state.record_provider_turn(
                idempotency_key=self._key(request),
                generation=generation,
                slot=slot,
                phase=phase,
                state="completed" if status == "completed" else "failed",
                artifact_path=str(directory),
                usage=cast(Mapping[str, Any], usage),
                provider_thread_id=str(value.get("provider_thread_id", value.get("thread_id")))
                if value.get("provider_thread_id", value.get("thread_id")) is not None
                else None,
                provider_turn_id=str(value.get("provider_turn_id", value.get("turn_id")))
                if value.get("provider_turn_id", value.get("turn_id")) is not None
                else None,
                error=str(value.get("error")) if value.get("error") else None,
            )
        return value

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        retained = self._retained(request)
        if retained is not None:
            return retained
        if self.session.budget_exhausted():
            raise KeyboardInterrupt
        payload = self._payload(request)
        try:
            value = self.provider.generate(payload)
            provider_result = value if isinstance(value, Mapping) else {"response": value}
            return self._record(request, cast(Mapping[str, Any], provider_result))
        except KeyboardInterrupt:
            raise
        except BaseException as error:
            evidence = getattr(error, "evidence", {})
            failure_result: dict[str, Any] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            if isinstance(evidence, Mapping):
                failure_result.update(cast(Mapping[str, Any], evidence))
            try:
                self._record(request, failure_result)
            except ArtifactIncompleteError as artifact_error:
                error.add_note(f"artifact retention also failed: {artifact_error}")
            raise

    def repair(
        self, request: Mapping[str, Any], diagnostics: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        retained = self._retained(request)
        if retained is not None:
            return retained
        if self.session.budget_exhausted():
            raise KeyboardInterrupt
        payload = self._payload(request)
        payload["diagnostics"] = [dict(item) for item in diagnostics]
        try:
            repair = getattr(self.provider, "repair", None)
            value = (
                repair(payload, diagnostics)
                if callable(repair)
                else self.provider.generate(payload)
            )
            result = value if isinstance(value, Mapping) else {"response": value}
            return self._record(request, cast(Mapping[str, Any], result))
        except KeyboardInterrupt:
            raise
        except BaseException as error:
            evidence = getattr(error, "evidence", {})
            result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            if isinstance(evidence, Mapping):
                result.update(dict(evidence))
            try:
                self._record(request, result)
            except ArtifactIncompleteError as artifact_error:
                error.add_note(f"artifact retention also failed: {artifact_error}")
            raise

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


class NativeExperimentAdapter:
    """The default production adapter for ``mforge experiment``."""

    def __init__(
        self,
        *,
        provider: Any | None = None,
        engine: Any | None = None,
        evaluator: Any | None = None,
        backend: Any | None = None,
    ) -> None:
        self.provider = provider
        self.engine = engine
        self.evaluator = evaluator
        self.backend = backend

    def preflight(self, config: ExperimentConfig) -> Mapping[str, Any]:
        system, schema, request, repair, context, proposal, semantic, baseline = _load_assets()
        if config.kind != "ranker-search":
            raise WorkspaceError(f"unsupported experiment kind {config.kind!r}")
        if config.preset != "heg-ranker-evolution-v1":
            raise WorkspaceError(f"unsupported native experiment preset {config.preset!r}")
        if config.model.provider != "codex":
            raise WorkspaceError(
                f"unsupported model.provider {config.model.provider!r}; native provider is codex"
            )
        if config.model.effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
            raise WorkspaceError(f"unsupported model.effort {config.model.effort!r}")
        if config.search.selection != "elite-diversity":
            raise WorkspaceError(f"unsupported native selection policy {config.search.selection!r}")
        unsupported_baselines = set(config.evaluation.baselines).difference(
            {"random", "structural"}
        )
        if unsupported_baselines:
            raise WorkspaceError(
                f"unsupported native baseline policies: {sorted(unsupported_baselines)!r}"
            )
        if (
            not system.strip()
            or not schema
            or not context
            or not proposal
            or not semantic
            or not baseline
        ):
            raise WorkspaceError("native preset assets are empty")
        heg = Path(__file__).resolve().parents[4] / "heg"
        if not (heg / "src").is_dir():
            raise WorkspaceError(f"HEG backend is unavailable at {heg}")
        try:
            from mutation_forge.backends.heg import HegBackend

            backend = HegBackend(heg)
            backend.close()
        except Exception as error:
            raise WorkspaceError(f"HEG backend is not functional at {heg}: {error}") from error
        return {
            "native_assets": {
                "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
                "request_prompt_sha256": hashlib.sha256(request.encode()).hexdigest(),
                "repair_prompt_sha256": hashlib.sha256(repair.encode()).hexdigest(),
                "output_schema_sha256": _sha256(schema),
                "context_schema_sha256": _sha256(context),
                "proposal_schema_sha256": _sha256(proposal),
                "semantic_descriptions_sha256": _sha256(semantic),
                "baseline_rankers_sha256": _sha256(baseline),
                "mutation_briefs_sha256": _sha256(_load_slot_briefs()),
            },
            "heg": {"repo": str(heg), "backend": "generic"},
        }

    @staticmethod
    def _parent_data(
        state: ExperimentStateStore, layout: ExperimentLayout
    ) -> tuple[dict[str, str], dict[str, Any]]:
        sources: dict[str, str] = {}
        records: dict[str, Any] = {}
        rows = state.connection.execute(
            "SELECT candidate_id,archive_path,metadata_json FROM candidates "
            "WHERE status NOT IN ('duplicate','invalid') ORDER BY generation DESC, candidate_id"
        ).fetchall()
        for row in rows:
            candidate_id = str(row["candidate_id"])
            archive_path = row["archive_path"]
            source_path = (
                Path(str(archive_path))
                if archive_path
                else layout.archive / "sources" / f"{candidate_id}.py"
            )
            try:
                source = source_path.read_text(encoding="utf-8")
            except OSError:
                continue
            sources[candidate_id] = source
            try:
                metadata = json.loads(str(row["metadata_json"]))
            except json.JSONDecodeError:
                metadata = {}
            records[candidate_id] = {
                "source": source,
                **(metadata if isinstance(metadata, Mapping) else {}),
            }
        return sources, records

    @staticmethod
    def _select_parents(
        generation: int,
        candidates: Sequence[Candidate],
        slots: int,
        selection: str = "elite-diversity",
        _results: Sequence[SlotResult] = (),
    ) -> Mapping[str, str]:
        del generation, _results
        if selection != "elite-diversity":
            raise NativeExperimentError(f"unsupported native selection policy {selection!r}")
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item.behavior_signature.get("score", 0.0))
                if isinstance(item.behavior_signature, Mapping)
                else 0.0,
                item.normalized_ast_sha256,
            ),
        )
        if not ordered:
            return {f"slot-{index:02d}": "native-baseline" for index in range(slots)}
        selected: list[Candidate] = [ordered[0]]
        while len(selected) < min(slots, len(ordered)):
            remaining = [candidate for candidate in ordered if candidate not in selected]
            selected.append(
                max(
                    remaining,
                    key=lambda candidate: (
                        min(
                            sum(
                                left != right
                                for left, right in zip(
                                    candidate.normalized_ast_sha256,
                                    parent.normalized_ast_sha256,
                                    strict=True,
                                )
                            )
                            for parent in selected
                        ),
                        candidate.behavior_signature.get("score", 0.0)
                        if isinstance(candidate.behavior_signature, Mapping)
                        else 0.0,
                        candidate.normalized_ast_sha256,
                    ),
                )
            )
        selected = (selected * ((slots + len(selected) - 1) // len(selected)))[:slots]
        return {
            f"slot-{index:02d}": f"g{candidate.generation:04d}-{candidate.slot}"
            for index, candidate in enumerate(selected)
        }

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
        *,
        observer: Any | None = None,
        event_callback: Any | None = None,
        profiling: bool | None = None,
    ) -> Mapping[str, Any]:
        (
            system_prompt,
            output_schema,
            request_prompt,
            repair_prompt,
            context_schema,
            proposal_schema,
            semantic_descriptions,
            baseline_rankers,
        ) = _load_assets()
        slot_briefs = _load_slot_briefs()
        auth_path = Path.home() / ".codex" / "auth.json"
        provider = self.provider or LocalCodexAppServerProvider(
            model=config.model.name,
            effort=config.model.effort,
            concurrency=config.model.concurrency,
            max_repairs=config.model.max_repairs,
            turn_timeout_base_seconds=config.run.turn_timeout_base_seconds,
            auth_json=auth_path if auth_path.is_file() else None,
            persist_artifacts=False,
        )
        wrapped = _NativeProvider(
            provider,
            layout,
            state,
            session,
            sandbox_limits=SandboxLimits(),
        )
        archive = _NativeArchive(layout.archive)
        parent_sources, parent_records = self._parent_data(state, layout)
        baseline_sources = archive.existing_sources()
        callback = observer if observer is not None else event_callback
        profiling_enabled = (
            config.run.profiling_enabled if profiling is None else bool(profiling)
        )

        def emit(event_type: str, **payload: Any) -> None:
            if not callable(callback):
                return
            try:
                callback(event_type, payload)
            except TypeError:
                try:
                    callback(event_type, **payload)
                except Exception:
                    return
            except Exception:
                return

        def pretty(value: object) -> str:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)

        def fenced(value: object, language: str = "json") -> str:
            rendered = pretty(value) if not isinstance(value, str) else value.rstrip()
            return f"```{language}\n{rendered}\n```"

        def render_prompt(**values: Any) -> str:
            generation = int(values.get("generation", 0))
            slot = str(values.get("slot", "slot-00"))
            phase = str(values.get("phase", "initial"))
            brief = str(values.get("brief", "")).strip()
            if not brief or brief.startswith("mutation brief generation"):
                brief = (
                    "Generate one deterministic ranker for this native mutation search. "
                    "The host supplies legal proposals and performs authoritative scoring; "
                    "the policy only ranks the supplied objects."
                )
            parent_source = str(values.get("parent_source", "")).strip()
            parent_metadata = values.get("parent_metadata", {})
            parent_metadata_view = (
                {
                    str(key): item
                    for key, item in parent_metadata.items()
                    if key not in {"source", "parent_id", "program_id", "candidate_id"}
                }
                if isinstance(parent_metadata, Mapping)
                else {}
            )
            feedback = str(values.get("search_feedback", "")).strip()
            archive_context = str(values.get("archive_context", "")).strip()
            config_view = {
                "experiment_id": config.exp_id,
                "generation": generation,
                "slot": slot,
                "search": {
                    "population_size": config.search.population_size,
                    "max_generations": config.search.max_generations,
                    "max_model_turns": config.search.max_model_turns,
                    "selection": config.search.selection,
                },
                "evaluation": {
                    "orders": list(config.evaluation.orders),
                    "graph_seeds": list(config.evaluation.graph_seeds),
                    "policy_seeds": list(config.evaluation.policy_seeds),
                    "horizon": config.evaluation.horizon,
                    "proposal_pool_size": config.evaluation.proposal_pool_size,
                    "baselines": list(config.evaluation.baselines),
                    "replay": config.evaluation.replay,
                },
                "resources": {
                    "workers": config.resources.workers,
                    "thread_count": config.resources.thread_count,
                },
            }
            feature_limits = FeatureLimits()
            numeric_scales = {
                "ctx.order": {
                    "values_in_this_experiment": list(config.evaluation.orders),
                },
                "ctx.forbidden_lengths": {
                    "values": list(DEFAULT_FORBIDDEN_LENGTHS),
                },
                "ctx.capped_cycle_counts": {
                    "per_item_minimum": 0,
                    "per_item_maximum": DEFAULT_WITNESS_CAP,
                },
                "ctx.weighted_penalty": {
                    "minimum": 0,
                    "note": "pool-constant current-state score; not a candidate signal",
                },
                "ctx.step": {
                    "minimum": 0,
                    "maximum": config.evaluation.horizon - 1,
                },
                "ctx.remaining_steps": {
                    "minimum": 0,
                    "maximum": config.evaluation.horizon - 1,
                },
                "ctx.stagnation": {
                    "minimum": 0,
                    "maximum": config.evaluation.horizon,
                },
                "ctx.recent_best_improvement": {
                    "minimum": 0,
                    "maximum": 1,
                },
                "ctx.recent_acceptance_rate": {"minimum": 0, "maximum": 1},
                "ctx.recent_duplicate_rate": {"minimum": 0, "maximum": 1},
                "proposal.k": {"values": [2, 3, 4]},
                "proposal.broken_sampled_witnesses_by_length": {
                    "per_item_minimum": 0,
                    "per_item_maximum": feature_limits.witness_sample_cap,
                },
                "proposal.removed_edge_load_sum_by_length": {
                    "per_item_minimum": 0,
                    "per_item_maximum": (
                        feature_limits.witness_sample_cap * 4
                    ),
                },
                "proposal.removed_edge_load_max_by_length": {
                    "per_item_minimum": 0,
                    "per_item_maximum": feature_limits.witness_sample_cap,
                },
                "proposal.distance_fields": {
                    "minimum": 0,
                    "maximum": "ctx.order (sentinel when the distance budget is exhausted)",
                },
                "proposal.local_triangle_risk": {
                    "minimum": 0,
                    "operation_budget": feature_limits.local_risk_budget,
                },
                "proposal.local_c4_risk": {
                    "minimum": 0,
                    "operation_budget": feature_limits.local_risk_budget,
                },
            }
            sections = [
                "# Mutation Forge native ranker task",
                "",
                "## Objective",
                "",
                request_prompt.strip(),
                "",
                "## Mutation brief",
                "",
                brief,
                "",
                "## Host boundary",
                "",
                "The host supplies the concrete `ctx` and legal `proposal` objects at evaluation "
                "time. It owns legality, authoritative scoring, and verification. Do not invent "
                "proposals, post-rewrite scores, proposal IDs, hidden state, or unavailable "
                "fields.",
                "",
                "## Parent policy",
                "",
                (
                    fenced(parent_source, "python")
                    if parent_source
                    else "This is root generation; no parent policy is available."
                ),
                "",
                "## Parent evaluation metadata",
                "",
                (
                    fenced(parent_metadata_view)
                    if parent_metadata_view
                    else "No parent evaluation metadata is available."
                ),
                "",
                "## Search feedback",
                "",
                feedback or "No prior search feedback is available.",
                "",
                "## Archive context",
                "",
                archive_context or "No prior archive context is available.",
                "",
                "## Experiment configuration",
                "",
                fenced(config_view),
                "",
                "## Context contract",
                "",
                "The host context follows this schema:",
                "",
                fenced(context_schema),
                "",
                "## Proposal contract",
                "",
                "The host proposal follows this schema; all proposals reaching the ranker are "
                "legal:",
                "",
                fenced(proposal_schema),
                "",
                "## Field semantics",
                "",
                fenced(semantic_descriptions),
                "",
                "## Numeric scales and bounds",
                "",
                fenced(numeric_scales),
                "",
                "## Baseline rankers",
                "",
                fenced(baseline_rankers),
                "",
                "## Output contract",
                "",
                "Return exactly one JSON object and no prose outside it. "
                "The object must match this schema:",
                "",
                fenced(output_schema),
            ]
            if phase == "repair":
                diagnostics = values.get("diagnostics", ())
                repair_source = str(values.get("repair_source", ""))
                repair_attempt = int(values.get("repair_attempt", 1))
                max_repairs = int(values.get("max_repairs", config.model.max_repairs))
                remaining_repairs = int(values.get("remaining_repairs", 0))
                sections.extend(
                    [
                        "",
                        "## Repair instructions",
                        "",
                        repair_prompt.strip(),
                        "",
                        f"Repair attempt {repair_attempt} of {max_repairs}; "
                        f"{remaining_repairs} repairs remain after this attempt.",
                        "",
                        "## Previous response/source",
                        "",
                        fenced(repair_source, "python"),
                        "",
                        "## Validation diagnostics",
                        "",
                        fenced([dict(item) for item in diagnostics], "json"),
                    ]
                )
            return "\n".join(sections).rstrip() + "\n"

        best_objective: float | None = None
        best_candidate_id: str | None = None
        last_ir: float | None = None
        last_timing_profile: Mapping[str, Any] | None = None

        def stored_evaluation_summary(candidate_id: str) -> dict[str, Any]:
            row = state.evaluation(f"{candidate_id}:development")
            raw_result = row.get("result_json") if isinstance(row, Mapping) else None
            if not isinstance(raw_result, str):
                return {}
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                return {}
            summary = result.get("summary") if isinstance(result, Mapping) else None
            return dict(summary) if isinstance(summary, Mapping) else {}

        def render_search_feedback(
            _generation: int,
            _slot: str,
            parent_id: str,
        ) -> str:
            summary = stored_evaluation_summary(parent_id)
            if not summary:
                return ""
            return pretty(
                {
                    "parent_id": parent_id,
                    "mean_auc": summary.get("mean_auc"),
                    "best_auc": summary.get("best_auc"),
                    "baseline_auc": summary.get("baseline_auc", {}),
                    "instruction": (
                        "Keep evidence-backed strengths, change the assigned brief's "
                        "mechanism, and avoid repeating recorded failure modes."
                    ),
                }
            )

        def render_archive_context(
            _generation: int,
            _slot: str,
            _parent_id: str,
        ) -> str:
            rows = state.connection.execute(
                "SELECT candidate_id FROM evaluations "
                "WHERE state='completed' ORDER BY completed_at, candidate_id"
            ).fetchall()
            summaries = [
                {
                    "candidate_id": str(row["candidate_id"]),
                    **stored_evaluation_summary(str(row["candidate_id"])),
                }
                for row in rows
                if row["candidate_id"]
            ]
            summaries.sort(
                key=lambda item: (
                    -(
                        float(item["mean_auc"])
                        if isinstance(item.get("mean_auc"), (int, float))
                        and not isinstance(item.get("mean_auc"), bool)
                        else 0.0
                    ),
                    str(item["candidate_id"]),
                )
            )
            return (
                pretty({"evaluated_candidates": summaries[:8]})
                if summaries
                else ""
            )

        def on_generation(
            generation: int, candidates: Sequence[Candidate], results: Sequence[SlotResult]
        ) -> Mapping[str, str]:
            nonlocal best_objective, best_candidate_id, last_ir, last_timing_profile
            if session.budget_exhausted():
                raise KeyboardInterrupt
            selection_candidates: list[Candidate] = []
            for candidate in candidates:
                if session.budget_exhausted():
                    raise KeyboardInterrupt
                program_id = f"g{candidate.generation:04d}-{candidate.slot}"
                archive.append(
                    {
                        "program_id": program_id,
                        "source": candidate.source,
                        "source_sha256": candidate.source_sha256,
                        "normalized_ast_sha256": candidate.normalized_ast_sha256,
                        "generation": candidate.generation,
                        "slot": candidate.slot,
                        "parent_id": candidate.parent_id,
                        "validation_status": "valid",
                        "probe_status": "passed",
                        "usage": dict(candidate.usage),
                        "behavior": dict(candidate.behavior_signature),
                    }
                )
                state.record_candidate(
                    program_id,
                    source_sha256=candidate.source_sha256,
                    archive_path=str(archive.sources / f"{program_id}.py"),
                    generation=candidate.generation,
                    slot=candidate.slot,
                    status="created",
                    metadata={"behavior": dict(candidate.behavior_signature), "search_metrics": {}},
                )
                emit(
                    "candidate_archived",
                    generation=generation,
                    slot=candidate.slot,
                    candidate_id=program_id,
                    status="accepted",
                    archive_size=len(archive.records()),
                    source_sha256=candidate.source_sha256,
                    normalized_ast_sha256=candidate.normalized_ast_sha256,
                    source_lines=len(candidate.source.splitlines()),
                )
                identity = f"{program_id}:development"
                if state.evaluation(identity) is None:
                    evaluator = self.evaluator or evaluate_candidate
                    emit(
                        "evaluation_started",
                        generation=generation,
                        slot=candidate.slot,
                        candidate_id=program_id,
                        phase="development",
                        evaluation_id=identity,
                        evaluations_queued=state.counts().get("evaluation_count", 0) + 1,
                        evaluation_total=(
                            len(config.evaluation.orders)
                            * len(config.evaluation.graph_seeds)
                            * len(config.evaluation.policy_seeds)
                        ),
                        development_progress=0.0,
                        replay_progress=0.0,
                        worker_count=config.resources.workers,
                        active_workers=1,
                    )
                    evaluation_started = time.monotonic()

                    def evaluation_progress(
                        payload: Mapping[str, Any],
                        *,
                        _candidate_id: str = program_id,
                        _generation: int = generation,
                        _slot: str = candidate.slot,
                        _identity: str = identity,
                    ) -> None:
                        progress_payload = dict(payload)
                        progress_payload.setdefault("candidate_id", _candidate_id)
                        progress_payload.setdefault("generation", _generation)
                        progress_payload.setdefault("slot", _slot)
                        pass_name = progress_payload.get("pass")
                        progress_payload.setdefault(
                            "phase", pass_name if isinstance(pass_name, str) else "development"
                        )
                        progress_payload.setdefault("evaluation_id", _identity)
                        progress_payload.setdefault("evaluations_active", 1)
                        emit(
                            "evaluation_progress",
                            **progress_payload,
                        )

                    evaluator_kwargs: dict[str, Any] = {
                        "artifact_root": layout.artifacts,
                        "backend": self.backend,
                        "sandbox_limits": SandboxLimits(),
                    }
                    try:
                        parameters = dict(inspect.signature(evaluator).parameters)
                    except (TypeError, ValueError):
                        parameters = {}
                    if "progress" in parameters:
                        evaluator_kwargs["progress"] = evaluation_progress
                    if "profiling_enabled" in parameters:
                        evaluator_kwargs["profiling_enabled"] = profiling_enabled
                    try:
                        result = evaluator(config, program_id, candidate.source, **evaluator_kwargs)
                    except BaseException as error:
                        emit(
                            "evaluation_failed",
                            generation=generation,
                            slot=candidate.slot,
                            candidate_id=program_id,
                            phase="development",
                            evaluation_id=identity,
                            status="failed",
                            evaluations_active=0,
                            error=f"{type(error).__name__}: {error}",
                            elapsed_seconds=time.monotonic() - evaluation_started,
                        )
                        raise
                    state.record_evaluation(
                        identity,
                        candidate_id=program_id,
                        kind="development",
                        state="completed",
                        result=cast(Mapping[str, Any], result),
                    )
                    session.evaluations_completed += 1
                    summary = result.get("summary") if isinstance(result, Mapping) else None
                    metric = summary.get("mean_auc") if isinstance(summary, Mapping) else None
                    if (
                        isinstance(metric, (int, float))
                        and not isinstance(metric, bool)
                        and (best_objective is None or float(metric) > best_objective)
                    ):
                        best_objective = float(metric)
                        best_candidate_id = program_id
                    timing_profile = (
                        result.get("timing_profile") if isinstance(result, Mapping) else None
                    )
                    if isinstance(timing_profile, Mapping):
                        last_timing_profile = dict(timing_profile)
                    raw_ir = (
                        result.get("ir")
                        if isinstance(result, Mapping)
                        else None
                    )
                    if raw_ir is None and isinstance(summary, Mapping):
                        raw_ir = summary.get("ir", summary.get("improvement_rate"))
                    if isinstance(raw_ir, (int, float)) and not isinstance(raw_ir, bool):
                        last_ir = float(raw_ir)
                    replay = result.get("replay") if isinstance(result, Mapping) else None
                    baseline_comparison = (
                        summary.get("baseline_auc") if isinstance(summary, Mapping) else None
                    )
                    emit(
                        "evaluation_completed",
                        generation=generation,
                        slot=candidate.slot,
                        candidate_id=program_id,
                        phase="development",
                        evaluation_id=identity,
                        status="completed",
                        evaluations_active=0,
                        evaluations_completed=session.evaluations_completed,
                        evaluation_count=state.counts().get("evaluation_count", 0),
                        mean_auc=metric,
                        best_auc=(
                            summary.get("best_auc") if isinstance(summary, Mapping) else None
                        ),
                        elapsed_seconds=time.monotonic() - evaluation_started,
                        timing_profile=timing_profile,
                        development_progress=1.0,
                        replay_progress=(
                            1.0
                            if isinstance(replay, Mapping) and replay.get("enabled") is True
                            else 0.0
                        ),
                        current_objective=metric,
                        best_objective=best_objective,
                        best_candidate_id=best_candidate_id,
                        best_score=best_objective,
                        baseline_comparison=baseline_comparison,
                        ir=last_ir,
                        worker_count=config.resources.workers,
                        active_workers=0,
                    )
                    metadata = {
                        "search_metrics": {"pooled_median_auc": metric}
                        if isinstance(metric, (int, float))
                        else {}
                    }
                    state.record_candidate(
                        program_id,
                        source_sha256=candidate.source_sha256,
                        archive_path=str(archive.sources / f"{program_id}.py"),
                        generation=candidate.generation,
                        slot=candidate.slot,
                        status="created",
                        metadata=metadata,
                    )
                evaluation_row = state.evaluation(identity)
                evaluation_result: Mapping[str, Any] = {}
                if isinstance(evaluation_row, Mapping):
                    raw_result = evaluation_row.get("result_json")
                    if isinstance(raw_result, str):
                        try:
                            decoded_result = json.loads(raw_result)
                        except json.JSONDecodeError:
                            decoded_result = {}
                        if isinstance(decoded_result, Mapping):
                            evaluation_result = decoded_result
                evaluation_summary = evaluation_result.get("summary")
                if not isinstance(evaluation_summary, Mapping):
                    evaluation_summary = {}
                evaluation_metric = evaluation_summary.get("mean_auc")
                numeric_metric = (
                    float(evaluation_metric)
                    if isinstance(evaluation_metric, (int, float))
                    and not isinstance(evaluation_metric, bool)
                    else 0.0
                )
                selection_candidates.append(
                    replace(
                        candidate,
                        behavior_signature={
                            **dict(candidate.behavior_signature),
                            "score": numeric_metric,
                        },
                    )
                )
            selected = self._select_parents(
                generation,
                selection_candidates or candidates,
                config.search.population_size,
                config.search.selection,
                results,
            )
            return selected

        try:
            engine = self.engine
            if engine is not None:
                engine_kwargs: dict[str, Any] = {
                    "config": config,
                    "archive": archive,
                    "on_generation": on_generation,
                    "layout": layout,
                    "state": state,
                    "session": session,
                }
                try:
                    engine_parameters = dict(inspect.signature(engine).parameters)
                except (TypeError, ValueError):
                    engine_parameters = {}
                if "observer" in engine_parameters:
                    engine_kwargs["observer"] = callback
                elif "event_callback" in engine_parameters:
                    engine_kwargs["event_callback"] = callback
                if "profiling" in engine_parameters:
                    engine_kwargs["profiling"] = profiling_enabled
                result = engine(wrapped, **engine_kwargs)
            else:
                generation_config = GenerationConfig(
                    campaign_id=config.exp_id,
                    generations=config.search.max_generations,
                    population_size=config.search.population_size,
                    concurrency=config.model.concurrency,
                    max_model_turns=config.search.max_model_turns,
                    max_repairs=config.model.max_repairs,
                    model=config.model.name,
                    effort=config.model.effort,
                    system_prompt=system_prompt,
                    output_schema=output_schema,
                    repair_prompt=repair_prompt,
                    sandbox_limits=SandboxLimits(),
                    checkpoint_path=layout.artifacts / "native-generation-checkpoint.json",
                    turn_timeout_seconds=config.turn_timeout_seconds,
                )
                coordinator = GenerationCoordinator(
                    wrapped,
                    config=generation_config,
                    briefs=slot_briefs,
                    parent_sources=parent_sources,
                    parent_records=parent_records,
                    existing_sources=baseline_sources,
                    prompt_renderer=render_prompt,
                    selection_callback=on_generation,
                    search_feedback=render_search_feedback,
                    archive_context=render_archive_context,
                    behavior_evaluator=_native_behavior,
                    retry_infrastructure=True,
                    budget_exhausted=session.budget_exhausted,
                    observer=callback,
                )
                generation_result = coordinator.run(resume=True)
                result = {
                    "status": generation_result.status,
                    "generation": generation_result.summary.get(
                        "completed_generation_count",
                        generation_result.summary.get("generation_count", 0),
                    ),
                    "summary": dict(generation_result.summary),
                }
        finally:
            wrapped.close()
        if not isinstance(result, Mapping):
            raise NativeExperimentError("native generation engine returned a non-object result")
        batch_completed = str(result.get("status", "completed")) == "completed"
        outcome: dict[str, Any] = {
            "state": "idle",
            "stop_reason": (
                "generation_batch_completed"
                if batch_completed
                else "infrastructure_failed"
                if str(result.get("status")) == "infrastructure_failed"
                else "budget_exhausted"
            ),
            "generation": int(result.get("generation", 0) or 0),
            "provider_turns": session.provider_turns_completed,
            "evaluations": [],
            "result": dict(result),
        }
        if str(result.get("status")) == "infrastructure_failed":
            outcome["last_error"] = (
                "native generation paused after an uncharged provider infrastructure failure"
            )
        if last_timing_profile is not None:
            outcome["timing_profile"] = last_timing_profile
        if last_ir is not None:
            outcome["ir"] = last_ir
        for field in ("deep_operator_profile", "deep_score_profile"):
            if field in result:
                outcome[field] = result[field]
        return outcome


__all__ = ["NativeExperimentAdapter", "NativeExperimentError"]
