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

from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.policy import probe_policy
from mutation_forge.sandbox.validation import validate_policy

from .artifacts import (
    ArtifactIncompleteError,
    TurnArtifactStore,
    generated_policy_diagnostics,
    is_generated_policy,
)
from .config import ExperimentConfig
from .evaluation import evaluate_candidate
from .generation import Candidate, GenerationConfig, GenerationCoordinator, SlotResult
from .layout import ExperimentLayout, WorkspaceError
from .provider import LocalCodexAppServerProvider
from .sessions import SessionContext
from .state import ExperimentStateStore


class NativeExperimentError(RuntimeError):
    """A native experiment could not complete its current safe boundary."""


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
        "context": root / "configs" / "native" / "context.schema.json",
        "proposal": root / "configs" / "native" / "proposal.schema.json",
        "semantic": root / "configs" / "native" / "semantic-descriptions.json",
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


def _native_behavior(
    source: str, limits: SandboxLimits
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result = probe_policy(source, limits)
    if result.get("status") != "completed":
        raise ValueError(str(result.get("status", "behavior probe failed")))
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
            return "repair-01"
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
        self.turns.verify_turn(directory)
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
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIncompleteError(
                "retained native provider evidence is unreadable"
            ) from exc
        if not isinstance(manifest, Mapping):
            return None
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
            if validation.valid:
                try:
                    behavior, telemetry = _native_behavior(source, self.sandbox_limits)
                except Exception as error:
                    value["behavior"] = {"status": "failed", "error": str(error)}
                    value["worker_telemetry"] = {}
                else:
                    value["behavior"] = behavior
                    value["worker_telemetry"] = telemetry
            value["canonical_response"] = dict(cast(Mapping[str, Any], response))
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
        elif (directory / "turn-manifest.json").is_file():
            self.turns.verify_turn(directory)
        else:
            usage_path = directory / f"{slot}.usage.json"
            if not usage_path.is_file() and usage:
                directory.mkdir(parents=True, exist_ok=True)
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
            self.turns.record_existing_turn(
                directory,
                generation=generation,
                slot=slot,
                phase=phase,
                request=request,
                result=value,
            )
            self.turns.verify_turn(directory)
        with self._lock:
            recorded = self.state.record_provider_turn(
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
            if recorded:
                self.session.provider_turns_attempted += 1
                if status == "completed":
                    self.session.provider_turns_completed += 1
                total = usage.get("totalTokens") if isinstance(usage, Mapping) else None
                if isinstance(total, int) and not isinstance(total, bool):
                    self.session.token_usage_delta += total
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
            self._record(request, failure_result)
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
            self._record(request, result)
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
        auth_path = Path.home() / ".codex" / "auth.json"
        provider = self.provider or LocalCodexAppServerProvider(
            model=config.model.name,
            effort=config.model.effort,
            concurrency=config.model.concurrency,
            max_repairs=config.model.max_repairs,
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
                sections.extend(
                    [
                        "",
                        "## Repair instructions",
                        "",
                        repair_prompt.strip(),
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
        last_timing_profile: Mapping[str, Any] | None = None

        def on_generation(
            generation: int, candidates: Sequence[Candidate], results: Sequence[SlotResult]
        ) -> Mapping[str, str]:
            nonlocal best_objective, best_candidate_id, last_timing_profile
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
                        worker_count=config.resources.workers,
                        active_workers=0,
                    )
                    metadata = {
                        "search_metrics": {"pooled_median_auc": metric}
                        if isinstance(metric, (int, float))
                        else {}
                    }
                    selection_candidates.append(
                        replace(
                            candidate,
                            behavior_signature={
                                **dict(candidate.behavior_signature),
                                "score": (
                                    float(metric) if isinstance(metric, (int, float)) else 0.0
                                ),
                            },
                        )
                    )
                    state.record_candidate(
                        program_id,
                        source_sha256=candidate.source_sha256,
                        archive_path=str(archive.sources / f"{program_id}.py"),
                        generation=candidate.generation,
                        slot=candidate.slot,
                        status="created",
                        metadata=metadata,
                    )
            return self._select_parents(
                generation,
                selection_candidates or candidates,
                config.search.population_size,
                config.search.selection,
                results,
            )

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
                )
                coordinator = GenerationCoordinator(
                    wrapped,
                    config=generation_config,
                    parent_sources=parent_sources,
                    parent_records=parent_records,
                    existing_sources=baseline_sources,
                    prompt_renderer=render_prompt,
                    selection_callback=on_generation,
                    behavior_evaluator=_native_behavior,
                    observer=callback,
                )
                generation_result = coordinator.run(resume=True)
                result = {
                    "status": generation_result.status,
                    "generation": generation_result.summary.get("generation_count", 0),
                    "summary": dict(generation_result.summary),
                }
        finally:
            wrapped.close()
        if not isinstance(result, Mapping):
            raise NativeExperimentError("native generation engine returned a non-object result")
        state_value = (
            "completed" if str(result.get("status", "completed")) == "completed" else "idle"
        )
        outcome: dict[str, Any] = {
            "state": state_value,
            "stop_reason": "generation_limit" if state_value == "completed" else "budget_exhausted",
            "generation": int(result.get("generation", 0) or 0),
            "provider_turns": session.provider_turns_completed,
            "evaluations": [],
            "result": dict(result),
        }
        if last_timing_profile is not None:
            outcome["timing_profile"] = last_timing_profile
        for field in ("deep_operator_profile", "deep_score_profile"):
            if field in result:
                outcome[field] = result[field]
        return outcome


__all__ = ["NativeExperimentAdapter", "NativeExperimentError"]
