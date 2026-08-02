"""Provider boundary for native experiment turns.

This module is intentionally independent of the historical Stage 3/4
campaigns.  A provider consumes the small request envelope used by a native
experiment adapter and returns a terminal, JSON-serialisable envelope.  The
``transport`` injection point keeps deterministic tests and local substitutes
useful while :class:`LocalCodexAppServerProvider` supplies the real local
Codex App Server implementation.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .artifacts import TurnArtifactStore, redact


class NativeProviderError(RuntimeError):
    """Base error raised at the native provider boundary."""


class AuthenticationError(NativeProviderError):
    """The local Codex profile is absent or did not authenticate."""


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_EFFORT = "high"
DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_REPAIRS = 2


class NativeTransport(Protocol):
    """Minimal transport contract, deliberately easy to fake in tests."""

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def repair(
        self, request: Mapping[str, Any], diagnostics: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NativeProviderConfig:
    """Scientific/provider identity exposed in run metadata."""

    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    concurrency: int = DEFAULT_CONCURRENCY
    max_repairs: int = DEFAULT_MAX_REPAIRS

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning effort")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")


class _CodexTransport:
    """Small lazy wrapper over the installed local App Server protocol."""

    def __init__(
        self,
        config: NativeProviderConfig,
        *,
        auth_json: str | Path | None,
        process_factory: Any | None,
        auth_checker: Any | None,
        sandbox_mode: str,
        approval_policy: str,
    ) -> None:
        self.config = config
        self.auth_json = auth_json
        self.process_factory = process_factory
        self.auth_checker = auth_checker
        self.sandbox_mode = sandbox_mode
        self.approval_policy = approval_policy
        self._adapters: list[Any] = []

    def _adapter(self, request: Mapping[str, Any]) -> Any:
        # This is the generic JSONL App Server transport.  It has no Stage 4
        # dependency; native prompts/schemas are supplied by the request.
        from mutation_forge.stage3.app_server import AppServerLimits, CodexAppServerAdapter

        artifact_dir = request.get("artifact_dir")
        prefix = str(request.get("artifact_prefix", "slot-00"))
        adapter = CodexAppServerAdapter(
            process_factory=self.process_factory,
            auth_checker=self.auth_checker,
            auth_json=self.auth_json,
            limits=AppServerLimits(max_turns=1, max_campaigns=1),
            base_instructions=str(request.get("system_prompt", "")),
            artifact_dir=artifact_dir,
            artifact_prefix=prefix,
            artifact_root=request.get("artifact_root")
            if isinstance(request.get("artifact_root"), (str, Path))
            else None,
            sandbox_mode=self.sandbox_mode,
            approval_policy=self.approval_policy,
        )
        self._adapters.append(adapter)
        return adapter

    @staticmethod
    def _usage(raw: Mapping[str, Any]) -> dict[str, Any]:
        usage = raw.get("usage")
        if not isinstance(usage, Mapping):
            return {
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
                "totalTokens": 0,
                "final": True,
                "partial": False,
            }
        value = dict(usage)
        for key in (
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        ):
            if not isinstance(value.get(key), int) or isinstance(value.get(key), bool):
                value[key] = 0
        value["final"] = value.get("final") is True
        value["partial"] = value.get("partial") is True
        return value

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        model = str(request.get("model", self.config.model))
        effort = str(request.get("effort", self.config.effort))
        prompt = request.get("prompt")
        system = request.get("system_prompt")
        schema = request.get("output_schema")
        if not isinstance(system, str) or not system:
            system_path = Path(__file__).resolve().parents[3] / "prompts" / "native" / "system.md"
            try:
                system = system_path.read_text(encoding="utf-8")
            except OSError:
                system = "Return one generated policy JSON object."
        if not isinstance(schema, Mapping):
            schema_path = (
                Path(__file__).resolve().parents[3]
                / "configs"
                / "native"
                / "generated-policy.schema.json"
            )
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                schema = {"type": "object"}
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("native provider prompt must be non-empty")
        from mutation_forge.stage3.app_server import ModelProfile

        adapter = self._adapter(request)
        profile = ModelProfile("codex", model, effort)
        if adapter.logger:
            adapter.logger.text("request.md", prompt)
            adapter.logger.document(
                "request.json",
                {
                    "model": model,
                    "reasoning_effort": effort,
                    "prompt": prompt,
                    "system_prompt": system,
                    "output_schema": dict(schema),
                },
            )
        try:
            result = adapter.generate(prompt, profile, output_schema=schema)
        except Exception as error:
            name = type(error).__name__.lower()
            message = str(error)
            if "auth" in name or "authenticated" in message.lower() or "login" in message.lower():
                raise AuthenticationError(str(redact(message))) from error
            if adapter.logger:
                adapter.logger.document(
                    "response.json",
                    {"status": "failed", "error": str(redact(message))},
                )
            raise
        response_text = str(result.text)
        response: Any = response_text
        try:
            decoded = json.loads(response_text)
        except (TypeError, ValueError):
            pass
        else:
            if isinstance(decoded, Mapping):
                response = dict(decoded)
        usage = self._usage({"usage": self._usage_from_result(result)})
        value = {
            "status": "completed",
            "accepted": True,
            "charged": usage.get("totalTokens", 0) > 0,
            "content": bool(response_text),
            "response": response,
            "response_text": response_text,
            "usage": usage,
            "provider_thread_id": result.thread_id,
            "provider_turn_id": result.turn_id,
            "provider_request_id": result.request_id,
            "model": model,
            "effort": effort,
            "transport_sha256": adapter.logger.transcript_sha256 if adapter.logger else None,
        }
        if adapter.logger:
            adapter.logger.text("response.md", response_text)
            adapter.logger.document("response.json", value)
            adapter.logger.document(
                "provider-raw.json",
                {
                    "usage": usage,
                    "thread_id": result.thread_id,
                    "turn_id": result.turn_id,
                    "request_id": result.request_id,
                    "diagnostics": list(result.diagnostics),
                },
            )
        return value

    @staticmethod
    def _usage_from_result(result: Any) -> Mapping[str, Any]:
        usage = result.usage
        return {
            "inputTokens": usage.input_tokens,
            "cachedInputTokens": usage.cached_input_tokens,
            "outputTokens": usage.output_tokens,
            "reasoningOutputTokens": usage.reasoning_output_tokens,
            "totalTokens": usage.total_tokens,
            "final": usage.final,
            "partial": usage.partial,
        }

    def repair(
        self,
        request: Mapping[str, Any],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if not diagnostics:
            raise ValueError("repair requires bounded diagnostics")
        value = dict(request)
        value["prompt"] = (
            str(request.get("repair_prompt", "Repair the generated policy using the diagnostics."))
            + "\n\n"
            + str(request.get("prompt", ""))
            + "\n\nDiagnostics:\n"
            + json.dumps(list(diagnostics), sort_keys=True, separators=(",", ":"))
        )
        return self.generate(value)

    def close(self) -> None:
        for adapter in self._adapters:
            with suppress(Exception):
                adapter.close()
        self._adapters.clear()


class LocalCodexAppServerProvider:
    """Native provider with a real local Codex transport and fake seam."""

    def __init__(
        self,
        *,
        config: NativeProviderConfig | None = None,
        model: str | None = None,
        effort: str | None = None,
        concurrency: int | None = None,
        max_repairs: int | None = None,
        transport: NativeTransport | None = None,
        auth_json: str | Path | None = None,
        process_factory: Any | None = None,
        auth_checker: Any | None = None,
        artifact_store: TurnArtifactStore | None = None,
        persist_artifacts: bool = True,
        sandbox_mode: str = "danger-full-access",
        approval_policy: str = "never",
    ) -> None:
        if config is None:
            config = NativeProviderConfig(
                model=model if model is not None else DEFAULT_MODEL,
                effort=effort if effort is not None else DEFAULT_EFFORT,
                concurrency=(concurrency if concurrency is not None else DEFAULT_CONCURRENCY),
                max_repairs=(max_repairs if max_repairs is not None else DEFAULT_MAX_REPAIRS),
            )
        elif any(value is not None for value in (model, effort, concurrency, max_repairs)):
            config = NativeProviderConfig(
                model=model if model is not None else config.model,
                effort=effort if effort is not None else config.effort,
                concurrency=concurrency if concurrency is not None else config.concurrency,
                max_repairs=max_repairs if max_repairs is not None else config.max_repairs,
            )
        self.config = config
        self.model = self.config.model
        self.effort = self.config.effort
        self.concurrency = self.config.concurrency
        self.max_repairs = self.config.max_repairs
        self.artifact_store = artifact_store
        self.persist_artifacts = persist_artifacts
        self._transport = transport or _CodexTransport(
            self.config,
            auth_json=auth_json,
            process_factory=process_factory,
            auth_checker=auth_checker,
            sandbox_mode=sandbox_mode,
            approval_policy=approval_policy,
        )
        self._retained: dict[str, Mapping[str, Any]] = {}
        self._lock = threading.RLock()
        self._repair_counts: dict[str, int] = {}

    def _key(self, request: Mapping[str, Any], phase: str) -> str:
        value = request.get("idempotency_key", request.get("request_idempotency_key"))
        if isinstance(value, str) and value:
            return value
        canonical = json.dumps(
            {"phase": phase, **dict(request)},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _persist(self, request: Mapping[str, Any], result: Mapping[str, Any], phase: str) -> None:
        root = request.get("artifact_dir")
        if not isinstance(root, (str, Path)):
            return
        path = Path(root)
        prefix = str(request.get("artifact_prefix", request.get("slot", "slot-00")))
        usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else None
        if usage is None:
            usage = {
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
                "totalTokens": 0,
                "final": True,
                "partial": False,
            }
        response = result.get("response")
        response_text = result.get("response_text")
        if not isinstance(response_text, str):
            response_text = (
                response
                if isinstance(response, str)
                else json.dumps(redact(response), sort_keys=True, separators=(",", ":"))
            )
        # The generic transport streams counters through its in-memory result;
        # materialise the exact usage file before the experiment manifest is
        # indexed so a completed turn never claims evidence it did not retain.
        usage_path = path / f"{prefix}.usage.json"
        if path.is_dir() and not usage_path.exists():
            usage_path.write_text(
                json.dumps(redact(usage), sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        # Native adapters normally validate generated Python immediately
        # after this call.  Do not create a terminal manifest before that
        # finalisation step; doing so would falsely claim validation evidence.
        if (
            isinstance(response, Mapping)
            and isinstance(response.get("source"), str)
            and result.get("validation_completed") is not True
        ):
            return
        if path.is_dir() and any(path.glob(f"{prefix}.wire.jsonl")):
            # The local transport has already flushed immutable streams.
            self.artifact_store_or_default().record_existing_turn(
                path,
                generation=int(request.get("generation", 0)),
                slot=str(request.get("slot", "slot-00")),
                phase=phase,
                request=request,
                result=result,
            )
            return
        store = self.artifact_store_or_default(path)
        store.write_turn(
            generation=int(request.get("generation", 0)),
            slot=str(request.get("slot", "slot-00")),
            phase=phase,
            request=request,
            request_text=str(request.get("prompt", "")),
            response=response,
            response_text=response_text,
            usage=cast(Mapping[str, Any], usage),
            provider_raw=result,
            codex_profile={
                "model": self.model,
                "effort": self.effort,
                "concurrency": self.concurrency,
            },
            rpc=result.get("rpc", []),
            events=result.get("events", []),
            wire=result.get("wire", []),
            stdout=result.get("stdout", []),
            stderr=result.get("stderr", ""),
            request_idempotency_key=self._key(request, phase),
            provider_thread_id=(
                str(result.get("provider_thread_id"))
                if result.get("provider_thread_id") is not None
                else None
            ),
            provider_turn_id=(
                str(result.get("provider_turn_id"))
                if result.get("provider_turn_id") is not None
                else None
            ),
            terminal_status=str(result.get("status", "completed")),
            request_accepted=bool(result.get("accepted", result.get("accepted_turn", False))),
            charged=result.get("charged") if isinstance(result.get("charged"), bool) else None,
            uncharged=(
                result.get("uncharged") if isinstance(result.get("uncharged"), bool) else None
            ),
            content_received=bool(result.get("content", bool(response_text))),
            error=str(result.get("error")) if result.get("error") else None,
        )

    def artifact_store_or_default(self, turn_path: Path | None = None) -> TurnArtifactStore:
        if self.artifact_store is None:
            if turn_path is not None and len(turn_path.parents) >= 4:
                # ExperimentLayout stores turns as root/generations/generation/slot/phase.
                root = turn_path.parents[3]
            else:
                root = Path.cwd() / ".native-artifacts"
            self.artifact_store = TurnArtifactStore(root)
        return self.artifact_store

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        key = self._key(request, "initial")
        with self._lock:
            retained = self._retained.get(key)
            if retained is not None:
                return retained
            result = self._transport.generate(request)
            if not isinstance(result, Mapping):
                result = {"status": "completed", "accepted": True, "response": result}
            value = dict(result)
            value.setdefault("status", "completed")
            value.setdefault("accepted", value.get("status") == "completed")
            if self.persist_artifacts:
                self._persist(request, value, "initial")
            self._retained[key] = value
            return value

    def repair(
        self,
        request: Mapping[str, Any],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if len(diagnostics) > 64:
            raise ValueError("repair diagnostics exceed bound")
        base = str(request.get("idempotency_key", request.get("request_idempotency_key", "repair")))
        with self._lock:
            count = self._repair_counts.get(base, 0)
            if count >= self.max_repairs:
                raise NativeProviderError("maximum native provider repairs exceeded")
            diagnostic_hash = hashlib.sha256(
                json.dumps(
                    list(diagnostics), sort_keys=True, separators=(",", ":"), default=str
                ).encode()
            ).hexdigest()
            key = self._key(
                {**dict(request), "repair_diagnostics_sha256": diagnostic_hash},
                "repair",
            )
            retained = self._retained.get(key)
            if retained is not None:
                return retained
            self._repair_counts[base] = count + 1
            result = self._transport.repair(request, diagnostics)
            value = (
                dict(result)
                if isinstance(result, Mapping)
                else {"status": "completed", "accepted": True, "response": result}
            )
            value.setdefault("status", "completed")
            value.setdefault("accepted", value.get("status") == "completed")
            if self.persist_artifacts:
                self._persist(request, value, f"repair-{count + 1:02d}")
            self._retained[key] = value
            return value

    def close(self) -> None:
        self._transport.close()


NativeExperimentProvider = LocalCodexAppServerProvider
CodexAppServerProvider = LocalCodexAppServerProvider

__all__ = [
    "AuthenticationError",
    "CodexAppServerProvider",
    "LocalCodexAppServerProvider",
    "NativeExperimentProvider",
    "NativeProviderConfig",
    "NativeProviderError",
    "NativeTransport",
]
