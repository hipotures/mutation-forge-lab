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
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .artifacts import (
    TurnArtifactStore,
    redact,
)
from .json_io import write_json


class NativeProviderError(RuntimeError):
    """Base error raised at the native provider boundary."""


class AuthenticationError(NativeProviderError):
    """The local Codex profile is absent or did not authenticate."""


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_EFFORT = "high"
DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_REPAIRS = 2
DEFAULT_TURN_TIMEOUT_BASE_SECONDS = 120.0


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
    turn_timeout_base_seconds: float = DEFAULT_TURN_TIMEOUT_BASE_SECONDS

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning effort")
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.max_repairs < 0:
            raise ValueError("max_repairs must be non-negative")
        if (
            isinstance(self.turn_timeout_base_seconds, bool)
            or not isinstance(self.turn_timeout_base_seconds, int | float)
            or self.turn_timeout_base_seconds <= 0
        ):
            raise ValueError("turn_timeout_base_seconds must be positive")

    @property
    def turn_timeout_seconds(self) -> float:
        return self.turn_timeout_base_seconds


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
        self._adapters_lock = threading.RLock()
        self._closed = False

    def preflight(self) -> None:
        if self.auth_json is None:
            raise AuthenticationError(
                "model.auth_json is required for the isolated Codex App Server"
            )
        from mutation_forge.stage3.app_server import CodexAppServerAdapter
        from mutation_forge.stage3.isolation import IsolatedCapsule, secure_capsule_parent

        capsule = IsolatedCapsule.create(
            secure_capsule_parent(),
            auth_json=self.auth_json,
            sandbox_mode=self.sandbox_mode,
            approval_policy=self.approval_policy,
        )
        try:
            checker = self.auth_checker or CodexAppServerAdapter._login_status
            if not checker(capsule):
                raise AuthenticationError(
                    "Codex authentication copied from model.auth_json is not logged in"
                )
        finally:
            capsule.cleanup()

    def _adapter(self, request: Mapping[str, Any]) -> Any:
        # This is the generic JSONL App Server transport.  It has no Stage 4
        # dependency; native prompts/schemas are supplied by the request.
        from mutation_forge.stage3.app_server import AppServerLimits, CodexAppServerAdapter

        request_limit = request.get("maximum_request_bytes")
        response_limit = request.get("maximum_encoded_response_bytes")
        if (
            isinstance(request_limit, bool)
            or not isinstance(request_limit, int)
            or not 1 <= request_limit <= 1024 * 1024
        ):
            request_limit = None
        if (
            isinstance(response_limit, bool)
            or not isinstance(response_limit, int)
            or not 1 <= response_limit <= 1024 * 1024
        ):
            response_limit = None

        artifact_dir = request.get("artifact_dir")
        prefix = str(request.get("artifact_prefix", "slot-00"))
        if isinstance(artifact_dir, (str, Path)):
            directory = Path(artifact_dir)
            if directory.is_dir() and any(
                path.name == prefix or path.name.startswith(f"{prefix}.")
                for path in directory.iterdir()
            ):
                attempt = 1
                while any(
                    path.name == f"{prefix}.retry-{attempt:02d}"
                    or path.name.startswith(f"{prefix}.retry-{attempt:02d}.")
                    for path in directory.iterdir()
                ):
                    attempt += 1
                prefix = f"{prefix}.retry-{attempt:02d}"
        adapter = CodexAppServerAdapter(
            process_factory=self.process_factory,
            auth_checker=self.auth_checker,
            auth_json=self.auth_json,
            limits=AppServerLimits(
                max_turns=1,
                max_campaigns=1,
                turn_timeout=self.config.turn_timeout_seconds,
                max_request_bytes=request_limit,
                max_response_bytes=response_limit,
                max_event_bytes=(
                    max(256 * 1024, response_limit) if response_limit is not None else None
                ),
            ),
            base_instructions=str(request.get("system_prompt", "")),
            artifact_dir=artifact_dir,
            artifact_prefix=prefix,
            # The transport bound applies to one turn.  A long-running
            # experiment legitimately grows beyond one turn's byte limit.
            artifact_root=None,
            compress_json_artifacts=True,
            sandbox_mode=self.sandbox_mode,
            approval_policy=self.approval_policy,
        )
        with self._adapters_lock:
            if self._closed:
                with suppress(Exception):
                    adapter.close()
                raise NativeProviderError("native transport is closed")
            self._adapters.append(adapter)
        return adapter

    @staticmethod
    def _usage(raw: Mapping[str, Any]) -> dict[str, Any]:
        usage = raw.get("usage")
        if not isinstance(usage, Mapping):
            return {
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "cacheWriteInputTokens": 0,
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
            "cacheWriteInputTokens",
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
                system = "Return the requested bounded JSON object."
        if not isinstance(schema, Mapping):
            schema_path = (
                Path(__file__).resolve().parents[3]
                / "configs"
                / "native"
                / "generated-program-batch.schema.json"
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
            # request.md is the exact final prompt.  The structured request
            # envelope, system prompt, and output schema are retained beside
            # it instead of being injected into Markdown.
            adapter.logger.raw_text("request.md", prompt)
            adapter.logger.document(
                "request.json",
                {
                    **dict(request),
                    "model": model,
                    "reasoning_effort": effort,
                    "prompt": prompt,
                    "system_prompt": system,
                    "output_schema": dict(schema),
                },
            )
            adapter.logger.raw_text("system-prompt.md", system)
            adapter.logger.document("output-schema.json", dict(schema))
        try:
            result = adapter.generate(prompt, profile, output_schema=schema)
        except Exception as error:
            name = type(error).__name__.lower()
            message = str(error)
            if "auth" in name or "authenticated" in message.lower() or "login" in message.lower():
                raise AuthenticationError(str(redact(message))) from error
            if adapter.logger:
                adapter.logger.raw_text("request.md", prompt)
                adapter.logger.document(
                    "provider-raw.json",
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
        response_projection_valid = isinstance(response, Mapping)
        response_diagnostics: tuple[dict[str, str], ...] = (
            ()
            if response_projection_valid
            else (
                {
                    "code": "invalid_json_object",
                    "path": "/",
                    "message": "provider response must decode to a JSON object",
                },
            )
        )
        usage = self._usage({"usage": self._usage_from_result(result)})
        value = {
            "status": "completed",
            "accepted": True,
            "charged": usage.get("totalTokens", 0) > 0,
            "content": bool(response_text),
            "response": response,
            "response_text": response_text,
            "response_projection_valid": response_projection_valid,
            "response_diagnostics": [dict(item) for item in response_diagnostics],
            "transport_diagnostics": [dict(item) for item in result.diagnostics],
            "usage": usage,
            "provider_thread_id": result.thread_id,
            "provider_turn_id": result.turn_id,
            "provider_request_id": result.request_id,
            "provider_duration_ms": result.duration_ms,
            "model": model,
            "effort": effort,
            "transport_sha256": adapter.logger.transcript_sha256 if adapter.logger else None,
        }
        if adapter.logger:
            # CodexAppServerAdapter also records the final message as a
            # transport-level response.md.  Replace that provisional text
            # with the native semantic projection, or remove it entirely for
            # malformed/schema-invalid responses.
            adapter.logger.raw_text("request.md", prompt)
            adapter.logger.raw_text("response.raw.txt", response_text)
            if response_projection_valid and isinstance(response, Mapping):
                adapter.logger.raw_text(
                    "response.md",
                    "```json\n"
                    + json.dumps(
                        response,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n```\n",
                )
            else:
                adapter.logger.remove("response.md")
            if isinstance(response, Mapping):
                adapter.logger.document("response.json", response)
            if response_diagnostics:
                adapter.logger.document(
                    "response-diagnostics.json",
                    [dict(item) for item in response_diagnostics],
                )
            if result.diagnostics:
                adapter.logger.document(
                    "transport-diagnostics.json",
                    [dict(item) for item in result.diagnostics],
                )
            adapter.logger.document(
                "provider-raw.json",
                {
                    "response_text": response_text,
                    "response_projection_valid": response_projection_valid,
                    "usage": usage,
                    "thread_id": result.thread_id,
                    "turn_id": result.turn_id,
                    "request_id": result.request_id,
                    "transport_diagnostics": list(result.diagnostics),
                },
            )
            value["artifact_refs"] = sorted(
                path.name
                for path in Path(adapter.logger.directory).iterdir()
                if path.name == adapter.logger.prefix
                or path.name.startswith(f"{adapter.logger.prefix}.")
            )
        return value

    @staticmethod
    def _usage_from_result(result: Any) -> Mapping[str, Any]:
        usage = result.usage
        return {
            "inputTokens": usage.input_tokens,
            "cachedInputTokens": usage.cached_input_tokens,
            "cacheWriteInputTokens": usage.cache_write_input_tokens,
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
        prompt = str(request.get("prompt", ""))
        # Native v3 requests already carry the locked repair instruction. The
        # fallback only materializes bounded diagnostics for direct callers.
        if "diagnostic" not in prompt.lower():
            pretty = json.dumps(list(diagnostics), ensure_ascii=False, sort_keys=True, indent=2)
            value["prompt"] = (
                str(request.get("repair_prompt", "Repair the policy ASTs using the diagnostics."))
                + "\n\n"
                + prompt
                + "\n\n## Repair diagnostics\n\n```json\n"
                + pretty
                + "\n```"
            )
        else:
            value["prompt"] = prompt
        value["artifact_prefix"] = "repair"
        return self.generate(value)

    def close(self) -> None:
        with self._adapters_lock:
            self._closed = True
            adapters, self._adapters = self._adapters, []
        if not adapters:
            return

        # A run can have one App Server per worker.  Closing these one by one
        # made Ctrl-C wait for every pipe-drain timeout in sequence.  Kill and
        # drain them concurrently, with a bounded wait for the coordinator;
        # the close workers are daemon threads so a broken pipe cannot keep
        # the CLI alive after an interrupt.
        def close_one(adapter: Any) -> None:
            with suppress(Exception):
                adapter.close(force=True)

        threads = [
            threading.Thread(
                target=close_one,
                args=(adapter,),
                name="mforge-provider-close",
                daemon=True,
            )
            for adapter in adapters
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 5.0
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))


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
        turn_timeout_base_seconds: float | None = None,
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
                turn_timeout_base_seconds=(
                    turn_timeout_base_seconds
                    if turn_timeout_base_seconds is not None
                    else DEFAULT_TURN_TIMEOUT_BASE_SECONDS
                ),
            )
        elif any(
            value is not None
            for value in (
                model,
                effort,
                concurrency,
                max_repairs,
                turn_timeout_base_seconds,
            )
        ):
            config = NativeProviderConfig(
                model=model if model is not None else config.model,
                effort=effort if effort is not None else config.effort,
                concurrency=concurrency if concurrency is not None else config.concurrency,
                max_repairs=max_repairs if max_repairs is not None else config.max_repairs,
                turn_timeout_base_seconds=(
                    turn_timeout_base_seconds
                    if turn_timeout_base_seconds is not None
                    else config.turn_timeout_base_seconds
                ),
            )
        self.config = config
        self.model = self.config.model
        self.effort = self.config.effort
        self.concurrency = self.config.concurrency
        self.max_repairs = self.config.max_repairs
        self.turn_timeout_seconds = self.config.turn_timeout_seconds
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

    def preflight(self) -> None:
        preflight = getattr(self._transport, "preflight", None)
        if callable(preflight):
            preflight()

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
        usage_path = path / f"{prefix}.usage.json.gz"
        if path.is_dir() and not usage_path.exists():
            write_json(usage_path, redact(usage))
        # Native adapters normally validate generated Python immediately
        # after this call.  Do not create a terminal manifest before that
        # finalisation step; doing so would falsely claim validation evidence.
        # The native adapter adds validation/projection evidence after the
        # transport returns.  Leave the already-flushed logger directory for
        # that finalization step, including malformed responses that have no
        # ``source`` field.
        if path.is_dir() and result.get("validation_completed") is not True:
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
            request_text_redact=False,
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
            return {**dict(retained), "retained": True}
        result = self._transport.generate(request)
        if not isinstance(result, Mapping):
            result = {"status": "completed", "accepted": True, "response": result}
        value = dict(result)
        value.setdefault("status", "completed")
        value.setdefault("accepted", value.get("status") == "completed")
        with self._lock:
            retained = self._retained.get(key)
            if retained is not None:
                return {**dict(retained), "retained": True}
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
        repair_attempt = int(request.get("repair_attempt", 1))
        max_repairs = int(request.get("max_repairs", self.max_repairs))
        if repair_attempt < 1 or repair_attempt > max_repairs:
            raise NativeProviderError(
                f"repair attempt {repair_attempt} exceeds configured maximum {max_repairs}"
            )
        diagnostic_hash = hashlib.sha256(
            json.dumps(
                list(diagnostics), sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        ).hexdigest()
        key = self._key(
            {**dict(request), "repair_diagnostics_sha256": diagnostic_hash},
            "repair",
        )
        with self._lock:
            retained = self._retained.get(key)
            if retained is not None:
                return {**dict(retained), "retained": True}
        result = self._transport.repair(request, diagnostics)
        value = (
            dict(result)
            if isinstance(result, Mapping)
            else {"status": "completed", "accepted": True, "response": result}
        )
        value.setdefault("status", "completed")
        value.setdefault("accepted", value.get("status") == "completed")
        with self._lock:
            retained = self._retained.get(key)
            if retained is not None:
                return {**dict(retained), "retained": True}
            if self.persist_artifacts:
                self._persist(request, value, f"repair-{repair_attempt:02d}")
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
