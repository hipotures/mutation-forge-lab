"""Thin, isolated JSON-RPC adapter for Codex app-server.

The adapter is intentionally transport-focused: it sends no model request on
construction, has no tool implementation, and rejects every server-initiated
request.  A process factory is injectable, which keeps unit tests entirely
offline.
"""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from .isolation import (
    THIN_APP_SERVER_ARGS,
    IsolatedCapsule,
    linux_resource_preexec,
    sanitized_environment,
)

Json = dict[str, Any]

_PASSIVE_ITEM_TYPES = frozenset({"userMessage", "agentMessage", "reasoning"})
_REASONING_DELTA_METHODS = frozenset(
    {
        "item/reasoning/summaryTextDelta",
        "item/reasoning/summaryPartAdded",
        "item/reasoning/textDelta",
    }
)


class AppServerError(RuntimeError):
    """Base error for protocol, isolation, and process failures."""


class ProtocolError(AppServerError):
    pass


class IsolationError(AppServerError):
    pass


class TurnError(AppServerError):
    pass


@dataclass(frozen=True, slots=True)
class AppServerLimits:
    message_bytes: int = 256 * 1024
    stdout_bytes: int = 2 * 1024 * 1024
    stderr_bytes: int = 64 * 1024
    transcript_bytes: int = 2 * 1024 * 1024
    max_turns: int = 1
    max_campaigns: int = 1
    max_events: int = 10_000
    turn_timeout: float = 120.0
    usage_grace: float = 1.0
    startup_timeout: float = 10.0
    resource_cpu_seconds: int = 120
    resource_address_space_bytes: int = 2 * 1024 * 1024 * 1024
    resource_file_bytes: int = 8 * 1024 * 1024
    resource_open_files: int = 256
    resource_processes: int = 1024
    # Explicit ``max_*`` spellings are accepted for callers that map limits
    # directly from a run configuration.
    max_message_bytes: int | None = None
    max_event_bytes: int | None = None
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None
    max_transcript_bytes: int | None = None

    @property
    def message_limit(self) -> int:
        return self.max_event_bytes or self.max_message_bytes or self.message_bytes

    @property
    def stdout_limit(self) -> int:
        return self.max_stdout_bytes or self.stdout_bytes

    @property
    def stderr_limit(self) -> int:
        return self.max_stderr_bytes or self.stderr_bytes

    @property
    def transcript_limit(self) -> int:
        return self.max_transcript_bytes or self.transcript_bytes


@dataclass(frozen=True, slots=True)
class ModelProfile:
    provider: str
    model: str
    effort: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Usage with every server field retained in ``raw``."""

    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    usage: TokenUsage
    thread_id: str
    session_id: str | None
    turn_id: str
    thread_path: str | None
    diagnostics: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderResult:
    slot: str
    accepted: bool
    content: str | None
    usage: TokenUsage | None
    thread_id: str | None
    turn_id: str | None
    status: str


class AppServerGenerationProvider:
    """Offline-testable provider used by the eight-slot host coordinator.

    The host owns concurrency.  Each call creates one independent capsule and
    one one-turn thread, so no candidate can observe another slot or inherit
    its conversation state.
    """

    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
        auth_checker: AuthChecker | None = None,
        auth_json: str | Path | None = None,
        limits: AppServerLimits | None = None,
    ) -> None:
        self.process_factory = process_factory
        self.auth_checker = auth_checker
        self.auth_json = auth_json
        self.limits = limits or AppServerLimits()

    @staticmethod
    def _usage_mapping(usage: TokenUsage) -> dict[str, Any]:
        return {
            "inputTokens": usage.input_tokens,
            "cachedInputTokens": usage.cached_input_tokens,
            "cacheWriteInputTokens": usage.cache_write_input_tokens,
            "outputTokens": usage.output_tokens,
            "reasoningOutputTokens": usage.reasoning_output_tokens,
            "totalTokens": usage.total_tokens,
            "raw": dict(usage.raw),
        }

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute one isolated turn and return the explicit host envelope."""
        prompt = request.get("prompt")
        system_prompt = request.get("system_prompt")
        output_schema = request.get("output_schema")
        model = request.get("model")
        effort = request.get("effort")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("request prompt must be non-empty")
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("request system_prompt must be non-empty")
        if not isinstance(output_schema, Mapping):
            raise ValueError("request output_schema must be an object")
        if not isinstance(model, str) or not isinstance(effort, str):
            raise ValueError("request model and effort must be strings")
        adapter = CodexAppServerAdapter(
            process_factory=self.process_factory,
            auth_checker=self.auth_checker,
            auth_json=self.auth_json,
            limits=self.limits,
            base_instructions=system_prompt,
        )
        try:
            result = adapter.generate(
                prompt,
                ModelProfile("codex", model, effort),
                output_schema=output_schema,
            )
            return {
                "response": result.text,
                "accepted": True,
                "charged": result.usage.total_tokens > 0,
                "content": bool(result.text),
                "usage": self._usage_mapping(result.usage),
                "status": "completed",
                "thread_id": result.thread_id,
                "session_id": result.session_id,
                "turn_id": result.turn_id,
                "model": model,
            }
        finally:
            adapter.close()

    def repair(
        self,
        request: Mapping[str, Any],
        diagnostics: tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, Any]:
        """Run the sole permitted schema/AST repair in a fresh capsule."""
        if not diagnostics:
            raise ValueError("repair requires bounded diagnostics")
        prompt = request.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("repair request prompt must be a string")
        repair_prompt = (
            f"{prompt}\n\nRepair only the output schema or Python AST/runtime "
            "violations listed below. Do not change the scientific objective "
            "and do not request benchmark feedback.\n"
            + json.dumps(list(diagnostics), sort_keys=True, separators=(",", ":"))
        )
        return self.generate({**request, "prompt": repair_prompt})

    def run_campaign(
        self,
        prompts: Mapping[str, str] | None = None,
        profile: ModelProfile | str = "codex/default:high",
        *,
        output_schema: Mapping[str, Any] | None = None,
    ) -> tuple[ProviderResult, ...]:
        slot_prompts = {
            f"slot-{index:02d}": (prompts or {}).get(f"slot-{index:02d}", "") for index in range(8)
        }

        def run(slot: str) -> ProviderResult:
            adapter = CodexAppServerAdapter(
                process_factory=self.process_factory,
                auth_checker=self.auth_checker,
                auth_json=self.auth_json,
                limits=self.limits,
            )
            try:
                result = adapter.generate(
                    slot_prompts[slot] or "Produce one answer.",
                    profile,
                    output_schema=output_schema,
                )
                return ProviderResult(
                    slot,
                    True,
                    result.text,
                    result.usage,
                    result.thread_id,
                    result.turn_id,
                    "completed",
                )
            except Exception as exc:
                return ProviderResult(slot, False, None, None, None, None, type(exc).__name__)
            finally:
                adapter.close()

        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="mforge-slot") as pool:
            futures = {pool.submit(run, slot): slot for slot in slot_prompts}
            results = [future.result() for future in as_completed(futures)]
        return tuple(sorted(results, key=lambda item: item.slot))


class Process(Protocol):
    stdin: Any
    stdout: Any
    stderr: Any

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


ProcessFactory = Callable[..., Process]
AuthChecker = Callable[[IsolatedCapsule], bool]


def resolve_model_profile(identifier: str, *, default_effort: str = "high") -> ModelProfile:
    """Parse ``provider/model[:effort]`` without contacting app-server."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("model identifier must be non-empty")
    value = identifier.strip()
    provider = "codex"
    if "/" in value:
        provider, value = value.split("/", 1)
    if not provider or not value:
        raise ValueError("invalid model identifier")
    effort = default_effort
    if ":" in value:
        value, effort = value.rsplit(":", 1)
    if not value or effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("invalid model identifier or reasoning effort")
    return ModelProfile(provider, value, effort)


def resolve_model_profiles(identifiers: list[str] | tuple[str, ...]) -> tuple[ModelProfile, ...]:
    return tuple(resolve_model_profile(item) for item in identifiers)


def list_model_profiles(identifiers: list[str] | tuple[str, ...]) -> tuple[ModelProfile, ...]:
    """Resolve an application-owned model list without contacting a model."""
    return resolve_model_profiles(identifiers)


_SECRET = re.compile(r"(?i)(token|secret|password|api[_-]?key|authorization)")


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): ("[REDACTED]" if _SECRET.search(str(k)) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        # Paths are diagnostics-only and must not reveal private homes.
        return "[PRIVATE_PATH]" if value.startswith("/") else value
    return value


class CodexAppServerAdapter:
    def __init__(
        self,
        *,
        capsule: IsolatedCapsule | None = None,
        process_factory: ProcessFactory | None = None,
        auth_checker: AuthChecker | None = None,
        auth_json: str | Path | None = None,
        limits: AppServerLimits | None = None,
        client_name: str = "mutation-forge-lab",
        client_title: str = "Mutation Forge Lab",
        client_version: str = "0.1.0",
        base_instructions: str = (
            "Answer the supplied request directly. Do not use tools or runtime context."
        ),
    ) -> None:
        if not base_instructions.strip():
            raise ValueError("base_instructions must be non-empty")
        if capsule is not None and auth_json is not None:
            raise ValueError("auth_json cannot be combined with an existing capsule")
        self._owns_capsule = capsule is None
        self.capsule = capsule or IsolatedCapsule.create(auth_json=auth_json)
        self.process_factory = process_factory or cast(ProcessFactory, subprocess.Popen)
        if auth_checker is None and process_factory is not None:
            raise ValueError("an injected process_factory requires an explicit auth_checker")
        self.auth_checker = auth_checker or self._login_status
        self.limits = limits or AppServerLimits()
        self.client_info = {"name": client_name, "title": client_title, "version": client_version}
        self.base_instructions = base_instructions
        self._process: Process | None = None
        self._next_id = 0
        self._thread: Json | None = None
        self._turns = 0
        self._campaigns = 0
        self._transcript_size = 0
        self._stdout_size = 0
        self._stderr_size = 0
        self._stderr_exceeded = False
        self._stdout_overflow = False
        self._stdout_queue: queue.Queue[Any] = queue.Queue(maxsize=256)
        self._reader_stop = threading.Event()
        self._event_count = 0
        self._active_items: dict[str, str] = {}
        self._completed_item_ids: set[str] = set()
        self._last_status: str = "new"
        self._diagnostics: list[Mapping[str, Any]] = []
        self._lock = threading.RLock()
        self._failed = False

    @staticmethod
    def _login_status(capsule: IsolatedCapsule) -> bool:
        """Check authentication without reading or copying credential material."""
        result = subprocess.run(
            [capsule.codex_executable, "login", "status"],
            cwd=capsule.workdir,
            env=sanitized_environment(capsule),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        output = (result.stdout + result.stderr)[:4096]
        return result.returncode == 0 and "logged in" in output.lower()

    @property
    def diagnostics(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._diagnostics)

    def _record(self, event: str, **fields: Any) -> None:
        if len(self._diagnostics) < 200:
            self._diagnostics.append(_redact({"event": event, **fields}))

    def start(self) -> None:
        with self._lock:
            if self._failed:
                raise AppServerError("failed adapter cannot be reused")
            if self._process is not None and self._process.poll() is None:
                return
            if self._process is not None:
                self._failed = True
                raise AppServerError("app-server process exited and cannot be reused")
            self._reader_stop.clear()
            self._stdout_queue = queue.Queue(maxsize=256)
            argv = [self.capsule.codex_executable, *THIN_APP_SERVER_ARGS[1:]]
            self._process = self.process_factory(
                argv,
                cwd=str(self.capsule.workdir),
                env=sanitized_environment(self.capsule),
                start_new_session=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=partial(
                    linux_resource_preexec,
                    cpu_seconds=self.limits.resource_cpu_seconds,
                    address_space_bytes=self.limits.resource_address_space_bytes,
                    file_bytes=self.limits.resource_file_bytes,
                    open_files=self.limits.resource_open_files,
                    processes=self.limits.resource_processes,
                ),
            )
            self._start_stdout_monitor()
            self._start_stderr_monitor()
            try:
                self._request(
                    "initialize",
                    {"clientInfo": self.client_info, "capabilities": {"experimentalApi": True}},
                    timeout=self.limits.startup_timeout,
                )
                self._notify("initialized", {})
                self._prepare_skills()
                self._last_status = "initialized"
            except Exception:
                self._failed = True
                self.close(force=True)
                raise

    def _start_stderr_monitor(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        def drain() -> None:
            while True:
                try:
                    line = process.stderr.readline()
                except (OSError, EOFError, TimeoutError, queue.Empty):
                    return
                if not line:
                    return
                size = len(line.encode() if isinstance(line, str) else line)
                self._stderr_size += size
                if self._stderr_size > self.limits.stderr_limit:
                    self._stderr_exceeded = True
                    return

        threading.Thread(target=drain, daemon=True).start()

    def _start_stdout_monitor(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        def drain() -> None:
            while not self._reader_stop.is_set():
                try:
                    line = process.stdout.readline()
                except (OSError, EOFError, TimeoutError, queue.Empty) as exc:
                    try:
                        self._stdout_queue.put_nowait(exc)
                    except queue.Full:
                        self._stdout_overflow = True
                    return
                if not line:
                    try:
                        self._stdout_queue.put_nowait(None)
                    except queue.Full:
                        self._stdout_overflow = True
                    return
                try:
                    self._stdout_queue.put(line, timeout=0.1)
                except queue.Full:
                    self._stdout_overflow = True
                    return

        threading.Thread(target=drain, daemon=True).start()

    def _prepare_skills(self) -> None:
        result = self._request(
            "skills/list", {"cwds": [str(self.capsule.workdir)], "forceReload": True}
        )
        data = result.get("data") if isinstance(result, Mapping) else None
        if not isinstance(data, list):
            raise IsolationError("skills/list result missing data")
        paths: set[str] = set()
        for entry in data:
            if not isinstance(entry, Mapping):
                raise IsolationError("invalid skills/list data")
            errors = entry.get("errors", [])
            if errors:
                raise IsolationError("skills/list returned errors")
            skills = entry.get("skills", [])
            if not isinstance(skills, list):
                raise IsolationError("invalid skills list")
            for skill in skills:
                if not isinstance(skill, Mapping) or not skill.get("enabled"):
                    continue
                path = skill.get("path")
                if not isinstance(path, str) or not os.path.isabs(path):
                    raise IsolationError("enabled skill path must be absolute")
                if path in paths:
                    continue
                paths.add(path)
                disabled = self._request("skills/config/write", {"path": path, "enabled": False})
                if disabled.get("effectiveEnabled") is not False:
                    raise IsolationError("skill was not disabled")
        verify = self._request(
            "skills/list", {"cwds": [str(self.capsule.workdir)], "forceReload": True}
        )
        verify_data = verify.get("data") if isinstance(verify, Mapping) else None
        if not isinstance(verify_data, list):
            raise IsolationError("skills verification missing data")
        for entry in verify_data:
            if not isinstance(entry, Mapping):
                raise IsolationError("invalid skills verification data")
            if entry.get("errors"):
                raise IsolationError("skills verification returned errors")
            skills = entry.get("skills", [])
            if not isinstance(skills, list):
                raise IsolationError("invalid skills verification list")
            if any(isinstance(skill, Mapping) and skill.get("enabled") for skill in skills):
                raise IsolationError("enabled skills remain")

    def model_catalog(self) -> tuple[Mapping[str, Any], ...]:
        """Read the installed server's model catalog without starting a thread."""
        self.start()
        result = self._request("model/list", {"limit": 100})
        data = result.get("data") if isinstance(result, Mapping) else None
        if not isinstance(data, list):
            raise ProtocolError("model/list result missing data")
        models: list[Mapping[str, Any]] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise ProtocolError("model/list returned a non-object model")
            if not isinstance(item.get("model"), str):
                raise ProtocolError("model/list entry missing model")
            efforts = item.get("supportedReasoningEfforts")
            if not isinstance(efforts, list):
                raise ProtocolError("model/list entry missing reasoning efforts")
            models.append(dict(item))
        return tuple(models)

    def start_thread(self, profile: ModelProfile | str, *, ephemeral: bool = False) -> Json:
        self.start()
        if not self.auth_checker(self.capsule):
            self._failed = True
            self.close(force=True)
            raise IsolationError("isolated Codex home is not authenticated")
        selected = resolve_model_profile(profile) if isinstance(profile, str) else profile
        params: Json = {
            "model": selected.model,
            "allowProviderModelFallback": False,
            "cwd": str(self.capsule.workdir),
            "sandbox": "read-only",
            "approvalPolicy": "never",
            "ephemeral": ephemeral,
            "baseInstructions": self.base_instructions,
            "developerInstructions": "",
            "personality": "none",
            "environments": [],
            "dynamicTools": [],
            "selectedCapabilityRoots": [],
            "runtimeWorkspaceRoots": [],
            "config": {"model_reasoning_effort": selected.effort},
        }
        result = self._request("thread/start", params, timeout=self.limits.startup_timeout)
        thread = result.get("thread")
        if not isinstance(thread, Mapping) or not isinstance(thread.get("id"), str):
            raise ProtocolError("thread/start returned no thread id")
        sandbox = result.get("sandbox")
        if (
            not isinstance(sandbox, Mapping)
            or sandbox.get("type") != "readOnly"
            or sandbox.get("networkAccess") is not False
        ):
            raise IsolationError("thread capabilities violate read-only isolation")
        for key, expected in (
            ("approvalPolicy", "never"),
            ("cwd", str(self.capsule.workdir)),
            ("model", selected.model),
        ):
            if result.get(key) != expected:
                raise IsolationError(f"thread returned invalid {key}")
        for key in ("instructionSources", "runtimeWorkspaceRoots"):
            if result.get(key, []) != []:
                raise IsolationError(f"thread returned non-empty {key}")
        self._thread = dict(thread)
        self._campaigns += 1
        if self._campaigns > self.limits.max_campaigns:
            raise TurnError("campaign limit exceeded")
        return dict(thread)

    def generate(
        self,
        prompt: str,
        profile: ModelProfile | str,
        *,
        output_schema: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty")
        if self._thread is None:
            self.start_thread(profile)
        if self._turns >= self.limits.max_turns:
            raise TurnError("turn limit exceeded")
        thread = self._thread
        if thread is None:
            raise AppServerError("thread failed to start")
        thread_id = cast(str, thread["id"])
        selected = resolve_model_profile(profile) if isinstance(profile, str) else profile
        params: Json = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt}],
            "model": selected.model,
            "effort": selected.effort,
            "cwd": str(self.capsule.workdir),
            "environments": [],
            "runtimeWorkspaceRoots": [],
        }
        if output_schema is not None:
            params["outputSchema"] = dict(output_schema)
        self._turns += 1
        try:
            result = self._run_turn(params, thread_id)
            self._last_status = "completed"
            return result
        except Exception:
            self._last_status = "failed"
            self._failed = True
            self.close(force=True)
            raise

    def _run_turn(self, params: Json, thread_id: str) -> GenerationResult:
        thread = self._thread
        if thread is None:
            raise AppServerError("thread failed to start")
        if self._active_items:
            raise ProtocolError("previous turn left active items")
        self._completed_item_ids.clear()
        request_id = self._send("turn/start", params)
        final: str | None = None
        turn_id: str | None = None
        pending_turn_id: str | None = None
        usage_raw: Mapping[str, Any] | None = None
        terminal = False
        deadline = time.monotonic() + self.limits.turn_timeout
        while time.monotonic() < deadline:
            message = self._read_message(deadline)
            if message is None:
                raise TurnError("app-server EOF before turn completion")
            if "id" in message:
                if message.get("id") == request_id:
                    if "error" in message:
                        raise TurnError("turn/start failed")
                    result = message.get("result")
                    if isinstance(result, Mapping) and isinstance(result.get("turn"), Mapping):
                        raw_turn_id = result["turn"].get("id")
                        if not isinstance(raw_turn_id, str) or not raw_turn_id:
                            raise ProtocolError("turn/start returned no turn id")
                        turn_id = raw_turn_id
                        if pending_turn_id is not None and pending_turn_id != turn_id:
                            raise ProtocolError("foreign turn event before turn/start response")
                    if turn_id is None:
                        raise ProtocolError("turn/start returned no turn")
                    if terminal:
                        break
                    continue
                self._deny_server_request(message)
                continue
            method = message.get("method")
            params_event = message.get("params")
            if not isinstance(method, str) or not isinstance(params_event, Mapping):
                raise ProtocolError("malformed notification")
            if method == "thread/started":
                if turn_id is not None:
                    raise ProtocolError("thread/started arrived after turn/start response")
                self._correlate_thread_started(params_event, thread_id)
                continue
            observed_turn_id = self._correlate_event(
                method,
                params_event,
                thread_id,
                turn_id,
            )
            if turn_id is None and observed_turn_id is not None:
                if pending_turn_id is not None and pending_turn_id != observed_turn_id:
                    raise ProtocolError("conflicting turn events before turn/start response")
                pending_turn_id = observed_turn_id
            if method == "turn/started":
                continue
            if method == "item/started":
                self._start_item(params_event)
                continue
            if method == "item/agentMessage/delta":
                self._correlate_item_delta(params_event, "agentMessage")
                continue
            if method in _REASONING_DELTA_METHODS:
                self._correlate_item_delta(params_event, "reasoning")
                continue
            if method == "item/completed":
                item = self._complete_item(params_event)
                if item.get("type") == "agentMessage":
                    phase = item.get("phase")
                    if phase == "final_answer" or phase is None:
                        content = item.get("text")
                        if isinstance(content, str):
                            final = content
                continue
            if method == "thread/tokenUsage/updated":
                last = (
                    params_event.get("tokenUsage", {}).get("last")
                    if isinstance(params_event.get("tokenUsage"), Mapping)
                    else None
                )
                if isinstance(last, Mapping):
                    usage_raw = dict(last)
                continue
            if method == "thread/status/changed":
                status = params_event.get("status")
                status_type = status.get("type") if isinstance(status, Mapping) else status
                if status_type in {"systemError", "failed", "interrupted", "cancelled"}:
                    raise TurnError(f"terminal turn status: {status_type}")
                continue
            if method == "turn/completed":
                status = (
                    params_event.get("turn", params_event).get("status")
                    if isinstance(params_event.get("turn", params_event), Mapping)
                    else None
                )
                if status != "completed":
                    raise TurnError(f"turn ended with status {status!r}")
                if self._active_items:
                    raise ProtocolError("turn completed with active items")
                terminal = True
                if turn_id is not None:
                    break
                continue
            if method == "error":
                error = params_event.get("error")
                will_retry = (
                    error.get("willRetry")
                    if isinstance(error, Mapping)
                    else params_event.get("willRetry")
                )
                if will_retry is False:
                    raise TurnError("terminal app-server error")
                continue
            # Unknown notifications are protocol failures: silently ignoring one
            # could leave a turn pending indefinitely.
            raise ProtocolError(f"unknown app-server notification: {method}")
        if not terminal:
            raise TurnError("turn timed out")
        if turn_id is None:
            raise ProtocolError("turn completion had no correlated turn id")
        usage_deadline = time.monotonic() + self.limits.usage_grace
        while time.monotonic() < usage_deadline:
            message = self._read_message(usage_deadline)
            if message is None:
                break
            if message.get("method") == "thread/tokenUsage/updated":
                event = message.get("params", {})
                if isinstance(event, Mapping):
                    self._correlate_event(
                        "thread/tokenUsage/updated",
                        event,
                        thread_id,
                        turn_id,
                    )
                last = (
                    event.get("tokenUsage", {}).get("last")
                    if isinstance(event, Mapping) and isinstance(event.get("tokenUsage"), Mapping)
                    else None
                )
                if isinstance(last, Mapping):
                    usage_raw = dict(last)
            elif message.get("method") == "thread/status/changed":
                event = message.get("params")
                if not isinstance(event, Mapping):
                    raise ProtocolError("malformed notification after turn completion")
                self._correlate_event(
                    "thread/status/changed",
                    event,
                    thread_id,
                    turn_id,
                )
                status = event.get("status")
                status_type = status.get("type") if isinstance(status, Mapping) else status
                if status_type in {"systemError", "failed", "interrupted", "cancelled"}:
                    raise TurnError(f"terminal turn status: {status_type}")
            elif "id" in message:
                self._deny_server_request(message)
            else:
                raise ProtocolError("unexpected notification after turn completion")
        if final is None:
            raise TurnError("no final_answer item")
        usage = self._usage(usage_raw)
        return GenerationResult(
            final,
            usage,
            thread_id,
            cast(str | None, thread.get("sessionId")),
            turn_id or "",
            cast(str | None, thread.get("path")),
            tuple(self._diagnostics),
        )

    def flush(self) -> None:
        """Flush client writes while retaining the persistent capsule."""
        if self._process is not None and self._process.stdin is not None:
            self._process.stdin.flush()

    def inspect_metadata(self) -> Mapping[str, Any]:
        """Return safe identifiers and status; rollout paths remain opaque."""
        return {
            "threadId": self._thread.get("id") if self._thread else None,
            "sessionId": self._thread.get("sessionId") if self._thread else None,
            "threadPath": self._thread.get("path") if self._thread else None,
            "status": self._last_status,
            "turns": self._turns,
        }

    def _usage(self, raw: Mapping[str, Any] | None) -> TokenUsage:
        required = (
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        )
        if raw is None or any(key not in raw for key in required):
            raise TurnError("exact tokenUsage.last with totalTokens is required")
        values = {
            name: raw.get(name, 0)
            for name in (
                "inputTokens",
                "cachedInputTokens",
                "cacheWriteInputTokens",
                "outputTokens",
                "reasoningOutputTokens",
            )
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values.values()
        ):
            raise TurnError("partial token usage")
        if (
            not isinstance(raw["totalTokens"], int)
            or isinstance(raw["totalTokens"], bool)
            or raw["totalTokens"] < 0
        ):
            raise TurnError("invalid total token usage")
        return TokenUsage(
            cast(int, values["inputTokens"]),
            cast(int, values["cachedInputTokens"]),
            cast(int, values["cacheWriteInputTokens"]),
            cast(int, values["outputTokens"]),
            cast(int, values["reasoningOutputTokens"]),
            raw["totalTokens"],
            dict(raw),
        )

    @staticmethod
    def _correlate_thread_started(params: Mapping[str, Any], thread_id: str) -> None:
        observed_thread = params.get("threadId", params.get("thread_id"))
        if observed_thread is None:
            thread = params.get("thread")
            if isinstance(thread, Mapping):
                observed_thread = thread.get("id")
        if not isinstance(observed_thread, str) or not observed_thread:
            raise ProtocolError("thread/started does not contain a valid thread ID")
        if observed_thread != thread_id:
            raise ProtocolError("foreign thread/started event")

    def _correlate_event(
        self,
        method: str,
        params: Mapping[str, Any],
        thread_id: str,
        turn_id: str | None,
    ) -> str | None:
        observed_thread = params.get("threadId", params.get("thread_id"))
        if observed_thread != thread_id:
            raise ProtocolError("missing or foreign thread event")
        if method == "thread/status/changed":
            return None
        observed_turn = params.get("turnId", params.get("turn_id"))
        if observed_turn is None and method in {"turn/started", "turn/completed"}:
            turn = params.get("turn")
            if isinstance(turn, Mapping):
                observed_turn = turn.get("id")
        if not isinstance(observed_turn, str) or not observed_turn:
            raise ProtocolError("missing or foreign turn event")
        if turn_id is not None and observed_turn != turn_id:
            raise ProtocolError("missing or foreign turn event")
        item = params.get("item")
        if isinstance(item, Mapping):
            item_id = item.get("id")
            if item_id is not None and (not isinstance(item_id, str) or not item_id):
                raise ProtocolError("invalid item id")
            for key in ("threadId", "thread_id"):
                if key in item and item[key] != thread_id:
                    raise ProtocolError("foreign item thread")
            if turn_id is not None:
                for key in ("turnId", "turn_id"):
                    if key in item and item[key] != turn_id:
                        raise ProtocolError("foreign item turn")
        item_id = params.get("itemId", params.get("item_id"))
        if item_id is not None and (not isinstance(item_id, str) or not item_id):
            raise ProtocolError("invalid item id")
        return observed_turn

    @staticmethod
    def _item_payload(params: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any]]:
        item = params.get("item")
        if not isinstance(item, Mapping):
            raise ProtocolError("item notification has no item")
        item_id = item.get("id")
        item_type = item.get("type")
        if not isinstance(item_id, str) or not item_id:
            raise ProtocolError("invalid item id")
        if not isinstance(item_type, str) or not item_type:
            raise ProtocolError("invalid item type")
        return item_id, item_type, item

    def _start_item(self, params: Mapping[str, Any]) -> None:
        item_id, item_type, _ = self._item_payload(params)
        if item_type not in _PASSIVE_ITEM_TYPES:
            raise IsolationError(f"unsupported app-server item type: {item_type}")
        if item_id in self._active_items or item_id in self._completed_item_ids:
            raise ProtocolError("duplicate item/started")
        self._active_items[item_id] = item_type

    def _correlate_item_delta(
        self,
        params: Mapping[str, Any],
        expected_type: str,
    ) -> None:
        item_id = params.get("itemId")
        if not isinstance(item_id, str) or not item_id:
            raise ProtocolError("item delta has no valid item ID")
        if self._active_items.get(item_id) != expected_type:
            raise ProtocolError("item delta does not match an active item")

    def _complete_item(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        item_id, item_type, item = self._item_payload(params)
        if self._active_items.get(item_id) != item_type:
            raise ProtocolError("item/completed does not match an active item")
        del self._active_items[item_id]
        self._completed_item_ids.add(item_id)
        return item

    def _request(self, method: str, params: Json, *, timeout: float | None = None) -> Json:
        request_id = self._send(method, params)
        deadline = time.monotonic() + (timeout if timeout is not None else self.limits.turn_timeout)
        while time.monotonic() < deadline:
            message = self._read_message(deadline)
            if message is None:
                raise ProtocolError("app-server EOF")
            if message.get("id") == request_id:
                if "error" in message:
                    raise ProtocolError(f"request {method} failed")
                result = message.get("result")
                return dict(result) if isinstance(result, Mapping) else {}
            if "id" in message:
                self._deny_server_request(message)
                continue
            if method == "thread/start" and message.get("method") == "thread/started":
                raise ProtocolError("thread/started arrived before thread/start response")
            if message.get("method") in {"error", "thread/status/changed"}:
                continue
            # During setup only notifications are tolerated.
        raise ProtocolError(f"timeout waiting for {method}")

    def _send(self, method: str, params: Json) -> int:
        process = self._process
        if process is None or process.stdin is None:
            raise AppServerError("app-server is not running")
        request_id = self._next_id
        self._next_id += 1
        payload = (
            json.dumps(
                {"id": request_id, "method": method, "params": params}, separators=(",", ":")
            )
            + "\n"
        )
        encoded = payload.encode()
        if len(encoded) > self.limits.message_limit:
            raise ProtocolError("outgoing message exceeds limit")
        process.stdin.write(encoded)
        process.stdin.flush()
        self._transcript_size += len(encoded)
        if self._transcript_size > self.limits.transcript_limit:
            raise ProtocolError("transcript limit exceeded")
        self._record("request", method=method, bytes=len(encoded))
        return request_id

    def _notify(self, method: str, params: Json) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise AppServerError("app-server is not running")
        payload = (
            json.dumps({"method": method, "params": params}, separators=(",", ":")) + "\n"
        ).encode()
        process.stdin.write(payload)
        process.stdin.flush()

    def _read_message(self, deadline: float) -> Json | None:
        process = self._process
        if process is None or process.stdout is None:
            raise AppServerError("app-server is not running")
        if self._stderr_exceeded:
            raise ProtocolError("stderr limit exceeded")
        if self._stdout_overflow:
            raise ProtocolError("stdout queue overflow")
        remaining = max(0.0, deadline - time.monotonic())
        if not remaining:
            return None
        try:
            line = self._stdout_queue.get(timeout=remaining)
        except queue.Empty:
            return None
        if isinstance(line, BaseException):
            return None
        if isinstance(line, str):
            line = line.encode()
        if not line:
            return None
        self._stdout_size += len(line)
        if self._stdout_size > self.limits.stdout_limit:
            raise ProtocolError("stdout limit exceeded")
        if len(line) > self.limits.message_limit:
            raise ProtocolError("incoming message exceeds limit")
        self._transcript_size += len(line)
        if self._transcript_size > self.limits.transcript_limit:
            raise ProtocolError("transcript limit exceeded")
        self._event_count += 1
        if self._event_count > self.limits.max_events:
            raise ProtocolError("event limit exceeded")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("malformed JSONL") from exc
        if not isinstance(value, dict):
            raise ProtocolError("JSON-RPC message must be an object")
        return cast(Json, value)

    def _deny_server_request(self, message: Json) -> None:
        process = self._process
        if process is None or process.stdin is None or "id" not in message:
            raise ProtocolError("malformed server request")
        response = {
            "id": message["id"],
            "error": {"code": -32601, "message": "server requests are disabled"},
        }
        process.stdin.write((json.dumps(response, separators=(",", ":")) + "\n").encode())
        process.stdin.flush()
        self._record("denied_server_request", method=message.get("method"))
        raise ProtocolError("unsupported server request")

    def close(self, *, force: bool = False) -> None:
        process, self._process = self._process, None
        if process is None:
            if self._owns_capsule:
                self.capsule.cleanup()
            return
        self._reader_stop.set()
        try:
            if not force and process.poll() is None:
                self._signal_process_group(process, signal.SIGTERM)
                process.wait(timeout=1.0)
            elif process.poll() is None:
                self._signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=1.0)
        except Exception:
            with suppress(Exception):
                self._signal_process_group(process, signal.SIGKILL)
        for stream in (process.stdin, process.stdout, process.stderr):
            with suppress(Exception):
                stream.close()
        if self._owns_capsule:
            self.capsule.cleanup()

    @staticmethod
    def _signal_process_group(process: Process, sig: signal.Signals) -> None:
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0 and hasattr(os, "killpg"):
            with suppress(ProcessLookupError):
                os.killpg(pid, sig)
            return
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()

    def __enter__(self) -> CodexAppServerAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


AppServerAdapter = CodexAppServerAdapter
