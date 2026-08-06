"""Strict, offline-testable JSONL transport for Codex app-server 0.145."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from .artifacts import TransportLogger, safe_value
from .isolation import (
    APP_SERVER_APPROVAL_POLICIES,
    APP_SERVER_SANDBOX_MODES,
    THIN_APP_SERVER_ARGS,
    IsolatedCapsule,
    IsolationError,
    linux_resource_preexec,
    sanitized_environment,
    secure_capsule_parent,
)

Json = dict[str, Any]
FROZEN_STAGE3_MODEL = "gpt-5.6-luna"
FROZEN_STAGE3_EFFORT = "high"


class AppServerError(RuntimeError):
    pass


class ProtocolError(AppServerError):
    pass


class TurnError(AppServerError):
    pass


@dataclass(frozen=True, slots=True)
class AppServerLimits:
    request_bytes: int = 64 * 1024
    response_bytes: int = 16 * 1024
    message_bytes: int = 256 * 1024
    stdout_bytes: int = 2 * 1024 * 1024
    stderr_bytes: int = 64 * 1024
    transcript_bytes: int = 2 * 1024 * 1024
    max_turns: int = 1
    max_campaigns: int = 1
    max_events: int = 10_000
    turn_timeout: float = 120.0
    usage_grace: float = 10.0
    startup_timeout: float = 10.0
    resource_cpu_seconds: int = 120
    resource_address_space_bytes: int = 2 * 1024 * 1024 * 1024
    resource_file_bytes: int = 8 * 1024 * 1024
    resource_open_files: int = 256
    resource_processes: int = 102_400
    max_message_bytes: int | None = None
    max_request_bytes: int | None = None
    max_response_bytes: int | None = None
    max_event_bytes: int | None = None
    max_stdout_bytes: int | None = None
    max_stderr_bytes: int | None = None
    max_transcript_bytes: int | None = None

    @property
    def message_limit(self) -> int:
        return self.max_event_bytes or self.max_message_bytes or self.message_bytes

    @property
    def request_limit(self) -> int:
        return self.max_request_bytes or self.request_bytes

    @property
    def response_limit(self) -> int:
        return self.max_response_bytes or self.response_bytes

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
    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int
    final: bool
    partial: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    usage: TokenUsage
    thread_id: str
    session_id: str | None
    turn_id: str
    request_id: int
    thread_path: str | None
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class CompactionResult:
    thread_id: str
    turn_id: str
    item_id: str
    request_id: int
    usage: TokenUsage | None
    duration_ms: int


@dataclass(frozen=True, slots=True)
class ForkResult:
    source_thread_id: str
    child_thread_id: str
    session_id: str
    thread_path: str | None
    last_turn_id: str
    included_turn_ids: tuple[str, ...]


ProcessFactory = Callable[..., Any]
AuthChecker = Callable[[IsolatedCapsule], bool]


def resolve_model_profile(identifier: str, *, default_effort: str = "high") -> ModelProfile:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("model identifier must be non-empty")
    val = identifier.strip()
    provider = "codex"
    if "/" in val:
        provider, val = val.split("/", 1)
    effort = default_effort
    if ":" in val:
        val, effort = val.rsplit(":", 1)
    if (
        not provider
        or not val
        or effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}
    ):
        raise ValueError("invalid model identifier or reasoning effort")
    return ModelProfile(provider, val, effort)


class _Proc(Protocol):
    stdin: Any
    stdout: Any
    stderr: Any

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


_PASSIVE = {"userMessage", "agentMessage", "reasoning"}
_GLOBAL = {
    "account/updated",
    "account/rateLimits/updated",
    "configWarning",
    "remoteControl/status/changed",
}
_DELTAS = {
    "item/agentMessage/delta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded",
    "item/reasoning/textDelta",
}
_SETUP_NOTIFICATIONS = {"initialized"}
_WAITING_NOTIFICATIONS = _GLOBAL | {
    "thread/started",
    "thread/status/changed",
    "thread/tokenUsage/updated",
}


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
        artifact_dir: str | Path | None = None,
        artifact_prefix: str = "",
        artifact_root: str | Path | None = None,
        artifact_max_bytes: int = 32 * 1024 * 1024,
        compress_json_artifacts: bool = False,
        protocol_audit_sha256: str | None = None,
        sandbox_mode: str = "danger-full-access",
        approval_policy: str = "never",
        copy_rollout_artifact: bool = True,
    ):
        if not base_instructions.strip():
            raise ValueError("base_instructions must be non-empty")
        if capsule is not None and auth_json is not None:
            raise ValueError("auth_json cannot be combined with an existing capsule")
        if sandbox_mode not in APP_SERVER_SANDBOX_MODES:
            raise ValueError("unsupported app-server sandbox mode")
        if approval_policy not in APP_SERVER_APPROVAL_POLICIES:
            raise ValueError("unsupported app-server approval policy")
        self._owns_capsule = capsule is None
        capsule_parent: Path | None = None
        if capsule is None:
            capsule_parent = secure_capsule_parent()
        self.capsule = capsule or IsolatedCapsule.create(
            capsule_parent,
            auth_json=auth_json,
            sandbox_mode=sandbox_mode,
            approval_policy=approval_policy,
        )
        self.process_factory = process_factory or cast(ProcessFactory, subprocess.Popen)
        if auth_checker is None and process_factory is not None:
            raise ValueError("an injected process_factory requires an explicit auth_checker")
        self.auth_checker = auth_checker or self._login_status
        self.limits = limits or AppServerLimits()
        self.client_info = {"name": client_name, "title": client_title, "version": client_version}
        self.base_instructions = base_instructions
        self.protocol_audit_sha256 = protocol_audit_sha256
        self.sandbox_mode = sandbox_mode
        self.approval_policy = approval_policy
        self.copy_rollout_artifact = copy_rollout_artifact
        self._process: _Proc | None = None
        self._next_id = 0
        self._thread: Json | None = None
        self._forked_threads: dict[str, Json] = {}
        self._completed_turn_ids: dict[str, list[str]] = {}
        self._turns = 0
        self._campaigns = 0
        self._failed = False
        self._event_count = 0
        self._transcript_size = 0
        self._stdout_size = 0
        self._stderr_size = 0
        self._stdout_lines: list[bytes] = []
        self._stderr_lines: list[bytes] = []
        self._stderr_exceeded = False
        self._stdout_overflow = False
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=5_120)
        self._queued_bytes = 0
        self._queue_lock = threading.Lock()
        self._stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._active: dict[str, str] = {}
        self._completed: set[str] = set()
        self._diag: list[Mapping[str, Any]] = []
        self._server_retries = 0
        self._server_warnings = 0
        self._last_status = "new"
        self.logger = (
            TransportLogger(
                Path(artifact_dir),
                artifact_prefix,
                max_bytes=self.limits.transcript_limit,
                max_events=self.limits.max_events,
                max_line_bytes=self.limits.message_limit,
                aggregate_root=Path(artifact_root) if artifact_root is not None else None,
                max_aggregate_bytes=artifact_max_bytes,
                compress_json=compress_json_artifacts,
            )
            if artifact_dir
            else None
        )
        self.partial_result: GenerationResult | None = None
        self._current_thread_id: str | None = None
        self._current_turn_id: str | None = None
        self._last_usage_raw: Mapping[str, Any] | None = None

    @staticmethod
    def _login_status(capsule: IsolatedCapsule) -> bool:
        try:
            p = subprocess.run(
                [capsule.codex_executable, "login", "status"],
                cwd=capsule.workdir,
                env=sanitized_environment(capsule),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return p.returncode == 0 and "logged in" in (p.stdout + p.stderr)[:4096].lower()
        except Exception:
            return False

    @property
    def diagnostics(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._diag)

    def _record(self, event: str, **fields: Any) -> None:
        if len(self._diag) < 200:
            self._diag.append(cast(Mapping[str, Any], safe_value({"event": event, **fields})))

    def start(self) -> None:
        if self._failed:
            raise AppServerError("failed adapter cannot be reused")
        if self._process is not None:
            if self._process.poll() is None:
                return
            self._failed = True
            raise AppServerError("app-server process exited and cannot be reused")
        argv = [self.capsule.codex_executable, *THIN_APP_SERVER_ARGS[1:]]
        self._process = cast(
            _Proc,
            self.process_factory(
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
            ),
        )
        self._stop.clear()
        self._start_reader()
        self._start_stderr()
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
            self._drain_evidence()
            self.close(force=True)
            raise

    def _start_reader(self) -> None:
        p = self._process

        def run() -> None:
            while not self._stop.is_set() and p is not None:
                try:
                    line = p.stdout.readline(self.limits.message_limit + 1)
                except queue.Empty:
                    self._enqueue(None)
                    return
                except BaseException as exc:
                    self._stdout_overflow = True
                    self._enqueue(exc)
                    return
                raw = line.encode() if isinstance(line, str) else bytes(line)
                if not raw:
                    self._enqueue(None)
                    return
                # Reject unterminated or oversized JSONL before retaining/parsing it.
                if (
                    not raw.endswith(b"\n")
                    or len(raw) > self.limits.message_limit
                    or len(raw) > self.limits.stdout_limit
                ):
                    self._stdout_overflow = True
                    return
                if not self._enqueue(raw):
                    self._stdout_overflow = True
                    return

        self._reader_thread = threading.Thread(target=run, daemon=True)
        self._reader_thread.start()

    def _enqueue(self, value: Any) -> bool:
        size = 0
        if value is not None and not isinstance(value, BaseException):
            size = len(value.encode() if isinstance(value, str) else bytes(value))
        with self._queue_lock:
            if self._queued_bytes + size > self.limits.stdout_limit:
                return False
            try:
                self._queue.put_nowait(value)
            except queue.Full:
                return False
            self._queued_bytes += size
            return True

    def _start_stderr(self) -> None:
        p = self._process

        def run() -> None:
            if p is None:
                return
            while True:
                try:
                    remaining = self.limits.stderr_limit - self._stderr_size
                    if remaining <= 0:
                        self._stderr_exceeded = True
                        return
                    line = p.stderr.readline(remaining + 1)
                except BaseException:
                    return
                if not line:
                    return
                raw = line.encode() if isinstance(line, str) else bytes(line)
                if len(raw) > remaining:
                    self._stderr_lines.append(raw[:remaining])
                    self._stderr_size += remaining
                    self._stderr_exceeded = True
                    return
                self._stderr_lines.append(raw)
                self._stderr_size += len(raw)
                if self.logger:
                    try:
                        self.logger.text(
                            "stderr.txt", b"".join(self._stderr_lines).decode("utf-8", "replace")
                        )
                    except ValueError:
                        return

        self._stderr_thread = threading.Thread(target=run, daemon=True)
        self._stderr_thread.start()

    def _prepare_skills(self) -> None:
        result = self._request(
            "skills/list", {"cwds": [str(self.capsule.workdir)], "forceReload": True}
        )
        data = result.get("data")
        if not isinstance(data, list):
            raise IsolationError("skills/list result missing data")
        paths = set()
        for entry in data:
            if not isinstance(entry, Mapping) or entry.get("errors"):
                raise IsolationError("skills/list returned errors")
            skills = entry.get("skills", [])
            if not isinstance(skills, list):
                raise IsolationError("invalid skills list")
            for s in skills:
                if not isinstance(s, Mapping) or not isinstance(s.get("enabled"), bool):
                    raise IsolationError("invalid skill entry")
                if s["enabled"]:
                    path = s.get("path")
                    if not isinstance(path, str) or not os.path.isabs(path):
                        raise IsolationError("enabled skill path must be absolute")
                    if path not in paths:
                        paths.add(path)
                        if (
                            self._request(
                                "skills/config/write", {"path": path, "enabled": False}
                            ).get("effectiveEnabled")
                            is not False
                        ):
                            raise IsolationError("skill was not disabled")
        verify = self._request(
            "skills/list", {"cwds": [str(self.capsule.workdir)], "forceReload": True}
        )
        vd = verify.get("data")
        if not isinstance(vd, list):
            raise IsolationError("skills verification missing data")
        for e in vd:
            if (
                not isinstance(e, Mapping)
                or e.get("errors")
                or not isinstance(e.get("skills"), list)
                or any(
                    not isinstance(s, Mapping)
                    or not isinstance(s.get("enabled"), bool)
                    or s["enabled"]
                    for s in e["skills"]
                )
            ):
                raise IsolationError("enabled skills remain")

    def model_catalog(self) -> tuple[Mapping[str, Any], ...]:
        self.start()
        data = self._request("model/list", {"limit": 100}).get("data")
        if not isinstance(data, list):
            raise ProtocolError("model/list result missing data")
        out = []
        for i in data:
            if (
                not isinstance(i, Mapping)
                or not isinstance(i.get("model"), str)
                or not isinstance(i.get("supportedReasoningEfforts"), list)
            ):
                raise ProtocolError("model/list returned malformed model")
            out.append(dict(i))
        return tuple(out)

    def start_thread(self, profile: ModelProfile | str, *, ephemeral: bool = True) -> Json:
        # Match the proven TheML one-shot transport: every generation receives
        # one ephemeral thread in its own private capsule.
        if self._failed:
            raise AppServerError("failed adapter cannot be reused")
        if self._campaigns >= self.limits.max_campaigns:
            raise TurnError("campaign limit exceeded")
        selected = resolve_model_profile(profile) if isinstance(profile, str) else profile
        if selected.provider != "codex":
            raise ValueError("only the installed Codex provider is supported")
        if not self.auth_checker(self.capsule):
            self._failed = True
            self.close(force=True)
            raise IsolationError("isolated Codex home is not authenticated")
        self.start()
        params = {
            "model": selected.model,
            "allowProviderModelFallback": False,
            "cwd": str(self.capsule.workdir),
            "sandbox": self.sandbox_mode,
            "approvalPolicy": self.approval_policy,
            "approvalsReviewer": "user",
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
        if self.logger:
            self.logger.profile(
                {
                    "model": selected.model,
                    "effort": selected.effort,
                    "ephemeral": ephemeral,
                    "sandbox": self.sandbox_mode,
                    "approvalPolicy": self.approval_policy,
                    "artifactPrefix": self.logger.prefix,
                    "protocolAuditSha256": self.protocol_audit_sha256,
                }
            )
        try:
            result = self._request("thread/start", params, timeout=self.limits.startup_timeout)
            thread = result.get("thread")
            if not isinstance(thread, Mapping) or not isinstance(thread.get("id"), str):
                raise ProtocolError("thread/start returned no thread id")
            if thread.get("ephemeral") is not ephemeral or thread.get("cwd") != str(
                self.capsule.workdir
            ):
                raise IsolationError("thread identity violates capsule persistence settings")
            thread_path = thread.get("path")
            if not ephemeral:
                if not isinstance(thread_path, str) or not Path(thread_path).is_absolute():
                    raise IsolationError(
                        "persistent capsule thread returned no absolute rollout path"
                    )
                try:
                    Path(thread_path).resolve(strict=False).relative_to(
                        self.capsule.root.resolve(strict=True)
                    )
                except ValueError as error:
                    raise IsolationError("thread rollout path escapes the capsule") from error
            sandbox = result.get("sandbox")
            expected_sandbox_type = (
                "readOnly" if self.sandbox_mode == "read-only" else "dangerFullAccess"
            )
            if (
                not isinstance(sandbox, Mapping)
                or sandbox.get("type") != expected_sandbox_type
                or (self.sandbox_mode == "read-only" and sandbox.get("networkAccess") is not False)
            ):
                raise IsolationError("thread capabilities do not match configured sandbox mode")
            for key, expected in (
                ("approvalPolicy", self.approval_policy),
                ("approvalsReviewer", "user"),
                ("cwd", str(self.capsule.workdir)),
                ("model", selected.model),
                ("reasoningEffort", selected.effort),
            ):
                if result.get(key) != expected:
                    raise IsolationError(f"thread returned invalid {key}")
            if (
                result.get("instructionSources", []) != []
                or result.get("runtimeWorkspaceRoots", []) != []
            ):
                raise IsolationError("thread returned non-empty context")
            self._thread = dict(thread)
            self._campaigns += 1
            return dict(thread)
        except Exception:
            self._failed = True
            self._drain_evidence()
            self.close(force=True)
            raise

    def resume_thread(
        self,
        profile: ModelProfile | str,
        *,
        thread_id: str,
        thread_path: str | None = None,
    ) -> Json:
        """Resume one durable thread in an experimental replacement process."""

        if self._thread is not None:
            raise TurnError("a durable thread is already active")
        thread = self._resume_thread(
            profile,
            thread_id=thread_id,
            thread_path=thread_path,
        )
        self._thread = dict(thread)
        return dict(thread)

    def resume_forked_thread(
        self,
        profile: ModelProfile | str,
        *,
        thread_id: str,
        thread_path: str | None = None,
    ) -> Json:
        """Load another durable worker into the same replacement process."""

        if self._thread is None:
            raise TurnError("additional resume requires an active durable thread")
        if thread_id in self._forked_threads:
            raise ValueError("durable thread is already loaded")
        return self._resume_thread(
            profile,
            thread_id=thread_id,
            thread_path=thread_path,
        )

    def _resume_thread(
        self,
        profile: ModelProfile | str,
        *,
        thread_id: str,
        thread_path: str | None,
    ) -> Json:
        if self._campaigns >= self.limits.max_campaigns:
            raise TurnError("campaign limit exceeded")
        selected = resolve_model_profile(profile) if isinstance(profile, str) else profile
        if selected.provider != "codex":
            raise ValueError("only the installed Codex provider is supported")
        if not self.auth_checker(self.capsule):
            raise IsolationError("isolated Codex home is not authenticated")
        self.start()
        params: Json = {
            "threadId": thread_id,
            "model": selected.model,
            "cwd": str(self.capsule.workdir),
            "sandbox": self.sandbox_mode,
            "approvalPolicy": self.approval_policy,
            "baseInstructions": self.base_instructions,
            "developerInstructions": "",
            "runtimeWorkspaceRoots": [],
            "config": {"model_reasoning_effort": selected.effort},
        }
        if thread_path is not None:
            params["path"] = thread_path
        result = self._request("thread/resume", params, timeout=self.limits.startup_timeout)
        thread = result.get("thread")
        if (
            not isinstance(thread, Mapping)
            or thread.get("id") != thread_id
            or thread.get("ephemeral") is not False
        ):
            raise IsolationError("thread/resume returned a different or ephemeral thread")
        self._forked_threads[thread_id] = dict(thread)
        self._campaigns += 1
        self._drain_resume_notifications(thread_id)
        return dict(thread)

    def rotate_logger(
        self,
        artifact_dir: str | Path,
        artifact_prefix: str,
        *,
        compress_json: bool = True,
    ) -> None:
        """Start an isolated artifact prefix for the next experimental turn."""

        if self._current_turn_id is not None and self._last_status not in {
            "completed",
            "initialized",
            "new",
        }:
            raise TurnError("cannot rotate artifacts during an active turn")
        self._stdout_lines.clear()
        self._stderr_lines.clear()
        self.logger = TransportLogger(
            Path(artifact_dir),
            artifact_prefix,
            max_bytes=self.limits.transcript_limit,
            max_events=self.limits.max_events,
            max_line_bytes=self.limits.message_limit,
            compress_json=compress_json,
        )

    def generate(
        self,
        prompt: str,
        profile: ModelProfile | str,
        *,
        output_schema: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        return self._generate_on_thread(
            prompt,
            profile,
            output_schema=output_schema,
            persistent=False,
            allow_completed_reasoning=False,
            allow_server_retry=False,
        )

    def generate_ephemeral_experiment(
        self,
        prompt: str,
        profile: ModelProfile | str,
        *,
        output_schema: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Run one isolated experimental turn against current CLI event ordering."""

        return self._generate_on_thread(
            prompt,
            profile,
            output_schema=output_schema,
            persistent=False,
            allow_completed_reasoning=True,
            allow_server_retry=False,
        )

    def generate_persistent(
        self,
        prompt: str,
        profile: ModelProfile | str,
        *,
        output_schema: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Run one turn on a durable experimental thread."""

        return self._generate_on_thread(
            prompt,
            profile,
            output_schema=output_schema,
            persistent=True,
            allow_completed_reasoning=True,
            allow_server_retry=True,
        )

    def compact_persistent_thread(self) -> CompactionResult:
        """Run one explicit compaction turn on an experimental durable thread."""

        if self._thread is None or self._thread.get("ephemeral") is not False:
            raise IsolationError("compaction requires a durable thread")
        if self._turns >= self.limits.max_turns:
            raise TurnError("turn limit exceeded")
        thread_id = cast(str, self._thread["id"])
        self._turns += 1
        try:
            result = self._run_compaction(thread_id)
            self._last_status = "completed"
            return result
        except Exception:
            self._last_status = "failed"
            self._failed = True
            if self._current_thread_id and self._current_turn_id and self._process is not None:
                with suppress(Exception):
                    self._request(
                        "turn/interrupt",
                        {
                            "threadId": self._current_thread_id,
                            "turnId": self._current_turn_id,
                        },
                        timeout=min(1.0, self.limits.usage_grace),
                    )
            self._drain_evidence()
            self.close(force=True)
            raise

    def fork_persistent_thread(
        self,
        profile: ModelProfile | str,
        *,
        last_turn_id: str,
        activate: bool = False,
    ) -> ForkResult:
        """Fork an experimental durable thread at one completed inclusive turn."""

        if (
            self._thread is None
            or self._thread.get("ephemeral") is not False
            or self._last_status != "completed"
        ):
            raise IsolationError("fork requires an idle durable thread")
        if not isinstance(last_turn_id, str) or not last_turn_id:
            raise ValueError("last_turn_id must be non-empty")
        if self._campaigns >= self.limits.max_campaigns:
            raise TurnError("campaign limit exceeded")
        selected = resolve_model_profile(profile) if isinstance(profile, str) else profile
        if selected.provider != "codex":
            raise ValueError("only the installed Codex provider is supported")
        source = self._thread
        source_thread_id = cast(str, source["id"])
        result = self._request(
            "thread/fork",
            {
                "threadId": source_thread_id,
                "lastTurnId": last_turn_id,
                "model": selected.model,
                "cwd": str(self.capsule.workdir),
                "sandbox": self.sandbox_mode,
                "approvalPolicy": self.approval_policy,
                "baseInstructions": self.base_instructions,
                "developerInstructions": "",
                "runtimeWorkspaceRoots": [],
                "ephemeral": False,
                "excludeTurns": False,
                "config": {"model_reasoning_effort": selected.effort},
            },
            timeout=self.limits.startup_timeout,
        )
        thread = result.get("thread")
        if not isinstance(thread, Mapping):
            raise ProtocolError("thread/fork returned no child thread")
        child_thread_id = thread.get("id")
        session_id = thread.get("sessionId")
        turns = thread.get("turns")
        if (
            not isinstance(child_thread_id, str)
            or not child_thread_id
            or child_thread_id == source_thread_id
            or thread.get("forkedFromId") != source_thread_id
            or thread.get("ephemeral") is not False
            or not isinstance(session_id, str)
            or not session_id
            or not isinstance(turns, list)
        ):
            raise ProtocolError("thread/fork returned invalid child identity")
        included_turn_ids: list[str] = []
        for raw_turn in turns:
            if not isinstance(raw_turn, Mapping):
                raise ProtocolError("thread/fork returned malformed history")
            turn, _ = self._validated_turn(
                {"turn": raw_turn},
                source="thread/fork",
            )
            if turn.get("status") != "completed":
                raise ProtocolError("thread/fork returned non-completed history")
            included_turn_ids.append(cast(str, turn["id"]))
        if not included_turn_ids or included_turn_ids[-1] != last_turn_id:
            raise ProtocolError("thread/fork did not honor lastTurnId")
        for key, expected in (
            ("approvalPolicy", self.approval_policy),
            ("approvalsReviewer", "user"),
            ("cwd", str(self.capsule.workdir)),
            ("model", selected.model),
        ):
            if result.get(key) != expected:
                raise IsolationError(f"thread/fork returned invalid {key}")
        if (
            result.get("instructionSources", []) != []
            or result.get("runtimeWorkspaceRoots", []) != []
        ):
            raise IsolationError("forked thread returned non-empty context")
        sandbox = result.get("sandbox")
        expected_sandbox_type = (
            "readOnly" if self.sandbox_mode == "read-only" else "dangerFullAccess"
        )
        if (
            not isinstance(sandbox, Mapping)
            or sandbox.get("type") != expected_sandbox_type
            or (self.sandbox_mode == "read-only" and sandbox.get("networkAccess") is not False)
        ):
            raise IsolationError("forked thread capabilities do not match")
        thread_path = thread.get("path")
        if (
            not isinstance(thread_path, str)
            or not Path(thread_path).is_absolute()
            or thread.get("cwd") != str(self.capsule.workdir)
        ):
            raise ProtocolError("forked thread returned invalid path or cwd")
        try:
            Path(thread_path).resolve(strict=False).relative_to(
                self.capsule.root.resolve(strict=True)
            )
        except ValueError as exc:
            raise IsolationError("forked rollout path escapes capsule") from exc
        fork = ForkResult(
            source_thread_id=source_thread_id,
            child_thread_id=child_thread_id,
            session_id=session_id,
            thread_path=thread_path,
            last_turn_id=last_turn_id,
            included_turn_ids=tuple(included_turn_ids),
        )
        self._forked_threads[child_thread_id] = dict(thread)
        self._completed_turn_ids[child_thread_id] = list(included_turn_ids)
        self._drain_fork_notifications(source_thread_id)
        self._campaigns += 1
        if activate:
            self.activate_forked_thread(child_thread_id)
        return fork

    def activate_forked_thread(self, child_thread_id: str) -> None:
        """Select one child previously returned by thread/fork."""

        thread = self._forked_threads.get(child_thread_id)
        if thread is None:
            raise ValueError("unknown forked thread")
        self._thread = dict(thread)
        self._current_thread_id = child_thread_id
        self._current_turn_id = None
        self._last_usage_raw = None
        self._active.clear()
        self._completed.clear()
        self._last_status = "initialized"

    def _generate_on_thread(
        self,
        prompt: str,
        profile: ModelProfile | str,
        *,
        output_schema: Mapping[str, Any] | None,
        persistent: bool,
        allow_completed_reasoning: bool,
        allow_server_retry: bool,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty")
        if self._thread is None:
            self.start_thread(profile, ephemeral=not persistent)
        elif self._thread.get("ephemeral") is persistent:
            expected = "durable" if persistent else "ephemeral"
            raise IsolationError(f"generation requires an {expected} thread")
        if self._turns >= self.limits.max_turns:
            raise TurnError("turn limit exceeded")
        thread = cast(Json, self._thread)
        selected = resolve_model_profile(profile) if isinstance(profile, str) else profile
        params = {
            "threadId": thread["id"],
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
            r = self._run_turn(
                params,
                cast(str, thread["id"]),
                allow_completed_reasoning=allow_completed_reasoning,
                allow_server_retry=allow_server_retry,
            )
            self._last_status = "completed"
            return r
        except Exception:
            self._last_status = "failed"
            self._failed = True
            if self._current_thread_id and self._current_turn_id and self._process is not None:
                with suppress(Exception):
                    self._request(
                        "turn/interrupt",
                        {"threadId": self._current_thread_id, "turnId": self._current_turn_id},
                        timeout=min(1.0, self.limits.usage_grace),
                    )
            self._drain_evidence()
            self.close(force=True)
            raise

    def _drain_evidence(self) -> None:
        """Persist already-emitted trailing JSONL without interpreting it."""
        if not self.logger:
            return
        while True:
            try:
                value = self._queue.get_nowait()
            except queue.Empty:
                break
            if value is not None and not isinstance(value, BaseException):
                size = len(value.encode() if isinstance(value, str) else bytes(value))
                with self._queue_lock:
                    self._queued_bytes = max(0, self._queued_bytes - size)
            if value is None or isinstance(value, BaseException):
                break
            raw = value.encode() if isinstance(value, str) else bytes(value)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    try:
                        self.logger.message(parsed, raw)
                    except ValueError:
                        break
            except (ValueError, TypeError, UnicodeDecodeError):
                continue

    def _run_turn(
        self,
        params: Json,
        thread_id: str,
        *,
        allow_completed_reasoning: bool = False,
        allow_server_retry: bool = False,
    ) -> GenerationResult:
        self._current_thread_id = thread_id
        request_id = self._send("turn/start", params)
        turn_id = None
        pending = None
        final = None
        final_item = None
        usage_raw = None
        turn_duration_ms: int | None = None
        terminal = False
        deadline = time.monotonic() + self.limits.turn_timeout
        self._active.clear()
        self._completed.clear()
        while time.monotonic() < deadline:
            msg = self._read_message(deadline)
            if msg is None:
                if self._stdout_overflow:
                    raise ProtocolError("incoming message exceeds limit")
                if self._stderr_exceeded:
                    raise ProtocolError("output limit exceeded")
                if (
                    time.monotonic() >= deadline
                    and self._process is not None
                    and self._process.poll() is None
                ):
                    raise TurnError("turn timed out")
                raise TurnError("app-server EOF before turn completion")
            if "id" in msg and "method" in msg:
                self._deny_server_request(msg)
            if "id" in msg:
                if msg.get("id") != request_id:
                    self._deny_server_request(msg)
                elif "error" in msg:
                    raise TurnError("turn/start failed")
                else:
                    tr, _ = self._validated_turn(
                        cast(Mapping[str, Any], msg.get("result", {})), source="turn/start"
                    )
                    turn_id = cast(str, tr["id"])
                    self._current_turn_id = turn_id
                    if tr.get("status") != "inProgress":
                        raise ProtocolError("turn/start returned non-running turn")
                    if pending is not None and pending != turn_id:
                        raise ProtocolError("foreign turn event before turn/start response")
                continue
            method = msg.get("method")
            ev = msg.get("params")
            if not isinstance(method, str) or not isinstance(ev, Mapping):
                raise ProtocolError("malformed notification")
            if self._consume_global_notification(method, ev):
                continue
            if method == "warning" and allow_server_retry:
                if (
                    not isinstance(ev.get("message"), str)
                    or not ev["message"]
                    or ev.get("threadId") != thread_id
                ):
                    raise ProtocolError("malformed app-server warning")
                self._server_warnings += 1
                continue
            if method == "thread/started":
                if turn_id is not None:
                    raise ProtocolError("thread/started arrived after turn/start response")
                self._correlate_thread_started(ev, thread_id)
                continue
            if (
                method
                not in {
                    "turn/started",
                    "item/started",
                    "item/completed",
                    *_DELTAS,
                    "thread/tokenUsage/updated",
                    "thread/status/changed",
                    "turn/completed",
                    "model/rerouted",
                    "error",
                }
                and ev.get("threadId", ev.get("thread_id")) is None
                and ev.get("turnId", ev.get("turn_id")) is None
                and not isinstance(ev.get("turn"), Mapping)
            ):
                raise ProtocolError(f"unknown app-server notification: {method}")
            observed = self._correlate_event(method, ev, thread_id, turn_id)
            if turn_id is None and observed is not None:
                if pending is not None and observed != pending:
                    raise ProtocolError("foreign turn event before turn/start response")
                pending = observed
            if method == "model/rerouted":
                raise IsolationError("model reroute is forbidden")
            if method == "turn/started":
                t, _ = self._validated_turn(ev, source="turn/started")
                if t.get("status") != "inProgress":
                    raise ProtocolError("turn/started returned non-running turn")
            elif method == "item/started":
                self._start_item(ev)
            elif method in _DELTAS:
                self._correlate_item_delta(
                    ev, "agentMessage" if method == "item/agentMessage/delta" else "reasoning"
                )
            elif method == "item/completed":
                item = self._complete_item(ev)
                if (
                    item.get("type") == "agentMessage"
                    and item.get("phase") in {"final_answer", None}
                    and isinstance(item.get("text"), str)
                ):
                    final = cast(str, item["text"])
                    final_item = cast(str, item["id"])
                    if self.logger:
                        self.logger.text("response.md", final)
            elif method == "thread/tokenUsage/updated":
                token = ev.get("tokenUsage")
                last = token.get("last") if isinstance(token, Mapping) else None
                if isinstance(last, Mapping):
                    usage_raw = dict(last)
                    self._last_usage_raw = usage_raw
            elif method == "thread/status/changed":
                status = ev.get("status")
                typ = status.get("type") if isinstance(status, Mapping) else status
                if typ in {"systemError", "failed", "interrupted", "cancelled"}:
                    raise TurnError(f"terminal turn status: {typ}")
            elif method == "turn/completed":
                t, ids = self._validated_turn(ev, source="turn/completed")
                if t.get("status") != "completed":
                    raise TurnError(f"turn ended with status {t.get('status')!r}")
                duration_ms = t.get("durationMs")
                if duration_ms is not None and (
                    not isinstance(duration_ms, int)
                    or isinstance(duration_ms, bool)
                    or duration_ms < 0
                ):
                    raise ProtocolError("turn/completed returned invalid durationMs")
                turn_duration_ms = duration_ms
                if self._active and not (
                    allow_completed_reasoning
                    and final is not None
                    and set(self._active.values()) == {"reasoning"}
                ):
                    raise ProtocolError("turn completed with active items")
                self._active.clear()
                if (
                    t.get("itemsView") != "notLoaded"
                    and final_item is not None
                    and final_item not in ids
                ):
                    raise ProtocolError("final agent item is absent from completed turn")
                terminal = True
                if turn_id is not None:
                    break
            elif method == "error":
                if not isinstance(ev.get("error"), Mapping) or not isinstance(
                    ev.get("willRetry"), bool
                ):
                    raise ProtocolError("malformed app-server error")
                if ev["willRetry"]:
                    if allow_server_retry:
                        self._server_retries += 1
                        continue
                    raise IsolationError("server retry is forbidden")
                raise TurnError("terminal app-server error")
            else:
                raise ProtocolError(f"unknown app-server notification: {method}")
        if not terminal:
            raise TurnError("turn timed out")
        if turn_id is None:
            raise ProtocolError("turn completion had no correlated turn id")
        end = time.monotonic() + self.limits.usage_grace
        while time.monotonic() < end:
            msg = self._read_message(end)
            if msg is None:
                break
            if "id" in msg:
                self._deny_server_request(msg)
            method, ev = msg.get("method"), msg.get("params")
            if method in _GLOBAL and isinstance(ev, Mapping):
                self._consume_global_notification(method, ev)
                continue
            if method == "thread/tokenUsage/updated" and isinstance(ev, Mapping):
                self._correlate_event(method, ev, thread_id, turn_id)
                tok = ev.get("tokenUsage")
                last = tok.get("last") if isinstance(tok, Mapping) else None
                if isinstance(last, Mapping):
                    usage_raw = dict(last)
                    self._last_usage_raw = usage_raw
            elif method == "thread/status/changed" and isinstance(ev, Mapping):
                self._correlate_event(method, ev, thread_id, turn_id)
                status = ev.get("status")
                if not isinstance(status, Mapping) or not isinstance(status.get("type"), str):
                    raise ProtocolError("malformed thread status notification")
            elif (
                isinstance(ev, Mapping)
                and ev.get("threadId", ev.get("thread_id")) is None
                and ev.get("turnId", ev.get("turn_id")) is None
            ):
                self._record("post_turn_notification", method=method)
            else:
                raise ProtocolError("unexpected notification after turn completion")
        if final is None:
            raise TurnError("no final_answer item")
        if len(final.encode("utf-8")) > self.limits.response_limit:
            raise ProtocolError("structured response exceeds limit")
        usage = self._usage(usage_raw)
        completed_turns = self._completed_turn_ids.setdefault(thread_id, [])
        if turn_id in completed_turns:
            raise ProtocolError("duplicate completed turn identity")
        completed_turns.append(turn_id)
        return GenerationResult(
            final,
            usage,
            thread_id,
            cast(str | None, cast(Json, self._thread).get("sessionId")),
            turn_id,
            request_id,
            cast(str | None, cast(Json, self._thread).get("path")),
            tuple(self._diag),
            turn_duration_ms,
        )

    def _run_compaction(self, thread_id: str) -> CompactionResult:
        self._current_thread_id = thread_id
        self._current_turn_id = None
        self._last_usage_raw = None
        request_id = self._send("thread/compact/start", {"threadId": thread_id})
        response_received = False
        turn_id = None
        pending = None
        item_id = None
        item_completed = False
        terminal = False
        usage_raw = None
        duration_ms = None
        started = time.monotonic()
        deadline = started + self.limits.turn_timeout
        self._active.clear()
        self._completed.clear()
        while time.monotonic() < deadline:
            msg = self._read_message(deadline)
            if msg is None:
                if self._stdout_overflow:
                    raise ProtocolError("incoming message exceeds limit")
                if self._stderr_exceeded:
                    raise ProtocolError("output limit exceeded")
                if (
                    time.monotonic() >= deadline
                    and self._process is not None
                    and self._process.poll() is None
                ):
                    raise TurnError("compaction timed out")
                raise TurnError("app-server EOF before compaction completion")
            if "id" in msg and "method" in msg:
                self._deny_server_request(msg)
            if "id" in msg:
                if msg.get("id") != request_id:
                    self._deny_server_request(msg)
                elif "error" in msg:
                    raise TurnError("thread/compact/start failed")
                elif msg.get("result") != {}:
                    raise ProtocolError("thread/compact/start returned malformed response")
                else:
                    response_received = True
                    if terminal:
                        break
                continue
            method = msg.get("method")
            event = msg.get("params")
            if not isinstance(method, str) or not isinstance(event, Mapping):
                raise ProtocolError("malformed compaction notification")
            if self._consume_global_notification(method, event):
                continue
            if method == "warning":
                if (
                    not isinstance(event.get("message"), str)
                    or not event["message"]
                    or event.get("threadId") != thread_id
                ):
                    raise ProtocolError("malformed app-server warning")
                continue
            if method == "thread/compacted":
                self._correlate_thread_started(event, thread_id)
                continue
            if method not in {
                "turn/started",
                "item/started",
                "item/completed",
                "thread/tokenUsage/updated",
                "thread/status/changed",
                "turn/completed",
                "error",
            }:
                raise ProtocolError(f"unknown compaction notification: {method}")
            observed = self._correlate_event(method, event, thread_id, turn_id)
            if turn_id is None and observed is not None:
                if pending is not None and observed != pending:
                    raise ProtocolError("foreign compaction event before response")
                pending = observed
            if method == "turn/started":
                turn, _ = self._validated_turn(event, source="turn/started")
                if turn.get("status") != "inProgress":
                    raise ProtocolError("compaction turn started non-running")
                turn_id = cast(str, turn["id"])
                self._current_turn_id = turn_id
            elif method == "item/started":
                current_id, item_type, _ = self._item_payload(event)
                if item_type != "contextCompaction":
                    raise IsolationError(f"unsupported compaction item type: {item_type}")
                if item_id is not None or current_id in self._completed:
                    raise ProtocolError("duplicate compaction item")
                item_id = current_id
                self._active[current_id] = item_type
            elif method == "item/completed":
                item = self._complete_item(event)
                if item.get("type") != "contextCompaction" or item.get("id") != item_id:
                    raise ProtocolError("completed a different compaction item")
                item_completed = True
            elif method == "thread/tokenUsage/updated":
                token = event.get("tokenUsage")
                last = token.get("last") if isinstance(token, Mapping) else None
                if isinstance(last, Mapping):
                    usage_raw = dict(last)
                    self._last_usage_raw = usage_raw
            elif method == "thread/status/changed":
                status = event.get("status")
                status_type = status.get("type") if isinstance(status, Mapping) else status
                if status_type in {
                    "systemError",
                    "failed",
                    "interrupted",
                    "cancelled",
                }:
                    raise TurnError(f"terminal compaction status: {status_type}")
            elif method == "turn/completed":
                turn, ids = self._validated_turn(event, source="turn/completed")
                if turn.get("status") != "completed":
                    raise TurnError(f"compaction ended with status {turn.get('status')!r}")
                if not item_completed or item_id is None or self._active:
                    raise ProtocolError("compaction item did not complete")
                if turn.get("itemsView") != "notLoaded" and item_id not in ids:
                    raise ProtocolError("contextCompaction item absent from completed turn")
                reported_duration = turn.get("durationMs")
                if reported_duration is not None and (
                    not isinstance(reported_duration, int)
                    or isinstance(reported_duration, bool)
                    or reported_duration < 0
                ):
                    raise ProtocolError("compaction turn returned invalid durationMs")
                duration_ms = (
                    reported_duration
                    if reported_duration is not None
                    else round((time.monotonic() - started) * 1000)
                )
                turn_id = cast(str, turn["id"])
                self._current_turn_id = turn_id
                terminal = True
                if response_received:
                    break
            elif method == "error":
                if not isinstance(event.get("error"), Mapping) or not isinstance(
                    event.get("willRetry"), bool
                ):
                    raise ProtocolError("malformed app-server error")
                if event["willRetry"]:
                    continue
                raise TurnError("terminal compaction error")
        if not terminal or not response_received:
            raise TurnError("compaction timed out")
        if turn_id is None or item_id is None:
            raise ProtocolError("compaction completion lacked correlated identity")
        end = time.monotonic() + self.limits.usage_grace
        while time.monotonic() < end:
            msg = self._read_message(end)
            if msg is None:
                break
            if "id" in msg:
                self._deny_server_request(msg)
            method = msg.get("method")
            event = msg.get("params")
            if method in _GLOBAL and isinstance(event, Mapping):
                self._consume_global_notification(method, event)
                continue
            if method == "thread/tokenUsage/updated" and isinstance(event, Mapping):
                self._correlate_event(method, event, thread_id, turn_id)
                token = event.get("tokenUsage")
                last = token.get("last") if isinstance(token, Mapping) else None
                if isinstance(last, Mapping):
                    usage_raw = dict(last)
                    self._last_usage_raw = usage_raw
            elif method == "thread/status/changed" and isinstance(event, Mapping):
                self._correlate_event(method, event, thread_id, turn_id)
            elif method == "thread/compacted" and isinstance(event, Mapping):
                self._correlate_thread_started(event, thread_id)
            else:
                raise ProtocolError("unexpected notification after compaction")
        return CompactionResult(
            thread_id=thread_id,
            turn_id=turn_id,
            item_id=item_id,
            request_id=request_id,
            usage=self._usage(usage_raw) if usage_raw is not None else None,
            duration_ms=cast(int, duration_ms),
        )

    def _usage(self, raw: Mapping[str, Any] | None) -> TokenUsage:
        keys = (
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        )
        if raw is None or any(k not in raw for k in keys):
            raise TurnError("exact tokenUsage.last with totalTokens is required")
        vals = [
            raw.get(k, 0)
            for k in (
                "inputTokens",
                "cachedInputTokens",
                "cacheWriteInputTokens",
                "outputTokens",
                "reasoningOutputTokens",
            )
        ]
        if (
            any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in vals)
            or not isinstance(raw["totalTokens"], int)
            or isinstance(raw["totalTokens"], bool)
            or raw["totalTokens"] < 0
        ):
            raise TurnError("invalid token usage")
        return TokenUsage(
            vals[0],
            vals[1],
            vals[2],
            vals[3],
            vals[4],
            raw["totalTokens"],
            True,
            False,
            dict(raw),
        )

    @staticmethod
    def _correlate_thread_started(p: Mapping[str, Any], tid: str) -> None:
        got = p.get("threadId", p.get("thread_id"))
        t = p.get("thread")
        got = t.get("id") if got is None and isinstance(t, Mapping) else got
        if not isinstance(got, str) or not got:
            raise ProtocolError("thread/started does not contain a valid thread ID")
        if got != tid:
            raise ProtocolError("foreign thread/started event")

    def _correlate_event(
        self, m: str, p: Mapping[str, Any], tid: str, turn: str | None
    ) -> str | None:
        thread_id = p.get("threadId", p.get("thread_id"))
        if thread_id is not None and thread_id != tid:
            raise ProtocolError("missing or foreign thread event")
        if m == "thread/status/changed":
            status = p.get("status")
            if not isinstance(status, Mapping) or not isinstance(status.get("type"), str):
                raise ProtocolError("malformed thread status notification")
            return None
        got = p.get("turnId", p.get("turn_id"))
        t = p.get("turn")
        got = t.get("id") if got is None and isinstance(t, Mapping) else got
        if got is not None and (not isinstance(got, str) or not got):
            raise ProtocolError("invalid turn event")
        if turn is not None and got is not None and got != turn:
            raise ProtocolError("missing or foreign turn event")
        item = p.get("item")
        if isinstance(item, Mapping):
            if item.get("id") is not None and (
                not isinstance(item.get("id"), str) or not item.get("id")
            ):
                raise ProtocolError("invalid item id")
            if item.get("threadId", item.get("thread_id", tid)) != tid:
                raise ProtocolError("foreign item thread")
            if turn is not None and item.get("turnId", item.get("turn_id", turn)) != turn:
                raise ProtocolError("foreign item turn")
        iid = p.get("itemId", p.get("item_id"))
        if iid is not None and (not isinstance(iid, str) or not iid):
            raise ProtocolError("invalid item id")
        return got

    @staticmethod
    def _validated_turn(p: Mapping[str, Any], *, source: str) -> tuple[Mapping[str, Any], set[str]]:
        t = p.get("turn")
        if (
            not isinstance(t, Mapping)
            or not isinstance(t.get("id"), str)
            or not isinstance(t.get("status"), str)
            or not isinstance(t.get("items"), list)
        ):
            raise ProtocolError(f"{source} returned malformed turn")
        ids = set()
        for i in t["items"]:
            if (
                not isinstance(i, Mapping)
                or not isinstance(i.get("id"), str)
                or not isinstance(i.get("type"), str)
                or i["id"] in ids
            ):
                raise ProtocolError(f"{source} returned malformed item")
            ids.add(cast(str, i["id"]))
        return t, ids

    @staticmethod
    def _item_payload(p: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any]]:
        i = p.get("item")
        if (
            not isinstance(i, Mapping)
            or not isinstance(i.get("id"), str)
            or not isinstance(i.get("type"), str)
        ):
            raise ProtocolError("invalid item id")
        return cast(str, i["id"]), cast(str, i["type"]), i

    def _start_item(self, p: Mapping[str, Any]) -> None:
        iid, typ, _ = self._item_payload(p)
        if typ not in _PASSIVE:
            raise IsolationError(f"unsupported app-server item type: {typ}")
        if iid in self._active or iid in self._completed:
            raise ProtocolError("duplicate item/started")
        self._active[iid] = typ

    def _correlate_item_delta(self, p: Mapping[str, Any], typ: str) -> None:
        iid = p.get("itemId")
        if not isinstance(iid, str) or self._active.get(iid) != typ:
            raise ProtocolError("item delta does not match an active item")

    def _complete_item(self, p: Mapping[str, Any]) -> Mapping[str, Any]:
        iid, typ, item = self._item_payload(p)
        if self._active.get(iid) != typ:
            raise ProtocolError("item/completed does not match an active item")
        del self._active[iid]
        self._completed.add(iid)
        return item

    @staticmethod
    def _consume_global_notification(m: str, p: Mapping[str, Any]) -> bool:
        if m not in _GLOBAL:
            return False
        if m == "account/rateLimits/updated" and not isinstance(p.get("rateLimits"), Mapping):
            raise ProtocolError("malformed account rate-limit notification")
        if m == "configWarning":
            if not isinstance(p.get("summary"), str):
                raise ProtocolError("malformed config warning notification")
            for key in ("details", "path"):
                if key in p and p[key] is not None and not isinstance(p[key], str):
                    raise ProtocolError("malformed config warning notification")
            text_range = p.get("range")
            if text_range is not None:
                if not isinstance(text_range, Mapping):
                    raise ProtocolError("malformed config warning notification")
                for position_name in ("start", "end"):
                    position = text_range.get(position_name)
                    if not isinstance(position, Mapping):
                        raise ProtocolError("malformed config warning notification")
                    for coordinate in ("line", "column"):
                        value = position.get(coordinate)
                        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                            raise ProtocolError("malformed config warning notification")
        if m == "remoteControl/status/changed" and (
            p.get("status") not in {"disabled", "connecting", "connected", "errored"}
            or not isinstance(p.get("serverName"), str)
            or not isinstance(p.get("installationId"), str)
            or (p.get("environmentId") is not None and not isinstance(p.get("environmentId"), str))
        ):
            raise ProtocolError("malformed remote-control status notification")
        return True

    def _send(self, m: str, p: Json) -> int:
        if self._process is None or self._process.stdin is None:
            raise AppServerError("app-server is not running")
        rid = self._next_id
        self._next_id += 1
        message = {"id": rid, "method": m, "params": p}
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        if len(payload) > self.limits.request_limit:
            raise ProtocolError("outgoing request exceeds limit")
        if self._transcript_size + len(payload) > self.limits.transcript_limit:
            raise ProtocolError("outgoing transcript exceeds limit")
        if self.logger:
            self.logger.sent(message, payload)
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        self._transcript_size += len(payload)
        self._record("request", method=m, bytes=len(payload))
        return rid

    def _notify(self, m: str, p: Json) -> None:
        if self._process is None or self._process.stdin is None:
            raise AppServerError("app-server is not running")
        if m not in _SETUP_NOTIFICATIONS or p != {}:
            raise ProtocolError("unsupported or malformed setup notification")
        message = {"method": m, "params": p}
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        if (
            len(payload) > self.limits.message_limit
            or self._transcript_size + len(payload) > self.limits.transcript_limit
        ):
            raise ProtocolError("outgoing notification exceeds limit")
        if self.logger:
            self.logger.sent(message, payload)
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        self._transcript_size += len(payload)

    def _request(self, m: str, p: Json, *, timeout: float | None = None) -> Json:
        rid = self._send(m, p)
        end = time.monotonic() + (timeout if timeout is not None else self.limits.turn_timeout)
        while time.monotonic() < end:
            msg = self._read_message(end)
            if msg is None:
                raise ProtocolError("app-server EOF")
            if "id" in msg and "method" in msg:
                self._deny_server_request(msg)
            if msg.get("id") == rid:
                if "error" in msg:
                    raise ProtocolError(f"request {m} failed")
                return cast(
                    Json, msg.get("result") if isinstance(msg.get("result"), Mapping) else {}
                )
            if "id" in msg:
                self._deny_server_request(msg)
            method = msg.get("method")
            params = msg.get("params")
            if not isinstance(method, str) or not isinstance(params, Mapping):
                raise ProtocolError(f"malformed notification while waiting for {m}")
            if method not in _WAITING_NOTIFICATIONS:
                raise ProtocolError(f"unsupported notification while waiting for {m}")
            if method in _GLOBAL:
                self._consume_global_notification(method, params)
            elif m == "thread/fork" and method in {
                "thread/started",
                "thread/tokenUsage/updated",
            }:
                self._validate_fork_waiting_notification(method, params, p)
            elif m == "thread/resume" and method in {
                "thread/status/changed",
                "thread/tokenUsage/updated",
            }:
                self._validate_resume_waiting_notification(method, params, p)
            elif method in {
                "thread/status/changed",
                "thread/tokenUsage/updated",
            }:
                raise ProtocolError(f"unsupported notification while waiting for {m}")
            if m == "thread/start" and method == "thread/started":
                raise ProtocolError("thread/started arrived before thread/start response")
            # The installed app-server may emit config, remote-control, account,
            # and thread notifications between an RPC request and its response.
            # The working TheML client records these notifications and keeps
            # waiting for the correlated response. Server requests remain
            # rejected above, and active-turn authority is checked in _run_turn.
            self._record("notification_while_waiting", request=m, method=method)
            continue
        raise ProtocolError(f"timeout waiting for {m}")

    def _validate_resume_waiting_notification(
        self,
        method: str,
        params: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> None:
        expected_thread_id = request.get("threadId")
        thread_id = params.get("threadId")
        if (
            not isinstance(expected_thread_id, str)
            or not expected_thread_id
            or thread_id != expected_thread_id
        ):
            raise ProtocolError("missing or foreign thread/resume notification")
        if method == "thread/status/changed":
            status = params.get("status")
            status_type = status.get("type") if isinstance(status, Mapping) else None
            if not isinstance(status_type, str) or not status_type:
                raise ProtocolError("malformed thread status during thread/resume")
            if status_type in {
                "systemError",
                "failed",
                "interrupted",
                "cancelled",
            }:
                raise TurnError(f"terminal thread/resume status: {status_type}")
            return
        token_usage = params.get("tokenUsage")
        turn_id = params.get("turnId")
        last = token_usage.get("last") if isinstance(token_usage, Mapping) else None
        total = token_usage.get("total") if isinstance(token_usage, Mapping) else None
        if (
            not isinstance(turn_id, str)
            or not turn_id
            or not isinstance(last, Mapping)
            or not isinstance(total, Mapping)
        ):
            raise ProtocolError("malformed token usage during thread/resume")
        self._usage(last)
        self._usage(total)

    def _validate_fork_waiting_notification(
        self,
        method: str,
        params: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> None:
        source_thread_id = request.get("threadId")
        if not isinstance(source_thread_id, str) or not source_thread_id:
            raise ProtocolError("fork notification has no source thread")
        if method == "thread/tokenUsage/updated":
            thread_id = params.get("threadId")
            turn_id = params.get("turnId")
            token_usage = params.get("tokenUsage")
            last = token_usage.get("last") if isinstance(token_usage, Mapping) else None
            total = token_usage.get("total") if isinstance(token_usage, Mapping) else None
            if (
                not isinstance(thread_id, str)
                or not thread_id
                or not isinstance(turn_id, str)
                or not turn_id
                or not isinstance(last, Mapping)
                or not isinstance(total, Mapping)
            ):
                raise ProtocolError("malformed token usage while waiting for fork")
            self._usage(last)
            self._usage(total)
            known_turns = set(self._completed_turn_ids.get(source_thread_id, []))
            known_turns.update(self._completed_turn_ids.get(thread_id, []))
            if turn_id not in known_turns:
                raise ProtocolError("foreign token usage while waiting for fork")
            return
        thread = params.get("thread")
        if (
            not isinstance(thread, Mapping)
            or not isinstance(thread.get("id"), str)
            or not thread["id"]
            or thread.get("forkedFromId") != source_thread_id
            or thread.get("ephemeral") is not False
            or thread.get("cwd") != str(self.capsule.workdir)
        ):
            raise ProtocolError("malformed thread/started while waiting for fork")
        raw_path = thread.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ProtocolError("fork thread/started has no absolute path")
        try:
            Path(raw_path).resolve(strict=False).relative_to(self.capsule.root.resolve(strict=True))
        except ValueError as exc:
            raise IsolationError("fork thread/started path escapes capsule") from exc
        known = self._forked_threads.get(cast(str, thread["id"]))
        if known is not None and any(
            thread.get(key) != known.get(key) for key in ("forkedFromId", "id", "path", "sessionId")
        ):
            raise ProtocolError("late fork thread/started changed child identity")

    def _drain_fork_notifications(self, source_thread_id: str) -> None:
        """Consume bounded notifications emitted just after a fork response."""

        end = time.monotonic() + min(1.0, self.limits.usage_grace)
        request = {"threadId": source_thread_id}
        while time.monotonic() < end:
            msg = self._read_message(end)
            if msg is None:
                return
            if "id" in msg:
                raise ProtocolError("unexpected response after thread/fork")
            method = msg.get("method")
            params = msg.get("params")
            if not isinstance(method, str) or not isinstance(params, Mapping):
                raise ProtocolError("malformed notification after thread/fork")
            if method in _GLOBAL:
                self._consume_global_notification(method, params)
            elif method in {
                "thread/started",
                "thread/tokenUsage/updated",
            }:
                self._validate_fork_waiting_notification(method, params, request)
            else:
                raise ProtocolError(f"unsupported notification after thread/fork: {method}")
            self._record("notification_after_fork", method=method)

    def _drain_resume_notifications(self, thread_id: str) -> None:
        """Consume bounded history notifications emitted after a resume response."""

        end = time.monotonic() + min(1.0, self.limits.usage_grace)
        request = {"threadId": thread_id}
        while time.monotonic() < end:
            msg = self._read_message(end)
            if msg is None:
                return
            if "id" in msg:
                raise ProtocolError("unexpected response after thread/resume")
            method = msg.get("method")
            params = msg.get("params")
            if not isinstance(method, str) or not isinstance(params, Mapping):
                raise ProtocolError("malformed notification after thread/resume")
            if method in _GLOBAL:
                self._consume_global_notification(method, params)
            elif method in {
                "thread/status/changed",
                "thread/tokenUsage/updated",
            }:
                self._validate_resume_waiting_notification(method, params, request)
            else:
                raise ProtocolError(f"unsupported notification after thread/resume: {method}")
            self._record("notification_after_resume", method=method)

    def _read_message(self, end: float) -> Json | None:
        if self._stderr_exceeded or self._stdout_overflow:
            if self._stdout_overflow:
                raise ProtocolError("incoming message exceeds limit")
            raise ProtocolError("output limit exceeded")
        try:
            v = self._queue.get(timeout=max(0.0, end - time.monotonic()))
        except queue.Empty:
            return None
        if v is not None and not isinstance(v, BaseException):
            size = len(v.encode() if isinstance(v, str) else bytes(v))
            with self._queue_lock:
                self._queued_bytes = max(0, self._queued_bytes - size)
        if v is None or isinstance(v, BaseException):
            if self._stdout_overflow:
                raise ProtocolError("incoming message exceeds limit")
            return None
        raw = v.encode() if isinstance(v, str) else bytes(v)
        self._stdout_lines.append(raw)
        self._stdout_size += len(raw)
        if len(raw) > self.limits.message_limit or self._stdout_size > self.limits.stdout_limit:
            raise ProtocolError("incoming message exceeds limit")
        if self._transcript_size + len(raw) > self.limits.transcript_limit:
            raise ProtocolError("bidirectional transcript exceeds limit")
        self._transcript_size += len(raw)
        self._event_count += 1
        if self._event_count > self.limits.max_events:
            raise ProtocolError("event limit exceeded")
        try:
            msg = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("malformed JSONL") from exc
        if not isinstance(msg, dict):
            raise ProtocolError("JSON-RPC message must be an object")
        if self.logger:
            try:
                self.logger.message(msg, raw)
                self.logger.text(
                    "stdout.jsonl", b"".join(self._stdout_lines).decode("utf-8", "replace")
                )
            except ValueError as exc:
                raise ProtocolError(str(exc)) from exc
        return cast(Json, msg)

    def _deny_server_request(self, msg: Json) -> None:
        if self._process is None or self._process.stdin is None:
            raise ProtocolError("malformed server request")
        response = {
            "id": msg.get("id"),
            "error": {"code": -32601, "message": "server requests are disabled"},
        }
        payload = (json.dumps(response, separators=(",", ":")) + "\n").encode()
        if (
            len(payload) > self.limits.response_limit
            or self._transcript_size + len(payload) > self.limits.transcript_limit
        ):
            raise ProtocolError("denied server response exceeds limit")
        if self.logger:
            self.logger.sent(response, payload)
        self._process.stdin.write(payload)
        self._process.stdin.flush()
        self._transcript_size += len(payload)
        self._record("denied_server_request", method=msg.get("method"))
        raise ProtocolError("unsupported server request")

    def inspect_metadata(self) -> Mapping[str, Any]:
        return {
            "threadId": self._thread.get("id") if self._thread else None,
            "sessionId": self._thread.get("sessionId") if self._thread else None,
            "threadPath": self._thread.get("path") if self._thread else None,
            "status": self._last_status,
            "turns": self._turns,
            "serverRetries": self._server_retries,
            "serverWarnings": self._server_warnings,
            "usage_final": self._last_status == "completed" and self._last_usage_raw is not None,
            "usage_partial": self._last_status != "completed" and self._last_usage_raw is not None,
        }

    def inspect_usage(self) -> Mapping[str, Any]:
        return {
            "final": self._last_status == "completed" and self._last_usage_raw is not None,
            "partial": self._last_status != "completed" and self._last_usage_raw is not None,
            "raw": dict(self._last_usage_raw) if self._last_usage_raw is not None else None,
        }

    def experimental_turn_identity(self) -> tuple[str | None, str | None]:
        """Expose the correlated identity needed by an opt-in experiment report."""

        return self._current_thread_id, self._current_turn_id

    def flush(self) -> None:
        if self._process and self._process.stdin:
            with suppress(Exception):
                self._process.stdin.flush()

    def close(self, *, force: bool = False) -> None:
        p, self._process = self._process, None
        if p is None:
            if self._owns_capsule:
                self.capsule.cleanup()
            return
        self._stop.set()
        try:
            if p.poll() is None:
                self._signal_process_group(p, signal.SIGKILL if force else signal.SIGTERM)
                p.wait(timeout=1.0)
        except Exception:
            with suppress(Exception):
                self._signal_process_group(p, signal.SIGKILL)
                p.wait(timeout=1.0)
        # Closing the pipes releases reader threads blocked in readline; join
        # them before copying evidence or removing the capsule.
        for s in (p.stdin, p.stdout, p.stderr):
            with suppress(Exception):
                s.close()
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
        self._reader_thread = None
        self._stderr_thread = None
        self._drain_evidence()
        if self.copy_rollout_artifact:
            self._copy_rollout()
        if self.logger:
            self.logger.cleanup_temporary_files()
        if self._owns_capsule:
            self.capsule.cleanup()

    def _copy_rollout(self) -> None:
        """Copy only the server-returned opaque rollout path, never derive one."""
        if not self.logger or not self._thread:
            return
        raw = self._thread.get("path")
        if (
            not isinstance(raw, str)
            or not raw
            or Path(raw).name.lower() in {"auth.json", "credentials.json"}
        ):
            return
        try:
            path = Path(raw)
            if not path.is_absolute():
                return
            path = path.resolve(strict=True)
            capsule_root = self.capsule.root.resolve(strict=True)
            try:
                path.relative_to(capsule_root)
            except ValueError:
                return
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size > self.limits.transcript_limit
            ):
                return
            self.logger.text("rollout.jsonl", path.read_bytes().decode("utf-8", "replace"))
        except (OSError, ValueError):
            return

    @staticmethod
    def _signal_process_group(p: _Proc, sig: signal.Signals) -> None:
        pid = getattr(p, "pid", None)
        if isinstance(pid, int) and pid > 0 and hasattr(os, "killpg"):
            with suppress(ProcessLookupError):
                os.killpg(pid, sig)
        elif sig == signal.SIGKILL:
            p.kill()
        else:
            p.terminate()

    def __enter__(self) -> CodexAppServerAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AppServerGenerationProvider:
    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
        auth_checker: AuthChecker | None = None,
        auth_json: str | Path | None = None,
        limits: AppServerLimits | None = None,
        artifact_dir: str | Path | None = None,
        artifact_prefix: str = "",
        artifact_max_bytes: int = 32 * 1024 * 1024,
        sandbox_mode: str = "danger-full-access",
        approval_policy: str = "never",
    ):
        self.process_factory = process_factory
        self.auth_checker = auth_checker
        self.auth_json = auth_json
        self.limits = limits or AppServerLimits()
        self.artifact_dir = artifact_dir
        self.artifact_prefix = artifact_prefix
        self.artifact_max_bytes = artifact_max_bytes
        if sandbox_mode not in APP_SERVER_SANDBOX_MODES:
            raise ValueError("unsupported app-server sandbox mode")
        if approval_policy not in APP_SERVER_APPROVAL_POLICIES:
            raise ValueError("unsupported app-server approval policy")
        self.sandbox_mode = sandbox_mode
        self.approval_policy = approval_policy

    @staticmethod
    def _usage(u: TokenUsage) -> dict[str, Any]:
        return {
            "inputTokens": u.input_tokens,
            "cachedInputTokens": u.cached_input_tokens,
            "cacheWriteInputTokens": u.cache_write_input_tokens,
            "outputTokens": u.output_tokens,
            "reasoningOutputTokens": u.reasoning_output_tokens,
            "totalTokens": u.total_tokens,
            "final": u.final,
            "partial": u.partial,
            "raw": dict(u.raw),
        }

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = request.get("prompt")
        system = request.get("system_prompt")
        schema = request.get("output_schema")
        model = request.get("model")
        effort = request.get("effort")
        if (
            not isinstance(prompt, str)
            or not prompt
            or not isinstance(system, str)
            or not system
            or not isinstance(schema, Mapping)
            or not isinstance(model, str)
            or not isinstance(effort, str)
        ):
            raise ValueError("invalid generation request")
        if model != FROZEN_STAGE3_MODEL or effort != FROZEN_STAGE3_EFFORT:
            raise IsolationError(
                "Stage 3 generation requires the frozen "
                f"{FROZEN_STAGE3_MODEL}:{FROZEN_STAGE3_EFFORT} profile"
            )
        artifact_dir = request.get("artifact_dir", self.artifact_dir)
        artifact_prefix = str(request.get("artifact_prefix", self.artifact_prefix))
        artifact_root = request.get("artifact_root")
        protocol_audit_sha256 = request.get("appserver_doctor_sha256")
        ad = CodexAppServerAdapter(
            process_factory=self.process_factory,
            auth_checker=self.auth_checker,
            auth_json=self.auth_json,
            limits=self.limits,
            base_instructions=system,
            artifact_dir=artifact_dir,
            artifact_prefix=artifact_prefix,
            artifact_root=artifact_root if isinstance(artifact_root, (str, Path)) else None,
            artifact_max_bytes=self.artifact_max_bytes,
            protocol_audit_sha256=(
                protocol_audit_sha256 if isinstance(protocol_audit_sha256, str) else None
            ),
            sandbox_mode=self.sandbox_mode,
            approval_policy=self.approval_policy,
        )
        try:
            if ad.logger:
                ad.logger.text("request.md", prompt)
                ad.logger.document(
                    "request.json",
                    {
                        "model": model,
                        "reasoning_effort": effort,
                        "prompt": prompt,
                        "system_prompt": system,
                        "output_schema": dict(schema),
                        "appserver_doctor_sha256": protocol_audit_sha256,
                    },
                )
            r = ad.generate(prompt, ModelProfile("codex", model, effort), output_schema=schema)
            result = {
                "response": r.text,
                "accepted": True,
                "charged": r.usage.total_tokens > 0,
                "content": bool(r.text),
                "usage": self._usage(r.usage),
                "status": "completed",
                "thread_id": r.thread_id,
                "session_id": r.session_id,
                "turn_id": r.turn_id,
                "request_id": r.request_id,
                "model": model,
                "effort": effort,
                "transport_sha256": ad.logger.transcript_sha256 if ad.logger else None,
                "appserver_doctor_sha256": protocol_audit_sha256,
            }
            if ad.logger:
                ad.logger.text("response.md", r.text)
                ad.logger.document("response.json", result)
                ad.logger.document(
                    "provider-raw.json",
                    {
                        "diagnostics": list(r.diagnostics),
                        "thread_id": r.thread_id,
                        "session_id": r.session_id,
                        "turn_id": r.turn_id,
                        "request_id": r.request_id,
                        "thread_path": r.thread_path,
                        "usage": self._usage(r.usage),
                        "transport_sha256": (ad.logger.transcript_sha256 if ad.logger else None),
                        "appserver_doctor_sha256": protocol_audit_sha256,
                    },
                )
            return result
        except Exception as error:
            if ad.logger:
                ad.logger.document(
                    "response.json",
                    {
                        "accepted": False,
                        "content": False,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error)[:512],
                        "metadata": dict(ad.inspect_metadata()),
                        "usage": dict(ad.inspect_usage()),
                        "diagnostics": list(ad.diagnostics),
                        "transport_sha256": (ad.logger.transcript_sha256 if ad.logger else None),
                        "appserver_doctor_sha256": protocol_audit_sha256,
                    },
                )
            raise
        finally:
            ad.close()

    def repair(
        self, request: Mapping[str, Any], diagnostics: tuple[Mapping[str, Any], ...]
    ) -> Mapping[str, Any]:
        if not diagnostics:
            raise ValueError("repair requires bounded diagnostics")
        p = request.get("prompt")
        if not isinstance(p, str):
            raise ValueError("repair request prompt must be a string")
        prefix = str(request.get("artifact_prefix", self.artifact_prefix)).rstrip(".")
        return self.generate(
            {
                **request,
                "artifact_prefix": f"{prefix}.repair" if prefix else "repair",
                "prompt": p + "\n\nRepair only the output schema or Python AST/runtime "
                "violations listed below.\n"
                + json.dumps(list(diagnostics), sort_keys=True, separators=(",", ":")),
            }
        )
