from __future__ import annotations

import argparse
import builtins
import json
import math
import os
import resource
import selectors
import signal
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import BinaryIO, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import (
    ContractError,
    RankerContext,
    RankerProposal,
    SandboxLimits,
    freeze_plain_data,
    validate_priority,
    validate_ranker_inputs,
)
from mutation_forge.sandbox.errors import (
    ProtocolError,
    UnsupportedPlatformError,
    WorkerCrashError,
    WorkerTimeoutError,
)
from mutation_forge.sandbox.validation import SAFE_BUILTINS, validate_policy

_HEADER = struct.Struct("!I")
_PROTOCOL_VERSION = "stage2a.worker.v1"


@dataclass(frozen=True, slots=True)
class WorkerCallResult:
    status: str
    priority: int | float | None
    elapsed_ns: int
    error: dict[str, JsonValue] | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "status": self.status,
            "priority": self.priority,
            "elapsed_ns": self.elapsed_ns,
            "error": self.error,
        }


def _encode_frame(payload: object, limit: int) -> bytes:
    try:
        body = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ProtocolError(f"protocol value is not JSON-compatible: {error}") from error
    if len(body) > limit:
        raise ProtocolError(f"protocol frame of {len(body)} bytes exceeds {limit}")
    return _HEADER.pack(len(body)) + body


def _child_read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("protocol stream closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _child_read_frame(stream: BinaryIO, limit: int) -> dict[str, object]:
    length = _HEADER.unpack(_child_read_exact(stream, _HEADER.size))[0]
    if length > limit:
        raise ProtocolError(f"request frame length {length} exceeds {limit}")
    try:
        payload = json.loads(_child_read_exact(stream, length))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid request JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ProtocolError("protocol request must be a JSON object")
    return cast(dict[str, object], payload)


def _child_write_frame(stream: BinaryIO, payload: object, limit: int) -> None:
    stream.write(_encode_frame(payload, limit))
    stream.flush()


def _safe_globals() -> dict[str, object]:
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    return {"__builtins__": safe}


def _control_record() -> dict[str, JsonValue]:
    return {
        "cwd": os.getcwd(),
        "environment_keys": cast(list[JsonValue], sorted(os.environ)),
        "stdin_mode": "protocol_pipe",
        "process_group_isolated": os.getpgrp() == os.getpid(),
        "rlimits": {
            "cpu": list(resource.getrlimit(resource.RLIMIT_CPU)),
            "address_space": list(resource.getrlimit(resource.RLIMIT_AS)),
            "file_size": list(resource.getrlimit(resource.RLIMIT_FSIZE)),
            "open_files": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
            "process_count": list(resource.getrlimit(resource.RLIMIT_NPROC)),
        },
    }


def _child_main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        initialize = _child_read_frame(stdin, 64 * 1024)
        if initialize.get("type") != "initialize":
            raise ProtocolError("first worker request must be initialize")
        source = initialize.get("source")
        limits_raw = initialize.get("limits")
        if not isinstance(source, str) or not isinstance(limits_raw, dict):
            raise ProtocolError("initialize requires source and limits")
        limits = SandboxLimits(
            **{
                key: value
                for key, value in limits_raw.items()
                if key in SandboxLimits.__dataclass_fields__
            }
        )
        validation = validate_policy(source, limits)
        if not validation.valid:
            _child_write_frame(
                stdout,
                {
                    "type": "ready",
                    "status": "invalid",
                    "validation": validation.as_dict(),
                },
                limits.response_bytes,
            )
            return 2
        namespace = _safe_globals()
        exec(compile(source, "<policy>", "exec"), namespace, namespace)
        function = namespace.get("priority")
        if not callable(function):
            raise ProtocolError("validated policy did not define priority")
        _child_write_frame(
            stdout,
            {
                "type": "ready",
                "status": "ok",
                "protocol_version": _PROTOCOL_VERSION,
                "identity": validation.identity.as_dict(),
                "controls": _control_record(),
            },
            limits.response_bytes,
        )
        while True:
            request = _child_read_frame(stdin, limits.request_bytes)
            request_type = request.get("type")
            if request_type == "shutdown":
                _child_write_frame(
                    stdout,
                    {"type": "shutdown", "status": "ok"},
                    limits.response_bytes,
                )
                return 0
            if request_type != "call":
                raise ProtocolError(f"unexpected request type: {request_type!r}")
            started_ns = time.perf_counter_ns()
            try:
                ctx, proposal = validate_ranker_inputs(
                    request.get("ctx"),
                    request.get("proposal"),
                    max_request_bytes=limits.request_bytes,
                )
                output = function(
                    freeze_plain_data(cast(JsonValue, ctx)),
                    freeze_plain_data(cast(JsonValue, proposal)),
                )
                priority = validate_priority(
                    output,
                    max_response_bytes=limits.response_bytes,
                )
                response: dict[str, object] = {
                    "type": "result",
                    "status": "ok",
                    "priority": priority,
                    "elapsed_ns": time.perf_counter_ns() - started_ns,
                }
            except BaseException as error:
                detail: dict[str, object]
                if isinstance(error, ContractError):
                    detail = cast(dict[str, object], error.as_dict())
                else:
                    detail = {
                        "code": "policy_exception",
                        "message": str(error)[:1024],
                        "error_type": type(error).__name__,
                    }
                response = {
                    "type": "result",
                    "status": "exception",
                    "priority": None,
                    "elapsed_ns": time.perf_counter_ns() - started_ns,
                    "error": detail,
                }
            _child_write_frame(stdout, response, limits.response_bytes)
    except (EOFError, BrokenPipeError):
        return 1
    except BaseException:
        return 3


def _require_linux_limits() -> None:
    required = (
        "RLIMIT_CPU",
        "RLIMIT_AS",
        "RLIMIT_FSIZE",
        "RLIMIT_NOFILE",
        "RLIMIT_NPROC",
    )
    if sys.platform != "linux" or any(not hasattr(resource, name) for name in required):
        raise UnsupportedPlatformError(
            "Stage 2A workers require Linux RLIMIT_CPU/AS/FSIZE/NOFILE/NPROC"
        )


def _limit_child(limits: SandboxLimits) -> None:
    os.setsid()
    os.umask(0o077)
    cpu_soft = max(1, math.ceil(limits.total_wall_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
    resource.setrlimit(
        resource.RLIMIT_AS,
        (limits.address_space_bytes, limits.address_space_bytes),
    )
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (limits.captured_output_bytes, limits.captured_output_bytes),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
    resource.setrlimit(
        resource.RLIMIT_NPROC,
        (limits.process_count, limits.process_count),
    )


class PolicyWorker:
    """One persistent, fail-closed subprocess for one validated policy."""

    def __init__(
        self,
        source: str,
        limits: SandboxLimits | None = None,
    ) -> None:
        _require_linux_limits()
        self.limits = limits or SandboxLimits()
        validation = validate_policy(source, self.limits)
        if not validation.valid:
            raise ValueError(json.dumps(validation.as_dict(), sort_keys=True))
        self.identity = validation.identity
        self._started = time.monotonic()
        self._failed = False
        self._closed = False
        self._calls = 0
        self._failures = 0
        self._total_elapsed_ns = 0
        self._max_elapsed_ns = 0
        self._controls: dict[str, JsonValue] = {}
        self._temporary = tempfile.TemporaryDirectory(prefix="mforge-policy-")
        self._stderr = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        environment = {
            "HOME": self._temporary.name,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
        }
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-u",
                    "-m",
                    "mutation_forge.sandbox.worker",
                    "--child",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                cwd=self._temporary.name,
                env=environment,
                close_fds=True,
                preexec_fn=lambda: _limit_child(self.limits),
            )
        except BaseException:
            self._stderr.close()
            self._temporary.cleanup()
            raise
        if self._process.stdin is None or self._process.stdout is None:
            self._terminate()
            raise WorkerCrashError("worker protocol pipes were not created")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        try:
            self._send(
                {
                    "type": "initialize",
                    "source": source,
                    "limits": self.limits.as_dict(),
                },
                self.limits.request_bytes,
            )
            ready = self._receive(min(5.0, self.limits.total_wall_seconds))
            if (
                ready.get("type") != "ready"
                or ready.get("status") != "ok"
                or ready.get("identity") != self.identity.as_dict()
            ):
                raise ProtocolError(f"worker initialization failed: {ready}")
            controls = ready.get("controls")
            if not isinstance(controls, dict) or not self._controls_match(controls):
                raise ProtocolError("worker resource/isolation controls did not match")
            self._controls = cast(dict[str, JsonValue], controls)
        except BaseException:
            self._failed = True
            self._terminate()
            self._closed = True
            self._stderr.close()
            self._temporary.cleanup()
            raise

    @property
    def usable(self) -> bool:
        return not self._failed and not self._closed and self._process.poll() is None

    def _controls_match(self, controls: dict[object, object]) -> bool:
        cpu_soft = max(1, math.ceil(self.limits.total_wall_seconds))
        expected_rlimits = {
            "cpu": [cpu_soft, cpu_soft + 1],
            "address_space": [
                self.limits.address_space_bytes,
                self.limits.address_space_bytes,
            ],
            "file_size": [
                self.limits.captured_output_bytes,
                self.limits.captured_output_bytes,
            ],
            "open_files": [self.limits.open_files, self.limits.open_files],
            "process_count": [
                self.limits.process_count,
                self.limits.process_count,
            ],
        }
        environment_keys = controls.get("environment_keys")
        return (
            controls.get("cwd") == self._temporary.name
            and controls.get("stdin_mode") == "protocol_pipe"
            and controls.get("process_group_isolated") is True
            and controls.get("rlimits") == expected_rlimits
            and isinstance(environment_keys, list)
            and set(environment_keys).issubset({"HOME", "LANG", "LC_ALL", "PATH"})
        )

    def _send(self, payload: object, limit: int) -> None:
        frame = _encode_frame(payload, limit)
        try:
            self._stdin.write(frame)
            self._stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise WorkerCrashError("worker closed its request pipe") from error

    def _read_exact(self, size: int, deadline: float) -> bytes:
        chunks = bytearray()
        selector = selectors.DefaultSelector()
        selector.register(self._stdout, selectors.EVENT_READ)
        try:
            while len(chunks) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerTimeoutError("worker response exceeded wall limit")
                if not selector.select(remaining):
                    raise WorkerTimeoutError("worker response exceeded wall limit")
                chunk = os.read(self._stdout.fileno(), size - len(chunks))
                if not chunk:
                    code = self._process.poll()
                    raise WorkerCrashError(f"worker exited before a response (code={code})")
                chunks.extend(chunk)
        finally:
            selector.close()
        return bytes(chunks)

    def _receive(self, wall_seconds: float) -> dict[str, object]:
        deadline = time.monotonic() + wall_seconds
        header = self._read_exact(_HEADER.size, deadline)
        length = _HEADER.unpack(header)[0]
        if length > self.limits.response_bytes:
            raise ProtocolError(
                f"worker response frame length {length} exceeds "
                f"{self.limits.response_bytes}"
            )
        body = self._read_exact(length, deadline)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError(f"worker returned invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ProtocolError("worker response must be a JSON object")
        return cast(dict[str, object], payload)

    def call(
        self,
        ctx: RankerContext,
        proposal: RankerProposal,
    ) -> WorkerCallResult:
        if not self.usable:
            raise WorkerCrashError("failed or closed workers cannot be reused")
        elapsed_total = time.monotonic() - self._started
        total_remaining = self.limits.total_wall_seconds - elapsed_total
        if total_remaining <= 0:
            self._failed = True
            self._terminate()
            raise WorkerTimeoutError("worker exceeded total wall limit")
        normalized_ctx, normalized_proposal = validate_ranker_inputs(
            ctx,
            proposal,
            max_request_bytes=self.limits.request_bytes,
        )
        try:
            self._send(
                {
                    "type": "call",
                    "ctx": normalized_ctx,
                    "proposal": normalized_proposal,
                },
                self.limits.request_bytes,
            )
            response = self._receive(
                min(self.limits.per_call_wall_seconds, total_remaining)
            )
            if response.get("type") != "result":
                raise ProtocolError(f"unexpected worker response: {response}")
            status = response.get("status")
            elapsed_ns = response.get("elapsed_ns")
            if (
                status not in {"ok", "exception"}
                or not isinstance(elapsed_ns, int)
                or isinstance(elapsed_ns, bool)
                or elapsed_ns < 0
            ):
                raise ProtocolError(f"malformed worker result: {response}")
            priority_raw = response.get("priority")
            priority = (
                validate_priority(
                    priority_raw,
                    max_response_bytes=self.limits.response_bytes,
                )
                if status == "ok"
                else None
            )
            error_raw = response.get("error")
            error = (
                cast(dict[str, JsonValue], error_raw)
                if isinstance(error_raw, dict)
                else None
            )
            result = WorkerCallResult(status, priority, elapsed_ns, error)
            self._calls += 1
            self._total_elapsed_ns += elapsed_ns
            self._max_elapsed_ns = max(self._max_elapsed_ns, elapsed_ns)
            if status != "ok":
                self._failures += 1
                self._failed = True
                self._terminate()
            return result
        except BaseException:
            self._failures += 1
            self._failed = True
            self._terminate()
            raise

    def telemetry(self) -> dict[str, JsonValue]:
        return {
            "protocol_version": _PROTOCOL_VERSION,
            "pid": self._process.pid,
            "calls": self._calls,
            "failures": self._failures,
            "total_policy_elapsed_ns": self._total_elapsed_ns,
            "max_policy_elapsed_ns": self._max_elapsed_ns,
            "usable": self.usable,
            "controls": self._controls,
            "captured_stderr_bytes": min(
                self._stderr_size(),
                self.limits.captured_output_bytes,
            ),
        }

    def _stderr_size(self) -> int:
        try:
            return os.fstat(self._stderr.fileno()).st_size
        except OSError:
            return 0

    def captured_stderr(self) -> str:
        self._stderr.seek(0)
        data = self._stderr.read(self.limits.captured_output_bytes)
        return data.decode("utf-8", errors="replace")

    def _terminate(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def close(self) -> None:
        if self._closed:
            return
        if self.usable:
            try:
                self._send({"type": "shutdown"}, self.limits.request_bytes)
                self._receive(0.2)
            except BaseException:
                self._failed = True
        self._terminate()
        self._closed = True
        self._stderr.close()
        self._temporary.cleanup()

    def __enter__(self) -> PolicyWorker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = _build_parser().parse_args()
    raise SystemExit(_child_main() if arguments.child else 2)
