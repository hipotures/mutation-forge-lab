"""Host coordinator for the isolated ordinary-Python policy worker."""

from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, cast

from mutation_forge.models import GraphState, JsonValue, RewritePlan

from .contracts import GraphViewV1, PolicyContextV1
from .runtime_contracts import (
    RUNTIME_PROTOCOL_ID,
    GraphFeatureInputV1,
    PolicyInfrastructureError,
    PolicyInvocationResultV1,
    PolicyProtocolError,
    PolicyRuntimeLimitsV1,
    PolicyWorkerStartupError,
    ProgramFailureV1,
    RewriteHostV1,
    UnsupportedPolicySandboxError,
)
from .safe_api import (
    SafeAPIInfrastructureError,
    SafeAPIProgramError,
    SafeGraphSessionV1,
    graph_view_v1,
)
from .validation import PythonProgramIdentityV1, validate_python_policy_source

_HEADER = struct.Struct("!I")
_WORKER_SCRIPT = Path(__file__).with_name("worker_main.py")
_REQUIRED_NAMESPACES = ("user", "mnt", "pid", "net", "ipc", "uts")
_SAFE_ENVIRONMENT = {
    "HOME": "/work",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin",
    "PWD": "/work",
}


class _WorkerTimeout(Exception):
    pass


class _WorkerCrash(Exception):
    pass


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise PolicyProtocolError(f"protocol value is not JSON-compatible: {error}") from error


def _canonical_frame(payload: object, limit: int) -> bytes:
    body = _canonical_json_bytes(payload)
    if len(body) > limit:
        raise PolicyProtocolError(f"protocol frame of {len(body)} bytes exceeds {limit}")
    return _HEADER.pack(len(body)) + body


def _strict_json_object(body: bytes) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate protocol key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"invalid protocol constant {value}")

    try:
        payload = json.loads(
            body,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise PolicyProtocolError(f"worker returned invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PolicyProtocolError("worker response must be a JSON object")
    if _canonical_json_bytes(payload) != body:
        raise PolicyProtocolError("worker response is not canonical JSON")
    return payload


def _parent_namespaces() -> dict[str, str]:
    try:
        return {
            name: os.readlink(f"/proc/self/ns/{name}")
            for name in _REQUIRED_NAMESPACES
        }
    except OSError as error:
        raise UnsupportedPolicySandboxError(
            "Linux namespace identities are unavailable"
        ) from error


def _require_linux_sandbox() -> tuple[Path, Path, Path]:
    if sys.platform != "linux":
        raise UnsupportedPolicySandboxError("ordinary-Python workers require Linux")
    if sys.version_info[:2] != (3, 12):
        raise UnsupportedPolicySandboxError("ordinary-Python workers require Python 3.12")
    bwrap_raw = shutil.which("bwrap")
    if bwrap_raw is None:
        raise UnsupportedPolicySandboxError("bubblewrap is required")
    bwrap = Path(bwrap_raw).resolve()
    runtime_root = Path(sys.base_prefix).resolve()
    python = runtime_root / "bin" / "python3.12"
    for path, label in ((bwrap, "bubblewrap"), (python, "Python executable")):
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise UnsupportedPolicySandboxError(f"{label} is unavailable") from error
        if not stat.S_ISREG(mode) or mode & stat.S_IWOTH:
            raise UnsupportedPolicySandboxError(
                f"{label} must be a regular file not writable by other"
            )
    try:
        root_mode = runtime_root.stat().st_mode
    except OSError as error:
        raise UnsupportedPolicySandboxError("Python runtime root is unavailable") from error
    if not stat.S_ISDIR(root_mode) or root_mode & stat.S_IWOTH:
        raise UnsupportedPolicySandboxError(
            "Python runtime root must be a directory not writable by other"
        )
    if not _WORKER_SCRIPT.is_file():
        raise UnsupportedPolicySandboxError("worker entry script is missing")
    return bwrap, runtime_root, python


def _worker_command(bwrap: Path, runtime_root: Path, python: Path) -> list[str]:
    sandbox_python = Path("/opt/python") / python.relative_to(runtime_root)
    return [
        str(bwrap),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--setenv",
        "HOME",
        "/work",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "LC_ALL",
        "C.UTF-8",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--dir",
        "/opt",
        "--dir",
        "/opt/mforge",
        "--ro-bind",
        str(runtime_root),
        "/opt/python",
        "--ro-bind",
        str(_WORKER_SCRIPT),
        "/opt/mforge/worker_main.py",
        "--chdir",
        "/work",
        "--",
        str(sandbox_python),
        "-I",
        "-S",
        "-u",
        "/opt/mforge/worker_main.py",
        "--child",
    ]


def _prepare_child() -> None:
    os.setsid()
    os.umask(0o077)


class IsolatedPolicyWorkerV1:
    """One fail-closed worker process dedicated to one validated candidate."""

    def __init__(
        self,
        source: str,
        limits: PolicyRuntimeLimitsV1 | None = None,
    ) -> None:
        self.limits = limits or PolicyRuntimeLimitsV1()
        validation = validate_python_policy_source(source)
        if not validation.valid or validation.identity is None:
            diagnostics = [item.as_dict() for item in validation.diagnostics]
            raise ValueError(
                json.dumps(
                    {"error": "CONTRACT_INVALID", "diagnostics": diagnostics},
                    sort_keys=True,
                )
            )
        self.identity: PythonProgramIdentityV1 = validation.identity
        if self.identity.program_hash is None:
            raise ValueError("valid policy identity is missing program_hash")
        self._program_hash = self.identity.program_hash
        self._source = validation.response.source if validation.response is not None else source
        self._failed = False
        self._closed = False
        self._calls = 0
        self._failures = 0
        self._rotations = 0
        self._started_at = 0.0
        self._startup_seconds = 0.0
        self._controls: dict[str, JsonValue] = {}
        self._last_rss_kib = 0
        try:
            self._spawn_process()
        except BaseException:
            self._failed = True
            self._closed = True
            raise

    def _spawn_process(self) -> None:
        self._started_at = time.monotonic()
        self._startup_seconds = 0.0
        self._controls = {}
        self._last_rss_kib = 0
        self._stderr = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        try:
            bwrap, runtime_root, python = _require_linux_sandbox()
            self._process = subprocess.Popen(
                _worker_command(bwrap, runtime_root, python),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                cwd="/",
                env={"PATH": "/usr/bin"},
                close_fds=True,
                preexec_fn=_prepare_child,
            )
        except BaseException:
            self._stderr.close()
            raise
        if self._process.stdin is None or self._process.stdout is None:
            self._terminate()
            diagnostic, diagnostic_bytes = self._startup_diagnostic()
            self._stderr.close()
            raise PolicyWorkerStartupError(
                "worker protocol pipes were not created",
                private_diagnostic=diagnostic,
                diagnostic_bytes=diagnostic_bytes,
            )
        self._stdin = cast(BinaryIO, self._process.stdin)
        self._stdout = cast(BinaryIO, self._process.stdout)
        try:
            self._send(
                {
                    "type": "initialize",
                    "protocol_id": RUNTIME_PROTOCOL_ID,
                    "source": self._source,
                    "limits": self.limits.as_dict(),
                    "parent_namespaces": _parent_namespaces(),
                },
                self.limits.request_bytes,
            )
            ready = self._receive(time.monotonic() + min(10.0, self.limits.worker_lifetime_seconds))
            self._accept_ready(ready)
            self._startup_seconds = time.monotonic() - self._started_at
        except (_WorkerTimeout, _WorkerCrash) as error:
            self._terminate()
            diagnostic, diagnostic_bytes = self._startup_diagnostic()
            self._stderr.close()
            raise PolicyWorkerStartupError(
                f"worker initialization failed: {error}",
                private_diagnostic=diagnostic,
                diagnostic_bytes=diagnostic_bytes,
            ) from error
        except BaseException:
            self._terminate()
            self._stderr.close()
            raise

    @property
    def usable(self) -> bool:
        return not self._failed and not self._closed and self._process.poll() is None

    def _shutdown_process(self, *, strict: bool) -> None:
        failure: BaseException | None = None
        exit_code = self._process.poll()
        if strict and exit_code is not None:
            failure = PolicyInfrastructureError(
                f"worker exited before rotation shutdown (code={exit_code})"
            )
        elif exit_code is None:
            try:
                self._send({"type": "shutdown"}, self.limits.request_bytes)
                response = self._receive(time.monotonic() + 0.2)
                if response != {"status": "ok", "type": "shutdown"}:
                    raise PolicyProtocolError("worker returned invalid shutdown response")
                self._process.wait(timeout=0.2)
            except BaseException as error:
                failure = error
        self._terminate()
        self._stderr.close()
        if strict and failure is not None:
            raise PolicyInfrastructureError(
                "worker process could not be cleanly rotated"
            ) from failure

    def _rotate_process(self) -> None:
        try:
            self._shutdown_process(strict=True)
            self._spawn_process()
        except BaseException:
            self._failed = True
            raise
        self._rotations += 1

    def _prepare_full_propose_window(self) -> float:
        exit_code = self._process.poll()
        if exit_code is not None:
            self._failed = True
            self._shutdown_process(strict=False)
            raise PolicyInfrastructureError(
                f"worker exited while idle before invocation (code={exit_code})"
            )
        process_deadline = self._started_at + self.limits.worker_lifetime_seconds
        now = time.monotonic()
        if process_deadline - now < self.limits.propose_wall_seconds:
            self._rotate_process()
            process_deadline = self._started_at + self.limits.worker_lifetime_seconds
            now = time.monotonic()
        if process_deadline - now < self.limits.propose_wall_seconds:
            self._failed = True
            self._shutdown_process(strict=False)
            raise PolicyInfrastructureError(
                "worker lifetime cannot guarantee a full propose wall-time window"
            )
        return now

    def _accept_ready(self, ready: dict[str, object]) -> None:
        if (
            ready.get("type") != "ready"
            or ready.get("status") != "ok"
            or ready.get("protocol_id") != RUNTIME_PROTOCOL_ID
        ):
            raise PolicyProtocolError(f"worker initialization failed: {ready}")
        controls = ready.get("controls")
        rss_kib = ready.get("rss_kib")
        if not isinstance(controls, dict) or not self._controls_match(controls):
            raise UnsupportedPolicySandboxError(
                "worker did not prove all required sandbox controls: "
                + json.dumps(controls, sort_keys=True)[:4_096]
            )
        if isinstance(rss_kib, bool) or not isinstance(rss_kib, int) or rss_kib < 0:
            raise PolicyProtocolError("worker returned invalid startup RSS")
        self._controls = cast(dict[str, JsonValue], controls)
        self._last_rss_kib = rss_kib

    def _controls_match(self, controls: dict[object, object]) -> bool:
        expected_rlimits = {
            "cpu": [self.limits.cpu_seconds, self.limits.cpu_seconds + 1],
            "address_space": [
                self.limits.address_space_bytes,
                self.limits.address_space_bytes,
            ],
            "file_size": [self.limits.file_size_bytes, self.limits.file_size_bytes],
            "open_files": [self.limits.open_files, self.limits.open_files],
            "process_count": [self.limits.process_count, self.limits.process_count],
        }
        namespaces = controls.get("namespaces")
        seccomp = controls.get("seccomp")
        environment = controls.get("environment")
        open_fds = controls.get("open_fds")
        return (
            controls.get("protocol_id") == RUNTIME_PROTOCOL_ID
            and controls.get("cwd") == "/work"
            and controls.get("cwd_empty") is True
            and controls.get("no_new_privileges") is True
            and controls.get("rlimits") == expected_rlimits
            and controls.get("configured_limits") == self.limits.as_dict()
            and isinstance(namespaces, dict)
            and set(namespaces) == set(_REQUIRED_NAMESPACES)
            and all(value is True for value in namespaces.values())
            and isinstance(seccomp, dict)
            and set(seccomp)
            == {
                "filesystem",
                "network",
                "process",
                "clock_syscalls",
                "ambient_randomness",
                "ptrace",
            }
            and all(value is True for value in seccomp.values())
            and environment == _SAFE_ENVIRONMENT
            and isinstance(open_fds, list)
            and open_fds == [0, 1, 2]
        )

    def _send(self, payload: object, limit: int) -> None:
        frame = _canonical_frame(payload, limit)
        try:
            self._stdin.write(frame)
            self._stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise _WorkerCrash("worker closed its request pipe") from error

    def _read_exact(self, size: int, deadline: float) -> bytes:
        chunks = bytearray()
        selector = selectors.DefaultSelector()
        selector.register(self._stdout, selectors.EVENT_READ)
        try:
            while len(chunks) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _WorkerTimeout("worker response exceeded wall limit")
                if not selector.select(remaining):
                    raise _WorkerTimeout("worker response exceeded wall limit")
                chunk = os.read(self._stdout.fileno(), size - len(chunks))
                if not chunk:
                    code = self._process.poll()
                    raise _WorkerCrash(f"worker exited before a response (code={code})")
                chunks.extend(chunk)
        finally:
            selector.close()
        return bytes(chunks)

    def _receive(self, deadline: float) -> dict[str, object]:
        header = self._read_exact(_HEADER.size, deadline)
        length = _HEADER.unpack(header)[0]
        if length > self.limits.response_bytes:
            raise PolicyProtocolError(
                f"worker response frame length {length} exceeds {self.limits.response_bytes}"
            )
        body = self._read_exact(length, deadline)
        return _strict_json_object(body)

    @staticmethod
    def _program_failure(code: str, message: str) -> ProgramFailureV1:
        bounded = message.encode("utf-8", errors="replace")[:1_024].decode(
            "utf-8",
            errors="replace",
        )
        return ProgramFailureV1(code=code, message=bounded)

    def _failure_result(
        self,
        session: SafeGraphSessionV1,
        failure: ProgramFailureV1,
        wall_seconds: float,
        *,
        guard_counts: dict[str, int] | None = None,
        rss_kib: int | None = None,
    ) -> PolicyInvocationResultV1:
        counts = guard_counts or {}
        return PolicyInvocationResultV1(
            outcome="PROGRAM_FAILURE",
            failure=failure,
            semantic_trace=session.semantic_trace,
            wall_seconds=wall_seconds,
            selector_wall_seconds=session.timing["selector_wall_seconds"],
            action_wall_seconds=session.timing["action_wall_seconds"],
            worker_rss_kib=rss_kib if rss_kib is not None else self._last_rss_kib,
            loop_body_entries=counts.get("loop_body_entries", 0),
            helper_invocations=counts.get("helper_invocations", 0),
        )

    def invoke(
        self,
        *,
        context: PolicyContextV1,
        graph: GraphState,
        rewrite_host: RewriteHostV1,
        seed: int,
        features: GraphFeatureInputV1 | None = None,
    ) -> PolicyInvocationResultV1:
        """Execute one policy call while the host services capability RPC."""

        if not self.usable:
            raise PolicyInfrastructureError("failed or closed workers cannot be reused")
        try:
            session = SafeGraphSessionV1(
                graph=graph,
                context=context,
                seed=seed,
                program_hash=self._program_hash,
                rewrite_host=rewrite_host,
                limits=self.limits,
                features=features or GraphFeatureInputV1(),
            )
        except (SafeAPIInfrastructureError, ValueError) as error:
            raise PolicyInfrastructureError(f"invalid host runtime input: {error}") from error
        graph_view: GraphViewV1 = graph_view_v1(graph)
        invocation = os.urandom(16).hex()
        started = self._prepare_full_propose_window()
        deadline = started + self.limits.propose_wall_seconds
        program_error_sent = False
        try:
            self._send(
                {
                    "type": "invoke",
                    "invocation": invocation,
                    "ctx": context.as_dict(),
                    "graph": {
                        "order": graph_view.order,
                        "edge_count": graph_view.edge_count,
                        "minimum_degree": graph_view.minimum_degree,
                        "maximum_degree": graph_view.maximum_degree,
                    },
                    "seed": seed,
                },
                self.limits.request_bytes,
            )
            while True:
                message = self._receive(deadline)
                message_type = message.get("type")
                if message_type == "api_call":
                    if program_error_sent:
                        raise PolicyProtocolError(
                            "worker issued an API call after a program error"
                        )
                    request_id = message.get("request_id")
                    method = message.get("method")
                    arguments = message.get("arguments")
                    if (
                        message.get("invocation") != invocation
                        or isinstance(request_id, bool)
                        or not isinstance(request_id, int)
                        or request_id < 0
                        or not isinstance(method, str)
                        or not isinstance(arguments, dict)
                    ):
                        raise PolicyProtocolError("malformed API call frame")
                    if time.monotonic() >= deadline:
                        raise _WorkerTimeout("propose exceeded wall limit during API call")
                    try:
                        value = session.handle_call(method, arguments)
                    except SafeAPIProgramError as error:
                        program_error_sent = True
                        self._send(
                            {
                                "type": "api_result",
                                "invocation": invocation,
                                "request_id": request_id,
                                "status": "program_error",
                                "code": error.code,
                                "message": str(error)[:1_024],
                            },
                            self.limits.request_bytes,
                        )
                    except SafeAPIInfrastructureError as error:
                        raise PolicyInfrastructureError(str(error)) from error
                    except Exception as error:
                        raise PolicyInfrastructureError(
                            f"safe graph API host failure: {type(error).__name__}"
                        ) from error
                    else:
                        self._send(
                            {
                                "type": "api_result",
                                "invocation": invocation,
                                "request_id": request_id,
                                "status": "ok",
                                "value": value,
                            },
                            self.limits.request_bytes,
                        )
                    if time.monotonic() >= deadline:
                        raise _WorkerTimeout("propose exceeded wall limit during API call")
                    continue
                if message_type != "result":
                    raise PolicyProtocolError(f"unexpected worker frame: {message_type!r}")
                wall_seconds = time.monotonic() - started
                guard_counts_raw = message.get("guard_counts")
                rss_kib = message.get("rss_kib")
                if (
                    not isinstance(guard_counts_raw, dict)
                    or set(guard_counts_raw)
                    != {"loop_body_entries", "helper_invocations"}
                    or any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 0
                        for value in guard_counts_raw.values()
                    )
                    or isinstance(rss_kib, bool)
                    or not isinstance(rss_kib, int)
                    or rss_kib < 0
                ):
                    raise PolicyProtocolError("malformed worker counters")
                guard_counts = cast(dict[str, int], guard_counts_raw)
                self._last_rss_kib = rss_kib
                self._calls += 1
                if message.get("status") == "program_failure":
                    failure_raw = message.get("failure")
                    if not isinstance(failure_raw, dict):
                        raise PolicyProtocolError("missing program failure payload")
                    code = failure_raw.get("code")
                    detail = failure_raw.get("message")
                    if (
                        failure_raw.get("classification") != "PROGRAM_FAILURE"
                        or not isinstance(code, str)
                        or not isinstance(detail, str)
                    ):
                        raise PolicyProtocolError("malformed program failure payload")
                    self._failed = True
                    self._failures += 1
                    result = self._failure_result(
                        session,
                        self._program_failure(code, detail),
                        wall_seconds,
                        guard_counts=guard_counts,
                        rss_kib=rss_kib,
                    )
                    self._terminate()
                    return result
                if message.get("status") != "ok":
                    raise PolicyProtocolError("malformed worker result status")
                result_value_raw = message.get("value")
                if (
                    not isinstance(result_value_raw, dict)
                    or set(result_value_raw) != {"$host_result", "kind"}
                ):
                    raise PolicyProtocolError("worker returned a non-minted result")
                token = result_value_raw.get("$host_result")
                kind = result_value_raw.get("kind")
                if not isinstance(token, str) or not isinstance(kind, str):
                    raise PolicyProtocolError("worker result token is malformed")
                try:
                    host_result = session.resolve_result(token, kind)
                except SafeAPIProgramError as error:
                    self._failed = True
                    self._failures += 1
                    result = self._failure_result(
                        session,
                        self._program_failure(error.code, str(error)),
                        wall_seconds,
                        guard_counts=guard_counts,
                        rss_kib=rss_kib,
                    )
                    self._terminate()
                    return result
                if isinstance(host_result, RewritePlan):
                    return PolicyInvocationResultV1(
                        outcome="REWRITE_PLAN",
                        rewrite_plan=host_result,
                        semantic_trace=session.semantic_trace,
                        wall_seconds=wall_seconds,
                        selector_wall_seconds=session.timing[
                            "selector_wall_seconds"
                        ],
                        action_wall_seconds=session.timing[
                            "action_wall_seconds"
                        ],
                        worker_rss_kib=rss_kib,
                        loop_body_entries=guard_counts["loop_body_entries"],
                        helper_invocations=guard_counts["helper_invocations"],
                    )
                return PolicyInvocationResultV1(
                    outcome="NO_PLAN",
                    no_plan=host_result,
                    semantic_trace=session.semantic_trace,
                    wall_seconds=wall_seconds,
                    selector_wall_seconds=session.timing[
                        "selector_wall_seconds"
                    ],
                    action_wall_seconds=session.timing[
                        "action_wall_seconds"
                    ],
                    worker_rss_kib=rss_kib,
                    loop_body_entries=guard_counts["loop_body_entries"],
                    helper_invocations=guard_counts["helper_invocations"],
                )
        except _WorkerTimeout:
            self._failed = True
            self._failures += 1
            self._terminate()
            return self._failure_result(
                session,
                self._program_failure(
                    "PROPOSE_TIMEOUT",
                    "propose exceeded its wall-time limit",
                ),
                time.monotonic() - started,
            )
        except _WorkerCrash as error:
            self._failed = True
            self._failures += 1
            self._terminate()
            return self._failure_result(
                session,
                self._program_failure("WORKER_CRASH", str(error)),
                time.monotonic() - started,
            )
        except BaseException:
            self._failed = True
            self._failures += 1
            self._terminate()
            raise

    def telemetry(self) -> dict[str, JsonValue]:
        return {
            "protocol_id": RUNTIME_PROTOCOL_ID,
            "pid": self._process.pid,
            "calls": self._calls,
            "failures": self._failures,
            "rotations": self._rotations,
            "usable": self.usable,
            "startup_seconds": self._startup_seconds,
            "worker_age_seconds": max(0.0, time.monotonic() - self._started_at),
            "worker_rss_kib": self._last_rss_kib,
            "controls": self._controls,
            "captured_stderr_bytes": min(
                self._stderr_size(),
                self.limits.diagnostics_bytes,
            ),
        }

    def _stderr_size(self) -> int:
        try:
            return os.fstat(self._stderr.fileno()).st_size
        except (OSError, ValueError):
            return 0

    def _startup_diagnostic(self) -> tuple[str, int]:
        try:
            self._stderr.seek(0)
            data = self._stderr.read(self.limits.diagnostics_bytes)
        except (OSError, ValueError):
            return "", 0
        return data.decode("utf-8", errors="replace"), len(data)

    def captured_stderr(self) -> str:
        if self._stderr.closed:
            return ""
        self._stderr.seek(0)
        data = self._stderr.read(self.limits.diagnostics_bytes)
        return data.decode("utf-8", errors="replace")

    def _terminate(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        for stream_name in ("_stdin", "_stdout"):
            stream = getattr(self, stream_name, None)
            if stream is not None and not stream.closed:
                with suppress(OSError):
                    stream.close()

    def close(self) -> None:
        if self._closed:
            return
        self._shutdown_process(strict=False)
        self._closed = True

    def __enter__(self) -> IsolatedPolicyWorkerV1:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
