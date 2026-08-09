"""Stdlib-only child entry point for the ordinary-Python policy sandbox.

This file is executed as a script inside bubblewrap. It intentionally imports
no Mutation Forge package modules and is the only M2 module that compiles or
executes generated policy source.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import ctypes
import errno
import json
import os
import resource
import socket
import struct
import sys
from typing import BinaryIO

_HEADER = struct.Struct("!I")
_PROTOCOL_ID = "mforge.native.python_policy_runtime.v1"
_SAFE_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "enumerate",
        "int",
        "len",
        "max",
        "min",
        "range",
        "reversed",
        "sum",
        "tuple",
    }
)
_CONTEXT_FIELDS = (
    "step_index",
    "horizon",
    "acceptance_profile_id",
    "stagnation_steps",
    "exploration_window_index",
    "accepted_rewrites",
    "accepted_non_improving_rewrites",
    "consecutive_non_improving_rewrites",
    "witness_cap",
    "invocation_ordinal",
    "forbidden_lengths",
)
_GRAPH_FIELDS = ("order", "edge_count", "minimum_degree", "maximum_degree")


class _ProtocolError(RuntimeError):
    pass


class _PolicyAPIError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
        raise _ProtocolError(f"non-canonical protocol value: {error}") from error


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("protocol stream closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_frame(stream: BinaryIO, limit: int) -> dict[str, object]:
    length = _HEADER.unpack(_read_exact(stream, _HEADER.size))[0]
    if length > limit:
        raise _ProtocolError(f"request frame length {length} exceeds {limit}")
    body = _read_exact(stream, length)

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
        raise _ProtocolError(f"invalid request JSON: {error}") from error
    if not isinstance(payload, dict):
        raise _ProtocolError("protocol request must be a JSON object")
    if _canonical_json_bytes(payload) != body:
        raise _ProtocolError("protocol request is not canonical JSON")
    return payload


def _write_frame(stream: BinaryIO, payload: object, limit: int) -> None:
    body = _canonical_json_bytes(payload)
    if len(body) > limit:
        raise _ProtocolError(f"response frame length {len(body)} exceeds {limit}")
    stream.write(_HEADER.pack(len(body)) + body)
    stream.flush()


class _FrozenRecord:
    __slots__ = ("_frozen",)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("policy inputs are immutable")
        object.__setattr__(self, name, value)


class _PolicyContext(_FrozenRecord):
    __slots__ = (*_CONTEXT_FIELDS, "_frozen")

    def __init__(self, values: dict[str, object]) -> None:
        if set(values) != set(_CONTEXT_FIELDS):
            raise _ProtocolError("policy context fields do not match PolicyContextV1")
        for name in _CONTEXT_FIELDS:
            value = values[name]
            if name == "forbidden_lengths":
                if not isinstance(value, list) or any(
                    isinstance(item, bool) or not isinstance(item, int) for item in value
                ):
                    raise _ProtocolError("invalid forbidden_lengths")
                value = tuple(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_frozen", True)


class _GraphView(_FrozenRecord):
    __slots__ = (*_GRAPH_FIELDS, "_frozen")

    def __init__(self, values: dict[str, object]) -> None:
        if set(values) != set(_GRAPH_FIELDS):
            raise _ProtocolError("graph fields do not match GraphViewV1")
        for name in _GRAPH_FIELDS:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_frozen", True)


class _OpaqueRef:
    __slots__ = ("__token", "__kind")
    __token: str
    __kind: str

    def __init__(self, token: str, kind: str) -> None:
        self.__token = token
        self.__kind = kind

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _OpaqueRef)
            and self.__token == other.__token
            and self.__kind == other.__kind
        )

    def __hash__(self) -> int:
        return hash((self.__token, self.__kind))

    def __repr__(self) -> str:
        return f"<{self.__kind}Ref opaque>"

    def encoded(self) -> dict[str, str]:
        return {"$ref": self.__token, "kind": self.__kind}


class _HostResult:
    __slots__ = ("__token", "__kind")
    __token: str
    __kind: str

    def __init__(self, token: str, kind: str) -> None:
        self.__token = token
        self.__kind = kind

    def encoded(self) -> dict[str, str]:
        return {"$host_result": self.__token, "kind": self.__kind}


def _decode_value(value: object) -> object:
    if isinstance(value, dict) and set(value) == {"$ref", "kind"}:
        token = value.get("$ref")
        kind = value.get("kind")
        if not isinstance(token, str) or not isinstance(kind, str):
            raise _ProtocolError("malformed reference response")
        return _OpaqueRef(token, kind)
    if isinstance(value, dict) and set(value) == {"$host_result", "kind"}:
        token = value.get("$host_result")
        kind = value.get("kind")
        if not isinstance(token, str) or kind not in {"rewrite_plan", "no_plan"}:
            raise _ProtocolError("malformed host result response")
        return _HostResult(token, kind)
    if isinstance(value, list):
        return tuple(_decode_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise _ProtocolError(f"unsupported API response value {type(value).__name__}")


def _encode_value(value: object) -> object:
    if isinstance(value, _OpaqueRef):
        return value.encoded()
    if isinstance(value, (tuple, list)):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise _PolicyAPIError(
        "INVALID_API_ARGUMENT",
        f"unsupported API argument type {type(value).__name__}",
    )


class _APIProxy:
    __slots__ = ("_stdin", "_stdout", "_request_limit", "_response_limit", "_invocation", "_next")

    def __init__(
        self,
        stdin: BinaryIO,
        stdout: BinaryIO,
        request_limit: int,
        response_limit: int,
        invocation: str,
    ) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._request_limit = request_limit
        self._response_limit = response_limit
        self._invocation = invocation
        self._next = 0

    def _call(self, method: str, arguments: dict[str, object]) -> object:
        request_id = self._next
        self._next += 1
        _write_frame(
            self._stdout,
            {
                "type": "api_call",
                "invocation": self._invocation,
                "request_id": request_id,
                "method": method,
                "arguments": {
                    key: _encode_value(value) for key, value in arguments.items()
                },
            },
            self._response_limit,
        )
        response = _read_frame(self._stdin, self._request_limit)
        if (
            response.get("type") != "api_result"
            or response.get("invocation") != self._invocation
            or response.get("request_id") != request_id
        ):
            raise _ProtocolError("API response correlation failed")
        status = response.get("status")
        if status == "ok":
            return _decode_value(response.get("value"))
        if status == "program_error":
            code = response.get("code")
            message = response.get("message")
            if not isinstance(code, str) or not isinstance(message, str):
                raise _ProtocolError("malformed API program error")
            raise _PolicyAPIError(code, message)
        raise _ProtocolError("malformed API response status")

    def vertices_degree_extreme(self, mode: str = "max") -> tuple[_OpaqueRef, ...]:
        return self._call("vertices_degree_extreme", {"mode": mode})  # type: ignore[return-value]

    def vertices_degree_class(self, degree: int) -> tuple[_OpaqueRef, ...]:
        return self._call("vertices_degree_class", {"degree": degree})  # type: ignore[return-value]

    def vertices_witness_load_extreme(
        self,
        length: int,
        mode: str = "max",
    ) -> tuple[_OpaqueRef, ...]:
        return self._call(  # type: ignore[return-value]
            "vertices_witness_load_extreme",
            {"length": length, "mode": mode},
        )

    def edges_witness_load_extreme(
        self,
        length: int,
        mode: str = "max",
    ) -> tuple[_OpaqueRef, ...]:
        return self._call(  # type: ignore[return-value]
            "edges_witness_load_extreme",
            {"length": length, "mode": mode},
        )

    def vertices_articulation_risk(self, mode: str = "max") -> tuple[_OpaqueRef, ...]:
        return self._call("vertices_articulation_risk", {"mode": mode})  # type: ignore[return-value]

    def edges_bridge_risk(self, mode: str = "max") -> tuple[_OpaqueRef, ...]:
        return self._call("edges_bridge_risk", {"mode": mode})  # type: ignore[return-value]

    def edges_removable(self) -> tuple[_OpaqueRef, ...]:
        return self._call("edges_removable", {})  # type: ignore[return-value]

    def vertices_distance_band(
        self,
        source: _OpaqueRef,
        minimum: int,
        maximum: int,
    ) -> tuple[_OpaqueRef, ...]:
        return self._call(  # type: ignore[return-value]
            "vertices_distance_band",
            {"source": source, "minimum": minimum, "maximum": maximum},
        )

    def non_edges_from_vertex(self, vertex: _OpaqueRef) -> tuple[_OpaqueRef, ...]:
        return self._call("non_edges_from_vertex", {"vertex": vertex})  # type: ignore[return-value]

    def non_edges_legal(self) -> tuple[_OpaqueRef, ...]:
        return self._call("non_edges_legal", {})  # type: ignore[return-value]

    def non_edges_local_cycle_risk(self, mode: str = "max") -> tuple[_OpaqueRef, ...]:
        return self._call("non_edges_local_cycle_risk", {"mode": mode})  # type: ignore[return-value]

    def paths_length_two(self) -> tuple[_OpaqueRef, ...]:
        return self._call("paths_length_two", {})  # type: ignore[return-value]

    def matching_k_switch_reconnections(self, k: int) -> tuple[_OpaqueRef, ...]:
        return self._call("matching_k_switch_reconnections", {"k": k})  # type: ignore[return-value]

    def matching_k_switch_reconnections_for_edge(
        self,
        edge: _OpaqueRef,
        k: int,
    ) -> tuple[_OpaqueRef, ...]:
        return self._call(  # type: ignore[return-value]
            "matching_k_switch_reconnections_for_edge",
            {"edge": edge, "k": k},
        )

    def relocations_legal(self) -> tuple[_OpaqueRef, ...]:
        return self._call("relocations_legal", {})  # type: ignore[return-value]

    def relocations_legal_for_edge(
        self,
        edge: _OpaqueRef,
    ) -> tuple[_OpaqueRef, ...]:
        return self._call(  # type: ignore[return-value]
            "relocations_legal_for_edge",
            {"edge": edge},
        )

    def edge_fanouts_legal(self) -> tuple[_OpaqueRef, ...]:
        return self._call("edge_fanouts_legal", {})  # type: ignore[return-value]

    def edge_fanouts_legal_for_edge(
        self,
        edge: _OpaqueRef,
    ) -> tuple[_OpaqueRef, ...]:
        return self._call(  # type: ignore[return-value]
            "edge_fanouts_legal_for_edge",
            {"edge": edge},
        )

    def pick(
        self,
        items: tuple[_OpaqueRef, ...],
        seed: int,
        salt: int | str,
        feature: str = "uniform",
    ) -> _OpaqueRef | None:
        return self._call(  # type: ignore[return-value]
            "pick",
            {"items": items, "seed": seed, "salt": salt, "feature": feature},
        )

    def add_edge(self, edge: _OpaqueRef) -> None:
        self._call("add_edge", {"edge": edge})

    def remove_edge(self, edge: _OpaqueRef) -> None:
        self._call("remove_edge", {"edge": edge})

    def relocate_endpoint(self, relocation: _OpaqueRef) -> None:
        self._call("relocate_endpoint", {"relocation": relocation})

    def k_switch(self, matching: _OpaqueRef) -> None:
        self._call("k_switch", {"matching": matching})

    def edge_fanout(self, fanout: _OpaqueRef) -> None:
        self._call("edge_fanout", {"fanout": fanout})

    def edge_fold(self, path: _OpaqueRef) -> None:
        self._call("edge_fold", {"path": path})

    def emit(self) -> _HostResult:
        return self._call("emit", {})  # type: ignore[return-value]

    def no_plan(self, reason: str = "EXPLICIT") -> _HostResult:
        return self._call("no_plan", {"reason": reason})  # type: ignore[return-value]


class _RuntimeGuard:
    __slots__ = (
        "loop_limit",
        "helper_limit",
        "helper_depth_limit",
        "loop_entries",
        "helper_invocations",
        "helper_depth",
    )

    def __init__(
        self,
        loop_limit: int,
        helper_limit: int,
        helper_depth_limit: int,
    ) -> None:
        self.loop_limit = loop_limit
        self.helper_limit = helper_limit
        self.helper_depth_limit = helper_depth_limit
        self.loop_entries = 0
        self.helper_invocations = 0
        self.helper_depth = 0

    def reset(self) -> None:
        self.loop_entries = 0
        self.helper_invocations = 0
        self.helper_depth = 0

    def loop(self) -> None:
        self.loop_entries += 1
        if self.loop_entries > self.loop_limit:
            raise _PolicyAPIError(
                "LOOP_BUDGET_EXCEEDED",
                "loop-body entry budget exceeded",
            )

    def helper_enter(self) -> None:
        self.helper_invocations += 1
        if self.helper_invocations > self.helper_limit:
            raise _PolicyAPIError(
                "HELPER_BUDGET_EXCEEDED",
                "helper invocation budget exceeded",
            )
        self.helper_depth += 1
        if self.helper_depth > self.helper_depth_limit:
            raise _PolicyAPIError(
                "HELPER_DEPTH_EXCEEDED",
                "helper call-depth budget exceeded",
            )

    def helper_exit(self) -> None:
        self.helper_depth -= 1


class _GuardTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._function_name: str | None = None

    @staticmethod
    def _guard_call(method: str) -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="_guard", ctx=ast.Load()),
                    attr=method,
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        previous = self._function_name
        self._function_name = node.name
        self.generic_visit(node)
        if node.name != "propose":
            docstring: list[ast.stmt] = []
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring = [body[0]]
                body = body[1:]
            node.body = [
                *docstring,
                self._guard_call("helper_enter"),
                ast.Try(
                    body=body or [ast.Pass()],
                    handlers=[],
                    orelse=[],
                    finalbody=[self._guard_call("helper_exit")],
                ),
            ]
        self._function_name = previous
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        node.body.insert(0, self._guard_call("loop"))
        return node


class _NoConstructMeta(type):
    def __call__(cls, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise TypeError(f"{cls.__name__} is host-minted and cannot be constructed")


class RewritePlan(metaclass=_NoConstructMeta):
    pass


class NoPlan(metaclass=_NoConstructMeta):
    pass


def _safe_globals(guard: _RuntimeGuard) -> dict[str, object]:
    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTINS}
    return {
        "__builtins__": safe,
        "__name__": "__mforge_policy__",
        "_guard": guard,
        "RewritePlan": RewritePlan,
        "NoPlan": NoPlan,
    }


def _limit_value(limits: dict[str, object], name: str) -> int:
    value = limits.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _ProtocolError(f"runtime limit {name} must be a positive integer")
    return value


def _set_limits(limits: dict[str, object]) -> None:
    cpu_seconds = _limit_value(limits, "cpu_seconds")
    address_space_bytes = _limit_value(limits, "address_space_bytes")
    file_size_bytes = _limit_value(limits, "file_size_bytes")
    open_files = _limit_value(limits, "open_files")
    process_count = _limit_value(limits, "process_count")
    pairs = (
        (resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1),
        (
            resource.RLIMIT_AS,
            address_space_bytes,
            address_space_bytes,
        ),
        (
            resource.RLIMIT_FSIZE,
            file_size_bytes,
            file_size_bytes,
        ),
        (resource.RLIMIT_NOFILE, open_files, open_files),
        (resource.RLIMIT_NPROC, process_count, process_count),
    )
    for limit, soft, hard in pairs:
        resource.setrlimit(limit, (soft, hard))


def _set_no_new_privileges() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    if libc.prctl(39, 0, 0, 0, 0) != 1:
        raise OSError(ctypes.get_errno(), "PR_GET_NO_NEW_PRIVS failed")


def _install_seccomp() -> tuple[ctypes.CDLL, dict[str, int]]:
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = (ctypes.c_uint32,)
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_load.argtypes = (ctypes.c_void_p,)
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = (ctypes.c_void_p,)
    library.seccomp_rule_add.restype = ctypes.c_int
    context = library.seccomp_init(0x7FFF0000)
    if not context:
        raise OSError("seccomp_init failed")
    denied_names = (
        "open",
        "openat",
        "openat2",
        "creat",
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "sendto",
        "recvfrom",
        "clone",
        "clone3",
        "fork",
        "vfork",
        "execve",
        "execveat",
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "clock_gettime",
        "clock_gettime64",
        "gettimeofday",
        "time",
        "clock_settime",
        "clock_settime64",
        "clock_nanosleep",
        "nanosleep",
        "getrandom",
        "mount",
        "umount2",
        "pivot_root",
        "chroot",
        "bpf",
        "keyctl",
        "perf_event_open",
        "io_uring_setup",
    )
    required = {"openat", "socket", "clone", "execve", "ptrace", "clock_gettime", "getrandom"}
    numbers: dict[str, int] = {}
    try:
        for name in denied_names:
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                if name in required:
                    raise OSError(f"required syscall {name} is unknown to libseccomp")
                continue
            numbers[name] = number
            action = 0x00050000 | errno.EPERM
            if library.seccomp_rule_add(context, action, number, 0) != 0:
                raise OSError(f"failed to deny syscall {name}")
        if library.seccomp_load(context) != 0:
            raise OSError(ctypes.get_errno(), "seccomp_load failed")
    finally:
        library.seccomp_release(context)
    return library, numbers


def _syscall_denied(number: int, *arguments: object) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    result = libc.syscall(number, *arguments)
    return result == -1 and ctypes.get_errno() == errno.EPERM


def _sandbox_probes(numbers: dict[str, int]) -> dict[str, bool]:
    try:
        os.open("/etc/passwd", os.O_RDONLY)
    except OSError as error:
        filesystem_denied = error.errno == errno.EPERM
    else:
        filesystem_denied = False
    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as error:
        network_denied = error.errno == errno.EPERM
    else:
        network_denied = False
    try:
        os.fork()
    except OSError as error:
        process_denied = error.errno == errno.EPERM
    else:
        process_denied = False
    clock_denied = _syscall_denied(numbers["clock_gettime"], 0, 0)
    randomness_denied = _syscall_denied(numbers["getrandom"], 0, 0, 0)
    ptrace_denied = _syscall_denied(numbers["ptrace"], 0, 0, 0, 0)
    return {
        "filesystem": filesystem_denied,
        "network": network_denied,
        "process": process_denied,
        "clock_syscalls": clock_denied,
        "ambient_randomness": randomness_denied,
        "ptrace": ptrace_denied,
    }


def _namespace_record(parent: dict[str, object]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name in ("user", "mnt", "pid", "net", "ipc", "uts"):
        current = os.readlink(f"/proc/self/ns/{name}")
        expected_parent = parent.get(name)
        result[name] = isinstance(expected_parent, str) and current != expected_parent
    return result


def _control_record(
    limits: dict[str, object],
    parent_namespaces: dict[str, object],
    prefilter_fds: list[int],
    cwd_empty: bool,
    probes: dict[str, bool],
) -> dict[str, object]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "cwd": os.getcwd(),
        "cwd_empty": cwd_empty,
        "environment": dict(sorted(os.environ.items())),
        "open_fds": prefilter_fds,
        "namespaces": _namespace_record(parent_namespaces),
        "no_new_privileges": True,
        "seccomp": probes,
        "rlimits": {
            "cpu": list(resource.getrlimit(resource.RLIMIT_CPU)),
            "address_space": list(resource.getrlimit(resource.RLIMIT_AS)),
            "file_size": list(resource.getrlimit(resource.RLIMIT_FSIZE)),
            "open_files": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
            "process_count": list(resource.getrlimit(resource.RLIMIT_NPROC)),
        },
        "configured_limits": limits,
    }


def _program_failure(code: str, message: str) -> dict[str, object]:
    bounded = message.encode("utf-8", errors="replace")[:1_024].decode(
        "utf-8",
        errors="replace",
    )
    return {
        "type": "result",
        "status": "program_failure",
        "failure": {
            "classification": "PROGRAM_FAILURE",
            "code": code,
            "message": bounded,
        },
    }


def _child_main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        initialize = _read_frame(stdin, 256 * 1024)
        if initialize.get("type") != "initialize":
            raise _ProtocolError("first request must be initialize")
        if initialize.get("protocol_id") != _PROTOCOL_ID:
            raise _ProtocolError("runtime protocol ID mismatch")
        source = initialize.get("source")
        limits = initialize.get("limits")
        parent_namespaces = initialize.get("parent_namespaces")
        if (
            not isinstance(source, str)
            or not isinstance(limits, dict)
            or not isinstance(parent_namespaces, dict)
        ):
            raise _ProtocolError("initialize requires source, limits, and namespaces")
        _set_limits(limits)
        cwd_empty = os.listdir(".") == []
        descriptor_directory = os.open(
            "/proc/self/fd",
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            listed_fds = tuple(
                int(name)
                for name in os.listdir(descriptor_directory)
                if name.isdigit()
            )
            prefilter_fds: list[int] = []
            for descriptor in listed_fds:
                if descriptor == descriptor_directory:
                    continue
                try:
                    os.fstat(descriptor)
                except OSError:
                    continue
                prefilter_fds.append(descriptor)
            prefilter_fds.sort()
        finally:
            os.close(descriptor_directory)
        _set_no_new_privileges()
        _, numbers = _install_seccomp()
        probes = _sandbox_probes(numbers)
        if not all(probes.values()):
            raise _ProtocolError("one or more sandbox denial probes failed")
        tree = ast.parse(
            source,
            filename="<generated-python-policy>",
            mode="exec",
            type_comments=True,
            feature_version=(3, 12),
        )
        tree = _GuardTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        guard = _RuntimeGuard(
            int(limits["loop_body_entries"]),
            int(limits["helper_invocations"]),
            int(limits["helper_call_depth"]),
        )
        namespace = _safe_globals(guard)
        exec(compile(tree, "<generated-python-policy>", "exec"), namespace, namespace)
        propose = namespace.get("propose")
        if not callable(propose):
            raise _ProtocolError("validated source did not define propose")
        controls = _control_record(
            limits,
            parent_namespaces,
            prefilter_fds,
            cwd_empty,
            probes,
        )
        _write_frame(
            stdout,
            {
                "type": "ready",
                "status": "ok",
                "protocol_id": _PROTOCOL_ID,
                "controls": controls,
                "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            int(limits["response_bytes"]),
        )
        while True:
            request = _read_frame(stdin, int(limits["request_bytes"]))
            request_type = request.get("type")
            if request_type == "shutdown":
                _write_frame(
                    stdout,
                    {"type": "shutdown", "status": "ok"},
                    int(limits["response_bytes"]),
                )
                return 0
            if request_type != "invoke":
                raise _ProtocolError(f"unexpected request type {request_type!r}")
            invocation = request.get("invocation")
            context_raw = request.get("ctx")
            graph_raw = request.get("graph")
            seed = request.get("seed")
            if (
                not isinstance(invocation, str)
                or not isinstance(context_raw, dict)
                or not isinstance(graph_raw, dict)
                or isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                raise _ProtocolError("malformed invoke request")
            guard.reset()
            api = _APIProxy(
                stdin,
                stdout,
                int(limits["request_bytes"]),
                int(limits["response_bytes"]),
                invocation,
            )
            try:
                result = propose(
                    _PolicyContext(context_raw),
                    _GraphView(graph_raw),
                    api,
                    seed,
                )
                if not isinstance(result, _HostResult):
                    response = _program_failure(
                        "INVALID_RETURN",
                        "propose must return a host-minted RewritePlan or NoPlan",
                    )
                else:
                    response = {
                        "type": "result",
                        "status": "ok",
                        "value": result.encoded(),
                    }
            except _PolicyAPIError as error:
                response = _program_failure(error.code, str(error))
            except MemoryError:
                response = _program_failure("MEMORY_LIMIT_EXCEEDED", "policy exhausted memory")
            except RecursionError:
                response = _program_failure("RECURSION_LIMIT_EXCEEDED", "policy recursion failed")
            except BaseException as error:
                response = _program_failure(
                    "POLICY_EXCEPTION",
                    f"{type(error).__name__}: {error}",
                )
            response["guard_counts"] = {
                "loop_body_entries": guard.loop_entries,
                "helper_invocations": guard.helper_invocations,
            }
            response["rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            _write_frame(stdout, response, int(limits["response_bytes"]))
            if response.get("status") != "ok":
                return 4
    except (EOFError, BrokenPipeError):
        return 1
    except BaseException:
        return 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = _build_parser().parse_args()
    raise SystemExit(_child_main() if arguments.child else 2)
