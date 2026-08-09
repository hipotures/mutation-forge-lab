"""Runtime contracts for the isolated ordinary-Python policy worker."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from mutation_forge.models import Edge, GraphState, JsonValue, RewritePlan

from .contracts import NoPlan

RUNTIME_PROTOCOL_ID = "mforge.native.python_policy_runtime.v1"
SAFE_GRAPH_API_PROTOCOL_ID = "mforge.native.python_safe_graph_api.v1"
SEMANTIC_TRACE_PROTOCOL_ID = "mforge.native.python_semantic_trace.v1"
RANDOM_PROTOCOL_ID = "mforge.native.python_api_random.v1"


class PolicyRuntimeError(RuntimeError):
    """Base class for host-visible policy runtime failures."""


class PolicyInfrastructureError(PolicyRuntimeError):
    """The host, sandbox, or worker protocol failed."""


class PolicyWorkerStartupError(PolicyInfrastructureError):
    """A worker failed before completing its trusted startup attestation."""

    def __init__(
        self,
        message: str,
        *,
        private_diagnostic: str,
        diagnostic_bytes: int,
    ) -> None:
        super().__init__(message)
        self.private_diagnostic = private_diagnostic
        self.diagnostic_bytes = diagnostic_bytes


class UnsupportedPolicySandboxError(PolicyInfrastructureError):
    """Required sandbox controls are unavailable."""


class PolicyProtocolError(PolicyInfrastructureError):
    """The host/worker framed protocol was malformed."""


class IllegalRewriteError(ValueError):
    """The candidate's proposed graph rewrite failed trusted validation."""


@dataclass(frozen=True, slots=True)
class PolicyRuntimeLimitsV1:
    """Provisional M2 runtime and capability limits."""

    propose_wall_seconds: float = 1.0
    address_space_bytes: int = 256 * 1024 * 1024
    worker_lifetime_seconds: float = 60.0
    cpu_seconds: int = 60
    file_size_bytes: int = 64 * 1024
    open_files: int = 16
    process_count: int = 1
    request_bytes: int = 256 * 1024
    response_bytes: int = 32 * 1024
    diagnostics_bytes: int = 64 * 1024
    total_api_calls: int = 256
    selector_calls: int = 64
    action_calls: int = 64
    selector_result_size: int = 64
    net_added_edges: int = 8
    net_removed_edges: int = 8
    random_draws: int = 2_048
    loop_body_entries: int = 4_096
    helper_invocations: int = 256
    helper_call_depth: int = 8
    graph_order: int = 128

    def __post_init__(self) -> None:
        positive_reals = ("propose_wall_seconds", "worker_lifetime_seconds")
        for field_name in positive_reals:
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be positive")
        positive_ints = (
            "address_space_bytes",
            "cpu_seconds",
            "file_size_bytes",
            "open_files",
            "process_count",
            "request_bytes",
            "response_bytes",
            "diagnostics_bytes",
            "total_api_calls",
            "selector_calls",
            "action_calls",
            "selector_result_size",
            "net_added_edges",
            "net_removed_edges",
            "random_draws",
            "loop_body_entries",
            "helper_invocations",
            "helper_call_depth",
            "graph_order",
        )
        for field_name in positive_ints:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        hard_caps = {
            "propose_wall_seconds": 1.0,
            "address_space_bytes": 256 * 1024 * 1024,
            "worker_lifetime_seconds": 60.0,
            "cpu_seconds": 60,
            "file_size_bytes": 64 * 1024,
            "open_files": 16,
            "process_count": 1,
            "request_bytes": 256 * 1024,
            "response_bytes": 32 * 1024,
            "diagnostics_bytes": 64 * 1024,
            "total_api_calls": 256,
            "selector_calls": 64,
            "action_calls": 64,
            "selector_result_size": 64,
            "net_added_edges": 8,
            "net_removed_edges": 8,
            "random_draws": 2_048,
            "loop_body_entries": 4_096,
            "helper_invocations": 256,
            "helper_call_depth": 8,
            "graph_order": 128,
        }
        for field_name, maximum in hard_caps.items():
            if getattr(self, field_name) > maximum:
                raise ValueError(f"{field_name} exceeds the M2 hard cap of {maximum}")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


type WitnessLoadProviderV1 = Callable[
    [GraphState],
    tuple[
        Mapping[tuple[int, int], int],
        Mapping[tuple[int, Edge], int],
    ],
]


@dataclass(frozen=True, slots=True)
class GraphFeatureInputV1:
    """Host-only graph features used by witness-load selectors."""

    vertex_witness_load: Mapping[tuple[int, int], int] = field(default_factory=dict)
    edge_witness_load: Mapping[tuple[int, Edge], int] = field(default_factory=dict)
    witness_load_provider: WitnessLoadProviderV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class RewriteHostV1(Protocol):
    """Host rewrite-validation boundary used by ``api.emit``."""

    # Candidate-invalid rewrites must raise IllegalRewriteError. Every other
    # exception is an infrastructure failure.
    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState: ...


@dataclass(frozen=True, slots=True)
class ProgramFailureV1:
    """Deterministic failure attributable to a candidate policy."""

    code: str
    message: str
    classification: str = "PROGRAM_FAILURE"

    def __post_init__(self) -> None:
        if self.classification != "PROGRAM_FAILURE":
            raise ValueError("program failures must use PROGRAM_FAILURE classification")
        if not self.code or not self.code.isascii():
            raise ValueError("program failure code must be non-empty ASCII")
        if len(self.message.encode("utf-8")) > 1_024:
            raise ValueError("program failure message must be at most 1,024 bytes")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "classification": self.classification,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class SemanticAPIEventV1:
    """Timing-free semantic record for one safe-API call."""

    ordinal: int
    method: str
    arguments: Mapping[str, JsonValue]
    result: JsonValue

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "ordinal": self.ordinal,
            "method": self.method,
            "arguments": dict(self.arguments),
            "result": self.result,
        }


@dataclass(frozen=True, slots=True)
class PolicyInvocationResultV1:
    """One isolated ``propose`` outcome."""

    outcome: str
    rewrite_plan: RewritePlan | None = None
    no_plan: NoPlan | None = None
    failure: ProgramFailureV1 | None = None
    semantic_trace: tuple[SemanticAPIEventV1, ...] = ()
    wall_seconds: float = 0.0
    worker_rss_kib: int = 0
    loop_body_entries: int = 0
    helper_invocations: int = 0
    protocol_id: str = RUNTIME_PROTOCOL_ID

    def __post_init__(self) -> None:
        expected = {
            "REWRITE_PLAN": self.rewrite_plan is not None
            and self.no_plan is None
            and self.failure is None,
            "NO_PLAN": self.rewrite_plan is None
            and self.no_plan is not None
            and self.failure is None,
            "PROGRAM_FAILURE": self.rewrite_plan is None
            and self.no_plan is None
            and self.failure is not None,
        }
        if self.outcome not in expected or not expected[self.outcome]:
            raise ValueError("invocation outcome and payload are inconsistent")
        if self.protocol_id != RUNTIME_PROTOCOL_ID:
            raise ValueError("invalid runtime protocol ID")
