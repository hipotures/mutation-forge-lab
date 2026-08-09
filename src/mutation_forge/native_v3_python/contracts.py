"""Non-executing contracts for the ordinary-Python Native v3 policy path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from mutation_forge.models import JsonValue, RewritePlan

PYTHON_RESPONSE_SCHEMA_VERSION = "mforge.native.python_policy_response.v1"
PYTHON_POLICY_PROTOCOL_ID = "mforge.native.python_policy.v1"
PYTHON_WORKSPACE_SCHEMA_VERSION = "mforge.experiment.native_python.v1"
PYTHON_EXPERIMENT_PROTOCOL_ID = "native-v3-python-v1"
BEHAVIOR_IDENTITY_PROTOCOL_ID = "mforge.native.python_policy_behavior.v1"
CURRENT_JSON_DSL_WORKSPACE_SCHEMA_VERSION = "mforge.experiment.v3"

MAX_FORBIDDEN_LENGTHS = 64
MAX_SELECTOR_RESULTS = 64
UINT64_MAX = (1 << 64) - 1
NO_PLAN_REASONS = frozenset({"EXPLICIT", "NO_MATCH", "ILLEGAL_FINAL_STATE", "NO_EFFECT"})


class PythonWorkspaceProtocolError(ValueError):
    """A workspace does not belong to the inactive ordinary-Python protocol."""


def require_python_workspace_schema_version(schema_version: object) -> str:
    """Accept only a newly-created ordinary-Python workspace identity."""

    if schema_version == CURRENT_JSON_DSL_WORKSPACE_SCHEMA_VERSION:
        raise PythonWorkspaceProtocolError(
            "JSON-DSL v3 workspaces are incompatible with the ordinary-Python protocol; "
            "create a new workspace"
        )
    if schema_version != PYTHON_WORKSPACE_SCHEMA_VERSION:
        raise PythonWorkspaceProtocolError(
            "unsupported ordinary-Python workspace schema: "
            f"{schema_version!r}; expected {PYTHON_WORKSPACE_SCHEMA_VERSION!r}"
        )
    return PYTHON_WORKSPACE_SCHEMA_VERSION


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= UINT64_MAX
    ):
        raise ValueError(f"{field_name} must be an unsigned 64-bit integer")
    return value


@dataclass(frozen=True, slots=True)
class PolicyContextV1:
    """Immutable, bounded scalar policy context."""

    step_index: int
    horizon: int
    acceptance_profile_id: str
    stagnation_steps: int
    exploration_window_index: int
    accepted_rewrites: int
    accepted_non_improving_rewrites: int
    consecutive_non_improving_rewrites: int
    witness_cap: int
    invocation_ordinal: int
    forbidden_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "step_index",
            "horizon",
            "stagnation_steps",
            "exploration_window_index",
            "accepted_rewrites",
            "accepted_non_improving_rewrites",
            "consecutive_non_improving_rewrites",
            "witness_cap",
            "invocation_ordinal",
        ):
            _require_nonnegative_int(getattr(self, field_name), field_name)
        if (
            not self.acceptance_profile_id
            or not self.acceptance_profile_id.isascii()
            or not self.acceptance_profile_id.isprintable()
        ):
            raise ValueError("acceptance_profile_id must be non-empty printable ASCII")
        if len(self.acceptance_profile_id.encode("ascii")) > 128:
            raise ValueError("acceptance_profile_id must be at most 128 bytes")
        if type(self.forbidden_lengths) is not tuple:
            raise ValueError("forbidden_lengths must be an immutable tuple")
        if len(self.forbidden_lengths) > MAX_FORBIDDEN_LENGTHS:
            raise ValueError(
                f"forbidden_lengths must contain at most {MAX_FORBIDDEN_LENGTHS} values"
            )
        for length in self.forbidden_lengths:
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or not 3 <= length <= UINT64_MAX
            ):
                raise ValueError(
                    "forbidden_lengths values must be unsigned 64-bit integers of at least 3"
                )
        if self.forbidden_lengths != tuple(sorted(set(self.forbidden_lengths))):
            raise ValueError("forbidden_lengths must be strictly increasing and unique")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "step_index": self.step_index,
            "horizon": self.horizon,
            "acceptance_profile_id": self.acceptance_profile_id,
            "stagnation_steps": self.stagnation_steps,
            "exploration_window_index": self.exploration_window_index,
            "accepted_rewrites": self.accepted_rewrites,
            "accepted_non_improving_rewrites": self.accepted_non_improving_rewrites,
            "consecutive_non_improving_rewrites": self.consecutive_non_improving_rewrites,
            "witness_cap": self.witness_cap,
            "invocation_ordinal": self.invocation_ordinal,
            "forbidden_lengths": list(self.forbidden_lengths),
        }


@dataclass(frozen=True, slots=True)
class GraphViewV1:
    """Immutable label-opaque graph scalars exposed to a policy."""

    order: int
    edge_count: int
    minimum_degree: int
    maximum_degree: int

    def __post_init__(self) -> None:
        for field_name in ("order", "edge_count", "minimum_degree", "maximum_degree"):
            _require_nonnegative_int(getattr(self, field_name), field_name)
        if self.minimum_degree > self.maximum_degree:
            raise ValueError("minimum_degree must not exceed maximum_degree")
        if self.order == 0 and self.maximum_degree != 0:
            raise ValueError("an empty graph must have maximum_degree zero")
        if self.order > 0 and self.maximum_degree >= self.order:
            raise ValueError("maximum_degree must be smaller than graph order")


class VertexRef(Protocol):
    """Opaque host-minted vertex reference."""


class EdgeRef(Protocol):
    """Opaque host-minted edge reference."""


class NonEdgeRef(Protocol):
    """Opaque host-minted absent-edge reference."""


class PathRef(Protocol):
    """Opaque host-minted path reference."""


class MatchingRef(Protocol):
    """Opaque host-minted matching reference."""


class RelocationRef(Protocol):
    """Opaque host-minted endpoint-relocation reference."""


class FanoutRef(Protocol):
    """Opaque host-minted edge-fanout reference."""


@dataclass(frozen=True, slots=True)
class NoPlan:
    """Host-minted terminal result declaration for a future runtime."""

    reason: str

    def __post_init__(self) -> None:
        if self.reason not in NO_PLAN_REASONS:
            raise ValueError(f"unsupported NoPlan reason: {self.reason!r}")


_Ref = TypeVar(
    "_Ref",
    VertexRef,
    EdgeRef,
    NonEdgeRef,
    PathRef,
    MatchingRef,
    RelocationRef,
    FanoutRef,
)


class SafeGraphAPIV1(Protocol):
    """Maximum initial capability surface; M1 supplies no implementation."""

    def vertices_degree_extreme(self, mode: str = "max") -> tuple[VertexRef, ...]: ...

    def vertices_degree_class(self, degree: int) -> tuple[VertexRef, ...]: ...

    def vertices_witness_load_extreme(
        self, length: int, mode: str = "max"
    ) -> tuple[VertexRef, ...]: ...

    def edges_witness_load_extreme(
        self, length: int, mode: str = "max"
    ) -> tuple[EdgeRef, ...]: ...

    def vertices_articulation_risk(self, mode: str = "max") -> tuple[VertexRef, ...]: ...

    def edges_bridge_risk(self, mode: str = "max") -> tuple[EdgeRef, ...]: ...

    def edges_removable(self) -> tuple[EdgeRef, ...]: ...

    def vertices_distance_band(
        self, source: VertexRef, minimum: int, maximum: int
    ) -> tuple[VertexRef, ...]: ...

    def non_edges_from_vertex(self, vertex: VertexRef) -> tuple[NonEdgeRef, ...]: ...

    def non_edges_legal(self) -> tuple[NonEdgeRef, ...]: ...

    def non_edges_local_cycle_risk(
        self, mode: str = "max"
    ) -> tuple[NonEdgeRef, ...]: ...

    def paths_length_two(self) -> tuple[PathRef, ...]: ...

    def matching_k_switch_reconnections(self, k: int) -> tuple[MatchingRef, ...]: ...

    def matching_k_switch_reconnections_for_edge(
        self, edge: EdgeRef, k: int
    ) -> tuple[MatchingRef, ...]: ...

    def relocations_legal(self) -> tuple[RelocationRef, ...]: ...

    def relocations_legal_for_edge(
        self, edge: EdgeRef
    ) -> tuple[RelocationRef, ...]: ...

    def edge_fanouts_legal(self) -> tuple[FanoutRef, ...]: ...

    def edge_fanouts_legal_for_edge(
        self, edge: EdgeRef
    ) -> tuple[FanoutRef, ...]: ...

    def pick(
        self,
        items: tuple[_Ref, ...],
        seed: int,
        salt: int | str,
        feature: str = "uniform",
    ) -> _Ref | None: ...

    def add_edge(self, edge: NonEdgeRef) -> None: ...

    def remove_edge(self, edge: EdgeRef) -> None: ...

    def relocate_endpoint(self, relocation: RelocationRef) -> None: ...

    def k_switch(self, matching: MatchingRef) -> None: ...

    def edge_fanout(self, fanout: FanoutRef) -> None: ...

    def edge_fold(self, path: PathRef) -> None: ...

    def emit(self) -> RewritePlan: ...

    def no_plan(self, reason: str = "EXPLICIT") -> NoPlan: ...


@dataclass(frozen=True, slots=True)
class BehaviorIdentityV1:
    """Separate future runtime product; M1 never computes it by execution."""

    probe_manifest_sha256: str
    behavior_signature: str
    protocol_id: str = BEHAVIOR_IDENTITY_PROTOCOL_ID

    def __post_init__(self) -> None:
        for field_name in ("probe_manifest_sha256", "behavior_signature"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        if self.protocol_id != BEHAVIOR_IDENTITY_PROTOCOL_ID:
            raise ValueError(
                f"protocol_id must be exactly {BEHAVIOR_IDENTITY_PROTOCOL_ID!r}"
            )
