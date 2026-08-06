"""Static Native v3 program validation and identity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import Any, cast

from mutation_forge.models import JsonValue

from .canonical import (
    CanonicalJsonError,
    canonical_json_bytes,
    json_value,
    parse_strict_json,
    program_hash,
)

PROGRAM_SCHEMA_VERSION = "mforge.native.program.v3"
VALIDATOR_PROTOCOL_ID = "native_v3_validator_v1"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_UINT32_MAX = (1 << 32) - 1


class ValueType(StrEnum):
    BOOL = "Bool"
    INT = "Int"
    RATIONAL = "Rational"
    STRING = "String"
    VERTEX = "VertexRef"
    EDGE = "EdgeRef"
    NON_EDGE = "NonEdgeRef"
    PATH = "PathRef"
    VERTEX_SET = "VertexSetRef"
    EDGE_SET = "EdgeSetRef"
    MATCHING = "MatchingRef"
    RELOCATION_SET = "RelocationSetRef"
    RELOCATION = "RelocationRef"
    FANOUT_SET = "FanoutSetRef"
    FANOUT = "FanoutRef"


SELECTOR_TYPES: dict[str, ValueType] = {
    "vertices_degree_extreme": ValueType.VERTEX_SET,
    "vertices_degree_class": ValueType.VERTEX_SET,
    "vertices_witness_load_extreme": ValueType.VERTEX_SET,
    "edges_witness_load_extreme": ValueType.EDGE_SET,
    "vertices_articulation_risk": ValueType.VERTEX_SET,
    "edges_bridge_risk": ValueType.EDGE_SET,
    "vertices_distance_band": ValueType.VERTEX_SET,
    "edges_removable": ValueType.EDGE_SET,
    "non_edges_legal": ValueType.NON_EDGE,
    "non_edges_from_vertex": ValueType.NON_EDGE,
    "non_edges_local_cycle_risk": ValueType.NON_EDGE,
    "paths_length_two": ValueType.PATH,
    "matching_k_switch_reconnections": ValueType.MATCHING,
}

SELECTOR_COSTS: dict[str, int] = {
    "vertices_degree_extreme": 1,
    "vertices_degree_class": 1,
    "vertices_witness_load_extreme": 32,
    "edges_witness_load_extreme": 32,
    "vertices_articulation_risk": 16,
    "edges_bridge_risk": 16,
    "vertices_distance_band": 8,
    "edges_removable": 1,
    "non_edges_legal": 2,
    "non_edges_from_vertex": 2,
    "non_edges_local_cycle_risk": 4,
    "paths_length_two": 2,
    "matching_k_switch_reconnections": 4,
}

SELECTOR_ARGUMENT_TYPES: dict[str, dict[str, ValueType]] = {
    "vertices_degree_extreme": {"mode": ValueType.STRING},
    "vertices_degree_class": {"degree": ValueType.INT},
    "vertices_witness_load_extreme": {
        "length": ValueType.INT,
        "mode": ValueType.STRING,
    },
    "edges_witness_load_extreme": {
        "length": ValueType.INT,
        "mode": ValueType.STRING,
    },
    "vertices_articulation_risk": {"mode": ValueType.STRING},
    "edges_bridge_risk": {"mode": ValueType.STRING},
    "vertices_distance_band": {
        "source": ValueType.VERTEX,
        "minimum": ValueType.INT,
        "maximum": ValueType.INT,
    },
    "edges_removable": {},
    "non_edges_legal": {},
    "non_edges_from_vertex": {"vertex": ValueType.VERTEX},
    "non_edges_local_cycle_risk": {"mode": ValueType.STRING},
    "paths_length_two": {},
    "matching_k_switch_reconnections": {"k": ValueType.INT},
}

ACTION_ARGUMENT_TYPES: dict[str, dict[str, ValueType]] = {
    "add_edge": {"edge": ValueType.NON_EDGE},
    "remove_edge": {"edge": ValueType.EDGE},
    "relocate_endpoint": {
        "edge": ValueType.EDGE,
        "keep": ValueType.VERTEX,
        "new": ValueType.VERTEX,
    },
    "k_switch": {"matching": ValueType.MATCHING},
    "edge_fanout": {"edge": ValueType.EDGE, "w": ValueType.VERTEX},
    "edge_fold": {"path": ValueType.PATH},
}

CTX_TYPES: dict[str, ValueType] = {
    "step_index": ValueType.INT,
    "horizon": ValueType.INT,
    "acceptance_profile_id": ValueType.STRING,
    "stagnation_steps": ValueType.INT,
    "exploration_window_index": ValueType.INT,
    "accepted_rewrites": ValueType.INT,
    "accepted_non_improving_rewrites": ValueType.INT,
    "consecutive_non_improving_rewrites": ValueType.INT,
    "witness_cap": ValueType.INT,
}

FEATURE_TYPES: dict[str, ValueType] = {
    "order": ValueType.INT,
    "edge_count": ValueType.INT,
    "minimum_degree": ValueType.INT,
    "maximum_degree": ValueType.INT,
}

PICK_MODES = ("seeded_uniform", "seeded_weighted", "require_singleton")
WEIGHT_FEATURES = ("uniform", "degree", "inverse_degree")
BINARY_OPERATIONS = (
    "add",
    "subtract",
    "multiply",
    "minimum",
    "maximum",
    "equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "and",
    "or",
)
UNARY_OPERATIONS = ("not", "exists")
NO_PLAN_REASONS = ("EXPLICIT", "NO_MATCH", "ILLEGAL_FINAL_STATE", "NO_EFFECT")

type LiteralValue = str | int


@dataclass(frozen=True, slots=True)
class SelectorDefinition:
    result_type: ValueType
    cost: int
    arguments: dict[str, ValueType]
    literal_domains: dict[str, tuple[LiteralValue, ...]] = field(default_factory=dict)
    relation: str = ""
    ordered_nonnegative_bounds: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    arguments: dict[str, ValueType]
    relation: str


@dataclass(frozen=True, slots=True)
class ProgramContract:
    selectors: dict[str, SelectorDefinition]
    actions: dict[str, ActionDefinition]
    context_fields: dict[str, ValueType]
    graph_features: dict[str, ValueType]
    pick_results: dict[ValueType, ValueType]


DEFAULT_SELECTOR_DEFINITIONS = {
    selector_id: SelectorDefinition(
        result_type=SELECTOR_TYPES[selector_id],
        cost=SELECTOR_COSTS[selector_id],
        arguments=SELECTOR_ARGUMENT_TYPES[selector_id],
        literal_domains={
            **({"mode": ("min", "max")} if "mode" in SELECTOR_ARGUMENT_TYPES[selector_id] else {}),
            **(
                {"k": (2, 3, 4)}
                if selector_id == "matching_k_switch_reconnections"
                else {}
            ),
        },
    )
    for selector_id in SELECTOR_TYPES
}

DEFAULT_ACTION_DEFINITIONS = {
    "add_edge": ActionDefinition(
        arguments=ACTION_ARGUMENT_TYPES["add_edge"],
        relation="edge must be absent from the current overlay",
    ),
    "remove_edge": ActionDefinition(
        arguments=ACTION_ARGUMENT_TYPES["remove_edge"],
        relation="edge must be present in the current overlay",
    ),
    "relocate_endpoint": ActionDefinition(
        arguments=ACTION_ARGUMENT_TYPES["relocate_endpoint"],
        relation=(
            "keep must be an endpoint of edge; new must not be an endpoint; "
            "the replacement edge must be absent"
        ),
    ),
    "k_switch": ActionDefinition(
        arguments=ACTION_ARGUMENT_TYPES["k_switch"],
        relation=(
            "matching contains 2, 3, or 4 vertex-disjoint source edges and "
            "endpoint-preserving absent target edges"
        ),
    ),
    "edge_fanout": ActionDefinition(
        arguments=ACTION_ARGUMENT_TYPES["edge_fanout"],
        relation=(
            "w must not be an endpoint of edge and both replacement edges must be absent"
        ),
    ),
    "edge_fold": ActionDefinition(
        arguments=ACTION_ARGUMENT_TYPES["edge_fold"],
        relation="path contains two present edges and its endpoint edge must be absent",
    ),
}

DEFAULT_PROGRAM_CONTRACT = ProgramContract(
    selectors=DEFAULT_SELECTOR_DEFINITIONS,
    actions=DEFAULT_ACTION_DEFINITIONS,
    context_fields=CTX_TYPES,
    graph_features=FEATURE_TYPES,
    pick_results={
        ValueType.VERTEX_SET: ValueType.VERTEX,
        ValueType.EDGE_SET: ValueType.EDGE,
        ValueType.NON_EDGE: ValueType.NON_EDGE,
        ValueType.PATH: ValueType.PATH,
        ValueType.MATCHING: ValueType.MATCHING,
    },
)


@dataclass(frozen=True, slots=True)
class ProgramLimits:
    maximum_decoded_bytes: int = 32 * 1024
    maximum_nodes: int = 256
    maximum_depth: int = 12
    maximum_bindings: int = 32
    maximum_integer_bits: int = 64
    maximum_denominator_bits: int = 32
    maximum_selector_calls: int = 64
    maximum_selector_cost_units: int = 128
    maximum_repeat: int = 8
    maximum_gross_actions: int = 8

    def __post_init__(self) -> None:
        values = (
            self.maximum_decoded_bytes,
            self.maximum_nodes,
            self.maximum_depth,
            self.maximum_bindings,
            self.maximum_selector_calls,
            self.maximum_selector_cost_units,
            self.maximum_repeat,
            self.maximum_gross_actions,
        )
        if any(value < 0 for value in values):
            raise ValueError("program limits must be non-negative")
        if self.maximum_integer_bits < 1 or self.maximum_denominator_bits < 1:
            raise ValueError("integer bit limits must be positive")


@dataclass(frozen=True, slots=True)
class ProgramDiagnostic:
    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True, slots=True)
class ValidatedProgram:
    raw: str
    ast: dict[str, Any]
    canonical_json: str
    program_hash: str
    node_count: int
    maximum_depth: int
    binding_count: int
    selector_calls: int
    selector_cost_units: int
    gross_actions: int


@dataclass(frozen=True, slots=True)
class ProgramValidation:
    program: ValidatedProgram | None
    diagnostics: tuple[ProgramDiagnostic, ...]

    @property
    def valid(self) -> bool:
        return self.program is not None and not self.diagnostics


class _InvalidProgram(Exception):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.diagnostic = ProgramDiagnostic(code, path, message)


@dataclass(slots=True)
class _Stats:
    nodes: int = 0
    depth: int = 0
    bindings: int = 0
    selector_calls: int = 0
    selector_cost: int = 0
    actions: int = 0


def _merge_structural(target: _Stats, branches: list[_Stats]) -> None:
    target.nodes += sum(branch.nodes for branch in branches)
    target.depth = max((target.depth, *(branch.depth for branch in branches)))


def _merge_runtime_max(target: _Stats, branches: list[_Stats]) -> None:
    _merge_structural(target, branches)
    target.bindings += max((branch.bindings for branch in branches), default=0)
    target.selector_calls += max((branch.selector_calls for branch in branches), default=0)
    target.selector_cost += max((branch.selector_cost for branch in branches), default=0)
    target.actions += max((branch.actions for branch in branches), default=0)


def _merge_runtime_sum(target: _Stats, branches: list[_Stats]) -> None:
    _merge_structural(target, branches)
    target.bindings += max((branch.bindings for branch in branches), default=0)
    target.selector_calls += sum(branch.selector_calls for branch in branches)
    target.selector_cost += sum(branch.selector_cost for branch in branches)
    target.actions += sum(branch.actions for branch in branches)


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _InvalidProgram("type", path, "expected an object")
    if not all(isinstance(key, str) for key in value):
        raise _InvalidProgram("type", path, "object keys must be strings")
    return cast(dict[str, object], value)


def _exact_keys(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    path: str,
) -> None:
    allowed_optional = optional or set()
    missing = required.difference(value)
    unknown = set(value).difference(required | allowed_optional)
    if missing:
        raise _InvalidProgram("missing_field", path, f"missing fields: {sorted(missing)}")
    if unknown:
        raise _InvalidProgram("unknown_field", path, f"unknown fields: {sorted(unknown)}")


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise _InvalidProgram("identifier", path, "invalid identifier")
    return value


def _signed_bit_length(value: int) -> int:
    return value.bit_length() + 1 if value >= 0 else (~value).bit_length() + 1


def _integer(value: object, path: str, limits: ProgramLimits) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _InvalidProgram("integer", path, "expected an integer")
    if _signed_bit_length(value) > limits.maximum_integer_bits:
        raise _InvalidProgram(
            "integer_bits",
            path,
            f"integer exceeds {limits.maximum_integer_bits} signed bits",
        )
    return value


def _record_node(
    stats: _Stats,
    *,
    path: str,
    depth: int,
    limits: ProgramLimits,
) -> None:
    stats.nodes += 1
    stats.depth = max(stats.depth, depth)
    if depth > limits.maximum_depth:
        raise _InvalidProgram("depth_limit", path, "AST depth limit exceeded")


def _expression(
    value: object,
    *,
    path: str,
    depth: int,
    environment: dict[str, ValueType],
    stats: _Stats,
    limits: ProgramLimits,
    contract: ProgramContract,
) -> ValueType:
    _record_node(stats, path=path, depth=depth, limits=limits)
    if isinstance(value, bool):
        return ValueType.BOOL
    if isinstance(value, int):
        _integer(value, path, limits)
        return ValueType.INT
    if isinstance(value, str):
        if not value.isascii() or not value.isprintable():
            raise _InvalidProgram("string", path, "strings must be printable ASCII")
        return ValueType.STRING
    expression = _object(value, path)
    operation = expression.get("op")
    if operation == "ref":
        _exact_keys(expression, required={"op", "name"}, path=path)
        name = _identifier(expression["name"], f"{path}/name")
        try:
            return environment[name]
        except KeyError as exc:
            raise _InvalidProgram("unbound_reference", path, f"unbound name: {name}") from exc
    if operation in {"ctx", "feature"}:
        _exact_keys(expression, required={"op", "field"}, path=path)
        field = _identifier(expression["field"], f"{path}/field")
        registry = (
            contract.context_fields
            if operation == "ctx"
            else contract.graph_features
        )
        if field not in registry:
            raise _InvalidProgram("unknown_field", path, f"unsupported {operation} field: {field}")
        return registry[field]
    if operation == "rational":
        _exact_keys(expression, required={"op", "numerator", "denominator"}, path=path)
        numerator = _integer(expression["numerator"], f"{path}/numerator", limits)
        denominator = _integer(expression["denominator"], f"{path}/denominator", limits)
        if (
            denominator <= 0
            or denominator.bit_length() > limits.maximum_denominator_bits
        ):
            raise _InvalidProgram(
                "rational",
                path,
                "denominator must be positive and within its bit limit",
            )
        normalized = Fraction(numerator, denominator)
        if normalized.numerator != numerator or normalized.denominator != denominator:
            raise _InvalidProgram("rational", path, "rational must be normalized")
        return ValueType.RATIONAL
    if operation == "selector":
        _exact_keys(expression, required={"op", "selector_id", "arguments"}, path=path)
        selector_id = _identifier(expression["selector_id"], f"{path}/selector_id")
        if selector_id not in contract.selectors:
            raise _InvalidProgram("unknown_selector", path, f"unknown selector: {selector_id}")
        selector = contract.selectors[selector_id]
        arguments = _object(expression["arguments"], f"{path}/arguments")
        expected_arguments = selector.arguments
        if set(arguments) != set(expected_arguments):
            raise _InvalidProgram(
                "selector_arguments",
                path,
                f"expected arguments {sorted(expected_arguments)}, got {sorted(arguments)}",
            )
        for key, expected_type in expected_arguments.items():
            actual_type = _expression(
                arguments[key],
                path=f"{path}/arguments/{key}",
                depth=depth + 1,
                environment=environment,
                stats=stats,
                limits=limits,
                contract=contract,
            )
            if actual_type != expected_type:
                raise _InvalidProgram(
                    "selector_argument_type",
                    f"{path}/arguments/{key}",
                    f"expected {expected_type}, got {actual_type}",
                )
        for argument_name, allowed_values in selector.literal_domains.items():
            argument_value = arguments[argument_name]
            if (
                isinstance(argument_value, bool)
                or argument_value not in allowed_values
            ):
                if argument_name == "mode":
                    message = "selector mode must be the literal min or max"
                elif (
                    selector_id == "matching_k_switch_reconnections"
                    and argument_name == "k"
                ):
                    message = "k-switch requires the literal 2, 3, or 4"
                else:
                    message = (
                        f"{selector_id}.{argument_name} must be one of "
                        f"{list(allowed_values)}"
                    )
                raise _InvalidProgram(
                    "selector_argument_value",
                    f"{path}/arguments/{argument_name}",
                    message,
                )
        if selector.ordered_nonnegative_bounds is not None:
            minimum_name, maximum_name = selector.ordered_nonnegative_bounds
            minimum = arguments[minimum_name]
            maximum = arguments[maximum_name]
            if (
                isinstance(minimum, int)
                and not isinstance(minimum, bool)
                and isinstance(maximum, int)
                and not isinstance(maximum, bool)
                and (minimum < 0 or maximum < minimum)
            ):
                raise _InvalidProgram(
                    "selector_argument_value",
                    f"{path}/arguments",
                    "distance band literals must satisfy 0 <= minimum <= maximum",
                )
        stats.selector_calls += 1
        stats.selector_cost += selector.cost
        return selector.result_type
    if operation == "pick":
        _exact_keys(
            expression,
            required={"op", "source", "mode"},
            optional={"weight_feature"},
            path=path,
        )
        source_type = _expression(
            expression["source"],
            path=f"{path}/source",
            depth=depth + 1,
            environment=environment,
            stats=stats,
            limits=limits,
            contract=contract,
        )
        mode = expression["mode"]
        if mode not in PICK_MODES:
            raise _InvalidProgram("pick_mode", path, "unsupported pick mode")
        if mode == "seeded_weighted":
            weight_feature = _identifier(
                expression.get("weight_feature"),
                f"{path}/weight_feature",
            )
            if weight_feature not in WEIGHT_FEATURES:
                raise _InvalidProgram(
                    "weight_feature",
                    f"{path}/weight_feature",
                    "unsupported or unbudgeted weight feature",
                )
            if weight_feature != "uniform" and source_type is not ValueType.VERTEX_SET:
                raise _InvalidProgram(
                    "weight_feature_type",
                    f"{path}/weight_feature",
                    f"{weight_feature} requires a vertex population",
                )
        elif "weight_feature" in expression:
            raise _InvalidProgram(
                "weight_feature",
                f"{path}/weight_feature",
                "weight_feature is valid only for seeded_weighted",
            )
        if source_type not in contract.pick_results:
            raise _InvalidProgram("pick_type", path, f"cannot pick from {source_type}")
        return contract.pick_results[source_type]
    if operation in BINARY_OPERATIONS:
        _exact_keys(expression, required={"op", "left", "right"}, path=path)
        left = _expression(
            expression["left"],
            path=f"{path}/left",
            depth=depth + 1,
            environment=environment,
            stats=stats,
            limits=limits,
            contract=contract,
        )
        right = _expression(
            expression["right"],
            path=f"{path}/right",
            depth=depth + 1,
            environment=environment,
            stats=stats,
            limits=limits,
            contract=contract,
        )
        if operation in {"and", "or"}:
            if left != ValueType.BOOL or right != ValueType.BOOL:
                raise _InvalidProgram("expression_type", path, "boolean operands required")
            return ValueType.BOOL
        if operation in {"equal", "less", "less_equal", "greater", "greater_equal"}:
            if left != right and {left, right} != {ValueType.INT, ValueType.RATIONAL}:
                raise _InvalidProgram("expression_type", path, "incompatible comparison operands")
            return ValueType.BOOL
        if left not in {ValueType.INT, ValueType.RATIONAL} or right not in {
            ValueType.INT,
            ValueType.RATIONAL,
        }:
            raise _InvalidProgram("expression_type", path, "numeric operands required")
        return ValueType.RATIONAL if ValueType.RATIONAL in {left, right} else ValueType.INT
    if operation in UNARY_OPERATIONS:
        _exact_keys(expression, required={"op", "value"}, path=path)
        operand = _expression(
            expression["value"],
            path=f"{path}/value",
            depth=depth + 1,
            environment=environment,
            stats=stats,
            limits=limits,
            contract=contract,
        )
        if operation == "not" and operand != ValueType.BOOL:
            raise _InvalidProgram("expression_type", path, "not requires a boolean")
        return ValueType.BOOL
    raise _InvalidProgram("unknown_expression", path, f"unknown expression: {operation!r}")


def _node(
    value: object,
    *,
    path: str,
    depth: int,
    environment: dict[str, ValueType],
    stats: _Stats,
    limits: ProgramLimits,
    contract: ProgramContract,
    allow_terminal: bool = True,
) -> frozenset[str]:
    _record_node(stats, path=path, depth=depth, limits=limits)
    node = _object(value, path)
    operation = node.get("op")
    if operation == "block":
        _exact_keys(node, required={"op", "children"}, path=path)
        children = node["children"]
        if not isinstance(children, list) or not children:
            raise _InvalidProgram("block", path, "block children must be non-empty")
        block_outcomes: set[str] = {"continue"}
        for index, child in enumerate(children):
            if "continue" not in block_outcomes:
                raise _InvalidProgram(
                    "unreachable",
                    f"{path}/children/{index}",
                    "node is unreachable",
                )
            child_outcomes = _node(
                child,
                path=f"{path}/children/{index}",
                depth=depth + 1,
                environment=environment.copy(),
                stats=stats,
                limits=limits,
                contract=contract,
                allow_terminal=allow_terminal,
            )
            block_outcomes.discard("continue")
            block_outcomes.update(child_outcomes)
        return frozenset(block_outcomes)
    if operation == "let":
        _exact_keys(node, required={"op", "name", "value", "body"}, path=path)
        name = _identifier(node["name"], f"{path}/name")
        if name in environment:
            raise _InvalidProgram("binding", path, f"binding already exists: {name}")
        value_type = _expression(
            node["value"],
            path=f"{path}/value",
            depth=depth + 1,
            environment=environment,
            stats=stats,
            limits=limits,
            contract=contract,
        )
        nested = environment.copy()
        nested[name] = value_type
        stats.bindings += 1
        return _node(
            node["body"],
            path=f"{path}/body",
            depth=depth + 1,
            environment=nested,
            stats=stats,
            limits=limits,
            contract=contract,
            allow_terminal=allow_terminal,
        )
    if operation == "if":
        _exact_keys(node, required={"op", "condition", "then", "else"}, path=path)
        condition_type = _expression(
            node["condition"],
            path=f"{path}/condition",
            depth=depth + 1,
            environment=environment,
            stats=stats,
            limits=limits,
            contract=contract,
        )
        if condition_type != ValueType.BOOL:
            raise _InvalidProgram("condition", path, "if condition must be boolean")
        then_stats = _Stats()
        then_outcomes = _node(
            node["then"],
            path=f"{path}/then",
            depth=depth + 1,
            environment=environment.copy(),
            stats=then_stats,
            limits=limits,
            contract=contract,
            allow_terminal=allow_terminal,
        )
        else_stats = _Stats()
        else_outcomes = _node(
            node["else"],
            path=f"{path}/else",
            depth=depth + 1,
            environment=environment.copy(),
            stats=else_stats,
            limits=limits,
            contract=contract,
            allow_terminal=allow_terminal,
        )
        _merge_runtime_max(stats, [then_stats, else_stats])
        return then_outcomes | else_outcomes
    if operation == "try":
        _exact_keys(node, required={"op", "branches"}, path=path)
        branches = node["branches"]
        if not isinstance(branches, list) or not 1 <= len(branches) <= 8:
            raise _InvalidProgram("try", path, "try requires 1..8 branches")
        try_outcomes: frozenset[str] = frozenset()
        try_branch_stats: list[_Stats] = []
        for index, branch in enumerate(branches):
            current_stats = _Stats()
            try_outcomes |= _node(
                branch,
                path=f"{path}/branches/{index}",
                depth=depth + 1,
                environment=environment.copy(),
                stats=current_stats,
                limits=limits,
                contract=contract,
                allow_terminal=allow_terminal,
            )
            try_branch_stats.append(current_stats)
        _merge_runtime_sum(stats, try_branch_stats)
        return try_outcomes | frozenset({"terminal"})
    if operation == "repeat":
        _exact_keys(node, required={"op", "count", "body"}, path=path)
        count = _integer(node["count"], f"{path}/count", limits)
        if not 1 <= count <= limits.maximum_repeat:
            raise _InvalidProgram("repeat_limit", path, "repeat count exceeds limit")
        body_stats = _Stats()
        repeat_outcomes = _node(
            node["body"],
            path=f"{path}/body",
            depth=depth + 1,
            environment=environment.copy(),
            stats=body_stats,
            limits=limits,
            contract=contract,
            allow_terminal=False,
        )
        if repeat_outcomes != frozenset({"continue"}):
            raise _InvalidProgram("repeat_terminal", path, "repeat body must be non-terminal")
        stats.nodes += body_stats.nodes
        stats.depth = max(stats.depth, body_stats.depth)
        stats.bindings += body_stats.bindings
        stats.selector_calls += body_stats.selector_calls * count
        stats.selector_cost += body_stats.selector_cost * count
        stats.actions += body_stats.actions * count
        return frozenset({"continue"})
    if operation == "choose":
        _exact_keys(node, required={"op", "branches"}, path=path)
        branches = node["branches"]
        if not isinstance(branches, list) or not 1 <= len(branches) <= 8:
            raise _InvalidProgram("choose", path, "choose requires 1..8 branches")
        total_weight = 0
        choose_outcomes: frozenset[str] = frozenset()
        choose_branch_stats: list[_Stats] = []
        for index, raw_branch in enumerate(branches):
            branch = _object(raw_branch, f"{path}/branches/{index}")
            _exact_keys(branch, required={"weight", "body"}, path=f"{path}/branches/{index}")
            weight = _integer(
                branch["weight"],
                f"{path}/branches/{index}/weight",
                limits,
            )
            if weight <= 0 or weight > _UINT32_MAX:
                raise _InvalidProgram("weight", path, "choose weight must be positive uint32")
            total_weight += weight
            maximum_weight_sum = (1 << (limits.maximum_integer_bits - 1)) - 1
            if total_weight > maximum_weight_sum:
                raise _InvalidProgram(
                    "weight_sum",
                    path,
                    "cumulative weights exceed the signed integer bit limit",
                )
            current_stats = _Stats()
            choose_outcomes |= _node(
                branch["body"],
                path=f"{path}/branches/{index}/body",
                depth=depth + 1,
                environment=environment.copy(),
                stats=current_stats,
                limits=limits,
                contract=contract,
                allow_terminal=allow_terminal,
            )
            choose_branch_stats.append(current_stats)
        _merge_runtime_max(stats, choose_branch_stats)
        return choose_outcomes
    if operation == "apply":
        _exact_keys(node, required={"op", "action_id", "arguments"}, path=path)
        action_id = _identifier(node["action_id"], f"{path}/action_id")
        if action_id not in contract.actions:
            raise _InvalidProgram("unknown_action", path, f"unknown action: {action_id}")
        arguments = _object(node["arguments"], f"{path}/arguments")
        expected = contract.actions[action_id].arguments
        if set(arguments) != set(expected):
            raise _InvalidProgram(
                "action_arguments",
                path,
                f"expected arguments {sorted(expected)}, got {sorted(arguments)}",
            )
        for name, expected_type in expected.items():
            actual = _expression(
                arguments[name],
                path=f"{path}/arguments/{name}",
                depth=depth + 1,
                environment=environment,
                stats=stats,
                limits=limits,
                contract=contract,
            )
            if actual != expected_type:
                raise _InvalidProgram(
                    "action_argument_type",
                    f"{path}/arguments/{name}",
                    f"expected {expected_type}, got {actual}",
                )
        stats.actions += 1
        return frozenset({"continue"})
    if operation == "emit":
        _exact_keys(node, required={"op"}, path=path)
        if not allow_terminal:
            raise _InvalidProgram("terminal", path, "terminal node is forbidden here")
        return frozenset({"terminal"})
    if operation == "no_plan":
        _exact_keys(node, required={"op", "reason"}, path=path)
        if node["reason"] not in NO_PLAN_REASONS:
            raise _InvalidProgram("no_plan_reason", path, "unsupported NoPlan reason")
        if not allow_terminal:
            raise _InvalidProgram("terminal", path, "terminal node is forbidden here")
        return frozenset({"terminal"})
    raise _InvalidProgram("unknown_node", path, f"unknown node: {operation!r}")


def _enforce_limits(stats: _Stats, limits: ProgramLimits) -> None:
    values = (
        ("node_limit", stats.nodes, limits.maximum_nodes),
        ("binding_limit", stats.bindings, limits.maximum_bindings),
        ("selector_call_limit", stats.selector_calls, limits.maximum_selector_calls),
        ("selector_cost_limit", stats.selector_cost, limits.maximum_selector_cost_units),
        ("action_limit", stats.actions, limits.maximum_gross_actions),
    )
    for code, actual, maximum in values:
        if actual > maximum:
            raise _InvalidProgram(code, "/entry", f"{actual} exceeds limit {maximum}")


def validate_program(
    raw: str,
    *,
    limits: ProgramLimits | None = None,
    contract: ProgramContract | None = None,
) -> ProgramValidation:
    effective_limits = limits or ProgramLimits()
    effective_contract = contract or DEFAULT_PROGRAM_CONTRACT
    try:
        parsed = parse_strict_json(
            raw,
            maximum_bytes=effective_limits.maximum_decoded_bytes,
        )
        document = _object(parsed, "/")
        _exact_keys(document, required={"schema_version", "entry"}, path="/")
        if document["schema_version"] != PROGRAM_SCHEMA_VERSION:
            raise _InvalidProgram(
                "schema_version",
                "/schema_version",
                f"expected {PROGRAM_SCHEMA_VERSION}",
            )
        stats = _Stats()
        outcomes = _node(
            document["entry"],
            path="/entry",
            depth=1,
            environment={},
            stats=stats,
            limits=effective_limits,
            contract=effective_contract,
        )
        if outcomes != frozenset({"terminal"}):
            raise _InvalidProgram(
                "unterminated_path",
                "/entry",
                "every reachable path must terminate exactly once",
            )
        _enforce_limits(stats, effective_limits)
        canonical = canonical_json_bytes(document)
        identity = program_hash(
            schema_version=PROGRAM_SCHEMA_VERSION,
            canonical_program=canonical,
        )
        return ProgramValidation(
            ValidatedProgram(
                raw=raw,
                ast=cast(dict[str, Any], document),
                canonical_json=canonical.decode("ascii"),
                program_hash=identity,
                node_count=stats.nodes,
                maximum_depth=stats.depth,
                binding_count=stats.bindings,
                selector_calls=stats.selector_calls,
                selector_cost_units=stats.selector_cost,
                gross_actions=stats.actions,
            ),
            (),
        )
    except CanonicalJsonError as exc:
        return ProgramValidation(None, (ProgramDiagnostic("json", "/", str(exc)),))
    except _InvalidProgram as exc:
        return ProgramValidation(None, (exc.diagnostic,))


def validated_program_artifact(program: ValidatedProgram) -> dict[str, JsonValue]:
    return {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "program_json_raw": program.raw,
        "program_ast": json_value(program.ast),
        "program_json_canonical": program.canonical_json,
        "program_hash": program.program_hash,
        "validator_protocol_id": VALIDATOR_PROTOCOL_ID,
        "node_count": program.node_count,
        "maximum_depth": program.maximum_depth,
        "binding_count": program.binding_count,
        "selector_calls": program.selector_calls,
        "selector_cost_units": program.selector_cost_units,
        "gross_actions": program.gross_actions,
    }
