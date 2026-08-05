"""Static Native v3 program validation and identity."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT32_MAX = (1 << 32) - 1


class ValueType(StrEnum):
    BOOL = "Bool"
    INT = "Int"
    RATIONAL = "Rational"
    STRING = "String"
    VERTEX = "VertexRef"
    EDGE = "EdgeRef"
    NON_EDGE = "NonEdgeRef"
    PATH2 = "Path2Ref"
    VERTEX_SET = "VertexSetRef"
    EDGE_SET = "EdgeSetRef"
    MATCHING = "MatchingRef"


SELECTOR_TYPES: dict[str, ValueType] = {
    "vertices_degree_extreme": ValueType.VERTEX_SET,
    "vertices_degree_class": ValueType.VERTEX_SET,
    "vertices_witness_load_extreme": ValueType.VERTEX_SET,
    "edges_witness_load_extreme": ValueType.EDGE_SET,
    "vertices_articulation_risk": ValueType.VERTEX_SET,
    "edges_bridge_risk": ValueType.EDGE_SET,
    "vertices_distance_band": ValueType.VERTEX_SET,
    "non_edges_from_vertex": ValueType.NON_EDGE,
    "non_edges_local_cycle_risk": ValueType.NON_EDGE,
    "paths_length_two": ValueType.PATH2,
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
    "edge_fold": {"path": ValueType.PATH2},
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


@dataclass(frozen=True, slots=True)
class ProgramLimits:
    maximum_decoded_bytes: int = 32 * 1024
    maximum_nodes: int = 256
    maximum_depth: int = 12
    maximum_bindings: int = 32
    maximum_selector_calls: int = 64
    maximum_selector_cost_units: int = 128
    maximum_repeat: int = 8
    maximum_gross_actions: int = 8
    maximum_net_removed_edges: int = 8
    maximum_net_added_edges: int = 8


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


def _int64(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _InvalidProgram("integer", path, "expected an integer")
    if value < _INT64_MIN or value > _INT64_MAX:
        raise _InvalidProgram("integer_range", path, "integer exceeds signed 64-bit")
    return value


def _expression(
    value: object,
    *,
    path: str,
    environment: dict[str, ValueType],
    stats: _Stats,
) -> ValueType:
    stats.nodes += 1
    if isinstance(value, bool):
        return ValueType.BOOL
    if isinstance(value, int):
        _int64(value, path)
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
        registry = CTX_TYPES if operation == "ctx" else FEATURE_TYPES
        if field not in registry:
            raise _InvalidProgram("unknown_field", path, f"unsupported {operation} field: {field}")
        return registry[field]
    if operation == "rational":
        _exact_keys(expression, required={"op", "numerator", "denominator"}, path=path)
        numerator = _int64(expression["numerator"], f"{path}/numerator")
        denominator = _int64(expression["denominator"], f"{path}/denominator")
        if denominator <= 0 or denominator > _UINT32_MAX:
            raise _InvalidProgram("rational", path, "denominator must be uint32 and positive")
        normalized = Fraction(numerator, denominator)
        if normalized.numerator != numerator or normalized.denominator != denominator:
            raise _InvalidProgram("rational", path, "rational must be normalized")
        return ValueType.RATIONAL
    if operation == "selector":
        _exact_keys(expression, required={"op", "selector_id", "arguments"}, path=path)
        selector_id = _identifier(expression["selector_id"], f"{path}/selector_id")
        if selector_id not in SELECTOR_TYPES:
            raise _InvalidProgram("unknown_selector", path, f"unknown selector: {selector_id}")
        arguments = _object(expression["arguments"], f"{path}/arguments")
        expected_arguments = SELECTOR_ARGUMENT_TYPES[selector_id]
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
                environment=environment,
                stats=stats,
            )
            if actual_type != expected_type:
                raise _InvalidProgram(
                    "selector_argument_type",
                    f"{path}/arguments/{key}",
                    f"expected {expected_type}, got {actual_type}",
                )
        stats.selector_calls += 1
        stats.selector_cost += SELECTOR_COSTS[selector_id]
        return SELECTOR_TYPES[selector_id]
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
            environment=environment,
            stats=stats,
        )
        mode = expression["mode"]
        if mode not in {"seeded_uniform", "seeded_weighted", "require_singleton"}:
            raise _InvalidProgram("pick_mode", path, "unsupported pick mode")
        if mode == "seeded_weighted":
            weight_feature = _identifier(
                expression.get("weight_feature"),
                f"{path}/weight_feature",
            )
            if weight_feature not in {"uniform", "degree", "inverse_degree"}:
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
        result_types = {
            ValueType.VERTEX_SET: ValueType.VERTEX,
            ValueType.EDGE_SET: ValueType.EDGE,
            ValueType.NON_EDGE: ValueType.NON_EDGE,
            ValueType.PATH2: ValueType.PATH2,
            ValueType.MATCHING: ValueType.MATCHING,
        }
        if source_type not in result_types:
            raise _InvalidProgram("pick_type", path, f"cannot pick from {source_type}")
        return result_types[source_type]
    binary_types = {
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
    }
    if operation in binary_types:
        _exact_keys(expression, required={"op", "left", "right"}, path=path)
        left = _expression(
            expression["left"],
            path=f"{path}/left",
            environment=environment,
            stats=stats,
        )
        right = _expression(
            expression["right"],
            path=f"{path}/right",
            environment=environment,
            stats=stats,
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
    if operation in {"not", "exists"}:
        _exact_keys(expression, required={"op", "value"}, path=path)
        operand = _expression(
            expression["value"],
            path=f"{path}/value",
            environment=environment,
            stats=stats,
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
    allow_terminal: bool = True,
) -> frozenset[str]:
    stats.nodes += 1
    stats.depth = max(stats.depth, depth)
    if depth > limits.maximum_depth:
        raise _InvalidProgram("depth_limit", path, "AST depth limit exceeded")
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
            environment=environment,
            stats=stats,
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
            allow_terminal=allow_terminal,
        )
    if operation == "if":
        _exact_keys(node, required={"op", "condition", "then", "else"}, path=path)
        condition_type = _expression(
            node["condition"],
            path=f"{path}/condition",
            environment=environment,
            stats=stats,
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
                allow_terminal=allow_terminal,
            )
            try_branch_stats.append(current_stats)
        _merge_runtime_sum(stats, try_branch_stats)
        return try_outcomes | frozenset({"terminal"})
    if operation == "repeat":
        _exact_keys(node, required={"op", "count", "body"}, path=path)
        count = _int64(node["count"], f"{path}/count")
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
            weight = _int64(branch["weight"], f"{path}/branches/{index}/weight")
            if weight <= 0 or weight > _UINT32_MAX:
                raise _InvalidProgram("weight", path, "choose weight must be positive uint32")
            total_weight += weight
            if total_weight > _INT64_MAX:
                raise _InvalidProgram("weight_sum", path, "cumulative weights exceed int64")
            current_stats = _Stats()
            choose_outcomes |= _node(
                branch["body"],
                path=f"{path}/branches/{index}/body",
                depth=depth + 1,
                environment=environment.copy(),
                stats=current_stats,
                limits=limits,
                allow_terminal=allow_terminal,
            )
            choose_branch_stats.append(current_stats)
        _merge_runtime_max(stats, choose_branch_stats)
        return choose_outcomes
    if operation == "apply":
        _exact_keys(node, required={"op", "action_id", "arguments"}, path=path)
        action_id = _identifier(node["action_id"], f"{path}/action_id")
        if action_id not in ACTION_ARGUMENT_TYPES:
            raise _InvalidProgram("unknown_action", path, f"unknown action: {action_id}")
        arguments = _object(node["arguments"], f"{path}/arguments")
        expected = ACTION_ARGUMENT_TYPES[action_id]
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
                environment=environment,
                stats=stats,
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
        if node["reason"] not in {"EXPLICIT", "NO_MATCH", "ILLEGAL_FINAL_STATE", "NO_EFFECT"}:
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
) -> ProgramValidation:
    effective_limits = limits or ProgramLimits()
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
