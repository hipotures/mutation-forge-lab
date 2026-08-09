"""Parse, validate, and identify ordinary-Python policies without executing them."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import cast

from mutation_forge.models import JsonValue

from .contracts import PYTHON_POLICY_PROTOCOL_ID, PYTHON_RESPONSE_SCHEMA_VERSION

PYTHON_SYNTAX_VERSION = "3.12"
VALIDATOR_VERSION = "mforge.native.python_policy_validator.v1"
IDENTITY_PROTOCOL_VERSION = "mforge.native.python_policy_identity.v1"
MAX_SOURCE_BYTES = 32 * 1024
MAX_AST_NODES = 2_000
MAX_AST_DEPTH = 128
MAX_HELPER_FUNCTIONS = 16
MAX_LOOP_TRIPS = 64
MAX_HELPER_CALL_DEPTH = 8
MAX_LITERAL_ITEMS = 256
MAX_STRING_BYTES = 1_024
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

ENTRY_POINT = "propose"
ENTRY_PARAMETERS = ("ctx", "graph", "api", "seed")
HELPER_NAME_PATTERN = re.compile(r"^helper_[A-Za-z][A-Za-z0-9_]{0,55}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

CONTEXT_FIELDS = frozenset(
    {
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
    }
)
GRAPH_FIELDS = frozenset({"order", "edge_count", "minimum_degree", "maximum_degree"})
SELECTOR_METHODS = frozenset(
    {
        "vertices_degree_extreme",
        "vertices_degree_class",
        "vertices_witness_load_extreme",
        "edges_witness_load_extreme",
        "vertices_articulation_risk",
        "edges_bridge_risk",
        "edges_removable",
        "vertices_distance_band",
        "non_edges_from_vertex",
        "non_edges_legal",
        "non_edges_local_cycle_risk",
        "paths_length_two",
        "matching_k_switch_reconnections",
        "matching_k_switch_reconnections_for_edge",
        "relocations_legal",
        "relocations_legal_for_edge",
        "edge_fanouts_legal",
        "edge_fanouts_legal_for_edge",
    }
)
ACTION_METHODS = frozenset(
    {
        "add_edge",
        "remove_edge",
        "relocate_endpoint",
        "k_switch",
        "edge_fanout",
        "edge_fold",
        "emit",
        "no_plan",
    }
)
API_METHODS = SELECTOR_METHODS | ACTION_METHODS | {"pick"}
SAFE_BUILTINS = frozenset(
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

_ALLOWED_NODE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Expr,
    ast.Name,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.IfExp,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.UAdd,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.BitOr,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Load,
    ast.Store,
    ast.TypeIgnore,
)


@dataclass(frozen=True, slots=True)
class PythonPolicyDiagnostic:
    code: str
    path: str
    message: str
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True, slots=True)
class PythonPolicyResponse:
    schema_version: str
    source: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {"schema_version": self.schema_version, "source": self.source}


@dataclass(frozen=True, slots=True)
class PythonProgramIdentityV1:
    source_sha256: str
    canonical_ast_sha256: str | None
    program_hash: str | None
    ast_node_count: int
    helper_function_count: int
    protocol_id: str = PYTHON_POLICY_PROTOCOL_ID
    identity_protocol: str = IDENTITY_PROTOCOL_VERSION
    validator_version: str = VALIDATOR_VERSION
    python_syntax_version: str = PYTHON_SYNTAX_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "source_sha256": self.source_sha256,
            "canonical_ast_sha256": self.canonical_ast_sha256,
            "program_hash": self.program_hash,
            "ast_node_count": self.ast_node_count,
            "helper_function_count": self.helper_function_count,
            "protocol_id": self.protocol_id,
            "identity_protocol": self.identity_protocol,
            "validator_version": self.validator_version,
            "python_syntax_version": self.python_syntax_version,
        }


@dataclass(frozen=True, slots=True)
class PythonPolicyValidation:
    valid: bool
    response: PythonPolicyResponse | None
    identity: PythonProgramIdentityV1 | None
    diagnostics: tuple[PythonPolicyDiagnostic, ...]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "valid": self.valid,
            "response": None if self.response is None else self.response.as_dict(),
            "identity": None if self.identity is None else self.identity.as_dict(),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
        }


class _DuplicateKeyError(ValueError):
    pass


def normalize_source_newlines(source: str) -> str:
    """Normalize CRLF and lone CR without otherwise changing exact source."""

    return source.replace("\r\n", "\n").replace("\r", "\n")


def _diagnostic(
    code: str,
    path: str,
    message: str,
    node: ast.AST | None = None,
) -> PythonPolicyDiagnostic:
    return PythonPolicyDiagnostic(
        code=code,
        path=path,
        message=message,
        line=None if node is None else getattr(node, "lineno", None),
        column=None if node is None else getattr(node, "col_offset", None),
        end_line=None if node is None else getattr(node, "end_lineno", None),
        end_column=None if node is None else getattr(node, "end_col_offset", None),
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_response(
    raw: str | bytes,
) -> tuple[PythonPolicyResponse | None, list[PythonPolicyDiagnostic]]:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            return None, [
                _diagnostic(
                    "INVALID_UTF8",
                    "$",
                    f"response is not valid UTF-8 at byte {error.start}",
                )
            ]
    else:
        text = raw
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as error:
        return None, [_diagnostic("INVALID_ENVELOPE_JSON", "$", str(error))]
    if not isinstance(value, dict):
        return None, [_diagnostic("INVALID_ENVELOPE_TYPE", "$", "response must be an object")]
    keys = set(value)
    expected = {"schema_version", "source"}
    if keys != expected:
        return None, [
            _diagnostic(
                "INVALID_ENVELOPE_FIELDS",
                "$",
                f"response fields must be exactly {sorted(expected)!r}; got {sorted(keys)!r}",
            )
        ]
    schema_version = value["schema_version"]
    source = value["source"]
    if schema_version != PYTHON_RESPONSE_SCHEMA_VERSION:
        return None, [
            _diagnostic(
                "INVALID_SCHEMA_VERSION",
                "$.schema_version",
                f"expected {PYTHON_RESPONSE_SCHEMA_VERSION!r}",
            )
        ]
    if not isinstance(source, str):
        return None, [
            _diagnostic("INVALID_SOURCE_TYPE", "$.source", "source must be a string")
        ]
    return PythonPolicyResponse(
        schema_version=PYTHON_RESPONSE_SCHEMA_VERSION,
        source=normalize_source_newlines(source),
    ), []


def _return_annotation_is_exact(node: ast.expr | None) -> bool:
    return node is None or (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and isinstance(node.left, ast.Name)
        and node.left.id == "RewritePlan"
        and isinstance(node.right, ast.Name)
        and node.right.id == "NoPlan"
    )


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _bound_names(function: ast.FunctionDef) -> set[str]:
    result = {argument.arg for argument in function.args.args}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            result.add(node.id)
    return result


class _PolicyValidator(ast.NodeVisitor):
    def __init__(self, functions: dict[str, ast.FunctionDef]) -> None:
        self.functions = functions
        self.helper_names = frozenset(functions) - {ENTRY_POINT}
        self.current_function: ast.FunctionDef | None = None
        self.current_bound_names: set[str] = set()
        self.selector_locals: set[str] = set()
        self.call_graph: dict[str, set[str]] = {name: set() for name in functions}
        self.diagnostics: list[PythonPolicyDiagnostic] = []
        self.loop_depth = 0

    def error(self, node: ast.AST, code: str, message: str) -> None:
        self.diagnostics.append(_diagnostic(code, "$.source", message, node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.current_function is not None:
            self.error(node, "FORBIDDEN_NESTED_FUNCTION", "nested functions are forbidden")
            return
        previous_function = self.current_function
        previous_names = self.current_bound_names
        previous_selectors = self.selector_locals
        self.current_function = node
        self.current_bound_names = _bound_names(node)
        self.selector_locals = set()

        if node.decorator_list:
            self.error(node, "FORBIDDEN_DECORATOR", "function decorators are forbidden")
        if node.type_params:
            self.error(node, "FORBIDDEN_TYPE_PARAMETER", "type parameters are forbidden")
        if node.args.posonlyargs or node.args.kwonlyargs:
            self.error(
                node,
                "INVALID_FUNCTION_SIGNATURE",
                "positional-only and keyword-only parameters are forbidden",
            )
        if node.args.vararg is not None or node.args.kwarg is not None:
            self.error(node, "INVALID_FUNCTION_SIGNATURE", "variadic parameters are forbidden")
        if node.args.defaults or node.args.kw_defaults:
            self.error(node, "INVALID_FUNCTION_SIGNATURE", "default arguments are forbidden")
        if any(argument.annotation is not None for argument in node.args.args):
            self.error(
                node,
                "FORBIDDEN_PARAMETER_ANNOTATION",
                "parameter annotations are forbidden",
            )
        parameter_names = tuple(argument.arg for argument in node.args.args)
        if len(set(parameter_names)) != len(parameter_names):
            self.error(node, "DUPLICATE_PARAMETER", "parameter names must be unique")
        if node.name == ENTRY_POINT:
            if parameter_names != ENTRY_PARAMETERS:
                self.error(
                    node,
                    "INVALID_ENTRY_POINT_SIGNATURE",
                    f"propose parameters must be exactly {ENTRY_PARAMETERS!r}",
                )
            if not _return_annotation_is_exact(node.returns):
                self.error(
                    node,
                    "INVALID_RETURN_ANNOTATION",
                    "return annotation must be omitted or exactly RewritePlan | NoPlan",
                )
        else:
            reserved_parameters = (
                set(ENTRY_PARAMETERS)
                | set(self.functions)
                | SAFE_BUILTINS
                | {"RewritePlan", "NoPlan"}
            )
            for argument in node.args.args:
                if argument.arg in reserved_parameters:
                    self.error(
                        argument,
                        "SHADOWED_RESERVED_NAME",
                        f"helper parameter {argument.arg!r} shadows a reserved name",
                    )
            if node.returns is not None:
                self.error(
                    node,
                    "FORBIDDEN_HELPER_ANNOTATION",
                    "helper return annotations are forbidden",
                )

        for argument in node.args.args:
            self._validate_identifier(argument, argument.arg)

        for index, statement in enumerate(node.body):
            if _is_docstring(statement) and index == 0:
                self.visit(cast(ast.Expr, statement).value)
                continue
            if _is_docstring(statement):
                self.error(
                    statement,
                    "FORBIDDEN_STRING_EXPRESSION",
                    "only a leading function docstring may be a string expression",
                )
                continue
            self.visit(statement)

        self.current_function = previous_function
        self.current_bound_names = previous_names
        self.selector_locals = previous_selectors

    def visit_Expr(self, node: ast.Expr) -> None:
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "api"
            and node.value.func.attr in ACTION_METHODS
        ):
            self.error(
                node,
                "FORBIDDEN_EXPRESSION_STATEMENT",
                "expression statements must be permitted API action calls",
            )
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.error(node, "FORBIDDEN_ASSIGNMENT_TARGET", "assignment target must be one name")
        else:
            target = node.targets[0]
            self._validate_local_target(target)
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "api"
                and node.value.func.attr in SELECTOR_METHODS
            ):
                self.selector_locals.add(target.id)
            else:
                self.selector_locals.discard(target.id)
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            self.error(
                node,
                "FORBIDDEN_ASSIGNMENT_TARGET",
                "augmented assignment target must be one local name",
            )
        else:
            self._validate_local_target(node.target)
            self.selector_locals.discard(node.target.id)
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.FloorDiv | ast.Mod):
            self.error(node, "FORBIDDEN_OPERATOR", "augmented assignment operator is forbidden")
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            self.error(node, "FORBIDDEN_LOOP_TARGET", "for-loop target must be one local name")
        else:
            self._validate_local_target(node.target)
        if node.orelse:
            self.error(node, "FORBIDDEN_FOR_ELSE", "for-else is not in the allowed subset")
        if not self._bounded_iterator(node.iter):
            self.error(
                node.iter,
                "UNBOUNDED_FOR_ITERATOR",
                f"for iterator must have a statically proven bound of at most {MAX_LOOP_TRIPS}",
            )
        self.visit(node.iter)
        self.loop_depth += 1
        for statement in node.body:
            self.visit(statement)
        self.loop_depth -= 1
        for statement in node.orelse:
            self.visit(statement)

    def visit_Break(self, node: ast.Break) -> None:
        if self.loop_depth == 0:
            self.error(node, "BREAK_OUTSIDE_LOOP", "break is permitted only inside a for loop")

    def visit_Continue(self, node: ast.Continue) -> None:
        if self.loop_depth == 0:
            self.error(
                node,
                "CONTINUE_OUTSIDE_LOOP",
                "continue is permitted only inside a for loop",
            )

    def visit_Call(self, node: ast.Call) -> None:
        if any(keyword.arg is None for keyword in node.keywords):
            self.error(node, "FORBIDDEN_KEYWORD_EXPANSION", "** keyword expansion is forbidden")
        if isinstance(node.func, ast.Name):
            target = node.func.id
            if target not in SAFE_BUILTINS and target not in self.helper_names:
                self.error(node, "FORBIDDEN_CALL", f"call target {target!r} is not allowlisted")
            if target in self.helper_names and self.current_function is not None:
                self.call_graph[self.current_function.name].add(target)
            self._validate_identifier(node.func, target)
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "api"
        ):
            if node.func.attr not in API_METHODS:
                self.error(
                    node.func,
                    "FORBIDDEN_API_CALL",
                    f"api method {node.func.attr!r} is not allowlisted",
                )
        else:
            self.error(node, "FORBIDDEN_CALL", "dynamic and method calls are forbidden")
            self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "ctx"
            and node.attr in CONTEXT_FIELDS
            and isinstance(node.ctx, ast.Load)
        ):
            return
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "graph"
            and node.attr in GRAPH_FIELDS
            and isinstance(node.ctx, ast.Load)
        ):
            return
        self.error(
            node,
            "FORBIDDEN_ATTRIBUTE",
            "only documented ctx and graph scalar attributes may be read",
        )

    def visit_Name(self, node: ast.Name) -> None:
        self._validate_identifier(node, node.id)
        allowed = (
            self.current_bound_names
            | self.helper_names
            | SAFE_BUILTINS
            | {"RewritePlan", "NoPlan"}
        )
        if isinstance(node.ctx, ast.Load) and node.id not in allowed:
            self.error(node, "UNKNOWN_NAME", f"name {node.id!r} is not defined by the contract")

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            if not INT64_MIN <= value <= INT64_MAX:
                self.error(node, "INTEGER_OUT_OF_RANGE", "integer must fit signed 64 bits")
            return
        if isinstance(value, str):
            if not value.isprintable() or len(value.encode("utf-8")) > MAX_STRING_BYTES:
                self.error(
                    node,
                    "INVALID_STRING_LITERAL",
                    f"strings must be printable and at most {MAX_STRING_BYTES} UTF-8 bytes",
                )
            return
        self.error(
            node,
            "FORBIDDEN_CONSTANT",
            "only None, Boolean, signed 64-bit integer, and bounded string literals are allowed",
        )

    def visit_List(self, node: ast.List) -> None:
        self._visit_sequence_literal(node, node.elts)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        self._visit_sequence_literal(node, node.elts)

    def visit_Dict(self, node: ast.Dict) -> None:
        if len(node.keys) > MAX_LITERAL_ITEMS:
            self.error(
                node,
                "LITERAL_TOO_LARGE",
                f"dictionary literals may contain at most {MAX_LITERAL_ITEMS} items",
            )
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                self.error(node, "FORBIDDEN_DICT_EXPANSION", "dictionary expansion is forbidden")
            else:
                if not (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str | int)
                    and not isinstance(key.value, bool)
                ):
                    self.error(
                        key,
                        "INVALID_DICT_KEY",
                        "dictionary keys must be strings or integers",
                    )
                self.visit(key)
            self.visit(value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Store):
            self.error(node, "FORBIDDEN_ASSIGNMENT_TARGET", "subscript mutation is forbidden")
        if isinstance(node.value, ast.Name) and node.value.id in {"ctx", "graph", "api"}:
            self.error(
                node,
                "FORBIDDEN_SUBSCRIPT",
                "ctx, graph, and api do not support subscription access",
            )
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.FloorDiv | ast.Mod):
            self.error(node, "FORBIDDEN_OPERATOR", "binary operator is forbidden")
        self.visit(node.left)
        self.visit(node.right)

    def _visit_sequence_literal(self, node: ast.AST, values: list[ast.expr]) -> None:
        if len(values) > MAX_LITERAL_ITEMS:
            self.error(
                node,
                "LITERAL_TOO_LARGE",
                f"literal containers may contain at most {MAX_LITERAL_ITEMS} items",
            )
        for value in values:
            self.visit(value)

    def _validate_identifier(self, node: ast.AST, name: str) -> None:
        if not IDENTIFIER_PATTERN.fullmatch(name):
            self.error(
                node,
                "INVALID_IDENTIFIER",
                "identifiers must be public ASCII names of at most 64 characters",
            )

    def _validate_local_target(self, node: ast.Name) -> None:
        self._validate_identifier(node, node.id)
        reserved = set(ENTRY_PARAMETERS) | set(self.functions) | SAFE_BUILTINS | {
            "RewritePlan",
            "NoPlan",
        }
        if node.id in reserved:
            self.error(node, "RESERVED_NAME_ASSIGNMENT", f"name {node.id!r} is reserved")

    def _bounded_iterator(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.selector_locals
        if isinstance(node, ast.List | ast.Tuple):
            return len(node.elts) <= MAX_LOOP_TRIPS
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return False
        if node.keywords:
            return False
        if node.func.id == "range":
            values = [_static_integer(argument) for argument in node.args]
            if not 1 <= len(values) <= 3 or any(value is None for value in values):
                return False
            integer_values = cast(list[int], values)
            try:
                return len(range(*integer_values)) <= MAX_LOOP_TRIPS
            except (OverflowError, ValueError):
                return False
        if node.func.id in {"enumerate", "reversed"} and len(node.args) == 1:
            return self._bounded_iterator(node.args[0])
        return False


def _static_integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value, bool
    ):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.UAdd | ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        return node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
    return None


def _validate_call_graph(
    validator: _PolicyValidator,
    functions: dict[str, ast.FunctionDef],
) -> None:
    def depth(name: str, path: tuple[str, ...]) -> int:
        if name in path:
            validator.error(
                functions[name],
                "RECURSIVE_HELPER_CALL",
                f"recursive helper call cycle: {' -> '.join((*path, name))}",
            )
            return MAX_HELPER_CALL_DEPTH + 1
        children = validator.call_graph[name]
        if not children:
            return 0
        return 1 + max(depth(child, (*path, name)) for child in sorted(children))

    maximum = max(depth(name, ()) for name in sorted(functions))
    if maximum > MAX_HELPER_CALL_DEPTH:
        validator.error(
            functions[ENTRY_POINT],
            "HELPER_CALL_DEPTH_EXCEEDED",
            f"helper call depth exceeds {MAX_HELPER_CALL_DEPTH}",
        )
    reachable: set[str] = set()
    pending = [ENTRY_POINT]
    while pending:
        current = pending.pop()
        for child in sorted(validator.call_graph[current]):
            if child not in reachable:
                reachable.add(child)
                pending.append(child)
    for helper in sorted(validator.helper_names - reachable):
        validator.error(
            functions[helper],
            "UNREACHABLE_HELPER",
            f"helper {helper!r} is not reachable from propose",
        )


def _identity_tree(tree: ast.Module) -> ast.Module:
    normalized = copy.deepcopy(tree)
    normalized.type_ignores.clear()
    for node in ast.walk(normalized):
        if hasattr(node, "type_comment"):
            node.type_comment = None
        if isinstance(node, ast.FunctionDef) and node.body and _is_docstring(node.body[0]):
            del node.body[0]
    return normalized


def _ast_depth(tree: ast.AST) -> int:
    maximum = 0
    pending: list[tuple[ast.AST, int]] = [(tree, 1)]
    while pending:
        node, depth = pending.pop()
        maximum = max(maximum, depth)
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return maximum


def _identity(
    source: str,
    tree: ast.Module | None,
    *,
    node_count: int,
    helper_count: int,
) -> PythonProgramIdentityV1:
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if tree is None:
        return PythonProgramIdentityV1(
            source_sha256=source_sha256,
            canonical_ast_sha256=None,
            program_hash=None,
            ast_node_count=node_count,
            helper_function_count=helper_count,
        )
    canonical_ast_bytes = ast.dump(
        _identity_tree(tree),
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    canonical_ast_sha256 = hashlib.sha256(canonical_ast_bytes).hexdigest()
    prefix = (
        b"mforge-native-python-policy-v1\0"
        + b"python-3.12\0"
        + b"validator-v1\0"
    )
    return PythonProgramIdentityV1(
        source_sha256=source_sha256,
        canonical_ast_sha256=canonical_ast_sha256,
        program_hash=hashlib.sha256(prefix + canonical_ast_bytes).hexdigest(),
        ast_node_count=node_count,
        helper_function_count=helper_count,
    )


def validate_python_policy_source(source: str) -> PythonPolicyValidation:
    """Validate and identify source with parsing only; generated code is never invoked."""

    normalized_source = normalize_source_newlines(source)
    response = PythonPolicyResponse(
        schema_version=PYTHON_RESPONSE_SCHEMA_VERSION,
        source=normalized_source,
    )
    try:
        source_bytes = normalized_source.encode("utf-8")
    except UnicodeEncodeError as error:
        return PythonPolicyValidation(
            valid=False,
            response=response,
            identity=None,
            diagnostics=(
                _diagnostic(
                    "INVALID_SOURCE_UTF8",
                    "$.source",
                    f"source cannot be encoded as UTF-8 at character {error.start}",
                ),
            ),
        )
    if len(source_bytes) > MAX_SOURCE_BYTES:
        return PythonPolicyValidation(
            valid=False,
            response=response,
            identity=_identity(normalized_source, None, node_count=0, helper_count=0),
            diagnostics=(
                _diagnostic(
                    "SOURCE_TOO_LARGE",
                    "$.source",
                    f"source is {len(source_bytes)} bytes; maximum is {MAX_SOURCE_BYTES}",
                ),
            ),
        )
    if "\x00" in normalized_source:
        return PythonPolicyValidation(
            valid=False,
            response=response,
            identity=_identity(normalized_source, None, node_count=0, helper_count=0),
            diagnostics=(
                _diagnostic("SOURCE_CONTAINS_NUL", "$.source", "source contains a NUL byte"),
            ),
        )
    try:
        tree = ast.parse(
            normalized_source,
            filename="<generated-python-policy>",
            mode="exec",
            type_comments=True,
            feature_version=(3, 12),
        )
    except RecursionError:
        return PythonPolicyValidation(
            valid=False,
            response=response,
            identity=_identity(normalized_source, None, node_count=0, helper_count=0),
            diagnostics=(
                _diagnostic(
                    "AST_TOO_DEEP",
                    "$.source",
                    f"AST exceeds the maximum supported depth of {MAX_AST_DEPTH}",
                ),
            ),
        )
    except SyntaxError as error:
        diagnostic = PythonPolicyDiagnostic(
            code="SYNTAX_ERROR",
            path="$.source",
            message=error.msg,
            line=error.lineno,
            column=error.offset,
            end_line=error.end_lineno,
            end_column=error.end_offset,
        )
        return PythonPolicyValidation(
            valid=False,
            response=response,
            identity=_identity(normalized_source, None, node_count=0, helper_count=0),
            diagnostics=(diagnostic,),
        )

    nodes = tuple(ast.walk(tree))
    tree_depth = _ast_depth(tree)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    helper_count = sum(function.name != ENTRY_POINT for function in functions)
    diagnostics: list[PythonPolicyDiagnostic] = []
    if len(nodes) > MAX_AST_NODES:
        diagnostics.append(
            _diagnostic(
                "AST_TOO_LARGE",
                "$.source",
                f"AST contains {len(nodes)} nodes; maximum is {MAX_AST_NODES}",
            )
        )
    if tree_depth > MAX_AST_DEPTH:
        diagnostics.append(
            _diagnostic(
                "AST_TOO_DEEP",
                "$.source",
                f"AST depth is {tree_depth}; maximum is {MAX_AST_DEPTH}",
            )
        )
    for node in nodes:
        if type(node) not in _ALLOWED_NODE_TYPES:
            diagnostics.append(
                _diagnostic(
                    "FORBIDDEN_AST_NODE",
                    "$.source",
                    f"AST node {type(node).__name__} is not allowlisted",
                    node,
                )
            )
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        diagnostics.append(
            _diagnostic(
                "FORBIDDEN_TOP_LEVEL",
                "$.source",
                "only function definitions are permitted at module scope",
            )
        )
    entry_points = [function for function in functions if function.name == ENTRY_POINT]
    if len(entry_points) != 1:
        diagnostics.append(
            _diagnostic(
                "INVALID_ENTRY_POINT_COUNT",
                "$.source",
                "source must define exactly one top-level propose function",
            )
        )
    if helper_count > MAX_HELPER_FUNCTIONS:
        diagnostics.append(
            _diagnostic(
                "TOO_MANY_HELPERS",
                "$.source",
                f"source defines {helper_count} helpers; maximum is {MAX_HELPER_FUNCTIONS}",
            )
        )
    for function in functions:
        if function.name != ENTRY_POINT and not HELPER_NAME_PATTERN.fullmatch(function.name):
            diagnostics.append(
                _diagnostic(
                    "INVALID_HELPER_NAME",
                    "$.source",
                    "module-local helpers must match helper_[A-Za-z][A-Za-z0-9_]{0,55}",
                    function,
                )
            )
    function_map = {function.name: function for function in functions}
    if len(function_map) != len(functions):
        diagnostics.append(
            _diagnostic(
                "DUPLICATE_FUNCTION_NAME",
                "$.source",
                "top-level function names must be unique",
            )
        )

    if (
        tree_depth <= MAX_AST_DEPTH
        and len(entry_points) == 1
        and len(function_map) == len(functions)
    ):
        validator = _PolicyValidator(function_map)
        for function in functions:
            validator.visit(function)
        _validate_call_graph(validator, function_map)
        diagnostics.extend(validator.diagnostics)

    identity = _identity(
        normalized_source,
        tree if tree_depth <= MAX_AST_DEPTH else None,
        node_count=len(nodes),
        helper_count=helper_count,
    )
    return PythonPolicyValidation(
        valid=not diagnostics,
        response=response,
        identity=identity,
        diagnostics=tuple(diagnostics),
    )


def validate_python_policy_response(raw: str | bytes) -> PythonPolicyValidation:
    """Validate the exact response envelope and its ordinary Python source."""

    response, diagnostics = _parse_response(raw)
    if response is None:
        return PythonPolicyValidation(
            valid=False,
            response=None,
            identity=None,
            diagnostics=tuple(diagnostics),
        )
    return validate_python_policy_source(response.source)


def accepted_ast_node_names() -> tuple[str, ...]:
    """Return the complete explicit allowlist for contract fixtures and reports."""

    return tuple(sorted(node_type.__name__ for node_type in _ALLOWED_NODE_TYPES))
