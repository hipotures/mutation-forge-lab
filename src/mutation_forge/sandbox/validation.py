from __future__ import annotations

import ast
import copy
import hashlib
import math
from dataclasses import dataclass

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import (
    MAX_INTEGER_BITS,
    MAX_SEQUENCE_ITEMS,
    MAX_STRING_BYTES,
    VALIDATOR_VERSION,
    SandboxLimits,
)

SAFE_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "len",
        "max",
        "min",
        "range",
        "round",
        "sum",
    }
)
PARAMETERS = ("ctx", "proposal")
FORBIDDEN_POLICY_FIELDS = frozenset(
    {
        "ctx.schema_version",
        "ctx.step",
        "ctx.remaining_steps",
        "proposal.schema_version",
        "proposal.proposal_id",
        "proposal.selector_tags",
        "proposal.anchor_forbidden_length",
    }
)
PROPOSAL_SIGNAL_FIELDS = frozenset(
    {
        "proposal.broken_sampled_witnesses_by_length",
        "proposal.removed_edge_load_sum_by_length",
        "proposal.removed_edge_load_max_by_length",
        "proposal.minimum_distance_between_removed_edges",
        "proposal.mean_distance_between_removed_edges",
        "proposal.minimum_preexisting_distance_for_new_edges",
        "proposal.mean_preexisting_distance_for_new_edges",
        "proposal.local_triangle_risk",
        "proposal.local_c4_risk",
        "proposal.reconnection_span",
    }
)


def render_policy_validator_contract(
    limits: SandboxLimits | None = None,
    *,
    scientific: bool = False,
) -> str:
    """Render the model-facing contract from the executable validator rules."""

    configured = limits or SandboxLimits()
    allowed_calls = ", ".join(f"`{name}`" for name in sorted(SAFE_BUILTINS))
    rules = [
        f"## Sandbox contract ({VALIDATOR_VERSION})",
        "",
        "- The source must contain exactly one top-level function "
        "`priority(ctx, proposal)`.",
        "- Do not define helper or nested functions.",
        "- Do not use imports.",
        "- Do not use names beginning with `_`.",
        "- Do not use attribute access or method calls.",
        "- Read `ctx` and `proposal` only with subscription syntax such as "
        '`ctx["field"]` and `proposal["field"]`; nested subscriptions are allowed.',
        "- Direct `ctx` and `proposal` field names must be string literals.",
    ]
    if scientific:
        rules.extend(
            [
                "- Do not read contract identifiers, trajectory counters, proposal IDs, "
                "selector provenance, or anchor provenance: "
                + ", ".join(f"`{name}`" for name in sorted(FORBIDDEN_POLICY_FIELDS))
                + ".",
                "- The returned priority must use at least one proposal-specific "
                "structural signal; a constant or context-only ranker is invalid.",
            ]
        )
    rules.extend(
        [
            "- The function must contain exactly one `return`, and it must be the final "
            "top-level statement.",
            f"- The complete allowed-call whitelist is: {allowed_calls}.",
            "- Do not call any other built-in, function, callable value, or method.",
            "- Generator expressions and all comprehensions are forbidden, including "
            "inside allowed built-ins such as `sum`.",
            f"- Source size must not exceed {configured.max_source_bytes} UTF-8 bytes.",
            f"- The parsed program must not exceed {configured.max_ast_nodes} AST nodes.",
        ]
    )
    return "\n".join(rules)


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
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.Call,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)


@dataclass(frozen=True, slots=True)
class ValidationError:
    code: str
    message: str
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True, slots=True)
class ProgramIdentity:
    source_sha256: str
    normalized_ast_sha256: str | None
    ast_node_count: int
    validator_version: str = VALIDATOR_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "source_sha256": self.source_sha256,
            "normalized_ast_sha256": self.normalized_ast_sha256,
            "ast_node_count": self.ast_node_count,
            "validator_version": self.validator_version,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    identity: ProgramIdentity
    errors: tuple[ValidationError, ...]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "valid": self.valid,
            "identity": self.identity.as_dict(),
            "errors": [error.as_dict() for error in self.errors],
        }


class _PolicyValidator(ast.NodeVisitor):
    def __init__(self, limits: SandboxLimits, *, scientific: bool = False) -> None:
        self.limits = limits
        self.scientific = scientific
        self.errors: list[ValidationError] = []
        self.local_names: set[str] = set()
        self.accessed_input_fields: set[str] = set()

    def error(self, node: ast.AST, code: str, message: str) -> None:
        self.errors.append(
            ValidationError(
                code=code,
                message=message,
                line=getattr(node, "lineno", None),
                column=getattr(node, "col_offset", None),
                end_line=getattr(node, "end_lineno", None),
                end_column=getattr(node, "end_col_offset", None),
            )
        )

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            self.error(
                node,
                "forbidden_syntax",
                f"{type(node).__name__} is not allowed",
            )
            return
        super().generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        if len(node.body) != 1 or not isinstance(node.body[0], ast.FunctionDef):
            self.error(
                node,
                "top_level_contract",
                "source must contain exactly one top-level function",
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name != "priority":
            self.error(node, "wrong_function_name", "function must be named priority")
        if node.name.startswith("_"):
            self.error(node, "private_name", "names beginning '_' are forbidden")
        args = node.args
        valid_args = (
            not args.posonlyargs
            and [arg.arg for arg in args.args] == list(PARAMETERS)
            and args.vararg is None
            and not args.kwonlyargs
            and args.kwarg is None
            and not args.defaults
            and not args.kw_defaults
        )
        if not valid_args:
            self.error(
                node,
                "wrong_signature",
                "signature must be exactly priority(ctx, proposal)",
            )
        if node.decorator_list:
            self.error(node, "decorator", "decorators are forbidden")
        if node.returns is not None or any(arg.annotation for arg in args.args):
            self.error(node, "annotation", "annotations are not allowed")
        return_nodes = [
            child for child in ast.walk(node) if isinstance(child, ast.Return)
        ]
        if len(return_nodes) != 1:
            self.error(
                node,
                "return_contract",
                "priority must contain exactly one return statement",
            )
        elif not node.body or node.body[-1] is not return_nodes[0]:
            self.error(
                return_nodes[0],
                "return_contract",
                "the single return must be the final top-level statement",
            )
        self._collect_locals(node)
        self.generic_visit(node)

    def _collect_locals(self, function: ast.FunctionDef) -> None:
        for node in ast.walk(function):
            target: ast.AST | None = None
            if isinstance(node, ast.Assign):
                for assign_target in node.targets:
                    self._collect_target(assign_target)
            elif isinstance(node, ast.AugAssign | ast.For):
                target = node.target
            if target is not None:
                self._collect_target(target)

    def _collect_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.local_names.add(target.id)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            self.error(node, "private_name", "names beginning '_' are forbidden")
        if isinstance(node.ctx, ast.Store):
            if node.id in PARAMETERS:
                self.error(
                    node,
                    "input_mutation",
                    f"cannot assign to {node.id}",
                )
            if node.id in SAFE_BUILTINS:
                self.error(
                    node,
                    "reserved_name",
                    f"cannot shadow safe built-in {node.id}",
                )
            return
        if node.id not in {*PARAMETERS, *SAFE_BUILTINS, *self.local_names}:
            self.error(node, "unknown_name", f"name {node.id!r} is not allowed")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.error(
                node,
                "assignment_target",
                "assignments may target one local name only",
            )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            self.error(
                node,
                "assignment_target",
                "augmented assignments may target a local name only",
            )
        self.generic_visit(node)

    @staticmethod
    def _literal_int(node: ast.AST) -> int | None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
        ):
            return -node.operand.value
        return None

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_BUILTINS:
            self.error(
                node,
                "forbidden_call",
                "only selected deterministic built-ins may be called",
            )
        if node.keywords:
            self.error(node, "call_keywords", "keyword arguments are not allowed")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in PARAMETERS:
            parameter = node.value.id
            if not (
                isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and node.slice.value
            ):
                self.error(
                    node,
                    "dynamic_input_field",
                    f"{parameter} field names must be non-empty string literals",
                )
            else:
                direct_field = f"{parameter}.{node.slice.value}"
                self.accessed_input_fields.add(direct_field)
                if self.scientific and direct_field in FORBIDDEN_POLICY_FIELDS:
                    self.error(
                        node,
                        "forbidden_input_field",
                        f"{direct_field} is provenance-only and cannot rank proposals",
                    )
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if not (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            self.error(node, "expression_statement", "only a docstring is allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            if value.bit_length() > MAX_INTEGER_BITS:
                self.error(
                    node,
                    "integer_literal_too_large",
                    f"integer literal exceeds {MAX_INTEGER_BITS} bits",
                )
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                self.error(node, "non_finite_literal", "non-finite literals are forbidden")
            return
        if isinstance(value, str):
            if len(value.encode("utf-8")) > MAX_STRING_BYTES:
                self.error(
                    node,
                    "string_literal_too_large",
                    f"string literal exceeds {MAX_STRING_BYTES} UTF-8 bytes",
                )
            return
        self.error(
            node,
            "literal_type",
            f"{type(value).__name__} literals are not allowed",
        )

    def visit_List(self, node: ast.List) -> None:
        if len(node.elts) > MAX_SEQUENCE_ITEMS:
            self.error(node, "literal_too_large", "list literal is too large")
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        if len(node.elts) > MAX_SEQUENCE_ITEMS:
            self.error(node, "literal_too_large", "tuple literal is too large")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if len(node.keys) > MAX_SEQUENCE_ITEMS:
            self.error(node, "literal_too_large", "dict literal is too large")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Pow):
            exponent = self._literal_int(node.right)
            if exponent is not None and abs(exponent) > self.limits.max_static_loop_bound:
                self.error(
                    node,
                    "power_bound",
                    "literal exponent exceeds the static bound",
                )
        if isinstance(node.op, ast.Mult):
            for operand in (node.left, node.right):
                value = self._literal_int(operand)
                if value is not None and abs(value) > self.limits.max_static_loop_bound:
                    self.error(
                        node,
                        "multiplication_bound",
                        "literal multiplier exceeds the static bound",
                    )
        self.generic_visit(node)


class _LocalNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def _canonical(self, name: str) -> str:
        if name in PARAMETERS or name in SAFE_BUILTINS:
            return name
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        normalized = copy.copy(node)
        normalized.id = self._canonical(node.id)
        return normalized


def _normalized_ast_hash(tree: ast.Module) -> str:
    normalized = _LocalNormalizer().visit(copy.deepcopy(tree))
    ast.fix_missing_locations(normalized)
    payload = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def accessed_policy_fields(source: str) -> tuple[str, ...]:
    """Return canonical direct input fields accessed by a valid policy source."""

    tree = ast.parse(source, mode="exec")
    fields = {
        f"{node.value.id}.{node.slice.value}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in PARAMETERS
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        and node.slice.value
    }
    return tuple(sorted(fields))


def validate_policy(
    source: str,
    limits: SandboxLimits | None = None,
    *,
    scientific: bool = False,
) -> ValidationResult:
    applied_limits = limits or SandboxLimits()
    source_bytes = source.encode("utf-8")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if len(source_bytes) > applied_limits.max_source_bytes:
        identity = ProgramIdentity(source_hash, None, 0)
        return ValidationResult(
            False,
            identity,
            (
                ValidationError(
                    "source_too_large",
                    f"source exceeds {applied_limits.max_source_bytes} UTF-8 bytes",
                ),
            ),
        )
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError) as error:
        identity = ProgramIdentity(source_hash, None, 0)
        return ValidationResult(
            False,
            identity,
            (
                ValidationError(
                    "syntax_error",
                    str(error),
                    getattr(error, "lineno", None),
                    getattr(error, "offset", None),
                    getattr(error, "end_lineno", None),
                    getattr(error, "end_offset", None),
                ),
            ),
        )
    node_count = sum(1 for _ in ast.walk(tree))
    validator = _PolicyValidator(applied_limits, scientific=scientific)
    if node_count > applied_limits.max_ast_nodes:
        validator.errors.append(
            ValidationError(
                "ast_too_large",
                f"AST has {node_count} nodes; limit is {applied_limits.max_ast_nodes}",
            )
        )
    validator.visit(tree)
    if scientific and not validator.accessed_input_fields.intersection(
        PROPOSAL_SIGNAL_FIELDS
    ):
        validator.errors.append(
            ValidationError(
                "proposal_signal_required",
                "priority must use at least one proposal-specific structural signal",
            )
        )
    identity = ProgramIdentity(
        source_sha256=source_hash,
        normalized_ast_sha256=_normalized_ast_hash(tree),
        ast_node_count=node_count,
    )
    errors = tuple(
        sorted(
            validator.errors,
            key=lambda item: (
                item.line is None,
                item.line or 0,
                item.column or 0,
                item.code,
            ),
        )
    )
    return ValidationResult(not errors, identity, errors)
