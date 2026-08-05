"""Bounded interpreter for validated Native v3 graph-mutation programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import Any, NoReturn, cast

from mutation_forge.models import GraphState, RewritePlan

from .contracts import (
    ACTION_ARGUMENT_TYPES,
    CTX_TYPES,
    FEATURE_TYPES,
    PROGRAM_SCHEMA_VERSION,
    SELECTOR_ARGUMENT_TYPES,
    SELECTOR_COSTS,
    SELECTOR_TYPES,
    ValidatedProgram,
    ValueType,
)
from .graph_runtime import (
    EdgeRef,
    EdgeSetRef,
    GraphFeatureInput,
    GraphFinalStateError,
    GraphPreconditionError,
    GraphResourceError,
    GraphRuntime,
    MatchingRef,
    NonEdgeRef,
    PathRef,
    Population,
    ReferenceValue,
    RewriteHost,
    SelectionPopulation,
    VertexRef,
    VertexSetRef,
    population_items,
    population_type,
    reference_type,
)
from .randomness import RANDOM_PROTOCOL_ID, derive_seed64, uniform_below, weighted_index

INTERPRETER_PROTOCOL_ID = "native_v3_graph_interpreter_v1"

type Scalar = bool | int | Fraction | str | ReferenceValue
type RuntimeValue = Scalar | Population


class BranchFailureCode(StrEnum):
    """The complete, closed class of failures that ``try`` may catch."""

    NO_MATCH = "NO_MATCH"
    LOCAL_PRECONDITION_FAILED = "LOCAL_PRECONDITION_FAILED"
    ILLEGAL_FINAL_STATE = "ILLEGAL_FINAL_STATE"
    NO_EFFECT = "NO_EFFECT"


class CatchableBranchFailure(Exception):
    """A documented selector/action failure eligible for ordered fallback."""

    def __init__(self, code: BranchFailureCode, message: str = "") -> None:
        super().__init__(message or code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProgramContext:
    step_index: int
    horizon: int
    acceptance_profile_id: str
    stagnation_steps: int = 0
    exploration_window_index: int = 0
    accepted_rewrites: int = 0
    accepted_non_improving_rewrites: int = 0
    consecutive_non_improving_rewrites: int = 0
    witness_cap: int = 0
    invocation_ordinal: int = 0

    def values(self) -> dict[str, Scalar]:
        return {
            "step_index": self.step_index,
            "horizon": self.horizon,
            "acceptance_profile_id": self.acceptance_profile_id,
            "stagnation_steps": self.stagnation_steps,
            "exploration_window_index": self.exploration_window_index,
            "accepted_rewrites": self.accepted_rewrites,
            "accepted_non_improving_rewrites": self.accepted_non_improving_rewrites,
            "consecutive_non_improving_rewrites": (self.consecutive_non_improving_rewrites),
            "witness_cap": self.witness_cap,
        }


@dataclass(frozen=True, slots=True)
class InterpreterLimits:
    maximum_steps: int = 1_024
    maximum_repeat_iterations: int = 64
    maximum_choices: int = 64
    maximum_bindings: int = 64
    maximum_selector_calls: int = 64
    maximum_selector_cost_units: int = 128
    maximum_actions: int = 64
    maximum_net_added_edges: int = 8
    maximum_net_removed_edges: int = 8
    maximum_random_draws: int = 2_048
    maximum_integer_bits: int = 64
    maximum_denominator_bits: int = 32

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.maximum_steps,
                self.maximum_repeat_iterations,
                self.maximum_choices,
                self.maximum_bindings,
                self.maximum_selector_calls,
                self.maximum_selector_cost_units,
                self.maximum_actions,
                self.maximum_net_added_edges,
                self.maximum_net_removed_edges,
                self.maximum_random_draws,
            )
        ):
            raise ValueError("interpreter limits must be non-negative")
        if self.maximum_integer_bits < 1 or self.maximum_denominator_bits < 1:
            raise ValueError("numeric bit limits must be positive")


@dataclass(frozen=True, slots=True)
class NoPlan:
    reason: str


@dataclass(frozen=True, slots=True)
class ProgramFailure:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class InvocationCounters:
    steps: int
    repeat_iterations: int
    choices: int
    bindings: int
    selector_calls: int
    selector_cost_units: int
    actions: int
    random_draws: int


@dataclass(frozen=True, slots=True)
class InvocationResult:
    rewrite: RewritePlan | None
    no_plan: NoPlan | None
    failure: ProgramFailure | None
    counters: InvocationCounters

    @property
    def successful(self) -> bool:
        return (self.rewrite is not None or self.no_plan is not None) and self.failure is None


class _ProgramFault(Exception):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.failure = ProgramFailure(code, path, message)


class _Terminal(Exception):
    def __init__(self, result: RewritePlan | NoPlan) -> None:
        super().__init__(type(result).__name__)
        self.result = result


@dataclass(slots=True)
class _Counters:
    steps: int = 0
    repeat_iterations: int = 0
    choices: int = 0
    bindings: int = 0
    selector_calls: int = 0
    selector_cost_units: int = 0
    actions: int = 0
    random_draws: int = 0

    def frozen(self) -> InvocationCounters:
        return InvocationCounters(
            self.steps,
            self.repeat_iterations,
            self.choices,
            self.bindings,
            self.selector_calls,
            self.selector_cost_units,
            self.actions,
            self.random_draws,
        )


@dataclass(slots=True)
class _Runtime:
    program: ValidatedProgram
    graph_runtime: GraphRuntime
    rewrite_host: RewriteHost
    context: ProgramContext
    episode_id: str
    limits: InterpreterLimits
    counters: _Counters = field(default_factory=_Counters)

    def budget(self, field_name: str, increment: int, maximum: int, path: str) -> None:
        value = cast(int, getattr(self.counters, field_name)) + increment
        setattr(self.counters, field_name, value)
        if value > maximum:
            raise _ProgramFault(
                "BUDGET_EXHAUSTED",
                path,
                f"{field_name} budget exceeded: {value} > {maximum}",
            )

    def step(self, path: str) -> None:
        self.budget("steps", 1, self.limits.maximum_steps, path)

    def random_index(self, path: str, weights: Sequence[int]) -> int:
        seed = derive_seed64(
            RANDOM_PROTOCOL_ID,
            INTERPRETER_PROTOCOL_ID,
            self.program.program_hash,
            self.episode_id,
            self.context.step_index,
            self.context.invocation_ordinal,
            path,
        )
        try:
            if all(weight == 1 for weight in weights):
                index, draws = uniform_below(seed, len(weights))
            else:
                index, draws = weighted_index(seed, weights)
        except ValueError as exc:
            raise _ProgramFault("TYPE_ERROR", path, str(exc)) from exc
        self.budget(
            "random_draws",
            draws,
            self.limits.maximum_random_draws,
            path,
        )
        return index


def _runtime_type(value: RuntimeValue) -> ValueType:
    if isinstance(value, bool):
        return ValueType.BOOL
    if isinstance(value, int):
        return ValueType.INT
    if isinstance(value, Fraction):
        return ValueType.RATIONAL
    if isinstance(value, str):
        return ValueType.STRING
    if isinstance(value, VertexRef | EdgeRef | NonEdgeRef | PathRef | MatchingRef):
        return reference_type(value)
    if isinstance(value, VertexSetRef | EdgeSetRef | SelectionPopulation):
        return population_type(value)
    raise TypeError(f"unsupported runtime value: {type(value).__name__}")


def _fault(code: str, path: str, message: str) -> NoReturn:
    raise _ProgramFault(code, path, message)


def _expect(value: RuntimeValue, expected: ValueType, path: str) -> RuntimeValue:
    try:
        actual = _runtime_type(value)
    except TypeError as exc:
        _fault("TYPE_ERROR", path, str(exc))
    if actual is not expected:
        _fault("TYPE_ERROR", path, f"expected {expected}, got {actual}")
    return value


def _number(runtime: _Runtime, value: int | Fraction, path: str) -> int | Fraction:
    numerator = value if isinstance(value, int) else value.numerator
    denominator = 1 if isinstance(value, int) else value.denominator
    if numerator.bit_length() > runtime.limits.maximum_integer_bits:
        _fault("BUDGET_EXHAUSTED", path, "integer bit budget exceeded")
    if denominator.bit_length() > runtime.limits.maximum_denominator_bits:
        _fault("BUDGET_EXHAUSTED", path, "rational denominator bit budget exceeded")
    return value


def _expression(
    runtime: _Runtime,
    raw: object,
    environment: Mapping[str, RuntimeValue],
    path: str,
) -> RuntimeValue:
    runtime.step(path)
    if isinstance(raw, bool | int | str):
        if isinstance(raw, int) and not isinstance(raw, bool):
            return _number(runtime, raw, path)
        return raw
    if not isinstance(raw, dict):
        _fault("INVALID_AST", path, "expression must be an object or scalar")
    expression = cast(dict[str, Any], raw)
    operation = expression.get("op")
    if operation == "ref":
        name = expression.get("name")
        if not isinstance(name, str) or name not in environment:
            _fault("INVALID_AST", path, "unbound or invalid reference")
        return environment[name]
    if operation == "ctx":
        field_name = expression.get("field")
        if not isinstance(field_name, str) or field_name not in CTX_TYPES:
            _fault("INVALID_AST", path, "unknown context field")
        return _expect(runtime.context.values()[field_name], CTX_TYPES[field_name], path)
    if operation == "feature":
        field_name = expression.get("field")
        if not isinstance(field_name, str) or field_name not in FEATURE_TYPES:
            _fault("INVALID_AST", path, "unknown feature field")
        return _expect(
            runtime.graph_runtime.feature_values()[field_name],
            FEATURE_TYPES[field_name],
            path,
        )
    if operation == "rational":
        numerator = expression.get("numerator")
        denominator = expression.get("denominator")
        if (
            not isinstance(numerator, int)
            or isinstance(numerator, bool)
            or not isinstance(denominator, int)
            or isinstance(denominator, bool)
            or denominator <= 0
        ):
            _fault("INVALID_AST", path, "invalid rational")
        return _number(runtime, Fraction(numerator, denominator), path)
    if operation == "selector":
        selector_id = expression.get("selector_id")
        if not isinstance(selector_id, str) or selector_id not in SELECTOR_TYPES:
            _fault("INVALID_AST", path, "unknown selector")
        raw_arguments = expression.get("arguments")
        if not isinstance(raw_arguments, dict):
            _fault("INVALID_AST", path, "selector arguments must be an object")
        expected_arguments = SELECTOR_ARGUMENT_TYPES[selector_id]
        if set(raw_arguments) != set(expected_arguments):
            _fault("INVALID_AST", path, "invalid selector arguments")
        arguments: dict[str, Scalar] = {}
        for name, expected in expected_arguments.items():
            value = _expression(
                runtime,
                raw_arguments[name],
                environment,
                f"{path}/arguments/{name}",
            )
            arguments[name] = cast(Scalar, _expect(value, expected, path))
        runtime.budget(
            "selector_calls",
            1,
            runtime.limits.maximum_selector_calls,
            path,
        )
        runtime.budget(
            "selector_cost_units",
            SELECTOR_COSTS[selector_id],
            runtime.limits.maximum_selector_cost_units,
            path,
        )
        try:
            selection = runtime.graph_runtime.select(
                selector_id,
                arguments,
                path=path,
                random_index=runtime.random_index,
            )
        except GraphPreconditionError as exc:
            raise CatchableBranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED,
                str(exc),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise _ProgramFault("TYPE_ERROR", path, str(exc)) from exc
        _expect(selection, SELECTOR_TYPES[selector_id], path)
        return selection
    if operation == "pick":
        source = _expression(runtime, expression.get("source"), environment, f"{path}/source")
        if not isinstance(source, VertexSetRef | EdgeSetRef | SelectionPopulation):
            _fault("TYPE_ERROR", path, "pick source is not a selection")
        items = population_items(source)
        if not items:
            raise CatchableBranchFailure(BranchFailureCode.NO_MATCH)
        mode = expression.get("mode")
        if mode == "require_singleton":
            if len(items) != 1:
                raise CatchableBranchFailure(
                    BranchFailureCode.LOCAL_PRECONDITION_FAILED,
                    "selection is not a singleton",
                )
            return items[0]
        runtime.budget("choices", 1, runtime.limits.maximum_choices, path)
        if mode == "seeded_uniform":
            index = runtime.random_index(path, [1] * len(items))
            return items[index]
        if mode == "seeded_weighted":
            feature = expression.get("weight_feature")
            if not isinstance(feature, str):
                _fault("INVALID_AST", path, "weighted pick lacks weight_feature")
            weights: list[int] = []
            for item in items:
                try:
                    weight = runtime.graph_runtime.weight(item, feature)
                except (TypeError, ValueError) as exc:
                    raise _ProgramFault("TYPE_ERROR", path, str(exc)) from exc
                if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                    _fault("TYPE_ERROR", path, "weight must be a positive integer")
                weights.append(weight)
            return items[runtime.random_index(path, weights)]
        _fault("INVALID_AST", path, "unknown pick mode")
    if operation in {
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
    }:
        left = _expression(runtime, expression.get("left"), environment, f"{path}/left")
        right = _expression(runtime, expression.get("right"), environment, f"{path}/right")
        if operation in {"and", "or"}:
            _expect(left, ValueType.BOOL, path)
            _expect(right, ValueType.BOOL, path)
            return bool(left and right) if operation == "and" else bool(left or right)
        if operation == "equal":
            left_type = _runtime_type(left)
            right_type = _runtime_type(right)
            if left_type is not right_type and {left_type, right_type} != {
                ValueType.INT,
                ValueType.RATIONAL,
            }:
                _fault("TYPE_ERROR", path, "incompatible equality operands")
            return left == right
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int | Fraction)
            or not isinstance(right, int | Fraction)
        ):
            _fault("TYPE_ERROR", path, "numeric operands required")
        if operation == "add":
            return _number(runtime, left + right, path)
        if operation == "subtract":
            return _number(runtime, left - right, path)
        if operation == "multiply":
            return _number(runtime, left * right, path)
        if operation == "minimum":
            return min(left, right)
        if operation == "maximum":
            return max(left, right)
        if operation == "less":
            return left < right
        if operation == "less_equal":
            return left <= right
        if operation == "greater":
            return left > right
        if operation == "greater_equal":
            return left >= right
    if operation == "not":
        value = _expression(runtime, expression.get("value"), environment, f"{path}/value")
        _expect(value, ValueType.BOOL, path)
        return not cast(bool, value)
    if operation == "exists":
        try:
            value = _expression(runtime, expression.get("value"), environment, f"{path}/value")
        except CatchableBranchFailure:
            return False
        if isinstance(value, VertexSetRef | EdgeSetRef | SelectionPopulation):
            return bool(population_items(value))
        return value is not None
    _fault("INVALID_AST", path, f"unknown expression: {operation!r}")


def _node(
    runtime: _Runtime,
    raw: object,
    environment: dict[str, RuntimeValue],
    path: str,
) -> None:
    runtime.step(path)
    if not isinstance(raw, dict):
        _fault("INVALID_AST", path, "node must be an object")
    node = cast(dict[str, Any], raw)
    operation = node.get("op")
    if operation == "block":
        children = node.get("children")
        if not isinstance(children, list) or not children:
            _fault("INVALID_AST", path, "block children must be non-empty")
        for index, child in enumerate(children):
            _node(runtime, child, environment.copy(), f"{path}/children/{index}")
        return
    if operation == "let":
        name = node.get("name")
        if not isinstance(name, str) or name in environment:
            _fault("INVALID_AST", path, "invalid binding")
        value = _expression(runtime, node.get("value"), environment, f"{path}/value")
        runtime.budget("bindings", 1, runtime.limits.maximum_bindings, path)
        nested = environment.copy()
        nested[name] = value
        _node(runtime, node.get("body"), nested, f"{path}/body")
        return
    if operation == "if":
        condition = _expression(
            runtime,
            node.get("condition"),
            environment,
            f"{path}/condition",
        )
        _expect(condition, ValueType.BOOL, path)
        branch_name = "then" if condition else "else"
        _node(runtime, node.get(branch_name), environment.copy(), f"{path}/{branch_name}")
        return
    if operation == "try":
        branches = node.get("branches")
        if not isinstance(branches, list) or not branches:
            _fault("INVALID_AST", path, "try branches must be non-empty")
        last_failure: CatchableBranchFailure | None = None
        for index, branch in enumerate(branches):
            graph_snapshot = runtime.graph_runtime.overlay.snapshot()
            environment_snapshot = environment.copy()
            try:
                _node(
                    runtime,
                    branch,
                    environment,
                    f"{path}/branches/{index}",
                )
                return
            except CatchableBranchFailure as exc:
                last_failure = exc
                runtime.graph_runtime.overlay.restore(graph_snapshot)
                environment.clear()
                environment.update(environment_snapshot)
        if last_failure is None:
            _fault("INVALID_AST", path, "try has no branches")
        raise last_failure
    if operation == "repeat":
        count = node.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            _fault("INVALID_AST", path, "repeat count must be positive")
        for index in range(count):
            runtime.budget(
                "repeat_iterations",
                1,
                runtime.limits.maximum_repeat_iterations,
                path,
            )
            _node(
                runtime,
                node.get("body"),
                environment.copy(),
                f"{path}/body@{index}",
            )
        return
    if operation == "choose":
        branches = node.get("branches")
        if not isinstance(branches, list) or not branches:
            _fault("INVALID_AST", path, "choose branches must be non-empty")
        weights: list[int] = []
        for branch in branches:
            if not isinstance(branch, dict):
                _fault("INVALID_AST", path, "choose branch must be an object")
            weight = branch.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                _fault("INVALID_AST", path, "choose weight must be positive")
            weights.append(weight)
        runtime.budget("choices", 1, runtime.limits.maximum_choices, path)
        index = runtime.random_index(path, weights)
        branch = cast(dict[str, Any], branches[index])
        _node(
            runtime,
            branch.get("body"),
            environment.copy(),
            f"{path}/branches/{index}/body",
        )
        return
    if operation == "apply":
        action_id = node.get("action_id")
        if not isinstance(action_id, str) or action_id not in ACTION_ARGUMENT_TYPES:
            _fault("INVALID_AST", path, "unknown action")
        raw_arguments = node.get("arguments")
        if not isinstance(raw_arguments, dict):
            _fault("INVALID_AST", path, "action arguments must be an object")
        expected_arguments = ACTION_ARGUMENT_TYPES[action_id]
        if set(raw_arguments) != set(expected_arguments):
            _fault("INVALID_AST", path, "invalid action arguments")
        arguments: dict[str, Scalar] = {}
        for name, expected in expected_arguments.items():
            value = _expression(
                runtime,
                raw_arguments[name],
                environment,
                f"{path}/arguments/{name}",
            )
            arguments[name] = cast(Scalar, _expect(value, expected, path))
        runtime.budget("actions", 1, runtime.limits.maximum_actions, path)
        try:
            runtime.graph_runtime.apply_action(action_id, arguments)
        except GraphPreconditionError as exc:
            raise CatchableBranchFailure(
                BranchFailureCode.LOCAL_PRECONDITION_FAILED,
                str(exc),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise _ProgramFault("TYPE_ERROR", path, str(exc)) from exc
        return
    if operation == "emit":
        try:
            rewrite = runtime.graph_runtime.emit(
                host=runtime.rewrite_host,
                program_hash=runtime.program.program_hash,
                gross_actions=runtime.counters.actions,
                selector_cost_units=runtime.counters.selector_cost_units,
                maximum_net_added_edges=runtime.limits.maximum_net_added_edges,
                maximum_net_removed_edges=runtime.limits.maximum_net_removed_edges,
            )
        except GraphPreconditionError as exc:
            raise CatchableBranchFailure(BranchFailureCode.NO_EFFECT, str(exc)) from exc
        except GraphFinalStateError as exc:
            raise CatchableBranchFailure(
                BranchFailureCode.ILLEGAL_FINAL_STATE,
                str(exc),
            ) from exc
        except GraphResourceError as exc:
            raise _ProgramFault("BUDGET_EXHAUSTED", path, str(exc)) from exc
        raise _Terminal(rewrite)
    if operation == "no_plan":
        reason = node.get("reason")
        if not isinstance(reason, str):
            _fault("INVALID_AST", path, "NoPlan reason must be a string")
        raise _Terminal(NoPlan(reason))
    _fault("INVALID_AST", path, f"unknown node: {operation!r}")


def _no_plan_reason(code: BranchFailureCode) -> str:
    if code is BranchFailureCode.LOCAL_PRECONDITION_FAILED:
        return BranchFailureCode.NO_MATCH.value
    return code.value


def invoke_program(
    program: ValidatedProgram,
    graph: GraphState,
    *,
    rewrite_host: RewriteHost,
    context: ProgramContext,
    episode_id: str,
    features: GraphFeatureInput | None = None,
    limits: InterpreterLimits | None = None,
) -> InvocationResult:
    """Execute one validated program against a private graph overlay."""

    try:
        graph_runtime = GraphRuntime(graph, features or GraphFeatureInput())
        invocation_snapshot = graph_runtime.overlay.snapshot()
        runtime = _Runtime(
            program=program,
            graph_runtime=graph_runtime,
            rewrite_host=rewrite_host,
            context=context,
            episode_id=episode_id,
            limits=limits or InterpreterLimits(),
        )
        document = program.ast
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != PROGRAM_SCHEMA_VERSION
            or "entry" not in document
        ):
            _fault("INVALID_AST", "/", "program is not a validated Native v3 envelope")
        _node(runtime, document["entry"], {}, "/entry")
        _fault("INVALID_AST", "/entry", "program completed without a terminal")
    except _Terminal as terminal:
        rewrite = terminal.result if isinstance(terminal.result, RewritePlan) else None
        no_plan = terminal.result if isinstance(terminal.result, NoPlan) else None
        return InvocationResult(
            rewrite,
            no_plan,
            None,
            runtime.counters.frozen(),
        )
    except CatchableBranchFailure as exc:
        runtime.graph_runtime.overlay.restore(invocation_snapshot)
        return InvocationResult(
            None,
            NoPlan(_no_plan_reason(exc.code)),
            None,
            runtime.counters.frozen(),
        )
    except _ProgramFault as exc:
        return InvocationResult(None, None, exc.failure, runtime.counters.frozen())
    except Exception as exc:
        counters = runtime.counters.frozen() if "runtime" in locals() else _Counters().frozen()
        return InvocationResult(
            None,
            None,
            ProgramFailure("INTERPRETER_FAULT", "/entry", str(exc)),
            counters,
        )
