"""Bounded interpreter for validated Native v3 programs over synthetic fixtures."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import Any, NoReturn, Protocol, cast

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
from .randomness import RANDOM_PROTOCOL_ID, derive_seed64, uniform_below, weighted_index

INTERPRETER_PROTOCOL_ID = "native_v3_synthetic_interpreter_v1"


@dataclass(frozen=True, slots=True, order=True)
class VertexRef:
    value: int


@dataclass(frozen=True, slots=True, order=True)
class EdgeRef:
    u: int
    v: int


@dataclass(frozen=True, slots=True, order=True)
class NonEdgeRef:
    u: int
    v: int


@dataclass(frozen=True, slots=True, order=True)
class Path2Ref:
    u: int
    v: int
    w: int


@dataclass(frozen=True, slots=True)
class MatchingRef:
    edges: tuple[EdgeRef, ...]


type Selectable = VertexRef | EdgeRef | NonEdgeRef | Path2Ref | MatchingRef
type Scalar = bool | int | Fraction | str | Selectable
type OverlayValue = bool | int | str
type RuntimeValue = Scalar | Selection


_SELECTION_ITEM_TYPES: dict[ValueType, ValueType] = {
    ValueType.VERTEX_SET: ValueType.VERTEX,
    ValueType.EDGE_SET: ValueType.EDGE,
    ValueType.NON_EDGE: ValueType.NON_EDGE,
    ValueType.PATH2: ValueType.PATH2,
    ValueType.MATCHING: ValueType.MATCHING,
}


@dataclass(frozen=True, slots=True)
class Selection:
    """A typed, ordered population returned by a synthetic selector."""

    value_type: ValueType
    items: tuple[Selectable, ...]

    def __post_init__(self) -> None:
        expected = _SELECTION_ITEM_TYPES.get(self.value_type)
        if expected is None:
            raise ValueError(f"{self.value_type} is not a selector population type")
        if any(_runtime_type(item) is not expected for item in self.items):
            raise ValueError(f"selection items do not match {self.value_type}")


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


class SyntheticFixture(Protocol):
    """Typed host boundary used only by synthetic interpreter fixtures."""

    fixture_id: str

    def select(
        self,
        selector_id: str,
        arguments: Mapping[str, Scalar],
        overlay: Mapping[str, OverlayValue],
    ) -> Selection:
        """Return an ordered typed selector population."""

    def weight(
        self,
        item: Selectable,
        feature: str,
        overlay: Mapping[str, OverlayValue],
    ) -> int:
        """Return a positive exact integer weight for a selectable item."""

    def apply(
        self,
        action_id: str,
        arguments: Mapping[str, Scalar],
        overlay: MutableMapping[str, OverlayValue],
    ) -> None:
        """Apply one action to the private invocation overlay."""

    def validate_emit(self, overlay: Mapping[str, OverlayValue]) -> None:
        """Validate the final private overlay before it is emitted."""


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
            "consecutive_non_improving_rewrites": (
                self.consecutive_non_improving_rewrites
            ),
            "witness_cap": self.witness_cap,
        }


@dataclass(frozen=True, slots=True)
class GraphFeatureInput:
    order: int
    edge_count: int
    minimum_degree: int
    maximum_degree: int

    def values(self) -> dict[str, Scalar]:
        return {
            "order": self.order,
            "edge_count": self.edge_count,
            "minimum_degree": self.minimum_degree,
            "maximum_degree": self.maximum_degree,
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
    maximum_random_draws: int = 128
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
                self.maximum_random_draws,
            )
        ):
            raise ValueError("interpreter limits must be non-negative")
        if self.maximum_integer_bits < 1 or self.maximum_denominator_bits < 1:
            raise ValueError("numeric bit limits must be positive")


class OutcomeKind(StrEnum):
    EMIT = "emit"
    NO_PLAN = "no_plan"


@dataclass(frozen=True, slots=True)
class InvocationOutcome:
    kind: OutcomeKind
    no_plan_reason: str | None
    overlay: tuple[tuple[str, OverlayValue], ...]


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
    outcome: InvocationOutcome | None
    failure: ProgramFailure | None
    counters: InvocationCounters

    @property
    def successful(self) -> bool:
        return self.outcome is not None and self.failure is None


class _ProgramFault(Exception):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.failure = ProgramFailure(code, path, message)


class _Terminal(Exception):
    def __init__(self, kind: OutcomeKind, reason: str | None = None) -> None:
        super().__init__(kind.value)
        self.kind = kind
        self.reason = reason


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
    fixture: SyntheticFixture
    context: ProgramContext
    features: GraphFeatureInput
    limits: InterpreterLimits
    overlay: dict[str, OverlayValue]
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
            self.fixture.fixture_id,
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
    if isinstance(value, VertexRef):
        return ValueType.VERTEX
    if isinstance(value, EdgeRef):
        return ValueType.EDGE
    if isinstance(value, NonEdgeRef):
        return ValueType.NON_EDGE
    if isinstance(value, Path2Ref):
        return ValueType.PATH2
    if isinstance(value, MatchingRef):
        return ValueType.MATCHING
    if isinstance(value, Selection):
        return value.value_type
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
        return _expect(runtime.features.values()[field_name], FEATURE_TYPES[field_name], path)
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
            selection = runtime.fixture.select(selector_id, arguments, runtime.overlay)
        except CatchableBranchFailure:
            raise
        except Exception as exc:
            raise _ProgramFault("INTERPRETER_FAULT", path, str(exc)) from exc
        if not isinstance(selection, Selection):
            _fault("TYPE_ERROR", path, "selector did not return a Selection")
        _expect(selection, SELECTOR_TYPES[selector_id], path)
        return selection
    if operation == "pick":
        source = _expression(runtime, expression.get("source"), environment, f"{path}/source")
        if not isinstance(source, Selection):
            _fault("TYPE_ERROR", path, "pick source is not a selection")
        if not source.items:
            raise CatchableBranchFailure(BranchFailureCode.NO_MATCH)
        mode = expression.get("mode")
        if mode == "require_singleton":
            if len(source.items) != 1:
                raise CatchableBranchFailure(
                    BranchFailureCode.LOCAL_PRECONDITION_FAILED,
                    "selection is not a singleton",
                )
            return source.items[0]
        runtime.budget("choices", 1, runtime.limits.maximum_choices, path)
        if mode == "seeded_uniform":
            index = runtime.random_index(path, [1] * len(source.items))
            return source.items[index]
        if mode == "seeded_weighted":
            feature = expression.get("weight_feature")
            if not isinstance(feature, str):
                _fault("INVALID_AST", path, "weighted pick lacks weight_feature")
            weights: list[int] = []
            for item in source.items:
                try:
                    weight = runtime.fixture.weight(item, feature, runtime.overlay)
                except CatchableBranchFailure:
                    raise
                except Exception as exc:
                    raise _ProgramFault("INTERPRETER_FAULT", path, str(exc)) from exc
                if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
                    _fault("TYPE_ERROR", path, "fixture weight must be a positive integer")
                weights.append(weight)
            return source.items[runtime.random_index(path, weights)]
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
        value = _expression(runtime, expression.get("value"), environment, f"{path}/value")
        return bool(value.items) if isinstance(value, Selection) else value is not None
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
            overlay_snapshot = runtime.overlay.copy()
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
                runtime.overlay.clear()
                runtime.overlay.update(overlay_snapshot)
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
            runtime.fixture.apply(action_id, arguments, runtime.overlay)
        except CatchableBranchFailure:
            raise
        except Exception as exc:
            raise _ProgramFault("INTERPRETER_FAULT", path, str(exc)) from exc
        return
    if operation == "emit":
        try:
            runtime.fixture.validate_emit(runtime.overlay)
        except CatchableBranchFailure:
            raise
        except Exception as exc:
            raise _ProgramFault("INTERPRETER_FAULT", path, str(exc)) from exc
        raise _Terminal(OutcomeKind.EMIT)
    if operation == "no_plan":
        reason = node.get("reason")
        if not isinstance(reason, str):
            _fault("INVALID_AST", path, "NoPlan reason must be a string")
        raise _Terminal(OutcomeKind.NO_PLAN, reason)
    _fault("INVALID_AST", path, f"unknown node: {operation!r}")


def _no_plan_reason(code: BranchFailureCode) -> str:
    if code is BranchFailureCode.LOCAL_PRECONDITION_FAILED:
        return BranchFailureCode.NO_MATCH.value
    return code.value


def invoke_program(
    program: ValidatedProgram,
    *,
    fixture: SyntheticFixture,
    context: ProgramContext,
    features: GraphFeatureInput,
    initial_overlay: Mapping[str, OverlayValue] | None = None,
    limits: InterpreterLimits | None = None,
) -> InvocationResult:
    """Execute one validated program without mutating fixture or caller state."""

    invocation_snapshot = dict(initial_overlay or {})
    runtime = _Runtime(
        program=program,
        fixture=fixture,
        context=context,
        features=features,
        limits=limits or InterpreterLimits(),
        overlay=invocation_snapshot.copy(),
    )
    try:
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
        return InvocationResult(
            InvocationOutcome(
                terminal.kind,
                terminal.reason,
                tuple(sorted(runtime.overlay.items())),
            ),
            None,
            runtime.counters.frozen(),
        )
    except CatchableBranchFailure as exc:
        runtime.overlay.clear()
        runtime.overlay.update(invocation_snapshot)
        return InvocationResult(
            InvocationOutcome(
                OutcomeKind.NO_PLAN,
                _no_plan_reason(exc.code),
                tuple(sorted(runtime.overlay.items())),
            ),
            None,
            runtime.counters.frozen(),
        )
    except _ProgramFault as exc:
        return InvocationResult(None, exc.failure, runtime.counters.frozen())
    except Exception as exc:
        return InvocationResult(
            None,
            ProgramFailure("INTERPRETER_FAULT", "/entry", str(exc)),
            runtime.counters.frozen(),
        )
