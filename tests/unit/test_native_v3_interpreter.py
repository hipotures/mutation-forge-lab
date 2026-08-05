from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from dataclasses import replace

import pytest

from mutation_forge.native_v3.contracts import (
    PROGRAM_SCHEMA_VERSION,
    ValidatedProgram,
    ValueType,
    validate_program,
)
from mutation_forge.native_v3.interpreter import (
    BranchFailureCode,
    CatchableBranchFailure,
    GraphFeatureInput,
    InterpreterLimits,
    NonEdgeRef,
    OutcomeKind,
    OverlayValue,
    ProgramContext,
    Scalar,
    Selectable,
    Selection,
    SyntheticFixture,
    invoke_program,
)
from mutation_forge.native_v3.randomness import (
    derive_seed64,
    splitmix64,
    uniform_below,
    weighted_index,
)


def _validated(entry: object) -> ValidatedProgram:
    validation = validate_program(
        json.dumps(
            {"schema_version": PROGRAM_SCHEMA_VERSION, "entry": entry},
            separators=(",", ":"),
        )
    )
    assert validation.valid
    assert validation.program is not None
    return validation.program


def _pick(mode: str) -> dict[str, object]:
    return {
        "op": "pick",
        "source": {
            "op": "selector",
            "selector_id": "non_edges_local_cycle_risk",
            "arguments": {"mode": mode},
        },
        "mode": "require_singleton",
    }


class _Fixture(SyntheticFixture):
    fixture_id = "synthetic-episode-17"

    def __init__(
        self,
        *,
        wrong_selection_type: bool = False,
        interpreter_fault: bool = False,
    ) -> None:
        self.wrong_selection_type = wrong_selection_type
        self.interpreter_fault = interpreter_fault

    def select(
        self,
        selector_id: str,
        arguments: Mapping[str, Scalar],
        overlay: Mapping[str, OverlayValue],
    ) -> Selection:
        del selector_id, overlay
        if self.wrong_selection_type:
            return Selection(ValueType.VERTEX_SET, ())
        edge = NonEdgeRef(1, 2) if arguments["mode"] == "min" else NonEdgeRef(9, 10)
        return Selection(ValueType.NON_EDGE, (edge,))

    def weight(
        self,
        item: Selectable,
        feature: str,
        overlay: Mapping[str, OverlayValue],
    ) -> int:
        del item, overlay
        return {"uniform": 1, "degree": 3, "inverse_degree": 2}[feature]

    def apply(
        self,
        action_id: str,
        arguments: Mapping[str, Scalar],
        overlay: MutableMapping[str, OverlayValue],
    ) -> None:
        del action_id
        edge = arguments["edge"]
        assert isinstance(edge, NonEdgeRef)
        overlay["applications"] = int(overlay.get("applications", 0)) + 1
        overlay["last_u"] = edge.u
        if self.interpreter_fault:
            raise RuntimeError("synthetic fixture bug")
        if edge.u == 9:
            overlay["failed_branch_leak"] = True
            raise CatchableBranchFailure(BranchFailureCode.NO_EFFECT)

    def validate_emit(self, overlay: Mapping[str, OverlayValue]) -> None:
        del overlay


CONTEXT = ProgramContext(
    step_index=7,
    horizon=100,
    acceptance_profile_id="synthetic",
    invocation_ordinal=3,
)
FEATURES = GraphFeatureInput(
    order=20,
    edge_count=31,
    minimum_degree=2,
    maximum_degree=6,
)


def test_random_protocol_has_frozen_vectors() -> None:
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert derive_seed64("program", "episode", 7) == 0x32083FE73F9CFA5B
    assert uniform_below(0x123456789ABCDEF0, 17) == (16, 1)
    assert weighted_index(0x123456789ABCDEF0, (1, 3, 2)) == (1, 1)


def test_same_input_replays_the_same_choice_and_output() -> None:
    program = _validated(
        {
            "op": "choose",
            "branches": [
                {"weight": 1, "body": {"op": "no_plan", "reason": "NO_MATCH"}},
                {"weight": 3, "body": {"op": "no_plan", "reason": "NO_EFFECT"}},
                {
                    "weight": 2,
                    "body": {"op": "no_plan", "reason": "ILLEGAL_FINAL_STATE"},
                },
            ],
        }
    )
    first = invoke_program(program, fixture=_Fixture(), context=CONTEXT, features=FEATURES)
    second = invoke_program(program, fixture=_Fixture(), context=CONTEXT, features=FEATURES)

    assert first == second
    assert first.successful
    assert first.outcome is not None
    assert first.outcome.kind is OutcomeKind.NO_PLAN
    assert first.outcome.no_plan_reason == "ILLEGAL_FINAL_STATE"
    assert first.counters.choices == 1
    assert first.counters.random_draws == 1


def test_seed_identity_includes_dynamic_path_and_invocation_ordinal() -> None:
    common = (
        "native_v3_splitmix64_v1",
        "native_v3_synthetic_interpreter_v1",
        "program-hash",
        "episode-id",
        5,
    )
    seeds = {
        derive_seed64(*common, 0, "/entry/body@0"),
        derive_seed64(*common, 0, "/entry/body@1"),
        derive_seed64(*common, 1, "/entry/body@0"),
    }
    assert seeds == {
        0x8C9DE88281E80C3B,
        0x9B87F0071E405640,
        0xE979DC0E98F72168,
    }


def test_try_restores_overlay_and_bindings_but_not_consumed_budgets() -> None:
    program = _validated(
        {
            "op": "let",
            "name": "outer_edge",
            "value": _pick("min"),
            "body": {
                "op": "try",
                "branches": [
                    {
                        "op": "let",
                        "name": "failed_edge",
                        "value": _pick("max"),
                        "body": {
                            "op": "block",
                            "children": [
                                {
                                    "op": "apply",
                                    "action_id": "add_edge",
                                    "arguments": {
                                        "edge": {"op": "ref", "name": "outer_edge"}
                                    },
                                },
                                {
                                    "op": "apply",
                                    "action_id": "add_edge",
                                    "arguments": {
                                        "edge": {"op": "ref", "name": "failed_edge"}
                                    },
                                },
                                {"op": "emit"},
                            ],
                        },
                    },
                    {
                        "op": "block",
                        "children": [
                            {
                                "op": "apply",
                                "action_id": "add_edge",
                                "arguments": {
                                    "edge": {"op": "ref", "name": "outer_edge"}
                                },
                            },
                            {"op": "emit"},
                        ],
                    },
                ],
            },
        }
    )
    result = invoke_program(
        program,
        fixture=_Fixture(),
        context=CONTEXT,
        features=FEATURES,
        initial_overlay={"original": True},
    )

    assert result.successful
    assert result.outcome is not None
    assert result.outcome.kind is OutcomeKind.EMIT
    assert result.outcome.overlay == (
        ("applications", 1),
        ("last_u", 1),
        ("original", True),
    )
    assert result.counters.bindings == 2
    assert result.counters.actions == 3
    assert result.counters.selector_calls == 2


def test_uncaught_catchable_failure_rolls_back_the_invocation_overlay() -> None:
    program = _validated(
        {
            "op": "block",
            "children": [
                {
                    "op": "apply",
                    "action_id": "add_edge",
                    "arguments": {"edge": _pick("max")},
                },
                {"op": "emit"},
            ],
        }
    )
    result = invoke_program(
        program,
        fixture=_Fixture(),
        context=CONTEXT,
        features=FEATURES,
        initial_overlay={"original": True},
    )
    assert result.failure is None
    assert result.outcome is not None
    assert result.outcome.kind is OutcomeKind.NO_PLAN
    assert result.outcome.no_plan_reason == "NO_EFFECT"
    assert result.outcome.overlay == (("original", True),)
    assert result.counters.actions == 1


def test_exact_rational_expression_and_if_semantics() -> None:
    program = _validated(
        {
            "op": "let",
            "name": "half",
            "value": {"op": "rational", "numerator": 1, "denominator": 2},
            "body": {
                "op": "if",
                "condition": {
                    "op": "equal",
                    "left": {
                        "op": "add",
                        "left": {"op": "ref", "name": "half"},
                        "right": {"op": "ref", "name": "half"},
                    },
                    "right": 1,
                },
                "then": {"op": "no_plan", "reason": "NO_EFFECT"},
                "else": {"op": "no_plan", "reason": "EXPLICIT"},
            },
        }
    )
    result = invoke_program(
        program,
        fixture=_Fixture(),
        context=CONTEXT,
        features=FEATURES,
    )
    assert result.outcome is not None
    assert result.outcome.no_plan_reason == "NO_EFFECT"


@pytest.mark.parametrize(
    ("program", "fixture", "limits", "expected_code"),
    [
        (
            _validated(
                {
                    "op": "try",
                    "branches": [
                        {
                            "op": "block",
                            "children": [
                                {
                                    "op": "repeat",
                                    "count": 1,
                                    "body": {
                                        "op": "apply",
                                        "action_id": "add_edge",
                                        "arguments": {"edge": _pick("min")},
                                    },
                                },
                                {"op": "emit"},
                            ],
                        },
                        {"op": "no_plan", "reason": "EXPLICIT"},
                    ],
                }
            ),
            _Fixture(),
            InterpreterLimits(maximum_repeat_iterations=0),
            "BUDGET_EXHAUSTED",
        ),
        (
            _validated(
                {
                    "op": "try",
                    "branches": [
                        {
                            "op": "let",
                            "name": "edge",
                            "value": _pick("min"),
                            "body": {"op": "emit"},
                        },
                        {"op": "no_plan", "reason": "EXPLICIT"},
                    ],
                }
            ),
            _Fixture(wrong_selection_type=True),
            InterpreterLimits(),
            "TYPE_ERROR",
        ),
        (
            _validated(
                {
                    "op": "try",
                    "branches": [
                        {
                            "op": "block",
                            "children": [
                                {
                                    "op": "apply",
                                    "action_id": "add_edge",
                                    "arguments": {"edge": _pick("min")},
                                },
                                {"op": "emit"},
                            ],
                        },
                        {"op": "no_plan", "reason": "EXPLICIT"},
                    ],
                }
            ),
            _Fixture(interpreter_fault=True),
            InterpreterLimits(),
            "INTERPRETER_FAULT",
        ),
    ],
)
def test_try_does_not_swallow_program_failures(
    program: ValidatedProgram,
    fixture: SyntheticFixture,
    limits: InterpreterLimits,
    expected_code: str,
) -> None:
    result = invoke_program(
        program,
        fixture=fixture,
        context=CONTEXT,
        features=FEATURES,
        limits=limits,
    )
    assert not result.successful
    assert result.outcome is None
    assert result.failure is not None
    assert result.failure.code == expected_code


def test_invalid_ast_is_a_program_failure_not_a_fallback_signal() -> None:
    valid = _validated({"op": "no_plan", "reason": "EXPLICIT"})
    invalid = replace(
        valid,
        ast={
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "entry": {
                "op": "try",
                "branches": [
                    {"op": "not_a_node"},
                    {"op": "no_plan", "reason": "EXPLICIT"},
                ],
            },
        },
    )
    result = invoke_program(invalid, fixture=_Fixture(), context=CONTEXT, features=FEATURES)
    assert result.outcome is None
    assert result.failure is not None
    assert result.failure.code == "INVALID_AST"


@pytest.mark.parametrize(
    ("entry", "kind", "reason"),
    [
        ({"op": "emit"}, OutcomeKind.EMIT, None),
        (
            {"op": "no_plan", "reason": "EXPLICIT"},
            OutcomeKind.NO_PLAN,
            "EXPLICIT",
        ),
    ],
)
def test_successful_invocation_has_exactly_one_terminal_outcome(
    entry: object,
    kind: OutcomeKind,
    reason: str | None,
) -> None:
    result = invoke_program(
        _validated(entry),
        fixture=_Fixture(),
        context=CONTEXT,
        features=FEATURES,
    )
    assert result.failure is None
    assert result.outcome is not None
    assert result.outcome.kind is kind
    assert result.outcome.no_plan_reason == reason
