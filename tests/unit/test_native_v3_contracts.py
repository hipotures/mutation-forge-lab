from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge.native_v3.canonical import (
    CanonicalJsonError,
    canonical_json_bytes,
    parse_strict_json,
)
from mutation_forge.native_v3.contracts import (
    ACTION_ARGUMENT_TYPES,
    PROGRAM_SCHEMA_VERSION,
    SELECTOR_COSTS,
    SELECTOR_TYPES,
    ProgramLimits,
    validate_program,
)
from mutation_forge.native_v3.randomness import (
    derive_seed64,
    splitmix64,
    uniform_below,
    weighted_index,
)


def _program(entry: object) -> str:
    return json.dumps(
        {"schema_version": PROGRAM_SCHEMA_VERSION, "entry": entry},
        separators=(",", ":"),
    )


def test_canonical_json_rejects_duplicate_keys_and_floats() -> None:
    with pytest.raises(CanonicalJsonError, match="duplicate"):
        parse_strict_json('{"a":1,"a":2}', maximum_bytes=100)
    with pytest.raises(CanonicalJsonError, match="floating-point"):
        parse_strict_json('{"a":1.25}', maximum_bytes=100)


def test_canonical_json_has_fixed_order_and_escaping() -> None:
    assert canonical_json_bytes({"z": 1, "a": 'x"y'}) == b'{"a":"x\\"y","z":1}'


def test_minimal_program_is_canonicalized_and_hashed() -> None:
    validation = validate_program(_program({"op": "no_plan", "reason": "EXPLICIT"}))
    assert validation.valid
    assert validation.program is not None
    assert validation.program.canonical_json == (
        '{"entry":{"op":"no_plan","reason":"EXPLICIT"},"schema_version":"mforge.native.program.v3"}'
    )
    assert len(validation.program.program_hash) == 64


def test_program_hash_ignores_input_object_key_order() -> None:
    first = validate_program(
        '{"schema_version":"mforge.native.program.v3","entry":{"op":"no_plan","reason":"EXPLICIT"}}'
    )
    second = validate_program(
        '{"entry":{"reason":"EXPLICIT","op":"no_plan"},"schema_version":"mforge.native.program.v3"}'
    )
    assert first.valid and second.valid
    assert first.program is not None and second.program is not None
    assert first.program.program_hash == second.program.program_hash


def test_validator_checks_typed_selector_bindings_and_action_arguments() -> None:
    entry = {
        "op": "let",
        "name": "vertices",
        "value": {
            "op": "selector",
            "selector_id": "vertices_degree_extreme",
            "arguments": {"mode": "min"},
        },
        "body": {
            "op": "let",
            "name": "u",
            "value": {
                "op": "pick",
                "source": {"op": "ref", "name": "vertices"},
                "mode": "seeded_uniform",
            },
            "body": {
                "op": "let",
                "name": "non_edges",
                "value": {
                    "op": "selector",
                    "selector_id": "non_edges_from_vertex",
                    "arguments": {"vertex": {"op": "ref", "name": "u"}},
                },
                "body": {
                    "op": "let",
                    "name": "edge",
                    "value": {
                        "op": "pick",
                        "source": {"op": "ref", "name": "non_edges"},
                        "mode": "seeded_uniform",
                    },
                    "body": {
                        "op": "block",
                        "children": [
                            {
                                "op": "apply",
                                "action_id": "add_edge",
                                "arguments": {
                                    "edge": {"op": "ref", "name": "edge"},
                                },
                            },
                            {"op": "emit"},
                        ],
                    },
                },
            },
        },
    }
    validation = validate_program(_program(entry))
    assert validation.valid, validation.diagnostics
    assert validation.program is not None
    assert validation.program.selector_calls == 2
    assert validation.program.selector_cost_units == 3
    assert validation.program.gross_actions == 1


def test_repeat_static_cost_is_multiplied_and_terminal_body_is_rejected() -> None:
    repeated = {
        "op": "block",
        "children": [
            {
                "op": "repeat",
                "count": 8,
                "body": {
                    "op": "let",
                    "name": "vertices",
                    "value": {
                        "op": "selector",
                        "selector_id": "vertices_articulation_risk",
                        "arguments": {"mode": "max"},
                    },
                    "body": {
                        "op": "apply",
                        "action_id": "remove_edge",
                        "arguments": {"edge": {"op": "ref", "name": "vertices"}},
                    },
                },
            },
            {"op": "no_plan", "reason": "EXPLICIT"},
        ],
    }
    validation = validate_program(_program(repeated))
    assert not validation.valid
    assert validation.diagnostics[0].code in {"action_argument_type", "selector_cost_limit"}

    terminal_repeat = validate_program(
        _program(
            {
                "op": "repeat",
                "count": 2,
                "body": {"op": "no_plan", "reason": "EXPLICIT"},
            }
        )
    )
    assert not terminal_repeat.valid
    assert terminal_repeat.diagnostics[0].code == "terminal"


def test_selector_cost_limit_is_static_and_versioned_registry_matches_code() -> None:
    limit = ProgramLimits(maximum_selector_cost_units=31)
    validation = validate_program(
        _program(
            {
                "op": "let",
                "name": "loaded",
                "value": {
                    "op": "selector",
                    "selector_id": "vertices_witness_load_extreme",
                    "arguments": {"length": 4, "mode": "max"},
                },
                "body": {"op": "no_plan", "reason": "EXPLICIT"},
            }
        ),
        limits=limit,
    )
    assert not validation.valid
    assert validation.diagnostics[0].code == "selector_cost_limit"

    registry = json.loads(
        Path("configs/native/native-v3-selector-registry.json").read_text(encoding="utf-8")
    )
    assert {item["selector_id"] for item in registry["selectors"]} == set(SELECTOR_TYPES)
    assert {
        item["selector_id"]: item["cost_units"] for item in registry["selectors"]
    } == SELECTOR_COSTS

    actions = json.loads(
        Path("configs/native/native-v3-action-registry.json").read_text(encoding="utf-8")
    )
    assert {item["operator_id"] for item in actions["actions"]} == set(ACTION_ARGUMENT_TYPES)


def test_weighted_pick_rejects_unbudgeted_witness_load_feature() -> None:
    validation = validate_program(
        _program(
            {
                "op": "let",
                "name": "vertices",
                "value": {
                    "op": "selector",
                    "selector_id": "vertices_degree_extreme",
                    "arguments": {"mode": "max"},
                },
                "body": {
                    "op": "let",
                    "name": "vertex",
                    "value": {
                        "op": "pick",
                        "source": {"op": "ref", "name": "vertices"},
                        "mode": "seeded_weighted",
                        "weight_feature": "witness_load",
                    },
                    "body": {"op": "no_plan", "reason": "EXPLICIT"},
                },
            }
        )
    )
    assert not validation.valid
    assert validation.diagnostics[0].code == "weight_feature"


def test_splitmix_and_unbiased_choice_are_replayable() -> None:
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    seed = derive_seed64("program", "episode", 7)
    assert uniform_below(seed, 0, 7) == uniform_below(seed, 0, 7)
    assert weighted_index(seed, 0, [1, 2, 3]) == weighted_index(seed, 0, [1, 2, 3])


def test_raw_vertex_labels_are_not_available_as_strategy_features() -> None:
    validation = validate_program(
        _program(
            {
                "op": "if",
                "condition": {
                    "op": "equal",
                    "left": {"op": "feature", "field": "vertex_id"},
                    "right": 0,
                },
                "then": {"op": "no_plan", "reason": "EXPLICIT"},
                "else": {"op": "no_plan", "reason": "EXPLICIT"},
            }
        )
    )
    assert not validation.valid
    assert validation.diagnostics[0].code == "unknown_field"
