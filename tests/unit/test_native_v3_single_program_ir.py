from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from mutation_forge.native_v3.canonical import canonical_json_bytes
from mutation_forge.native_v3.single_program_ir import (
    BRIEF_OPERATORS,
    FLAT_IR_SCHEMA_VERSION,
    MAXIMUM_FLAT_LOGICAL_STEPS,
    SLOT_SPECIFIC_SCHEMA_VERSION,
    CandidateContractError,
    CandidateKind,
    build_candidate_request,
    build_flat_ir_output_schema,
    build_schema_complexity_inventory,
    build_slot_specific_output_schema,
    compile_candidate_response,
    compile_flat_ir_response,
    compile_slot_specific_response,
)

FORBIDDEN_LENGTHS = (4, 8, 16)
BRIEF_IDS = tuple(BRIEF_OPERATORS)
SELECTOR_ARGUMENTS = {
    "add-edge": {"mode": "min"},
    "remove-edge": {"mode": "min"},
    "relocation": {},
    "fanout": {},
}
ACTION_ARGUMENTS = {
    "add-edge": "edge",
    "remove-edge": "edge",
    "relocation": "relocation",
    "fanout": "fanout",
}
SLOT_SCHEMA_HASHES = {
    "add-edge": "ee716c45d85014445854432a3ac13dec7e790b93b3ce2c40124a46ff583ee3af",
    "remove-edge": "f5bbe1483f8a999a63c4b16e913d61eebfc99d81a5d5f2f1e195757bb44f1f94",
    "relocation": "4d77a3c84a23ff5a59cb44ceee90b069c88fef15f0543c1421ff8ef63454ff56",
    "fanout": "307e53ea8225897b57236a559f60b4a30e0d687748efdfc9d37fa38992636087",
}
FLAT_SCHEMA_HASH = "2906f517c522e5ee97596c5c90d104fa1a8bda2a1b443467cf37df45f105e5c8"


def _common(*, schema_version: str, slot_id: str, brief_id: str) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "slot_id": slot_id,
        "brief_id": brief_id,
        "active_forbidden_lengths": list(FORBIDDEN_LENGTHS),
        "design_summary": f"Use the bounded {brief_id} operator family.",
        "hypothesis": "The relation-certified rewrite may improve the candidate graph.",
    }


def _slot_response(brief_id: str, slot_id: str) -> dict[str, object]:
    operators = BRIEF_OPERATORS[brief_id]
    argument = ACTION_ARGUMENTS[brief_id]
    return {
        **_common(
            schema_version=SLOT_SPECIFIC_SCHEMA_VERSION,
            slot_id=slot_id,
            brief_id=brief_id,
        ),
        "plan": {
            "selector": {
                "selector_id": operators.selector_id,
                "arguments": dict(SELECTOR_ARGUMENTS[brief_id]),
            },
            "pick": {"mode": "seeded_uniform"},
            "action": {
                "action_id": operators.action_id,
                "arguments": {argument: "selected"},
            },
            "terminal": {"kind": "emit"},
        },
    }


def _flat_response(brief_id: str, slot_id: str) -> dict[str, object]:
    operators = BRIEF_OPERATORS[brief_id]
    argument = ACTION_ARGUMENTS[brief_id]
    return {
        **_common(
            schema_version=FLAT_IR_SCHEMA_VERSION,
            slot_id=slot_id,
            brief_id=brief_id,
        ),
        "bindings": [
            {
                "kind": "selector",
                "id": "candidates",
                "selector_id": operators.selector_id,
                "arguments": dict(SELECTOR_ARGUMENTS[brief_id]),
            },
            {
                "kind": "pick",
                "id": "selected",
                "source": "candidates",
                "mode": "seeded_uniform",
            },
        ],
        "steps": [
            {
                "op": "apply",
                "action_id": operators.action_id,
                "arguments": {argument: "selected"},
            }
        ],
        "terminal": {"kind": "emit"},
    }


def _text(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("brief_index,brief_id", enumerate(BRIEF_IDS))
def test_slot_specific_schema_golden_and_compiler(
    brief_index: int,
    brief_id: str,
) -> None:
    slot_id = f"slot-{brief_index:02d}"
    response = _slot_response(brief_id, slot_id)
    schema = build_slot_specific_output_schema(
        brief_id=brief_id,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(response)
    assert hashlib.sha256(canonical_json_bytes(schema)).hexdigest() == SLOT_SCHEMA_HASHES[
        brief_id
    ]
    compiled = compile_slot_specific_response(
        _text(response),
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert "$ref" not in _text(schema)
    assert compiled.program.ast["entry"]["op"] == "try"
    assert compiled.program.gross_actions == 1
    assert compiled.representation == response
    assert (
        compile_slot_specific_response(
            canonical_json_bytes(response).decode("ascii"),
            slot_id=slot_id,
            brief_id=brief_id,
            forbidden_lengths=FORBIDDEN_LENGTHS,
        ).program.program_hash
        == compiled.program.program_hash
    )


@pytest.mark.parametrize("brief_index,brief_id", enumerate(BRIEF_IDS))
def test_flat_ir_schema_golden_and_compiler(
    brief_index: int,
    brief_id: str,
) -> None:
    slot_id = f"slot-{brief_index:02d}"
    response = _flat_response(brief_id, slot_id)
    schema = build_flat_ir_output_schema(FORBIDDEN_LENGTHS)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(response)
    assert hashlib.sha256(canonical_json_bytes(schema)).hexdigest() == FLAT_SCHEMA_HASH
    compiled = compile_flat_ir_response(
        _text(response),
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert compiled.program.ast["entry"]["op"] == "try"
    assert compiled.program.gross_actions == 1
    assert compiled.representation == response
    assert (
        compile_candidate_response(
            "flat_ir",
            canonical_json_bytes(response).decode("ascii"),
            slot_id=slot_id,
            brief_id=brief_id,
            forbidden_lengths=FORBIDDEN_LENGTHS,
        ).program.program_hash
        == compiled.program.program_hash
    )


def test_candidate_request_is_non_recursive_and_does_not_change_rich_builder() -> None:
    slot_specific = build_candidate_request(
        candidate="slot_specific",
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    flat = build_candidate_request(
        candidate="flat_ir",
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert "counterexample" in slot_specific.system_prompt
    assert slot_specific.system_prompt == flat.system_prompt
    assert '"candidate":"slot_specific"' in slot_specific.prompt
    assert '"candidate":"flat_ir"' in flat.prompt
    assert "recursive" not in _text(slot_specific.output_schema)
    assert "#/$defs/" not in _text(flat.output_schema)
    assert '"prefixItems"' not in _text(slot_specific.output_schema)
    assert '"items":false' not in _text(slot_specific.output_schema)
    assert '"oneOf"' not in _text(slot_specific.output_schema)
    assert '"prefixItems"' not in _text(flat.output_schema)
    assert '"items":false' not in _text(flat.output_schema)
    assert '"oneOf"' not in _text(flat.output_schema)


def test_schema_complexity_inventory_is_deterministic_and_has_all_candidates() -> None:
    first = build_schema_complexity_inventory(FORBIDDEN_LENGTHS)
    second = build_schema_complexity_inventory(FORBIDDEN_LENGTHS)

    assert first == second
    assert len(first) == 12
    assert {
        (item["candidate"], item["brief_id"]) for item in first
    } == {
        (candidate, brief_id)
        for candidate in ("rich_recursive_control", "slot_specific", "flat_ir")
        for brief_id in BRIEF_IDS
    }
    rich = [item for item in first if item["candidate"] == "rich_recursive_control"]
    slots = [item for item in first if item["candidate"] == "slot_specific"]
    flat = [item for item in first if item["candidate"] == "flat_ir"]
    assert {item["schema_bytes"] for item in rich} == {12_125}
    assert all(item["recursive_reference_count"] > 0 for item in rich)
    assert all(item["recursive_reference_count"] == 0 for item in slots + flat)
    assert all(item["schema_bytes"] < 12_125 for item in slots + flat)
    assert all(item["maximum_generated_program_step_count"] == 1 for item in slots)
    assert all(
        item["maximum_generated_program_step_count"] == MAXIMUM_FLAT_LOGICAL_STEPS
        for item in flat
    )


@pytest.mark.parametrize("candidate", ("slot_specific", "flat_ir"))
def test_schema_and_compiler_reject_missing_terminal(candidate: CandidateKind) -> None:
    response = (
        _slot_response("add-edge", "slot-00")
        if candidate == "slot_specific"
        else _flat_response("add-edge", "slot-00")
    )
    if candidate == "slot_specific":
        plan = cast(dict[str, Any], response["plan"])
        del plan["terminal"]
        schema = build_slot_specific_output_schema(
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )
    else:
        del response["terminal"]
        schema = build_flat_ir_output_schema(FORBIDDEN_LENGTHS)
    assert not Draft202012Validator(schema).is_valid(response)
    with pytest.raises(CandidateContractError, match="terminal"):
        compile_candidate_response(
            candidate,
            _text(response),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_slot_schema_cannot_accept_a_missing_rewrite_field_as_another_variant() -> None:
    response = _slot_response("add-edge", "slot-00")
    plan = cast(dict[str, Any], response["plan"])
    del plan["action"]
    schema = build_slot_specific_output_schema(
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert not Draft202012Validator(schema).is_valid(response)
    with pytest.raises(CandidateContractError, match="fields"):
        compile_slot_specific_response(
            _text(response),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_schema_and_compiler_reject_enum_alias_and_invalid_forbidden_length() -> None:
    response = _slot_response("add-edge", "slot-00")
    response["plan"]["selector"]["arguments"]["mode"] = "low"  # type: ignore[index]
    schema = build_slot_specific_output_schema(
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    assert not Draft202012Validator(schema).is_valid(response)
    with pytest.raises(CandidateContractError, match="validator domain"):
        compile_slot_specific_response(
            _text(response),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )

    response = _slot_response("add-edge", "slot-00")
    response["active_forbidden_lengths"] = [4, 6, 16]
    assert not Draft202012Validator(schema).is_valid(response)
    with pytest.raises(CandidateContractError, match="active set"):
        compile_slot_specific_response(
            _text(response),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_flat_compiler_rejects_unresolved_and_incompatible_action_bindings() -> None:
    unresolved = _flat_response("add-edge", "slot-00")
    unresolved["steps"][0]["arguments"]["edge"] = "missing"  # type: ignore[index]
    with pytest.raises(CandidateContractError, match="unresolved"):
        compile_flat_ir_response(
            _text(unresolved),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )

    incompatible = _flat_response("remove-edge", "slot-01")
    incompatible["bindings"][0] = {  # type: ignore[index]
        "kind": "selector",
        "id": "candidates",
        "selector_id": "non_edges_local_cycle_risk",
        "arguments": {"mode": "min"},
    }
    with pytest.raises(CandidateContractError, match="expects EdgeRef, got NonEdgeRef"):
        compile_flat_ir_response(
            _text(incompatible),
            slot_id="slot-01",
            brief_id="remove-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_flat_compiler_rejects_alias_excessive_work_and_unterminated_noop() -> None:
    alias = _flat_response("add-edge", "slot-00")
    alias["bindings"][0]["selector_id"] = "low_risk_non_edges"  # type: ignore[index]
    with pytest.raises(CandidateContractError, match="unsupported"):
        compile_flat_ir_response(
            _text(alias),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )

    excessive = _flat_response("add-edge", "slot-00")
    excessive["bindings"] = [
        *cast(list[object], excessive["bindings"]),
        *[
            {
                "kind": "selector",
                "id": f"extra_{index}",
                "selector_id": "non_edges_local_cycle_risk",
                "arguments": {"mode": "min"},
            }
            for index in range(MAXIMUM_FLAT_LOGICAL_STEPS)
        ],
    ]
    with pytest.raises(CandidateContractError, match="exceeds eight"):
        compile_flat_ir_response(
            _text(excessive),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )

    noop = _flat_response("add-edge", "slot-00")
    noop["bindings"] = []
    noop["steps"] = []
    with pytest.raises(CandidateContractError, match="requires at least one"):
        compile_flat_ir_response(
            _text(noop),
            slot_id="slot-00",
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


@pytest.mark.parametrize("candidate", ("slot_specific", "flat_ir"))
def test_no_plan_is_explicit_and_cannot_carry_hidden_work(
    candidate: CandidateKind,
) -> None:
    if candidate == "slot_specific":
        response = _slot_response("add-edge", "slot-00")
        response["plan"] = {
            "terminal": {"kind": "no_plan", "reason": "NO_MATCH"}
        }
        schema = build_slot_specific_output_schema(
            brief_id="add-edge",
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )
    else:
        response = _flat_response("add-edge", "slot-00")
        response["bindings"] = []
        response["steps"] = []
        response["terminal"] = {"kind": "no_plan", "reason": "NO_MATCH"}
        schema = build_flat_ir_output_schema(FORBIDDEN_LENGTHS)
    Draft202012Validator(schema).validate(response)
    compiled = compile_candidate_response(
        candidate,
        _text(response),
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    assert compiled.program.ast["entry"] == {"op": "no_plan", "reason": "NO_MATCH"}

    hidden = copy.deepcopy(response)
    if candidate == "slot_specific":
        hidden["plan"]["selector"] = {}  # type: ignore[index]
        assert not Draft202012Validator(schema).is_valid(hidden)
    else:
        hidden["bindings"] = _flat_response("add-edge", "slot-00")["bindings"]
        with pytest.raises(CandidateContractError, match="cannot contain"):
            compile_flat_ir_response(
                _text(hidden),
                slot_id="slot-00",
                brief_id="add-edge",
                forbidden_lengths=FORBIDDEN_LENGTHS,
            )
