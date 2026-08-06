from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mutation_forge.native_v3.cohort import PROVIDER_PARTITION, render_batch_prompt
from mutation_forge.native_v3.single_program_contract import (
    SINGLE_PROGRAM_BRIEFS,
    SingleProgramContractError,
    build_single_program_contract,
    build_single_program_output_schema,
    build_single_program_request,
    model_facing_contract,
    single_program_request_size_bytes,
    validate_single_program_response,
)

FORBIDDEN_LENGTHS = (4, 8, 16)


def _fixtures() -> dict[str, dict[str, object]]:
    value = json.loads(
        Path("tests/fixtures/native_v3_single_program_responses.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def _response(value: dict[str, object]) -> str:
    return json.dumps(value, separators=(",", ":"))


def test_system_role_is_mathematical_and_separate_from_dynamic_slot_request() -> None:
    request = build_single_program_request(
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert "Erdős–Gyárfás conjecture" in request.system_prompt
    assert "counterexample" in request.system_prompt
    assert "The host owns" in request.system_prompt
    assert "Native v3" not in request.system_prompt
    assert "slot-00" not in request.system_prompt
    assert "slot-00" in request.prompt
    assert SINGLE_PROGRAM_BRIEFS["add-edge"] in request.prompt


def test_dynamic_request_contains_contract_and_checklist_but_not_recursive_schema() -> None:
    request = build_single_program_request(
        slot_id="slot-02",
        brief_id="relocation",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert "Every selector and action identifier is exact" in request.prompt
    assert "Prefer a valid `no_plan`" in request.prompt
    assert '"literals":["min","max"]' in request.prompt
    assert '"literals":[4,8,16]' in request.prompt
    assert '"relocations_legal"' in request.prompt
    assert '"relation"' in request.prompt
    assert '"$defs"' not in request.prompt
    assert '"oneOf"' not in request.prompt
    assert "program_json_raw" not in request.prompt


def test_model_contract_is_projected_from_the_validator_registry() -> None:
    contract = build_single_program_contract(FORBIDDEN_LENGTHS)
    projected = model_facing_contract(contract)

    assert set(projected["selectors"]) == set(contract.selectors)
    assert set(projected["actions"]) == set(contract.actions)
    for selector_id, definition in contract.selectors.items():
        selector = projected["selectors"][selector_id]
        assert selector["result_type"] == str(definition.result_type)
        assert set(selector["arguments"]) == set(definition.arguments)
        for name, literals in definition.literal_domains.items():
            assert selector["arguments"][name]["literals"] == list(literals)
    for action_id, definition in contract.actions.items():
        assert projected["actions"][action_id]["arguments"] == {
            name: str(value_type)
            for name, value_type in definition.arguments.items()
        }


def test_direct_output_schema_has_one_program_and_no_string_encoded_program() -> None:
    schema = build_single_program_output_schema(FORBIDDEN_LENGTHS)

    assert schema["required"] == ["program", "design_summary", "hypothesis"]
    assert schema["properties"]["program"] == {"$ref": "#/$defs/program"}
    assert "source" not in schema["properties"]
    serialized = json.dumps(schema, sort_keys=True)
    assert "program_json_raw" not in serialized
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("brief_id", ("add-edge", "remove-edge", "relocation", "fanout"))
def test_golden_single_program_responses_match_schema_and_validator(
    brief_id: str,
) -> None:
    fixture = _fixtures()[brief_id]
    schema = build_single_program_output_schema(FORBIDDEN_LENGTHS)

    Draft202012Validator(schema).validate(fixture)
    validated = validate_single_program_response(
        _response(fixture),
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert validated.program.ast == fixture["program"]
    assert validated.design_summary == fixture["design_summary"]
    assert validated.hypothesis == fixture["hypothesis"]


def test_schema_rejects_enum_alias_witness_length_unknown_node_and_field() -> None:
    validator = Draft202012Validator(
        build_single_program_output_schema(FORBIDDEN_LENGTHS)
    )
    base = _fixtures()["add-edge"]

    alias = copy.deepcopy(base)
    alias["program"]["entry"]["branches"][0]["value"]["arguments"]["mode"] = "low"
    assert not validator.is_valid(alias)

    witness = copy.deepcopy(base)
    witness["program"]["entry"]["branches"][0]["value"] = {
        "op": "selector",
        "selector_id": "vertices_witness_load_extreme",
        "arguments": {"length": 6, "mode": "min"},
    }
    assert not validator.is_valid(witness)

    unknown_node = copy.deepcopy(base)
    unknown_node["program"]["entry"] = {"op": "finish"}
    assert not validator.is_valid(unknown_node)

    invalid_field = copy.deepcopy(base)
    invalid_field["program"]["unexpected"] = True
    assert not validator.is_valid(invalid_field)


def test_validator_rejects_unrelated_relocation_binding() -> None:
    fixture = copy.deepcopy(_fixtures()["relocation"])
    first_branch = fixture["program"]["entry"]["branches"][0]
    first_branch["value"] = {
        "op": "selector",
        "selector_id": "vertices_degree_extreme",
        "arguments": {"mode": "min"},
    }
    first_branch["body"]["value"]["source"]["name"] = "candidate_relocations"

    with pytest.raises(SingleProgramContractError, match="action_argument_type"):
        validate_single_program_response(
            _response(fixture),
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_validator_rejects_missing_terminal_path() -> None:
    fixture = copy.deepcopy(_fixtures()["add-edge"])
    first_branch = fixture["program"]["entry"]["branches"][0]
    first_branch["body"]["body"] = {
        "op": "apply",
        "action_id": "add_edge",
        "arguments": {"edge": {"op": "ref", "name": "edge"}},
    }
    fixture["program"]["entry"] = first_branch

    with pytest.raises(SingleProgramContractError, match="unterminated_path"):
        validate_single_program_response(
            _response(fixture),
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_validator_rejects_invalid_active_witness_length() -> None:
    fixture = copy.deepcopy(_fixtures()["add-edge"])
    first_branch = fixture["program"]["entry"]["branches"][0]
    first_branch["value"] = {
        "op": "selector",
        "selector_id": "vertices_witness_load_extreme",
        "arguments": {"length": 6, "mode": "min"},
    }

    with pytest.raises(SingleProgramContractError, match="selector_argument_value"):
        validate_single_program_response(
            _response(fixture),
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_summary_and_hypothesis_are_bounded_to_three_sentences() -> None:
    fixture = copy.deepcopy(_fixtures()["add-edge"])
    fixture["design_summary"] = "One. Two. Three. Four."

    with pytest.raises(SingleProgramContractError, match="at most three sentences"):
        validate_single_program_response(
            _response(fixture),
            forbidden_lengths=FORBIDDEN_LENGTHS,
        )


def test_request_contract_size_is_deterministic_and_bounded() -> None:
    first = build_single_program_request(
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    second = build_single_program_request(
        slot_id="slot-00",
        brief_id="add-edge",
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )

    assert first == second
    assert single_program_request_size_bytes(first) == single_program_request_size_bytes(
        second
    )
    assert single_program_request_size_bytes(first) < 65_536


def test_existing_production_batch_contract_remains_byte_identical() -> None:
    assert [
        hashlib.sha256(render_batch_prompt(partition).encode("utf-8")).hexdigest()
        for partition in PROVIDER_PARTITION
    ] == [
        "6585f468ee304679f22af87639f70a3b2cd00834b36e6eb9195d9a108d4834d8",
        "b2d20776bbb739d524c1d78a092cde9a70a893bfe61908ccb430216a597b77ab",
    ]
    assert hashlib.sha256(
        Path("prompts/native-v3/cohort-system.md").read_bytes()
    ).hexdigest() == "86f6182b52c99596487f936bc59933ab6fa30eb45ce34f13ce1d3b485bb2f9ef"
    assert hashlib.sha256(
        Path("configs/native/native-v3-cohort-envelope.schema.json").read_bytes()
    ).hexdigest() == "a364205f589733e3974b184cb7d5c9b2ed20cb06d873787069b696566379a484"
