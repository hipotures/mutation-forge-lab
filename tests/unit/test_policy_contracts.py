from __future__ import annotations

import math

import pytest

from mutation_forge.sandbox.contracts import (
    ContractError,
    canonical_json_bytes,
    canonical_plain_data,
    validate_priority,
    validate_probe_inputs,
)


def _ctx() -> dict[str, object]:
    return {
        "probe_id": "unit",
        "step": 0,
        "budget_remaining": 3,
        "features": {"values": (1, 2, 3)},
    }


def _proposal() -> dict[str, object]:
    return {
        "proposal_id": "p1",
        "kind": "probe",
        "features": {"weight": 1.0},
    }


def test_contract_canonicalizes_tuples_and_mapping_order() -> None:
    ctx, proposal = validate_probe_inputs(
        _ctx(),
        _proposal(),
        max_request_bytes=65536,
    )
    assert ctx["features"]["values"] == [1, 2, 3]
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert proposal["proposal_id"] == "p1"


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (math.nan, "invalid_float"),
        (math.inf, "invalid_float"),
        ({1: "bad"}, "invalid_key"),
        (object(), "invalid_type"),
        ("x" * 4097, "string_too_large"),
    ],
)
def test_plain_data_rejects_invalid_or_oversized_values(
    value: object,
    code: str,
) -> None:
    with pytest.raises(ContractError) as caught:
        canonical_plain_data(value)
    assert caught.value.code == code


def test_probe_schema_is_exact_and_request_is_bounded() -> None:
    invalid = _ctx()
    invalid["extra"] = 1
    with pytest.raises(ContractError, match="keys"):
        validate_probe_inputs(invalid, _proposal(), max_request_bytes=65536)
    with pytest.raises(ContractError) as caught:
        validate_probe_inputs(_ctx(), _proposal(), max_request_bytes=16)
    assert caught.value.code == "request_too_large"


@pytest.mark.parametrize("value", [True, None, complex(1), [], {}, math.nan, math.inf])
def test_priority_must_be_a_finite_non_bool_number(value: object) -> None:
    with pytest.raises(ContractError):
        validate_priority(value, max_response_bytes=16384)


def test_priority_rejects_oversized_integer() -> None:
    with pytest.raises(ContractError) as caught:
        validate_priority(1 << 4097, max_response_bytes=16384)
    assert caught.value.code == "output_integer_too_large"
