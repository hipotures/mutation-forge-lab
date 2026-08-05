from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from mutation_forge.native_v3.canonical import (
    PROGRAM_HASH_DOMAIN,
    CanonicalJsonError,
    canonical_json_bytes,
    domain_hash,
    parse_strict_json,
)
from mutation_forge.native_v3.contracts import (
    PROGRAM_SCHEMA_VERSION,
    ProgramLimits,
    validate_program,
    validated_program_artifact,
)


def _program(entry: object, *, schema_version: str = PROGRAM_SCHEMA_VERSION) -> str:
    return json.dumps(
        {"schema_version": schema_version, "entry": entry},
        separators=(",", ":"),
    )


def _terminal(reason: str = "EXPLICIT") -> dict[str, object]:
    return {"op": "no_plan", "reason": reason}


def _reverse_objects(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _reverse_objects(item)
            for key, item in reversed(tuple(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_objects(item) for item in value]
    return value


@pytest.mark.parametrize(
    "raw",
    (
        '{"a":1,"a":2}',
        '{"a":1.0}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":-Infinity}',
    ),
)
def test_strict_json_rejects_ambiguous_numeric_and_object_syntax(raw: str) -> None:
    with pytest.raises(CanonicalJsonError):
        parse_strict_json(raw, maximum_bytes=100)


def test_canonical_json_has_normative_order_escaping_and_domain_hashing() -> None:
    canonical = canonical_json_bytes({"z": 1, "a": 'x"y'})
    assert canonical == b'{"a":"x\\"y","z":1}'
    assert domain_hash(b"test-domain\0", canonical) == hashlib.sha256(
        b"test-domain\0" + canonical
    ).hexdigest()
    assert domain_hash(b"a\0", b"bc") != domain_hash(b"ab\0", b"c")
    with pytest.raises(CanonicalJsonError, match="NUL-terminated"):
        domain_hash(b"test-domain", canonical)
    with pytest.raises(CanonicalJsonError, match="keys"):
        canonical_json_bytes({"a": 1, 2: "b"})
    with pytest.raises(CanonicalJsonError, match="printable ASCII"):
        canonical_json_bytes({"text": "zażółć"})


@pytest.mark.parametrize(
    "entry",
    (
        {"op": "no_plan", "reason": "EXPLICIT"},
        {
            "op": "if",
            "condition": True,
            "then": {"op": "no_plan", "reason": "NO_MATCH"},
            "else": {"op": "emit"},
        },
        {
            "op": "let",
            "name": "ratio",
            "value": {"op": "rational", "numerator": -3, "denominator": 5},
            "body": {"op": "no_plan", "reason": "NO_EFFECT"},
        },
    ),
)
def test_equivalent_ast_objects_have_identical_canonical_bytes_and_hash(
    entry: dict[str, object],
) -> None:
    document = {"schema_version": PROGRAM_SCHEMA_VERSION, "entry": entry}
    reversed_document = _reverse_objects(document)
    raw_variants = (
        json.dumps(document),
        json.dumps(document, indent=2),
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        json.dumps(reversed_document, separators=(",", ":")),
    )

    validations = [validate_program(raw) for raw in raw_variants]
    assert all(item.valid for item in validations)
    programs = [item.program for item in validations]
    assert all(program is not None for program in programs)
    assert len({program.canonical_json for program in programs if program is not None}) == 1
    assert len({program.program_hash for program in programs if program is not None}) == 1


def test_non_equivalent_program_corpus_has_distinct_domain_separated_hashes() -> None:
    entries = (
        _terminal("EXPLICIT"),
        _terminal("NO_MATCH"),
        _terminal("ILLEGAL_FINAL_STATE"),
        {
            "op": "if",
            "condition": True,
            "then": _terminal("EXPLICIT"),
            "else": _terminal("NO_MATCH"),
        },
        {
            "op": "if",
            "condition": False,
            "then": _terminal("EXPLICIT"),
            "else": _terminal("NO_MATCH"),
        },
    )
    validations = [validate_program(_program(entry)) for entry in entries]
    assert all(item.valid and item.program is not None for item in validations)
    hashes = {
        item.program.program_hash
        for item in validations
        if item.program is not None
    }
    assert len(hashes) == len(entries)
    assert validations[0].program is not None
    assert validations[0].program.program_hash == (
        "f4d0a38a1f60860825aa672a4956e939d30f40553bf0c50091cede6c2adae81f"
    )
    for item in validations:
        assert item.program is not None
        raw_digest = hashlib.sha256(
            item.program.canonical_json.encode("ascii")
        ).hexdigest()
        assert item.program.program_hash != raw_digest
        assert len(item.program.program_hash) == 64
    assert PROGRAM_HASH_DOMAIN.endswith(b"\0")


@pytest.mark.parametrize(
    ("raw", "code"),
    (
        (
            '{"schema_version":"mforge.native.program.v3","entry":{"op":"emit"},'
            '"extra":true}',
            "unknown_field",
        ),
        (_program(_terminal(), schema_version="mforge.native.program.v2"), "schema_version"),
        (_program({"op": "unknown"}), "unknown_node"),
        (
            _program(
                {
                    "op": "let",
                    "name": "x",
                    "value": {"op": "ref", "name": "missing"},
                    "body": _terminal(),
                }
            ),
            "unbound_reference",
        ),
        (
            _program({"op": "emit", "unexpected": True}),
            "unknown_field",
        ),
    ),
)
def test_invalid_envelope_nodes_fields_and_references_fail_closed(
    raw: str,
    code: str,
) -> None:
    validation = validate_program(raw)
    assert not validation.valid
    assert validation.program is None
    assert validation.diagnostics[0].code == code


@pytest.mark.parametrize(
    ("numerator", "denominator", "code"),
    (
        (2, 4, "rational"),
        (0, 2, "rational"),
        (1, 0, "rational"),
        (1, -2, "rational"),
        (1 << 63, 1, "integer_bits"),
        (1, 1 << 32, "rational"),
    ),
)
def test_rationals_require_normalized_bounded_integer_components(
    numerator: int,
    denominator: int,
    code: str,
) -> None:
    validation = validate_program(
        _program(
            {
                "op": "let",
                "name": "ratio",
                "value": {
                    "op": "rational",
                    "numerator": numerator,
                    "denominator": denominator,
                },
                "body": _terminal(),
            }
        )
    )
    assert not validation.valid
    assert validation.diagnostics[0].code == code


def test_static_decoded_byte_node_depth_binding_repeat_and_integer_limits() -> None:
    minimal = _program(_terminal())
    over_bytes = validate_program(
        minimal,
        limits=ProgramLimits(maximum_decoded_bytes=len(minimal.encode("utf-8")) - 1),
    )
    assert over_bytes.diagnostics[0].code == "json"

    over_nodes = validate_program(minimal, limits=ProgramLimits(maximum_nodes=0))
    assert over_nodes.diagnostics[0].code == "node_limit"

    nested_expression = {
        "op": "if",
        "condition": {
            "op": "not",
            "value": {"op": "not", "value": True},
        },
        "then": _terminal(),
        "else": _terminal("NO_MATCH"),
    }
    over_depth = validate_program(
        _program(nested_expression),
        limits=ProgramLimits(maximum_depth=3),
    )
    assert over_depth.diagnostics[0].code == "depth_limit"

    bound = {
        "op": "let",
        "name": "value",
        "value": 1,
        "body": _terminal(),
    }
    over_bindings = validate_program(
        _program(bound),
        limits=ProgramLimits(maximum_bindings=0),
    )
    assert over_bindings.diagnostics[0].code == "binding_limit"

    over_repeat = validate_program(
        _program({"op": "repeat", "count": 2, "body": {"op": "apply"}}),
        limits=ProgramLimits(maximum_repeat=1),
    )
    assert over_repeat.diagnostics[0].code == "repeat_limit"

    over_integer = validate_program(
        _program({**bound, "value": 8}),
        limits=ProgramLimits(maximum_integer_bits=4),
    )
    assert over_integer.diagnostics[0].code == "integer_bits"

    over_denominator = validate_program(
        _program(
            {
                **bound,
                "value": {"op": "rational", "numerator": 1, "denominator": 9},
            }
        ),
        limits=ProgramLimits(maximum_denominator_bits=3),
    )
    assert over_denominator.diagnostics[0].code == "rational"


def test_typed_selector_binding_action_and_static_cost_contract() -> None:
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

    artifact = validated_program_artifact(validation.program)
    assert artifact["schema_version"] == PROGRAM_SCHEMA_VERSION
    assert artifact["program_hash"] == validation.program.program_hash

    invalid_mode = validate_program(
        _program(
            {
                "op": "let",
                "name": "vertices",
                "value": {
                    "op": "selector",
                    "selector_id": "vertices_degree_extreme",
                    "arguments": {"mode": "middle"},
                },
                "body": _terminal(),
            }
        )
    )
    assert invalid_mode.diagnostics[0].code == "selector_argument_value"


@pytest.mark.parametrize("k", (1, 5))
def test_k_switch_selector_rejects_non_2_3_4_literal(k: int) -> None:
    validation = validate_program(
        _program(
            {
                "op": "let",
                "name": "matching",
                "value": {
                    "op": "selector",
                    "selector_id": "matching_k_switch_reconnections",
                    "arguments": {"k": k},
                },
                "body": _terminal(),
            }
        )
    )
    assert not validation.valid
    assert validation.diagnostics[0].code == "selector_argument_value"


def test_program_schema_freezes_non_recursive_envelope_and_recursive_ast() -> None:
    schema = json.loads(
        Path("configs/native/native-v3-program.schema.json").read_text(encoding="utf-8")
    )
    assert schema["required"] == ["schema_version", "entry"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == PROGRAM_SCHEMA_VERSION
    assert schema["properties"]["entry"] == {"$ref": "#/$defs/node"}
    assert {"expression", "node"} <= set(schema["$defs"])


def test_native_v2_and_v3_imports_are_isolated() -> None:
    commands = (
        (
            "import sys; import mutation_forge.experiment; "
            "print(any(name.startswith('mutation_forge.native_v3') for name in sys.modules))"
        ),
        (
            "import sys; import mutation_forge.native_v3; "
            "print(any(name.startswith('mutation_forge.experiment') for name in sys.modules))"
        ),
    )
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-c", command],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == "False"
