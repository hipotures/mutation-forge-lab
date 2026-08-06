"""Direct one-program prompt and structured-output contract for Native v3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import CanonicalJsonError, canonical_json_bytes, parse_strict_json
from .contracts import (
    BINARY_OPERATIONS,
    DEFAULT_PROGRAM_CONTRACT,
    NO_PLAN_REASONS,
    PICK_MODES,
    PROGRAM_SCHEMA_VERSION,
    UNARY_OPERATIONS,
    WEIGHT_FEATURES,
    ActionDefinition,
    ProgramContract,
    SelectorDefinition,
    ValidatedProgram,
    ValueType,
    validate_program,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SINGLE_PROGRAM_RESPONSE_SCHEMA_VERSION = "mforge.native.single_program_response.v1"
MAXIMUM_RESPONSE_BYTES = 65_536
MAXIMUM_SUMMARY_CHARACTERS = 600

SINGLE_PROGRAM_BRIEFS = {
    "add-edge": (
        "Add one legal non-edge selected by low local-cycle risk, then emit; "
        "return no_plan when no candidate exists."
    ),
    "remove-edge": (
        "Remove one present removable edge while avoiding high bridge risk, then emit; "
        "return no_plan when no candidate exists."
    ),
    "relocation": (
        "Choose one relation-safe relocation candidate, apply it, then emit; "
        "return no_plan when no candidate exists."
    ),
    "fanout": (
        "Choose one relation-safe edge-fanout candidate, apply it, then emit; "
        "return no_plan when no candidate exists."
    ),
}

_SELECTOR_RELATIONS = {
    "vertices_degree_extreme": "returns all vertices tied at the selected degree extreme",
    "vertices_degree_class": "returns all vertices whose degree equals degree",
    "vertices_witness_load_extreme": (
        "length is one active forbidden length; returns vertices tied at its load extreme"
    ),
    "edges_witness_load_extreme": (
        "length is one active forbidden length; returns edges tied at its load extreme"
    ),
    "vertices_articulation_risk": "returns vertices tied at the selected articulation risk",
    "edges_bridge_risk": "returns edges tied at the selected bridge risk",
    "vertices_distance_band": "source is bound; literals satisfy 0 <= minimum <= maximum",
    "edges_removable": "returns present edges; final host validation still applies",
    "non_edges_legal": "returns absent non-loop edges",
    "non_edges_from_vertex": "returns absent non-loop edges incident to vertex",
    "non_edges_local_cycle_risk": (
        "returns absent non-loop edges tied at the selected local-cycle-risk extreme"
    ),
    "paths_length_two": (
        "returns paths with two present edges and an absent endpoint edge"
    ),
    "matching_k_switch_reconnections": (
        "returns endpoint-preserving reconnections of k vertex-disjoint present edges"
    ),
    "relocations_legal": (
        "each result contains a present edge, one kept endpoint, and a distinct new "
        "vertex whose replacement edge is absent"
    ),
    "edge_fanouts_legal": (
        "each result contains a present edge and a distinct vertex for which both "
        "replacement edges are absent"
    ),
}


class SingleProgramContractError(ValueError):
    """A direct one-program response does not satisfy its contract."""


@dataclass(frozen=True, slots=True)
class SingleProgramRequest:
    system_prompt: str
    prompt: str
    output_schema: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "prompt": self.prompt,
            "output_schema": self.output_schema,
        }


@dataclass(frozen=True, slots=True)
class SingleProgramResponse:
    program: ValidatedProgram
    design_summary: str
    hypothesis: str


def _load_text(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _active_forbidden_lengths(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values or values != tuple(sorted(set(values))):
        raise ValueError("forbidden lengths must be a non-empty sorted unique tuple")
    if any(isinstance(value, bool) or value < 3 for value in values):
        raise ValueError("forbidden lengths must contain integers of at least three")
    return values


def build_single_program_contract(
    forbidden_lengths: tuple[int, ...],
) -> ProgramContract:
    """Build the validator registry for one active counterexample-search domain."""

    active_lengths = _active_forbidden_lengths(forbidden_lengths)
    selectors = {
        selector_id: SelectorDefinition(
            result_type=definition.result_type,
            cost=definition.cost,
            arguments=dict(definition.arguments),
            literal_domains={
                **definition.literal_domains,
                **(
                    {"length": active_lengths}
                    if selector_id
                    in {
                        "vertices_witness_load_extreme",
                        "edges_witness_load_extreme",
                    }
                    else {}
                ),
            },
            relation=_SELECTOR_RELATIONS[selector_id],
            ordered_nonnegative_bounds=(
                ("minimum", "maximum")
                if selector_id == "vertices_distance_band"
                else None
            ),
        )
        for selector_id, definition in DEFAULT_PROGRAM_CONTRACT.selectors.items()
    }
    selectors["relocations_legal"] = SelectorDefinition(
        result_type=ValueType.RELOCATION_SET,
        cost=2,
        arguments={},
        relation=_SELECTOR_RELATIONS["relocations_legal"],
    )
    selectors["edge_fanouts_legal"] = SelectorDefinition(
        result_type=ValueType.FANOUT_SET,
        cost=2,
        arguments={},
        relation=_SELECTOR_RELATIONS["edge_fanouts_legal"],
    )

    actions = dict(DEFAULT_PROGRAM_CONTRACT.actions)
    actions["relocate_endpoint"] = ActionDefinition(
        arguments={"relocation": ValueType.RELOCATION},
        relation=(
            "relocation must come from relocations_legal in the current private overlay"
        ),
    )
    actions["edge_fanout"] = ActionDefinition(
        arguments={"fanout": ValueType.FANOUT},
        relation="fanout must come from edge_fanouts_legal in the current private overlay",
    )

    return ProgramContract(
        selectors=selectors,
        actions=actions,
        context_fields=dict(DEFAULT_PROGRAM_CONTRACT.context_fields),
        graph_features=dict(DEFAULT_PROGRAM_CONTRACT.graph_features),
        pick_results={
            **DEFAULT_PROGRAM_CONTRACT.pick_results,
            ValueType.RELOCATION_SET: ValueType.RELOCATION,
            ValueType.FANOUT_SET: ValueType.FANOUT,
        },
    )


def model_facing_contract(contract: ProgramContract) -> dict[str, Any]:
    """Project the exact validator registry without repository-internal names."""

    return {
        "selectors": {
            selector_id: {
                "result_type": str(definition.result_type),
                "cost_units": definition.cost,
                "arguments": {
                    name: {
                        "type": str(value_type),
                        **(
                            {"literals": list(definition.literal_domains[name])}
                            if name in definition.literal_domains
                            else {}
                        ),
                    }
                    for name, value_type in definition.arguments.items()
                },
                "relation": definition.relation,
            }
            for selector_id, definition in sorted(contract.selectors.items())
        },
        "actions": {
            action_id: {
                "arguments": {
                    name: str(value_type)
                    for name, value_type in definition.arguments.items()
                },
                "relation": definition.relation,
            }
            for action_id, definition in sorted(contract.actions.items())
        },
        "context_fields": {
            name: str(value_type)
            for name, value_type in sorted(contract.context_fields.items())
        },
        "graph_features": {
            name: str(value_type)
            for name, value_type in sorted(contract.graph_features.items())
        },
        "expressions": {
            "reference": {"op": "ref", "fields": ["name"]},
            "context": {"op": "ctx", "field_literals": sorted(contract.context_fields)},
            "feature": {"op": "feature", "field_literals": sorted(contract.graph_features)},
            "rational": {
                "op": "rational",
                "fields": ["numerator", "denominator"],
                "relation": "denominator is positive and the fraction is normalized",
            },
            "selector": {
                "op": "selector",
                "fields": ["selector_id", "arguments"],
            },
            "pick": {
                "op": "pick",
                "mode_literals": list(PICK_MODES),
                "weight_feature_literals": list(WEIGHT_FEATURES),
            },
            "binary_op_literals": list(BINARY_OPERATIONS),
            "unary_op_literals": list(UNARY_OPERATIONS),
        },
        "nodes": {
            "op_literals": [
                "block",
                "let",
                "if",
                "try",
                "repeat",
                "choose",
                "apply",
                "emit",
                "no_plan",
            ],
            "no_plan_reason_literals": list(NO_PLAN_REASONS),
            "terminal_rule": "every reachable path ends exactly once in emit or no_plan",
        },
    }


def _identifier_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,63}$",
    }


def _expression_reference() -> dict[str, Any]:
    return {"$ref": "#/$defs/expression"}


def _exact_object(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }


def _selector_schema(
    selector_id: str,
    definition: SelectorDefinition,
) -> dict[str, Any]:
    argument_properties = {
        name: (
            {"enum": list(definition.literal_domains[name])}
            if name in definition.literal_domains
            else _expression_reference()
        )
        for name in definition.arguments
    }
    return _exact_object(
        {
            "op": {"const": "selector"},
            "selector_id": {"const": selector_id},
            "arguments": _exact_object(
                argument_properties,
                required=tuple(definition.arguments),
            ),
        },
        required=("op", "selector_id", "arguments"),
    )


def _action_schema(
    action_id: str,
    definition: ActionDefinition,
) -> dict[str, Any]:
    return _exact_object(
        {
            "op": {"const": "apply"},
            "action_id": {"const": action_id},
            "arguments": _exact_object(
                {
                    name: _expression_reference()
                    for name in definition.arguments
                },
                required=tuple(definition.arguments),
            ),
        },
        required=("op", "action_id", "arguments"),
    )


def _expression_schema(contract: ProgramContract) -> dict[str, Any]:
    return {
        "oneOf": [
            {
                "type": "integer",
                "minimum": -(1 << 63),
                "maximum": (1 << 63) - 1,
            },
            {"type": "boolean"},
            {"type": "string", "pattern": "^[ -~]*$"},
            _exact_object(
                {
                    "op": {"const": "ref"},
                    "name": {"$ref": "#/$defs/identifier"},
                },
                required=("op", "name"),
            ),
            _exact_object(
                {
                    "op": {"const": "ctx"},
                    "field": {"enum": sorted(contract.context_fields)},
                },
                required=("op", "field"),
            ),
            _exact_object(
                {
                    "op": {"const": "feature"},
                    "field": {"enum": sorted(contract.graph_features)},
                },
                required=("op", "field"),
            ),
            _exact_object(
                {
                    "op": {"const": "rational"},
                    "numerator": {
                        "type": "integer",
                        "minimum": -(1 << 63),
                        "maximum": (1 << 63) - 1,
                    },
                    "denominator": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": (1 << 32) - 1,
                    },
                },
                required=("op", "numerator", "denominator"),
            ),
            *[
                _selector_schema(selector_id, definition)
                for selector_id, definition in sorted(contract.selectors.items())
            ],
            _exact_object(
                {
                    "op": {"const": "pick"},
                    "source": _expression_reference(),
                    "mode": {"enum": ["seeded_uniform", "require_singleton"]},
                },
                required=("op", "source", "mode"),
            ),
            _exact_object(
                {
                    "op": {"const": "pick"},
                    "source": _expression_reference(),
                    "mode": {"const": "seeded_weighted"},
                    "weight_feature": {"enum": list(WEIGHT_FEATURES)},
                },
                required=("op", "source", "mode", "weight_feature"),
            ),
            _exact_object(
                {
                    "op": {"enum": list(BINARY_OPERATIONS)},
                    "left": _expression_reference(),
                    "right": _expression_reference(),
                },
                required=("op", "left", "right"),
            ),
            _exact_object(
                {
                    "op": {"enum": list(UNARY_OPERATIONS)},
                    "value": _expression_reference(),
                },
                required=("op", "value"),
            ),
        ]
    }


def _node_schema(contract: ProgramContract) -> dict[str, Any]:
    node = {"$ref": "#/$defs/node"}
    return {
        "oneOf": [
            _exact_object(
                {
                    "op": {"const": "block"},
                    "children": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 256,
                        "items": node,
                    },
                },
                required=("op", "children"),
            ),
            _exact_object(
                {
                    "op": {"const": "let"},
                    "name": {"$ref": "#/$defs/identifier"},
                    "value": _expression_reference(),
                    "body": node,
                },
                required=("op", "name", "value", "body"),
            ),
            _exact_object(
                {
                    "op": {"const": "if"},
                    "condition": _expression_reference(),
                    "then": node,
                    "else": node,
                },
                required=("op", "condition", "then", "else"),
            ),
            _exact_object(
                {
                    "op": {"const": "try"},
                    "branches": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": node,
                    },
                },
                required=("op", "branches"),
            ),
            _exact_object(
                {
                    "op": {"const": "repeat"},
                    "count": {"type": "integer", "minimum": 1, "maximum": 8},
                    "body": node,
                },
                required=("op", "count", "body"),
            ),
            _exact_object(
                {
                    "op": {"const": "choose"},
                    "branches": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": _exact_object(
                            {
                                "weight": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": (1 << 32) - 1,
                                },
                                "body": node,
                            },
                            required=("weight", "body"),
                        ),
                    },
                },
                required=("op", "branches"),
            ),
            *[
                _action_schema(action_id, definition)
                for action_id, definition in sorted(contract.actions.items())
            ],
            _exact_object({"op": {"const": "emit"}}, required=("op",)),
            _exact_object(
                {
                    "op": {"const": "no_plan"},
                    "reason": {"enum": list(NO_PLAN_REASONS)},
                },
                required=("op", "reason"),
            ),
        ]
    }


def build_single_program_output_schema(
    forbidden_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Build the direct one-program outputSchema supplied to App Server."""

    contract = build_single_program_contract(forbidden_lengths)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://mutation-forge.invalid/schemas/"
            "native-v3-single-program-response.json"
        ),
        "title": "One typed graph-rewrite program",
        "type": "object",
        "additionalProperties": False,
        "required": ["program", "design_summary", "hypothesis"],
        "properties": {
            "program": {"$ref": "#/$defs/program"},
            "design_summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAXIMUM_SUMMARY_CHARACTERS,
                "pattern": "^[ -~]+$",
            },
            "hypothesis": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAXIMUM_SUMMARY_CHARACTERS,
                "pattern": "^[ -~]+$",
            },
        },
        "$defs": {
            "identifier": _identifier_schema(),
            "expression": _expression_schema(contract),
            "node": _node_schema(contract),
            "program": _exact_object(
                {
                    "schema_version": {"const": PROGRAM_SCHEMA_VERSION},
                    "entry": {"$ref": "#/$defs/node"},
                },
                required=("schema_version", "entry"),
            ),
        },
    }


def build_single_program_request(
    *,
    slot_id: str,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
) -> SingleProgramRequest:
    """Build prompts and direct outputSchema without contacting a provider."""

    if brief_id not in SINGLE_PROGRAM_BRIEFS:
        raise ValueError(f"unknown single-program brief: {brief_id}")
    if not re.fullmatch(r"slot-[0-9]{2}", slot_id):
        raise ValueError("slot_id must match slot-NN")
    contract = build_single_program_contract(forbidden_lengths)
    dynamic = {
        "slot_id": slot_id,
        "brief_id": brief_id,
        "brief": SINGLE_PROGRAM_BRIEFS[brief_id],
        "active_forbidden_lengths": list(forbidden_lengths),
        "executable_contract": model_facing_contract(contract),
    }
    request = _load_text("prompts/native-v3/single-program-request.md").rstrip()
    prompt = (
        request
        + "\n\nRequest:\n"
        + json.dumps(dynamic, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )
    return SingleProgramRequest(
        system_prompt=_load_text("prompts/native-v3/single-program-system.md").rstrip(),
        prompt=prompt,
        output_schema=build_single_program_output_schema(forbidden_lengths),
    )


def single_program_request_size_bytes(request: SingleProgramRequest) -> int:
    """Return compact UTF-8 request-contract bytes, including RPC framing newline."""

    payload = json.dumps(
        request.as_dict(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return len((payload + "\n").encode("utf-8"))


def _bounded_summary(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAXIMUM_SUMMARY_CHARACTERS
        or not value.isascii()
        or not value.isprintable()
    ):
        raise SingleProgramContractError(
            f"{field_name} must be 1..{MAXIMUM_SUMMARY_CHARACTERS} printable ASCII characters"
        )
    sentence_marks = len(re.findall(r"[.!?](?:\s|$)", value))
    sentence_count = max(1, sentence_marks)
    if sentence_count > 3:
        raise SingleProgramContractError(f"{field_name} must contain at most three sentences")
    return value


def validate_single_program_response(
    response_text: str,
    *,
    forbidden_lengths: tuple[int, ...],
) -> SingleProgramResponse:
    """Validate a direct response against the same registry shown to the model."""

    try:
        parsed = parse_strict_json(
            response_text,
            maximum_bytes=MAXIMUM_RESPONSE_BYTES,
        )
    except CanonicalJsonError as exc:
        raise SingleProgramContractError(f"invalid response JSON: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != {
        "program",
        "design_summary",
        "hypothesis",
    }:
        raise SingleProgramContractError(
            "response must contain exactly program, design_summary, and hypothesis"
        )
    program_value = parsed["program"]
    if not isinstance(program_value, dict):
        raise SingleProgramContractError("program must be an object")
    try:
        program_raw = canonical_json_bytes(program_value).decode("ascii")
    except CanonicalJsonError as exc:
        raise SingleProgramContractError(f"invalid program JSON: {exc}") from exc
    validation = validate_program(
        program_raw,
        contract=build_single_program_contract(forbidden_lengths),
    )
    if validation.program is None:
        summary = "; ".join(
            f"{item.code} at {item.path}: {item.message}"
            for item in validation.diagnostics
        )
        raise SingleProgramContractError(f"invalid program: {summary}")
    return SingleProgramResponse(
        program=validation.program,
        design_summary=_bounded_summary(parsed["design_summary"], "design_summary"),
        hypothesis=_bounded_summary(parsed["hypothesis"], "hypothesis"),
    )
