"""Non-recursive model-facing contracts for one Native v3 program."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, cast

from .canonical import CanonicalJsonError, canonical_json_bytes, parse_strict_json
from .contracts import (
    NO_PLAN_REASONS,
    PROGRAM_SCHEMA_VERSION,
    ProgramContract,
    SelectorDefinition,
    ValidatedProgram,
    ValueType,
)
from .single_program_contract import (
    MAXIMUM_RESPONSE_BYTES,
    MAXIMUM_SUMMARY_CHARACTERS,
    SINGLE_PROGRAM_BRIEFS,
    SingleProgramRequest,
    build_single_program_contract,
    build_single_program_output_schema,
    build_single_program_request,
    validate_single_program_response,
)

CandidateKind = Literal["slot_specific", "flat_ir"]

SLOT_SPECIFIC_OUTPUT_CONTRACT: Literal["slot_specific"] = "slot_specific"
SLOT_SPECIFIC_SCHEMA_VERSION = "mforge.native.slot_program.v1"
FLAT_IR_SCHEMA_VERSION = "mforge.native.flat_ir.v1"
MAXIMUM_FLAT_LOGICAL_STEPS = 8

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_SLOT_ID = re.compile(r"^slot-[0-9]{2}$")
_PICK_MODES = ("seeded_uniform", "require_singleton")


@dataclass(frozen=True, slots=True)
class BriefOperators:
    selector_id: str
    action_id: str


BRIEF_OPERATORS = {
    "add-edge": BriefOperators("non_edges_local_cycle_risk", "add_edge"),
    "remove-edge": BriefOperators("edges_bridge_risk", "remove_edge"),
    "relocation": BriefOperators("relocations_legal", "relocate_endpoint"),
    "fanout": BriefOperators("edge_fanouts_legal", "edge_fanout"),
}


class CandidateContractError(ValueError):
    """An experimental response cannot compile to the existing Native v3 AST."""


@dataclass(frozen=True, slots=True)
class CompiledCandidateResponse:
    candidate: CandidateKind
    representation: dict[str, Any]
    representation_sha256: str
    program: ValidatedProgram
    design_summary: str
    hypothesis: str


def _exact_object(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties if required is None else required),
        "properties": properties,
    }


def _identifier_schema() -> dict[str, Any]:
    return {"type": "string", "pattern": _IDENTIFIER.pattern}


def _summary_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": MAXIMUM_SUMMARY_CHARACTERS,
        "pattern": "^[ -~]+$",
    }


def _active_lengths_schema(forbidden_lengths: tuple[int, ...]) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": len(forbidden_lengths),
        "maxItems": len(forbidden_lengths),
        "items": {"enum": list(forbidden_lengths)},
    }


def _literal_schema(definition: SelectorDefinition, name: str) -> dict[str, Any]:
    literals = definition.literal_domains.get(name)
    if literals is None:
        raise ValueError(f"experimental selector argument is not literal: {name}")
    return {"enum": list(literals)}


def _selector_arguments_schema(definition: SelectorDefinition) -> dict[str, Any]:
    return _exact_object(
        {
            name: _literal_schema(definition, name)
            for name in definition.arguments
        }
    )


def _strict_literal_types(value: object) -> None:
    if isinstance(value, dict):
        if "type" not in value and "const" in value:
            literal = value["const"]
            value["type"] = (
                "boolean"
                if isinstance(literal, bool)
                else "integer"
                if isinstance(literal, int)
                else "string"
            )
        if "type" not in value and isinstance(value.get("enum"), list):
            types = {
                "boolean"
                if isinstance(item, bool)
                else "integer"
                if isinstance(item, int)
                else "string"
                for item in value["enum"]
            }
            if len(types) != 1:
                raise ValueError("structured-output enums must use one JSON type")
            value["type"] = types.pop()
        for item in value.values():
            _strict_literal_types(item)
    elif isinstance(value, list):
        for item in value:
            _strict_literal_types(item)


def _base_properties(
    schema_version: str,
    forbidden_lengths: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": {"const": schema_version},
        "slot_id": {"type": "string", "pattern": _SLOT_ID.pattern},
        "brief_id": {"enum": list(BRIEF_OPERATORS)},
        "active_forbidden_lengths": _active_lengths_schema(forbidden_lengths),
        "design_summary": _summary_schema(),
        "hypothesis": _summary_schema(),
    }


def _terminal_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            _exact_object({"kind": {"const": "emit"}}),
            _exact_object(
                {
                    "kind": {"const": "no_plan"},
                    "reason": {"enum": list(NO_PLAN_REASONS)},
                }
            ),
        ]
    }


def build_slot_specific_output_schema(
    *,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Build one direct, non-recursive schema for the selected operator family."""

    contract = build_single_program_contract(forbidden_lengths)
    operators = _brief_operators(brief_id, contract)
    selector = contract.selectors[operators.selector_id]
    action = contract.actions[operators.action_id]
    if len(action.arguments) != 1:
        raise ValueError("slot-specific candidate requires one relation-preserving input")
    action_argument = next(iter(action.arguments))
    rewrite = _exact_object(
        {
            "selector": _exact_object(
                {
                    "selector_id": {"const": operators.selector_id},
                    "arguments": _selector_arguments_schema(selector),
                }
            ),
            "pick": _exact_object({"mode": {"enum": list(_PICK_MODES)}}),
            "action": _exact_object(
                {
                    "action_id": {"const": operators.action_id},
                    "arguments": _exact_object(
                        {action_argument: {"const": "selected"}}
                    ),
                }
            ),
            "terminal": _exact_object({"kind": {"const": "emit"}}),
        }
    )
    no_plan = _exact_object(
        {
            "terminal": _exact_object(
                {
                    "kind": {"const": "no_plan"},
                    "reason": {"enum": list(NO_PLAN_REASONS)},
                }
            )
        }
    )
    properties = _base_properties(SLOT_SPECIFIC_SCHEMA_VERSION, forbidden_lengths)
    properties["brief_id"] = {"const": brief_id}
    properties["plan"] = {"anyOf": [rewrite, no_plan]}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://mutation-forge.invalid/schemas/"
            f"native-v3-slot-program-{brief_id}.json"
        ),
        "title": f"One {brief_id} graph-rewrite program",
        **_exact_object(properties),
    }
    _strict_literal_types(schema)
    return schema


def slot_specific_schema_hashes(
    forbidden_lengths: tuple[int, ...],
) -> dict[str, str]:
    """Return the canonical schema identity for every supported brief."""

    return {
        brief_id: hashlib.sha256(
            canonical_json_bytes(
                build_slot_specific_output_schema(
                    brief_id=brief_id,
                    forbidden_lengths=forbidden_lengths,
                )
            )
        ).hexdigest()
        for brief_id in BRIEF_OPERATORS
    }


def slot_specific_contract_sha256(
    forbidden_lengths: tuple[int, ...],
) -> str:
    """Return one canonical identity for the complete slot-specific contract."""

    return hashlib.sha256(
        canonical_json_bytes(slot_specific_schema_hashes(forbidden_lengths))
    ).hexdigest()


def _flat_selector_variants(contract: ProgramContract) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for selector_id in sorted({item.selector_id for item in BRIEF_OPERATORS.values()}):
        definition = contract.selectors[selector_id]
        variants.append(
            _exact_object(
                {
                    "kind": {"const": "selector"},
                    "id": _identifier_schema(),
                    "selector_id": {"const": selector_id},
                    "arguments": _selector_arguments_schema(definition),
                }
            )
        )
    return variants


def _flat_action_variants(contract: ProgramContract) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for action_id in sorted({item.action_id for item in BRIEF_OPERATORS.values()}):
        definition = contract.actions[action_id]
        variants.append(
            _exact_object(
                {
                    "op": {"const": "apply"},
                    "action_id": {"const": action_id},
                    "arguments": _exact_object(
                        {name: _identifier_schema() for name in definition.arguments}
                    ),
                }
            )
        )
    return variants


def build_flat_ir_output_schema(
    forbidden_lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Build the generic flat bounded IR schema used by all four briefs."""

    contract = build_single_program_contract(forbidden_lengths)
    binding = {
        "anyOf": [
            *_flat_selector_variants(contract),
            _exact_object(
                {
                    "kind": {"const": "pick"},
                    "id": _identifier_schema(),
                    "source": _identifier_schema(),
                    "mode": {"enum": list(_PICK_MODES)},
                }
            ),
        ]
    }
    properties = _base_properties(FLAT_IR_SCHEMA_VERSION, forbidden_lengths)
    properties.update(
        {
            "bindings": {
                "type": "array",
                "maxItems": MAXIMUM_FLAT_LOGICAL_STEPS,
                "items": binding,
            },
            "steps": {
                "type": "array",
                "maxItems": MAXIMUM_FLAT_LOGICAL_STEPS,
                "items": {"anyOf": _flat_action_variants(contract)},
            },
            "terminal": _terminal_schema(),
        }
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mutation-forge.invalid/schemas/native-v3-flat-ir.json",
        "title": "One flat bounded graph-rewrite program",
        **_exact_object(properties),
    }
    _strict_literal_types(schema)
    return schema


def _brief_operators(brief_id: str, contract: ProgramContract) -> BriefOperators:
    operators = BRIEF_OPERATORS.get(brief_id)
    if operators is None:
        raise ValueError(f"unknown single-program brief: {brief_id}")
    if operators.selector_id not in contract.selectors:
        raise ValueError(f"brief selector is absent from validator contract: {brief_id}")
    if operators.action_id not in contract.actions:
        raise ValueError(f"brief action is absent from validator contract: {brief_id}")
    selector = contract.selectors[operators.selector_id]
    action = contract.actions[operators.action_id]
    if len(action.arguments) != 1:
        raise ValueError(f"brief action must have one typed relation input: {brief_id}")
    expected_type = next(iter(action.arguments.values()))
    selected_type = contract.pick_results.get(selector.result_type)
    if selected_type != expected_type:
        raise ValueError(f"brief selector/action types do not compose: {brief_id}")
    return operators


def _candidate_contract_projection(
    *,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
) -> dict[str, Any]:
    contract = build_single_program_contract(forbidden_lengths)
    operators = _brief_operators(brief_id, contract)
    selector = contract.selectors[operators.selector_id]
    action = contract.actions[operators.action_id]
    return {
        "brief_id": brief_id,
        "objective": SINGLE_PROGRAM_BRIEFS[brief_id],
        "selector": {
            "id": operators.selector_id,
            "result_type": str(selector.result_type),
            "arguments": {
                name: {
                    "type": str(value_type),
                    "literals": list(selector.literal_domains.get(name, ())),
                }
                for name, value_type in selector.arguments.items()
            },
            "relation": selector.relation,
        },
        "action": {
            "id": operators.action_id,
            "arguments": {
                name: str(value_type) for name, value_type in action.arguments.items()
            },
            "relation": action.relation,
        },
    }


def schema_experiment_anchor_prompt(forbidden_lengths: tuple[int, ...]) -> str:
    """Return the shared non-recursive specification anchor for both candidates."""

    return json.dumps(
        {
            "instruction": (
                "Retain this bounded graph-rewrite specification for later turns. "
                "Do not generate a program. Return only the required acknowledgement."
            ),
            "active_forbidden_lengths": list(forbidden_lengths),
            "briefs": [
                _candidate_contract_projection(
                    brief_id=brief_id,
                    forbidden_lengths=forbidden_lengths,
                )
                for brief_id in BRIEF_OPERATORS
            ],
            "compiler_boundary": (
                "The host compiles the non-recursive response into its existing typed AST "
                "and rejects unresolved or incompatible references before execution."
            ),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def candidate_followup_prompt(
    *,
    candidate: CandidateKind,
    slot_id: str,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
    accepted_behavior_signatures: tuple[str, ...] = (),
) -> str:
    if candidate not in {"slot_specific", "flat_ir"}:
        raise ValueError(f"unknown candidate: {candidate}")
    if not _SLOT_ID.fullmatch(slot_id):
        raise ValueError("slot_id must match slot-NN")
    projection = _candidate_contract_projection(
        brief_id=brief_id,
        forbidden_lengths=forbidden_lengths,
    )
    return (
        "Generate exactly one program using the retained specification.\n"
        "Return only the structured response. Prefer no_plan over an illegal rewrite.\n\n"
        + json.dumps(
            {
                "candidate": candidate,
                "slot_id": slot_id,
                "brief": projection,
                "active_forbidden_lengths": list(forbidden_lengths),
                "accepted_behavior_signatures": list(accepted_behavior_signatures),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def build_candidate_request(
    *,
    candidate: CandidateKind,
    slot_id: str,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
    accepted_behavior_signatures: tuple[str, ...] = (),
) -> SingleProgramRequest:
    """Build an experimental request without changing the production request builder."""

    system_prompt = build_single_program_request(
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=forbidden_lengths,
    ).system_prompt
    schema = (
        build_slot_specific_output_schema(
            brief_id=brief_id,
            forbidden_lengths=forbidden_lengths,
        )
        if candidate == "slot_specific"
        else build_flat_ir_output_schema(forbidden_lengths)
    )
    return SingleProgramRequest(
        system_prompt=system_prompt,
        prompt=candidate_followup_prompt(
            candidate=candidate,
            slot_id=slot_id,
            brief_id=brief_id,
            forbidden_lengths=forbidden_lengths,
            accepted_behavior_signatures=accepted_behavior_signatures,
        ),
        output_schema=schema,
    )


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateContractError(f"{path} must be an object")
    return cast(dict[str, Any], value)


def _exact(value: dict[str, Any], keys: set[str], path: str) -> None:
    if set(value) != keys:
        raise CandidateContractError(
            f"{path} fields must be {sorted(keys)}, got {sorted(value)}"
        )


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CandidateContractError(f"{path} must be an identifier")
    return value


def _parse_candidate(
    response_text: str,
    *,
    schema_version: str,
    slot_id: str,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
    payload_fields: set[str],
) -> dict[str, Any]:
    try:
        value = parse_strict_json(response_text, maximum_bytes=MAXIMUM_RESPONSE_BYTES)
    except CanonicalJsonError as exc:
        raise CandidateContractError(f"invalid response JSON: {exc}") from exc
    envelope = _object(value, "/")
    base = {
        "schema_version",
        "slot_id",
        "brief_id",
        "active_forbidden_lengths",
        "design_summary",
        "hypothesis",
    }
    _exact(envelope, base | payload_fields, "/")
    if envelope["schema_version"] != schema_version:
        raise CandidateContractError(f"/schema_version must be {schema_version}")
    if envelope["slot_id"] != slot_id:
        raise CandidateContractError("/slot_id does not match the requested slot")
    if envelope["brief_id"] != brief_id:
        raise CandidateContractError("/brief_id does not match the requested brief")
    lengths = envelope["active_forbidden_lengths"]
    if lengths != list(forbidden_lengths):
        raise CandidateContractError("/active_forbidden_lengths does not match the active set")
    return envelope


def _selector_arguments(
    value: object,
    *,
    definition: SelectorDefinition,
    path: str,
) -> dict[str, Any]:
    arguments = _object(value, path)
    _exact(arguments, set(definition.arguments), path)
    for name, argument in arguments.items():
        literals = definition.literal_domains.get(name)
        if literals is None or argument not in literals or isinstance(argument, bool):
            raise CandidateContractError(f"{path}/{name} is outside the validator domain")
    return arguments


def _validated_response(
    *,
    candidate: CandidateKind,
    representation: dict[str, Any],
    program: dict[str, Any],
    forbidden_lengths: tuple[int, ...],
) -> CompiledCandidateResponse:
    response_text = json.dumps(
        {
            "program": program,
            "design_summary": representation["design_summary"],
            "hypothesis": representation["hypothesis"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        validated = validate_single_program_response(
            response_text,
            forbidden_lengths=forbidden_lengths,
        )
        canonical = canonical_json_bytes(representation)
    except (CanonicalJsonError, ValueError) as exc:
        raise CandidateContractError(str(exc)) from exc
    return CompiledCandidateResponse(
        candidate=candidate,
        representation=representation,
        representation_sha256=hashlib.sha256(canonical).hexdigest(),
        program=validated.program,
        design_summary=validated.design_summary,
        hypothesis=validated.hypothesis,
    )


def _compiled_rewrite(
    *,
    bindings: list[tuple[str, dict[str, Any]]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "op": "block",
        "children": [*actions, {"op": "emit"}],
    }
    for name, expression in reversed(bindings):
        body = {"op": "let", "name": name, "value": expression, "body": body}
    return {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "entry": {
            "op": "try",
            "branches": [
                body,
                {"op": "no_plan", "reason": "NO_MATCH"},
            ],
        },
    }


def compile_slot_specific_response(
    response_text: str,
    *,
    slot_id: str,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
) -> CompiledCandidateResponse:
    """Compile one slot-specific response into the existing validated AST."""

    representation = _parse_candidate(
        response_text,
        schema_version=SLOT_SPECIFIC_SCHEMA_VERSION,
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=forbidden_lengths,
        payload_fields={"plan"},
    )
    plan = _object(representation["plan"], "/plan")
    terminal = _object(plan.get("terminal"), "/plan/terminal")
    kind = terminal.get("kind")
    if kind == "no_plan":
        _exact(plan, {"terminal"}, "/plan")
        _exact(terminal, {"kind", "reason"}, "/plan/terminal")
        if terminal["reason"] not in NO_PLAN_REASONS:
            raise CandidateContractError("/plan/terminal/reason is unsupported")
        return _validated_response(
            candidate="slot_specific",
            representation=representation,
            program={
                "schema_version": PROGRAM_SCHEMA_VERSION,
                "entry": {"op": "no_plan", "reason": terminal["reason"]},
            },
            forbidden_lengths=forbidden_lengths,
        )
    if kind != "emit":
        raise CandidateContractError("/plan/terminal/kind must be emit or no_plan")
    _exact(plan, {"selector", "pick", "action", "terminal"}, "/plan")
    _exact(terminal, {"kind"}, "/plan/terminal")
    contract = build_single_program_contract(forbidden_lengths)
    operators = _brief_operators(brief_id, contract)
    selector = _object(plan["selector"], "/plan/selector")
    _exact(selector, {"selector_id", "arguments"}, "/plan/selector")
    if selector["selector_id"] != operators.selector_id:
        raise CandidateContractError("/plan/selector/selector_id does not match the brief")
    selector_definition = contract.selectors[operators.selector_id]
    selector_arguments = _selector_arguments(
        selector["arguments"],
        definition=selector_definition,
        path="/plan/selector/arguments",
    )
    pick = _object(plan["pick"], "/plan/pick")
    _exact(pick, {"mode"}, "/plan/pick")
    if pick["mode"] not in _PICK_MODES:
        raise CandidateContractError("/plan/pick/mode is unsupported")
    action = _object(plan["action"], "/plan/action")
    _exact(action, {"action_id", "arguments"}, "/plan/action")
    if action["action_id"] != operators.action_id:
        raise CandidateContractError("/plan/action/action_id does not match the brief")
    definition = contract.actions[operators.action_id]
    arguments = _object(action["arguments"], "/plan/action/arguments")
    _exact(arguments, set(definition.arguments), "/plan/action/arguments")
    if any(value != "selected" for value in arguments.values()):
        raise CandidateContractError("/plan/action arguments must reference selected")
    action_arguments = {
        name: {"op": "ref", "name": "selected"} for name in definition.arguments
    }
    program = _compiled_rewrite(
        bindings=[
            (
                "candidates",
                {
                    "op": "selector",
                    "selector_id": operators.selector_id,
                    "arguments": selector_arguments,
                },
            ),
            (
                "selected",
                {
                    "op": "pick",
                    "source": {"op": "ref", "name": "candidates"},
                    "mode": pick["mode"],
                },
            ),
        ],
        actions=[
            {
                "op": "apply",
                "action_id": operators.action_id,
                "arguments": action_arguments,
            }
        ],
    )
    return _validated_response(
        candidate="slot_specific",
        representation=representation,
        program=program,
        forbidden_lengths=forbidden_lengths,
    )


def compile_flat_ir_response(
    response_text: str,
    *,
    slot_id: str,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
) -> CompiledCandidateResponse:
    """Compile the flat bounded IR into the existing validated AST."""

    representation = _parse_candidate(
        response_text,
        schema_version=FLAT_IR_SCHEMA_VERSION,
        slot_id=slot_id,
        brief_id=brief_id,
        forbidden_lengths=forbidden_lengths,
        payload_fields={"bindings", "steps", "terminal"},
    )
    bindings_value = representation["bindings"]
    steps_value = representation["steps"]
    if not isinstance(bindings_value, list) or not isinstance(steps_value, list):
        raise CandidateContractError("/bindings and /steps must be arrays")
    if len(bindings_value) + len(steps_value) > MAXIMUM_FLAT_LOGICAL_STEPS:
        raise CandidateContractError("flat program exceeds eight logical steps")
    terminal = _object(representation["terminal"], "/terminal")
    terminal_kind = terminal.get("kind")
    if terminal_kind == "no_plan":
        _exact(terminal, {"kind", "reason"}, "/terminal")
        if bindings_value or steps_value:
            raise CandidateContractError("no_plan cannot contain bindings or steps")
        if terminal["reason"] not in NO_PLAN_REASONS:
            raise CandidateContractError("/terminal/reason is unsupported")
        return _validated_response(
            candidate="flat_ir",
            representation=representation,
            program={
                "schema_version": PROGRAM_SCHEMA_VERSION,
                "entry": {"op": "no_plan", "reason": terminal["reason"]},
            },
            forbidden_lengths=forbidden_lengths,
        )
    if terminal_kind != "emit":
        raise CandidateContractError("/terminal/kind must be emit or no_plan")
    _exact(terminal, {"kind"}, "/terminal")
    if not steps_value:
        raise CandidateContractError("emit requires at least one action step")

    contract = build_single_program_contract(forbidden_lengths)
    operators = _brief_operators(brief_id, contract)
    environment: dict[str, ValueType] = {}
    compiled_bindings: list[tuple[str, dict[str, Any]]] = []
    selector_ids: list[str] = []
    for index, raw_binding in enumerate(bindings_value):
        path = f"/bindings/{index}"
        binding = _object(raw_binding, path)
        kind = binding.get("kind")
        value_type: ValueType
        if kind == "selector":
            _exact(binding, {"kind", "id", "selector_id", "arguments"}, path)
            name = _identifier(binding["id"], f"{path}/id")
            selector_id = _identifier(binding["selector_id"], f"{path}/selector_id")
            selector_definition = contract.selectors.get(selector_id)
            if selector_definition is None or selector_id not in {
                item.selector_id for item in BRIEF_OPERATORS.values()
            }:
                raise CandidateContractError(f"{path}/selector_id is unsupported")
            arguments = _selector_arguments(
                binding["arguments"],
                definition=selector_definition,
                path=f"{path}/arguments",
            )
            value_type = selector_definition.result_type
            expression = {
                "op": "selector",
                "selector_id": selector_id,
                "arguments": arguments,
            }
            selector_ids.append(selector_id)
        elif kind == "pick":
            _exact(binding, {"kind", "id", "source", "mode"}, path)
            name = _identifier(binding["id"], f"{path}/id")
            source = _identifier(binding["source"], f"{path}/source")
            if source not in environment:
                raise CandidateContractError(f"{path}/source is unresolved")
            picked_type = contract.pick_results.get(environment[source])
            if picked_type is None:
                raise CandidateContractError(f"{path}/source cannot be picked")
            value_type = picked_type
            if binding["mode"] not in _PICK_MODES:
                raise CandidateContractError(f"{path}/mode is unsupported")
            expression = {
                "op": "pick",
                "source": {"op": "ref", "name": source},
                "mode": binding["mode"],
            }
        else:
            raise CandidateContractError(f"{path}/kind is unsupported")
        if name in environment:
            raise CandidateContractError(f"{path}/id duplicates an earlier binding")
        environment[name] = value_type
        compiled_bindings.append((name, expression))

    compiled_actions: list[dict[str, Any]] = []
    action_ids: list[str] = []
    for index, raw_step in enumerate(steps_value):
        path = f"/steps/{index}"
        step = _object(raw_step, path)
        _exact(step, {"op", "action_id", "arguments"}, path)
        if step["op"] != "apply":
            raise CandidateContractError(f"{path}/op must be apply")
        action_id = _identifier(step["action_id"], f"{path}/action_id")
        action_definition = contract.actions.get(action_id)
        if action_definition is None or action_id not in {
            item.action_id for item in BRIEF_OPERATORS.values()
        }:
            raise CandidateContractError(f"{path}/action_id is unsupported")
        arguments = _object(step["arguments"], f"{path}/arguments")
        _exact(arguments, set(action_definition.arguments), f"{path}/arguments")
        compiled_arguments: dict[str, Any] = {}
        for argument_name, expected_type in action_definition.arguments.items():
            binding_name = _identifier(
                arguments[argument_name],
                f"{path}/arguments/{argument_name}",
            )
            actual_type = environment.get(binding_name)
            if actual_type is None:
                raise CandidateContractError(
                    f"{path}/arguments/{argument_name} is unresolved"
                )
            if actual_type != expected_type:
                raise CandidateContractError(
                    f"{path}/arguments/{argument_name} expects {expected_type}, "
                    f"got {actual_type}"
                )
            compiled_arguments[argument_name] = {"op": "ref", "name": binding_name}
        action_ids.append(action_id)
        compiled_actions.append(
            {
                "op": "apply",
                "action_id": action_id,
                "arguments": compiled_arguments,
            }
        )
    if selector_ids != [operators.selector_id]:
        raise CandidateContractError("flat program must use the brief's selector exactly once")
    if action_ids != [operators.action_id]:
        raise CandidateContractError("flat program must use the brief's action exactly once")
    program = _compiled_rewrite(bindings=compiled_bindings, actions=compiled_actions)
    return _validated_response(
        candidate="flat_ir",
        representation=representation,
        program=program,
        forbidden_lengths=forbidden_lengths,
    )


def compile_candidate_response(
    candidate: CandidateKind,
    response_text: str,
    *,
    slot_id: str,
    brief_id: str,
    forbidden_lengths: tuple[int, ...],
) -> CompiledCandidateResponse:
    if candidate == "slot_specific":
        return compile_slot_specific_response(
            response_text,
            slot_id=slot_id,
            brief_id=brief_id,
            forbidden_lengths=forbidden_lengths,
        )
    if candidate == "flat_ir":
        return compile_flat_ir_response(
            response_text,
            slot_id=slot_id,
            brief_id=brief_id,
            forbidden_lengths=forbidden_lengths,
        )
    raise ValueError(f"unknown candidate: {candidate}")


def _schema_graph(schema: dict[str, Any]) -> tuple[dict[str, set[str]], list[tuple[str, str]]]:
    definitions = schema.get("$defs")
    graph: dict[str, set[str]] = {}
    references: list[tuple[str, str]] = []
    if not isinstance(definitions, dict):
        return graph, references

    def visit(value: object, source: str) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = ref.removeprefix("#/$defs/").split("/", 1)[0]
                graph.setdefault(source, set()).add(target)
                references.append((source, target))
            for item in value.values():
                visit(item, source)
        elif isinstance(value, list):
            for item in value:
                visit(item, source)

    for name, definition in definitions.items():
        if isinstance(name, str):
            graph.setdefault(name, set())
            visit(definition, name)
    return graph, references


def _reaches(graph: dict[str, set[str]], source: str, target: str) -> bool:
    pending = [source]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, ()))
    return False


def inspect_schema_complexity(
    schema: dict[str, Any],
    *,
    prompt: str,
    system_prompt: str,
    maximum_generated_program_step_count: int,
) -> dict[str, Any]:
    """Return deterministic structural and model-facing input metrics."""

    metrics = {
        "object_variants": 0,
        "any_of_count": 0,
        "any_of_variants": 0,
        "one_of_count": 0,
        "one_of_variants": 0,
        "ref_count": 0,
        "required_field_count": 0,
        "optional_field_count": 0,
        "enum_value_count": 0,
        "const_count": 0,
        "maximum_schema_nesting_depth": 0,
    }

    def visit(value: object, depth: int) -> None:
        metrics["maximum_schema_nesting_depth"] = max(
            metrics["maximum_schema_nesting_depth"], depth
        )
        if isinstance(value, dict):
            if value.get("type") == "object":
                metrics["object_variants"] += 1
                properties = value.get("properties")
                required = value.get("required")
                if isinstance(properties, dict):
                    required_names = set(required) if isinstance(required, list) else set()
                    metrics["required_field_count"] += len(required_names)
                    metrics["optional_field_count"] += len(set(properties) - required_names)
            any_of = value.get("anyOf")
            if isinstance(any_of, list):
                metrics["any_of_count"] += 1
                metrics["any_of_variants"] += len(any_of)
            one_of = value.get("oneOf")
            if isinstance(one_of, list):
                metrics["one_of_count"] += 1
                metrics["one_of_variants"] += len(one_of)
            if isinstance(value.get("$ref"), str):
                metrics["ref_count"] += 1
            enum = value.get("enum")
            if isinstance(enum, list):
                metrics["enum_value_count"] += len(enum)
            if "const" in value:
                metrics["const_count"] += 1
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                visit(item, depth + 1)

    visit(schema, 1)
    graph, references = _schema_graph(schema)
    recursive = sum(
        1 for source, target in references if _reaches(graph, target, source)
    )
    canonical = canonical_json_bytes(schema)
    prompt_bytes = prompt.encode("utf-8")
    system_bytes = system_prompt.encode("utf-8")
    return {
        "schema_sha256": hashlib.sha256(canonical).hexdigest(),
        "schema_bytes": len(canonical),
        **metrics,
        "recursive_reference_count": recursive,
        "maximum_generated_program_step_count": maximum_generated_program_step_count,
        "prompt_bytes": len(prompt_bytes),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "system_prompt_bytes": len(system_bytes),
        "system_prompt_sha256": hashlib.sha256(system_bytes).hexdigest(),
    }


def build_schema_complexity_inventory(
    forbidden_lengths: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Inspect the rich control and both experimental candidates for four briefs."""

    inventory: list[dict[str, Any]] = []
    for brief_index, brief_id in enumerate(BRIEF_OPERATORS):
        slot_id = f"slot-{brief_index:02d}"
        rich = build_single_program_request(
            slot_id=slot_id,
            brief_id=brief_id,
            forbidden_lengths=forbidden_lengths,
        )
        inventory.append(
            {
                "candidate": "rich_recursive_control",
                "brief_id": brief_id,
                **inspect_schema_complexity(
                    build_single_program_output_schema(forbidden_lengths),
                    prompt=rich.prompt,
                    system_prompt=rich.system_prompt,
                    maximum_generated_program_step_count=256,
                ),
            }
        )
        candidates: tuple[CandidateKind, ...] = ("slot_specific", "flat_ir")
        for candidate in candidates:
            request = build_candidate_request(
                candidate=candidate,
                slot_id=slot_id,
                brief_id=brief_id,
                forbidden_lengths=forbidden_lengths,
            )
            inventory.append(
                {
                    "candidate": candidate,
                    "brief_id": brief_id,
                    **inspect_schema_complexity(
                        request.output_schema,
                        prompt=request.prompt,
                        system_prompt=request.system_prompt,
                        maximum_generated_program_step_count=(
                            1
                            if candidate == "slot_specific"
                            else MAXIMUM_FLAT_LOGICAL_STEPS
                        ),
                    ),
                }
            )
    return inventory
