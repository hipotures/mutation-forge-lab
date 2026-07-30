"""Schema-derived, deliberately thin prompts for one-shot ranker generation."""
# Prompt prose intentionally contains long contract lines.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.validation import _ALLOWED_NODE_TYPES, SAFE_BUILTINS

PROMPT_VERSION = "ranker_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONTEXT_SCHEMA = REPO_ROOT / "configs/schemas/stage2b-context.schema.json"
PROPOSAL_SCHEMA = REPO_ROOT / "configs/schemas/stage2b-proposal.schema.json"
SEMANTICS_GLOSSARY = REPO_ROOT / "configs/stage3-field-semantics.v1.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts/ranker_v1_system.md"
REQUEST_PROMPT_PATH = REPO_ROOT / "prompts/ranker_v1_request.md"
OUTPUT_SCHEMA_PATH = REPO_ROOT / "prompts/ranker_v1_output_schema.json"


def _schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"schema {path} must be an object")
    return cast(dict[str, Any], value)


def schema_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_semantic_glossary(
    path: Path = SEMANTICS_GLOSSARY,
    *,
    context_schema: Path = CONTEXT_SCHEMA,
    proposal_schema: Path = PROPOSAL_SCHEMA,
) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > 64 * 1024:
        raise ValueError("semantic glossary must be non-empty and at most 64 KiB")
    value = json.loads(raw)
    root_keys = {
        "schema_version",
        "context_schema_sha256",
        "proposal_schema_sha256",
        "decision_problem",
        "vector_alignments",
        "aliases",
        "budget_caveats",
        "fields",
    }
    if (
        not isinstance(value, dict)
        or set(value) != root_keys
        or value.get("schema_version") != "stage3.field_semantics.v1"
        or value.get("context_schema_sha256") != schema_sha256(context_schema)
        or value.get("proposal_schema_sha256") != schema_sha256(proposal_schema)
    ):
        raise ValueError("semantic glossary root or schema identity is invalid")
    decision = value.get("decision_problem")
    required_decision = {
        "objective",
        "pool_protocol",
        "selection_rule",
        "scoring_boundary",
        "context_scope",
        "proposal_scope",
    }
    if not isinstance(decision, dict) or set(decision) != required_decision:
        raise ValueError("semantic glossary decision problem is incomplete")
    fields = value.get("fields")
    if not isinstance(fields, dict) or set(fields) != {"ctx", "proposal"}:
        raise ValueError("semantic glossary fields must contain ctx and proposal")
    known: set[str] = set()
    for prefix, schema_path in (("ctx", context_schema), ("proposal", proposal_schema)):
        properties = _schema(schema_path).get("properties")
        entries = fields.get(prefix)
        if (
            not isinstance(properties, dict)
            or not isinstance(entries, dict)
            or set(entries) != set(properties)
        ):
            raise ValueError(f"semantic glossary does not exactly cover {prefix} schema fields")
        for name, entry in entries.items():
            if (
                not isinstance(entry, dict)
                or set(entry) != {"scope", "semantics", "direction"}
                or any(
                    not isinstance(entry.get(key), str) or not entry[key].strip()
                    for key in ("scope", "semantics", "direction")
                )
            ):
                raise ValueError(f"invalid semantic glossary entry {prefix}.{name}")
            known.add(f"{prefix}.{name}")
    for name in ("vector_alignments", "budget_caveats"):
        entries = value.get(name)
        if (
            not isinstance(entries, list)
            or not entries
            or any(not isinstance(item, str) or not item.strip() for item in entries)
        ):
            raise ValueError(f"semantic glossary {name} must be non-empty strings")
    aliases = value.get("aliases")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError("semantic glossary aliases must be non-empty")
    for alias in aliases:
        if (
            not isinstance(alias, dict)
            or set(alias) != {"fields", "relationship"}
            or not isinstance(alias.get("fields"), list)
            or len(alias["fields"]) < 2
            or any(field not in known for field in alias["fields"])
            or not isinstance(alias.get("relationship"), str)
            or not alias["relationship"].strip()
        ):
            raise ValueError("invalid semantic glossary alias")
    return cast(dict[str, Any], value)


def _resolve_local_ref(root: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    resolved = raw
    seen: set[str] = set()
    while "$ref" in resolved:
        reference = resolved.get("$ref")
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise ValueError(f"unsupported schema reference {reference!r}")
        if reference in seen:
            raise ValueError(f"cyclic schema reference {reference!r}")
        seen.add(reference)
        definitions = root.get("$defs")
        key = reference.removeprefix("#/$defs/")
        if not isinstance(definitions, dict) or not isinstance(definitions.get(key), dict):
            raise ValueError(f"unresolved schema reference {reference!r}")
        siblings = {key: value for key, value in resolved.items() if key != "$ref"}
        resolved = {**cast(dict[str, Any], definitions[key]), **siblings}
    return resolved


def _bounded_type(raw: dict[str, Any], root: dict[str, Any]) -> str:
    raw = _resolve_local_ref(root, raw)
    variants = raw.get("anyOf")
    if isinstance(variants, list):
        descriptions = [
            _bounded_type(cast(dict[str, Any], item), root)
            for item in variants
            if isinstance(item, dict)
        ]
        if len(descriptions) != len(variants) or not descriptions:
            raise ValueError("unsupported anyOf schema")
        return " or ".join(descriptions)
    if "const" in raw:
        value = raw["const"]
        value_type = "string" if isinstance(value, str) else type(value).__name__
        return f"{value_type} constant {value!r}"
    if "enum" in raw:
        values = raw["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError("enum must contain values")
        value_type = "integer" if all(isinstance(item, int) for item in values) else "string"
        return f"{value_type}; allowed values: {values!r}"
    property_type = raw.get("type")
    if property_type == "null":
        return "null"
    if property_type == "array":
        items = raw.get("items")
        if not isinstance(items, dict):
            raise ValueError("array schema requires items")
        item_description = _bounded_type(items, root)
        minimum = raw.get("minItems")
        maximum = raw.get("maxItems")
        if isinstance(minimum, int) and isinstance(maximum, int):
            bounds = f"{minimum}..{maximum} items"
        elif isinstance(minimum, int):
            bounds = f"at least {minimum} items"
        elif isinstance(maximum, int):
            bounds = f"at most {maximum} items"
        else:
            bounds = "items"
        unique = "; unique" if raw.get("uniqueItems") is True else ""
        return f"array ({bounds}{unique}) of {item_description}"
    if property_type not in {"string", "integer", "number", "boolean"}:
        raise ValueError(f"unsupported property schema {raw!r}")
    details = [str(property_type)]
    minimum, maximum = raw.get("minimum"), raw.get("maximum")
    if minimum is not None and maximum is not None:
        details.append(f"range [{minimum}, {maximum}]")
    elif minimum is not None:
        details.append(f"minimum {minimum}")
    elif maximum is not None:
        details.append(f"maximum {maximum}")
    if "pattern" in raw:
        details.append(f"pattern {raw['pattern']!r}")
    minimum_length, maximum_length = raw.get("minLength"), raw.get("maxLength")
    if isinstance(minimum_length, int):
        details.append(f"minimum length {minimum_length}")
    if isinstance(maximum_length, int):
        details.append(f"maximum length {maximum_length}")
    return "; ".join(details)


def _describe(name: str, schema: dict[str, Any]) -> str:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [f"{name} schema ({schema.get('$id', 'unknown')}):"]
    if not isinstance(properties, dict):
        return "\n".join(lines)
    for field, raw in properties.items():
        if not isinstance(raw, dict):
            continue
        lines.append(
            f"- {field}: {_bounded_type(raw, schema)} "
            f"({'required' if field in required else 'optional'})"
        )
    if schema.get("x-alignment"):
        lines.append(f"Alignment: {schema['x-alignment']}")
    if schema.get("x-authority"):
        lines.append(f"Authority boundary: {schema['x-authority']}")
    return "\n".join(lines)


def schema_field_descriptions(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    return (
        _describe("Context", _schema(context_schema))
        + "\n\n"
        + _describe("Proposal", _schema(proposal_schema))
    )


def render_semantic_glossary(
    glossary: dict[str, Any],
    *,
    context_schema: Path = CONTEXT_SCHEMA,
    proposal_schema: Path = PROPOSAL_SCHEMA,
) -> str:
    decision = cast(dict[str, str], glossary["decision_problem"])
    lines = [
        "SCIENTIFIC DECISION PROBLEM",
        "",
        decision["objective"],
        "",
        decision["pool_protocol"],
        decision["selection_rule"],
        decision["scoring_boundary"],
        "",
        "IMPORTANT WITHIN-POOL DISTINCTION",
        "",
        decision["context_scope"],
        decision["proposal_scope"],
        "",
    ]
    schemas = {"ctx": _schema(context_schema), "proposal": _schema(proposal_schema)}
    headings = {
        "ctx": "CONTEXT FIELDS (POOL-CONSTANT)",
        "proposal": "PROPOSAL FIELDS (CANDIDATE-SPECIFIC OR PROVENANCE)",
    }
    fields = cast(dict[str, dict[str, dict[str, str]]], glossary["fields"])
    for prefix in ("ctx", "proposal"):
        lines.extend([headings[prefix], ""])
        properties = cast(dict[str, dict[str, Any]], schemas[prefix]["properties"])
        for name, entry in fields[prefix].items():
            lines.append(
                f"- {prefix}.{name} [{_bounded_type(properties[name], schemas[prefix])}; "
                f"scope={entry['scope']}]:"
            )
            lines.append(f"  {entry['semantics']}")
            lines.append(f"  Interpretation: {entry['direction']}.")
        lines.append("")
    lines.extend(["VECTOR ALIGNMENT", ""])
    lines.extend(f"- {item}" for item in glossary["vector_alignments"])
    lines.extend(["", "ALIASES AND REDUNDANCIES", ""])
    for alias in glossary["aliases"]:
        lines.append(f"- {', '.join(alias['fields'])}: {alias['relationship']}.")
    lines.extend(["", "BOUNDED-FEATURE CAVEATS", ""])
    lines.extend(f"- {item}" for item in glossary["budget_caveats"])
    return "\n".join(lines)


def validator_limits() -> str:
    limits = SandboxLimits()
    nodes = ", ".join(sorted(n.__name__ for n in _ALLOWED_NODE_TYPES))
    builtins = ", ".join(sorted(SAFE_BUILTINS))
    return (
        f"safe builtins: {builtins}; max source bytes={limits.max_source_bytes}; max AST nodes={limits.max_ast_nodes}; "
        f"max static loop bound={limits.max_static_loop_bound}; per-call wall seconds={limits.per_call_wall_seconds}; "
        f"total wall seconds={limits.total_wall_seconds}; allowed AST nodes: {nodes}"
    )


def program_contract() -> str:
    limits = SandboxLimits()
    builtins = ", ".join(sorted(SAFE_BUILTINS))
    return (
        "PROGRAM CONTRACT\n\n"
        "Return source containing exactly one top-level function with this exact unannotated signature:\n\n"
        "def priority(ctx, proposal):\n"
        "    ...\n"
        "    return finite_number\n\n"
        "- The source must contain exactly one return statement, and it must be the final top-level statement in priority.\n"
        "- Return a finite int or float; bool, NaN, infinity, complex values, and containers are rejected.\n"
        "- Read ctx and proposal only by indexing or slicing. Do not mutate either input.\n"
        "- Allowed local control flow: assignments, arithmetic, comparisons, Boolean logic, conditionals, and statically bounded for loops.\n"
        f"- Allowed built-ins only: {builtins}.\n"
        "- No imports, attributes or method calls, comprehensions, lambda, recursion, while, try, with, yield, async, decorators, annotations, classes, nested functions, reflection, dynamic execution, I/O, environment, process, network, database, RNG, or hidden state.\n"
        f"- Source <= {limits.max_source_bytes} bytes; AST <= {limits.max_ast_nodes} nodes; "
        f"every loop bound <= {limits.max_static_loop_bound}; per-call wall <= "
        f"{limits.per_call_wall_seconds} seconds; total smoke wall <= "
        f"{limits.total_wall_seconds} seconds."
    )


def render_system_prompt(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    return (
        "You are a thin, no-tool inference provider designing one deterministic structural "
        "ranking heuristic under the supplied scientific and Python contracts. Return only "
        "the strict JSON object requested by the user. Do not claim benchmark evidence, use "
        "unavailable scores, or request tools, files, repositories, network, or runtime context."
    )


def render_request_prompt(
    context_schema: Path = CONTEXT_SCHEMA,
    proposal_schema: Path = PROPOSAL_SCHEMA,
    semantics_glossary: Path = SEMANTICS_GLOSSARY,
) -> str:
    glossary = load_semantic_glossary(
        semantics_glossary,
        context_schema=context_schema,
        proposal_schema=proposal_schema,
    )
    return (
        "Generate one new deterministic ranker for this frozen slot.\n"
        "Generation mode: {generation_mode}\n"
        "Candidate output schema version: stage3.generated_policy.v1\n"
        "Task brief: {task_instruction}\n\n"
        + render_semantic_glossary(
            glossary,
            context_schema=context_schema,
            proposal_schema=proposal_schema,
        )
        + "\n\n"
        + program_contract()
        + "\n\nOUTPUT FIELD REQUIREMENTS\n\n"
        "- schema_version: exactly \"stage3.generated_policy.v1\".\n"
        "- source: only the complete priority(ctx, proposal) function; no imports or other definitions.\n"
        "- design_summary: begin with \"Hypothesis:\" and state why this ranking should select better mutations than an unstructured selection rule, in a falsifiable way.\n"
        "- used_fields: list every accessed field exactly once as ctx.<field> or proposal.<field>.\n"
        "- assumptions: explicitly list each assumed direction of effect that is not guaranteed by the field definition; use an empty array if none.\n\n"
        "Return exactly one JSON object with keys schema_version, source, design_summary, "
        "used_fields, assumptions. Return no Markdown, commentary, benchmark claim, or extra key."
    )


@dataclass(frozen=True, slots=True)
class PromptBundle:
    version: str
    system: str
    request: str
    output_schema: str
    context_schema_sha256: str
    proposal_schema_sha256: str
    semantic_glossary_sha256: str

    @property
    def base_instructions(self) -> str:
        return self.system

    def render_slot_request(
        self,
        slot_id: str,
        brief: str,
        *,
        generation_mode: str = "new_strategy",
        focus: str | None = None,
    ) -> str:
        if slot_id not in {f"slot-{i:02d}" for i in range(8)}:
            raise ValueError("slot_id must be slot-00 through slot-07")
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError("slot brief must not be empty")
        if generation_mode != "new_strategy":
            raise ValueError("Stage 3 slots require generation_mode=new_strategy")
        if focus is not None and (
            not isinstance(focus, str) or not focus or len(focus.encode("utf-8")) > 128
        ):
            raise ValueError("slot focus must be a bounded non-empty string")
        task = f"{slot_id}: {brief}"
        if focus is not None:
            task += f"\nPreregistered focus: {focus}"
        return self.request.replace("{generation_mode}", generation_mode).replace(
            "{task_instruction}", task
        )

    def stable_hash(self) -> str:
        payload = {
            "version": self.version,
            "system": self.system,
            "request": self.request,
            "output_schema": self.output_schema,
            "context_schema_sha256": self.context_schema_sha256,
            "proposal_schema_sha256": self.proposal_schema_sha256,
            "semantic_glossary_sha256": self.semantic_glossary_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_prompt_bundle(
    *,
    context_schema: Path = CONTEXT_SCHEMA,
    proposal_schema: Path = PROPOSAL_SCHEMA,
    semantics_glossary: Path = SEMANTICS_GLOSSARY,
    output_schema: Path = OUTPUT_SCHEMA_PATH,
) -> PromptBundle:
    return PromptBundle(
        PROMPT_VERSION,
        render_system_prompt(context_schema, proposal_schema),
        render_request_prompt(context_schema, proposal_schema, semantics_glossary),
        output_schema.read_text(encoding="utf-8"),
        schema_sha256(context_schema),
        schema_sha256(proposal_schema),
        schema_sha256(semantics_glossary),
    )


__all__ = [
    "PromptBundle",
    "PROMPT_VERSION",
    "schema_field_descriptions",
    "render_semantic_glossary",
    "load_semantic_glossary",
    "program_contract",
    "validator_limits",
    "render_system_prompt",
    "render_request_prompt",
    "load_prompt_bundle",
    "schema_sha256",
]
