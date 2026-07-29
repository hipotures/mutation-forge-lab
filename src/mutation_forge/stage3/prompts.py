"""Versioned prompt bundle built from the frozen Stage 2B schemas."""

# Prompt prose intentionally contains long contract lines; lint generated code,
# while keeping the exact wording readable in the resulting prompt.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.sandbox.contracts import (
    SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
    SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
    SandboxLimits,
)
from mutation_forge.sandbox.validation import _ALLOWED_NODE_TYPES, SAFE_BUILTINS

PROMPT_VERSION = "ranker_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONTEXT_SCHEMA = REPO_ROOT / "configs/schemas/stage2b-context.schema.json"
PROPOSAL_SCHEMA = REPO_ROOT / "configs/schemas/stage2b-proposal.schema.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts/ranker_v1_system.md"
REQUEST_PROMPT_PATH = REPO_ROOT / "prompts/ranker_v1_request.md"
OUTPUT_SCHEMA_PATH = REPO_ROOT / "prompts/ranker_v1_output_schema.json"


def _schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"schema {path} must be an object")
    return cast(dict[str, Any], value)


def schema_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _describe(name: str, schema: dict[str, Any]) -> str:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [f"{name} schema ({schema.get('$id', 'unknown')}):"]
    if not isinstance(properties, dict):
        return "\n".join(lines)
    for field, raw in properties.items():
        if not isinstance(raw, dict):
            continue
        bits: list[str] = []
        if "const" in raw:
            bits.append(f"const={raw['const']!r}")
        elif "enum" in raw:
            bits.append(f"one of {raw['enum']!r}")
        elif "$ref" in raw:
            bits.append(str(raw["$ref"]))
        elif "type" in raw:
            bits.append(str(raw["type"]))
        for bound in ("minimum", "maximum", "minItems", "maxItems", "minLength", "maxLength"):
            if bound in raw:
                bits.append(f"{bound}={raw[bound]}")
        mark = "required" if field in required else "optional"
        lines.append(f"- {field}: {', '.join(bits)} ({mark})")
    if schema.get("x-alignment"):
        lines.append(f"Alignment: {schema['x-alignment']}")
    return "\n".join(lines)


def schema_field_descriptions(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    """Render field names, types, bounds and schema annotations directly from JSON schemas."""
    return (
        _describe("Context", _schema(context_schema))
        + "\n\n"
        + _describe("Proposal", _schema(proposal_schema))
    )


def validator_limits() -> str:
    limits = SandboxLimits()
    nodes = ", ".join(sorted(node.__name__ for node in _ALLOWED_NODE_TYPES))
    builtins = ", ".join(sorted(SAFE_BUILTINS))
    return (
        f"safe builtins: {builtins}; max source bytes={limits.max_source_bytes}; max AST nodes={limits.max_ast_nodes}; "
        f"max static loop bound={limits.max_static_loop_bound}; request bytes={limits.request_bytes}; response bytes={limits.response_bytes}; "
        f"per-call wall seconds={limits.per_call_wall_seconds}; total wall seconds={limits.total_wall_seconds}; "
        f"address space bytes={limits.address_space_bytes}; captured stdout/stderr bytes={limits.captured_output_bytes}; "
        f"allowed AST nodes: {nodes}"
    )


def render_system_prompt(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    return f"""You are a controlled program-synthesis component for a deterministic graph-search experiment.

Produce exactly one Python function named `priority(ctx, proposal)` returning one finite int or float; larger values are preferred. Inputs are immutable plain data. The function ranks an already-legal proposal and cannot mutate a graph, call a scorer/verifier, access files, network, processes, environment, terminal or repository data, import modules, or inspect hidden state.

Before validator dispatch, the host requires exact Stage 2B schema versions `{SCIENTIFIC_CONTEXT_SCHEMA_VERSION}` and `{SCIENTIFIC_PROPOSAL_SCHEMA_VERSION}`. Use only documented fields below, local arithmetic/control flow, and the safe built-ins. Never use absolute identifiers as semantic information.

Objective: minimize capped forbidden-cycle witness counts, with the evaluator's primary total count and a secondary shorter-cycle weighted penalty. Ties are deterministic by proposal_id after descending finite priority; do not claim a score improvement. Return a falsifiable design hypothesis in metadata.

{schema_field_descriptions(context_schema, proposal_schema)}

Safety contract: {validator_limits()}. No imports, attributes, while loops, recursion, lambdas, nested definitions, exceptions, I/O, mutation, reflection, dynamic execution, unbounded loops, NaN or infinity. Source must satisfy the validator and runtime limits.

Return only one JSON object matching the output schema; no Markdown or additional text. The host ignores claims and validates source independently."""


def render_request_prompt(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    return f"""Generation mode: {{generation_mode}}
Candidate output schema version: stage3.generated_policy.v1
Required input schemas: {SCIENTIFIC_CONTEXT_SCHEMA_VERSION}, {SCIENTIFIC_PROPOSAL_SCHEMA_VERSION}
Allowed built-ins and validator/runtime limits: {validator_limits()}

Documented fields (derived mechanically from the frozen schemas):
{schema_field_descriptions(context_schema, proposal_schema)}

Task brief: {{task_instruction}}
Use no baseline source, Stage2C or Stage2D outcome, other candidate, repository code/result, hidden test data, or absolute vertex identifier. The evaluator supplies equal deterministic budgets and checks the source before execution.

Return exactly one object with keys schema_version, source, design_summary, used_fields, assumptions. No extra keys, Markdown, or prose outside JSON. Keep source <= 12 KiB and metadata bounded. State a testable hypothesis rather than claiming improvement."""


@dataclass(frozen=True, slots=True)
class PromptBundle:
    version: str
    system: str
    request: str
    output_schema: str
    context_schema_sha256: str
    proposal_schema_sha256: str

    def render_slot_request(self, slot_id: str, brief: str) -> str:
        if slot_id not in {f"slot-{index:02d}" for index in range(8)}:
            raise ValueError("slot_id must be slot-00 through slot-07")
        if not brief.strip():
            raise ValueError("slot brief must not be empty")
        return self.request.replace("{generation_mode}", "new_strategy").replace(
            "{task_instruction}", f"{slot_id}: {brief}"
        )

    def stable_hash(self) -> str:
        payload = {
            "version": self.version,
            "system": self.system,
            "request": self.request,
            "output_schema": self.output_schema,
            "context_schema_sha256": self.context_schema_sha256,
            "proposal_schema_sha256": self.proposal_schema_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_prompt_bundle() -> PromptBundle:
    return PromptBundle(
        PROMPT_VERSION,
        render_system_prompt(),
        render_request_prompt(),
        OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"),
        schema_sha256(CONTEXT_SCHEMA),
        schema_sha256(PROPOSAL_SCHEMA),
    )


__all__ = [
    "PromptBundle",
    "PROMPT_VERSION",
    "schema_field_descriptions",
    "validator_limits",
    "render_system_prompt",
    "render_request_prompt",
    "load_prompt_bundle",
    "schema_sha256",
]
