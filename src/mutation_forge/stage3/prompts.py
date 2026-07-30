"""Schema-derived, deliberately thin prompts for one-shot ranker generation."""
# Prompt prose intentionally contains long contract lines.
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
    value = json.loads(path.read_text(encoding="utf-8"))
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
        lines.append(
            f"- {field}: {', '.join(bits)} ({'required' if field in required else 'optional'})"
        )
    if schema.get("x-alignment"):
        lines.append(f"Alignment: {schema['x-alignment']}")
    return "\n".join(lines)


def schema_field_descriptions(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    return (
        _describe("Context", _schema(context_schema))
        + "\n\n"
        + _describe("Proposal", _schema(proposal_schema))
    )


def validator_limits() -> str:
    limits = SandboxLimits()
    nodes = ", ".join(sorted(n.__name__ for n in _ALLOWED_NODE_TYPES))
    builtins = ", ".join(sorted(SAFE_BUILTINS))
    return (
        f"safe builtins: {builtins}; max source bytes={limits.max_source_bytes}; max AST nodes={limits.max_ast_nodes}; "
        f"max static loop bound={limits.max_static_loop_bound}; per-call wall seconds={limits.per_call_wall_seconds}; "
        f"total wall seconds={limits.total_wall_seconds}; allowed AST nodes: {nodes}"
    )


def render_system_prompt(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    return (
        f"You write exactly one Python function priority(ctx, proposal) returning one finite number; larger is preferred. "
        f"Inputs are immutable plain data with exact schemas {SCIENTIFIC_CONTEXT_SCHEMA_VERSION} and {SCIENTIFIC_PROPOSAL_SCHEMA_VERSION}. "
        "Use only the documented fields, local arithmetic/control flow and safe built-ins. No imports, attributes, I/O, "
        "network, processes, mutation, hidden state, reflection, dynamic execution, while loops, recursion, NaN or infinity. "
        "The function ranks an already-legal proposal to minimise capped forbidden-cycle witness counts; ties are resolved by proposal_id. "
        "Return only the JSON object required by the output schema.\n\n"
        + schema_field_descriptions(context_schema, proposal_schema)
        + "\n\nSafety limits: "
        + validator_limits()
    )


def render_request_prompt(
    context_schema: Path = CONTEXT_SCHEMA, proposal_schema: Path = PROPOSAL_SCHEMA
) -> str:
    return (
        "Generation mode: {generation_mode}\nCandidate output schema version: stage3.generated_policy.v1\n"
        "Task brief: {task_instruction}\n\n"
        + schema_field_descriptions(context_schema, proposal_schema)
        + "\n\nImplement only priority(ctx, proposal) under the function contract above. State a falsifiable hypothesis in metadata. "
        "Return exactly keys schema_version, source, design_summary, used_fields, assumptions; no extra keys or prose."
    )


@dataclass(frozen=True, slots=True)
class PromptBundle:
    version: str
    system: str
    request: str
    output_schema: str
    context_schema_sha256: str
    proposal_schema_sha256: str

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
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_prompt_bundle(
    *,
    context_schema: Path = CONTEXT_SCHEMA,
    proposal_schema: Path = PROPOSAL_SCHEMA,
    output_schema: Path = OUTPUT_SCHEMA_PATH,
) -> PromptBundle:
    return PromptBundle(
        PROMPT_VERSION,
        render_system_prompt(context_schema, proposal_schema),
        render_request_prompt(context_schema, proposal_schema),
        output_schema.read_text(encoding="utf-8"),
        schema_sha256(context_schema),
        schema_sha256(proposal_schema),
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
