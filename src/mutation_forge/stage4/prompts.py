"""Deterministic Stage 4 generation and repair prompt rendering."""
# Prompt contract lines intentionally remain readable.
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutation_forge.stage3.prompts import (
    load_semantic_glossary,
    render_semantic_glossary,
    schema_field_descriptions,
    validator_limits,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_VERSION = "stage4.search.v1"
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "stage4-system.md"
REQUEST_PROMPT_PATH = REPO_ROOT / "prompts" / "stage4-request.md"
REPAIR_PROMPT_PATH = REPO_ROOT / "prompts" / "stage4-repair.md"
OUTPUT_SCHEMA_PATH = REPO_ROOT / "configs" / "schemas" / "stage4-generated-policy.schema.json"
BRIEFS_DIR = REPO_ROOT / "configs" / "stage4-briefs"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def schema_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PromptBundle:
    version: str
    system: str
    request: str
    repair: str
    output_schema: dict[str, Any]

    def stable_hash(self) -> str:
        return canonical_hash({"version": self.version, "system": self.system, "request": self.request, "repair": self.repair, "output_schema": self.output_schema})


def render_system_prompt() -> str:
    return (
        "You are proposing one safe ranker policy for Stage 4 search.\n\n"
        "PROGRAM CONTRACT\n\n"
        "Return exactly one JSON object matching stage4.generated_policy.v1. The source must define exactly priority(ctx, proposal) and no other top-level code.\n"
        "Only the selected proposal is applied and authoritatively scored; never assume oracle access or inspect a full candidate pool.\n"
        f"{validator_limits()}\n"
        "Use only the supplied schema, semantic glossary, one parent policy, assigned brief, compact search-training feedback, and bounded archive context."
    )


def render_request_prompt(slot_id: str = "slot-00", brief: str = "", parent_source: str = "", parent_metadata: dict[str, Any] | None = None, search_feedback: str = "", archive_context: str = "") -> str:
    if not slot_id.startswith("slot-"):
        raise ValueError("slot_id must be slot-00..slot-07")
    metadata = json.dumps(parent_metadata or {}, sort_keys=True, separators=(",", ":"))
    glossary = render_semantic_glossary(load_semantic_glossary())
    return (f"ASSIGNED SLOT: {slot_id}\n\nASSIGNED BRIEF\n{brief}\n\nPARENT SOURCE\n{parent_source}\n\nPARENT METADATA\n{metadata}\n\nSEARCH-TRAINING FEEDBACK (COMPACT)\n{search_feedback}\n\nBOUNDED ARCHIVE CONTEXT\n{archive_context}\n\n{schema_field_descriptions()}\n\n{glossary}\n\nWrite the policy and exact response fields now.")


def render_repair_prompt(diagnostics: dict[str, Any] | None = None, source: str = "") -> str:
    safe = diagnostics or {}
    allowed = {k: safe[k] for k in ("schema", "ast", "signature", "finite_output") if k in safe}
    return ("Repair the supplied policy using only bounded schema, AST, signature, and finite-output diagnostics.\n"
            "Performance, fitness, validation-set, archive, and trace feedback is unavailable and must not be inferred.\n\n"
            f"DIAGNOSTICS\n{json.dumps(allowed, sort_keys=True, separators=(',', ':'))}\n\nSOURCE\n{source}\n\n"
            "Return exactly one stage4.generated_policy.v1 object.")


def load_prompt_bundle() -> PromptBundle:
    output_schema = json.loads(OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
    return PromptBundle(PROMPT_VERSION, render_system_prompt(), render_request_prompt(), render_repair_prompt(), output_schema)


__all__ = ["PromptBundle", "PROMPT_VERSION", "render_system_prompt", "render_request_prompt", "render_repair_prompt", "load_prompt_bundle", "schema_sha256", "canonical_hash"]
