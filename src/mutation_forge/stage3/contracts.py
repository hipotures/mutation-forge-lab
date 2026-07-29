"""Frozen Stage 3 generation contracts.

The generation boundary is deliberately small: model output is one JSON object
containing source and advisory metadata.  This module performs the strict
shape/size checks before a response can reach Stage 2's validator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import (
    SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
    SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
    ContractError,
    RankerContext,
    RankerProposal,
    validate_ranker_inputs,
)

GENERATED_POLICY_SCHEMA_VERSION: Final = "stage3.generated_policy.v1"
GENERATION_EVENT_SCHEMA_VERSION: Final = "stage3.generation_event.v1"
GENERATION_TRANSCRIPT_SCHEMA_VERSION: Final = "stage3.transcript.v1"
MAX_SOURCE_BYTES: Final = 12 * 1024
MAX_DESIGN_SUMMARY_BYTES: Final = 2048
MAX_ASSUMPTION_BYTES: Final = 512
MAX_FIELD_NAME_BYTES: Final = 128
MAX_USED_FIELDS: Final = 32
MAX_ASSUMPTIONS: Final = 16
MAX_RESPONSE_BYTES: Final = 16 * 1024
MAX_EVENT_BYTES: Final = 64 * 1024

_SCHEMA_ROOT = Path(__file__).resolve().parents[3] / "configs" / "schemas"


def _known_fields() -> frozenset[str]:
    names: set[str] = set()
    for prefix, filename in (
        ("ctx", "stage2b-context.schema.json"),
        ("proposal", "stage2b-proposal.schema.json"),
    ):
        try:
            schema = json.loads((_SCHEMA_ROOT / filename).read_text(encoding="utf-8"))
            names.update(f"{prefix}.{name}" for name in schema.get("properties", {}))
        except (OSError, json.JSONDecodeError):
            return frozenset()
    return frozenset(names)


class Stage3ContractError(ValueError):
    """Raised when a Stage 3 value does not satisfy its frozen contract."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def as_dict(self) -> dict[str, JsonValue]:
        return {"code": self.code, "message": str(self), "path": self.path}


def _string(value: object, *, name: str, limit: int, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise Stage3ContractError("invalid_string", f"{name} must be a non-empty string", path)
    if len(value.encode("utf-8")) > limit:
        raise Stage3ContractError("string_too_large", f"{name} exceeds {limit} UTF-8 bytes", path)
    return value


def _string_list(
    value: object, *, name: str, max_items: int, item_limit: int, path: str
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_items:
        raise Stage3ContractError(
            "invalid_array", f"{name} must be an array of at most {max_items} strings", path
        )
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(
            _string(item, name=f"{name}[{index}]", limit=item_limit, path=f"{path}[{index}]")
        )
    if len(set(result)) != len(result):
        raise Stage3ContractError("duplicate_value", f"{name} must not contain duplicates", path)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GeneratedPolicy:
    """Validated, immutable model response passed to source validation."""

    source: str
    design_summary: str
    used_fields: tuple[str, ...]
    assumptions: tuple[str, ...]
    schema_version: str = GENERATED_POLICY_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "design_summary": self.design_summary,
            "used_fields": list(self.used_fields),
            "assumptions": list(self.assumptions),
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def parse_generated_policy(
    value: object, *, max_response_bytes: int = MAX_RESPONSE_BYTES
) -> GeneratedPolicy:
    """Parse model JSON strictly; unknown keys and malformed values are rejected."""

    if not isinstance(value, dict):
        raise Stage3ContractError("invalid_output", "generated policy must be a JSON object")
    expected = {"schema_version", "source", "design_summary", "used_fields", "assumptions"}
    if set(value) != expected:
        raise Stage3ContractError("invalid_keys", f"output keys must be exactly {sorted(expected)}")
    if value["schema_version"] != GENERATED_POLICY_SCHEMA_VERSION:
        raise Stage3ContractError(
            "invalid_schema_version",
            f"schema_version must be {GENERATED_POLICY_SCHEMA_VERSION!r}",
            "$.schema_version",
        )
    source = _string(value["source"], name="source", limit=MAX_SOURCE_BYTES, path="$.source")
    design = _string(
        value["design_summary"],
        name="design_summary",
        limit=MAX_DESIGN_SUMMARY_BYTES,
        path="$.design_summary",
    )
    fields = _string_list(
        value["used_fields"],
        name="used_fields",
        max_items=MAX_USED_FIELDS,
        item_limit=MAX_FIELD_NAME_BYTES,
        path="$.used_fields",
    )
    known = _known_fields()
    unknown = [field for field in fields if field not in known]
    if unknown:
        raise Stage3ContractError(
            "unknown_field",
            f"used_fields contains undocumented field {unknown[0]!r}",
            "$.used_fields",
        )
    assumptions = _string_list(
        value["assumptions"],
        name="assumptions",
        max_items=MAX_ASSUMPTIONS,
        item_limit=MAX_ASSUMPTION_BYTES,
        path="$.assumptions",
    )
    result = GeneratedPolicy(
        source=source, design_summary=design, used_fields=fields, assumptions=assumptions
    )
    if len(result.canonical_bytes()) > max_response_bytes:
        raise Stage3ContractError(
            "response_too_large", f"response exceeds {max_response_bytes} bytes"
        )
    return result


def parse_stage2b_inputs(
    ctx: object, proposal: object, *, max_request_bytes: int = 64 * 1024
) -> tuple[RankerContext, RankerProposal]:
    """Require exact Stage 2B schemas before dispatching to the validator."""

    if not isinstance(ctx, dict) or not isinstance(proposal, dict):
        raise Stage3ContractError("invalid_input", "ctx and proposal must be JSON objects")
    if ctx.get("schema_version") != SCIENTIFIC_CONTEXT_SCHEMA_VERSION:
        raise Stage3ContractError(
            "schema_version",
            f"ctx schema_version must be {SCIENTIFIC_CONTEXT_SCHEMA_VERSION!r}",
            "$.ctx.schema_version",
        )
    if proposal.get("schema_version") != SCIENTIFIC_PROPOSAL_SCHEMA_VERSION:
        raise Stage3ContractError(
            "schema_version",
            f"proposal schema_version must be {SCIENTIFIC_PROPOSAL_SCHEMA_VERSION!r}",
            "$.proposal.schema_version",
        )
    try:
        return validate_ranker_inputs(ctx, proposal, max_request_bytes=max_request_bytes)
    except ContractError as error:
        raise Stage3ContractError(error.code, str(error), error.path) from error


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    event_type: str
    slot_id: str
    status: str
    detail: str = ""
    schema_version: str = GENERATION_EVENT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "slot_id": self.slot_id,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    role: str
    content: str
    sequence: int
    schema_version: str = GENERATION_TRANSCRIPT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "content": self.content,
            "sequence": self.sequence,
        }


def freeze_contract(value: JsonValue) -> object:
    """Return recursively immutable data for callers retaining parsed payloads."""

    if isinstance(value, dict):
        return MappingProxyType({key: freeze_contract(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_contract(item) for item in value)
    return value


__all__ = [
    "GeneratedPolicy",
    "GenerationEvent",
    "TranscriptRecord",
    "Stage3ContractError",
    "GENERATED_POLICY_SCHEMA_VERSION",
    "GENERATION_EVENT_SCHEMA_VERSION",
    "GENERATION_TRANSCRIPT_SCHEMA_VERSION",
    "MAX_SOURCE_BYTES",
    "MAX_RESPONSE_BYTES",
    "parse_generated_policy",
    "parse_stage2b_inputs",
    "freeze_contract",
]
