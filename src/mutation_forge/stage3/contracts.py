"""Frozen, fail-closed contracts at the Stage 3 generation boundary.

The transport layer is intentionally kept separate from these value objects.  Every
object in this module is immutable, JSON serialisable, size bounded and rejects
unknown fields.  No model output is trusted until it has passed these checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

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
STRUCTURED_ENVELOPE_SCHEMA_VERSION: Final = "stage3.structured_envelope.v1"
USAGE_SCHEMA_VERSION: Final = "stage3.usage.v1"
PROVENANCE_SCHEMA_VERSION: Final = "stage3.provenance.v1"
SLOT_STATE_SCHEMA_VERSION: Final = "stage3.slot_state.v1"
GENERATION_SUMMARY_SCHEMA_VERSION: Final = "stage3.generation_summary.v1"
BEHAVIOR_IDENTITY_SCHEMA_VERSION: Final = "stage3.behavior_identity.v1"

MAX_SOURCE_BYTES: Final = 12 * 1024
MAX_DESIGN_SUMMARY_BYTES: Final = 2048
MAX_ASSUMPTION_BYTES: Final = 512
MAX_FIELD_NAME_BYTES: Final = 128
MAX_USED_FIELDS: Final = 32
MAX_ASSUMPTIONS: Final = 16
MAX_RESPONSE_BYTES: Final = 16 * 1024
MAX_EVENT_BYTES: Final = 64 * 1024
MAX_DETAIL_BYTES: Final = 2048
MAX_METADATA_BYTES: Final = 8192

_SCHEMA_ROOT = Path(__file__).resolve().parents[3] / "configs" / "schemas"


class Stage3ContractError(ValueError):
    """Raised when a Stage 3 value does not satisfy its frozen contract."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.code, self.path = code, path

    def as_dict(self) -> dict[str, JsonValue]:
        return {"code": self.code, "message": str(self), "path": self.path}


def canonical_bytes(value: object) -> bytes:
    """Canonical UTF-8 JSON bytes; NaN and infinity are never accepted."""
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage3ContractError("invalid_json", "value is not finite JSON") from exc


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _known_fields() -> frozenset[str]:
    names: set[str] = set()
    for prefix, filename in (
        ("ctx", "stage2b-context.schema.json"),
        ("proposal", "stage2b-proposal.schema.json"),
    ):
        try:
            raw = json.loads((_SCHEMA_ROOT / filename).read_text(encoding="utf-8"))
            names.update(f"{prefix}.{name}" for name in raw.get("properties", {}))
        except (OSError, json.JSONDecodeError):
            return frozenset()
    return frozenset(names)


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
        return canonical_bytes(self.as_dict())

    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def parse_generated_policy(
    value: object, *, max_response_bytes: int = MAX_RESPONSE_BYTES
) -> GeneratedPolicy:
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
    unknown = [name for name in fields if name not in _known_fields()]
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
    result = GeneratedPolicy(source, design, fields, assumptions)
    if len(result.canonical_bytes()) > max_response_bytes:
        raise Stage3ContractError(
            "response_too_large", f"response exceeds {max_response_bytes} bytes"
        )
    return result


def parse_stage2b_inputs(
    ctx: object, proposal: object, *, max_request_bytes: int = 64 * 1024
) -> tuple[RankerContext, RankerProposal]:
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


def _bounded_mapping(
    value: object, *, name: str, max_bytes: int = MAX_METADATA_BYTES
) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise Stage3ContractError("invalid_mapping", f"{name} must be an object")
    try:
        normalized = json.loads(canonical_bytes(dict(value)))
    except Stage3ContractError as exc:
        raise Stage3ContractError(exc.code, f"{name} must contain finite JSON") from exc
    if len(canonical_bytes(normalized)) > max_bytes:
        raise Stage3ContractError("mapping_too_large", f"{name} exceeds {max_bytes} bytes")
    return cast(dict[str, JsonValue], normalized)


@dataclass(frozen=True, slots=True)
class StructuredEnvelope:
    """Exact structured model envelope retained by generation artifacts."""

    response: GeneratedPolicy
    usage: Mapping[str, JsonValue]
    provenance: Mapping[str, JsonValue]
    schema_version: str = STRUCTURED_ENVELOPE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "response": self.response.as_dict(),
            "usage": dict(self.usage),
            "provenance": dict(self.provenance),
        }

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "usage", MappingProxyType(_bounded_mapping(self.usage, name="usage"))
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(_bounded_mapping(self.provenance, name="provenance")),
        )


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    event_type: str
    slot_id: str
    status: str
    detail: str = ""
    schema_version: str = GENERATION_EVENT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        _string(self.event_type, name="event_type", limit=128, path="$.event_type")
        _string(self.slot_id, name="slot_id", limit=64, path="$.slot_id")
        _string(self.status, name="status", limit=64, path="$.status")
        _string(
            self.detail, name="detail", limit=MAX_DETAIL_BYTES, path="$.detail", allow_empty=True
        )
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
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise Stage3ContractError("invalid_sequence", "sequence must be non-negative")
        _string(self.role, name="role", limit=64, path="$.role")
        _string(
            self.content, name="content", limit=MAX_DETAIL_BYTES, path="$.content", allow_empty=True
        )
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "content": self.content,
            "sequence": self.sequence,
        }


SLOT_STATES: Final = frozenset(
    {"pending", "running", "repairing", "succeeded", "failed", "skipped"}
)


@dataclass(frozen=True, slots=True)
class SlotState:
    slot_id: str
    state: str
    attempts: int = 0
    response_sha256: str | None = None
    error: str | None = None
    schema_version: str = SLOT_STATE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        if (
            self.state not in SLOT_STATES
            or isinstance(self.attempts, bool)
            or not 0 <= self.attempts <= 2
        ):
            raise Stage3ContractError("invalid_slot_state", "invalid slot state or attempts")
        _string(self.slot_id, name="slot_id", limit=64, path="$.slot_id")
        if self.response_sha256 is not None and (
            len(self.response_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.response_sha256)
        ):
            raise Stage3ContractError("invalid_hash", "response_sha256 must be lowercase SHA-256")
        if self.error is not None:
            _string(self.error, name="error", limit=MAX_DETAIL_BYTES, path="$.error")
        return {
            "schema_version": self.schema_version,
            "slot_id": self.slot_id,
            "state": self.state,
            "attempts": self.attempts,
            "response_sha256": self.response_sha256,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    slots: tuple[SlotState, ...]
    succeeded: int
    failed: int
    invalid: int = 0
    schema_version: str = GENERATION_SUMMARY_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        if any(isinstance(x, bool) or x < 0 for x in (self.succeeded, self.failed, self.invalid)):
            raise Stage3ContractError(
                "invalid_count", "summary counts must be non-negative integers"
            )
        if self.succeeded + self.failed + self.invalid > len(self.slots):
            raise Stage3ContractError("invalid_count", "summary counts exceed slot count")
        return {
            "schema_version": self.schema_version,
            "slots": [s.as_dict() for s in self.slots],
            "succeeded": self.succeeded,
            "failed": self.failed,
            "invalid": self.invalid,
        }


@dataclass(frozen=True, slots=True)
class BehaviorIdentity:
    source_sha256: str
    ast_sha256: str
    canonical_policy_sha256: str
    schema_version: str = BEHAVIOR_IDENTITY_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        for name in ("source_sha256", "ast_sha256", "canonical_policy_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise Stage3ContractError("invalid_hash", f"{name} must be lowercase SHA-256")
        return {
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "ast_sha256": self.ast_sha256,
            "canonical_policy_sha256": self.canonical_policy_sha256,
        }


def freeze_contract(value: JsonValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_contract(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_contract(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise Stage3ContractError("invalid_float", "non-finite values are forbidden")
    return value


__all__ = [
    "GeneratedPolicy",
    "StructuredEnvelope",
    "GenerationEvent",
    "TranscriptRecord",
    "SlotState",
    "GenerationSummary",
    "BehaviorIdentity",
    "SLOT_STATES",
    "Stage3ContractError",
    "canonical_bytes",
    "canonical_hash",
    "parse_generated_policy",
    "parse_stage2b_inputs",
    "freeze_contract",
]
