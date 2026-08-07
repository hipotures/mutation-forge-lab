"""Bounded host-owned Search Memory for the Native v3 Step 12D experiment."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .canonical import canonical_json_bytes

SEARCH_MEMORY_SCHEMA_VERSION = "mforge.native.search_memory.v3"
MODEL_SEARCH_MEMORY_SCHEMA_VERSION = "mforge.native.search_memory.model.v3"
MAX_SEEN_IDENTITIES = 64
MAX_PATTERNS_PER_OUTCOME = 8
MAX_ACTIVE_LINEAGES = 16
MAX_ARCHIVE_IDS = 16
MAX_SEARCH_MEMORY_BYTES = 16 * 1024
SCIENTIFIC_OUTCOMES = frozenset(
    {
        "NOT_EVALUATED",
        "VERIFIED_COUNTEREXAMPLE",
        "ACCEPTED_IMPROVEMENT",
        "REJECTED_WORSE",
        "REJECTED_EQUAL",
        "REJECTED_NOT_PROVED",
        "NO_PLAN",
        "NO_PLAN_AFTER_ILLEGAL_FINAL_STATE",
        "ILLEGAL_FINAL_STATE",
        "INCONCLUSIVE_SCORE",
        "PROGRAM_FAILURE",
    }
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PRINTABLE = re.compile(r"^[\x20-\x7e]+$")


class SearchMemoryError(ValueError):
    """The host Search Memory is invalid or exceeds its fixed bound."""


class DuplicateCandidateError(SearchMemoryError):
    """A generated candidate repeats an identity already held by the host."""


def program_families(
    ast: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return bounded selector and action family names from a validated AST."""

    selectors: set[str] = set()
    actions: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            selector = value.get("selector_id")
            action = value.get("action_id")
            if isinstance(selector, str):
                selectors.add(selector)
            if isinstance(action, str):
                actions.add(action)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(ast)
    return tuple(sorted(selectors)), tuple(sorted(actions))


def program_control_flow(ast: Mapping[str, Any]) -> tuple[str, ...]:
    """Return a bounded preorder summary of AST operation kinds."""

    operations: list[str] = []

    def visit(value: object) -> None:
        if len(operations) >= 32:
            return
        if isinstance(value, Mapping):
            operation = value.get("op")
            if isinstance(operation, str):
                operations.append(operation)
            for key in sorted(value):
                visit(value[key])
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(ast)
    return tuple(operations)


def _validated_hash(value: str, field: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise SearchMemoryError(f"{field} must be a lowercase SHA-256")
    return value


def _validated_identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SearchMemoryError(f"{field} is invalid")
    return value


def _validated_summary(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) > 600 or _PRINTABLE.fullmatch(value) is None:
        raise SearchMemoryError(f"{field} must be 1-600 printable ASCII characters")
    sentence_count = sum(value.count(mark) for mark in ".!?")
    if sentence_count < 1 or sentence_count > 3:
        raise SearchMemoryError(f"{field} must contain one to three sentences")
    return value


def _bounded_unique(
    values: tuple[str, ...],
    *,
    field: str,
    maximum: int,
    hash_values: bool = False,
) -> tuple[str, ...]:
    if len(values) > maximum:
        raise SearchMemoryError(f"{field} exceeds {maximum} entries")
    checked = tuple(
        _validated_hash(value, field) if hash_values else _validated_identifier(value, field)
        for value in values
    )
    if len(set(checked)) != len(checked):
        raise SearchMemoryError(f"{field} contains duplicates")
    return tuple(sorted(checked))


@dataclass(frozen=True, slots=True)
class PatternSummary:
    pattern_id: str
    selector_families: tuple[str, ...]
    action_families: tuple[str, ...]
    control_flow: tuple[str, ...]
    summary: str
    contract_status: Literal["VALID"]
    scientific_outcome: str
    model_hypothesis: str
    observed_effect: str | None
    primary_failure_code: str | None
    terminal_fallback_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pattern_id",
            _validated_identifier(self.pattern_id, "pattern_id"),
        )
        for field in ("selector_families", "action_families"):
            values = _bounded_unique(
                getattr(self, field),
                field=field,
                maximum=32,
            )
            object.__setattr__(self, field, values)
        if len(self.control_flow) > 32:
            raise SearchMemoryError("control_flow exceeds 32 entries")
        object.__setattr__(
            self,
            "control_flow",
            tuple(_validated_identifier(value, "control_flow") for value in self.control_flow),
        )
        if not self.selector_families and not self.action_families:
            raise SearchMemoryError("pattern must name a selector or action family")
        if not self.control_flow:
            raise SearchMemoryError("pattern must summarize control flow")
        _validated_summary(self.summary, "summary")
        if self.contract_status != "VALID":
            raise SearchMemoryError("contract_status must be VALID")
        if self.scientific_outcome not in SCIENTIFIC_OUTCOMES:
            raise SearchMemoryError("scientific_outcome is invalid")
        _validated_summary(self.model_hypothesis, "model_hypothesis")
        if self.observed_effect is not None:
            _validated_summary(self.observed_effect, "observed_effect")
        for field in ("primary_failure_code", "terminal_fallback_reason"):
            value = getattr(self, field)
            if value is not None:
                _validated_identifier(value, field)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "selector_families": list(self.selector_families),
            "action_families": list(self.action_families),
            "control_flow": list(self.control_flow),
            "summary": self.summary,
            "contract_status": self.contract_status,
            "scientific_outcome": self.scientific_outcome,
            "model_hypothesis": self.model_hypothesis,
            "observed_effect": self.observed_effect,
            "primary_failure_code": self.primary_failure_code,
            "terminal_fallback_reason": self.terminal_fallback_reason,
        }

    def model_facing_dict(self) -> dict[str, Any]:
        result = {
            "id": self.pattern_id,
            "selectors": list(self.selector_families),
            "actions": list(self.action_families),
            "control_flow": list(self.control_flow),
            "summary": self.summary,
            "contract_status": self.contract_status,
            "scientific_outcome": self.scientific_outcome,
            "model_hypothesis": self.model_hypothesis,
            "observed_effect": self.observed_effect,
        }
        if self.primary_failure_code is not None:
            result["primary_failure_code"] = self.primary_failure_code
        if self.terminal_fallback_reason is not None:
            result["terminal_fallback_reason"] = self.terminal_fallback_reason
        return result


@dataclass(frozen=True, slots=True)
class LineageSummary:
    candidate_id: str
    parent_id: str | None
    program_hash: str
    behavior_signature: str
    generation: int
    slot: int
    contract_status: Literal["VALID"]
    scientific_outcome: str
    summary: str

    def __post_init__(self) -> None:
        _validated_identifier(self.candidate_id, "candidate_id")
        if self.parent_id is not None:
            _validated_identifier(self.parent_id, "parent_id")
        _validated_hash(self.program_hash, "program_hash")
        _validated_hash(self.behavior_signature, "behavior_signature")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
            or isinstance(self.slot, bool)
            or not isinstance(self.slot, int)
            or self.slot < 0
        ):
            raise SearchMemoryError("generation and slot must be non-negative integers")
        if self.contract_status != "VALID":
            raise SearchMemoryError("contract_status must be VALID")
        if self.scientific_outcome not in SCIENTIFIC_OUTCOMES:
            raise SearchMemoryError("scientific_outcome is invalid")
        _validated_summary(self.summary, "summary")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "program_hash": self.program_hash,
            "behavior_signature": self.behavior_signature,
            "generation": self.generation,
            "slot": self.slot,
            "contract_status": self.contract_status,
            "scientific_outcome": self.scientific_outcome,
            "summary": self.summary,
        }

    def model_facing_dict(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "slot": self.slot,
            "contract_status": self.contract_status,
            "scientific_outcome": self.scientific_outcome,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class ActiveParentReference:
    candidate_id: str
    program_hash: str

    def __post_init__(self) -> None:
        _validated_identifier(self.candidate_id, "active_parent.candidate_id")
        _validated_hash(self.program_hash, "active_parent.program_hash")

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "program_hash": self.program_hash,
        }


@dataclass(frozen=True, slots=True)
class SearchMemoryV1:
    protocol_hash: str
    seen_program_hashes: tuple[str, ...]
    seen_behavior_signatures: tuple[str, ...]
    successful_patterns: tuple[PatternSummary, ...]
    tested_patterns: tuple[PatternSummary, ...]
    pending_patterns: tuple[PatternSummary, ...]
    active_lineages: tuple[LineageSummary, ...]
    validated_archive_ids: tuple[str, ...]
    active_parent: ActiveParentReference | None = None

    def __post_init__(self) -> None:
        _validated_hash(self.protocol_hash, "protocol_hash")
        object.__setattr__(
            self,
            "seen_program_hashes",
            _bounded_unique(
                self.seen_program_hashes,
                field="seen_program_hashes",
                maximum=MAX_SEEN_IDENTITIES,
                hash_values=True,
            ),
        )
        object.__setattr__(
            self,
            "seen_behavior_signatures",
            _bounded_unique(
                self.seen_behavior_signatures,
                field="seen_behavior_signatures",
                maximum=MAX_SEEN_IDENTITIES,
                hash_values=True,
            ),
        )
        for field in (
            "successful_patterns",
            "tested_patterns",
            "pending_patterns",
        ):
            patterns = getattr(self, field)
            if len(patterns) > MAX_PATTERNS_PER_OUTCOME:
                raise SearchMemoryError(f"{field} exceeds {MAX_PATTERNS_PER_OUTCOME} entries")
            ordered = tuple(sorted(patterns, key=lambda item: item.pattern_id))
            if len({item.pattern_id for item in ordered}) != len(ordered):
                raise SearchMemoryError(f"{field} contains duplicate pattern IDs")
            object.__setattr__(self, field, ordered)
        pattern_ids = [
            item.pattern_id
            for field in (
                "successful_patterns",
                "tested_patterns",
                "pending_patterns",
            )
            for item in getattr(self, field)
        ]
        if len(set(pattern_ids)) != len(pattern_ids):
            raise SearchMemoryError("pattern IDs must be unique across outcome groups")
        if len(self.active_lineages) > MAX_ACTIVE_LINEAGES:
            raise SearchMemoryError(f"active_lineages exceeds {MAX_ACTIVE_LINEAGES} entries")
        lineages = tuple(
            sorted(
                self.active_lineages,
                key=lambda item: (item.generation, item.slot, item.candidate_id),
            )
        )
        if len({item.candidate_id for item in lineages}) != len(lineages):
            raise SearchMemoryError("active_lineages contains duplicate candidates")
        object.__setattr__(self, "active_lineages", lineages)
        object.__setattr__(
            self,
            "validated_archive_ids",
            _bounded_unique(
                self.validated_archive_ids,
                field="validated_archive_ids",
                maximum=MAX_ARCHIVE_IDS,
            ),
        )
        if len(self.canonical_bytes()) > MAX_SEARCH_MEMORY_BYTES:
            raise SearchMemoryError(
                f"Search Memory exceeds {MAX_SEARCH_MEMORY_BYTES} canonical bytes"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEARCH_MEMORY_SCHEMA_VERSION,
            "protocol_hash": self.protocol_hash,
            "seen_program_hashes": list(self.seen_program_hashes),
            "seen_behavior_signatures": list(self.seen_behavior_signatures),
            "successful_patterns": [item.as_dict() for item in self.successful_patterns],
            "tested_patterns": [item.as_dict() for item in self.tested_patterns],
            "pending_patterns": [item.as_dict() for item in self.pending_patterns],
            "active_lineages": [item.as_dict() for item in self.active_lineages],
            "validated_archive_ids": list(self.validated_archive_ids),
            "active_parent": (None if self.active_parent is None else self.active_parent.as_dict()),
        }

    def model_facing_dict(self) -> dict[str, Any]:
        """Project semantic guidance without exposing host-owned identities."""

        active_parent: dict[str, Any] | None = None
        if self.active_parent is not None:
            lineage = next(
                (
                    item
                    for item in self.active_lineages
                    if item.candidate_id == self.active_parent.candidate_id
                ),
                None,
            )
            active_parent = {
                "id": self.active_parent.candidate_id,
                **(
                    {}
                    if lineage is None
                    else {
                        "summary": lineage.summary,
                        "contract_status": lineage.contract_status,
                        "scientific_outcome": lineage.scientific_outcome,
                    }
                ),
            }
        return {
            "schema_version": MODEL_SEARCH_MEMORY_SCHEMA_VERSION,
            "successful_patterns": [item.model_facing_dict() for item in self.successful_patterns],
            "tested_patterns": [item.model_facing_dict() for item in self.tested_patterns],
            "pending_patterns": [item.model_facing_dict() for item in self.pending_patterns],
            "active_lineages": [item.model_facing_dict() for item in self.active_lineages],
            "active_parent": active_parent,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def reject_duplicate(
    memory: SearchMemoryV1,
    *,
    program_hash: str,
    behavior_signature: str,
) -> None:
    """Apply the authoritative host duplicate gate."""

    _validated_hash(program_hash, "program_hash")
    _validated_hash(behavior_signature, "behavior_signature")
    if program_hash in memory.seen_program_hashes:
        raise DuplicateCandidateError("canonical program hash already exists")
    if behavior_signature in memory.seen_behavior_signatures:
        raise DuplicateCandidateError("behavior signature already exists")
