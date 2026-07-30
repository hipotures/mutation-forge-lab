"""Append-only generation checkpoints and idempotent resume bookkeeping."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mutation_forge.models import JsonValue

from .archive import canonical_bytes, canonical_hash


@dataclass(frozen=True, slots=True)
class Checkpoint:
    generation: int
    completed_slots: tuple[str, ...] = ()
    pending_slots: tuple[str, ...] = ()
    parent_assignments: Mapping[str, str] = field(default_factory=dict)
    archive_hash: str = ""
    evaluation_status: Mapping[str, str] = field(default_factory=dict)
    request_idempotency_keys: tuple[str, ...] = ()
    turn_count: int = 0
    usage: Mapping[str, JsonValue] = field(default_factory=dict)
    token_count: int = 0
    checkpoint_id: str | None = None
    sequence: int = 0

    @property
    def completed(self) -> frozenset[str]:
        return frozenset(self.completed_slots)

    @property
    def pending(self) -> frozenset[str]:
        return frozenset(self.pending_slots)

    @property
    def request_keys(self) -> frozenset[str]:
        return frozenset(self.request_idempotency_keys)

    @property
    def generation_boundary(self) -> int:
        return self.generation

    @property
    def turn_usage(self) -> Mapping[str, JsonValue]:
        return self.usage

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "stage4.checkpoint.v1",
            "generation": self.generation,
            "completed_slots": list(self.completed_slots),
            "pending_slots": list(self.pending_slots),
            "parent_assignments": dict(self.parent_assignments),
            "archive_hash": self.archive_hash,
            "evaluation_status": dict(self.evaluation_status),
            "request_idempotency_keys": list(self.request_idempotency_keys),
            "turn_count": self.turn_count,
            "usage": dict(self.usage),
            "token_count": self.token_count,
            "checkpoint_id": self.checkpoint_id,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Checkpoint:
        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name, ())
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{name} must be an array")
            return tuple(str(item) for item in raw)

        assignments = value.get("parent_assignments", {})
        status = value.get("evaluation_status", {})
        usage = value.get("usage", {})
        if (
            not isinstance(assignments, Mapping)
            or not isinstance(status, Mapping)
            or not isinstance(usage, Mapping)
        ):
            raise ValueError("checkpoint mappings are invalid")
        return cls(
            generation=int(value["generation"]),
            completed_slots=strings("completed_slots"),
            pending_slots=strings("pending_slots"),
            parent_assignments={str(k): str(v) for k, v in assignments.items()},
            archive_hash=str(value.get("archive_hash", "")),
            evaluation_status={str(k): str(v) for k, v in status.items()},
            request_idempotency_keys=strings("request_idempotency_keys"),
            turn_count=int(value.get("turn_count", 0)),
            usage=dict(usage),
            token_count=int(value.get("token_count", 0)),
            checkpoint_id=value.get("checkpoint_id"),
            sequence=int(value.get("sequence", 0)),
        )


class CheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        sequence = checkpoint.sequence
        if sequence == 0:
            existing = self.list()
            sequence = existing[-1].sequence + 1 if existing else 1
            checkpoint = replace(checkpoint, sequence=sequence)
        payload = canonical_bytes(checkpoint.as_dict())
        path = self.root / f"checkpoint-{checkpoint.sequence:012d}.json"
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            final_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(final_fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return checkpoint

    append = save

    def list(self) -> list[Checkpoint]:
        result: list[Checkpoint] = []
        for path in sorted(self.root.glob("checkpoint-*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, Mapping):
                    result.append(Checkpoint.from_dict(raw))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
                continue
        return sorted(result, key=lambda item: (item.generation, item.sequence))

    def latest(self) -> Checkpoint | None:
        values = self.list()
        return values[-1] if values else None

    load = latest

    def archive_hash(self) -> str:
        return canonical_hash([checkpoint.as_dict() for checkpoint in self.list()])

    def should_request(self, key: str, checkpoint: Checkpoint | None = None) -> bool:
        current = checkpoint or self.latest()
        return current is None or key not in current.request_keys

    def should_evaluate(self, slot: str, checkpoint: Checkpoint | None = None) -> bool:
        current = checkpoint or self.latest()
        return current is None or slot not in current.completed

    def resume(self, checkpoint: Checkpoint | None = None) -> dict[str, JsonValue]:
        current = checkpoint or self.latest()
        if current is None:
            return {
                "generation": 0,
                "completed_slots": [],
                "pending_slots": [],
                "request_idempotency_keys": [],
                "turn_count": 0,
                "usage": {},
            }
        return {
            "generation": current.generation,
            "completed_slots": list(current.completed_slots),
            "pending_slots": list(current.pending_slots),
            "parent_assignments": dict(current.parent_assignments),
            "archive_hash": current.archive_hash,
            "evaluation_status": dict(current.evaluation_status),
            "request_idempotency_keys": list(current.request_idempotency_keys),
            "turn_count": current.turn_count,
            "usage": dict(current.usage),
            "token_count": current.token_count,
        }


CheckpointManager = CheckpointStore
