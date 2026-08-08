"""Representation-independent serial execution records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mutation_forge.models import JsonValue


@dataclass(frozen=True, slots=True)
class ProgramFailure:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    kind: str
    path: str
    payload: Mapping[str, JsonValue]

    def as_dict(self) -> dict[str, JsonValue]:
        return {"kind": self.kind, "path": self.path, "payload": dict(self.payload)}
