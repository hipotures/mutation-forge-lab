"""Generic append-only experiment checkpoints with chain verification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

CHECKPOINT_SCHEMA_VERSION = "mforge.experiment.checkpoint.v1"


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint is missing, corrupt, or no longer forms a valid chain."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class CheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, value: Mapping[str, Any]) -> dict[str, Any]:
        existing = self.list(verify=False)
        sequence = int(value.get("sequence", 0)) or (
            existing[-1]["sequence"] + 1 if existing else 1
        )
        if existing and sequence <= existing[-1]["sequence"]:
            raise CheckpointIntegrityError("checkpoint sequence must be append-only")
        previous = existing[-1] if existing else None
        payload: dict[str, Any] = dict(value)
        payload.update(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "sequence": sequence,
                "created_at": str(payload.get("created_at", datetime.now(UTC).isoformat())),
                "previous_sha256": previous.get("checkpoint_sha256") if previous else None,
            }
        )
        payload["checkpoint_id"] = str(
            payload.get("checkpoint_id") or f"checkpoint-{sequence:012d}"
        )
        payload["checkpoint_sha256"] = _digest(payload)
        path = self.root / f"checkpoint-{sequence:012d}.json"
        if path.exists():
            raise CheckpointIntegrityError(f"checkpoint already exists: {path.name}")
        fd, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(_canonical(payload) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            try:
                descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        return payload

    def list(self, *, verify: bool = True) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("checkpoint-*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                if verify:
                    raise CheckpointIntegrityError(f"cannot read checkpoint: {path.name}") from exc
                continue
            if not isinstance(value, dict):
                if verify:
                    raise CheckpointIntegrityError(f"checkpoint must be an object: {path.name}")
                continue
            result.append(cast(dict[str, Any], value))
        result.sort(key=lambda item: int(item.get("sequence", 0)))
        if verify:
            self._verify(result)
        return result

    @staticmethod
    def _verify(values: Sequence[dict[str, Any]]) -> None:
        previous: str | None = None
        for expected_sequence, value in enumerate(values, start=1):
            sequence = int(value.get("sequence", 0))
            if sequence != expected_sequence:
                raise CheckpointIntegrityError("checkpoint sequence has a gap or duplicate")
            if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointIntegrityError("unsupported checkpoint schema")
            if value.get("previous_sha256") != previous:
                raise CheckpointIntegrityError(f"checkpoint {sequence} has an invalid predecessor")
            expected_hash = value.get("checkpoint_sha256")
            unsigned = {key: item for key, item in value.items() if key != "checkpoint_sha256"}
            if expected_hash != _digest(unsigned):
                raise CheckpointIntegrityError(f"checkpoint {sequence} hash mismatch")
            previous = str(expected_hash)

    def latest(self) -> dict[str, Any] | None:
        values = self.list()
        return values[-1] if values else None

    def verify(self) -> bool:
        self.list()
        return True


__all__ = ["CHECKPOINT_SCHEMA_VERSION", "CheckpointIntegrityError", "CheckpointStore"]
