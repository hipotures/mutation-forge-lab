"""Bounded, atomic artifacts for the Stage 3 generation harness.

The writer intentionally has no knowledge of an App Server implementation.  It
stores JSON records supplied by the orchestrator and strips values that look like
credentials or private machine paths before writing them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_SECRET = re.compile(
    r"(?i)(^|[_-])(access[_-]?token|refresh[_-]?token|api[_-]?key|"
    r"authorization|password|secret|cookie)($|[_-])"
)
_SECRET_VALUE = re.compile(r"(?i)(bearer\s+\S+|sk-[A-Za-z0-9_-]{12,})")
_PRIVATE = re.compile(r"(?:/home/[^/]+|/Users/[^/]+|[A-Za-z]:\\Users\\[^\\]+)")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe(value: object, key: str = "") -> object:
    if _SECRET.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): _safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v, key) for v in value]
    if isinstance(value, tuple):
        return [_safe(v, key) for v in value]
    if isinstance(value, str):
        return _PRIVATE.sub("[PRIVATE_PATH]", _SECRET_VALUE.sub("[REDACTED]", value))
    return value


class GenerationArtifacts:
    """Create and atomically update a bounded run directory.

    ``start`` writes a terminally-readable ``failed`` skeleton first.  A later
    ``finish`` updates the status and summary, so an interrupted process leaves
    useful artifacts rather than a half-written JSON document.
    """

    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        max_file_bytes: int = 1_048_576,
        max_total_bytes: int = 8_388_608,
    ) -> None:
        self.root = root / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self._started = False
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes

    def _bounded(self, payload: str) -> None:
        if len(payload.encode("utf-8")) > self.max_file_bytes:
            raise ValueError("artifact exceeds per-file byte bound")
        total = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        if total + len(payload.encode("utf-8")) > self.max_total_bytes:
            raise ValueError("artifact run exceeds total byte bound")

    def _write(self, relative: str, value: object) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(_safe(value)) + "\n"
        self._bounded(payload)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def start(self, metadata: Mapping[str, object]) -> None:
        if self._started:
            return
        self._started = True
        skeleton = dict(metadata)
        skeleton.setdefault("status", "failed")
        skeleton.setdefault("created_at", datetime.now(UTC).isoformat())
        self._write("generation_summary.json", skeleton)
        self._write("run_summary.json", {"status": "failed", "run_id": self.root.name})

    def write(self, name: str, value: object) -> Path:
        if not self._started:
            self.start({"run_id": self.root.name})
        return self._write(name, value)

    def write_text(self, name: str, value: str) -> Path:
        if not self._started:
            self.start({"run_id": self.root.name})
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        self._bounded(value)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def finish(self, status: str, summary: Mapping[str, object]) -> None:
        existing: dict[str, Any] = {}
        summary_path = self.root / "generation_summary.json"
        if summary_path.is_file():
            existing = cast(
                dict[str, Any],
                json.loads(summary_path.read_text(encoding="utf-8")),
            )
        final = {**existing, **dict(summary)}
        final["status"] = status
        final.setdefault("run_id", self.root.name)
        final.setdefault("finished_at", datetime.now(UTC).isoformat())
        self._write("generation_summary.json", final)
        self._write("run_summary.json", {"status": status, "run_id": self.root.name, **final})

    @staticmethod
    def read_summary(root: Path) -> dict[str, Any]:
        path = root / "generation_summary.json"
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def replay_generation(root: Path) -> dict[str, Any]:
    """Load and validate a persisted timing-stripped summary without a provider."""
    summary = GenerationArtifacts.read_summary(root)
    slots = summary.get("slots", [])
    if not isinstance(slots, list) or [s.get("slot") for s in slots if isinstance(s, dict)] != [
        f"slot-{i:02d}" for i in range(8)
    ]:
        raise ValueError("generation summary does not contain ordered slots 00..07")
    mismatches: list[str] = []
    smoke_calls = 0
    try:
        from mutation_forge.sandbox.contracts import SandboxLimits
        from mutation_forge.sandbox.validation import validate_policy

        generation_config = cast(
            dict[str, Any],
            json.loads((root / "generation_config.json").read_text(encoding="utf-8")),
        )
        raw_limits = generation_config.get("sandbox_limits")
        if not isinstance(raw_limits, dict):
            raise ValueError("replay is missing frozen sandbox limits")
        limits = SandboxLimits(**raw_limits)
        raw_smoke_calls = generation_config.get("smoke_calls")
        if (
            not isinstance(raw_smoke_calls, int)
            or isinstance(raw_smoke_calls, bool)
            or raw_smoke_calls < 0
        ):
            raise ValueError("replay smoke_calls is invalid")
        smoke_calls = raw_smoke_calls
        # Delayed import avoids a module cycle at import time.
        from mutation_forge.stage3.generation import _behavior

        for slot in slots:
            if not isinstance(slot, dict) or slot.get("status") not in {"accepted", "duplicate"}:
                continue
            name = str(slot["slot"])
            source_path = root / "slots" / name / "source.py"
            source = source_path.read_text(encoding="utf-8")
            identity = validate_policy(source, limits)
            if not identity.valid:
                mismatches.append(name)
                continue
            source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if source_hash != slot.get(
                "source_sha256"
            ) or identity.identity.normalized_ast_sha256 != slot.get("normalized_ast_sha256"):
                mismatches.append(name)
                continue
            persisted_behavior = cast(
                dict[str, Any],
                json.loads((root / "slots" / name / "behavior.json").read_text(encoding="utf-8")),
            )
            behavior, _ = _behavior(source, limits, smoke_calls)
            if persisted_behavior.get("signature") != behavior:
                mismatches.append(name)
    except (OSError, ValueError, KeyError) as error:
        mismatches.append(str(error))
    if mismatches:
        raise ValueError(f"replay validation mismatch: {mismatches}")
    return {
        **summary,
        "replay_validated": True,
        "replay_smoke_calls_per_candidate": smoke_calls,
        "provider_calls": 0,
    }
