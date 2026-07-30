"""Deterministic, append-only filesystem archive for Stage 4 programs.

The archive deliberately has no database dependency.  JSON records are the authority;
an index is rebuilt from those records whenever it is needed.  This makes interrupted
writes harmless and permits a byte-for-byte equality check after a rebuild.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def deterministic_program_id(
    *, generation: int, slot: str | int, source_sha256: str, parent_id: str | None = None
) -> str:
    """Return an id stable across machines and process restarts."""
    payload = {
        "generation": generation,
        "slot": str(slot),
        "source_sha256": source_sha256,
        "parent_id": parent_id,
    }
    return "program-" + canonical_hash(payload)[:24]


@dataclass(frozen=True, slots=True)
class ProgramRecord:
    program_id: str
    source_path: str
    source_sha256: str
    normalized_ast_sha256: str
    behavior_signature: str | Mapping[str, JsonValue]
    generation: int
    slot: str
    parent_id: str | None = None
    parent_program_id: str | None = None
    mutation_brief_id: str | None = None
    request_id: str | None = None
    app_server_request_id: str | None = None
    thread_id: str | None = None
    app_server_thread_id: str | None = None
    turn_id: str | None = None
    app_server_turn_id: str | None = None
    usage: Mapping[str, JsonValue] = field(default_factory=dict)
    validation_status: str = "unknown"
    probe_status: str = "unknown"
    smoke_10k_status: str = "unknown"
    replay_status: str = "unknown"
    duplicate_of: str | None = None
    search_metrics: Mapping[str, JsonValue] = field(default_factory=dict)
    metrics: Mapping[str, JsonValue] = field(default_factory=dict)
    fitness_status: str = "unknown"
    tombstone: bool = False
    error: str | None = None
    seed_id: str | None = None
    generation_mode: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def behavior_signature_sha256(self) -> str:
        if isinstance(self.behavior_signature, str):
            return self.behavior_signature
        value = self.behavior_signature.get("signature_sha256")
        return value if isinstance(value, str) else canonical_hash(self.behavior_signature)

    @property
    def is_seed(self) -> bool:
        return self.generation == 0 or self.effective_parent_id is None

    @property
    def effective_parent_id(self) -> str | None:
        return self.parent_id or self.parent_program_id

    @property
    def effective_request_id(self) -> str | None:
        return self.request_id or self.app_server_request_id

    @property
    def unique(self) -> bool:
        return self.duplicate_of is None and not self.tombstone

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "stage4.program.v1",
            "program_id": self.program_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "normalized_ast_sha256": self.normalized_ast_sha256,
            "behavior_signature": cast(JsonValue, self.behavior_signature),
            "generation": self.generation,
            "slot": self.slot,
            "parent_id": self.effective_parent_id,
            "parent_program_id": self.effective_parent_id,
            "mutation_brief_id": self.mutation_brief_id,
            "request_id": self.effective_request_id,
            "app_server_request_id": self.effective_request_id,
            "thread_id": self.app_server_thread_id or self.thread_id,
            "app_server_thread_id": self.app_server_thread_id or self.thread_id,
            "turn_id": self.app_server_turn_id or self.turn_id,
            "app_server_turn_id": self.app_server_turn_id or self.turn_id,
            "usage": dict(self.usage),
            "validation_status": self.validation_status,
            "probe_status": self.probe_status,
            "smoke_10k_status": self.smoke_10k_status,
            "replay_status": self.replay_status,
            "duplicate_of": self.duplicate_of,
            "search_metrics": dict(self.search_metrics),
            "metrics": dict(self.metrics),
            "fitness_status": self.fitness_status,
            "tombstone": self.tombstone,
            "error": self.error,
            "seed_id": self.seed_id,
            "generation_mode": self.generation_mode,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProgramRecord:
        required = (
            "program_id",
            "source_path",
            "source_sha256",
            "normalized_ast_sha256",
            "behavior_signature",
            "generation",
            "slot",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(f"missing program fields: {', '.join(missing)}")
        kwargs: dict[str, Any] = {name: value[name] for name in required}
        for name in (
            "parent_id",
            "mutation_brief_id",
            "request_id",
            "thread_id",
            "turn_id",
            "duplicate_of",
            "error",
            "seed_id",
            "generation_mode",
        ):
            kwargs[name] = value.get(name)
        kwargs["parent_program_id"] = value.get("parent_program_id")
        kwargs["app_server_request_id"] = value.get("app_server_request_id")
        kwargs["app_server_thread_id"] = value.get("app_server_thread_id")
        kwargs["app_server_turn_id"] = value.get("app_server_turn_id")
        for name in ("usage", "search_metrics", "metadata"):
            raw = value.get(name, {})
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be an object")
            kwargs[name] = dict(raw)
        raw_metrics = value.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("metrics must be an object")
        kwargs["metrics"] = dict(raw_metrics)
        for name in (
            "validation_status",
            "probe_status",
            "smoke_10k_status",
            "replay_status",
            "fitness_status",
        ):
            kwargs[name] = str(value.get(name, "unknown"))
        kwargs["tombstone"] = bool(value.get("tombstone", False))
        kwargs["generation"] = int(kwargs["generation"])
        kwargs["slot"] = str(kwargs["slot"])
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class ReindexReport:
    records: tuple[ProgramRecord, ...]
    archive_hash: str
    errors: tuple[str, ...] = ()
    duplicate_program_ids: tuple[str, ...] = ()
    duplicate_slots: tuple[str, ...] = ()
    duplicate_requests: tuple[str, ...] = ()
    missing_requests: tuple[str, ...] = ()
    missing_parents: tuple[str, ...] = ()
    missing_sources: tuple[str, ...] = ()
    source_hash_mismatches: tuple[str, ...] = ()
    corrupt_files: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            not self.errors
            and not self.corrupt_files
            and not self.missing_parents
            and not self.missing_sources
            and not self.source_hash_mismatches
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "archive_hash": self.archive_hash,
            "record_count": len(self.records),
            "errors": list(self.errors),
            "duplicate_program_ids": list(self.duplicate_program_ids),
            "duplicate_slots": list(self.duplicate_slots),
            "duplicate_requests": list(self.duplicate_requests),
            "missing_requests": list(self.missing_requests),
            "missing_parents": list(self.missing_parents),
            "missing_sources": list(self.missing_sources),
            "source_hash_mismatches": list(self.source_hash_mismatches),
            "corrupt_files": list(self.corrupt_files),
        }


class ArchiveError(ValueError):
    pass


class ProgramArchive:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.records_dir = self.root / "programs"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: ProgramRecord | Mapping[str, Any]) -> ProgramRecord:
        item = record if isinstance(record, ProgramRecord) else ProgramRecord.from_dict(record)
        # The first record for an AST is its immutable representative.  A later
        # submission remains a full record (including usage/evaluation evidence),
        # but is marked as a duplicate before it is written.
        if item.duplicate_of is None:
            try:
                existing = self.reindex().records
            except Exception:
                existing = ()
            representative = next(
                (
                    prior
                    for prior in existing
                    if prior.normalized_ast_sha256 == item.normalized_ast_sha256
                    or prior.source_sha256 == item.source_sha256
                ),
                None,
            )
            if representative is not None:
                item = replace(item, duplicate_of=representative.program_id)
        path = self.records_dir / f"{item.program_id}.json"
        payload = canonical_bytes(item.as_dict())
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".program-", suffix=".tmp", dir=self.records_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                fd_final = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError as exc:
                raise ArchiveError(f"program record already exists: {item.program_id}") from exc
            try:
                with os.fdopen(fd_final, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.unlink(temp_name)
            dir_fd = os.open(self.records_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return item

    add = append

    def records(self) -> tuple[ProgramRecord, ...]:
        return self.reindex().records

    def reindex(self) -> ReindexReport:
        records: list[ProgramRecord] = []
        errors: list[str] = []
        corrupt: list[str] = []
        files = sorted(
            path for path in self.records_dir.glob("*.json") if not path.name.startswith(".")
        )
        for path in files:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, Mapping):
                    raise ValueError("record is not an object")
                records.append(ProgramRecord.from_dict(raw))
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                corrupt.append(path.name)
                errors.append(f"{path.name}: {exc}")
        records.sort(key=lambda item: (item.generation, item.slot, item.program_id))
        ids = [item.program_id for item in records]
        slots = [f"{item.generation}:{item.slot}" for item in records]
        requests = [item.effective_request_id for item in records if item.effective_request_id]
        missing_requests = tuple(
            sorted(
                item.program_id
                for item in records
                if item.generation > 0 and item.effective_request_id is None
            )
        )
        dup_ids = tuple(sorted(key for key, count in Counter(ids).items() if count > 1))
        dup_slots = tuple(sorted(key for key, count in Counter(slots).items() if count > 1))
        dup_requests = tuple(sorted(key for key, count in Counter(requests).items() if count > 1))
        known = set(ids)
        missing = tuple(
            sorted(
                {
                    item.effective_parent_id
                    for item in records
                    if item.effective_parent_id and item.effective_parent_id not in known
                }
            )
        )
        missing_sources: list[str] = []
        source_hash_mismatches: list[str] = []
        for item in records:
            if item.tombstone:
                continue
            source = Path(item.source_path)
            resolved = source if source.is_absolute() else self.root.parent / source
            if not resolved.is_file():
                missing_sources.append(item.program_id)
                continue
            if hashlib.sha256(resolved.read_bytes()).hexdigest() != item.source_sha256:
                source_hash_mismatches.append(item.program_id)
        for label, values in (
            ("duplicate program ids", dup_ids),
            ("duplicate slots", dup_slots),
            ("duplicate requests", dup_requests),
        ):
            if values:
                errors.append(f"{label}: {', '.join(values)}")
        if missing:
            errors.append("missing parents: " + ", ".join(missing))
        if missing_requests:
            errors.append("missing requests: " + ", ".join(missing_requests))
        if missing_sources:
            errors.append("missing sources: " + ", ".join(sorted(missing_sources)))
        if source_hash_mismatches:
            errors.append(
                "source hash mismatches: " + ", ".join(sorted(source_hash_mismatches))
            )
        digest = canonical_hash([item.as_dict() for item in records])
        return ReindexReport(
            tuple(records),
            digest,
            tuple(errors),
            dup_ids,
            dup_slots,
            dup_requests,
            missing_requests,
            missing,
            tuple(sorted(missing_sources)),
            tuple(sorted(source_hash_mismatches)),
            tuple(corrupt),
        )

    rebuild_index = reindex

    def lineage(self, program_id: str) -> tuple[str, ...]:
        by_id = {item.program_id: item for item in self.records()}
        if program_id not in by_id:
            raise ArchiveError(f"missing program: {program_id}")
        chain: list[str] = []
        seen: set[str] = set()
        current = program_id
        while True:
            if current in seen:
                raise ArchiveError(f"lineage cycle at {current}")
            seen.add(current)
            chain.append(current)
            item = by_id.get(current)
            if item is None:
                raise ArchiveError(f"missing parent: {current}")
            if item.generation == 0 or item.effective_parent_id is None:
                if item.generation != 0 and item.effective_parent_id is None:
                    raise ArchiveError(f"nonseed without parent: {current}")
                return tuple(chain)
            current = item.effective_parent_id

    lineage_lookup = lineage

    def inspect(self) -> dict[str, JsonValue]:
        report = self.reindex()
        records = report.records
        unique = [item for item in records if item.unique]
        usage: dict[str, JsonValue] = {}
        for item in records:
            for key, value in item.usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = cast(int | float, usage.get(key, 0)) + value
        leaders = sorted(unique, key=lambda item: item.program_id)[:8]
        return {
            "archive_hash": report.archive_hash,
            "counts": {
                "records": len(records),
                "unique": len(unique),
                "duplicates": sum(item.duplicate_of is not None for item in records),
                "tombstones": sum(item.tombstone for item in records),
                "errors": len(report.errors),
            },
            "leaders": [item.program_id for item in leaders],
            "lineage": {item.program_id: list(self.lineage(item.program_id)) for item in leaders},
            "usage": usage,
        }


Archive = ProgramArchive
