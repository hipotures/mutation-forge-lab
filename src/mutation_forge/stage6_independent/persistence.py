"""Canonical, resumable persistence for the provider-free verification pass.

This module deliberately has no dependency on any stage runner.  Shards are
newline-delimited canonical JSON members in deterministic gzip containers;
every write is read back and hashed before the state file is advanced.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "stage6.independent.persistence.v1"
MAX_SHARD_BYTES = 32 * 1024 * 1024
TIMING_ONLY_FIELDS = frozenset(
    {
        "timing_ns",
        "first_improvement_ns",
        "ranker_elapsed_ns",
        "selected_scoring_ns",
        "pool_legality_ns",
        "pool_feature_ns",
        "elapsed_ns",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "timing",
        "timing_profile",
    }
)


def canonical_bytes(value: object) -> bytes:
    """Encode JSON-compatible data using the frozen byte representation."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def timing_stripped(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): timing_stripped(item)
            for key, item in value.items()
            if str(key) not in TIMING_ONLY_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [timing_stripped(item) for item in value]
    return value


def canonical_record_hash(record: Mapping[str, Any]) -> str:
    without = {key: value for key, value in record.items() if key not in {"canonical_hash", "canonical_episode_sha256"}}
    return hashlib.sha256(canonical_bytes(timing_stripped(without))).hexdigest()


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def safe_component(value: str, *, name: str = "name") -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"unsafe {name}")
    return value


def safe_child(root: Path, name: str) -> Path:
    component = safe_component(name, name="artifact name")
    root = root.resolve()
    candidate = (root / component).resolve()
    if candidate.parent != root:
        raise ValueError("artifact path escapes output directory")
    return candidate


def _atomic_write(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == data:
            return
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False) -> str:
    target = Path(path)
    if any(part in {"..", "."} for part in target.parts):
        raise ValueError("JSON artifact path contains traversal components")
    data = canonical_bytes(dict(value)) + b"\n"
    _atomic_write(target, data, overwrite=overwrite)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("persisted JSON must be an object")
    return cast(dict[str, Any], value)


def _gzip_payload(records: Iterable[Mapping[str, Any]]) -> tuple[bytes, list[str], int]:
    rows = list(records)
    ids = [str(row.get("episode_id", "")) for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("shard records require unique non-empty episode IDs")
    payload = b"".join(canonical_bytes(dict(row)) + b"\n" for row in rows)
    if len(payload) > MAX_SHARD_BYTES:
        raise ValueError("shard exceeds the uncompressed artifact limit")
    compressed = bytearray()
    # gzip.GzipFile with filename="" and mtime=0 is deterministic across hosts.
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(payload)
    compressed.extend(buffer.getvalue())
    return bytes(compressed), ids, len(payload)


def write_shard(
    root: str | Path,
    name: str,
    records: Iterable[Mapping[str, Any]],
    *,
    expected_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    directory = Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    filename = safe_component(name, name="shard filename")
    target = safe_child(directory, filename)
    compressed, ids, uncompressed_bytes = _gzip_payload(records)
    if expected_ids is not None and ids != [str(item) for item in expected_ids]:
        raise ValueError("shard record roster differs from manifest")
    _atomic_write(target, compressed)
    entry = {
        "path": filename,
        "record_count": len(ids),
        "episode_ids": ids,
        "uncompressed_bytes": uncompressed_bytes,
        "compressed_bytes": target.stat().st_size,
        "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    read_shard(directory, entry, ids)
    return entry


def read_shard(root: str | Path, entry: Mapping[str, Any], expected_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    directory = Path(root).resolve()
    filename = safe_component(str(entry.get("path", "")), name="shard filename")
    target = safe_child(directory, filename)
    if not target.is_file():
        raise ValueError("shard artifact is missing")
    actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual_hash != str(entry.get("file_sha256", "")):
        raise ValueError("shard file hash mismatch")
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("shard row must be an object")
                    row = cast(dict[str, Any], value)
                    declared = row.get("canonical_hash", row.get("canonical_episode_sha256"))
                    if declared is not None and declared != canonical_record_hash(row):
                        raise ValueError("canonical record hash mismatch")
                    rows.append(row)
    except (OSError, EOFError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid gzip shard: {error}") from error
    ids = [str(row.get("episode_id", "")) for row in rows]
    declared_ids = [str(item) for item in (expected_ids if expected_ids is not None else entry.get("episode_ids", []))]
    if ids != declared_ids or len(rows) != int(entry.get("record_count", -1)):
        raise ValueError("shard roster mismatch")
    return rows


def write_state(root: str | Path, state: Mapping[str, Any], *, name: str = "state.json", overwrite: bool = True) -> str:
    target = safe_child(Path(root).resolve(), name)
    payload = dict(state)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    return write_json(target, payload, overwrite=overwrite)


def load_state(root: str | Path, *, name: str = "state.json") -> dict[str, Any] | None:
    target = safe_child(Path(root).resolve(), name)
    if not target.is_file():
        return None
    state = read_json(target)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("state schema mismatch")
    return state


def reduction_hash(records: Iterable[Mapping[str, Any]], *, timing_only: bool = False) -> str:
    rows = sorted(records, key=lambda item: str(item.get("episode_id", "")))
    values = [timing_stripped(row) if timing_only else row for row in rows]
    return sha256_value(values)


__all__ = [
    "MAX_SHARD_BYTES",
    "SCHEMA_VERSION",
    "TIMING_ONLY_FIELDS",
    "canonical_bytes",
    "canonical_record_hash",
    "load_state",
    "read_json",
    "read_shard",
    "reduction_hash",
    "safe_child",
    "safe_component",
    "sha256_value",
    "timing_stripped",
    "write_json",
    "write_shard",
    "write_state",
]
