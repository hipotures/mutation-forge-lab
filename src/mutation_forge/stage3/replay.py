"""Timing-insensitive replay verification for Stage 3 evaluation artifacts.

Replay is a verification pass, never an additional statistical sample.  This
module intentionally performs no graph evaluation or model/App Server calls;
it only loads canonical records and compares their deterministic projections.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evaluation import canonical_projection, read_records
from .manifest import canonical_bytes, sha256


def _project(value: Any, *, top_level: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _project(item)
            for key, item in value.items()
            if key not in {"started_at", "finished_at", "elapsed_seconds", "timing", "path"}
            and not str(key).endswith("_ns")
            and not (top_level and key in {"canonical_episode_sha256", "canonical_hash"})
        }
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    return value


def canonical_hash(value: Mapping[str, Any] | list[Any]) -> str:
    """Hash an ordered record projection with timing/path fields removed recursively."""
    if isinstance(value, list):
        projected: Any = [
            canonical_projection(item) if isinstance(item, Mapping) else _project(item)
            for item in value
        ]
    else:
        projected = _project(value, top_level=True)
    return hashlib.sha256(canonical_bytes(projected)).hexdigest()


def _load_shards(manifest_path: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    shards = manifest.get("shards")
    if (
        manifest.get("schema_version") != "stage3.evaluation_shards.v1"
        or not isinstance(shards, list)
        or len(shards) != 8
    ):
        raise ValueError("invalid evaluation shard manifest")
    records: list[dict[str, Any]] = []
    for item in shards:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("invalid evaluation shard entry")
        relative = Path(item["path"])
        if relative.is_absolute() or relative.name != str(relative):
            raise ValueError("evaluation shard path escapes manifest directory")
        path = manifest_path.parent / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("file_sha256"):
            raise ValueError("evaluation shard file hash mismatch")
        shard_records = read_records(path, max_records=128)
        if len(shard_records) != item.get("record_count"):
            raise ValueError("evaluation shard record count mismatch")
        records.extend(shard_records)
    ordered = sorted(records, key=lambda record: str(record["episode_id"]))
    if len(ordered) != manifest.get("record_count"):
        raise ValueError("evaluation aggregate record count mismatch")
    if sha256([canonical_projection(record) for record in ordered]) != manifest.get(
        "canonical_records_sha256"
    ):
        raise ValueError("evaluation aggregate record hash mismatch")
    return ordered


def _load(value: str | Path | Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.suffix == ".gz":
        return read_records(path)
    if path.is_dir():
        candidate = path / "evaluation-primary.jsonl.gz"
        if candidate.is_file():
            return read_records(candidate)
        candidate = path / "evaluation-primary-shards.json"
        if candidate.is_file():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("evaluation shard manifest must be an object")
            return _load_shards(candidate, raw)
        candidate = path / "evaluation_summary.json"
    else:
        candidate = path
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and raw.get("schema_version") == "stage3.evaluation_shards.v1":
        return _load_shards(candidate, raw)
    if not isinstance(raw, (dict, list)):
        raise ValueError("replay artifact must contain an object or record list")
    return raw


def verify_replay(
    primary: str | Path | Mapping[str, Any], replay: str | Path | Mapping[str, Any]
) -> dict[str, Any]:
    """Return a deterministic, side-effect-free replay comparison report."""
    try:
        left, right = _load(primary), _load(replay)
        left_hash, right_hash = canonical_hash(left), canonical_hash(right)
        exact = left_hash == right_hash
        # Summary artifacts carry their own decision and hash fields.  A replay
        # must not silently alter either, while record lists are compared by
        # their complete canonical projection.
        decision_match = True
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            decision_match = left.get("decision") == right.get("decision")
            exact = exact and decision_match
        return {
            "status": "completed" if exact else "failed",
            "decision": "exact" if exact else "mismatch",
            "primary_sha256": left_hash,
            "replay_sha256": right_hash,
            "decision_match": decision_match,
            "provider_calls": 0,
            "exact": exact,
        }
    except Exception as error:
        return {
            "status": "failed",
            "decision": "mismatch",
            "exact": False,
            "provider_calls": 0,
            "error": f"{type(error).__name__}: {error}",
        }


__all__ = ["canonical_hash", "verify_replay"]
