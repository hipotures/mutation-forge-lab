"""Timing-insensitive Stage 4 replay verification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutation_forge.stage3.manifest import sha256


def _project(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _project(item)
            for key, item in value.items()
            if key not in {"started_at", "finished_at", "elapsed_seconds", "timing", "path"}
            and not str(key).endswith("_ns")
        }
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    return value


def canonical_replay_hash(value: Any) -> str:
    """Canonical hash after recursively removing timing and artifact paths."""
    if isinstance(value, Mapping) and value.get("schema_version") == "stage4.evaluation.v1":
        # Pass names, cache keys, worker scheduling and file paths are run
        # metadata. Episode/reduction and metrics projections are the replay
        # contract and remain stable between primary and replay passes.
        stable = {
            key: value.get(key)
            for key in (
                "schema_version",
                "candidate_id",
                "manifest_sha256",
                "config_sha256",
                "record_count",
                "canonical_reduction_sha256",
                "metrics_input_sha256",
                "counts",
                "fitness_immutable",
            )
        }
        return sha256(_project(stable))
    return sha256(_project(value))


def _load(value: Mapping[str, Any] | str | Path) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.is_dir():
        candidates = sorted(path.glob("stage4-*-summary.json"))
        if len(candidates) != 1:
            raise ValueError("Stage 4 replay directory must contain exactly one summary")
        path = candidates[0]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("replay artifact must be an object")
    return raw


def verify_replay(
    primary: Mapping[str, Any] | str | Path, replay: Mapping[str, Any] | str | Path
) -> dict[str, Any]:
    """Compare primary/replay reduction, metric and worker-independent hashes."""
    try:
        left, right = _load(primary), _load(replay)
        left_hash, right_hash = canonical_replay_hash(left), canonical_replay_hash(right)
        reduction_match = left.get("canonical_reduction_sha256") == right.get(
            "canonical_reduction_sha256"
        )
        metrics_match = left.get("metrics_input_sha256") == right.get("metrics_input_sha256")
        exact = left_hash == right_hash and reduction_match and metrics_match
        return {
            "status": "completed" if exact else "failed",
            "decision": "exact" if exact else "mismatch",
            "exact": exact,
            "primary_sha256": left_hash,
            "replay_sha256": right_hash,
            "canonical_reduction_match": reduction_match,
            "metrics_input_match": metrics_match,
            "provider_calls": 0,
        }
    except Exception as error:
        return {
            "status": "failed",
            "decision": "mismatch",
            "exact": False,
            "provider_calls": 0,
            "error": f"{type(error).__name__}: {error}",
        }


def replay_exact(primary: Mapping[str, Any], replay: Mapping[str, Any]) -> bool:
    return bool(verify_replay(primary, replay).get("exact"))


__all__ = ["canonical_replay_hash", "replay_exact", "verify_replay"]
