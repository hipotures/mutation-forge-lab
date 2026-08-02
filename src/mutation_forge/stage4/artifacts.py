"""Deterministic, bounded persistence for the Stage 4 search campaign.

This module deliberately contains no search or model code.  It is a small
serialization layer: raw App Server records, compact evaluation evidence,
campaign manifests, and the provider-free technical-amendment proof.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

MAX_SHARD_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_CAMPAIGN_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_CAMPAIGN_COMPRESSED_BYTES = 128 * 1024 * 1024
PROJECTION_FRACTION = 0.50
MANIFEST_NAME = "EVIDENCE_MANIFEST.json"

_SECRET_KEY = re.compile(
    r"(?i)(?:access[_-]?token|refresh[_-]?token|api[_-]?key|authorization|password|secret|cookie|credential|jwt|private[_-]?key|client[_-]?secret)"
)
_SECRET_VALUE = re.compile(
    r"(?ix)(bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|(?:access[_-]?token|refresh[_-]?token|api[_-]?key|authorization|password|secret|credential|jwt|private[_-]?key|client[_-]?secret|token)\s*[:=]\s*[\"']?[^,\s\"']+)"
)
_PRIVATE_PATH = re.compile(r"(?:/home/[^/]+|/Users/[^/]+|[A-Za-z]:\\Users\\[^\\]+)")


def canonical_bytes(value: object) -> bytes:
    """Return stable UTF-8 JSON bytes (and reject NaN/Infinity)."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_json(value: object) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_value(value: object, key: str = "") -> object:
    """Recursively redact credentials and machine-private path prefixes."""

    if _SECRET_KEY.search(key) or key.lower().replace("_", "").replace("-", "") in {
        "token",
        "authtoken",
        "accesstoken",
        "refreshtoken",
        "apikey",
        "authorization",
        "jwt",
        "privatekey",
    }:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): safe_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v, key) for v in value]
    if isinstance(value, str):
        return _PRIVATE_PATH.sub("[PRIVATE_PATH]", _SECRET_VALUE.sub("[REDACTED]", value))
    return value


def _relative(root: Path, name: str | Path) -> Path:
    candidate = Path(name)
    if candidate.is_absolute():
        raise ValueError("artifact path must be relative")
    base = root.resolve()
    result = (base / candidate).resolve()
    try:
        result.relative_to(base)
    except ValueError as exc:
        raise ValueError("artifact path escapes root") from exc
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _gzip_json_lines(records: Sequence[Mapping[str, Any]]) -> tuple[bytes, bytes]:
    lines = b"".join(canonical_bytes(safe_value(record)) + b"\n" for record in records)
    if len(lines) > MAX_SHARD_UNCOMPRESSED_BYTES:
        raise ValueError("shard exceeds 32 MiB uncompressed bound")
    # mtime=0 makes the gzip stream reproducible across runs.
    return lines, gzip.compress(lines, compresslevel=9, mtime=0)


def _slot_path(generation: int, slot: int) -> str:
    if generation < 0 or slot < 0:
        raise ValueError("generation and slot must be non-negative")
    return f"raw/generation-{generation:04d}/slot-{slot:04d}.json.gz"


def write_raw_slot_record(
    root: Path,
    generation: int,
    slot: int,
    record: Mapping[str, Any],
) -> Path:
    """Persist one independently bounded raw App Server record.

    Only the six reconstructible transport fields are retained.  Unknown
    metadata is intentionally not copied into raw records; nested credentials
    are redacted as a final defence.
    """

    fields = ("source", "request", "response", "transcript", "usage", "reference")
    payload = {name: safe_value(record.get(name), name) for name in fields}
    payload["generation"] = generation
    payload["slot"] = slot
    uncompressed = canonical_bytes(payload)
    if len(uncompressed) > MAX_SHARD_UNCOMPRESSED_BYTES:
        raise ValueError("raw App Server record exceeds 32 MiB uncompressed bound")
    path = _relative(Path(root), _slot_path(generation, slot))
    _atomic_write(path, gzip.compress(uncompressed, compresslevel=9, mtime=0))
    return path


def read_raw_slot_record(path: Path) -> dict[str, Any]:
    data = gzip.decompress(path.read_bytes())
    if len(data) > MAX_SHARD_UNCOMPRESSED_BYTES:
        raise ValueError("raw record exceeds 32 MiB uncompressed bound")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("raw record must be an object")
    return cast(dict[str, Any], value)


def write_evaluation_shard(
    root: Path,
    kind: str,
    shard_id: int | str,
    records: Iterable[Mapping[str, Any]],
) -> Path:
    """Write a deterministic candidate/episode JSONL gzip shard.

    Candidate and episode records should refer to shared proposal/score/trace
    objects by ``*_ref``.  Full traces and score objects are rejected when
    embedded in a policy record, preventing accidental duplication.
    """

    if kind not in {"candidate", "episode", "validation", "shared"}:
        raise ValueError("unknown evidence shard kind")
    materialized = [dict(record) for record in records]
    for record in materialized:
        if any(key in record for key in ("trace", "score", "proposal_trace", "score_object")):
            raise ValueError("evidence records must reference shared traces/scores")
    materialized.sort(key=lambda value: str(value.get("record_id", value.get("episode_id", ""))))
    uncompressed, compressed = _gzip_json_lines(materialized)
    path = _relative(Path(root), f"evidence/{kind}-shard-{str(shard_id)}.jsonl.gz")
    _atomic_write(path, compressed)
    return path


def _reconstruction_for(path: Path) -> str:
    name = path.name
    if path.as_posix().startswith("raw/") or "/raw/" in path.as_posix():
        return (
            "gzip-decompress canonical JSON object; fields "
            "source/request/response/transcript/usage/reference"
        )
    if "candidate" in name:
        return "gzip-decompress JSONL; join proposal_ref and score_ref against shared shards"
    if "episode" in name:
        return "gzip-decompress JSONL; join candidate_id, proposal_ref, score_ref and policy"
    if "validation" in name:
        return "gzip-decompress canonical JSONL validation records"
    if "shared" in name:
        return "gzip-decompress JSONL keyed by object_id"
    return "read canonical JSON or JSONL bytes"


def _category(path: Path) -> str:
    p = path.as_posix()
    if p.startswith("raw/"):
        return "raw_app_server"
    if p.startswith("evidence/candidate-"):
        return "candidate_evidence"
    if p.startswith("evidence/episode-"):
        return "episode_evidence"
    if p.startswith("evidence/validation-"):
        return "final_validation"
    if p.startswith("evidence/shared-"):
        return "shared_evidence"
    return "artifact"


def _manifest_base(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME or relative.startswith("."):
            continue
        data = path.read_bytes()
        row: dict[str, Any] = {
            "path": relative,
            "size": len(data),
            "sha256": sha256_bytes(data),
            "category": _category(Path(relative)),
            "reconstruction": _reconstruction_for(Path(relative)),
        }
        if path.suffix == ".gz":
            try:
                row["uncompressed_size"] = len(gzip.decompress(data))
            except OSError as exc:
                raise ValueError(f"invalid gzip artifact: {relative}") from exc
        if int(row.get("uncompressed_size", row["size"])) > MAX_SHARD_UNCOMPRESSED_BYTES:
            raise ValueError(f"artifact exceeds 32 MiB uncompressed bound: {relative}")
        files.append(row)
    uncompressed = sum(int(row.get("uncompressed_size", row["size"])) for row in files)
    compressed = sum(int(row["size"]) for row in files)
    if uncompressed > MAX_CAMPAIGN_UNCOMPRESSED_BYTES:
        raise ValueError("campaign exceeds 512 MiB uncompressed bound")
    if compressed > MAX_CAMPAIGN_COMPRESSED_BYTES:
        raise ValueError("campaign exceeds 128 MiB compressed bound")
    return {
        "schema_version": "stage4.evidence_manifest.v1",
        "files": files,
        "campaign_uncompressed_size": uncompressed,
        "campaign_compressed_size": compressed,
    }


def build_evidence_manifest(root: Path) -> dict[str, Any]:
    """Create the deterministic manifest and write ``EVIDENCE_MANIFEST.json``."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = _manifest_base(root)
    manifest = {**base, "manifest_sha256": sha256_bytes(canonical_bytes(base))}
    _atomic_write(root / MANIFEST_NAME, canonical_bytes(manifest) + b"\n")
    return manifest


def verify_evidence_manifest(root: Path, manifest: Mapping[str, Any] | None = None) -> bool:
    """Verify every listed artifact and reject missing/extra/corrupt entries."""

    root = Path(root)
    raw: Mapping[str, Any] = manifest or cast(
        Mapping[str, Any], json.loads((root / MANIFEST_NAME).read_text())
    )
    if raw.get("schema_version") != "stage4.evidence_manifest.v1":
        raise ValueError("unexpected evidence manifest schema")
    entries = raw.get("files")
    if not isinstance(entries, list):
        raise ValueError("manifest files must be a list")
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError("invalid manifest entry")
        rel = str(entry["path"])
        _relative(root, rel)
        paths.append(rel)
        if not entry.get("reconstruction") or not entry.get("category"):
            raise ValueError("manifest entry lacks category/reconstruction")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate manifest entries")
    expected_hash = raw.get("manifest_sha256")
    base = {key: value for key, value in raw.items() if key != "manifest_sha256"}
    if expected_hash != sha256_bytes(canonical_bytes(base)):
        raise ValueError("manifest hash mismatch")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() != MANIFEST_NAME
        and not path.name.startswith(".")
    }
    listed = set(paths)
    if listed != actual:
        missing, extra = sorted(listed - actual), sorted(actual - listed)
        raise ValueError(f"manifest file set mismatch (missing={missing}, extra={extra})")
    total_u = total_c = 0
    for entry in entries:
        path = _relative(root, str(entry["path"]))
        data = path.read_bytes()
        if int(entry.get("size", -1)) != len(data) or entry.get("sha256") != sha256_bytes(data):
            raise ValueError(f"artifact digest/size mismatch: {entry['path']}")
        expected_category = _category(Path(str(entry["path"])))
        if entry.get("category") != expected_category:
            raise ValueError(f"artifact category mismatch: {entry['path']}")
        uncompressed = len(gzip.decompress(data)) if path.suffix == ".gz" else len(data)
        if "uncompressed_size" in entry and int(entry["uncompressed_size"]) != uncompressed:
            raise ValueError(f"artifact uncompressed size mismatch: {entry['path']}")
        if uncompressed > MAX_SHARD_UNCOMPRESSED_BYTES:
            raise ValueError(f"artifact exceeds 32 MiB uncompressed bound: {entry['path']}")
        total_c += len(data)
        total_u += uncompressed
    if int(raw.get("campaign_compressed_size", -1)) != total_c:
        raise ValueError("campaign compressed size mismatch")
    if int(raw.get("campaign_uncompressed_size", -1)) != total_u:
        raise ValueError("campaign uncompressed size mismatch")
    if total_u > MAX_CAMPAIGN_UNCOMPRESSED_BYTES:
        raise ValueError("campaign exceeds 512 MiB uncompressed bound")
    if total_c > MAX_CAMPAIGN_COMPRESSED_BYTES:
        raise ValueError("campaign exceeds 128 MiB compressed bound")
    return True


@dataclass(frozen=True, slots=True)
class SizeProjection:
    shard_uncompressed: int
    campaign_uncompressed: int
    campaign_compressed: int
    shard_headroom: float
    campaign_uncompressed_headroom: float
    campaign_compressed_headroom: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "shard_uncompressed": self.shard_uncompressed,
            "campaign_uncompressed": self.campaign_uncompressed,
            "campaign_compressed": self.campaign_compressed,
            "shard_headroom": self.shard_headroom,
            "campaign_uncompressed_headroom": self.campaign_uncompressed_headroom,
            "campaign_compressed_headroom": self.campaign_compressed_headroom,
        }


def validate_size_projection(
    shard_uncompressed: int, campaign_uncompressed: int, campaign_compressed: int
) -> SizeProjection:
    values = (shard_uncompressed, campaign_uncompressed, campaign_compressed)
    if any(value < 0 for value in values):
        raise ValueError("projected sizes must be non-negative")
    limits = (
        MAX_SHARD_UNCOMPRESSED_BYTES,
        MAX_CAMPAIGN_UNCOMPRESSED_BYTES,
        MAX_CAMPAIGN_COMPRESSED_BYTES,
    )
    if any(
        value > limit * PROJECTION_FRACTION for value, limit in zip(values, limits, strict=True)
    ):
        raise ValueError("dry-run projection exceeds 50% of a Stage 4 size limit")
    result = SizeProjection(
        shard_uncompressed,
        campaign_uncompressed,
        campaign_compressed,
        MAX_SHARD_UNCOMPRESSED_BYTES / max(1, shard_uncompressed),
        MAX_CAMPAIGN_UNCOMPRESSED_BYTES / max(1, campaign_uncompressed),
        MAX_CAMPAIGN_COMPRESSED_BYTES / max(1, campaign_compressed),
    )
    if (
        min(
            result.shard_headroom,
            result.campaign_uncompressed_headroom,
            result.campaign_compressed_headroom,
        )
        < 2
    ):
        raise ValueError("projection must report at least 2x headroom")
    return result


def project_real_shape(root: Path | None = None) -> dict[str, Any]:
    """Build a deterministic compact campaign with the full Stage 4 matrix."""

    target = (
        Path(root)
        if root is not None
        else Path(tempfile.mkdtemp(prefix="mforge-stage4-projection-"))
    )
    target.mkdir(parents=True, exist_ok=True)
    # Shared records are written once; policy records contain references only.
    shared = [
        {"object_id": f"proposal-{i:03d}", "operator": "synthetic", "parent": i % 8}
        for i in range(32)
    ]
    scores = [
        {"object_id": f"score-{i:03d}", "weighted_penalty": i % 7, "complete": True}
        for i in range(32)
    ]
    write_evaluation_shard(target, "shared", "proposals", shared)
    write_evaluation_shard(target, "shared", "scores", scores)
    seeds = [
        {
            "record_id": f"seed-{i:02d}",
            "generation": 0,
            "parent_ids": [],
            "proposal_ref": f"proposal-{i:03d}",
            "score_ref": f"score-{i:03d}",
        }
        for i in range(8)
    ]
    offspring = [
        {
            "record_id": f"offspring-{i:02d}",
            "generation": 1 + i // 8,
            "parent_ids": [f"seed-{i % 8:02d}"],
            "proposal_ref": f"proposal-{i:03d}",
            "score_ref": f"score-{i:03d}",
        }
        for i in range(32)
    ]
    write_evaluation_shard(target, "candidate", "00", seeds + offspring)
    # Every seed policy and every offspring policy is evaluated.  Search has
    # 40 policies (8 seeds + 32 offspring), each with 128 episodes in both
    # primary and replay passes.  Records contain references only; shared
    # proposal/score objects above are never duplicated here.
    search_policies = [f"seed-{i:02d}" for i in range(8)] + [
        f"offspring-{i:02d}" for i in range(32)
    ]
    episode_records: list[dict[str, Any]] = []
    for policy_id in search_policies:
        for mode in ("primary", "replay"):
            for episode in range(128):
                episode_records.append(
                    {
                        "record_id": f"{policy_id}-e{episode:03d}-{mode}",
                        "episode_id": episode,
                        "policy_id": policy_id,
                        "pass": mode,
                        "candidate_id": policy_id,
                        "proposal_ref": f"proposal-{episode % 32:03d}",
                        "score_ref": f"score-{episode % 32:03d}",
                    }
                )
    for mode in ("primary", "replay"):
        # Eight deterministic shards per pass (16 episodes per policy/shard).
        for shard in range(8):
            rows = [
                row
                for row in episode_records
                if row["pass"] == mode and int(row["episode_id"]) // 16 == shard
            ]
            write_evaluation_shard(target, "episode", f"{mode}-{shard:02d}", rows)
    validation_records = [
        {
            "record_id": f"validation-{policy}-{episode:03d}",
            "policy_id": f"policy-{policy}",
            "episode_id": episode,
            "status": "verified",
            "score_ref": f"score-{episode % 32:03d}",
        }
        for policy in range(4)
        for episode in range(128)
    ]
    for shard in range(8):
        rows = [row for row in validation_records if cast(int, row["episode_id"]) // 16 == shard]
        write_evaluation_shard(target, "validation", f"{shard:02d}", rows)
    for generation in range(4):
        for slot in range(8):
            write_raw_slot_record(
                target,
                generation,
                slot,
                {
                    "source": "synthetic",
                    "request": {"generation": generation, "slot": slot},
                    "response": {"ok": True},
                    "transcript": [],
                    "usage": {"input": 0, "output": 0},
                    "reference": f"offspring-{slot:02d}",
                },
            )
    manifest = build_evidence_manifest(target)
    projection = validate_size_projection(
        max(
            (int(entry.get("uncompressed_size", entry["size"])) for entry in manifest["files"]),
            default=0,
        ),
        int(manifest["campaign_uncompressed_size"]),
        int(manifest["campaign_compressed_size"]),
    )
    return {
        "counts": {
            "seeds": 8,
            "offspring": 32,
            "generations": 4,
            "episodes_per_policy_pass": 128,
            "policies": 40,
            "search_policies": 40,
            "search_episode_records": 10_240,
            "search_records": 10_240,
            "raw_records": 32,
            "final_validation": 4,
            "validation_policies": 4,
            "validation_episode_records": 512,
            "validation_records": 512,
        },
        "manifest": manifest,
        "projection": projection.as_dict(),
        "root": str(target),
    }


ALLOWED_AMENDMENT_CATEGORIES = frozenset(
    {
        "persistence",
        "layout",
        "sharding",
        "compression",
        "path",
        "resume",
        "telemetry",
        "report_reconstruction",
    }
)
_FORBIDDEN_AMENDMENT_KEYS = frozenset(
    {
        "scientific",
        "prompt",
        "prompts",
        "model",
        "brief",
        "briefs",
        "selection",
        "data",
        "threshold",
        "thresholds",
        "gate",
        "gates",
        "champion",
        "source",
        "policy",
        "metrics_changed",
    }
)
_IDENTITY_FIELDS = (
    "artifact_identity_before",
    "artifact_identity_after",
    "source_identity_before",
    "source_identity_after",
)
_INVARIANT_FIELDS = (
    "parent_assignments",
    "evaluation_semantics",
    "metrics",
    "decisions",
    "raw_outputs",
)


def validate_technical_amendment(amendment: Mapping[str, Any]) -> bool:
    category = amendment.get("category")
    if category not in ALLOWED_AMENDMENT_CATEGORIES:
        raise ValueError("technical amendment category is not allowed")
    if not amendment.get("regression_test_ref"):
        raise ValueError("technical amendment requires regression_test_ref")
    if any(key in amendment for key in _FORBIDDEN_AMENDMENT_KEYS):
        raise ValueError("scientific or source changes are forbidden in technical amendments")
    for field in _IDENTITY_FIELDS + _INVARIANT_FIELDS:
        if field not in amendment:
            raise ValueError(f"technical amendment missing {field}")
    if amendment["artifact_identity_before"] != amendment["artifact_identity_after"]:
        raise ValueError("artifact identity set changed")
    if amendment["source_identity_before"] != amendment["source_identity_after"]:
        raise ValueError("source identity set changed")
    if amendment.get("model_calls", 0) not in (0, False, None):
        raise ValueError("technical amendment must not make model calls")
    return True


__all__ = [
    "MAX_SHARD_UNCOMPRESSED_BYTES",
    "MAX_CAMPAIGN_UNCOMPRESSED_BYTES",
    "MAX_CAMPAIGN_COMPRESSED_BYTES",
    "canonical_bytes",
    "canonical_json",
    "sha256_bytes",
    "safe_value",
    "write_raw_slot_record",
    "read_raw_slot_record",
    "write_evaluation_shard",
    "build_evidence_manifest",
    "verify_evidence_manifest",
    "SizeProjection",
    "validate_size_projection",
    "project_real_shape",
    "ALLOWED_AMENDMENT_CATEGORIES",
    "validate_technical_amendment",
]
