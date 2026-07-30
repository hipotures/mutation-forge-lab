from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from mutation_forge.stage4.artifacts import (
    MAX_CAMPAIGN_COMPRESSED_BYTES,
    MAX_CAMPAIGN_UNCOMPRESSED_BYTES,
    MAX_SHARD_UNCOMPRESSED_BYTES,
    build_evidence_manifest,
    project_real_shape,
    validate_size_projection,
    validate_technical_amendment,
    verify_evidence_manifest,
    write_evaluation_shard,
    write_raw_slot_record,
)


def test_raw_records_are_deterministic_gzip_and_redacted(tmp_path: Path) -> None:
    first = write_raw_slot_record(
        tmp_path,
        0,
        1,
        {
            "source": "app-server",
            "request": {"authorization": "Bearer secret-value"},
            "response": {"ok": True},
            "transcript": [],
            "usage": {"input": 1},
            "reference": "r1",
        },
    )
    first_bytes = first.read_bytes()
    write_raw_slot_record(
        tmp_path,
        0,
        1,
        {
            "source": "app-server",
            "request": {"authorization": "Bearer secret-value"},
            "response": {"ok": True},
            "transcript": [],
            "usage": {"input": 1},
            "reference": "r1",
        },
    )
    assert first_bytes == first.read_bytes()
    decoded = gzip.decompress(first_bytes).decode()
    assert "secret-value" not in decoded


def test_evidence_shard_rejects_trace_duplication(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference"):
        write_evaluation_shard(tmp_path, "candidate", 0, [{"record_id": "a", "trace": {}}])


def test_size_projection_enforces_half_limits_and_headroom() -> None:
    result = validate_size_projection(1, 1, 1)
    assert result.campaign_compressed_headroom >= 2
    with pytest.raises(ValueError, match="50%"):
        validate_size_projection(MAX_SHARD_UNCOMPRESSED_BYTES // 2 + 1, 1, 1)
    with pytest.raises(ValueError, match="50%"):
        validate_size_projection(1, MAX_CAMPAIGN_UNCOMPRESSED_BYTES // 2 + 1, 1)
    with pytest.raises(ValueError, match="50%"):
        validate_size_projection(1, 1, MAX_CAMPAIGN_COMPRESSED_BYTES // 2 + 1)


def test_manifest_success_and_corruption_missing_extra_duplicate_and_traversal(
    tmp_path: Path,
) -> None:
    write_raw_slot_record(tmp_path, 0, 0, {"source": "x"})
    manifest = build_evidence_manifest(tmp_path)
    assert verify_evidence_manifest(tmp_path, manifest)
    raw = tmp_path / "EVIDENCE_MANIFEST.json"
    tampered = json.loads(raw.read_text())
    tampered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_evidence_manifest(tmp_path, tampered)
    (tmp_path / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="file set mismatch"):
        verify_evidence_manifest(tmp_path)
    (tmp_path / "extra.txt").unlink()
    duplicate = json.loads(raw.read_text())
    duplicate["files"].append(dict(duplicate["files"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        verify_evidence_manifest(tmp_path, duplicate)
    traversal = json.loads(raw.read_text())
    traversal["files"][0]["path"] = "../escape"
    with pytest.raises(ValueError, match="escapes"):
        verify_evidence_manifest(tmp_path, traversal)


def test_real_shape_projection_is_deterministic_and_reconstructible(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    first = project_real_shape(left)
    second = project_real_shape(right)
    assert first["counts"] == second["counts"]
    assert first["counts"]["seeds"] == 8
    assert first["counts"]["offspring"] == 32
    assert first["counts"]["search_policies"] == 40
    assert first["counts"]["search_episode_records"] == 10_240
    assert first["counts"]["validation_episode_records"] == 512
    assert first["counts"]["raw_records"] == 32
    search_rows = []
    for path in sorted((left / "evidence").glob("episode-*.jsonl.gz")):
        search_rows.extend(
            json.loads(line)
            for line in gzip.decompress(path.read_bytes()).decode().splitlines()
        )
    validation_rows = []
    for path in sorted((left / "evidence").glob("validation-*.jsonl.gz")):
        validation_rows.extend(
            json.loads(line)
            for line in gzip.decompress(path.read_bytes()).decode().splitlines()
        )
    assert len(search_rows) == 10_240
    assert len({row["policy_id"] for row in search_rows}) == 40
    assert {row["pass"] for row in search_rows} == {"primary", "replay"}
    assert len(validation_rows) == 512
    assert len({row["policy_id"] for row in validation_rows}) == 4
    assert first["projection"] == second["projection"]
    assert (left / "EVIDENCE_MANIFEST.json").read_bytes() == (
        right / "EVIDENCE_MANIFEST.json"
    ).read_bytes()
    assert verify_evidence_manifest(left)


def _amendment() -> dict[str, object]:
    return {
        "category": "compression",
        "regression_test_ref": (
            "tests/unit/test_stage4_artifacts.py::"
            "test_real_shape_projection_is_deterministic_and_reconstructible"
        ),
        "artifact_identity_before": ["a"],
        "artifact_identity_after": ["a"],
        "source_identity_before": ["s"],
        "source_identity_after": ["s"],
        "parent_assignments": {"x": "y"},
        "evaluation_semantics": {"equal_budget": True},
        "metrics": {"score": 1},
        "decisions": {"winner": "x"},
        "raw_outputs": {"x": "same"},
        "model_calls": 0,
    }


def test_technical_amendment_invariance_and_forbidden_changes() -> None:
    assert validate_technical_amendment(_amendment())
    changed = _amendment()
    changed["source_identity_after"] = ["changed"]
    with pytest.raises(ValueError, match="identity"):
        validate_technical_amendment(changed)
    forbidden = _amendment()
    forbidden["thresholds"] = {"x": 1}
    with pytest.raises(ValueError, match="forbidden"):
        validate_technical_amendment(forbidden)
