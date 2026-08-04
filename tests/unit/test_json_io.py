from __future__ import annotations

from pathlib import Path

import pytest

from mutation_forge.experiment.json_io import read_json, write_json


def test_json_gzip_round_trip_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json.gz"
    value = {"schema_version": "test.v1", "items": [1, 2, 3]}

    write_json(path, value)
    first = path.read_bytes()
    write_json(path, value)

    assert first.startswith(b"\x1f\x8b")
    assert path.read_bytes() == first
    assert read_json(path) == value


def test_json_gzip_rejects_uncompressed_paths(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"

    with pytest.raises(ValueError, match=r"\.json\.gz"):
        write_json(path, {})
    with pytest.raises(ValueError, match=r"\.json\.gz"):
        read_json(path)
