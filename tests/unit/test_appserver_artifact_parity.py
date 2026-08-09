from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_parity_module() -> ModuleType:
    path = Path("scripts/appserver_artifact_parity.py")
    spec = importlib.util.spec_from_file_location("appserver_artifact_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initial_turn(root: Path) -> Path:
    return (
        root
        / "initial-success"
        / "artifacts"
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "initial"
    )


def test_frozen_fixture_is_byte_identical_and_structurally_complete(
    tmp_path: Path,
) -> None:
    parity = _load_parity_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    parity.generate_fixture(first)
    parity.generate_fixture(second)

    first_snapshot = parity.snapshot_fixture(first)
    second_snapshot = parity.snapshot_fixture(second)

    assert first_snapshot == second_snapshot
    parity.compare_contract(parity.load_contract(), first_snapshot)
    real_shape = parity.verify_real_provider_workspace(first / "initial-success")
    assert real_shape["turn_count"] == 1


def _mutate_fixture(root: Path, mutation: str) -> None:
    turn = _initial_turn(root)
    request_path = turn / "slot-00.request.json.gz"
    if mutation == "removed":
        (turn / "slot-00.response.raw.txt").unlink()
    elif mutation == "renamed":
        request_path.rename(turn / "slot-00.request-envelope.json.gz")
    elif mutation == "added":
        (turn / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "recompressed":
        request_path.write_bytes(gzip.compress(gzip.decompress(request_path.read_bytes()), mtime=1))
    elif mutation == "schema-key":
        request = json.loads(gzip.decompress(request_path.read_bytes()).decode("utf-8"))
        assert isinstance(request, dict)
        request["unexpected_contract_key"] = True
        payload = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        request_path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    ["removed", "renamed", "added", "recompressed", "schema-key"],
)
def test_parity_rejects_tree_byte_compression_and_schema_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    parity = _load_parity_module()
    root = tmp_path / mutation
    parity.generate_fixture(root)
    _mutate_fixture(root, mutation)

    with pytest.raises(parity.ParityError, match="artifact parity failed"):
        parity.verify_fixture(root)


def test_contract_contains_all_required_provider_turn_cases() -> None:
    parity = _load_parity_module()
    contract: dict[str, Any] = parity.load_contract()
    cases = {
        str(path).split("/", 1)[0]
        for path in contract["raw_sha256"]
    }

    assert cases == {
        "initial-success",
        "repair-success",
        "retry-success",
        "terminal-failure",
    }
    profiles = set(contract["structural_profiles"])
    assert any("/repair-01" in profile for profile in profiles)
    assert any(profile.startswith("retry-success/") for profile in profiles)
