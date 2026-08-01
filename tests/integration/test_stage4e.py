from __future__ import annotations

from pathlib import Path

from mutation_forge import cli
from mutation_forge.stage4e_config import load_manifest, load_stage4e_config


def test_stage4e_parser_exposes_freeze_and_confirm() -> None:
    parser = cli.build_parser()
    for command in ("freeze", "confirm"):
        parsed = parser.parse_args(["stage4e", command, "--json"])
        assert parsed.command == "stage4e"
        assert parsed.stage4e_command == command


def test_stage4e_parser_exposes_retained_recovery() -> None:
    parser = cli.build_parser()
    parsed = parser.parse_args(
        ["stage4e", "recover-retained", "--run", "/tmp/preserved-stage4e", "--json"]
    )
    assert parsed.command == "stage4e"
    assert parsed.stage4e_command == "recover-retained"


def test_stage4e_manifest_is_frozen_and_disjoint() -> None:
    config = load_stage4e_config(Path("configs/stage4e-confirmation.toml"))
    manifest = load_manifest(config)
    assert manifest["manifest_sha256"] == (
        "d80164cc4e0f26e2a2999adb7b1f8ff4b40a194e6f2576962190bd7b7bd22a34"
    )
    assert manifest["episode_count"] == 1536
    assert manifest["shard_count"] == 24
    assert manifest["episodes_per_shard"] == 64
