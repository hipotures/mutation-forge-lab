from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mutation_forge import cli
from mutation_forge.stage4 import commands
from mutation_forge.stage4.archive import ProgramArchive, ProgramRecord


def test_stage4_parser_exposes_every_frozen_command(tmp_path: Path) -> None:
    config = Path("configs/stage4-search.toml")
    run = tmp_path / "run"
    cases = (
        ["stage4", "doctor", "--config", str(config), "--json"],
        ["stage4", "freeze", "--config", str(config), "--json"],
        [
            "stage4",
            "evolve",
            "--config",
            str(config),
            "--concurrency",
            "8",
            "--json",
        ],
        ["stage4", "resume", str(run), "--json"],
        ["stage4", "archive", "inspect", str(run), "--json"],
        ["stage4", "archive", "reindex", str(run), "--json"],
        [
            "stage4",
            "evaluate-candidate",
            str(run),
            "program-1",
            "--pass",
            "primary",
            "--workers",
            "8",
            "--json",
        ],
        ["stage4", "freeze-validation", str(run), "--json"],
        ["stage4", "validate", str(run), "--workers", "8", "--json"],
        ["stage4", "verify-replay", str(run), "--json"],
    )
    parser = cli.build_parser()
    for arguments in cases:
        parsed = parser.parse_args(arguments)
        assert parsed.command == "stage4"


def test_stage4_json_cli_is_jsonl_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = {
        "schema_version": "stage4.doctor.v1",
        "status": "completed",
        "decision": "READY",
    }
    monkeypatch.setattr(commands, "doctor", lambda _: result)
    assert (
        cli.main(
            [
                "stage4",
                "doctor",
                "--config",
                "configs/stage4-search.toml",
                "--json",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == result


def test_stage4_json_failure_is_one_json_object(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_: Path) -> dict[str, object]:
        raise RuntimeError("freeze unavailable")

    monkeypatch.setattr(commands, "freeze", fail)
    assert (
        cli.main(
            [
                "stage4",
                "freeze",
                "--config",
                "configs/stage4-search.toml",
                "--json",
            ]
        )
        == 1
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    value = json.loads(lines[0])
    assert value["event_type"] == "run_failed"
    assert value["error_type"] == "RuntimeError"


def test_archive_commands_share_authoritative_files(tmp_path: Path) -> None:
    run = tmp_path / "campaign"
    archive = ProgramArchive(run / "archive")
    source = run / "archive" / "sources" / "stage3-slot-00.py"
    source.parent.mkdir(parents=True)
    source.write_text("seed\n", encoding="utf-8")
    archive.append(
        ProgramRecord(
            program_id="stage3-slot-00",
            source_path="archive/sources/stage3-slot-00.py",
            source_sha256=hashlib.sha256(b"seed\n").hexdigest(),
            normalized_ast_sha256="b" * 64,
            behavior_signature="c" * 64,
            generation=0,
            slot="slot-00",
            validation_status="valid",
            probe_status="passed",
            smoke_10k_status="passed",
            replay_status="verified",
            fitness_status="verified",
            seed_id="stage3-slot-00",
        )
    )
    inspection = commands.archive_inspect(run)
    reindex = commands.archive_reindex(run)
    assert inspection["counts"]["records"] == 1
    assert inspection["archive_hash"] == reindex["archive_hash"]
    assert reindex["status"] == "completed"


def test_doctor_runs_full_shape_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Path("configs/stage4-search.toml").resolve()
    monkeypatch.setattr(commands, "campaign_root", lambda _: tmp_path / "campaign")
    original_git_state = commands._git_state

    def clean_git_state(repo: Path) -> dict[str, object]:
        return {**original_git_state(repo), "dirty": False}

    monkeypatch.setattr(commands, "_git_state", clean_git_state)
    result = commands.doctor(config, check_auth=False, write=False)
    assert result["status"] == "completed"
    assert result["inference"] is False
    assert result["live_model_results_observed"] is False
    assert result["checks"]["manifest_matrix"] is True
    assert result["projection"]["counts"]["search_policies"] == 40
    assert result["projection"]["counts"]["search_records"] == 10_240
    assert result["projection"]["counts"]["validation_records"] == 512


def test_validation_and_generation_worker_counts_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        commands.evolve(
            "configs/stage4-search.toml",
            concurrency=7,
            provider=object(),
        )
    with pytest.raises(ValueError, match="workers"):
        commands.validate(tmp_path, workers=7)


def test_exact_usage_rejects_partial_or_incomplete_envelopes() -> None:
    complete = {
        "inputTokens": 1,
        "cachedInputTokens": 0,
        "outputTokens": 1,
        "reasoningOutputTokens": 0,
        "totalTokens": 2,
        "final": True,
        "partial": False,
    }
    assert commands._usage_complete(complete)
    assert not commands._usage_complete({**complete, "partial": True})
    assert not commands._usage_complete({"totalTokens": 2, "final": True})
