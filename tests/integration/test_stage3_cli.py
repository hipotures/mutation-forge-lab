from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge.cli import _emit_stage3, build_parser
from mutation_forge.stage3.commands import verify_replay


def test_stage3_parser_surfaces_the_six_frozen_commands() -> None:
    parser = build_parser()
    commands = (
        ["appserver-doctor", "--config", "config.toml", "--json"],
        ["freeze", "--config", "config.toml", "--json"],
        ["generate", "--config", "config.toml", "--concurrency", "8", "--json"],
        ["validate", "RUN", "--json"],
        ["evaluate", "--config", "config.toml", "--run", "RUN", "--workers", "8", "--json"],
        ["verify-replay", "PRIMARY", "REPLAY", "--json"],
    )
    for command in commands:
        args = parser.parse_args(["stage3", *command])
        assert args.command == "stage3"
        assert args.json is True


def test_verify_replay_is_model_free_and_compares_declared_content(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    replay = tmp_path / "replay.json"
    payload = {"schema_version": "stage3.generation.v1", "status": "completed", "slots": []}
    primary.write_text(json.dumps(payload), encoding="utf-8")
    replay.write_text(json.dumps({**payload, "finished_at": "different"}), encoding="utf-8")
    result = verify_replay(primary, replay)
    assert result["exact"] is True
    assert result["provider_calls"] == 0


def test_generation_and_evaluation_worker_values_are_strict() -> None:
    from mutation_forge.stage3 import commands

    with pytest.raises(ValueError, match="concurrency"):
        commands.generate("unused.toml", concurrency=4)
    with pytest.raises(ValueError, match="workers"):
        commands.evaluate("unused.toml", "unused-run", workers=4)


def test_stage3_rich_and_json_render_the_same_canonical_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = {
        "schema_version": "stage3.command.v1",
        "status": "completed",
        "slots": [{"slot": "slot-00", "status": "accepted"}],
        "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
    }
    _emit_stage3(result, json_output=True)
    json_value = json.loads(capsys.readouterr().out)
    _emit_stage3(result, json_output=False)
    rich_value = json.loads(capsys.readouterr().out)
    assert json_value == rich_value == result
