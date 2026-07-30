from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mutation_forge.cli import _emit_stage3, build_parser
from mutation_forge.stage3 import commands as stage3_commands
from mutation_forge.stage3.commands import verify_replay


def test_stage3_parser_surfaces_the_offline_revalidation_command() -> None:
    parser = build_parser()
    commands = (
        ["appserver-doctor", "--config", "config.toml", "--json"],
        ["freeze", "--config", "config.toml", "--json"],
        ["generate", "--config", "config.toml", "--concurrency", "8", "--json"],
        ["validate", "RUN", "--json"],
        ["revalidate", "--config", "config.toml", "--run", "RUN", "--json"],
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


def test_stage3_auth_json_is_explicit_for_doctor_and_generation() -> None:
    parser = build_parser()
    for command in ("appserver-doctor", "generate"):
        args = parser.parse_args(
            [
                "stage3",
                command,
                "--config",
                "config.toml",
                "--auth-json",
                "/private/auth.json",
            ]
        )
        assert args.auth_json == Path("/private/auth.json")


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


def _generation_config(run_root: Path) -> SimpleNamespace:
    return SimpleNamespace(run_root=run_root, stable_hash=lambda: "a" * 64)


def _write_generation_attempt(
    run_root: Path,
    *,
    accepted: int = 0,
    usage: dict[str, int] | None = None,
) -> Path:
    attempt = run_root / f"stage3-generation-{'a' * 12}-attempt-01"
    attempt.mkdir()
    summary = {
        "status": "infrastructure_failure",
        "provider_calls": 1,
        "initial_turn_count": 1,
        "accepted_model_turns": accepted,
        "usage_totals": usage if usage is not None else {"totalTokens": 0},
    }
    (attempt / "generation_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return attempt


@pytest.mark.parametrize(
    "evidence",
    ("content", "agent_message", "nonzero_usage", "accepted_turn"),
)
def test_generation_run_id_refuses_any_model_evidence(
    tmp_path: Path,
    evidence: str,
) -> None:
    usage = {"totalTokens": 1} if evidence == "nonzero_usage" else {"totalTokens": 0}
    accepted = 1 if evidence == "accepted_turn" else 0
    attempt = _write_generation_attempt(tmp_path, accepted=accepted, usage=usage)
    slot = attempt / "slots" / "slot-00"
    slot.mkdir(parents=True)
    if evidence == "content":
        (slot / "events.json").write_text(
            json.dumps([{"initial": {"content": True}}]),
            encoding="utf-8",
        )
    elif evidence == "agent_message":
        (slot / "slot-00.events.jsonl").write_text(
            json.dumps({"method": "item/agentMessage/delta"}) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="replacement attempts are forbidden"):
        stage3_commands._generation_run_id(_generation_config(tmp_path))


def test_generation_run_id_allows_only_one_reconciled_zero_evidence_retry(
    tmp_path: Path,
) -> None:
    first = _write_generation_attempt(tmp_path)
    slot = first / "slots" / "slot-00"
    slot.mkdir(parents=True)
    (slot / "events.json").write_text(
        json.dumps([{"initial": {"accepted": False, "content": False}}]),
        encoding="utf-8",
    )

    expected = f"stage3-generation-{'a' * 12}-attempt-02"
    assert stage3_commands._generation_run_id(_generation_config(tmp_path)) == expected

    second = tmp_path / expected
    second.mkdir()
    (second / "generation_summary.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "provider_calls": 1,
                "initial_turn_count": 1,
                "accepted_model_turns": 0,
                "usage_totals": {"totalTokens": 0},
            }
        ),
        encoding="utf-8",
    )
    second_slot = second / "slots" / "slot-00"
    second_slot.mkdir(parents=True)
    (second_slot / "events.json").write_text(
        json.dumps([{"initial": {"accepted": False, "content": False}}]),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="two bounded.*attempts are exhausted"):
        stage3_commands._generation_run_id(_generation_config(tmp_path))


def test_generation_run_id_rejects_nonterminal_zero_evidence(
    tmp_path: Path,
) -> None:
    attempt = _write_generation_attempt(tmp_path)
    summary_path = attempt / "generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "running"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(RuntimeError, match="replacement attempts are forbidden"):
        stage3_commands._generation_run_id(_generation_config(tmp_path))


def test_appserver_doctor_artifact_equals_returned_canonical_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        run_root=tmp_path,
        model=SimpleNamespace(name="gpt-5.6-luna", effort="high"),
        app_server=SimpleNamespace(
            sandbox_mode="danger-full-access",
            approval_policy="never",
        ),
        limits=SimpleNamespace(artifact_bytes=1024 * 1024),
    )

    class FakeCapsule:
        env = {"PATH": "/offline"}

        def cleanup(self) -> None:
            return None

    class FakeAdapter:
        def __init__(self, **_: object) -> None:
            pass

        def model_catalog(self) -> tuple[dict[str, object], ...]:
            return (
                {
                    "model": "gpt-5.6-luna",
                    "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
                },
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(stage3_commands, "load_stage3_config", lambda _: config)
    monkeypatch.setattr(stage3_commands.shutil, "which", lambda _: "/offline/codex")
    monkeypatch.setattr(
        stage3_commands.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="codex 0.test\n"),
    )
    monkeypatch.setattr(
        stage3_commands.IsolatedCapsule,
        "create",
        lambda *_, **__: FakeCapsule(),
    )
    monkeypatch.setattr(
        stage3_commands,
        "_auth_status",
        lambda **_: {"authenticated": True, "source": "offline-test"},
    )
    monkeypatch.setattr(stage3_commands, "CodexAppServerAdapter", FakeAdapter)
    monkeypatch.setattr(stage3_commands, "read_cpu_topology", lambda: tuple(range(16)))

    result = stage3_commands.appserver_doctor("offline.toml", auth_json="/offline/auth.json")

    artifact = Path(result["artifact"])
    assert json.loads(artifact.read_text(encoding="utf-8")) == result
    assert result["inference"] is False


def test_output_schema_preflight_requires_explicit_schema_version_type(
    tmp_path: Path,
) -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mutation-forge.invalid/test.json",
        "title": "test",
        "description": "test",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "source",
            "design_summary",
            "used_fields",
            "assumptions",
        ],
        "properties": {
            "schema_version": {"const": "stage3.generated_policy.v1"},
            "source": {"type": "string"},
            "design_summary": {"type": "string"},
            "used_fields": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
    }
    path = tmp_path / "output-schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    config = SimpleNamespace(output_schema_path=path)

    with pytest.raises(RuntimeError, match="frozen strict schema"):
        stage3_commands._validate_output_schema(config)

    schema["properties"]["schema_version"]["type"] = "string"
    path.write_text(json.dumps(schema), encoding="utf-8")
    assert stage3_commands._validate_output_schema(config) == schema

    schema["properties"]["used_fields"]["uniqueItems"] = True
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported schema keywords"):
        stage3_commands._validate_output_schema(config)


def _evaluation_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_repo=tmp_path / "project",
        heg_repo=tmp_path / "heg",
        frozen_heg_commit="h" * 40,
        manifest_path=tmp_path / "manifest.json",
    )


def test_evaluate_persists_atomic_inconclusive_on_evaluation_pass_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config = _evaluation_config(tmp_path)
    replacements: list[Path] = []
    real_replace = stage3_commands.os.replace

    monkeypatch.setattr(stage3_commands, "load_stage3_config", lambda _: config)
    monkeypatch.setattr(stage3_commands, "_load_freeze", lambda _: {})
    monkeypatch.setattr(
        stage3_commands,
        "_git_state",
        lambda repo: {
            "commit": config.frozen_heg_commit if repo == config.heg_repo else "p" * 40,
            "dirty": False,
        },
    )
    monkeypatch.setattr(
        stage3_commands,
        "replay_generation",
        lambda _: {"status": "completed", "replay_validated": True},
    )
    monkeypatch.setattr(stage3_commands, "_sources_for_run", lambda *_: {"random": "source"})
    monkeypatch.setattr(
        stage3_commands,
        "load_manifest",
        lambda *_: {"episodes": [{"episode_id": "episode-00"}]},
    )
    monkeypatch.setattr(
        stage3_commands,
        "_run_evaluation_pass",
        Mock(side_effect=RuntimeError("offline worker failure")),
    )

    def record_replace(source: Path, destination: Path) -> None:
        replacements.append(Path(destination))
        real_replace(source, destination)

    monkeypatch.setattr(stage3_commands.os, "replace", record_replace)

    result = stage3_commands.evaluate("offline.toml", run)

    assert result["decision"] == "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"
    assert result["provider_calls"] == 0
    assert json.loads((run / "evaluation_summary.json").read_text(encoding="utf-8")) == result
    assert json.loads((run / "gate.json").read_text(encoding="utf-8"))["decision"] == result[
        "decision"
    ]
    assert replacements == [run / "evaluation_summary.json", run / "gate.json"]
    assert not (run / ".evaluation_summary.json.tmp").exists()
    assert not (run / ".gate.json.tmp").exists()


@pytest.mark.parametrize("failure", ("freeze_drift", "dirty_repository"))
def test_evaluate_rejects_provenance_failures_before_cpu_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config = _evaluation_config(tmp_path)
    cpu_work = Mock(side_effect=AssertionError("CPU work must not start"))
    replay = Mock(side_effect=AssertionError("generation replay must not start"))

    monkeypatch.setattr(stage3_commands, "load_stage3_config", lambda _: config)
    if failure == "freeze_drift":
        monkeypatch.setattr(
            stage3_commands,
            "_load_freeze",
            Mock(side_effect=RuntimeError("freeze artifact hash mismatch")),
        )
    else:
        monkeypatch.setattr(stage3_commands, "_load_freeze", lambda _: {})
        monkeypatch.setattr(
            stage3_commands,
            "_git_state",
            lambda _: {"commit": "p" * 40, "dirty": True},
        )
    monkeypatch.setattr(stage3_commands, "replay_generation", replay)
    monkeypatch.setattr(stage3_commands, "_run_evaluation_pass", cpu_work)

    result = stage3_commands.evaluate("offline.toml", run)

    assert result["decision"] == "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"
    assert result["provider_calls"] == 0
    assert result["failure"]["code"] == "RuntimeError"
    cpu_work.assert_not_called()
    replay.assert_not_called()
