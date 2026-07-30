from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge.stage3.app_server import CodexAppServerAdapter
from mutation_forge.stage3.artifacts import GenerationArtifacts, TransportLogger, safe_value
from mutation_forge.stage3.isolation import IsolatedCapsule


def test_transport_logger_writes_theml_style_files_incrementally_and_redacts(
    tmp_path: Path,
) -> None:
    logger = TransportLogger(
        tmp_path, "slot-00.", max_bytes=4096, max_events=4, max_line_bytes=1024
    )
    logger.profile({"model": "gpt-5.6-luna", "authorization": "Bearer private-token-value"})
    logger.message({"id": 1, "result": {"access_token": "private-token-value"}}, b'{"id":1}\n')
    assert (tmp_path / "slot-00.codex-rpc.jsonl").is_file()
    logger.message({"method": "turn/started", "params": {"threadId": "thread-1"}}, b"{}\n")
    logger.text("stdout.jsonl", '{"ok":true}\n')
    logger.text("stderr.txt", "failure")
    for name in (
        "slot-00.codex-profile.json",
        "slot-00.codex-rpc.jsonl",
        "slot-00.events.jsonl",
        "slot-00.stdout.jsonl",
        "slot-00.stderr.txt",
    ):
        assert (tmp_path / name).is_file(), name
    retained = "".join(path.read_text() for path in tmp_path.iterdir())
    assert "private-token-value" not in retained
    assert "[REDACTED]" in retained


def test_transport_logger_enforces_event_and_payload_bounds(tmp_path: Path) -> None:
    logger = TransportLogger(tmp_path, max_bytes=128, max_events=1, max_line_bytes=16)
    logger.message({"method": "x"}, b"{}")
    with pytest.raises(ValueError, match="limit"):
        logger.message({"method": "x"}, b"{}")
    oversized = TransportLogger(tmp_path / "large", max_line_bytes=2)
    with pytest.raises(ValueError, match="limit"):
        oversized.message({"method": "x"}, b"123")


def test_transport_loggers_share_an_aggregate_run_byte_cap(tmp_path: Path) -> None:
    root = tmp_path / "run"
    first = TransportLogger(
        root / "slot-00", aggregate_root=root, max_aggregate_bytes=260, max_bytes=1024
    )
    second = TransportLogger(
        root / "slot-01", aggregate_root=root, max_aggregate_bytes=260, max_bytes=1024
    )
    first.text("request.md", "x" * 100)
    with pytest.raises(ValueError, match="aggregate"):
        second.text("request.md", "y" * 100)


def test_generation_artifacts_are_bounded_redacted_and_canonically_finished(tmp_path: Path) -> None:
    artifacts = GenerationArtifacts(tmp_path, "run-1", max_file_bytes=1024, max_total_bytes=4096)
    artifacts.start({"authorization": "Bearer private-token-value"})
    artifacts.write("nested/value.json", {"api_key": "private-token-value", "answer": 1})
    artifacts.finish("completed", {"answer": 1})
    summary = GenerationArtifacts.read_summary(artifacts.root)
    assert summary["status"] == "completed"
    assert "private-token-value" not in (artifacts.root / "generation_summary.json").read_text()
    assert json.loads((artifacts.root / "run_summary.json").read_text())["answer"] == 1
    with pytest.raises(ValueError, match="per-file"):
        artifacts.write_text("too-large.txt", "x" * 2048)


def test_rollout_is_copied_before_capsule_cleanup_and_never_copies_auth(tmp_path: Path) -> None:
    root = tmp_path / "mutation-forge-codex-fixture"
    home, sqlite, work = (root / "home", root / "sqlite", root / "work")
    for path in (home, sqlite, work):
        path.mkdir(parents=True, exist_ok=True)
    rollout = root / "server-rollout.jsonl"
    rollout.write_text('{"event":"done"}\n')
    capsule = IsolatedCapsule(
        root=root,
        codex_home=home,
        sqlite_home=sqlite,
        workdir=work,
        env={"PATH": "/usr/bin", "CODEX_HOME": str(home), "CODEX_SQLITE_HOME": str(sqlite)},
        codex_executable="/trusted/codex",
    )
    adapter = CodexAppServerAdapter(capsule=capsule, artifact_dir=tmp_path / "logs")
    adapter._thread = {"id": "thread-1", "path": str(rollout)}
    adapter._copy_rollout()
    assert (tmp_path / "logs" / "rollout.jsonl").read_text() == '{"event":"done"}\n'
    adapter._thread = {"id": "thread-1", "path": str(home / "auth.json")}
    (home / "auth.json").write_text('{"secret":"never-copy"}')
    adapter._copy_rollout()
    assert "never-copy" not in (tmp_path / "logs" / "rollout.jsonl").read_text()
    capsule.cleanup()
    assert not root.exists()


def test_safe_value_redacts_secrets_and_private_paths() -> None:
    value = safe_value(
        {"password": "secret", "path": "/home/alice/private", "nested": ["Bearer abc"]}
    )
    assert value == {
        "password": "[REDACTED]",
        "path": "[PRIVATE_PATH]/private",
        "nested": ["[REDACTED]"],
    }


def test_artifact_and_transport_paths_cannot_escape_root(tmp_path: Path) -> None:
    artifacts = GenerationArtifacts(tmp_path, "safe-run")
    with pytest.raises(ValueError, match="escapes"):
        artifacts.write("../outside.json", {"ok": False})
    with pytest.raises(ValueError, match="prefix"):
        TransportLogger(tmp_path / "logs", "../escape")
    logger = TransportLogger(tmp_path / "logs", max_bytes=128)
    with pytest.raises(ValueError, match="escapes"):
        logger.text("../outside.txt", "nope")


def test_safe_value_redacts_auth_tokens_and_jwts() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature"
    value = safe_value({"authToken": "private", "jwt": jwt, "message": f"authToken={jwt}"})
    assert value["authToken"] == "[REDACTED]"
    assert value["jwt"] == "[REDACTED]"
    assert "private" not in value["message"]
