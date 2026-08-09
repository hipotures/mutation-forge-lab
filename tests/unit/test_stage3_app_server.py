"""Offline protocol tests for the Stage 3 Codex app-server adapter.

The fixture is an in-memory JSONL peer: these tests do not start Codex and
cannot make a model or network request.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.stage3.app_server import (
    AppServerError,
    AppServerGenerationProvider,
    AppServerLimits,
    CodexAppServerAdapter,
    IsolationError,
    ModelProfile,
    ProtocolError,
    TurnError,
)
from mutation_forge.stage3.config import load_stage3_config
from mutation_forge.stage3.contracts import parse_generated_policy
from mutation_forge.stage3.isolation import THIN_APP_SERVER_ARGS, IsolatedCapsule
from mutation_forge.stage3.prompts import load_prompt_bundle

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location("stage3_fake_app_server", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario
FORK_PROFILE = ModelProfile("codex", "gpt-5.6-luna", "high")


def _capsule(tmp_path: Path) -> IsolatedCapsule:
    root = tmp_path / "capsule"
    home, sqlite, work = (root / "home", root / "sqlite", root / "work")
    for path in (home, sqlite, work):
        path.mkdir(parents=True, exist_ok=True)
    return IsolatedCapsule(
        root=root,
        codex_home=home,
        sqlite_home=sqlite,
        workdir=work,
        env={"PATH": "/usr/bin", "CODEX_HOME": str(home), "CODEX_SQLITE_HOME": str(sqlite)},
        codex_executable="/trusted/codex",
    )


def _adapter(
    tmp_path: Path,
    scenario: Any | None = None,
    *,
    limits: AppServerLimits | None = None,
    artifacts: bool = False,
    sandbox_mode: str = "danger-full-access",
) -> tuple[CodexAppServerAdapter, list[tuple[tuple[Any, ...], dict[str, Any]]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def factory(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return FakeProcess(scenario, **kwargs)

    adapter = CodexAppServerAdapter(
        capsule=_capsule(tmp_path),
        process_factory=factory,
        auth_checker=lambda _: True,
        limits=limits,
        artifact_dir=tmp_path / "logs" if artifacts else None,
        artifact_prefix="slot-00.",
        sandbox_mode=sandbox_mode,
    )
    return adapter, calls


def _fixed_process_factory(process: Any) -> Any:
    def factory(*_: Any, **kwargs: Any) -> Any:
        process.environment = kwargs.get("env", {})
        return process

    return factory


def test_switching_back_to_completed_parent_allows_exact_child_fork(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(
        tmp_path,
        FakeScenario(),
        limits=AppServerLimits(max_turns=8, max_campaigns=4),
    )
    try:
        anchor = adapter.generate_persistent("anchor", FORK_PROFILE)
        parent_branch = adapter.fork_persistent_thread(
            FORK_PROFILE,
            last_turn_id=anchor.turn_id,
            activate=True,
        )
        parent = adapter.generate_persistent("parent", FORK_PROFILE)
        sibling_branch = adapter.fork_persistent_thread(
            FORK_PROFILE,
            last_turn_id=anchor.turn_id,
            activate=True,
        )
        adapter.generate_persistent("sibling", FORK_PROFILE)
        adapter.activate_forked_thread(
            parent_branch.child_thread_id,
            completed_turn_ids=(anchor.turn_id, parent.turn_id),
        )
        child = adapter.fork_persistent_thread(
            FORK_PROFILE,
            last_turn_id=parent.turn_id,
        )
    finally:
        adapter.close()

    assert child.source_thread_id == parent_branch.child_thread_id
    assert child.included_turn_ids == (anchor.turn_id, parent.turn_id)
    assert sibling_branch.child_thread_id != child.source_thread_id


def test_switching_back_to_original_anchor_allows_another_root_fork(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(
        tmp_path,
        FakeScenario(),
        limits=AppServerLimits(max_turns=8, max_campaigns=3),
    )
    try:
        anchor = adapter.generate_persistent("anchor", FORK_PROFILE)
        first_root = adapter.fork_persistent_thread(
            FORK_PROFILE,
            last_turn_id=anchor.turn_id,
            activate=True,
        )
        adapter.activate_forked_thread(
            anchor.thread_id,
            completed_turn_ids=(anchor.turn_id,),
        )
        second_root = adapter.fork_persistent_thread(
            FORK_PROFILE,
            last_turn_id=anchor.turn_id,
        )
    finally:
        adapter.close()

    assert first_root.source_thread_id == anchor.thread_id
    assert second_root.source_thread_id == anchor.thread_id
    assert second_root.included_turn_ids == (anchor.turn_id,)


def test_switching_parent_rejects_changed_completed_history(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(tmp_path, FakeScenario())
    try:
        anchor = adapter.generate_persistent("anchor", FORK_PROFILE)
        branch = adapter.fork_persistent_thread(
            FORK_PROFILE,
            last_turn_id=anchor.turn_id,
            activate=True,
        )
        parent = adapter.generate_persistent("parent", FORK_PROFILE)
        with pytest.raises(ProtocolError, match="history changed"):
            adapter.activate_forked_thread(
                branch.child_thread_id,
                completed_turn_ids=(anchor.turn_id, "different-turn"),
            )
    finally:
        adapter.close()

    assert parent.turn_id != "different-turn"


@pytest.mark.parametrize("sandbox_mode", ["read-only", "danger-full-access"])
def test_thread_start_uses_and_verifies_configured_sandbox_mode(
    tmp_path: Path, sandbox_mode: str
) -> None:
    adapter, _ = _adapter(tmp_path, sandbox_mode=sandbox_mode)
    with adapter:
        thread = adapter.start_thread("codex/gpt-5.6-luna:high")
    assert thread["id"] == "thread-1"


def test_enabled_skills_are_disabled_before_thread_start(tmp_path: Path) -> None:
    scenario = FakeScenario(enabled_skills=["/system/skill/SKILL.md"])
    adapter, _ = _adapter(tmp_path, scenario)
    with adapter:
        adapter.start_thread("codex/gpt-5.6-luna:high")
    assert scenario.enabled_skills == []


def test_strict_argv_private_cwd_and_lifecycle_ids(tmp_path: Path) -> None:
    adapter, calls = _adapter(tmp_path)
    with adapter:
        result = adapter.generate(
            "return one policy", ModelProfile("codex", "gpt-5.6-luna", "high")
        )

    assert calls[0][0][0] == ["/trusted/codex", *THIN_APP_SERVER_ARGS[1:]]
    assert calls[0][1]["cwd"] == str(adapter.capsule.workdir)
    assert calls[0][1]["env"] == dict(adapter.capsule.env)
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["preexec_fn"].keywords["processes"] == 102_400
    assert result.text == "fixture answer"
    assert (result.thread_id, result.session_id, result.turn_id) == (
        "thread-1",
        "session-1",
        "turn-1",
    )
    assert result.usage.input_tokens == 2
    assert result.usage.cached_input_tokens == 0
    assert result.usage.cache_write_input_tokens == 0
    assert result.usage.output_tokens == 3
    assert result.usage.reasoning_output_tokens == 1
    assert result.usage.total_tokens == 5
    assert result.usage.raw["totalTokens"] == 5
    assert result.usage.final is True
    assert result.usage.partial is False
    assert result.request_id >= 0


def test_not_loaded_completed_items_accepts_already_persisted_final(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(
        tmp_path,
        FakeScenario(completed_items_view="notLoaded"),
        artifacts=True,
    )
    with adapter:
        result = adapter.generate("prompt", "codex/gpt-5.6-luna:high")
    assert result.text == "fixture answer"
    assert (tmp_path / "logs" / "slot-00.response.md").read_text() == "fixture answer"


def test_json_transport_text_is_retained_without_markdown_wrapper(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        tmp_path,
        FakeScenario(final_text='{"z":1,"a":{"b":2}}'),
        artifacts=True,
    )
    with adapter:
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")
    response_markdown = (tmp_path / "logs" / "slot-00.response.md").read_text(encoding="utf-8")
    assert response_markdown == '{"z":1,"a":{"b":2}}'


@pytest.mark.parametrize("shape", ["top-level", "nested"])
def test_thread_started_accepts_both_schema_shapes_in_legal_window(
    tmp_path: Path, shape: str
) -> None:
    adapter, _ = _adapter(tmp_path, FakeScenario(thread_started_notification=shape))
    with adapter:
        assert adapter.generate("prompt", "codex/gpt-5.6-luna:high").text == "fixture answer"


def test_thread_started_before_response_is_rejected(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        tmp_path, FakeScenario(thread_started_notification="nested-before-thread-response")
    )
    with adapter, pytest.raises(ProtocolError, match="before thread/start response"):
        adapter.start_thread("codex/gpt-5.6-luna:high")


def test_thread_started_after_turn_response_is_rejected(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        tmp_path, FakeScenario(thread_started_notification="nested-after-turn-response")
    )
    with adapter, pytest.raises(ProtocolError, match="after turn/start response"):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")


def test_thread_started_mismatched_normalized_id_is_rejected(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    with adapter:
        adapter.start_thread("codex/gpt-5.6-luna:high")
        process = adapter._process
        assert process is not None
        process.stdout.put(
            {"method": "thread/started", "params": {"thread": {"id": "foreign-thread"}}}
        )
        with pytest.raises(ProtocolError, match="foreign thread/started"):
            adapter.generate("prompt", "codex/gpt-5.6-luna:high")


def test_matching_id_server_request_is_rejected_before_response_handling(tmp_path: Path) -> None:
    process = FakeProcess(FakeScenario())
    original_receive = process.receive

    def receive(line: bytes) -> None:
        value = json.loads(line)
        if value.get("method") == "turn/start":
            process.stdout.put({"id": value["id"], "method": "item/toolCall", "params": {}})
        original_receive(line)

    process.receive = receive  # type: ignore[method-assign]
    adapter = CodexAppServerAdapter(
        capsule=_capsule(tmp_path),
        process_factory=_fixed_process_factory(process),
        auth_checker=lambda _: True,
    )
    with adapter, pytest.raises(ProtocolError, match="unsupported server request"):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")


def test_malformed_thread_status_is_rejected_and_marks_usage_partial(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    with adapter:
        adapter.start_thread("codex/gpt-5.6-luna:high")
        process = adapter._process
        assert process is not None
        process.stdout.put(
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "tokenUsage": {"last": FakeScenario().usage},
                },
            }
        )
        process.stdout.put(
            {
                "method": "thread/status/changed",
                "params": {"threadId": "thread-1", "status": {}},
            }
        )
        with pytest.raises(ProtocolError, match="malformed thread status"):
            adapter.generate("prompt", "codex/gpt-5.6-luna:high")
    assert adapter.inspect_metadata()["usage_final"] is False
    assert adapter.inspect_metadata()["usage_partial"] is True


def test_private_ephemeral_thread_is_the_default(
    tmp_path: Path,
) -> None:
    process = FakeProcess(FakeScenario())
    seen: list[dict[str, Any]] = []
    original_receive = process.receive

    def receive(line: bytes) -> None:
        seen.append(json.loads(line))
        original_receive(line)

    process.receive = receive  # type: ignore[method-assign]
    adapter = CodexAppServerAdapter(
        capsule=_capsule(tmp_path),
        process_factory=_fixed_process_factory(process),
        auth_checker=lambda _: True,
    )
    with adapter:
        thread = adapter.start_thread("codex/gpt-5.6-luna:high")
    start = next(value for value in seen if value.get("method") == "thread/start")
    assert start["params"]["ephemeral"] is True
    assert thread["ephemeral"] is True
    assert thread["cwd"] == str(adapter.capsule.workdir)
    assert Path(thread["path"]).is_relative_to(adapter.capsule.root)


def test_thread_cwd_and_rollout_escape_are_rejected(tmp_path: Path) -> None:
    process = FakeProcess(FakeScenario())

    def escaped_factory(*_: Any, **kwargs: Any) -> Any:
        process.environment = {**kwargs.get("env", {}), "CODEX_HOME": "/outside-capsule"}
        return process

    adapter = CodexAppServerAdapter(
        capsule=_capsule(tmp_path / "escape"),
        process_factory=escaped_factory,
        auth_checker=lambda _: True,
    )
    with adapter, pytest.raises(IsolationError, match="escapes the capsule"):
        adapter.start_thread("codex/gpt-5.6-luna:high", ephemeral=False)

    process = FakeProcess(FakeScenario())
    original_receive = process.receive

    def wrong_cwd(line: bytes) -> None:
        value = json.loads(line)
        if value.get("method") == "thread/start":
            value["params"]["cwd"] = "/wrong-cwd"
            line = json.dumps(value).encode()
        original_receive(line)

    process.receive = wrong_cwd  # type: ignore[method-assign]
    adapter = CodexAppServerAdapter(
        capsule=_capsule(tmp_path / "cwd"),
        process_factory=_fixed_process_factory(process),
        auth_checker=lambda _: True,
    )
    with adapter, pytest.raises(IsolationError, match="identity violates|invalid cwd"):
        adapter.start_thread("codex/gpt-5.6-luna:high")


@pytest.mark.parametrize(
    ("scenario", "error"),
    [
        (FakeScenario(unknown_notification=True), ProtocolError),
        (FakeScenario(server_request=True), ProtocolError),
        (FakeScenario(model_rerouted=True), IsolationError),
        (FakeScenario(terminal_status="failed"), TurnError),
        (FakeScenario(delta_item_id="foreign-item"), ProtocolError),
        (FakeScenario(completed_item_id="foreign-item"), ProtocolError),
        (FakeScenario(late_item=True), ProtocolError),
        (FakeScenario(malformed=True), ProtocolError),
    ],
)
def test_protocol_abuse_fails_only_the_adapter(
    tmp_path: Path, scenario: Any, error: type[Exception]
) -> None:
    adapter, _ = _adapter(tmp_path, scenario)
    with adapter, pytest.raises(error):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")
    assert adapter.inspect_metadata()["status"] == "failed"
    with pytest.raises(AppServerError, match="failed adapter cannot be reused"):
        adapter.start()


def test_legal_warning_with_completed_turn_preserves_success_requirements(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(
        tmp_path,
        FakeScenario(warning_message="fixture legal warning"),
        artifacts=True,
    )
    with adapter:
        result = adapter.generate("prompt", "codex/gpt-5.6-luna:high")

    metadata = adapter.inspect_metadata()
    assert result.text == "fixture answer"
    assert result.usage.final is True
    assert metadata["status"] == "completed"
    assert metadata["serverWarnings"] == 1
    assert metadata["usage_final"] is True
    stdout = (tmp_path / "logs" / "slot-00.stdout.jsonl").read_text(encoding="utf-8")
    assert '"method":"warning"' in stdout
    assert "fixture legal warning" in stdout


def test_legal_warning_with_interrupted_turn_fails_closed(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        tmp_path,
        FakeScenario(
            warning_message="fixture legal warning",
            terminal_status="interrupted",
        ),
    )
    with adapter, pytest.raises(TurnError, match="interrupted"):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")

    metadata = adapter.inspect_metadata()
    assert metadata["status"] == "failed"
    assert metadata["serverWarnings"] == 1
    assert metadata["usage_final"] is False


def test_legal_warning_with_system_error_fails_closed(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        tmp_path,
        FakeScenario(
            warning_message="fixture legal warning",
            thread_status_after_warning="systemError",
        ),
    )
    with adapter, pytest.raises(TurnError, match="systemError"):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")

    metadata = adapter.inspect_metadata()
    assert metadata["status"] == "failed"
    assert metadata["serverWarnings"] == 1
    assert metadata["usage_final"] is False


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        (
            FakeScenario(warning_message="fixture legal warning", final_text=None),
            "no final_answer",
        ),
        (
            FakeScenario(warning_message="fixture legal warning", usage=None),
            "exact tokenUsage",
        ),
    ],
)
def test_legal_warning_does_not_relax_final_response_or_usage(
    tmp_path: Path,
    scenario: Any,
    message: str,
) -> None:
    adapter, _ = _adapter(tmp_path, scenario)
    with adapter, pytest.raises(TurnError, match=message):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")

    metadata = adapter.inspect_metadata()
    assert metadata["status"] == "failed"
    assert metadata["serverWarnings"] == 1
    assert metadata["usage_final"] is False


def test_system_error_and_oversized_message_are_rejected(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    with adapter:
        adapter.start_thread("codex/gpt-5.6-luna:high")
        process = adapter._process
        assert process is not None
        process.stdout.put(
            {
                "method": "thread/status/changed",
                "params": {"threadId": "thread-1", "status": {"type": "systemError"}},
            }
        )
        with pytest.raises(TurnError, match="systemError"):
            adapter.generate("prompt", "codex/gpt-5.6-luna:high")

    limits = AppServerLimits(message_bytes=1024)
    adapter, _ = _adapter(tmp_path / "oversized", FakeScenario(oversized=True), limits=limits)
    with adapter, pytest.raises(ProtocolError, match="incoming message exceeds limit"):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")


def test_many_small_frames_may_exceed_single_frame_limit_in_aggregate(tmp_path: Path) -> None:
    process = FakeProcess()
    original_receive = process.receive

    def receive(line: bytes) -> None:
        payload = json.loads(line)
        if payload.get("method") == "initialize":
            for _ in range(20):
                process.stdout.put(
                    {
                        "method": "account/rateLimits/updated",
                        "params": {"rateLimits": {}},
                    }
                )
        original_receive(line)

    process.receive = receive  # type: ignore[method-assign]
    limits = AppServerLimits(
        message_bytes=1024,
        stdout_bytes=32_768,
        transcript_bytes=65_536,
    )
    adapter = CodexAppServerAdapter(
        capsule=_capsule(tmp_path),
        process_factory=_fixed_process_factory(process),
        auth_checker=lambda _: True,
        limits=limits,
    )
    with adapter:
        result = adapter.generate("prompt", "codex/gpt-5.6-luna:high")
        assert result.text == "fixture answer"
        assert adapter._stdout_size > limits.message_limit
        assert adapter._stdout_size < limits.stdout_limit


def test_timeout_interrupts_and_failed_adapter_is_not_reused(tmp_path: Path) -> None:
    scenario = FakeScenario()
    process = FakeProcess(scenario)
    original_receive = process.receive

    def receive(line: bytes) -> None:
        payload = json.loads(line)
        if payload.get("method") == "turn/start":
            return
        original_receive(line)

    process.receive = receive  # type: ignore[method-assign]
    calls: list[Any] = []

    def factory(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        process.environment = kwargs.get("env", {})
        return process

    adapter = CodexAppServerAdapter(
        capsule=_capsule(tmp_path),
        process_factory=factory,
        auth_checker=lambda _: True,
        limits=AppServerLimits(turn_timeout=0.02, usage_grace=0.01),
    )
    with adapter, pytest.raises(TurnError, match="turn timed out"):
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")
    assert process.returncode in {-15, -9}
    with pytest.raises(AppServerError, match="failed adapter cannot be reused"):
        adapter.start()


def test_unterminated_stdout_and_stderr_frames_are_bounded(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        tmp_path,
        limits=AppServerLimits(message_bytes=1024, stderr_bytes=64, turn_timeout=0.1),
    )
    with adapter:
        adapter.start_thread("codex/gpt-5.6-luna:high")
        process = adapter._process
        assert process is not None
        process.stdout.put(b"x" * 1025)
        with pytest.raises(ProtocolError, match="output limit|incoming message"):
            adapter.generate("prompt", "codex/gpt-5.6-luna:high")

    adapter, _ = _adapter(
        tmp_path / "stderr",
        limits=AppServerLimits(stderr_bytes=64, turn_timeout=0.1),
    )
    with adapter:
        adapter.start_thread("codex/gpt-5.6-luna:high")
        process = adapter._process
        assert process is not None
        process.stderr.put(b"x" * 65)
        deadline = time.monotonic() + 1.0
        while not adapter._stderr_exceeded and time.monotonic() < deadline:
            time.sleep(0.001)
        assert adapter._stderr_exceeded is True
        with pytest.raises(ProtocolError, match="output limit"):
            adapter.generate("prompt", "codex/gpt-5.6-luna:high")


def test_bidirectional_transcript_limit_is_enforced_for_send_and_receive(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    with adapter:
        adapter.start()
        adapter.limits = AppServerLimits(transcript_bytes=adapter._transcript_size)
        with pytest.raises(ProtocolError, match="outgoing transcript"):
            adapter._send("fixture/request", {})

    adapter, _ = _adapter(tmp_path / "incoming")
    with adapter:
        adapter.start()
        process = adapter._process
        assert process is not None
        adapter.limits = AppServerLimits(transcript_bytes=adapter._transcript_size)
        process.stdout.put({"method": "fixture", "params": {}})
        with pytest.raises(ProtocolError, match="bidirectional transcript"):
            adapter._read_message(time.monotonic() + 1.0)


def test_logs_persist_incrementally_on_success_and_failure(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path, artifacts=True)
    with adapter:
        adapter.generate("prompt", "codex/gpt-5.6-luna:high")
    root = tmp_path / "logs"
    for name in (
        "slot-00.codex-profile.json",
        "slot-00.codex-rpc.jsonl",
        "slot-00.events.jsonl",
        "slot-00.stdout.jsonl",
    ):
        assert (root / name).is_file(), name
    assert "gpt-5.6-luna" in (root / "slot-00.codex-profile.json").read_text()
    assert "turn/completed" in (root / "slot-00.events.jsonl").read_text()

    failed, _ = _adapter(
        tmp_path / "failed", FakeScenario(unknown_notification=True), artifacts=True
    )
    with failed, pytest.raises(ProtocolError):
        failed.generate("prompt", "codex/gpt-5.6-luna:high")
    assert (tmp_path / "failed" / "logs" / "slot-00.events.jsonl").is_file()
    assert (tmp_path / "failed" / "logs" / "slot-00.codex-rpc.jsonl").is_file()


def test_rollout_outside_capsule_is_never_copied_and_rollout_text_is_redacted(
    tmp_path: Path,
) -> None:
    adapter, _ = _adapter(tmp_path, artifacts=True)
    outside = tmp_path / "outside-rollout.jsonl"
    outside.write_text("Bearer private-rollout-token", encoding="utf-8")
    adapter._thread = {"id": "thread-1", "path": str(outside)}
    adapter._copy_rollout()
    assert not (tmp_path / "logs" / "slot-00.rollout.jsonl").exists()
    assert adapter.logger is not None
    adapter.logger.text("rollout.jsonl", "Bearer private-rollout-token")
    assert "private-rollout-token" not in (tmp_path / "logs" / "slot-00.rollout.jsonl").read_text(
        encoding="utf-8"
    )


def test_provider_persists_theml_style_initial_and_repair_artifacts(tmp_path: Path) -> None:
    auth_path = tmp_path / "authorized-auth.json"
    secret = "private-test-token"
    auth_path.write_text(json.dumps({"access_token": secret}), encoding="utf-8")
    auth_path.chmod(0o600)

    def factory(*_: Any, **kwargs: Any) -> Any:
        return FakeProcess(FakeScenario(), **kwargs)

    provider = AppServerGenerationProvider(
        process_factory=factory,
        auth_checker=lambda _: True,
        auth_json=auth_path,
        artifact_dir=tmp_path / "logs",
    )
    request = {
        "prompt": "Return one structured policy.",
        "system_prompt": "Return only the requested structured output.",
        "output_schema": {"type": "object", "additionalProperties": False},
        "model": "gpt-5.6-luna",
        "effort": "high",
        "artifact_prefix": "slot-00",
    }
    initial = provider.generate(request)
    repaired = provider.repair(request, ({"code": "schema", "message": "missing field"},))
    assert initial["status"] == repaired["status"] == "completed"

    logs = tmp_path / "logs"
    for prefix in ("slot-00", "slot-00.repair"):
        for name in (
            "request.md",
            "request.json",
            "response.md",
            "response.json",
            "provider-raw.json",
            "codex-profile.json",
            "codex-rpc.jsonl",
            "events.jsonl",
            "stdout.jsonl",
            "transcript.sha256",
        ):
            assert (logs / f"{prefix}.{name}").is_file(), f"{prefix}.{name}"
    retained = "".join(path.read_text(encoding="utf-8") for path in logs.iterdir())
    assert secret not in retained
    assert str(auth_path) not in retained
    assert initial["request_id"] >= 0
    assert initial["usage"]["final"] is True
    assert initial["usage"]["partial"] is False
    assert len((logs / "slot-00.transcript.sha256").read_text().strip()) == 64


def test_exact_v8_bundle_completes_nested_thread_started_structured_path(
    tmp_path: Path,
) -> None:
    config = load_stage3_config("configs/stage3-generation.toml")
    bundle = load_prompt_bundle(
        context_schema=config.context_schema_path,
        proposal_schema=config.proposal_schema_path,
        semantics_glossary=config.semantic_glossary_path,
        output_schema=config.output_schema_path,
    )
    brief = json.loads((config.slot_briefs_dir / "slot-00.json").read_text(encoding="utf-8"))
    prompt = bundle.render_slot_request(
        brief["slot_id"],
        brief["brief"],
        generation_mode=brief["generation_mode"],
        focus=brief["focus"],
    )
    output_schema = json.loads(bundle.output_schema)
    envelope = {
        "schema_version": "stage3.generated_policy.v1",
        "source": "def priority(ctx, proposal):\n    return 0.0\n",
        "design_summary": ("Hypothesis: a constant ranker matches an unstructured selection rule."),
        "used_fields": [],
        "assumptions": [],
    }
    process = FakeProcess(
        FakeScenario(
            final_text=json.dumps(envelope, sort_keys=True),
            thread_started_notification="nested",
        )
    )
    seen: list[dict[str, Any]] = []
    original_receive = process.receive

    def receive(line: bytes) -> None:
        seen.append(json.loads(line))
        original_receive(line)

    process.receive = receive
    provider = AppServerGenerationProvider(
        process_factory=_fixed_process_factory(process),
        auth_checker=lambda _: True,
        sandbox_mode=config.app_server.sandbox_mode,
        approval_policy=config.app_server.approval_policy,
    )
    result = provider.generate(
        {
            "prompt": prompt,
            "system_prompt": bundle.system,
            "output_schema": output_schema,
            "model": config.model.name,
            "effort": config.model.effort,
        }
    )

    assert result["status"] == "completed"
    assert result["accepted"] is True
    assert parse_generated_policy(json.loads(result["response"])).as_dict() == envelope
    thread_start = next(item for item in seen if item.get("method") == "thread/start")
    turn_start = next(item for item in seen if item.get("method") == "turn/start")
    assert thread_start["params"]["baseInstructions"] == bundle.system
    assert thread_start["params"]["dynamicTools"] == []
    assert thread_start["params"]["selectedCapabilityRoots"] == []
    assert turn_start["params"]["input"] == [{"type": "text", "text": prompt}]
    assert turn_start["params"]["outputSchema"] == output_schema
    assert turn_start["params"]["model"] == "gpt-5.6-luna"
    assert turn_start["params"]["effort"] == "high"
    assert (
        len(json.dumps(turn_start, separators=(",", ":")).encode("utf-8"))
        < config.limits.request_bytes
    )


def test_provider_request_and_response_bounds_fail_closed(tmp_path: Path) -> None:
    def request_factory(*_: Any, **kwargs: Any) -> Any:
        return FakeProcess(FakeScenario(), **kwargs)

    request_provider = AppServerGenerationProvider(
        process_factory=request_factory,
        auth_checker=lambda _: True,
        limits=AppServerLimits(request_bytes=2048),
    )
    request = {
        "prompt": "x" * 4096,
        "system_prompt": "short system",
        "output_schema": {"type": "object"},
        "model": "gpt-5.6-luna",
        "effort": "high",
    }
    with pytest.raises(ProtocolError, match="outgoing request exceeds limit"):
        request_provider.generate(request)

    def response_factory(*_: Any, **kwargs: Any) -> Any:
        return FakeProcess(FakeScenario(final_text="x" * 512), **kwargs)

    response_provider = AppServerGenerationProvider(
        process_factory=response_factory,
        auth_checker=lambda _: True,
        limits=AppServerLimits(response_bytes=64),
    )
    with pytest.raises(ProtocolError, match="structured response exceeds limit"):
        response_provider.generate({**request, "prompt": "short prompt"})


@pytest.mark.parametrize(
    ("model", "effort"),
    [("gpt-5.6-sol", "high"), ("gpt-5.6-luna", "medium")],
)
def test_provider_rejects_non_frozen_profile_before_spawn(
    tmp_path: Path, model: str, effort: str
) -> None:
    spawned = False

    def factory(*_: Any, **__: Any) -> Any:
        nonlocal spawned
        spawned = True
        raise AssertionError("provider must reject the profile before spawning")

    provider = AppServerGenerationProvider(
        process_factory=factory,
        auth_checker=lambda _: True,
        artifact_dir=tmp_path / "logs",
    )
    with pytest.raises(IsolationError, match="frozen gpt-5.6-luna:high profile"):
        provider.generate(
            {
                "prompt": "return a policy",
                "system_prompt": "return structured output",
                "output_schema": {"type": "object"},
                "model": model,
                "effort": effort,
            }
        )
    assert spawned is False
