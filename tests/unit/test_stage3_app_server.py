from __future__ import annotations

import pytest
from fixtures.fake_stage3_app_server import FakeScenario, fake_process_factory

from mutation_forge.stage3.app_server import (
    AppServerError,
    AppServerLimits,
    CodexAppServerAdapter,
    IsolationError,
    ProtocolError,
    TurnError,
    resolve_model_profile,
)


def make_adapter(scenario: FakeScenario) -> CodexAppServerAdapter:
    return CodexAppServerAdapter(
        process_factory=fake_process_factory(scenario),
        auth_checker=lambda _: True,
        limits=AppServerLimits(startup_timeout=0.25, turn_timeout=1, usage_grace=0.05),
    )


def test_profile_resolution_is_local() -> None:
    assert resolve_model_profile("codex/gpt-5.6-sol:high").model == "gpt-5.6-sol"


def test_model_catalog_uses_protocol_without_starting_a_turn() -> None:
    adapter = make_adapter(FakeScenario())
    catalog = adapter.model_catalog()
    assert catalog[0]["model"] == "gpt-5.6-luna"
    assert adapter.inspect_metadata()["turns"] == 0
    adapter.close()


def test_fixture_generation_and_late_usage() -> None:
    scenario = FakeScenario(enabled_skills=["/private/skill/SKILL.md"])
    adapter = make_adapter(scenario)
    result = adapter.generate("hello", "codex/test:high", output_schema={"type": "object"})
    assert result.text == "fixture answer"
    assert result.usage.total_tokens == 5
    assert scenario.enabled_skills == []
    adapter.close()


def test_thread_started_accepts_top_level_thread_id() -> None:
    adapter = make_adapter(FakeScenario())
    adapter._correlate_thread_started({"threadId": "thread-1"}, "thread-1")
    adapter.close()


def test_thread_started_accepts_nested_thread_id() -> None:
    adapter = make_adapter(FakeScenario())
    adapter._correlate_thread_started({"thread": {"id": "thread-1"}}, "thread-1")
    adapter.close()


def test_thread_started_rejects_missing_thread_id() -> None:
    adapter = make_adapter(FakeScenario())
    with pytest.raises(ProtocolError, match="valid thread ID"):
        adapter._correlate_thread_started({}, "thread-1")
    adapter.close()


def test_thread_started_rejects_foreign_thread_id() -> None:
    adapter = make_adapter(FakeScenario())
    with pytest.raises(ProtocolError, match="foreign"):
        adapter._correlate_thread_started({"thread": {"id": "thread-2"}}, "thread-1")
    adapter.close()


def test_queued_nested_thread_started_is_consumed_before_turn_response() -> None:
    adapter = make_adapter(FakeScenario(thread_started_notification="nested"))
    assert adapter.generate("hello", "test:high").text == "fixture answer"
    adapter.close()


def test_thread_started_before_thread_start_response_is_rejected() -> None:
    adapter = make_adapter(
        FakeScenario(thread_started_notification="nested-before-thread-response")
    )
    with pytest.raises(ProtocolError, match="before thread/start response"):
        adapter.generate("hello", "test:high")


def test_thread_started_after_turn_start_response_is_rejected() -> None:
    adapter = make_adapter(
        FakeScenario(thread_started_notification="nested-after-turn-response")
    )
    with pytest.raises(ProtocolError, match="after turn/start response"):
        adapter.generate("hello", "test:high")


def test_missing_usage_fails_closed() -> None:
    with pytest.raises(TurnError):
        make_adapter(FakeScenario(usage=None)).generate("hello", "test:high")


def test_terminal_failure_fails_closed() -> None:
    with pytest.raises(TurnError):
        make_adapter(FakeScenario(terminal_status="failed")).generate("hello", "test:high")


def test_server_request_is_denied() -> None:
    with pytest.raises((IsolationError, Exception)):
        make_adapter(FakeScenario(server_request=True)).generate("hello", "test:high")


@pytest.mark.parametrize(
    "scenario",
    [
        FakeScenario(usage=None),
        FakeScenario(malformed=True),
        FakeScenario(oversized=True),
        FakeScenario(crash=True),
        FakeScenario(unknown_notification=True),
        FakeScenario(terminal_status="cancelled"),
        FakeScenario(terminal_status="interrupted"),
    ],
)
def test_protocol_abuse_fails_only_the_adapter(scenario: FakeScenario) -> None:
    with pytest.raises(AppServerError):
        make_adapter(scenario).generate("hello", "test:high")
    healthy = make_adapter(FakeScenario())
    assert healthy.generate("hello", "test:high").text == "fixture answer"
    healthy.close()
