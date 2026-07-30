"""Provider-independent Stage 4 wave tests (all providers are local fakes)."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import mutation_forge.stage4.generation as generation
from mutation_forge.stage4.generation import (
    GenerationCoordinator,
    ProviderResult,
    cached_pre_turn_auth_retry_allowed,
    infrastructure_retry_allowed,
)

SOURCE = "def priority(ctx, proposal):\n    return 1.0\n"


def envelope(source: str = SOURCE) -> dict[str, Any]:
    return {
        "schema_version": "stage4.generated_policy.v1",
        "source": source,
        "design_summary": "offline",
        "change_summary": "offline change",
        "hypothesis": "offline hypothesis",
        "used_fields": ["proposal.k"],
        "assumptions": ["validated inputs"],
        "expected_failure_modes": ["none"],
    }


class FakeProvider:
    def __init__(self, *, invalid: set[tuple[int, str]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.repairs: list[dict[str, Any]] = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.invalid = invalid or set()

    def generate(self, request: dict[str, Any]) -> ProviderResult:
        with self.lock:
            self.calls.append(dict(request))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.001)
            key = (int(request["generation"]), str(request["slot"]))
            value: Any = {"not": "a policy"} if key in self.invalid else envelope()
            return ProviderResult(
                response=value,
                usage={
                    "inputTokens": 1,
                    "cachedInputTokens": 0,
                    "outputTokens": 1,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 2,
                    "final": True,
                    "partial": False,
                },
                request_id=f"r-{request['generation']}-{request['slot']}",
                thread_id="thread",
                turn_id=f"t-{request['generation']}-{request['slot']}",
            )
        finally:
            with self.lock:
                self.active -= 1

    def repair(
        self, request: dict[str, Any], diagnostics: tuple[dict[str, Any], ...]
    ) -> ProviderResult:
        self.repairs.append(dict(request))
        return ProviderResult(
            response=envelope(),
            usage={
                "inputTokens": 1,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
                "totalTokens": 1,
                "final": True,
                "partial": False,
            },
        )


def test_four_waves_have_exact_order_and_eight_way_concurrency(monkeypatch: Any) -> None:
    provider = FakeProvider()
    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "a" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    result = GenerationCoordinator(provider).run()
    assert result.status == "completed"
    assert len(provider.calls) == 32
    assert provider.peak >= 2
    assert provider.peak <= 8
    assert [item.slot for item in result.slots[:8]] == list(generation.SLOTS)
    assert [item.generation for item in result.slots] == [g for g in range(4) for _ in range(8)]


def test_one_repair_and_duplicate_without_replacement(monkeypatch: Any) -> None:
    provider = FakeProvider(invalid={(0, "slot-00")})
    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "b" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    result = GenerationCoordinator(provider).run()
    assert len(provider.calls) == 32
    assert len(provider.repairs) == 1
    assert result.summary["total_live_turns"] == 33
    assert result.summary["unique_count"] == 1
    assert sum(item.status == "duplicate" for item in result.slots) == 31


def test_checkpoint_resume_does_not_repeat_completed_requests(
    monkeypatch: Any, tmp_path: Any
) -> None:
    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "c" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    path = tmp_path / "checkpoint.json"
    first = FakeProvider()
    GenerationCoordinator(first, checkpoint_path=path).run()
    second = FakeProvider()
    resumed = GenerationCoordinator(second, checkpoint_path=path).run()
    assert second.calls == []
    assert resumed.summary["initial_turn_count"] == 0


def test_infrastructure_retry_predicate_is_fail_closed() -> None:
    assert infrastructure_retry_allowed(
        ProviderResult(
            status="infrastructure",
            accepted=False,
            charged=False,
            content=False,
            uncharged=True,
            usage={},
        )
    )
    assert not infrastructure_retry_allowed(
        ProviderResult(
            status="infrastructure", accepted=False, charged=False, content=False, usage={}
        )
    )
    assert not infrastructure_retry_allowed(
        ProviderResult(
            status="infrastructure",
            accepted=False,
            charged=False,
            content=False,
            usage={"totalTokens": 1},
        )
    )
    assert not infrastructure_retry_allowed(
        ProviderResult(status="infrastructure", accepted=True, charged=True, content=True)
    )
    assert not infrastructure_retry_allowed(
        ProviderResult(
            status="infrastructure",
            accepted=False,
            charged=False,
            content=False,
            uncharged=True,
            unauthorized_tool_approval=True,
            usage={},
        )
    )


def test_retained_auth_failure_retries_same_requests_only(
    monkeypatch: Any, tmp_path: Any
) -> None:
    class LegacyAuthFailureProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate(self, request: dict[str, Any]) -> ProviderResult:
            self.calls.append(dict(request))
            return ProviderResult(
                status="infrastructure",
                accepted=False,
                charged=False,
                content=False,
                uncharged=False,
                usage={},
                error="IsolationError: isolated Codex home is not authenticated",
            )

    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "3" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    path = tmp_path / "checkpoint.json"
    failed = LegacyAuthFailureProvider()
    first = GenerationCoordinator(failed, checkpoint_path=path).run()
    assert len(failed.calls) == 32
    assert all(cached_pre_turn_auth_retry_allowed(slot.as_dict()) for slot in first.slots)

    replacement = FakeProvider()
    resumed = GenerationCoordinator(
        replacement,
        checkpoint_path=path,
        retry_infrastructure=True,
    ).run()
    assert len(replacement.calls) == 32
    assert {
        request["idempotency_key"] for request in replacement.calls
    } == {
        request["idempotency_key"] for request in failed.calls
    }
    assert resumed.summary["initial_turn_count"] == 32


def test_retained_auth_failure_classifier_is_exact() -> None:
    value = {
        "status": "failed",
        "repairs": 0,
        "repair": None,
        "initial": {
            "status": "infrastructure",
            "accepted": False,
            "charged": False,
            "content": False,
            "uncharged": False,
            "usage": {},
            "response": None,
            "request_id": None,
            "thread_id": None,
            "turn_id": None,
            "session_id": None,
            "provider_request_id": None,
            "provider_thread_id": None,
            "provider_turn_id": None,
            "error": "IsolationError: isolated Codex home is not authenticated",
        },
    }
    assert cached_pre_turn_auth_retry_allowed(value)
    assert not cached_pre_turn_auth_retry_allowed(
        {
            **value,
            "initial": {
                **value["initial"],
                "usage": {"totalTokens": 1},
            },
        }
    )
    assert not cached_pre_turn_auth_retry_allowed(
        {
            **value,
            "initial": {
                **value["initial"],
                "turn_id": "turn-1",
            },
        }
    )
    assert not cached_pre_turn_auth_retry_allowed(
        {
            **value,
            "initial": {
                **value["initial"],
                "error": "different infrastructure failure",
            },
        }
    )


def test_accepted_artifact_recovery_never_submits_a_replacement(
    monkeypatch: Any,
) -> None:
    class RetainedProvider:
        def __init__(self) -> None:
            self.loaded: list[dict[str, Any]] = []
            self.generated = 0

        def load_retained_result(
            self,
            request: dict[str, Any],
        ) -> ProviderResult:
            self.loaded.append(dict(request))
            return ProviderResult(
                response=envelope(),
                usage={
                    "inputTokens": 1,
                    "cachedInputTokens": 0,
                    "outputTokens": 1,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 2,
                    "final": True,
                    "partial": False,
                },
                request_id=f"retained-{request['generation']}-{request['slot']}",
                thread_id="retained-thread",
                turn_id=f"retained-turn-{request['generation']}-{request['slot']}",
            )

        def generate(self, request: dict[str, Any]) -> ProviderResult:
            self.generated += 1
            raise AssertionError("retained accepted request must not be submitted again")

    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "4" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    provider = RetainedProvider()
    result = GenerationCoordinator(provider).run()
    assert provider.generated == 0
    assert len(provider.loaded) == 32
    assert result.summary["initial_turn_count"] == 32
    assert result.summary["recovered_initial_turn_count"] == 32
    assert result.summary["accepted_live_turns"] == 32


def test_json_text_response_and_real_parent_prompt_inputs(monkeypatch: Any) -> None:
    class TextProvider(FakeProvider):
        def generate(self, request: dict[str, Any]) -> ProviderResult:
            result = super().generate(request)
            return ProviderResult(
                response=json.dumps(result.response),
                usage=result.usage,
                request_id=result.request_id,
                thread_id=result.thread_id,
                turn_id=result.turn_id,
            )

    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "d" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    provider = TextProvider()
    GenerationCoordinator(
        provider,
        parent_assignments={0: {slot: "parent-a" for slot in generation.SLOTS}},
        parent_sources={"parent-a": SOURCE},
        parent_records={"parent-a": {"program_id": "parent-a", "generation": 0}},
        briefs={0: {slot: "brief" for slot in generation.SLOTS}},
        search_feedback="compact feedback",
        archive_context="bounded archive",
    ).run()
    first = provider.calls[0]
    assert SOURCE in first["prompt"]
    assert "parent-a" in first["prompt"]
    assert "brief" in first["prompt"]
    assert "compact feedback" in first["prompt"]


def test_repair_prompt_contains_source_and_only_bounded_diagnostics(monkeypatch: Any) -> None:
    provider = FakeProvider(invalid={(0, "slot-00")})
    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "e" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    GenerationCoordinator(provider).run()
    repair = provider.repairs[0]
    assert "Repair the supplied policy" in repair["prompt"]
    assert "SOURCE" in repair["prompt"]
    assert "performance" in repair["prompt"].lower()
    assert "fitness" in repair["prompt"].lower()


def test_static_ast_failure_gets_one_repair(monkeypatch: Any) -> None:
    class AstInvalidProvider(FakeProvider):
        def generate(self, request: dict[str, Any]) -> ProviderResult:
            result = super().generate(request)
            return ProviderResult(
                response=envelope("import os\n\ndef priority(ctx, proposal):\n    return 1.0\n"),
                usage=result.usage,
                request_id=result.request_id,
                thread_id=result.thread_id,
                turn_id=result.turn_id,
            )

    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "e" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    provider = AstInvalidProvider()
    GenerationCoordinator(provider).run()
    assert len(provider.calls) == 32
    assert len(provider.repairs) == 32


def test_runtime_failure_never_gets_a_repair(monkeypatch: Any) -> None:
    provider = FakeProvider()

    def fail_behavior(source: str, limits: Any, smoke_calls: int) -> Any:
        raise RuntimeError("input_mutation")

    monkeypatch.setattr(generation, "_behavior", fail_behavior)
    result = GenerationCoordinator(provider).run()
    assert result.status == "completed"
    assert len(provider.calls) == 32
    assert provider.repairs == []


def test_archive_source_path_is_written_atomically(monkeypatch: Any, tmp_path: Any) -> None:
    class Archive:
        def __init__(self) -> None:
            self.root = tmp_path / "archive"
            self.records: list[dict[str, Any]] = []

        def append(self, record: dict[str, Any]) -> None:
            self.records.append(record)

    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "f" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    archive = Archive()
    GenerationCoordinator(FakeProvider(), archive=archive).run()
    record = archive.records[0]
    assert (archive.root / record["source_path"]).read_text() == SOURCE


def test_generation_callback_runs_in_order_before_parent_selection(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "1" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    events: list[tuple[str, int]] = []
    state: dict[int, bool] = {}

    def select(generation_index: int) -> dict[str, str]:
        events.append(("select", generation_index))
        if generation_index > 0:
            assert state[generation_index - 1]
        return {slot: f"p-{generation_index}" for slot in generation.SLOTS}

    def completed(index: int, slots: tuple[Any, ...], candidates: tuple[Any, ...]) -> None:
        events.append(("callback", index))
        state[index] = True
        assert len(slots) == 8

    GenerationCoordinator(
        FakeProvider(), parent_selector=select, generation_completed=completed
    ).run()
    assert [event for event in events if event[0] == "callback"] == [
        ("callback", index) for index in range(4)
    ]
    assert events.index(("callback", 0)) < events.index(("select", 1))


def test_callback_interruption_resumes_without_repeating_turns(
    monkeypatch: Any, tmp_path: Any
) -> None:
    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "2" * 64},
            {"smoke_calls": smoke_calls},
        ),
    )
    path = tmp_path / "checkpoint.json"
    interrupted = {"value": True}

    def callback(index: int, slots: tuple[Any, ...], candidates: tuple[Any, ...]) -> None:
        if interrupted["value"] and index == 0:
            interrupted["value"] = False
            raise RuntimeError("evaluation interrupted")

    first = FakeProvider()
    try:
        GenerationCoordinator(first, checkpoint_path=path, generation_completed=callback).run()
    except RuntimeError as error:
        assert str(error) == "evaluation interrupted"
    else:
        raise AssertionError("callback interruption should propagate")
    second = FakeProvider()
    result = GenerationCoordinator(
        second, checkpoint_path=path, generation_completed=callback
    ).run()
    assert result.status == "completed"
    assert len(first.calls) == 8
    assert len(second.calls) == 24
    assert not {request["idempotency_key"] for request in first.calls} & {
        request["idempotency_key"] for request in second.calls
    }
