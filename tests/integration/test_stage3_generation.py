from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from mutation_forge.stage3.artifacts import GenerationArtifacts, replay_generation
from mutation_forge.stage3.generation import GenerationConfig, OneShotGenerator

SOURCE = (
    "def priority(ctx, proposal):\n"
    '    return proposal["broken_sampled_witnesses_by_length"][0]'
    ' - proposal["local_c4_risk"] * 0.25'
)
ENVELOPE = {
    "schema_version": "stage3.generated_policy.v1",
    "source": SOURCE,
    "design_summary": "Prefer sampled witness removal while controlling local C4 risk.",
    "used_fields": [
        "proposal.broken_sampled_witnesses_by_length",
        "proposal.local_c4_risk",
    ],
    "assumptions": ["Host validation guarantees legal proposals."],
}
USAGE = {
    "inputTokens": 10,
    "cachedInputTokens": 0,
    "cacheWriteInputTokens": 0,
    "outputTokens": 10,
    "reasoningOutputTokens": 2,
    "totalTokens": 20,
}


class FakeProvider:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = responses or [ENVELOPE]
        self.calls = 0
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def generate(self, request: dict[str, object]) -> dict[str, object]:
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            index = self.calls
            self.calls += 1
        try:
            time.sleep(0.01)
            response = self.responses[min(index, len(self.responses) - 1)]
            return {
                "response": response,
                "accepted": True,
                "charged": True,
                "content": True,
                "usage": USAGE,
                "status": "completed",
                "request_id": f"request-{index}",
                "thread_id": f"thread-{index}",
                "session_id": f"session-{index}",
                "turn_id": f"turn-{index}",
                "model": "gpt-5.6-luna",
            }
        finally:
            with self.lock:
                self.active -= 1


class RepairProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([{"source": SOURCE}])
        self.initial_completed = 0
        self.repairs = 0

    def generate(self, request: dict[str, object]) -> dict[str, object]:
        result = super().generate(request)
        with self.lock:
            self.initial_completed += 1
        return result

    def repair(
        self,
        request: dict[str, object],
        diagnostics: tuple[dict[str, object], ...],
    ) -> dict[str, object]:
        assert self.initial_completed == 8
        assert diagnostics
        with self.lock:
            self.repairs += 1
            index = 100 + self.repairs
        return {
            "response": ENVELOPE,
            "accepted": True,
            "charged": True,
            "content": True,
            "usage": USAGE,
            "status": "completed",
            "request_id": f"request-{index}",
            "thread_id": f"thread-{index}",
            "session_id": f"session-{index}",
            "turn_id": f"turn-{index}",
            "model": "gpt-5.6-luna",
        }


def test_eight_slots_are_concurrent_and_deduplicated() -> None:
    provider = FakeProvider()
    result = OneShotGenerator(provider, config=GenerationConfig(smoke_calls=0)).run()
    assert result.status == "completed"
    assert [slot.slot for slot in result.slots] == [f"slot-{i:02d}" for i in range(8)]
    assert len(result.unique_candidates) == 1
    assert provider.maximum == 8


def test_artifacts_are_readable_and_replay_is_model_free(tmp_path: Path) -> None:
    artifacts = GenerationArtifacts(tmp_path, "run")
    result = OneShotGenerator(
        FakeProvider(), config=GenerationConfig(smoke_calls=0), artifacts=artifacts
    ).run()
    summary = replay_generation(tmp_path / "run")
    assert summary["status"] == result.status
    assert json.loads((tmp_path / "run" / "run_summary.json").read_text())["status"] == "completed"


def test_repairs_start_only_after_all_initial_turns() -> None:
    provider = RepairProvider()
    result = OneShotGenerator(provider, config=GenerationConfig(smoke_calls=0)).run()
    assert provider.initial_completed == 8
    assert provider.repairs == 8
    assert result.status == "completed"
    assert all(slot.repairs == 1 for slot in result.slots)


def test_artifacts_redact_credentials_preserve_usage_and_stay_terminal(
    tmp_path: Path,
) -> None:
    artifacts = GenerationArtifacts(tmp_path, "bounded", max_file_bytes=512)
    artifacts.start({"accessToken": "secret", "inputTokens": 7})
    persisted = json.loads((artifacts.root / "generation_summary.json").read_text())
    assert persisted["accessToken"] == "[REDACTED]"
    assert persisted["inputTokens"] == 7
    assert persisted["status"] == "failed"
    with pytest.raises(ValueError, match="per-file"):
        artifacts.write("oversized.json", {"value": "x" * 1024})
