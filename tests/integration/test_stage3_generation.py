"""Offline tests for the frozen Stage 3 one-shot generation campaign.

These tests inject a small provider in place of the App Server.  They exercise
the campaign accounting and safety rules without making a model, network, or
Codex App Server call.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import mutation_forge.stage3.generation as generation
from mutation_forge.stage3.artifacts import GenerationArtifacts, replay_generation
from mutation_forge.stage3.generation import (
    GenerationConfig,
    OneShotGenerator,
    Turn,
)

SOURCE = "def priority(ctx, proposal):\n    return 1.0\n"


def _usage(total: int = 3) -> dict[str, int]:
    return {
        "inputTokens": 1,
        "cachedInputTokens": 0,
        "outputTokens": 1,
        "reasoningOutputTokens": 1,
        "totalTokens": total,
        "final": True,
        "partial": False,
    }


def _envelope(source: str = SOURCE) -> dict[str, Any]:
    return {
        "schema_version": "stage3.generated_policy.v1",
        "source": source,
        "design_summary": "constant deterministic policy",
        "used_fields": ["proposal.k"],
        "assumptions": ["inputs are validated"],
    }


def _turn(
    response: Any,
    *,
    status: str = "completed",
    accepted: bool = True,
    content: bool = True,
    charged: bool = True,
    slot: str = "slot-00",
    usage_final: bool = True,
    usage_partial: bool = False,
    transport_sha256: str | None = None,
) -> Turn:
    usage = _usage() if status == "completed" else {}
    if status == "completed":
        usage["final"] = usage_final
        usage["partial"] = usage_partial
    return Turn(
        response=response,
        accepted=accepted,
        charged=charged,
        content=content,
        usage=usage,
        status=status,
        request_id=f"request-{slot}",
        thread_id=f"thread-{slot}",
        session_id=f"session-{slot}",
        turn_id=f"turn-{slot}",
        model="gpt-5.6-luna",
        transport_sha256=transport_sha256 or f"transport-{slot}",
    )


class RecordingProvider:
    """Thread-safe fake provider used by all tests in this module."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.requests: list[dict[str, Any]] = []
        self.repairs: list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = []
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def generate(self, request: dict[str, Any]) -> Turn:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.requests.append(dict(request))
        try:
            # Give the executor an opportunity to overlap all eight initial slots.
            time.sleep(0.002)
            slot = str(request["slot"])
            fallback = _envelope(SOURCE.replace("1.0", f"{int(slot[-2:]) + 1}.0"))
            response = self.responses.get(slot, fallback)
            if isinstance(response, Turn):
                return response
            return _turn(response, slot=slot)
        finally:
            with self._lock:
                self.active -= 1

    def repair(self, request: dict[str, Any], diagnostics: tuple[dict[str, Any], ...]) -> Turn:
        with self._lock:
            self.repairs.append((dict(request), diagnostics))
        return _turn(_envelope(), slot=str(request["slot"]))


@pytest.fixture(autouse=True)
def cheap_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid launching 10,000 real sandbox calls in unit tests."""

    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"rank_order": ["proposal"], "signature_sha256": "offline"},
            {"persistent_smoke": {"calls": smoke_calls}, "smoke_calls": smoke_calls},
        ),
    )


def test_exactly_eight_ordered_initial_slots_and_bounded_concurrency() -> None:
    provider = RecordingProvider()
    result = OneShotGenerator(
        provider,
        config=GenerationConfig(model="gpt-5.6-luna", effort="high", smoke_calls=10_000),
    ).run(run_id="offline-eight")

    assert result.status == "completed"
    assert [slot.slot for slot in result.slots] == [f"slot-{i:02d}" for i in range(8)]
    assert provider.calls == 8
    assert provider.repairs == []
    assert provider.max_active <= 8
    assert provider.max_active >= 2
    assert result.summary["initial_turn_count"] == 8
    assert result.summary["total_live_turns"] == 8
    assert result.summary["max_active"] <= 8
    assert all(
        slot.candidate and slot.candidate.worker_telemetry["persistent_smoke"]["calls"] == 10_000
        for slot in result.slots
    )
    assert all(request["model"] == "gpt-5.6-luna" for request in provider.requests)
    assert all(request["effort"] == "high" for request in provider.requests)


def test_frozen_generation_config_rejects_wrong_budget_or_profile() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(smoke_calls=9_999)
    with pytest.raises(ValueError):
        GenerationConfig(model="other")
    with pytest.raises(ValueError):
        GenerationConfig(effort="medium")


def test_doctor_provenance_mismatch_is_terminal_and_not_repaired() -> None:
    provider = RecordingProvider()
    requests = {
        slot: {
            "slot": slot,
            "model": "gpt-5.6-luna",
            "effort": "high",
            "protocol_version": "stage3.generation.v1",
            "prompt": f"prompt-{slot}",
            "system_prompt": "system",
            "output_schema": {"type": "object"},
            "appserver_doctor_sha256": "doctor-a",
        }
        for slot in generation.SLOTS
    }
    result = OneShotGenerator(provider, slot_requests=requests).run(run_id="offline-doctor")
    assert provider.repairs == []
    assert result.status == "failed"
    assert any(error["code"] == "turn_provenance" for error in result.slots[0].errors)


def test_one_schema_repair_only_and_no_infrastructure_retry() -> None:
    invalid = {"not": "a generated-policy envelope"}
    provider = RecordingProvider(
        {
            "slot-00": invalid,
            "slot-01": _turn(
                {},
                status="crashed",
                accepted=False,
                content=False,
                charged=False,
                slot="slot-01",
            ),
        }
    )
    result = OneShotGenerator(provider).run(run_id="offline-repair")

    assert provider.calls == 8
    assert len(provider.repairs) == 1
    assert provider.repairs[0][0]["slot"] == "slot-00"
    # The repair channel receives the original immutable request plus
    # diagnostics; the provider does not receive a mutable retry marker.
    assert provider.repairs[0][1]
    assert result.summary["repair_turn_count"] == 1
    assert result.summary["total_live_turns"] == 9
    assert result.slots[0].status == "accepted"
    assert result.slots[1].status == "failed"
    assert result.slots[1].repairs == 0


@pytest.mark.parametrize(
    "source",
    [
        "def priority(ctx, proposal):\n    return proposal[\"k\"] * 1000\n",
    ],
)
def test_static_ast_validation_failure_receives_one_repair(source: str) -> None:
    provider = RecordingProvider({"slot-00": _envelope(source)})
    result = OneShotGenerator(provider).run(run_id="offline-ast-repair")

    assert len(provider.repairs) == 1
    assert provider.repairs[0][0]["slot"] == "slot-00"
    assert provider.repairs[0][1][0]["code"] == "multiplication_bound"
    assert result.slots[0].status == "accepted"
    assert result.slots[0].repairs == 1
    assert result.summary["repair_turn_count"] == 1


def test_runtime_failures_and_nonfinal_usage_never_trigger_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for code in (
        "worker_timeout",
        "worker_crash",
        "worker_protocol",
        "runtime_exception",
        "finite_probe",
    ):
        provider = RecordingProvider()

        def fail_behavior(source: str, limits: Any, smoke_calls: int, *, _code: str = code) -> Any:
            raise ValueError(_code)

        monkeypatch.setattr(generation, "_behavior", fail_behavior)
        result = OneShotGenerator(provider).run(run_id=f"offline-{code}")
        assert provider.repairs == [], code
        assert all(slot.status == "failed" for slot in result.slots), code

    monkeypatch.setattr(
        generation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": "offline"},
            {"smoke_calls": smoke_calls},
        ),
    )
    provider = RecordingProvider(
        {"slot-00": _turn(_envelope(), usage_final=False, usage_partial=True)}
    )
    result = OneShotGenerator(provider).run(run_id="offline-partial-usage")
    assert provider.repairs == []
    assert result.slots[0].status == "failed"
    assert any(error["code"] == "usage_missing" for error in result.slots[0].errors)


def test_weak_and_duplicate_candidates_are_not_replaced() -> None:
    # slot-00 and slot-01 are semantically identical (whitespace differs), and
    # slot-02 is weak output.  The baseline duplicate is rejected as well.
    equivalent = "def priority(ctx, proposal):\n\n    return 1.0\n"
    provider = RecordingProvider(
        {
            **{f"slot-{i:02d}": _envelope(SOURCE) for i in range(8)},
            **{
                "slot-00": _envelope(SOURCE),
                "slot-01": _envelope(equivalent),
                "slot-02": _turn({}, accepted=False, content=False, charged=False, slot="slot-02"),
            },
        }
    )
    result = OneShotGenerator(provider).run(run_id="offline-dedup")

    assert provider.calls == 8
    assert provider.repairs == []
    assert len(result.unique_candidates) == 1
    assert result.slots[0].status == "accepted"
    assert result.slots[1].status == "duplicate"
    assert result.slots[2].status == "failed"
    assert result.summary["unique_count"] == 1

    baseline_provider = RecordingProvider({f"slot-{i:02d}": _envelope(SOURCE) for i in range(8)})
    baseline = OneShotGenerator(baseline_provider, existing_sources=(SOURCE,)).run(
        run_id="offline-baseline-dedup"
    )
    assert baseline_provider.repairs == []
    assert baseline.summary["unique_count"] == 0
    assert all(slot.status == "duplicate" for slot in baseline.slots)


def test_exact_usage_and_provenance_accounting() -> None:
    provider = RecordingProvider()
    result = OneShotGenerator(provider).run(run_id="offline-usage")
    assert result.summary["exact_usage_complete"] is True
    assert result.summary["usage_totals"]["totalTokens"] == 8 * 3
    assert result.summary["usage_totals"]["inputTokens"] == 8
    assert all(slot.candidate is not None for slot in result.slots)
    assert all(
        slot.candidate and slot.candidate.provenance["model"] == "gpt-5.6-luna"
        for slot in result.slots
    )
    assert all(
        slot.candidate
        and slot.candidate.provenance["request_id"] == f"request-{slot.slot}"
        and slot.candidate.provenance["transport_sha256"] == f"transport-{slot.slot}"
        for slot in result.slots
    )


def test_artifacts_are_written_and_replay_makes_zero_provider_calls(tmp_path: Path) -> None:
    provider = RecordingProvider()
    artifacts = GenerationArtifacts(tmp_path, "primary")
    result = OneShotGenerator(provider, artifacts=artifacts).run(run_id="primary")
    root = artifacts.root
    assert (root / "generation_summary.json").is_file()
    assert (root / "slots.json").is_file()
    config = json.loads((root / "generation_config.json").read_text())
    assert config["model"] == "gpt-5.6-luna"
    assert config["effort"] == "high"
    assert config["smoke_calls"] == 10_000
    assert (root / "slots/slot-00/events.json").is_file()
    replay = replay_generation(root)
    assert replay["replay_validated"] is True
    assert replay["provider_calls"] == 0
    assert replay["status"] == result.status


def test_replay_rejects_changed_source_identity_and_behavior(tmp_path: Path) -> None:
    artifacts = GenerationArtifacts(tmp_path, "replay-integrity")
    OneShotGenerator(RecordingProvider(), artifacts=artifacts).run(run_id="replay-integrity")
    root = artifacts.root
    slot = root / "slots" / "slot-00"
    source = (slot / "source.py").read_text()
    identity = (slot / "identity.json").read_text()
    behavior = (slot / "behavior.json").read_text()

    (slot / "source.py").write_text(source + "\n")
    changed_source = replay_generation(root)
    assert changed_source["replay_validated"] is False
    assert changed_source["provider_calls"] == 0
    (slot / "source.py").write_text(source)

    identity_value = json.loads(identity)
    identity_value["source_sha256"] = "0" * 64
    (slot / "identity.json").write_text(json.dumps(identity_value))
    changed_identity = replay_generation(root)
    assert changed_identity["replay_validated"] is False
    (slot / "identity.json").write_text(identity)

    behavior_value = json.loads(behavior)
    behavior_value["signature"]["signature_sha256"] = "changed"
    (slot / "behavior.json").write_text(json.dumps(behavior_value))
    changed_behavior = replay_generation(root)
    assert changed_behavior["replay_validated"] is False
    assert changed_behavior["provider_calls"] == 0


def test_terminal_artifact_survives_interrupted_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = GenerationArtifacts(tmp_path, "interrupted")
    provider = RecordingProvider()
    original = OneShotGenerator._assess

    def interrupt(self: OneShotGenerator, slot: str, turn: Turn, **kwargs: Any) -> Any:
        if slot == "slot-00":
            raise RuntimeError("interrupted test")
        return original(self, slot, turn, **kwargs)

    monkeypatch.setattr(OneShotGenerator, "_assess", interrupt)
    result = OneShotGenerator(provider, artifacts=artifacts).run(run_id="interrupted")
    assert result.status == "failed"
    summary = json.loads((artifacts.root / "generation_summary.json").read_text())
    assert summary["status"] == "failed"
    assert (artifacts.root / "run_summary.json").is_file()
