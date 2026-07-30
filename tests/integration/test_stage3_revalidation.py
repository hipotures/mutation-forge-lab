from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.stage3 import revalidation
from mutation_forge.stage3.config import load_stage3_config


def _response(slot: str, index: int) -> dict[str, Any]:
    candidate = {
        "schema_version": "stage3.generated_policy.v1",
        "source": (
            "def priority(ctx, proposal):\n"
            f"    return proposal[\"k\"] + {index}\n"
        ),
        "design_summary": "Hypothesis: this deterministic test policy is distinguishable.",
        "used_fields": ["proposal.k"],
        "assumptions": ["Larger k is useful only for this test fixture."],
    }
    return {
        "status": "completed",
        "accepted": True,
        "charged": True,
        "content": True,
        "model": "gpt-5.6-luna",
        "effort": "high",
        "request_id": f"request-{slot}",
        "thread_id": f"thread-{slot}",
        "turn_id": f"turn-{slot}",
        "response": json.dumps(candidate),
        "usage": {
            "inputTokens": 1,
            "cachedInputTokens": 0,
            "cacheWriteInputTokens": 0,
            "outputTokens": 1,
            "reasoningOutputTokens": 0,
            "totalTokens": 2,
            "final": True,
            "partial": False,
        },
    }


def test_saved_generation_revalidation_is_provider_free_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "retained-run"
    (run / "slots").mkdir(parents=True)
    generation_summary = {
        "status": "failed",
        "total_live_turns": 13,
        "initial_max_active": 8,
        "max_active": 8,
    }
    (run / "generation_summary.json").write_text(
        json.dumps(generation_summary), encoding="utf-8"
    )
    (run / "freeze.json").write_text(
        json.dumps({"preregistration_tag": "stage3-generation-frozen-v11"}),
        encoding="utf-8",
    )
    raw_hashes: dict[Path, str] = {}
    for index in range(8):
        slot = f"slot-{index:02d}"
        slot_root = run / "slots" / slot
        slot_root.mkdir()
        initial = slot_root / f"{slot}.response.json"
        initial.write_text(json.dumps(_response(slot, index)), encoding="utf-8")
        raw_hashes[initial] = hashlib.sha256(initial.read_bytes()).hexdigest()
        if index % 2:
            repair = slot_root / f"{slot}.repair.response.json"
            repair.write_text(json.dumps(_response(slot, index)), encoding="utf-8")
            raw_hashes[repair] = hashlib.sha256(repair.read_bytes()).hexdigest()

    monkeypatch.setattr(
        revalidation,
        "_behavior",
        lambda source, limits, smoke_calls: (
            {"signature_sha256": hashlib.sha256(source.encode()).hexdigest()},
            {"smoke_calls": smoke_calls, "persistent_smoke": {"calls": smoke_calls}},
        ),
    )
    config = load_stage3_config("configs/stage3-generation.toml")
    summary = revalidation.revalidate_saved_generation(config, run, persist=True)

    assert summary["status"] == "completed"
    assert summary["provider_calls"] == 0
    assert summary["model_calls"] == 0
    assert summary["app_server_calls"] == 0
    assert summary["all_valid"] is True
    assert summary["unique_count"] == 8
    assert summary["total_smoke_calls"] == 80_000
    assert (run / "revalidation_summary.json").is_file()
    assert all(
        (run / "revalidation" / "slots" / f"slot-{index:02d}" / "source.py").is_file()
        for index in range(8)
    )
    assert raw_hashes == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_hashes
    }

    replay = revalidation.replay_saved_revalidation(config, run)
    assert replay["replay_validated"] is True
    assert replay["provider_calls"] == 0
    assert replay["model_calls"] == 0
