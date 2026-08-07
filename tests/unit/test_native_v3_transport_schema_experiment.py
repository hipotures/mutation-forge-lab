from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from mutation_forge.native_v3.persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    BOOTSTRAP_ACK_VALUE,
)
from mutation_forge.native_v3.single_program_ir import (
    BRIEF_OPERATORS,
    FLAT_IR_SCHEMA_VERSION,
    SLOT_SPECIFIC_SCHEMA_VERSION,
)
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
)

_SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "native_v3_transport_schema_experiment.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "native_v3_transport_schema_experiment",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = _SCRIPT
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)

_FAKE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_FAKE_SPEC = importlib.util.spec_from_file_location(
    "step12e0_fake_app_server",
    _FAKE_PATH,
)
assert _FAKE_SPEC is not None and _FAKE_SPEC.loader is not None
_FAKE = importlib.util.module_from_spec(_FAKE_SPEC)
sys.modules[_FAKE_SPEC.name] = _FAKE
_FAKE_SPEC.loader.exec_module(_FAKE)
FakeProcess = _FAKE.FakeProcess
FakeScenario = _FAKE.FakeScenario

FORBIDDEN_LENGTHS = (4, 8, 16)
SELECTOR_ARGUMENTS = {
    "add-edge": {"mode": "min"},
    "remove-edge": {"mode": "min"},
    "relocation": {},
    "fanout": {},
}
ACTION_ARGUMENTS = {
    "add-edge": "edge",
    "remove-edge": "edge",
    "relocation": "relocation",
    "fanout": "fanout",
}


def _common(
    *,
    schema_version: str,
    slot_id: str,
    brief_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "slot_id": slot_id,
        "brief_id": brief_id,
        "active_forbidden_lengths": list(FORBIDDEN_LENGTHS),
        "design_summary": f"Use the bounded {brief_id} operator family.",
        "hypothesis": "The relation-certified rewrite may improve the graph.",
    }


def _response(candidate: str, brief_id: str, slot_id: str) -> str:
    operators = BRIEF_OPERATORS[brief_id]
    action_argument = ACTION_ARGUMENTS[brief_id]
    common = _common(
        schema_version=(
            SLOT_SPECIFIC_SCHEMA_VERSION
            if candidate == "slot_specific"
            else FLAT_IR_SCHEMA_VERSION
        ),
        slot_id=slot_id,
        brief_id=brief_id,
    )
    if candidate == "slot_specific":
        value = {
            **common,
            "plan": {
                "selector": {
                    "selector_id": operators.selector_id,
                    "arguments": dict(SELECTOR_ARGUMENTS[brief_id]),
                },
                "pick": {"mode": "seeded_uniform"},
                "action": {
                    "action_id": operators.action_id,
                    "arguments": {action_argument: "selected"},
                },
                "terminal": {"kind": "emit"},
            },
        }
    else:
        value = {
            **common,
            "bindings": [
                {
                    "kind": "selector",
                    "id": "candidates",
                    "selector_id": operators.selector_id,
                    "arguments": dict(SELECTOR_ARGUMENTS[brief_id]),
                },
                {
                    "kind": "pick",
                    "id": "selected",
                    "source": "candidates",
                    "mode": "seeded_uniform",
                },
            ],
            "steps": [
                {
                    "op": "apply",
                    "action_id": operators.action_id,
                    "arguments": {action_argument: "selected"},
                }
            ],
            "terminal": {"kind": "emit"},
        }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_fake_lifecycle_gates_max_and_preserves_artifact_parity(
    tmp_path: Path,
) -> None:
    acknowledgement = json.dumps(
        {
            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
            "ack": BOOTSTRAP_ACK_VALUE,
        },
        separators=(",", ":"),
    )

    def factory(
        base_instructions: str,
        candidate: str,
        effort: str,
    ) -> CodexAppServerAdapter:
        briefs = (
            ("add-edge", "remove-edge", "relocation", "fanout")
            if effort == "medium"
            else ("add-edge", "relocation")
        )
        texts = [
            acknowledgement,
            *[
                _response(candidate, brief_id, f"slot-{index:02d}")
                for index, brief_id in enumerate(briefs)
            ],
        ]
        scenario = FakeScenario(
            final_texts=texts,
            thread_id=f"{candidate}-{effort}-thread",
        )
        return CodexAppServerAdapter(
            process_factory=lambda *_args, **kwargs: FakeProcess(scenario, **kwargs),
            auth_checker=lambda _capsule: True,
            limits=AppServerLimits(
                max_turns=5,
                max_campaigns=1,
                usage_grace=0.01,
            ),
            base_instructions=base_instructions,
            compress_json_artifacts=True,
            sandbox_mode="read-only",
            approval_policy="never",
        )

    report = _SCRIPT.run_transport_schema_benchmark(
        tmp_path / "benchmark",
        model="gpt-5.6-luna",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
        run_max=True,
    )

    assert report["max_turn_count"] == 4
    assert report["recommendation"]["decision"] == "GO"
    assert set(report["max"]) == {"slot_specific", "flat_ir"}
    for candidate in ("slot_specific", "flat_ir"):
        medium = report["medium"][candidate]
        maximum = report["max"][candidate]
        assert medium["medium_eligible"] is True
        assert medium["transport_completed"] == 4
        assert medium["schema_conformant"] == 4
        assert medium["compiler_successes"] == 4
        assert medium["semantically_valid"] == 4
        assert medium["artifact_parity_turns"] == 4
        assert medium["bootstrap"]["artifact_parity"] is True
        assert maximum["transport_completed"] == 2
        assert maximum["reconnects"] == 0
        assert maximum["semantically_valid"] == 2
    assert len(report["medium"]["flat_ir"]["repeated_schema_observations"]) == 1
    assert report["repository_status_unchanged"] is True
    assert (tmp_path / "benchmark" / "benchmark-report.json.gz").is_file()
    assert (tmp_path / "benchmark" / "benchmark-report.md").read_text(
        encoding="utf-8"
    ).endswith("STOP — waiting for operator acceptance\n")


def test_medium_failure_blocks_max_without_fallback(tmp_path: Path) -> None:
    acknowledgement = json.dumps(
        {
            "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
            "ack": BOOTSTRAP_ACK_VALUE,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    def factory(
        base_instructions: str,
        candidate: str,
        effort: str,
    ) -> CodexAppServerAdapter:
        briefs = (
            ("add-edge", "remove-edge", "relocation", "fanout")
            if effort == "medium"
            else ("add-edge", "relocation")
        )
        texts = [
            acknowledgement,
            *[
                (
                    '{"invalid":true}'
                    if candidate == "flat_ir" and effort == "medium" and index == 1
                    else _response(candidate, brief_id, f"slot-{index:02d}")
                )
                for index, brief_id in enumerate(briefs)
            ],
        ]
        scenario = FakeScenario(final_texts=texts)
        return CodexAppServerAdapter(
            process_factory=lambda *_args, **kwargs: FakeProcess(scenario, **kwargs),
            auth_checker=lambda _capsule: True,
            limits=AppServerLimits(
                max_turns=5,
                max_campaigns=1,
                usage_grace=0.01,
            ),
            base_instructions=base_instructions,
            compress_json_artifacts=True,
            sandbox_mode="read-only",
            approval_policy="never",
        )

    report = _SCRIPT.run_transport_schema_benchmark(
        tmp_path / "blocked",
        model="gpt-5.6-luna",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        adapter_factory=factory,
        run_max=True,
    )

    assert report["medium"]["flat_ir"]["medium_eligible"] is False
    assert "flat_ir" not in report["max"]
    assert report["max_turn_count"] == 2


def test_event_metrics_use_top_level_server_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "turn.events.jsonl"
    events = [
        {
            "emittedAtMs": 1_000,
            "method": "turn/started",
            "params": {"turn": {"id": "turn-1"}},
        },
        {
            "emittedAtMs": 1_125,
            "method": "item/started",
            "params": {"item": {"id": "reasoning-1", "type": "reasoning"}},
        },
        {
            "emittedAtMs": 1_250,
            "method": "item/agentMessage/delta",
            "params": {"itemId": "message-1", "delta": "x"},
        },
        {
            "emittedAtMs": 1_375,
            "method": "thread/tokenUsage/updated",
            "params": {"tokenUsage": {"last": {}}},
        },
    ]
    path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )

    metrics = _SCRIPT._event_metrics(path)

    assert metrics["turn_start_emitted_at_ms"] == 1_000
    assert metrics["time_to_first_reasoning_item_ms"] == 125
    assert metrics["time_to_first_agent_delta_ms"] == 250
    assert metrics["time_to_first_token_usage_ms"] == 375
