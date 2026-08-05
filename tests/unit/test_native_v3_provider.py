from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from mutation_forge.native_v3.contracts import ValidatedProgram, validate_program
from mutation_forge.native_v3.provider import (
    FrozenProviderRequest,
    NativeV3Provider,
    ProviderContractError,
    ProviderInputProfile,
    ProviderSlotSpec,
    build_provider_request,
)
from mutation_forge.native_v3.scheduler import EpochSnapshot, ProviderCall


def _raw_program(reason: str = "EXPLICIT") -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.program.v3",
            "entry": {"op": "no_plan", "reason": reason},
        },
        separators=(",", ":"),
    )


def _program() -> ValidatedProgram:
    result = validate_program(_raw_program())
    assert result.program is not None
    return result.program


def _call(slot_ids: tuple[str, ...] = ("slot-00", "slot-01")) -> ProviderCall:
    snapshot = EpochSnapshot(
        "epoch",
        1,
        ("parent",),
        "archive",
        "development",
        "protocol",
        slot_ids,
    )
    return ProviderCall("call", slot_ids, snapshot)


def test_request_profile_includes_no_more_than_four_complete_parent_asts() -> None:
    call = _call()
    parents = {f"parent-{index}": _program() for index in range(5)}
    slots = (
        ProviderSlotSpec("slot-00", ("parent-0", "parent-1"), "brief 0"),
        ProviderSlotSpec("slot-01", ("parent-2", "parent-3"), "brief 1"),
    )
    request = build_provider_request(
        call=call,
        slots=slots,
        parent_programs=parents,
        archive_summary={"parent-4": {"fitness_lower": 1}},
        system_prompt="system",
        output_schema={"type": "object"},
    )
    prompt = json.loads(str(request.request["prompt"]))
    assert len(prompt["complete_parent_programs"]) == 4
    assert request.conservative_token_bound <= 32 * 1024
    assert len(request.encoded_bytes) <= 128 * 1024


def test_provider_transport_artifact_path_is_part_of_frozen_request() -> None:
    call = _call(("slot-00",))
    request = build_provider_request(
        call=call,
        slots=(ProviderSlotSpec("slot-00", (), "brief"),),
        parent_programs={},
        archive_summary={},
        system_prompt="system",
        output_schema={"type": "object"},
        artifact_dir="/workspace/artifacts/provider-v3/epoch/call/transport",
        artifact_prefix="initial",
    )

    assert request.request["artifact_dir"] == (
        "/workspace/artifacts/provider-v3/epoch/call/transport"
    )
    assert request.request["artifact_prefix"] == "initial"
    assert b'"artifact_prefix":"initial"' in request.encoded_bytes


def test_request_is_rejected_instead_of_silently_changing_parent_set() -> None:
    call = _call(("slot-00",))
    with pytest.raises(ProviderContractError, match="too many parent"):
        build_provider_request(
            call=call,
            slots=(
                ProviderSlotSpec(
                    "slot-00",
                    ("parent-0", "parent-1", "parent-2"),
                    "brief",
                ),
            ),
            parent_programs={f"parent-{index}": _program() for index in range(3)},
            archive_summary={},
            system_prompt="system",
            output_schema={"type": "object"},
        )


class _FakeProvider:
    def __init__(self, response: Mapping[str, Any], repair: Mapping[str, Any] | None = None):
        self.response = response
        self.repair_response = repair
        self.repair_calls = 0

    def generate(self, _request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.response

    def repair(
        self,
        _request: Mapping[str, Any],
        _diagnostics: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        self.repair_calls += 1
        assert self.repair_response is not None
        return self.repair_response


def _frozen(call: ProviderCall) -> FrozenProviderRequest:
    return FrozenProviderRequest(call, {"prompt": "prompt"}, b"{}", 1)


def _response(programs: list[dict[str, str]]) -> Mapping[str, Any]:
    raw = json.dumps({"schema_version": "mforge.native.program_batch.v3", "programs": programs})
    return {"response_text": raw, "usage": {"outputTokens": 10}}


def test_valid_entry_starts_independently_of_invalid_batch_sibling() -> None:
    call = _call()
    provider = _FakeProvider(
        _response(
            [
                {
                    "slot_id": "slot-00",
                    "program_json_raw": _raw_program(),
                    "design_summary": "valid",
                },
                {
                    "slot_id": "slot-01",
                    "program_json_raw": "{",
                    "design_summary": "invalid",
                },
            ]
        )
    )
    artifacts = []
    batch = NativeV3Provider(
        provider,
        request_factory=_frozen,
        artifact_sink=artifacts.append,
    )(call)
    assert batch.entries[0].program is not None
    assert batch.entries[1].error is not None
    assert provider.repair_calls == 0
    assert artifacts[0].raw_response


def test_streamed_entry_is_published_only_after_raw_response_is_retained() -> None:
    call = _call()
    provider = _FakeProvider(
        _response(
            [
                {
                    "slot_id": slot,
                    "program_json_raw": _raw_program(),
                    "design_summary": "valid",
                }
                for slot in call.slot_ids
            ]
        )
    )
    observed: list[str] = []
    batch = NativeV3Provider(
        provider,
        request_factory=_frozen,
        raw_artifact_sink=lambda _artifact: observed.append("raw"),
    ).call_streaming(
        call,
        lambda entry: observed.append(entry.slot_id),
    )
    assert all(entry.program is not None for entry in batch.entries)
    assert observed == ["raw", "slot-00", "slot-01"]


def test_one_repair_reuses_the_frozen_slots_only_for_fully_invalid_batch() -> None:
    call = _call()
    provider = _FakeProvider(
        _response(
            [
                {
                    "slot_id": slot,
                    "program_json_raw": "{",
                    "design_summary": "invalid",
                }
                for slot in call.slot_ids
            ]
        ),
        _response(
            [
                {
                    "slot_id": slot,
                    "program_json_raw": _raw_program(),
                    "design_summary": "repaired",
                }
                for slot in call.slot_ids
            ]
        ),
    )
    batch = NativeV3Provider(provider, request_factory=_frozen)(call)
    assert provider.repair_calls == 1
    assert tuple(entry.slot_id for entry in batch.entries) == call.slot_ids
    assert all(entry.program is not None for entry in batch.entries)


def test_actual_request_preflight_enforces_token_ceiling() -> None:
    call = _call(("slot-00",))
    with pytest.raises(ProviderContractError, match="tokens"):
        build_provider_request(
            call=call,
            slots=(ProviderSlotSpec("slot-00", (), "brief"),),
            parent_programs={},
            archive_summary={},
            system_prompt="system",
            output_schema={"type": "object"},
            input_profile=ProviderInputProfile(maximum_request_tokens=1),
            token_counter=lambda _encoded: 2,
        )
