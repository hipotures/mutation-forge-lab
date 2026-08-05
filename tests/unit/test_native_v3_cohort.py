from __future__ import annotations

import json

import pytest

from mutation_forge.native_v3.cohort import (
    PROVIDER_PARTITION,
    SLOT_IDS,
    build_epoch_manifest,
    cohort_outcome,
    deduplicate_entries,
    parse_batch_response,
    render_batch_prompt,
)


def _program(index: int) -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.program.v3",
            "entry": {
                "op": "if",
                "condition": {
                    "op": "less",
                    "left": {"op": "feature", "field": "order"},
                    "right": 30 + index,
                },
                "then": {"op": "no_plan", "reason": "NO_MATCH"},
                "else": {"op": "no_plan", "reason": "EXPLICIT"},
            },
        },
        separators=(",", ":"),
    )


def _response(items: list[dict[str, str]]) -> str:
    batch = json.dumps(
        {
            "schema_version": "mforge.native.program_batch.v3",
            "programs": items,
        },
        separators=(",", ":"),
    )
    return json.dumps(
        {
            "schema_version": "mforge.native.generated_policy.v1",
            "source": batch,
            "design_summary": "Four independent mutation programs.",
            "hypothesis": "At least one mechanism improves the graph.",
            "used_fields": [],
            "assumptions": [],
            "expected_failure_modes": [],
        },
        separators=(",", ":"),
    )


def _item(slot_id: str, index: int) -> dict[str, str]:
    return {
        "slot_id": slot_id,
        "program_json_raw": _program(index),
        "design_summary": f"Mechanism {index}.",
    }


def test_response_order_does_not_change_programs_or_lineage() -> None:
    slots = PROVIDER_PARTITION[0]
    ordered = [_item(slot_id, index) for index, slot_id in enumerate(slots)]
    forward = parse_batch_response(_response(ordered), slots)
    reverse = parse_batch_response(_response(list(reversed(ordered))), slots)

    assert [entry.slot_id for entry in forward.entries] == list(slots)
    assert [entry.as_dict() for entry in reverse.entries] == [
        entry.as_dict() for entry in forward.entries
    ]
    assert deduplicate_entries(reverse.entries) == deduplicate_entries(
        forward.entries
    )


def test_partial_invalidity_keeps_valid_siblings() -> None:
    slots = PROVIDER_PARTITION[0]
    items = [_item(slot_id, index) for index, slot_id in enumerate(slots)]
    items[2]["program_json_raw"] = "{}"

    parsed = parse_batch_response(_response(items), slots)

    assert [entry.program is not None for entry in parsed.entries] == [
        True,
        True,
        False,
        True,
    ]
    assert parsed.entries[2].error is not None


def test_duplicate_program_retains_aliases_and_counts_once() -> None:
    slots = PROVIDER_PARTITION[0]
    items = [_item(slot_id, index) for index, slot_id in enumerate(slots)]
    items[1]["program_json_raw"] = items[0]["program_json_raw"]
    parsed = parse_batch_response(_response(items), slots)

    programs, aliases = deduplicate_entries(parsed.entries)

    assert len(programs) == 3
    duplicate_hash = parsed.entries[0].program
    assert duplicate_hash is not None
    assert aliases[duplicate_hash.program_hash] == ("slot-00", "slot-01")


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "INCONCLUSIVE"),
        (3, "INCONCLUSIVE"),
        (4, "DEGRADED"),
        (7, "DEGRADED"),
        (8, "COMPLETE"),
    ],
)
def test_exact_cohort_thresholds(count: int, expected: str) -> None:
    assert cohort_outcome(count) == expected


def test_manifest_freezes_eight_slots_two_calls_and_prompt_hashes() -> None:
    manifest = build_epoch_manifest(model="gpt-5.6-luna", effort="high")

    assert manifest["planned_slot_ids"] == list(SLOT_IDS)
    assert [call["slot_ids"] for call in manifest["provider_calls"]] == [
        list(PROVIDER_PARTITION[0]),
        list(PROVIDER_PARTITION[1]),
    ]
    assert all(slot["parent_program_hashes"] == [] for slot in manifest["slots"])
    assert all(len(call["prompt_sha256"]) == 64 for call in manifest["provider_calls"])


def test_model_prompt_contains_semantics_but_not_host_bookkeeping() -> None:
    prompt = render_batch_prompt(PROVIDER_PARTITION[0])

    assert "add_edge" in prompt
    assert "non_edges_local_cycle_risk" in prompt
    assert "slot-00" in prompt
    assert "proposal_id" not in prompt
    assert "sha256" not in prompt.lower()
    assert '"usage"' not in prompt
