from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mutation_forge.stage3.config import load_stage3_config
from mutation_forge.stage3.prompts import (
    load_prompt_bundle,
    load_semantic_glossary,
    render_request_prompt,
    render_system_prompt,
    schema_field_descriptions,
)

EXPECTED_SLOT_PROMPT_SHA256 = {
    "slot-00": "3e9a27d72f6047f4438e2480f2688e8e6d8938da2b1ba26e81a4f5582602afc8",
    "slot-01": "417e6127300b8022fe9a48713e1f440b92a4aa0c0bcfcbcd112d22f37cac342a",
    "slot-02": "f7a0366daf8130d9d4e99e82b5cc8ff7f47adceed32851ef35b113f531501d70",
    "slot-03": "8f2875bde98a7be004f1fc34b9dfbb91764f637d73601ffd2b1b2dbf03ebe134",
    "slot-04": "7475f6a3e5bea9ecb6c9297c0f0d2864745485d6e9b0329dc382d0f14f354245",
    "slot-05": "25cf310142b3d563b916622e6b875b602381f3840af48f50dfb4d6e3b3762e4a",
    "slot-06": "05b47b52cec8b264bf6e2907cb466c44bc24d01acf9146a3e777644f97fac867",
    "slot-07": "91ec153bcd2c427147673fc2e2d0ed42b4774094c71b2dc7a5fc9db5e9fc9175",
}


def test_checked_in_prompts_match_schema_renderer() -> None:
    assert Path("prompts/ranker_v1_system.md").read_text(encoding="utf-8").rstrip(
        "\n"
    ) == render_system_prompt()
    assert Path("prompts/ranker_v1_request.md").read_text(encoding="utf-8").rstrip(
        "\n"
    ) == render_request_prompt()


def test_all_frozen_slot_prompts_are_complete_and_snapshotted() -> None:
    config = load_stage3_config("configs/stage3-generation.toml")
    bundle = load_prompt_bundle(
        context_schema=config.context_schema_path,
        proposal_schema=config.proposal_schema_path,
        output_schema=config.output_schema_path,
    )
    observed: dict[str, str] = {}
    for path in sorted(config.slot_briefs_dir.glob("slot-*.json")):
        brief = json.loads(path.read_text(encoding="utf-8"))
        prompt = bundle.render_slot_request(
            brief["slot_id"],
            brief["brief"],
            generation_mode=brief["generation_mode"],
            focus=brief["focus"],
        )
        assert "$ref" not in prompt
        assert ":  (required)" not in prompt
        assert "contract above" not in prompt
        assert "{generation_mode}" not in prompt
        assert "{task_instruction}" not in prompt
        assert "def priority(ctx, proposal):" in prompt
        assert 'design_summary: begin with "Hypothesis:"' in prompt
        assert "ctx describes the current graph and is identical for every proposal" in prompt
        assert "Only the selected proposal is applied and authoritatively scored" in prompt
        assert "operator_family is exactly legal_{k}_switch" in prompt
        assert "the two fields are computed from the same arithmetic mean" in prompt
        assert "Allowed built-ins only: abs, all, any, len, max, min, range, round, sum" in prompt
        assert "No imports, attributes or method calls, comprehensions" in prompt
        assert "proposal.anchor_forbidden_length [null or integer; minimum 1;" in prompt
        assert (
            "proposal.broken_sampled_witnesses_by_length "
            "[array (1..16 items) of integer; minimum 0;"
        ) in prompt
        assert (
            Path("prompts/stage3-slots") / f"{brief['slot_id']}.md"
        ).read_text(encoding="utf-8") == prompt + "\n"
        observed[brief["slot_id"]] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert observed == EXPECTED_SLOT_PROMPT_SHA256


def test_semantic_glossary_is_exactly_schema_bound(tmp_path: Path) -> None:
    glossary = load_semantic_glossary()
    assert set(glossary["fields"]["ctx"]) == {
        "schema_version",
        "order",
        "forbidden_lengths",
        "capped_cycle_counts",
        "weighted_penalty",
        "step",
        "remaining_steps",
        "stagnation",
        "recent_best_improvement",
        "recent_acceptance_rate",
        "recent_duplicate_rate",
    }
    assert set(glossary["fields"]["proposal"]) == {
        "schema_version",
        "proposal_id",
        "k",
        "operator_family",
        "selector_tags",
        "anchor_forbidden_length",
        "broken_sampled_witnesses_by_length",
        "removed_edge_load_sum_by_length",
        "removed_edge_load_max_by_length",
        "minimum_distance_between_removed_edges",
        "mean_distance_between_removed_edges",
        "minimum_preexisting_distance_for_new_edges",
        "mean_preexisting_distance_for_new_edges",
        "local_triangle_risk",
        "local_c4_risk",
        "reconnection_span",
    }
    broken = json.loads(json.dumps(glossary))
    del broken["fields"]["proposal"]["local_c4_risk"]
    path = tmp_path / "broken-glossary.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover proposal"):
        load_semantic_glossary(path)


def test_semantic_glossary_rejects_wrong_schema_hash_and_extra_field(
    tmp_path: Path,
) -> None:
    glossary = load_semantic_glossary()
    wrong_hash = json.loads(json.dumps(glossary))
    wrong_hash["proposal_schema_sha256"] = "0" * 64
    wrong_hash_path = tmp_path / "wrong-hash.json"
    wrong_hash_path.write_text(json.dumps(wrong_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="schema identity"):
        load_semantic_glossary(wrong_hash_path)

    extra_field = json.loads(json.dumps(glossary))
    extra_field["fields"]["ctx"]["nonexistent"] = {
        "scope": "pool_constant",
        "semantics": "Not a real schema field.",
        "direction": "neutral",
    }
    extra_field_path = tmp_path / "extra-field.json"
    extra_field_path.write_text(json.dumps(extra_field), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover ctx"):
        load_semantic_glossary(extra_field_path)


def test_schema_renderer_rejects_unresolved_and_cyclic_local_refs(tmp_path: Path) -> None:
    context = json.loads(
        Path("configs/schemas/stage2b-context.schema.json").read_text(encoding="utf-8")
    )
    proposal = json.loads(
        Path("configs/schemas/stage2b-proposal.schema.json").read_text(encoding="utf-8")
    )
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")

    unresolved = json.loads(json.dumps(proposal))
    unresolved["properties"]["broken_sampled_witnesses_by_length"] = {
        "$ref": "#/$defs/missing"
    }
    unresolved_path = tmp_path / "unresolved.json"
    unresolved_path.write_text(json.dumps(unresolved), encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved schema reference"):
        schema_field_descriptions(context_path, unresolved_path)

    cyclic = json.loads(json.dumps(proposal))
    cyclic["$defs"]["first"] = {"$ref": "#/$defs/second"}
    cyclic["$defs"]["second"] = {"$ref": "#/$defs/first"}
    cyclic["properties"]["broken_sampled_witnesses_by_length"] = {
        "$ref": "#/$defs/first"
    }
    cyclic_path = tmp_path / "cyclic.json"
    cyclic_path.write_text(json.dumps(cyclic), encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic schema reference"):
        schema_field_descriptions(context_path, cyclic_path)


def test_prompt_contract_does_not_request_unsupported_or_undeclared_output() -> None:
    prompt = render_request_prompt()
    assert "zip" not in prompt
    assert "enumerate" not in prompt
    assert "proposal.get" not in prompt
    assert "math." not in prompt
    assert '"hypothesis"' not in prompt
    assert "design_summary: begin with \"Hypothesis:\"" in prompt
    assert "exactly one return statement" in prompt
    assert "final top-level statement" in prompt
