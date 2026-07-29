from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from mutation_forge.stage3.config import load_stage3_config
from mutation_forge.stage3.contracts import (
    GeneratedPolicy,
    Stage3ContractError,
    parse_generated_policy,
    parse_stage2b_inputs,
)
from mutation_forge.stage3.prompts import load_prompt_bundle, schema_field_descriptions


def _output() -> dict[str, object]:
    return {
        "schema_version": "stage3.generated_policy.v1",
        "source": "def priority(ctx, proposal):\n    return 0.0",
        "design_summary": "A finite deterministic baseline.",
        "used_fields": ["proposal.k"],
        "assumptions": ["The host provides legal proposals."],
    }


def test_generated_policy_is_strict_and_immutable() -> None:
    result = parse_generated_policy(_output())
    assert isinstance(result, GeneratedPolicy)
    assert result.used_fields == ("proposal.k",)
    with pytest.raises(AttributeError):
        result.source = "changed"  # type: ignore[misc]
    with pytest.raises(Stage3ContractError):
        parse_generated_policy({**_output(), "extra": True})
    with pytest.raises(Stage3ContractError, match="undocumented"):
        parse_generated_policy({**_output(), "used_fields": ["ctx.not_a_field"]})


def test_stage2b_schema_versions_are_required() -> None:
    with pytest.raises(Stage3ContractError, match="schema_version"):
        parse_stage2b_inputs({}, {})


def test_config_manifest_and_prompt_bundle_are_frozen(project_root: Path) -> None:
    config = load_stage3_config(project_root / "configs/stage3-generation.toml")
    assert config.model.slots == tuple(f"slot-{i:02d}" for i in range(8))
    assert config.model.max_repairs == 1
    assert config.experiment.episode_count == 128
    assert config.experiment.episodes_per_shard == 16
    assert config.episode_shard(10, 301, 3001) == 0
    assert config.episode_shard(12, 304, 3016) == 7
    bundle = load_prompt_bundle()
    assert bundle.system == config.system_prompt_path.read_text().rstrip("\n")
    assert bundle.request == config.request_prompt_path.read_text().rstrip("\n")
    assert bundle.output_schema == config.output_schema_path.read_text()
    assert "forbidden_lengths" in bundle.system
    assert "proposal_id" in schema_field_descriptions()
    assert "slot-00" in bundle.render_slot_request("slot-00", "sparse linear")


def test_development_seeds_are_disjoint_from_stage2c_and_stage2d(project_root: Path) -> None:
    stage3 = load_stage3_config(project_root / "configs/stage3-generation.toml")
    prior_seeds: set[int] = set()
    for name, table in (
        ("stage2c-diagnostic.toml", "matrix"),
        ("stage2d-preregistered.toml", "experiment"),
    ):
        with (project_root / "configs" / name).open("rb") as handle:
            experiment = tomllib.load(handle)[table]
        prior_seeds.update(experiment["graph_seeds"])
        prior_seeds.update(experiment["policy_seeds"])
    assert set(stage3.experiment.graph_seeds).isdisjoint(prior_seeds)
    assert set(stage3.experiment.policy_seeds).isdisjoint(prior_seeds)
