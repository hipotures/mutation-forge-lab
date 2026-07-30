import json
import re
from pathlib import Path

from mutation_forge.stage4.config import load_stage4_config
from mutation_forge.stage4.contracts import load_seed_capsule
from mutation_forge.stage4.prompts import render_repair_prompt, render_system_prompt


def test_stage4_frozen_config_and_capsule() -> None:
    config = load_stage4_config("configs/stage4-search.toml")
    assert config.model.name == "gpt-5.6-luna"
    assert config.model.generations == 4
    assert config.experiment.episode_count == 128
    assert len(load_seed_capsule()) == 8


def test_repair_prompt_excludes_feedback() -> None:
    prompt = render_repair_prompt(diagnostics={"schema": "bad", "performance": "omit"})
    assert "performance" not in prompt
    assert "validation-set" in prompt
    assert "stage4.generated_policy.v1" in render_system_prompt()


def test_generated_schema_avoids_unsupported_unique_items_keyword() -> None:
    schema = json.loads(Path("configs/schemas/stage4-generated-policy.schema.json").read_text())
    assert "uniqueItems" not in json.dumps(schema)


def test_checked_in_manifests_match_frozen_shape() -> None:
    for name, split, held_out, graph_start, policy_start in (
        ("stage4-search-v1.json", "search", False, 401, 4001),
        ("stage4-validation-v1.json", "validation", True, 451, 4501),
    ):
        manifest = json.loads(Path(f"configs/manifests/{name}").read_text())
        assert manifest["schema_version"] == "stage4.manifest.v1"
        assert manifest["split"] == split and manifest["held_out"] is held_out
        assert manifest["dataset"] == "retained-stage2d-toy-trajectories"
        assert manifest["orders"] == [10, 12]
        assert manifest["graph_seeds"] == list(range(graph_start, graph_start + 4))
        assert manifest["policy_seeds"] == list(range(policy_start, policy_start + 16))
        assert manifest["horizon"] == 32 and manifest["episode_count"] == 128
        assert manifest["shard_count"] == 8 and manifest["episodes_per_shard"] == 16
        assert len(manifest["episodes"]) == 128
        for episode in manifest["episodes"]:
            assert set(episode) == {
                "episode_id", "order", "graph_seed", "policy_seed", "horizon", "shard_id"
            }
            assert re.fullmatch(r"o\d{2}-g\d{4}-p\d{4}", episode["episode_id"])
            assert re.fullmatch(r"shard-[0-7]{2}", episode["shard_id"])
