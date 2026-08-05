from __future__ import annotations

from pathlib import Path

import pytest

from mutation_forge.experiment.config import load_experiment_config


def _config(schema: str = "mforge.experiment.v3") -> str:
    return f'''schema_version = "{schema}"
exp_id = "native-v3-test"
workspace = "./workspace"
kind = "heg"
preset = "native"

[run]
wall_seconds = 30

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "high"
concurrency = 2
max_repairs = 1

[search]
population_size = 8
max_generations = 1
max_model_turns = 8
selection = "persistent-elite-weighted-diversity"

[evaluation]
graph_mode = "unrestricted_min_degree_3"
order_schedule = "static"
orders = [8]
graph_seeds = [401]
policy_seeds = [4001]
validation_graph_seeds = [402]
validation_policy_seeds = [4002]
horizon = 4
baselines = [
  "add-low-local-cycle-risk",
  "remove-low-bridge-risk",
  "random-valid",
  "degree-fanout",
]
replay = true

[resources]
workers = 4
thread_count = 4

[native_v3]
provider_batch_size = 4
candidate_queue_capacity = 16
evaluation_queue_capacity = 32
target_evaluation_backlog = 16
candidate_shard_size = 1
auxiliary_shard_size = 1
witness_cap = 64
'''


def test_native_v3_config_locks_scheduler_and_disjoint_panels(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(), encoding="utf-8")
    config = load_experiment_config(path)
    assert config.native_v3.provider_batch_size == 4
    assert config.native_v3.target_evaluation_backlog == 16
    assert set(config.evaluation.graph_seeds).isdisjoint(config.evaluation.validation_graph_seeds)


def test_native_v2_config_is_rejected_instead_of_migrated(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config("mforge.experiment.v2"), encoding="utf-8")
    with pytest.raises(ValueError, match="accepts only mforge.experiment.v3"):
        load_experiment_config(path)


def test_development_and_validation_seed_overlap_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        _config().replace(
            "validation_graph_seeds = [402]",
            "validation_graph_seeds = [401]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        load_experiment_config(path)
