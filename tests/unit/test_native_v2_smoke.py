from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from mutation_forge.experiment.config import load_experiment_config


def _load_smoke_module() -> ModuleType:
    path = Path("scripts/native_v2_smoke.py")
    spec = importlib.util.spec_from_file_location("native_v2_smoke", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_config_is_bounded_and_uses_native_v2(tmp_path: Path) -> None:
    smoke = _load_smoke_module()
    config_path = tmp_path / "smoke.toml"
    config_path.write_text(
        smoke.render_config(exp_id="native-v2-smoke-test", workspace=tmp_path),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.schema_version == "mforge.experiment.v2"
    assert config.kind == "heg"
    assert config.preset == "native"
    assert config.model.provider == "codex"
    assert config.model.concurrency == 1
    assert config.model.max_repairs == 1
    assert config.search.population_size == 1
    assert config.search.max_generations == 1
    assert config.search.max_model_turns == 2
    assert config.evaluation.orders == (4,)
    assert config.evaluation.graph_seeds == (401,)
    assert config.evaluation.policy_seeds == (4001,)
    assert config.evaluation.horizon == 1
    assert config.resources.workers == 1
    assert config.resources.thread_count == 1


def test_provider_artifact_snapshot_detects_changes(tmp_path: Path) -> None:
    smoke = _load_smoke_module()
    turn = (
        tmp_path
        / "artifacts"
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "initial"
    )
    turn.mkdir(parents=True)
    artifact = turn / "slot-00.response.json.gz"
    artifact.write_bytes(b"first")

    first = smoke.provider_artifact_snapshot(tmp_path)
    artifact.write_bytes(b"second")
    second = smoke.provider_artifact_snapshot(tmp_path)

    relative = "generation-0000/slot-00/initial/slot-00.response.json.gz"
    assert list(first) == [relative]
    assert first[relative] != second[relative]


def test_fresh_experiment_id_is_safe() -> None:
    smoke = _load_smoke_module()

    exp_id = smoke.fresh_experiment_id(datetime(2026, 8, 5, 12, 30, tzinfo=UTC))

    assert exp_id.startswith("native-v2-smoke-20260805T123000Z-")
    assert "/" not in exp_id
    assert len(exp_id.rsplit("-", 1)[1]) == 8
