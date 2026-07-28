from __future__ import annotations

from pathlib import Path

import pytest

from mutation_forge.config import load_config


def test_stage1_smoke_config_loads(project_root: Path) -> None:
    config = load_config(project_root / "configs" / "stage1-smoke.toml")
    assert config.schema_version == "1.0"
    assert config.dataset.orders == (30,)
    assert config.dataset.graph_seeds == (101, 102, 103, 104)
    assert config.proposals.k_values == (2,)
    assert config.search.profiling_enabled
    assert len(config.stable_hash()) == 64


def test_stage1_rejects_k_switch(tmp_path: Path, project_root: Path) -> None:
    source = (project_root / "configs" / "stage1-smoke.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("k_values = [2]", "k_values = [2, 3]"))
    with pytest.raises(ValueError, match="k_values"):
        load_config(path)


def test_stage1_rejects_unknown_operator(tmp_path: Path, project_root: Path) -> None:
    source = (project_root / "configs" / "stage1-smoke.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("heg_uniform_two_switch", "generated_ranker"))
    with pytest.raises(ValueError, match="unsupported"):
        load_config(path)


def test_stage1_defaults_profiling_on_for_legacy_config(
    tmp_path: Path, project_root: Path
) -> None:
    source = (project_root / "configs" / "stage1-smoke.toml").read_text()
    path = tmp_path / "legacy.toml"
    path.write_text(source.replace("profiling_enabled = true\n", ""))
    assert load_config(path).search.profiling_enabled


def test_stage1_rejects_non_boolean_profiling(
    tmp_path: Path, project_root: Path
) -> None:
    source = (project_root / "configs" / "stage1-smoke.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(source.replace("profiling_enabled = true", "profiling_enabled = 1"))
    with pytest.raises(ValueError, match="profiling_enabled"):
        load_config(path)
