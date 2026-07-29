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
    assert not config.search.deep_profiling_enabled
    assert config.search.score_cache_enabled
    assert config.search.score_cutoff_enabled
    assert config.search.prepared_graph_cache_enabled
    assert config.search.prepared_proposal_handoff_enabled
    assert config.search.score_longest_first_enabled
    assert config.search.score_compact_dominated_enabled
    assert config.search.score_prepared_request_cache_enabled
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


def test_stage1_defaults_deep_profiling_off_for_legacy_config(
    tmp_path: Path, project_root: Path
) -> None:
    source = (project_root / "configs" / "stage1-smoke.toml").read_text()
    path = tmp_path / "legacy.toml"
    path.write_text(source.replace("deep_profiling_enabled = false\n", ""))
    assert not load_config(path).search.deep_profiling_enabled


def test_stage1_rejects_non_boolean_deep_profiling(
    tmp_path: Path, project_root: Path
) -> None:
    source = (project_root / "configs" / "stage1-smoke.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(
        source.replace(
            "deep_profiling_enabled = false",
            "deep_profiling_enabled = 1",
        )
    )
    with pytest.raises(ValueError, match="deep_profiling_enabled"):
        load_config(path)


@pytest.mark.parametrize(
    "field",
    [
        "score_cache_enabled",
        "score_cutoff_enabled",
        "prepared_graph_cache_enabled",
        "prepared_proposal_handoff_enabled",
        "score_longest_first_enabled",
        "score_compact_dominated_enabled",
        "score_prepared_request_cache_enabled",
    ],
)
def test_stage1_defaults_scoring_optimizations_on_and_rejects_non_boolean(
    field: str,
    tmp_path: Path,
    project_root: Path,
) -> None:
    source = (project_root / "configs" / "stage1-smoke.toml").read_text()
    line = f"{field} = true\n"
    legacy_path = tmp_path / f"legacy-{field}.toml"
    legacy_path.write_text(source.replace(line, ""))
    assert getattr(load_config(legacy_path).search, field)

    invalid_path = tmp_path / f"invalid-{field}.toml"
    invalid_path.write_text(source.replace(line, f"{field} = 1\n"))
    with pytest.raises(ValueError, match=field):
        load_config(invalid_path)
