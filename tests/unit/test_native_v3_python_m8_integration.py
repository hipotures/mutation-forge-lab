from __future__ import annotations

from pathlib import Path

import pytest

from mutation_forge.native_v3_python.contracts import (
    PYTHON_EXPERIMENT_PROTOCOL_ID,
)
from mutation_forge.native_v3_python.preview import (
    experiment_protocol,
    load_python_preview_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = (
    PROJECT_ROOT / "configs/examples/native-v3-python-preview-v1.toml"
)


def test_versioned_python_preview_example_is_explicit() -> None:
    config = load_python_preview_config(EXAMPLE_CONFIG)
    assert config.protocol == PYTHON_EXPERIMENT_PROTOCOL_ID
    assert config.exp_id == "native-v3-python-preview-example"
    assert config.workspace == PROJECT_ROOT / "runs/native-v3-python-preview"
    assert config.heg_repo == PROJECT_ROOT.parent / "heg"
    assert config.model == "gpt-5.6-luna"
    assert config.effort == "medium"


def test_removed_json_dsl_selector_has_a_clear_unsupported_error(
    tmp_path: Path,
) -> None:
    config = tmp_path / "removed-json-dsl.toml"
    config.write_text('protocol = "v3"\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="superseded JSON-DSL experiment protocol was removed",
    ):
        experiment_protocol(config)


def test_m8_operator_documents_cover_required_safe_workflows() -> None:
    guide = (
        PROJECT_ROOT / "docs/native-v3-python-operator-guide.md"
    ).read_text(encoding="utf-8")
    for required in (
        "Native v2 is the production default",
        'protocol = "native-v3-python-v1"',
        "def propose(ctx, graph, api, seed):",
        "--request-stop",
        "experiment status",
        "provider_failed",
        "PROGRAM_FAILURE",
        "evaluation_infrastructure_failure",
        "exact_verifier_only",
        "Old JSON-DSL",
    ):
        assert required in guide

    report = (
        PROJECT_ROOT / "docs/native-v3-python-integration-report.md"
    ).read_text(encoding="utf-8")
    assert "daf7ab5c95a36c29842e6703705a381080edcfd5" in report
    assert "728ea5f222c72184a9bca8ef69dfea90121e6fd4" in report
    assert "`experiment.toml`" in report

    roadmap = (
        PROJECT_ROOT / "docs/native-v3-python-post-migration-roadmap.md"
    ).read_text(encoding="utf-8")
    for issue in range(35, 43):
        assert f"# {issue}" not in roadmap
        assert f"| #{issue} |" in roadmap
