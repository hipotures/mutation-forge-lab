from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from mutation_forge.native_v3_python.contracts import (
    PYTHON_EXPERIMENT_PROTOCOL_ID,
)
from mutation_forge.native_v3_python.preview import (
    V2_PROTOCOL,
    experiment_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
REMOVED_MODULES = (
    "contracts",
    "graph_runtime",
    "interpreter",
    "single_program_ir",
    "single_program_contract",
    "cohort",
    "persistent_experiment",
    "preview",
    "lineage_experiment",
    "compaction_experiment",
    "search_memory",
    "provider_smoke",
    "provider_evaluation",
    "experiment",
)
REMOVED_ASSETS = (
    "configs/native/native-v3-program.schema.json",
    "configs/native/native-v3-provider-envelope.schema.json",
    "configs/native/native-v3-cohort-envelope.schema.json",
    "configs/native/native-v3-semantics.md",
    "prompts/native-v3",
)


def test_superseded_json_dsl_modules_and_assets_are_absent() -> None:
    for name in REMOVED_MODULES:
        qualified = f"mutation_forge.native_v3.{name}"
        assert importlib.util.find_spec(qualified) is None, qualified
        assert not (ROOT / "src/mutation_forge/native_v3" / f"{name}.py").exists()
    for relative in REMOVED_ASSETS:
        assert not (ROOT / relative).exists(), relative

    assert (
        ROOT / "configs/native/native-v3-python-policy-response.schema.json"
    ).is_file()
    assert (ROOT / "configs/native/generated-policy.schema.json").is_file()
    assert (ROOT / "prompts/native/system.md").is_file()


def test_production_has_no_json_dsl_import_or_runtime_dispatch() -> None:
    forbidden_modules = {
        f"mutation_forge.native_v3.{name}" for name in REMOVED_MODULES
    }
    for path in sorted((ROOT / "src/mutation_forge").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert forbidden_modules.isdisjoint(
                    {alias.name for alias in node.names}
                ), path
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module not in forbidden_modules, path

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src/mutation_forge/native_v3_python").glob("*.py"))
    )
    for token in (
        "invoke_program(",
        "evaluate_serial_program(",
        "compile_program(",
        "native-v3-program-batch",
    ):
        assert token not in source


def test_native_v2_remains_default_and_removed_selector_fails_closed(
    tmp_path: Path,
) -> None:
    default = tmp_path / "default.toml"
    default.write_text('[run]\nexp_id = "native-v2-default"\n', encoding="utf-8")
    assert experiment_protocol(default) == V2_PROTOCOL

    python = tmp_path / "python.toml"
    python.write_text(
        f'protocol = "{PYTHON_EXPERIMENT_PROTOCOL_ID}"\n',
        encoding="utf-8",
    )
    assert experiment_protocol(python) == PYTHON_EXPERIMENT_PROTOCOL_ID

    removed = tmp_path / "removed.toml"
    removed.write_text('protocol = "v3"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="superseded JSON-DSL"):
        experiment_protocol(removed)
