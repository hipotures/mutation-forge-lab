from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.validation import validate_policy


@pytest.mark.parametrize(
    "name",
    ["constant.py", "weighted.py", "bounded_loop.py", "conditional.py"],
)
def test_reviewed_rankers_validate(project_root: Path, name: str) -> None:
    source = (project_root / "fixtures" / "rankers" / name).read_text()
    result = validate_policy(source)
    assert result.valid, result.as_dict()
    assert result.identity.ast_node_count > 0
    assert result.identity.normalized_ast_sha256 is not None


@pytest.mark.parametrize(
    "name",
    [
        "import_os.py",
        "file_access.py",
        "environment_access.py",
        "process_access.py",
        "network_access.py",
        "random_access.py",
        "dunder_reflection.py",
        "dynamic_execution.py",
        "infinite_loop.py",
        "recursion.py",
        "wrong_signature.py",
        "multiple_functions.py",
        "hidden_state.py",
        "input_mutation.py",
        "output_print.py",
    ],
)
def test_static_adversarial_rankers_are_rejected(
    project_root: Path,
    name: str,
) -> None:
    source = (project_root / "fixtures" / "adversarial" / name).read_text()
    result = validate_policy(source)
    assert not result.valid
    assert result.errors
    assert any(error.line is not None for error in result.errors)


def test_source_and_ast_limits_are_structured() -> None:
    source = "def priority(ctx, proposal):\n    return 0\n"
    source_result = validate_policy(source, SandboxLimits(max_source_bytes=8))
    assert source_result.errors[0].code == "source_too_large"
    node_result = validate_policy(source, SandboxLimits(max_ast_nodes=2))
    assert any(error.code == "ast_too_large" for error in node_result.errors)


def test_versioned_probe_json_schema_is_well_formed(project_root: Path) -> None:
    schema = json.loads(
        (
            project_root
            / "configs"
            / "schemas"
            / "stage2a-probe.schema.json"
        ).read_text()
    )
    assert schema["x-runtime-bounds"]["schema_version"] == "stage2a.probe.v1"
    assert schema["x-runtime-bounds"]["max_request_bytes"] == 65536


def test_static_loop_bound_is_enforced() -> None:
    source = (
        "def priority(ctx, proposal):\n"
        "    total = 0\n"
        "    for value in range(257):\n"
        "        total += value\n"
        "    return total\n"
    )
    result = validate_policy(source)
    assert not result.valid
    assert any(error.code == "loop_bound" for error in result.errors)


@pytest.mark.parametrize(
    "name",
    [
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "print",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
    ],
)
def test_forbidden_dynamic_and_reflection_builtins_are_rejected(name: str) -> None:
    result = validate_policy(
        f"def priority(ctx, proposal):\n    return {name}('x')\n"
    )
    assert not result.valid
    assert any(
        error.code in {"forbidden_call", "private_name"} for error in result.errors
    )


@pytest.mark.parametrize(
    "body",
    [
        "    return lambda: 1\n",
        "    yield 1\n    return 1\n",
        "    try:\n        value = 1\n    except Exception:\n        value = 0\n"
        "    return value\n",
        "    with ctx:\n        value = 1\n    return value\n",
        "    def nested(ctx, proposal):\n        return 1\n    return 1\n",
    ],
)
def test_forbidden_control_and_nested_syntax_is_rejected(body: str) -> None:
    result = validate_policy(f"def priority(ctx, proposal):\n{body}")
    assert not result.valid
    assert result.errors


def test_normalized_identity_ignores_formatting_and_local_names() -> None:
    left = (
        "def priority(ctx, proposal):\n"
        "    total = proposal['features']['weight'] + 1\n"
        "    return total\n"
    )
    right = (
        "def priority(ctx, proposal):\n\n"
        "    renamed=proposal[\"features\"][\"weight\"]+1\n"
        "    return renamed\n"
    )
    left_result = validate_policy(left)
    right_result = validate_policy(right)
    assert left_result.valid and right_result.valid
    assert (
        left_result.identity.normalized_ast_sha256
        == right_result.identity.normalized_ast_sha256
    )
    assert left_result.identity.source_sha256 != right_result.identity.source_sha256


def test_normalized_identity_distinguishes_semantic_change() -> None:
    plus = validate_policy(
        "def priority(ctx, proposal):\n"
        "    score = proposal['features']['weight'] + 1\n"
        "    return score\n"
    )
    minus = validate_policy(
        "def priority(ctx, proposal):\n"
        "    score = proposal['features']['weight'] - 1\n"
        "    return score\n"
    )
    assert plus.valid and minus.valid
    assert plus.identity.normalized_ast_sha256 != minus.identity.normalized_ast_sha256


@pytest.mark.parametrize(
    "expression",
    ["True", "None", "1j", "{'value': 1}", "[1, 2]", "(1, 2)"],
)
def test_output_contract_is_checked_at_runtime_not_claimed_static(
    expression: str,
) -> None:
    result = validate_policy(
        f"def priority(ctx, proposal):\n    return {expression}\n"
    )
    if expression == "1j":
        assert not result.valid
    else:
        assert result.valid
