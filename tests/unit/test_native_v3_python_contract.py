from __future__ import annotations

import ast
import inspect
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from mutation_forge.native_v3.experiment import V2_PROTOCOL, experiment_protocol
from mutation_forge.native_v3_python import (
    ACTION_METHODS,
    API_METHODS,
    BEHAVIOR_IDENTITY_PROTOCOL_ID,
    CONTEXT_FIELDS,
    CURRENT_JSON_DSL_WORKSPACE_SCHEMA_VERSION,
    GRAPH_FIELDS,
    IDENTITY_PROTOCOL_VERSION,
    MAX_AST_DEPTH,
    MAX_AST_NODES,
    MAX_FORBIDDEN_LENGTHS,
    MAX_HELPER_FUNCTIONS,
    MAX_SOURCE_BYTES,
    NO_PLAN_REASONS,
    PYTHON_EXPERIMENT_PROTOCOL_ID,
    PYTHON_RESPONSE_SCHEMA_VERSION,
    PYTHON_SYNTAX_VERSION,
    PYTHON_WORKSPACE_SCHEMA_VERSION,
    SELECTOR_METHODS,
    VALIDATOR_VERSION,
    BehaviorIdentityV1,
    GraphViewV1,
    NoPlan,
    PolicyContextV1,
    PythonWorkspaceProtocolError,
    SafeGraphAPIV1,
    accepted_ast_node_names,
    require_python_workspace_schema_version,
    validate_python_policy_response,
    validate_python_policy_source,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "configs/native/native-v3-python-policy-response.schema.json"
FIXTURE_PATH = ROOT / "tests/fixtures/native_v3_python_policy_contract_v1.json"

PLAIN_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.non_edges_legal()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    edge = api.pick(candidates, seed, 0)
    if edge == None:
        return api.no_plan()
    api.add_edge(edge)
    return api.emit()
"""

ANNOTATED_SOURCE = PLAIN_SOURCE.replace(
    "def propose(ctx, graph, api, seed):",
    "def propose(ctx, graph, api, seed) -> RewritePlan | NoPlan:",
)


def _codes(source: str) -> set[str]:
    return {diagnostic.code for diagnostic in validate_python_policy_source(source).diagnostics}


def _source_with_node_count(target: int) -> str:
    prefix = "def propose(ctx, graph, api, seed):\n"
    one_pass = prefix + "    pass\n"
    overhead = sum(1 for _ in ast.walk(ast.parse(one_pass))) - 1
    assert target > overhead
    return prefix + ("    pass\n" * (target - overhead))


def _context(**changes: object) -> PolicyContextV1:
    values: dict[str, object] = {
        "step_index": 0,
        "horizon": 10,
        "acceptance_profile_id": "strict",
        "stagnation_steps": 0,
        "exploration_window_index": 0,
        "accepted_rewrites": 0,
        "accepted_non_improving_rewrites": 0,
        "consecutive_non_improving_rewrites": 0,
        "witness_cap": 100,
        "invocation_ordinal": 0,
        "forbidden_lengths": (3, 4, 5),
    }
    values.update(changes)
    return PolicyContextV1(**values)  # type: ignore[arg-type]


def test_response_schema_and_exact_two_field_envelope() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    response = {"schema_version": PYTHON_RESPONSE_SCHEMA_VERSION, "source": PLAIN_SOURCE}
    assert not list(validator.iter_errors(response))

    result = validate_python_policy_response(json.dumps(response))
    assert result.valid
    assert result.response is not None
    assert result.response.as_dict() == response

    malformed = (
        '{"schema_version":"mforge.native.python_policy_response.v1",'
        '"source":"x","source":"y"}'
    )
    assert [item.code for item in validate_python_policy_response(malformed).diagnostics] == [
        "INVALID_ENVELOPE_JSON"
    ]
    for value, code in (
        ({}, "INVALID_ENVELOPE_FIELDS"),
        (
            {
                "schema_version": PYTHON_RESPONSE_SCHEMA_VERSION,
                "source": PLAIN_SOURCE,
                "extra": 1,
            },
            "INVALID_ENVELOPE_FIELDS",
        ),
        (
            {"schema_version": "wrong", "source": PLAIN_SOURCE},
            "INVALID_SCHEMA_VERSION",
        ),
        (
            {"schema_version": PYTHON_RESPONSE_SCHEMA_VERSION, "source": 1},
            "INVALID_SOURCE_TYPE",
        ),
    ):
        result = validate_python_policy_response(json.dumps(value))
        assert not result.valid
        assert [item.code for item in result.diagnostics] == [code]


@pytest.mark.parametrize("source", [PLAIN_SOURCE, ANNOTATED_SOURCE])
def test_exact_public_entry_point_accepts_optional_exact_annotation(source: str) -> None:
    result = validate_python_policy_source(source)
    assert result.valid, result.as_dict()


@pytest.mark.parametrize(
    ("definition", "code"),
    [
        ("def propose(graph, ctx, api, seed):", "INVALID_ENTRY_POINT_SIGNATURE"),
        ("def propose(ctx, graph, api, seed=0):", "INVALID_FUNCTION_SIGNATURE"),
        ("def propose(ctx, graph, api, *seed):", "INVALID_FUNCTION_SIGNATURE"),
        ("def propose(ctx, graph, api, **seed):", "INVALID_FUNCTION_SIGNATURE"),
        ("def propose(ctx, graph, *, api, seed):", "INVALID_FUNCTION_SIGNATURE"),
        ("def propose(ctx, graph, api, seed, /):", "INVALID_FUNCTION_SIGNATURE"),
        ("async def propose(ctx, graph, api, seed):", "FORBIDDEN_AST_NODE"),
        ("@decorator\ndef propose(ctx, graph, api, seed):", "FORBIDDEN_DECORATOR"),
        ("def propose(ctx, graph, api, seed) -> RewritePlan:", "INVALID_RETURN_ANNOTATION"),
        (
            "def propose(ctx, graph, api, seed) -> NoPlan | RewritePlan:",
            "INVALID_RETURN_ANNOTATION",
        ),
        ("def propose(ctx: int, graph, api, seed):", "FORBIDDEN_PARAMETER_ANNOTATION"),
    ],
)
def test_entry_point_signature_fails_closed_with_location(definition: str, code: str) -> None:
    result = validate_python_policy_source(f"{definition}\n    return api.no_plan()\n")
    assert not result.valid
    matches = [item for item in result.diagnostics if item.code == code]
    assert matches
    assert matches[0].line == 1 or code == "FORBIDDEN_DECORATOR"


def test_source_byte_limit_below_at_and_above() -> None:
    base = "def propose(ctx, graph, api, seed):\n    return api.no_plan()\n#"
    for size in (MAX_SOURCE_BYTES - 1, MAX_SOURCE_BYTES):
        source = base + ("x" * (size - len(base.encode("utf-8"))))
        assert len(source.encode("utf-8")) == size
        assert validate_python_policy_source(source).valid
    source = base + ("x" * (MAX_SOURCE_BYTES + 1 - len(base.encode("utf-8"))))
    result = validate_python_policy_source(source)
    assert not result.valid
    assert [item.code for item in result.diagnostics] == ["SOURCE_TOO_LARGE"]


def test_ast_node_limit_below_at_and_above() -> None:
    for count in (MAX_AST_NODES - 1, MAX_AST_NODES):
        source = _source_with_node_count(count)
        result = validate_python_policy_source(source)
        assert result.valid, result.as_dict()
        assert result.identity is not None
        assert result.identity.ast_node_count == count
    result = validate_python_policy_source(_source_with_node_count(MAX_AST_NODES + 1))
    assert not result.valid
    assert result.identity is not None
    assert result.identity.ast_node_count == MAX_AST_NODES + 1
    assert "AST_TOO_LARGE" in {item.code for item in result.diagnostics}


def test_helper_limit_below_at_and_above() -> None:
    def source(helper_count: int) -> str:
        helpers = "".join(
            f"def helper_h{index}(value):\n    return value\n\n"
            for index in range(helper_count)
        )
        calls = "".join(
            f"    value = helper_h{index}(seed)\n" for index in range(helper_count)
        )
        return (
            helpers
            + "def propose(ctx, graph, api, seed):\n"
            + calls
            + "    return api.no_plan()\n"
        )

    for helper_count in (MAX_HELPER_FUNCTIONS - 1, MAX_HELPER_FUNCTIONS):
        result = validate_python_policy_source(source(helper_count))
        assert result.valid, result.as_dict()
        assert result.identity is not None
        assert result.identity.helper_function_count == helper_count
    result = validate_python_policy_source(source(MAX_HELPER_FUNCTIONS + 1))
    assert not result.valid
    assert "TOO_MANY_HELPERS" in {item.code for item in result.diagnostics}


@pytest.mark.parametrize(
    "body",
    [
        "items = [1, 2]\n    total = sum(items)",
        "mapping = {'a': 1}\n    value = mapping['a']",
        "value = 1 if graph.order else 0",
        "value = (graph.minimum_degree + graph.maximum_degree) // 2",
        "value = not (ctx.step_index < ctx.horizon and graph.edge_count != 0)",
        "for value in range(64):\n"
        "        if value == 3:\n"
        "            continue\n"
        "        if value == 4:\n"
        "            break",
        "items = api.edges_removable()\n"
        "    for edge in reversed(items):\n"
        "        api.remove_edge(edge)",
    ],
)
def test_allowed_ast_corpus(body: str) -> None:
    source = (
        "def propose(ctx, graph, api, seed):\n"
        f"    {body}\n"
        "    return api.no_plan()\n"
    )
    result = validate_python_policy_source(source)
    assert result.valid, result.as_dict()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("import os", "FORBIDDEN_AST_NODE"),
        ("from os import path", "FORBIDDEN_AST_NODE"),
        ("global danger", "FORBIDDEN_AST_NODE"),
        ("class Danger: pass", "FORBIDDEN_AST_NODE"),
        ("while True: break", "FORBIDDEN_AST_NODE"),
        ("value = [x for x in [1]]", "FORBIDDEN_AST_NODE"),
        ("value = {x: x for x in [1]}", "FORBIDDEN_AST_NODE"),
        ("value = {x for x in [1]}", "FORBIDDEN_AST_NODE"),
        ("value = (x for x in [1])", "FORBIDDEN_AST_NODE"),
        ("value = lambda: 1", "FORBIDDEN_AST_NODE"),
        ("yield 1", "FORBIDDEN_AST_NODE"),
        ("yield from items", "FORBIDDEN_AST_NODE"),
        ("await task", "FORBIDDEN_AST_NODE"),
        ("async for item in items:\n        pass", "FORBIDDEN_AST_NODE"),
        ("async with manager:\n        pass", "FORBIDDEN_AST_NODE"),
        ("match seed:\n        case 0: pass", "FORBIDDEN_AST_NODE"),
        ("try:\n        pass\n    except Exception:\n        pass", "FORBIDDEN_AST_NODE"),
        ("try:\n        pass\n    except* Exception:\n        pass", "FORBIDDEN_AST_NODE"),
        ("raise RuntimeError()", "FORBIDDEN_AST_NODE"),
        ("assert seed", "FORBIDDEN_AST_NODE"),
        ("with thing:\n        pass", "FORBIDDEN_AST_NODE"),
        ("del value", "FORBIDDEN_AST_NODE"),
        ("value = (other := 1)", "FORBIDDEN_AST_NODE"),
        ("value = {1, 2}", "FORBIDDEN_AST_NODE"),
        ("value = graph.__dict__", "FORBIDDEN_ATTRIBUTE"),
        ("value = graph.edges", "FORBIDDEN_ATTRIBUTE"),
        ("value = api.unknown()", "FORBIDDEN_API_CALL"),
        ("value = seed.real", "FORBIDDEN_ATTRIBUTE"),
        ("items[0] = 1", "FORBIDDEN_ASSIGNMENT_TARGET"),
        ("left, right = [1, 2]", "FORBIDDEN_ASSIGNMENT_TARGET"),
        ("*items, = [1, 2]", "FORBIDDEN_ASSIGNMENT_TARGET"),
        ("for value in range(65):\n        pass", "UNBOUNDED_FOR_ITERATOR"),
        ("for value in unknown:\n        pass", "UNBOUNDED_FOR_ITERATOR"),
        ("break", "BREAK_OUTSIDE_LOOP"),
        ("continue", "CONTINUE_OUTSIDE_LOOP"),
        ("value = 1.5", "FORBIDDEN_CONSTANT"),
        ("value = 1j", "FORBIDDEN_CONSTANT"),
        ("value = 2 ** 8", "FORBIDDEN_AST_NODE"),
        ("value = 1 / 2", "FORBIDDEN_AST_NODE"),
        ("value = 1 @ 2", "FORBIDDEN_AST_NODE"),
        ("value = 1 << 2", "FORBIDDEN_AST_NODE"),
        ("value = 1 | 2", "FORBIDDEN_OPERATOR"),
        ("value = seed is None", "FORBIDDEN_AST_NODE"),
        ("value = _private", "INVALID_IDENTIFIER"),
        ("value = missing", "UNKNOWN_NAME"),
        ("value = object.method()", "FORBIDDEN_CALL"),
        ("value = api.no_plan(**mapping)", "FORBIDDEN_KEYWORD_EXPANSION"),
        ("def nested():\n        return 1", "FORBIDDEN_NESTED_FUNCTION"),
        ("nonlocal danger", "FORBIDDEN_AST_NODE"),
    ],
)
def test_adversarial_forbidden_ast_corpus(body: str, expected: str) -> None:
    source = (
        "def propose(ctx, graph, api, seed):\n"
        "    items = [1]\n"
        f"    {body}\n"
        "    return api.no_plan()\n"
    )
    assert expected in _codes(source)


@pytest.mark.parametrize(
    "target",
    [
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "print",
        "__import__",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
    ],
)
def test_dynamic_execution_reflection_and_io_calls_are_rejected(target: str) -> None:
    source = f"def propose(ctx, graph, api, seed):\n    return {target}('x')\n"
    result = validate_python_policy_source(source)
    assert not result.valid
    assert {"FORBIDDEN_CALL", "INVALID_IDENTIFIER"} & _codes(source)


def test_helper_recursion_and_call_depth_are_rejected() -> None:
    recursive = """\
def helper_loop():
    return helper_loop()

def propose(ctx, graph, api, seed):
    return helper_loop()
"""
    assert "RECURSIVE_HELPER_CALL" in _codes(recursive)

    helpers = "".join(
        f"def helper_h{index}():\n"
        f"    return {'api.no_plan()' if index == 8 else f'helper_h{index + 1}()'}\n\n"
        for index in range(9)
    )
    deep = helpers + "def propose(ctx, graph, api, seed):\n    return helper_h0()\n"
    assert "HELPER_CALL_DEPTH_EXCEEDED" in _codes(deep)

    unreachable = """\
def helper_unused():
    return 1

def propose(ctx, graph, api, seed):
    return api.no_plan()
"""
    assert "UNREACHABLE_HELPER" in _codes(unreachable)

    wrong_name = """\
def utility():
    return 1

def propose(ctx, graph, api, seed):
    value = utility()
    return api.no_plan()
"""
    assert "INVALID_HELPER_NAME" in _codes(wrong_name)


@pytest.mark.parametrize("parameter", ["ctx", "graph", "api", "seed", "len", "NoPlan"])
def test_helper_parameters_must_not_shadow_reserved_names(parameter: str) -> None:
    source = (
        f"def helper_value({parameter}):\n"
        "    return 1\n\n"
        "def propose(ctx, graph, api, seed):\n"
        "    value = helper_value(1)\n"
        "    return api.no_plan()\n"
    )
    assert "SHADOWED_RESERVED_NAME" in _codes(source)


def test_non_function_module_statements_are_rejected() -> None:
    source = (
        "MODULE_STATE = 1\n\n"
        "def propose(ctx, graph, api, seed):\n"
        "    return api.no_plan()\n"
    )
    assert "FORBIDDEN_TOP_LEVEL" in _codes(source)


def test_fail_closed_allowlist_is_exact_and_unknown_node_is_rejected() -> None:
    expected = {
        "Add",
        "And",
        "Assign",
        "Attribute",
        "AugAssign",
        "BinOp",
        "BitOr",
        "BoolOp",
        "Break",
        "Call",
        "Compare",
        "Constant",
        "Continue",
        "Dict",
        "Eq",
        "Expr",
        "FloorDiv",
        "For",
        "FunctionDef",
        "Gt",
        "GtE",
        "If",
        "IfExp",
        "In",
        "List",
        "Load",
        "Lt",
        "LtE",
        "Mod",
        "Module",
        "Mult",
        "Name",
        "Not",
        "NotEq",
        "NotIn",
        "Or",
        "Pass",
        "Return",
        "Slice",
        "Store",
        "Sub",
        "Subscript",
        "Tuple",
        "TypeIgnore",
        "UAdd",
        "USub",
        "UnaryOp",
        "arg",
        "arguments",
        "keyword",
    }
    assert set(accepted_ast_node_names()) == expected
    assert "FORBIDDEN_AST_NODE" in _codes(
        "def propose(ctx, graph, api, seed):\n    value = {1}\n    return api.no_plan()\n"
    )


def test_identity_ignores_only_accepted_nonsemantic_source_features() -> None:
    base = """\
def helper_value(item):
    value = item
    return value

def propose(ctx, graph, api, seed):
    candidate = helper_value(seed)
    return api.no_plan()
"""
    decorated = """\
# a module comment

def helper_value(item):
    "function documentation"
    value = item  # type: int
    return value  # a comment


def propose(ctx, graph, api, seed):
    "entry documentation"
    candidate = helper_value(seed)
    return api.no_plan()
"""
    type_ignore = base.replace("value = item", "value = item  # type: ignore")
    blank_lines = base.replace("    candidate", "\n\n    candidate")
    crlf = base.replace("\n", "\r\n")
    identities = [
        validate_python_policy_source(source).identity
        for source in (base, decorated, type_ignore, blank_lines, crlf)
    ]
    assert all(identity is not None for identity in identities)
    hashes = {identity.program_hash for identity in identities if identity is not None}
    assert len(hashes) == 1
    assert identities[0] is not None and identities[1] is not None and identities[-1] is not None
    assert identities[0].source_sha256 == identities[-1].source_sha256
    assert identities[0].source_sha256 != identities[1].source_sha256


def test_identity_v1_preserves_helper_parameter_and_local_names() -> None:
    base = """\
def helper_value(item):
    value = item
    return value

def propose(ctx, graph, api, seed):
    result = helper_value(seed)
    return api.no_plan()
"""
    variants = (
        base.replace("helper_value", "helper_candidate"),
        base.replace("item", "candidate"),
        base.replace("value", "candidate"),
        base.replace("result", "outcome"),
    )
    base_identity = validate_python_policy_source(base).identity
    assert base_identity is not None
    for variant in variants:
        identity = validate_python_policy_source(variant).identity
        assert identity is not None
        assert identity.program_hash != base_identity.program_hash


def test_semantic_changes_change_program_identity() -> None:
    one = validate_python_policy_source(
        "def propose(ctx, graph, api, seed):\n    value = 1\n    return api.no_plan()\n"
    )
    two = validate_python_policy_source(
        "def propose(ctx, graph, api, seed):\n    value = 2\n    return api.no_plan()\n"
    )
    assert one.valid and two.valid
    assert one.identity is not None and two.identity is not None
    assert one.identity.program_hash != two.identity.program_hash


def test_identity_and_diagnostics_match_offline_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == "mforge.native.python_policy_contract_fixture.v1"
    for case in fixture["cases"]:
        result = validate_python_policy_response(json.dumps(case["response"]))
        assert result.as_dict() == case["expected"]


def test_policy_context_forbidden_lengths_is_immutable_bounded_and_deterministic() -> None:
    context = _context()
    assert context.forbidden_lengths == (3, 4, 5)
    assert context.as_dict()["forbidden_lengths"] == [3, 4, 5]
    with pytest.raises(FrozenInstanceError):
        context.forbidden_lengths = (6,)  # type: ignore[misc]
    with pytest.raises(ValueError, match="immutable tuple"):
        _context(forbidden_lengths=[3, 4])
    with pytest.raises(ValueError, match="strictly increasing"):
        _context(forbidden_lengths=(4, 3))
    with pytest.raises(ValueError, match="strictly increasing"):
        _context(forbidden_lengths=(3, 3))
    with pytest.raises(ValueError, match="at most"):
        _context(forbidden_lengths=tuple(range(3, 3 + MAX_FORBIDDEN_LENGTHS + 1)))


def test_no_plan_reason_declaration_is_exact() -> None:
    assert {"EXPLICIT", "NO_MATCH", "ILLEGAL_FINAL_STATE", "NO_EFFECT"} == NO_PLAN_REASONS
    assert NoPlan("NO_MATCH").reason == "NO_MATCH"
    with pytest.raises(ValueError, match="unsupported"):
        NoPlan("ARBITRARY")


def test_graph_view_and_safe_api_expose_only_the_accepted_maximum_surface() -> None:
    assert {item.name for item in fields(GraphViewV1)} == {
        "order",
        "edge_count",
        "minimum_degree",
        "maximum_degree",
    }
    assert {item.name for item in fields(PolicyContextV1)} == set(CONTEXT_FIELDS)
    public_api = {
        name
        for name, member in SafeGraphAPIV1.__dict__.items()
        if callable(member) and not name.startswith("_")
    }
    assert public_api == set(API_METHODS)
    assert SELECTOR_METHODS | ACTION_METHODS | {"pick"} == API_METHODS
    prohibited = {
        "labels",
        "adjacency",
        "edges",
        "scorer",
        "verifier",
        "backend",
        "filesystem",
        "workspace",
        "provider",
        "held_out",
    }
    assert prohibited.isdisjoint(GRAPH_FIELDS | CONTEXT_FIELDS | API_METHODS)


def test_behavior_identity_is_separate_and_nonexecuting() -> None:
    identity = BehaviorIdentityV1(
        probe_manifest_sha256="a" * 64,
        behavior_signature="b" * 64,
    )
    assert identity.protocol_id == BEHAVIOR_IDENTITY_PROTOCOL_ID
    assert BEHAVIOR_IDENTITY_PROTOCOL_ID != IDENTITY_PROTOCOL_VERSION


def test_inactive_python_protocol_rejects_json_dsl_workspace_and_leaves_v2_default(
    tmp_path: Path,
) -> None:
    assert PYTHON_EXPERIMENT_PROTOCOL_ID != "v3"
    assert require_python_workspace_schema_version(PYTHON_WORKSPACE_SCHEMA_VERSION) == (
        PYTHON_WORKSPACE_SCHEMA_VERSION
    )
    with pytest.raises(PythonWorkspaceProtocolError, match="JSON-DSL"):
        require_python_workspace_schema_version(CURRENT_JSON_DSL_WORKSPACE_SCHEMA_VERSION)

    config = tmp_path / "experiment.toml"
    config.write_text('[run]\nexp_id = "unchanged-v2-default"\n', encoding="utf-8")
    assert experiment_protocol(config) == V2_PROTOCOL
    active_routing_source = inspect.getsource(
        __import__(
            "mutation_forge.native_v3.experiment",
            fromlist=["experiment_protocol"],
        )
    )
    assert "native_v3_python" not in active_routing_source


def test_m1_validation_never_executes_generated_source(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    source = (
        "def propose(ctx, graph, api, seed):\n"
        f"    open({str(marker)!r}, 'w')\n"
        "    return api.no_plan()\n"
    )
    result = validate_python_policy_source(source)
    assert not result.valid
    assert not marker.exists()

    package = ROOT / "src/mutation_forge/native_v3_python"
    forbidden_calls = {"exec", "eval", "compile"}
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, (path, node.lineno, node.func.id)
            if isinstance(node, ast.Name):
                assert node.id != "importlib", (path, node.lineno)


def test_unencodable_source_fails_closed() -> None:
    result = validate_python_policy_source("\ud800")
    assert not result.valid
    assert result.identity is None
    assert [item.code for item in result.diagnostics] == ["INVALID_SOURCE_UTF8"]


def test_deeply_nested_source_fails_closed_without_recursion_error() -> None:
    source = (
        "def propose(ctx, graph, api, seed):\n"
        f"    value = {'not ' * (MAX_AST_DEPTH * 8)}True\n"
        "    return api.no_plan()\n"
    )
    result = validate_python_policy_source(source)
    assert not result.valid
    assert "AST_TOO_DEEP" in {item.code for item in result.diagnostics}


def test_deeply_nested_json_envelope_fails_closed_without_recursion_error() -> None:
    raw = (
        '{"schema_version":"mforge.native.python_policy_response.v1","source":'
        + ("[" * 10_000)
        + "0"
        + ("]" * 10_000)
        + "}"
    )
    result = validate_python_policy_response(raw)
    assert not result.valid
    assert [item.code for item in result.diagnostics] == ["INVALID_ENVELOPE_JSON"]


def test_extreme_static_range_fails_closed_without_overflow() -> None:
    source = """\
def propose(ctx, graph, api, seed):
    for value in range(-9223372036854775808, 9223372036854775807):
        pass
    return api.no_plan()
"""
    result = validate_python_policy_source(source)
    assert not result.valid
    assert "UNBOUNDED_FOR_ITERATOR" in {item.code for item in result.diagnostics}


def test_fixture_versions_and_limits_are_frozen() -> None:
    assert MAX_SOURCE_BYTES == 32 * 1024
    assert MAX_AST_NODES == 2_000
    assert MAX_HELPER_FUNCTIONS == 16
    assert PYTHON_SYNTAX_VERSION == "3.12"
    assert VALIDATOR_VERSION == "mforge.native.python_policy_validator.v1"
    assert IDENTITY_PROTOCOL_VERSION == "mforge.native.python_policy_identity.v1"
