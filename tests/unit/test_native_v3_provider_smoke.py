from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import read_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3.provider_smoke import (
    build_request,
    parse_provider_response,
    run_provider_smoke,
)

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location("native_v3_fake_app_server", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario


def _program() -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.program.v3",
            "entry": {"op": "no_plan", "reason": "EXPLICIT"},
        },
        separators=(",", ":"),
    )


def _response() -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.generated_policy.v1",
            "source": _program(),
            "design_summary": "Emit an explicit no-plan result.",
            "hypothesis": "A minimal AST proves the transport boundary.",
            "used_fields": [],
            "assumptions": [],
            "expected_failure_modes": [],
        },
        separators=(",", ":"),
    )


def _provider(
    tmp_path: Path,
    *,
    authenticated: bool = True,
    auth_state: list[bool] | None = None,
) -> LocalCodexAppServerProvider:
    scenario = FakeScenario(final_text=_response())

    def process_factory(*_: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(scenario, **kwargs)

    return LocalCodexAppServerProvider(
        model="gpt-5.6-luna",
        effort="high",
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=1,
        process_factory=process_factory,
        auth_checker=lambda _: authenticated if auth_state is None else auth_state[0],
        persist_artifacts=False,
    )


def test_recorded_response_parses_to_canonical_native_v3_ast() -> None:
    envelope, program = parse_provider_response(_response())

    assert envelope["source"] == _program()
    assert json.loads(program.canonical_json) == json.loads(_program())
    assert len(program.program_hash) == 64
    assert program.node_count == 1


def test_fake_app_server_runs_one_v2_turn_and_writes_derivative_outside_turn(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    try:
        report = run_provider_smoke(provider, tmp_path)
    finally:
        provider.close()

    assert report["status"] == "completed"
    assert report["model_turns"] == 1
    assert report["graph_evaluations"] == 0
    assert report["valid_ast"] is True
    turn = Path(str(report["turn_directory"]))
    assert turn.is_dir()
    assert Path(str(report["validated_program"])).parent == tmp_path / "native-v3-output"
    assert not (turn / "validated-program.json.gz").exists()
    manifest = read_json(turn / "turn-manifest.json.gz")
    assert manifest["artifact_complete"] is True
    assert manifest["response_projection_valid"] is True
    assert manifest["validation_completed"] is True
    assert {path.name for path in turn.iterdir()} == {
        "canonical_response.json.gz",
        "identity.json.gz",
        "provenance.json.gz",
        "slot-00.codex-profile.json.gz",
        "slot-00.codex-rpc.jsonl",
        "slot-00.events.jsonl",
        "slot-00.output-schema.json.gz",
        "slot-00.provider-raw.json.gz",
        "slot-00.request.json.gz",
        "slot-00.request.md",
        "slot-00.response.json.gz",
        "slot-00.response.md",
        "slot-00.response.raw.txt",
        "slot-00.stderr.txt",
        "slot-00.stdout.jsonl",
        "slot-00.system-prompt.md",
        "slot-00.transcript.sha256",
        "slot-00.transport-diagnostics.json.gz",
        "slot-00.usage.json.gz",
        "slot-00.wire.jsonl",
        "source.py",
        "turn-manifest.json.gz",
        "validation.json.gz",
    }


def test_unauthenticated_fixture_reports_provider_error_before_scientific_work(
    tmp_path: Path,
) -> None:
    auth_state = [False]
    provider = _provider(tmp_path, auth_state=auth_state)
    try:
        report = run_provider_smoke(provider, tmp_path)
        auth_state[0] = True
        resumed = run_provider_smoke(provider, tmp_path)
    finally:
        provider.close()

    assert report["status"] == "provider_error"
    assert report["error_classification"] == "authentication"
    assert "auth" in str(report["error"]).lower()
    assert report["model_turns"] == 0
    assert report["graph_evaluations"] == 0
    assert report["resumable"] is True
    assert resumed["status"] == "completed"
    assert resumed["valid_ast"] is True
    assert resumed["model_turns"] == 1
    assert resumed["graph_evaluations"] == 0
    assert (tmp_path / "native-v3-output" / "validated-program.json.gz").is_file()


def test_request_uses_dedicated_v3_prompt_schema_and_v2_provider_shape(tmp_path: Path) -> None:
    request = build_request(tmp_path, model="gpt-5.6-luna", effort="high")

    assert "Native v3" in request["system_prompt"]
    assert request["output_schema"]["$id"] == "mforge.native-v3-provider-envelope.v1"
    assert request["output_schema"]["properties"]["schema_version"]["const"] == (
        "mforge.native.generated_policy.v1"
    )
    assert request["output_schema"]["properties"]["schema_version"]["type"] == "string"
    assert request["artifact_prefix"] == "slot-00"
    assert request["max_repairs"] == 0
