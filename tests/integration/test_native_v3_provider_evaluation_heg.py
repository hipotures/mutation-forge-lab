"""Recorded provider-to-HEG integration coverage for Native v3."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.json_io import read_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3.provider_evaluation import (
    run_provider_evaluation_smoke,
)
from mutation_forge.native_v3.serial_evaluator import SerialEpisodeConfig

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location(
    "native_v3_provider_evaluation_heg_fake",
    _FIXTURE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario


def _response() -> str:
    source = json.dumps(
        {
            "schema_version": "mforge.native.program.v3",
            "entry": {"op": "no_plan", "reason": "EXPLICIT"},
        },
        separators=(",", ":"),
    )
    return json.dumps(
        {
            "schema_version": "mforge.native.generated_policy.v1",
            "source": source,
            "design_summary": "Return one explicit no-plan program.",
            "hypothesis": "The initial HEG graph still receives an authoritative score.",
            "used_fields": [],
            "assumptions": [],
            "expected_failure_modes": [],
        },
        separators=(",", ":"),
    )


def test_recorded_provider_turn_completes_one_real_heg_evaluation(
    tmp_path: Path,
    heg_repo: Path,
) -> None:
    scenario = FakeScenario(final_text=_response())

    def process_factory(*_: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(scenario, **kwargs)

    provider = LocalCodexAppServerProvider(
        model="gpt-5.6-luna",
        effort="high",
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=1,
        process_factory=process_factory,
        auth_checker=lambda _: True,
        persist_artifacts=False,
    )
    try:
        report = run_provider_evaluation_smoke(
            provider,
            tmp_path,
            backend_factory=lambda: HegBackend(heg_repo),
            config=SerialEpisodeConfig(
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                episode_id="native-v3-step09-recorded-heg",
            ),
        )
    finally:
        provider.close()

    assert report["status"] == "completed"
    assert report["model_turns"] == 1
    assert report["graph_evaluations"] == 1
    payload = read_json(Path(str(report["evaluation_result"])))
    assert payload["backend"]["graph_backend_id"].startswith("heg-erdos-gyarfas-")
    assert payload["backend"]["score_implementation"] == "heg-cpp-score-worker"
    assert payload["evaluation"]["failure"] is None
    assert payload["evaluation"]["steps"][0]["outcome"] == "no_plan"
