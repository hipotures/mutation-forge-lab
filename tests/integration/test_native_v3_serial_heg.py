from __future__ import annotations

import json
from pathlib import Path

from mutation_forge.backends.heg import HegBackend
from mutation_forge.native_v3.contracts import (
    PROGRAM_SCHEMA_VERSION,
    ValidatedProgram,
    validate_program,
)
from mutation_forge.native_v3.heg_scoring import HegScoreEvidenceAdapter
from mutation_forge.native_v3.serial_evaluator import (
    SerialEpisodeConfig,
    evaluate_serial_program,
)
from mutation_forge.native_v3_python import (
    PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
    PythonSerialEpisodeConfigV1,
    evaluate_serial_python_policy,
)

PYTHON_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "native_v3_python_m3"
    / "add_edge.py"
)


def _known_valid_program() -> ValidatedProgram:
    validation = validate_program(
        json.dumps(
            {
                "schema_version": PROGRAM_SCHEMA_VERSION,
                "entry": {
                    "op": "let",
                    "name": "edge",
                    "value": {
                        "op": "pick",
                        "source": {
                            "op": "selector",
                            "selector_id": "non_edges_legal",
                            "arguments": {},
                        },
                        "mode": "seeded_uniform",
                    },
                    "body": {
                        "op": "block",
                        "children": [
                            {
                                "op": "apply",
                                "action_id": "add_edge",
                                "arguments": {
                                    "edge": {"op": "ref", "name": "edge"}
                                },
                            },
                            {"op": "emit"},
                        ],
                    },
                },
            },
            separators=(",", ":"),
        )
    )
    assert validation.valid
    assert validation.program is not None
    return validation.program


def test_one_serial_native_v3_episode_uses_current_heg_backend(
    heg_repo: Path,
) -> None:
    backend = HegBackend(heg_repo)
    try:
        scorer = HegScoreEvidenceAdapter(backend)
        result = evaluate_serial_program(
            backend=backend,
            scorer=scorer,
            program=_known_valid_program(),
            config=SerialEpisodeConfig(
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                episode_id="native-v3-step08-heg-smoke",
            ),
        )
    finally:
        backend.close()

    assert result.failure is None
    assert len(result.steps) == 1
    assert result.steps[0].outcome == "rewrite"
    assert result.steps[0].rewrite is not None
    assert result.steps[0].candidate_evidence is not None
    assert result.terminal_evidence is not None
    assert result.initial_evidence is not None
    assert result.initial_evidence.scientifically_bounded
    assert result.terminal_evidence.scientifically_bounded
    assert result.fitness_interval.lower <= result.fitness_interval.upper


def test_fixture_python_runs_through_sandbox_and_current_heg_scoring(
    heg_repo: Path,
) -> None:
    backend = HegBackend(heg_repo)
    try:
        scorer = HegScoreEvidenceAdapter(backend)
        result = evaluate_serial_python_policy(
            backend=backend,
            scorer=scorer,
            source=PYTHON_FIXTURE.read_text(encoding="utf-8"),
            config=PythonSerialEpisodeConfigV1(
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                episode_id="native-v3-python-m3-heg-smoke",
                forbidden_lengths=backend.target_forbidden_lengths(30),
            ),
        )
    finally:
        backend.close()

    scientific = result.scientific_result
    assert scientific.protocol_id == PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID
    assert scientific.failure is None
    assert len(scientific.steps) == 1
    assert scientific.steps[0].outcome == "rewrite"
    assert scientific.steps[0].rewrite is not None
    assert scientific.steps[0].candidate_evidence is not None
    assert scientific.initial_evidence is not None
    assert scientific.terminal_evidence is not None
    assert scientific.initial_evidence.scientifically_bounded
    assert scientific.terminal_evidence.scientifically_bounded
    assert scientific.fitness_interval.lower <= scientific.fitness_interval.upper
    assert result.as_dict()["external_activity"] == {
        "provider_turns": 0,
        "model_turns": 0,
        "app_server_calls": 0,
    }
