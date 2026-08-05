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
