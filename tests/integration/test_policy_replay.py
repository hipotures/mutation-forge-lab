from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge.cli import main
from mutation_forge.sandbox.config import (
    PolicyEvaluationConfig,
    load_policy_config,
)
from mutation_forge.sandbox.contracts import SandboxLimits
from mutation_forge.sandbox.policy import (
    evaluate_policy,
    probe_policy,
    replay_policy,
)


def test_behavior_signature_has_deterministic_tie_breaking(project_root: Path) -> None:
    source = (project_root / "fixtures" / "rankers" / "constant.py").read_text()
    first = probe_policy(source)
    second = probe_policy(source)
    assert first["status"] == "completed"
    assert first["identity"] == second["identity"]
    assert first["behavior_signature"] == second["behavior_signature"]
    signature = first["behavior_signature"]
    assert isinstance(signature, dict)
    probes = signature["probes"]
    assert isinstance(probes, list)
    assert probes[0]["rank_order"] == ["p-alpha", "p-beta", "p-gamma"]
    assert probes[0]["selected_proposal_id"] == "p-alpha"


def test_evaluation_persists_canonical_artifacts_and_replays(
    tmp_path: Path,
    project_root: Path,
    heg_repo: Path,
) -> None:
    config_source = tmp_path / "config.toml"
    config_source.write_text("# test config provenance\n")
    config = PolicyEvaluationConfig(
        source_path=config_source,
        run_root=tmp_path / "runs",
        output="json",
        project_repo=project_root,
        heg_repo=heg_repo,
        frozen_project_commit="3b9beba058f472d6f0cad5b6210f34c6dbf96731",
        frozen_heg_commit="fd97451b0f3d87400d1d955a2c6b1b18303344ff",
        limits=SandboxLimits(max_ast_nodes=499),
    )
    policy = project_root / "fixtures" / "rankers" / "weighted.py"
    result = evaluate_policy(policy, config)
    assert result["status"] == "completed"
    run_path = Path(str(result["run_path"]))
    persisted = json.loads((run_path / "result.json").read_text())
    assert persisted == result
    assert (run_path / "artifacts" / "programs" / "policy.py").read_text() == (
        policy.read_text()
    )
    assert json.loads((run_path / "terminal_status.json").read_text())[
        "status"
    ] == "completed"
    for artifact in (
        "validation.json",
        "identity.json",
        "limits.json",
        "behavior_signature.json",
        "worker_telemetry.json",
        "provenance.json",
        "result.json",
        "terminal_status.json",
    ):
        assert (run_path / artifact).is_file()
    assert json.loads((run_path / "validation.json").read_text()) == result[
        "validation"
    ]
    assert json.loads((run_path / "identity.json").read_text()) == result["identity"]
    assert json.loads((run_path / "limits.json").read_text()) == result["limits"]
    assert json.loads((run_path / "behavior_signature.json").read_text()) == (
        result["behavior_signature"]
    )
    assert json.loads((run_path / "worker_telemetry.json").read_text()) == (
        result["worker_telemetry"]
    )
    assert result["provenance"]["model_calls"] == 0
    assert result["provenance"]["network_calls"] == 0
    assert result["provenance"]["heg_pin_verified"]
    assert result["provenance"]["project_base_verified"]
    assert result["provenance"]["execution_gate_verified"]
    replay = replay_policy(run_path)
    assert replay["status"] == "completed"
    assert replay["source_sha256_match"]
    assert replay["normalized_ast_sha256_match"]
    assert replay["behavior_signature_match"]
    assert replay["replayed"]["limits"]["max_ast_nodes"] == 499


def test_policy_config_and_cli_json_rich_have_same_canonical_payload(
    project_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_policy_config(project_root / "configs" / "stage2a-probe.toml")
    assert config.limits.address_space_bytes == 128 * 1024 * 1024
    policy = project_root / "fixtures" / "rankers" / "conditional.py"
    assert main(["policy", "validate", str(policy), "--json"]) == 0
    json_output = capsys.readouterr().out
    assert main(["policy", "validate", str(policy)]) == 0
    rich_output = capsys.readouterr().out
    assert json.loads(json_output) == json.loads(rich_output)
