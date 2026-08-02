from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mutation_forge import cli
from mutation_forge.experiment.artifacts import (
    ArtifactIncompleteError,
    TurnArtifactStore,
    copy_canonical_source,
)
from mutation_forge.experiment.config import (
    MAX_EXPERIMENT_ID_BYTES,
    load_experiment_config,
    validate_experiment_id,
)
from mutation_forge.experiment.layout import WorkspaceError
from mutation_forge.experiment.service import ExperimentService, NullExperimentAdapter
from mutation_forge.experiment.state import ActiveSessionError, ExperimentStateStore
from mutation_forge.experiment.status import STATUS_SCHEMA_VERSION, experiment_status


def _config(*, exp_id: str = "demo", workspace: str = "./workspace", wall: int = 1) -> str:
    return f'''schema_version = "mforge.experiment.v1"
exp_id = "{exp_id}"
workspace = "{workspace}"
kind = "ranker-search"
preset = "heg-ranker-evolution-v1"

[run]
wall_seconds = {wall}

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "high"
concurrency = 1
max_repairs = 0

[search]
population_size = 2
max_generations = 2
max_model_turns = 4
selection = "elite-diversity"

[evaluation]
orders = [10]
graph_seeds = [401]
policy_seeds = [4001]
horizon = 4
proposal_pool_size = 2
baselines = ["random", "structural"]
replay = true

[resources]
workers = 1
thread_count = 1
'''


def _write_config(tmp_path: Path, **kwargs: Any) -> Path:
    path = tmp_path / "configs" / "experiment.toml"
    path.parent.mkdir()
    path.write_text(_config(**kwargs), encoding="utf-8")
    return path


def test_config_resolves_workspace_relative_to_config(tmp_path: Path) -> None:
    path = _write_config(tmp_path, workspace="./workspace")
    config = load_experiment_config(path)
    assert config.experiment_root == path.parent / "workspace" / "demo"


@pytest.mark.parametrize("value", ["", ".", "..", "a/b", r"a\\b", "/tmp/demo", "bad\x01"])
def test_exp_id_rejects_unsafe_names(value: str) -> None:
    with pytest.raises(ValueError):
        validate_experiment_id(value)


def test_exp_id_preserves_spelling_and_has_documented_limit() -> None:
    value = "é" * (MAX_EXPERIMENT_ID_BYTES // len("é".encode()))
    assert validate_experiment_id(value) == value
    with pytest.raises(ValueError):
        validate_experiment_id(value + "é")


def test_first_run_creates_atomic_workspace_and_session(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    result = ExperimentService(adapter=NullExperimentAdapter()).run(path)
    root = tmp_path / "configs" / "workspace" / "demo"
    assert result["state"] == "idle"
    assert (root / "experiment.toml").read_bytes() == path.read_bytes()
    assert (root / "experiment.lock.json").is_file()
    assert (root / "state.sqlite3").is_file()
    assert (root / "checkpoints" / "checkpoint-000000000001.json").is_file()
    assert (
        root / "artifacts" / "sessions" / "session-000001" / "input-config.toml"
    ).read_bytes() == path.read_bytes()


def test_second_run_continues_and_run_budget_is_mutable(tmp_path: Path) -> None:
    path = _write_config(tmp_path, wall=1)
    ExperimentService(adapter=NullExperimentAdapter()).run(path)
    path.write_text(_config(wall=2), encoding="utf-8")
    result = ExperimentService(adapter=NullExperimentAdapter()).run(path)
    assert result["session_id"] == "session-000002"
    assert (
        tmp_path / "configs" / "workspace" / "demo" / "artifacts" / "sessions" / "session-000002"
    ).is_dir()


def test_immutable_change_fails_before_adapter(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    ExperimentService(adapter=NullExperimentAdapter()).run(path)
    path.write_text(_config().replace('selection = "elite-diversity"', 'selection = "other"'))
    calls: list[str] = []

    class Adapter:
        def run(self, *_: object) -> dict[str, str]:
            calls.append("called")
            return {"state": "completed"}

    with pytest.raises(ValueError, match="differs from the locked"):
        ExperimentService(adapter=Adapter()).run(path)
    assert calls == []


def test_completed_experiment_makes_no_adapter_call(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    class Complete:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, *_: object) -> dict[str, str]:
            self.calls += 1
            return {"state": "completed", "stop_reason": "generation_limit"}

    adapter = Complete()
    service = ExperimentService(adapter=adapter)
    service.run(path)
    result = service.run(path)
    assert adapter.calls == 1
    assert result["provider_calls"] == 0


def test_active_owner_and_stale_recovery(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    service = ExperimentService(adapter=NullExperimentAdapter())
    service.run(path)
    state_path = tmp_path / "configs" / "workspace" / "demo" / "state.sqlite3"
    with ExperimentStateStore(state_path) as state:
        state.create_session(
            number=99, session_id="session-stale", wall_seconds=1, starting_checkpoint=None
        )
        state.acquire_owner(
            exp_id="demo", session_id="session-stale", pid=987654, alive=lambda _: True
        )
        with pytest.raises(ActiveSessionError):
            state.acquire_owner(exp_id="demo", session_id="session-other", alive=lambda _: True)
        state.release_owner("session-stale")
        state.acquire_owner(
            exp_id="demo", session_id="session-other", pid=12345, alive=lambda _: False
        )
        assert state.owner()["session_id"] == "session-other"


def test_status_is_versioned_and_read_only(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    before = experiment_status(path)
    assert before["schema_version"] == STATUS_SCHEMA_VERSION
    assert before["state"] == "not_created"
    ExperimentService(adapter=NullExperimentAdapter()).run(path)
    after = experiment_status(path)
    assert after["state"] == "idle"
    assert after["provider_turns"] == 0


def test_manifest_reconciliation_rejects_modified_committed_artifact(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    service = ExperimentService(adapter=NullExperimentAdapter())
    service.run(path)
    artifact = (
        tmp_path
        / "configs"
        / "workspace"
        / "demo"
        / "artifacts"
        / "sessions"
        / "session-000001"
        / "input-config.toml"
    )
    artifact.write_text("tampered", encoding="utf-8")

    status = experiment_status(path)
    assert status["state"] == "failed"
    assert "digest mismatch" in str(status["last_error"])
    with pytest.raises(WorkspaceError, match="digest mismatch"):
        service.run(path)


def test_status_reads_nested_stage4_search_metrics(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    class CandidateAdapter:
        def run(
            self,
            _config: object,
            _layout: object,
            state: ExperimentStateStore,
            _session: object,
        ) -> dict[str, str]:
            state.record_candidate(
                "program-1",
                status="created",
                metadata={"search_metrics": {"pooled_median_auc": 0.75}},
            )
            return {"state": "idle"}

    ExperimentService(adapter=CandidateAdapter()).run(path)
    status = experiment_status(path)
    assert status["best_program_id"] == "program-1"
    assert status["best_primary_metric"] == 0.75


def test_cli_public_help_is_stage_free() -> None:
    help_text = cli.build_parser().format_help()
    assert "doctor" in help_text and "experiment" in help_text
    for stage in (
        "stage2d",
        "stage3",
        "stage4",
        "stage4r",
        "stage4e",
        "stage5",
        "stage6",
        "stage7",
    ):
        assert stage not in help_text


def test_cli_rejects_experiment_positional_id() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["experiment", "run", "demo"])


def _full_turn_kwargs() -> dict[str, object]:
    return {
        "request_text": "rendered prompt",
        "response_text": '{"source": "ok"}',
        "source": "def priority(ctx, proposal):\n    return 0\n",
        "usage": {
            "inputTokens": 1,
            "cachedInputTokens": 0,
            "outputTokens": 1,
            "reasoningOutputTokens": 0,
            "totalTokens": 3,
            "final": True,
            "partial": False,
        },
        "identity": {"source_sha256": "a" * 64},
        "behavior": {"signature": "b" * 64},
        "provenance": {"provider": "codex"},
        "validation": {"status": "valid"},
        "worker_telemetry": {"runtime_seconds": 0.1},
        "canonical_response": {"source": "ok"},
        "provider_raw": {"content": "ok"},
        "codex_profile": {"model": "gpt-5.6-luna"},
        "rpc": [{"method": "turn/start"}],
        "events": [{"event": "completed"}],
        "wire": [
            {"direction": "client_to_server", "message": {"id": 1}},
            {"direction": "server_to_client", "message": {"id": 1}},
        ],
        "stdout": [{"event": "stdout"}],
        "stderr": "",
        "request_idempotency_key": "turn-1",
        "provider_thread_id": "thread-1",
        "provider_turn_id": "turn-1",
        "request_accepted": True,
        "content_received": True,
        "validation_completed": True,
    }


def test_full_turn_artifacts_and_canonical_source(tmp_path: Path) -> None:
    store = TurnArtifactStore(tmp_path / "artifacts")
    manifest = store.write_turn(generation=2, slot=2, **_full_turn_kwargs())
    directory = store.turn_directory(2, 2)
    assert manifest["artifact_complete"] is True
    assert (directory / "slot-02.request.md").read_text() == "rendered prompt"
    assert (directory / "slot-02.response.md").is_file()
    assert (directory / "slot-02.wire.jsonl").is_file()
    assert (directory / "slot-02.transcript.sha256").is_file()
    assert store.verify_turn(directory)
    digest = copy_canonical_source(directory, tmp_path / "archive", "program-1")
    assert (
        digest == __import__("hashlib").sha256((directory / "source.py").read_bytes()).hexdigest()
    )
    assert (tmp_path / "archive" / "sources" / "program-1.py").read_bytes() == (
        directory / "source.py"
    ).read_bytes()


def test_incomplete_turn_fails_closed_but_retains_manifest(tmp_path: Path) -> None:
    store = TurnArtifactStore(tmp_path / "artifacts", max_bytes=4)
    with pytest.raises(ArtifactIncompleteError):
        store.write_turn(generation=0, slot=0, request_text="too long")
    manifest = json.loads(
        (store.turn_directory(0, 0) / "turn-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_complete"] is False
