from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from mutation_forge import cli
from mutation_forge.experiment.artifacts import TurnArtifactStore
from mutation_forge.experiment.layout import WorkspaceError
from mutation_forge.experiment.service import ExperimentService, LegacyStage4Adapter
from mutation_forge.experiment.status import experiment_status


def _config(path: Path) -> Path:
    path.write_text(
        """schema_version = \"mforge.experiment.v1\"
exp_id = \"continuation\"
workspace = \"./workspace\"
kind = \"ranker-search\"
preset = \"heg-ranker-evolution-v1\"

[run]
wall_seconds = 30

[model]
provider = \"codex\"
name = \"gpt-5.6-luna\"
effort = \"high\"
concurrency = 8
max_repairs = 1

[search]
population_size = 8
max_generations = 4
max_model_turns = 64
selection = \"elite-diversity\"

[evaluation]
orders = [10, 12]
graph_seeds = [401, 402, 403, 404]
policy_seeds = [
  4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008,
  4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016,
]
horizon = 32
proposal_pool_size = 8
baselines = [\"random\", \"structural\"]
replay = true

[resources]
workers = 8
thread_count = 1
""",
        encoding="utf-8",
    )
    return path


class _Provider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        root = Path(request["artifact_dir"])
        prefix = str(request["artifact_prefix"])
        root.mkdir(parents=True, exist_ok=True)
        for name in (
            "request.md",
            "request.json",
            "response.md",
            "response.json",
            "provider-raw.json",
            "codex-rpc.jsonl",
            "events.jsonl",
            "wire.jsonl",
            "stdout.jsonl",
            "stderr.txt",
            "transcript.sha256",
            "codex-profile.json",
        ):
            (root / f"{prefix}.{name}").write_text("{}\n", encoding="utf-8")
        usage = {
            "inputTokens": 1,
            "cachedInputTokens": 0,
            "outputTokens": 1,
            "reasoningOutputTokens": 0,
            "totalTokens": 2,
            "final": True,
            "partial": False,
        }
        (root / f"{prefix}.usage.json").write_text(
            json.dumps(usage), encoding="utf-8"
        )
        return {
            "status": "completed",
            "accepted": True,
            "content": True,
            "response": {"source": "def priority(ctx, proposal):\n    return 0\n"},
            "response_text": "generated",
            "usage": usage,
            "provider_thread_id": "thread-1",
            "provider_turn_id": "turn-1",
        }

    def close(self) -> None:
        return None


class _InterruptingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        _config_path: Path,
        *,
        provider: Any,
        concurrency: int,
        resume: bool,
        observer: Any,
        run_override: Path,
    ) -> dict[str, Any]:
        assert (concurrency, resume) == (8, True)
        self.calls += 1
        provider.generate(
            {
                "campaign_id": "ticket18",
                "generation": 1,
                "slot": "slot-00",
                "phase": "initial",
                "idempotency_key": "ticket18-turn-1",
            }
        )
        if self.calls == 1:
            raise KeyboardInterrupt
        observer({"event": "generation_completed"})
        return {"status": "completed", "run": str(run_override), "generation": {}}


def test_forced_interrupt_resumes_without_repeating_provider_turn(tmp_path: Path) -> None:
    config = _config(tmp_path / "experiment.toml")
    provider = _Provider()
    engine = _InterruptingEngine()
    service = ExperimentService(adapter=LegacyStage4Adapter(provider=provider, engine=engine))

    with suppress(KeyboardInterrupt):
        service.run(config)
    root = tmp_path / "workspace" / "continuation"
    assert experiment_status(config)["state"] == "interrupted"
    # Deliberately leave the global index stale, as a kill can happen between
    # the fsync of this file and the next index refresh.
    (root / "artifacts" / "sessions" / "session-000001" / "late.log").write_text(
        "late evidence", encoding="utf-8"
    )

    result = service.run(config)
    assert result["state"] == "completed"
    assert provider.calls == 1
    assert experiment_status(config)["provider_turns"] == 1
    lock = json.loads(
        (root / "experiment.lock.json").read_text(encoding="utf-8")
    )
    assert lock["preset_identity"]["resolved"] is True
    assert set(lock["baseline_identities"]) == {"random", "structural"}
    assert lock["proposal_schema_identities"]
    assert lock["context_schema_identities"]
    assert lock["app_server"]["model"] == "gpt-5.6-luna"
    manifest = (
        root
        / "artifacts"
        / "generations"
        / "generation-0001"
        / "slot-00"
        / "initial"
        / "turn-manifest.json"
    )
    assert manifest.is_file()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["artifact_complete"] is True
    assert value["validation_completed"] is True
    assert (manifest.parent / "identity.json").is_file()
    assert (manifest.parent / "behavior.json").is_file()
    assert (manifest.parent / "worker_telemetry.json").is_file()


def test_incompatible_stage4_config_fails_before_workspace_creation(tmp_path: Path) -> None:
    config = _config(tmp_path / "experiment.toml")
    config.write_text(
        config.read_text(encoding="utf-8").replace("concurrency = 8", "concurrency = 1"),
        encoding="utf-8",
    )
    service = ExperimentService(adapter=LegacyStage4Adapter(provider=_Provider(), engine=object()))

    with pytest.raises(WorkspaceError, match="incompatible"):
        service.run(config)

    assert not (tmp_path / "workspace" / "continuation").exists()


def test_failed_pre_response_is_complete_and_text_is_redacted(tmp_path: Path) -> None:
    store = TurnArtifactStore(tmp_path / "artifacts")
    manifest = store.write_turn(
        generation=0,
        slot=0,
        request_text=(
            "Bearer sk-abcdefghijklmnop token=plainsecret from /home/user/private"
        ),
        terminal_status="failed",
        request_accepted=True,
        error="connection closed before content",
    )
    assert manifest["artifact_complete"] is True
    assert manifest["usage_final_exact"] is False
    request = (store.turn_directory(0, 0) / "slot-00.request.md").read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnop" not in request
    assert "plainsecret" not in request
    assert "/home/user" not in request
    assert store.verify_turn(store.turn_directory(0, 0))


def test_public_parser_rejects_historical_stage_commands() -> None:
    for command in ("stage3", "stage4", "stage4r", "stage7"):
        try:
            cli.main([command, "--help"])
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"public CLI accepted removed command {command}")
