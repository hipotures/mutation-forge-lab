from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from mutation_forge import cli
from mutation_forge.experiment.artifacts import TurnArtifactStore
from mutation_forge.experiment.config import load_experiment_config
from mutation_forge.experiment.layout import ExperimentLayout, WorkspaceError
from mutation_forge.experiment.service import (
    ExperimentService,
    LegacyStage4Adapter,
    _WorkspaceStage4Provider,
)
from mutation_forge.experiment.sessions import SessionContext
from mutation_forge.experiment.state import ExperimentStateStore
from mutation_forge.experiment.status import experiment_status
from mutation_forge.stage4.config import load_stage4_config
from mutation_forge.stage4.generation import GenerationConfig, GenerationCoordinator


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
proposal_pool_size = 12
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
        result = {
            "status": "completed",
            "accepted": True,
            "content": True,
            "response": {"source": "def priority(ctx, proposal):\n    return 0\n"},
            "response_text": "generated",
            "usage": usage,
            "provider_thread_id": "thread-1",
            "provider_turn_id": "turn-1",
        }
        (root / f"{prefix}.response.md").write_text(
            result["response_text"], encoding="utf-8"
        )
        (root / f"{prefix}.response.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

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


class _RepairingProvider(_Provider):
    """Controlled transport with one repairable source followed by a valid repair."""

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        result = super().generate(request)
        source = (
            "def priority(ctx, proposal)\n    return 0\n"
            if request["phase"] == "initial"
            else "def priority(ctx, proposal):\n    return 0\n"
        )
        result["response"] = {
            "schema_version": "stage4.generated_policy.v1",
            "source": source,
            "design_summary": "test policy",
            "change_summary": "test change",
            "hypothesis": "test hypothesis",
            "used_fields": ["proposal.k"],
            "assumptions": ["valid inputs"],
            "expected_failure_modes": ["none"],
        }
        result["response_text"] = source
        root = Path(request["artifact_dir"])
        prefix = str(request["artifact_prefix"])
        (root / f"{prefix}.response.md").write_text(source, encoding="utf-8")
        (root / f"{prefix}.response.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result


class _CoordinatorInterruptingEngine:
    """Exercise the production coordinator while stopping at a durable boundary."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        config_path: Path,
        *,
        provider: Any,
        concurrency: int,
        resume: bool,
        observer: Any,
        run_override: Path,
    ) -> dict[str, Any]:
        assert (concurrency, resume) == (8, True)
        self.calls += 1
        stage4 = load_stage4_config(config_path)
        coordinator = GenerationCoordinator(
            provider,
            config=GenerationConfig(
                campaign_id="ticket18-coordinator",
                sandbox_limits=stage4.sandbox,
                checkpoint_path=run_override / "generation-checkpoint.json",
            ),
        )
        request = coordinator.build_request(0, "slot-00", "parent-0")
        result = coordinator.run_request(
            request,
            allow_repair=True,
            allow_infrastructure_retry=False,
        )
        assert result.status == "accepted"
        observer({"event": "generation_completed", "generation": 0})
        if self.calls == 1:
            raise KeyboardInterrupt
        return {
            "status": "completed",
            "run": str(run_override),
            "generation": {"generation_count": 1},
        }


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


def test_adapter_runs_real_coordinator_repair_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path / "experiment.toml")
    provider = _RepairingProvider()
    engine = _CoordinatorInterruptingEngine()
    service = ExperimentService(adapter=LegacyStage4Adapter(provider=provider, engine=engine))

    with suppress(KeyboardInterrupt):
        service.run(config)

    root = tmp_path / "workspace" / "continuation"
    assert list((root / "checkpoints").glob("checkpoint-*.json"))
    initial = (
        root
        / "artifacts"
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "initial"
    )
    repair = initial.parent / "repair-01"
    initial_validation = json.loads((initial / "validation.json").read_text(encoding="utf-8"))
    assert initial_validation["valid"] is False
    assert not (initial / "behavior.json").exists()
    assert (repair / "behavior.json").is_file()
    assert (repair / "worker_telemetry.json").is_file()

    result = service.run(config)
    assert result["state"] == "completed"
    assert provider.calls == 2
    assert engine.calls == 2


def test_manifest_only_turn_recovery_accounts_usage_once(tmp_path: Path) -> None:
    config_path = _config(tmp_path / "experiment.toml")
    config = load_experiment_config(config_path)
    layout = ExperimentLayout.from_config(config)
    ExperimentStateStore.initialize(
        layout.state,
        exp_id=config.exp_id,
        lock_hash="test-lock",
        root=layout.root,
    )
    state = ExperimentStateStore(layout.state)
    session = SessionContext(1, "session-000001", tmp_path, 30, "now", None)
    stage4 = load_stage4_config(
        Path(__file__).resolve().parents[2] / "configs" / "stage4-search.toml"
    )
    provider = _RepairingProvider()
    wrapped = _WorkspaceStage4Provider(
        provider,
        layout,
        state,
        session,
        sandbox_limits=stage4.sandbox,
    )
    request = {
        "campaign_id": "ticket18-coordinator",
        "generation": 0,
        "slot": "slot-00",
        "phase": "initial",
        "idempotency_key": "manifest-only-turn",
    }
    raw = provider.generate(wrapped._payload(request))
    directory = layout.generation_slot_phase(0, "slot-00", "initial")
    wrapped.turns.record_existing_turn(
        directory,
        generation=0,
        slot="slot-00",
        phase="initial",
        request=request,
        result=wrapped._with_validation_evidence(raw),
    )

    retained = wrapped.generate(request)
    assert retained["retained"] is True
    assert state.provider_turn("manifest-only-turn") is not None
    assert (session.provider_turns_attempted, session.provider_turns_completed) == (1, 1)
    assert session.token_usage_delta == 2
    wrapped.generate(request)
    assert (session.provider_turns_attempted, session.provider_turns_completed) == (1, 1)
    assert session.token_usage_delta == 2
    assert provider.calls == 1
    state.close()


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


def test_semantically_invalid_freeze_fails_before_workspace_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mutation_forge.stage4 import commands

    config = _config(tmp_path / "experiment.toml")
    freeze_root = tmp_path / "stage4-run"
    freeze_root.mkdir()
    (freeze_root / "search-freeze.json").write_text(
        json.dumps({"schema_version": "stage4.search.freeze.v1", "verified": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(commands, "campaign_root", lambda _config: freeze_root)

    with pytest.raises(WorkspaceError, match="frozen search metadata is invalid"):
        ExperimentService().run(config)

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
