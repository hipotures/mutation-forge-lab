from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mutation_forge import cli
from mutation_forge.events import Event
from mutation_forge.experiment import state as state_module
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
from mutation_forge.experiment.generation import GenerationConfig, GenerationCoordinator
from mutation_forge.experiment.layout import ExperimentLayout, WorkspaceError
from mutation_forge.experiment.lock import canonical_bytes, sha256_bytes, verify_lock
from mutation_forge.experiment.provider import NativeProviderConfig, _CodexTransport
from mutation_forge.experiment.service import (
    ExperimentService,
    NullExperimentAdapter,
    final_stop_experiment,
)
from mutation_forge.experiment.state import ActiveSessionError, ExperimentStateStore
from mutation_forge.experiment.status import (
    STATUS_SCHEMA_VERSION,
    experiment_status,
    render_status,
)


def _config(*, exp_id: str = "demo", workspace: str = "./workspace", wall: int = 1) -> str:
    return f'''schema_version = "mforge.experiment.v2"
exp_id = "{exp_id}"
workspace = "{workspace}"
kind = "heg"
preset = "native"

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


def test_config_accepts_and_serializes_unbounded_limits(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config()
        .replace("max_generations = 2", 'max_generations = "unbounded"')
        .replace("max_model_turns = 4", 'max_model_turns = "unbounded"'),
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert config.search.max_generations is None
    assert config.search.max_model_turns is None
    assert config.resolved_dict()["search"] == {
        "population_size": 2,
        "max_generations": "unbounded",
        "max_model_turns": "unbounded",
        "selection": "elite-diversity",
    }
    assert "effort" not in config.immutable_projection()["model"]


def test_lock_without_v2_schema_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config().replace('effort = "high"', 'effort = "xhigh"'),
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    layout = ExperimentLayout.from_config(config)
    legacy_projection = config.immutable_projection()
    legacy_projection["model"]["effort"] = "max"
    lock = {
        "exp_id": config.exp_id,
        "experiment_root": str(layout.root.resolve()),
        "normalized_immutable_config": legacy_projection,
        "immutable_config_sha256": sha256_bytes(canonical_bytes(legacy_projection)),
    }

    with pytest.raises(ValueError, match="Unsupported experiment lock schema"):
        verify_lock(lock, config, layout)


def test_config_accepts_hourly_token_limit(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config().replace(
            "wall_seconds = 1",
            "wall_seconds = 1\nmax_total_tokens_per_hour = 1_000_000",
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert config.run.max_total_tokens_per_hour == 1_000_000
    assert "run" not in config.immutable_projection()


def test_config_accepts_explicit_unbounded_hourly_token_limit(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config().replace(
            "wall_seconds = 1",
            'wall_seconds = 1\nmax_total_tokens_per_hour = "unbounded"',
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(path)
    assert config.run.max_total_tokens_per_hour is None
    assert config.resolved_dict()["run"]["max_total_tokens_per_hour"] == "unbounded"


@pytest.mark.parametrize("value", ["0", "-1", "true", '"infinite"'])
def test_config_rejects_invalid_hourly_token_limit(
    tmp_path: Path, value: str
) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config().replace(
            "wall_seconds = 1",
            f"wall_seconds = 1\nmax_total_tokens_per_hour = {value}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_total_tokens_per_hour"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_generations", "0"),
        ("max_generations", "-1"),
        ("max_generations", '"infinite"'),
        ("max_generations", "true"),
        ("max_model_turns", "0"),
        ("max_model_turns", '"none"'),
    ],
)
def test_config_rejects_invalid_search_limits(tmp_path: Path, field: str, value: str) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config().replace(
            f"{field} = {2 if field == 'max_generations' else 4}", f"{field} = {value}"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_config_rejects_v1_explicitly(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config().replace("mforge.experiment.v2", "mforge.experiment.v1"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="accepts only mforge.experiment.v2"):
        load_experiment_config(path)


def test_config_does_not_infer_missing_search_limits(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    path.write_text(
        _config().replace("max_generations = 2\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_generations"):
        load_experiment_config(path)


def test_durable_events_are_idempotent_by_key(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite3"
    ExperimentStateStore.initialize(
        state_path,
        exp_id="demo",
        lock_hash="0" * 64,
        root=tmp_path,
    )
    with ExperimentStateStore(state_path) as state:
        first = state.write_event(
            "counterexample_candidate_found",
            {"candidate_id": "cx-test", "idempotency_key": "cx-test:candidate"},
        )
        repeated = state.write_event(
            "counterexample_candidate_found",
            {"candidate_id": "cx-test", "idempotency_key": "cx-test:candidate"},
        )
        count = state.connection.execute("SELECT COUNT(*) FROM events").fetchone()

    assert repeated == first
    assert count is not None and count[0] == 1


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
    events = [
        json.loads(line)
        for line in (root / "artifacts" / "sessions" / "session-000001" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    started = next(event for event in events if event["event_type"] == "session_started")
    assert started["worker_count"] == 1
    assert started["active_workers"] == 0


def test_second_run_continues_and_invocation_fields_are_mutable(tmp_path: Path) -> None:
    path = _write_config(tmp_path, wall=1)
    ExperimentService(adapter=NullExperimentAdapter()).run(path)
    path.write_text(
        _config(wall=2).replace('effort = "high"', 'effort = "xhigh"'),
        encoding="utf-8",
    )
    seen_efforts: list[str] = []

    class Adapter:
        def run(self, config: Any, *_: object) -> dict[str, str]:
            seen_efforts.append(config.model.effort)
            return {"state": "idle", "stop_reason": "session_wall_seconds"}

    result = ExperimentService(adapter=Adapter()).run(path)
    assert result["session_id"] == "session-000002"
    assert seen_efforts == ["xhigh"]
    assert (
        tmp_path / "configs" / "workspace" / "demo" / "artifacts" / "sessions" / "session-000002"
    ).is_dir()


def test_second_run_allows_runtime_parallelism_changes(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    ExperimentService(adapter=NullExperimentAdapter()).run(path)
    path.write_text(
        _config()
        .replace('effort = "high"', 'effort = "xhigh"')
        .replace("concurrency = 1", "concurrency = 2")
        .replace("max_repairs = 0", "max_repairs = 1")
        .replace("workers = 1", "workers = 8")
        .replace("thread_count = 1", "thread_count = 2"),
        encoding="utf-8",
    )
    seen: list[tuple[str, int, int, int, int]] = []

    class Adapter:
        def run(self, config: Any, *_: object) -> dict[str, str]:
            seen.append(
                (
                    config.model.effort,
                    config.model.concurrency,
                    config.model.max_repairs,
                    config.resources.workers,
                    config.resources.thread_count,
                )
            )
            return {"state": "idle", "stop_reason": "session_wall_seconds"}

    result = ExperimentService(adapter=Adapter()).run(path)
    assert result["state"] == "idle"
    assert seen == [("xhigh", 2, 1, 8, 2)]


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
            return {"state": "completed", "stop_reason": "counterexample_verified"}

    adapter = Complete()
    service = ExperimentService(adapter=adapter)
    service.run(path)
    result = service.run(path)
    assert adapter.calls == 1
    assert result["provider_calls"] == 0


def test_operator_final_stop_is_terminal_and_resumeless(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    adapter = NullExperimentAdapter()
    ExperimentService(adapter=adapter).run(path)

    result = final_stop_experiment(path)
    status = experiment_status(path)

    assert result["changed"] is True
    assert result["stop_reason"] == "operator_final_stop"
    assert status["state"] == "completed"
    assert status["terminal"] is True
    assert status["resumable"] is False
    assert status["last_stop_reason"] == "operator_final_stop"


def test_verified_event_follows_terminal_checkpoint(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    class Verified:
        def run(self, *_: object) -> dict[str, object]:
            return {
                "state": "completed",
                "stop_reason": "counterexample_verified",
                "counterexample": {
                    "candidate_id": "cx-" + "a" * 64,
                    "certificate_path": "/evidence/certificate.json",
                    "certificate_sha256": "b" * 64,
                },
            }

    class Sink:
        def __init__(self) -> None:
            self.events: list[str] = []

        def write(self, event: Event) -> None:
            self.events.append(event.event_type)

        def close(self) -> None:
            return None

    sink = Sink()
    ExperimentService(adapter=Verified(), event_sinks=[sink]).run(path)
    checkpoint_index = sink.events.index("checkpoint_written")
    verified_index = sink.events.index("counterexample_verified")
    completed_index = sink.events.index("experiment_completed")
    assert checkpoint_index < verified_index < completed_index


def test_model_turn_boundary_is_exhausted_and_config_is_immutable(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    class Adapter:
        def __init__(self) -> None:
            self.calls: list[int | None] = []

        def run(
            self,
            _config: object,
            _layout: object,
            _state: object,
            _session: object,
            *,
            effective_max_model_turns: int | None = None,
        ) -> dict[str, str]:
            self.calls.append(effective_max_model_turns)
            return {"state": "exhausted", "stop_reason": "max_model_turns"}

    adapter = Adapter()
    service = ExperimentService(adapter=adapter)
    first = service.run(path)
    assert first["state"] == "exhausted"
    assert first["status"] == "exhausted"
    assert first["stop_reason"] == "max_model_turns"

    path.write_text(_config().replace("max_model_turns = 4", "max_model_turns = 8"))
    with pytest.raises(ValueError, match="differs from the locked"):
        service.run(path)
    assert adapter.calls == [4]


def test_generation_model_turn_boundary_is_not_completed(tmp_path: Path) -> None:
    class Provider:
        def generate(self, _request: object) -> object:
            raise AssertionError("model provider must not run at an exhausted cap")

    result = GenerationCoordinator(
        Provider(),
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=0,
            max_repairs=0,
            checkpoint_path=tmp_path / "generation.json",
        ),
    ).run()
    assert result.status == "budget_exhausted"
    assert result.summary["stop_reason"] == "max_model_turns"


def test_generation_hourly_token_boundary_stops_before_provider(
    tmp_path: Path,
) -> None:
    class Provider:
        def generate(self, _request: object) -> object:
            raise AssertionError("provider must not run at the hourly token limit")

    result = GenerationCoordinator(
        Provider(),
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=None,
            max_repairs=0,
            checkpoint_path=tmp_path / "generation.json",
        ),
        budget_exhausted=lambda: "hourly_token_limit",
    ).run()
    assert result.status == "budget_exhausted"
    assert result.summary["stop_reason"] == "hourly_token_limit"


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


def test_completed_turn_accounting_is_atomic_and_finish_is_idempotent(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.sqlite3"
    ExperimentStateStore.initialize(
        state_path,
        exp_id="demo",
        lock_hash="test-lock",
        root=tmp_path,
    )
    with ExperimentStateStore(state_path) as state:
        state.create_session(
            number=1,
            session_id="session-000001",
            wall_seconds=30,
            starting_checkpoint=None,
        )
        values: dict[str, Any] = {
            "idempotency_key": "turn-1",
            "generation": 0,
            "slot": "slot-00",
            "phase": "initial",
            "state": "completed",
            "usage": {
                "inputTokens": 1,
                "cachedInputTokens": 0,
                "outputTokens": 2,
                "reasoningOutputTokens": 0,
                "totalTokens": 3,
                "final": True,
                "partial": False,
            },
        }
        assert state.record_provider_turn(**values) is True
        assert state.record_provider_turn(**values) is False
        session = state.session("session-000001")
        assert session is not None
        assert session["provider_turns_attempted"] == 1
        assert session["provider_turns_completed"] == 1
        assert session["token_usage_delta"] == 3
        assert state.cumulative() == {
            "provider_turns": 1,
            "total_tokens": 3,
            "compute_seconds": 0.0,
        }
        assert state.token_usage() == {
            "inputTokens": 1,
            "cachedInputTokens": 0,
            "cacheWriteInputTokens": 0,
            "outputTokens": 2,
            "reasoningOutputTokens": 0,
            "totalTokens": 3,
            "quality": "exact",
            "chargedFailedTurns": 0,
        }
        assert state.record_provider_turn(
            idempotency_key="turn-failed",
            generation=0,
            slot="slot-01",
            phase="initial",
            state="failed",
            usage={
                "inputTokens": 4,
                "cachedInputTokens": 0,
                "outputTokens": 1,
                "reasoningOutputTokens": 0,
                "totalTokens": 5,
                "final": False,
                "partial": True,
            },
            error="transport timeout",
        )
        failed = state.provider_turn("turn-failed")
        assert failed is not None
        failed_usage = json.loads(failed["usage_json"])
        assert failed_usage["quality"] == "partial"
        assert state.cumulative()["provider_turns"] == 2
        assert state.cumulative()["total_tokens"] == 8
        assert state.token_usage() == {
            "inputTokens": 5,
            "cachedInputTokens": 0,
            "cacheWriteInputTokens": 0,
            "outputTokens": 3,
            "reasoningOutputTokens": 0,
            "totalTokens": 8,
            "quality": "partial",
            "chargedFailedTurns": 1,
        }
        state.finish_session(
            "session-000001",
            status="idle",
            ending_state="idle",
            ending_checkpoint=None,
            provider_turns_attempted=2,
            provider_turns_completed=1,
            token_usage_delta=8,
            cumulative_tokens=8,
        )
        assert state.cumulative()["provider_turns"] == 2
        assert state.cumulative()["total_tokens"] == 8


def test_hourly_token_usage_is_rolling_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.sqlite3"
    ExperimentStateStore.initialize(
        state_path,
        exp_id="demo",
        lock_hash="test-lock",
        root=tmp_path,
    )
    started = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(state_module, "_now", lambda: started.isoformat())
    with ExperimentStateStore(state_path) as state:
        state.create_session(
            number=1,
            session_id="session-000001",
            wall_seconds=30,
            starting_checkpoint=None,
        )
        values = {
            "idempotency_key": "turn-hourly",
            "generation": 0,
            "slot": "slot-00",
            "phase": "initial",
            "state": "completed",
            "usage": {
                "inputTokens": 600_000,
                "cachedInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 400_000,
                "reasoningOutputTokens": 200_000,
                "totalTokens": 1_000_000,
                "final": True,
                "partial": False,
            },
        }
        assert state.record_provider_turn(**values)
        assert not state.record_provider_turn(**values)
        charges = state.connection.execute(
            "SELECT payload_json FROM events "
            "WHERE event_type='model_token_charge_recorded'"
        ).fetchall()
        assert len(charges) == 1
        charge = json.loads(str(charges[0]["payload_json"]))
        assert charge["token_delta"] == 1_000_000
        assert charge["usage"]["reasoningOutputTokens"] == 200_000
        at_limit = state.hourly_token_usage(1_000_000, now=started)
        assert at_limit["hourly_tokens_used"] == 1_000_000
        assert at_limit["hourly_limit_reached"] is True
        assert at_limit["hourly_retry_after"] == (
            started + timedelta(hours=1)
        ).isoformat()
        expired = state.hourly_token_usage(
            1_000_000,
            now=started + timedelta(hours=1, microseconds=1),
        )
        assert expired["hourly_tokens_used"] == 0
        assert expired["hourly_limit_reached"] is False

def test_status_is_versioned_and_read_only(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    before = experiment_status(path)
    assert before["schema_version"] == STATUS_SCHEMA_VERSION
    assert before["state"] == "not_created"
    ExperimentService(adapter=NullExperimentAdapter()).run(path)
    after = experiment_status(path)
    assert after["state"] == "idle"
    assert after["provider_turns"] == 0
    rendered = render_status(after)
    assert "Results: 0 accepted candidates, 0 evaluations" in rendered
    assert "Winner: none — no candidate was accepted" in rendered
    assert "Tokens: input 0, cached 0, output 0, reasoning 0, total 0" in rendered


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
    source_path = tmp_path / "program-1.py"
    source_path.write_text("def priority(ctx, proposal):\n    return 1.0\n", encoding="utf-8")

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
                archive_path=str(source_path),
                generation=2,
                slot="slot-01",
                status="created",
                metadata={"search_metrics": {"pooled_median_auc": 0.75}},
            )
            state.record_evaluation(
                "program-1:development",
                candidate_id="program-1",
                kind="development",
                state="completed",
                result={"summary": {"mean_auc": 0.75}},
            )
            state.record_provider_turn(
                idempotency_key="turn-1",
                generation=2,
                slot="slot-01",
                phase="initial",
                state="completed",
                usage={
                    "inputTokens": 10,
                    "cachedInputTokens": 2,
                    "outputTokens": 5,
                    "reasoningOutputTokens": 3,
                    "totalTokens": 15,
                    "final": True,
                    "partial": False,
                },
            )
            return {"state": "idle"}

    ExperimentService(adapter=CandidateAdapter()).run(path)
    status = experiment_status(path)
    assert status["best_program_id"] == "program-1"
    assert status["best_primary_metric"] == 0.75
    assert status["winner_source"] == str(source_path)
    assert status["evaluation_count"] == 1
    assert status["ranked_candidates"] == [
        {
            "candidate_id": "program-1",
            "metric": 0.75,
            "generation": 2,
            "slot": "slot-01",
            "status": "created",
            "source_path": str(source_path),
        }
    ]
    assert status["token_usage"] == {
        "inputTokens": 10,
        "cachedInputTokens": 2,
        "cacheWriteInputTokens": 0,
        "outputTokens": 5,
        "reasoningOutputTokens": 3,
        "totalTokens": 15,
        "quality": "exact",
        "chargedFailedTurns": 0,
    }
    rendered = render_status(status)
    assert "Winner: program-1, primary metric 0.75" in rendered
    assert f"Winner code: {source_path}" in rendered
    assert "Best mutations:" in rendered
    assert "Tokens: input 10, cached 2, output 5, reasoning 3, total 15 (exact)" in rendered


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
    response = {
        "schema_version": "mforge.native.generated_policy.v1",
        "source": "def priority(ctx, proposal):\n    return 0\n",
        "design_summary": "A bounded deterministic ranker.",
        "hypothesis": "Structural proposal signals improve selection.",
        "used_fields": ["proposal.k"],
        "assumptions": ["The host supplies legal proposals."],
        "expected_failure_modes": ["Constant ranking may underperform."],
    }
    return {
        "request_text": "rendered prompt",
        "response_text": json.dumps(response, ensure_ascii=False, sort_keys=True),
        "response": response,
        "source": response["source"],
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
        "canonical_response": response,
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


def test_turn_prompt_is_exact_and_response_is_semantic_projection(tmp_path: Path) -> None:
    store = TurnArtifactStore(tmp_path / "artifacts")
    response = {
        "schema_version": "mforge.native.generated_policy.v1",
        "source": "def priority(ctx, proposal):\n    return 0\n",
        "design_summary": "A bounded deterministic ranker.",
        "hypothesis": "Structural proposal signals improve selection.",
        "used_fields": ["proposal.k"],
        "assumptions": ["The host supplies legal proposals."],
        "expected_failure_modes": ["Constant ranking may underperform."],
    }
    request_text = '{"brief":"native context"}'
    response_text = json.dumps(response, ensure_ascii=False, sort_keys=True)
    store.write_turn(
        generation=0,
        slot=0,
        request={"brief": "native context"},
        request_text=request_text,
        response=response,
        response_text=response_text,
    )
    directory = store.turn_directory(0, 0)
    request_markdown = (directory / "slot-00.request.md").read_text(encoding="utf-8")
    response_markdown = (directory / "slot-00.response.md").read_text(encoding="utf-8")
    assert request_markdown == request_text
    assert response_markdown.startswith("# Generated policy\n\n")
    assert "## Design summary" in response_markdown
    assert "```python\ndef priority(ctx, proposal):" in response_markdown
    assert (directory / "slot-00.response.raw.txt").read_text() == response_text
    assert json.loads((directory / "slot-00.response.json").read_text()) == response
    request_json = json.loads((directory / "slot-00.request.json").read_text(encoding="utf-8"))
    assert request_json["brief"] == "native context"


def test_invalid_response_retains_raw_text_and_diagnostics_without_projection(
    tmp_path: Path,
) -> None:
    store = TurnArtifactStore(tmp_path / "artifacts")
    raw = '{"source": "unterminated"'
    store.write_turn(
        generation=0,
        slot=0,
        request_text="# Native task\n",
        response_text=raw,
        response=raw,
        response_diagnostics=({"code": "invalid_json", "message": "unterminated"},),
        content_received=True,
    )
    directory = store.turn_directory(0, 0)
    assert not (directory / "slot-00.response.md").exists()
    assert (directory / "slot-00.response.raw.txt").read_text() == raw
    diagnostics = json.loads((directory / "slot-00.response-diagnostics.json").read_text())
    assert diagnostics[0]["code"] == "invalid_json"
    assert store.verify_turn(directory)


def test_incomplete_turn_fails_closed_but_retains_manifest(tmp_path: Path) -> None:
    store = TurnArtifactStore(tmp_path / "artifacts", max_bytes=4)
    with pytest.raises(ArtifactIncompleteError):
        store.write_turn(generation=0, slot=0, request_text="too long")
    manifest = json.loads(
        (store.turn_directory(0, 0) / "turn-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_complete"] is False


def test_retry_archives_incomplete_turn_manifest(tmp_path: Path) -> None:
    store = TurnArtifactStore(tmp_path / "artifacts", max_bytes=4)
    with pytest.raises(ArtifactIncompleteError):
        store.write_turn(generation=0, slot=0, request_text="too long")
    directory = store.turn_directory(0, 0)

    archived = store.archive_retryable_manifest(directory)

    assert archived.name == "turn-manifest.attempt-01.json"
    assert json.loads(archived.read_text(encoding="utf-8"))["artifact_complete"] is False
    assert not (directory / "turn-manifest.json").exists()
    assert (
        store.artifact_prefix(
            directory,
            {"artifact_refs": ["slot-00.retry-01.request.md"]},
            "slot-00",
        )
        == "slot-00.retry-01"
    )


def test_native_transport_uses_per_turn_limit_and_retry_prefix(tmp_path: Path) -> None:
    artifact_root = tmp_path / "all-artifacts"
    turn = artifact_root / "generations" / "generation-0000" / "slot-00" / "initial"
    turn.mkdir(parents=True)
    (turn / "slot-00.request.md").write_text("partial", encoding="utf-8")
    transport = _CodexTransport(
        NativeProviderConfig(),
        auth_json=None,
        process_factory=None,
        auth_checker=lambda _capsule: True,
        sandbox_mode="danger-full-access",
        approval_policy="never",
    )

    adapter = transport._adapter(
        {
            "artifact_dir": str(turn),
            "artifact_root": str(artifact_root),
            "artifact_prefix": "slot-00",
            "system_prompt": "Return one generated policy object.",
        }
    )
    try:
        assert adapter.logger is not None
        assert adapter.logger.prefix == "slot-00.retry-01"
        assert adapter.logger.aggregate_root == turn.resolve()
    finally:
        adapter.close(force=True)


def test_artifact_manifest_ignores_atomic_write_temporary_files(tmp_path: Path) -> None:
    layout = ExperimentLayout(tmp_path, "manifest-temporary-files")
    layout.artifacts.mkdir(parents=True)
    (layout.artifacts / "result.json").write_text("{}", encoding="utf-8")
    (layout.artifacts / ".log.interrupted").write_text("partial", encoding="utf-8")

    manifest = layout.write_artifact_manifest()

    assert [entry["path"] for entry in manifest["files"]] == ["result.json"]
