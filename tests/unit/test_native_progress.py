from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from mutation_forge import cli
from mutation_forge.events import Event, JsonlSink
from mutation_forge.experiment.observer import ExperimentEventHub
from mutation_forge.output.rich_live import ProgressLineSink, RichLiveSink


class _FlushStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def _event(event_type: str, **payload: object) -> Event:
    return Event(
        schema_version="1.0",
        timestamp="2026-08-02T00:00:00+00:00",
        run_id="native-progress",
        event_type=event_type,
        payload=payload,
    )


def test_native_rich_field_coverage_and_profile() -> None:
    output = io.StringIO()
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    hub = ExperimentEventHub(
        "native-progress",
        [sink],
        profiling_enabled=True,
    )
    try:
        hub.emit(
            "preflight_started",
            experiment_id="native-progress",
            workspace="/tmp/native-progress",
        )
        hub.emit(
            "preflight_completed",
            experiment_id="native-progress",
            workspace="/tmp/native-progress",
            status="completed",
        )
        hub.emit(
            "session_started",
            experiment_id="native-progress",
            workspace="/tmp/native-progress",
            session_id="session-000001",
            run_mode="fresh",
            configured_wall_seconds=30.0,
            remaining_seconds=29.5,
            model="gpt-test",
            effort="high",
            configured_concurrency=2,
            effective_concurrency=2,
            max_model_turns=10,
            remaining_model_turns=10,
        )
        hub.emit(
            "generation_started",
            generation=0,
            generation_limit=2,
            population_size=2,
            phase="initial",
        )
        hub.emit(
            "slot_queued",
            generation=0,
            slot="slot-00",
            status="running",
            parent_id="root",
            completed_slots=0,
            population_size=2,
        )
        hub.emit(
            "provider_turn_started",
            generation=0,
            slot="slot-00",
            phase="initial",
        )
        hub.emit(
            "provider_turn_completed",
            generation=0,
            slot="slot-00",
            phase="initial",
            usage={
                "inputTokens": 10,
                "cachedInputTokens": 2,
                "outputTokens": 5,
                "reasoningOutputTokens": 3,
                "totalTokens": 18,
            },
            usage_quality="exact",
        )
        hub.emit(
            "validation_completed",
            generation=0,
            slot="slot-00",
            valid=True,
            parse_outcome="valid",
            schema_outcome="valid",
        )
        hub.emit(
            "behavior_probe_completed",
            generation=0,
            slot="slot-00",
            valid=True,
        )
        hub.emit(
            "candidate_archived",
            generation=0,
            slot="slot-00",
            candidate_id="g0000-slot-00",
            status="accepted",
            archive_size=1,
        )
        hub.emit(
            "evaluation_started",
            candidate_id="g0000-slot-00",
            evaluations_queued=1,
        )
        hub.emit(
            "evaluation_progress",
            order=4,
            graph_seed=1,
            policy_seed=2,
            development_progress=0.5,
            replay_progress=0.25,
            evaluations=2,
            evaluations_per_second=4.0,
        )
        hub.emit(
            "evaluation_completed",
            candidate_id="g0000-slot-00",
            evaluations_completed=1,
            mean_auc=0.75,
            best_auc=0.8,
            current_objective=0.75,
            best_objective=0.8,
            best_candidate_id="g0000-slot-00",
            best_score=0.8,
            baseline_comparison={"random": 0.2},
        )
        hub.emit(
            "checkpoint_written",
            checkpoint="checkpoint-000001.json",
            completed_slots=1,
        )
        hub.emit(
            "budget_boundary_reached",
            state="idle",
            stop_reason="max_model_turns",
        )
        assert sink.state["state"] == "idle"
        hub.emit(
            "experiment_completed",
            state="completed",
            stop_reason="generation_limit",
            token_usage_delta=18,
            cumulative_tokens=18,
        )

        Console(file=output, force_terminal=False, width=240).print(sink._render())
        rendered = output.getvalue()
        for field in (
            "Experiment ID",
            "Workspace",
            "Session",
            "Generation",
            "Slots completed",
            "Provider turns",
            "Input tokens",
            "Reasoning tokens",
            "Candidates accepted",
            "Evaluation coordinates",
            "Development / replay",
            "Current / best objective",
            "Latest checkpoint",
            "g0000-slot-00",
            "Runtime profile",
            "Top hotspots",
        ):
            assert field in rendered
        assert sink.state["usage"] == {
            "inputTokens": 10,
            "cachedInputTokens": 2,
            "outputTokens": 5,
            "reasoningOutputTokens": 3,
            "totalTokens": 18,
            "quality": "exact",
        }
    finally:
        hub.close()
        sink.close()


def test_profiling_disabled_omits_profile_panel() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    hub = ExperimentEventHub("native-unprofiled", [sink], profiling_enabled=False)
    try:
        hub.emit("session_started", experiment_id="native-unprofiled")
        rendered = io.StringIO()
        Console(file=rendered, force_terminal=False, width=160).print(sink._render())
        assert "Runtime profile" not in rendered.getvalue()
        assert "Deep operator profile" not in rendered.getvalue()
        assert "Deep score profile" not in rendered.getvalue()
    finally:
        hub.close()
        sink.close()


def test_pre_session_events_are_replayed_to_durable_session_observer() -> None:
    class Session:
        session_id = "session-000001"

    class Manager:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def event(self, _session: Session, event_type: str, **payload: object) -> None:
            self.events.append((event_type, payload))

    manager = Manager()
    hub = ExperimentEventHub("native-durable")
    try:
        hub.emit("preflight_started", experiment_id="native-durable")
        hub.attach_session(manager, Session())
        hub.emit("session_started", experiment_id="native-durable")
        assert [event_type for event_type, _payload in manager.events] == [
            "preflight_started",
            "session_started",
        ]
    finally:
        hub.close()


def test_non_tty_progress_lines_flush_and_jsonl_is_parseable() -> None:
    stream = _FlushStream()
    ProgressLineSink(stream).write(
        _event(
            "evaluation_progress",
            generation=1,
            slot="slot-00",
            evaluations_completed=3,
            best_score=0.9,
        )
    )
    assert stream.flushes == 1
    assert "evaluation_progress" in stream.getvalue()
    assert "generation=1" in stream.getvalue()

    json_stream = io.StringIO()
    JsonlSink(json_stream).write(_event("evaluation_progress", evaluations=3))
    parsed = json.loads(json_stream.getvalue())
    assert parsed["event_type"] == "evaluation_progress"
    assert parsed["evaluations"] == 3


def test_json_cli_keeps_one_final_stdout_object_and_progress_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "experiment.toml"
    config.write_text(
        """schema_version = "mforge.experiment.v1"
exp_id = "json-progress"
workspace = "./workspace"
kind = "ranker-search"
preset = "heg-ranker-evolution-v1"

[run]
wall_seconds = 30
output = "json"

[model]
provider = "codex"
name = "gpt-test"
effort = "high"
concurrency = 1
max_repairs = 0

[search]
population_size = 1
max_generations = 1
max_model_turns = 1
selection = "elite-diversity"

[evaluation]
orders = [4]
graph_seeds = [1]
policy_seeds = [2]
horizon = 1
proposal_pool_size = 2
baselines = ["random"]
replay = false

[resources]
workers = 1
thread_count = 1
""",
        encoding="utf-8",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    def fake_run(
        *_args: object, event_sinks: list[object], **_kwargs: object
    ) -> dict[str, object]:
        event = _event("evaluation_progress", evaluations=1)
        for sink in event_sinks:
            sink.write(event)  # type: ignore[attr-defined]
        return {"status": "completed", "state": "completed", "exp_id": "json-progress"}

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    assert cli._experiment_run(config, json_output=True) == 0

    final = json.loads(stdout.getvalue())
    assert final == {
        "exp_id": "json-progress",
        "state": "completed",
        "status": "completed",
    }
    progress = json.loads(stderr.getvalue())
    assert progress["event_type"] == "evaluation_progress"
