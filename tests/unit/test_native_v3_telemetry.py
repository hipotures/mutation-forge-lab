from __future__ import annotations

from mutation_forge.events import Event
from mutation_forge.native_v3.scheduler import TelemetryEvent
from mutation_forge.native_v3.telemetry import (
    Bottleneck,
    summarize_scheduler_telemetry,
)
from mutation_forge.output.interactive_dashboard import (
    DashboardState,
    reduce_dashboard_event,
)


def test_dashboard_metrics_distinguish_provider_bound_execution() -> None:
    events = (
        TelemetryEvent("provider_call_started", 0, {"provider_calls_in_flight": 1}),
        TelemetryEvent("cpu_idle_provider_starvation_started", 10),
        TelemetryEvent(
            "cpu_idle_provider_starvation_ended",
            90,
            {"idle_ns": 80},
        ),
        TelemetryEvent(
            "provider_call_completed",
            100,
            {
                "latency_ns": 100,
                "programs_returned": 4,
                "valid_programs": 4,
                "provider_calls_in_flight": 0,
            },
        ),
        TelemetryEvent("epoch_terminal", 110),
    )
    summary = summarize_scheduler_telemetry(
        events,
        provider_concurrency=1,
        evaluator_workers=4,
    )
    assert summary.bottleneck is Bottleneck.PROVIDER_BOUND
    assert summary.cpu_idle_provider_starvation_ns == 80
    assert summary.as_dict()["programs_returned_per_call"] == {
        "numerator": 4,
        "denominator": 1,
    }


def test_dashboard_metrics_can_report_persistence_bound_execution() -> None:
    events = (
        TelemetryEvent("provider_call_started", 0, {"provider_calls_in_flight": 1}),
        TelemetryEvent(
            "provider_call_completed",
            100,
            {"latency_ns": 10, "programs_returned": 1, "valid_programs": 1},
        ),
    )
    summary = summarize_scheduler_telemetry(
        events,
        provider_concurrency=1,
        evaluator_workers=1,
        persistence_wall_ns=80,
    )
    assert summary.bottleneck is Bottleneck.PERSISTENCE_BOUND


def test_interactive_dashboard_reduces_native_v3_metrics() -> None:
    state = reduce_dashboard_event(
        DashboardState(),
        Event(
            "mforge.experiment.events.v3",
            "2026-01-01T00:00:00+00:00",
            "run",
            "native_v3_metrics",
            {
                "bottleneck": "EVALUATOR_BOUND",
                "provider_utilization": {"numerator": 1, "denominator": 2},
                "evaluator_utilization": {"numerator": 3, "denominator": 4},
                "candidate_queue_depth": 2,
                "evaluation_shard_queue_depth": 9,
                "raw_graph_score_calls": 20,
                "unique_graph_scores": 15,
                "accepted_rewrites": 3,
                "score_cache_hit_rate": {"numerator": 1, "denominator": 4},
                "active_cpp_scorers": 4,
                "scorer_restarts": 1,
                "forbidden_fallback_count": 0,
            },
        ),
    )
    assert state.native_v3_bottleneck == "EVALUATOR_BOUND"
    assert state.provider_utilization == 0.5
    assert state.evaluator_utilization == 0.75
    assert state.evaluation_shard_queue_depth == 9


def test_dashboard_surfaces_native_v3_provider_call_failure() -> None:
    state = reduce_dashboard_event(
        DashboardState(),
        Event(
            "mforge.experiment.events.v3",
            "2026-01-01T00:00:00+00:00",
            "run",
            "provider_call_started",
            {"call_id": "epoch-1:provider:0000", "provider_calls_in_flight": 1},
        ),
    )
    state = reduce_dashboard_event(
        state,
        Event(
            "mforge.experiment.events.v3",
            "2026-01-01T00:00:01+00:00",
            "run",
            "provider_call_failed",
            {
                "call_id": "epoch-1:provider:0000",
                "error_type": "AuthenticationError",
                "error_message": "model.auth_json is not logged in",
                "provider_calls_in_flight": 0,
            },
        ),
    )

    assert state.active_provider_turns == 0
    assert state.provider_turns_attempted == 1
    assert state.phase == "provider error"
    assert state.status_message == "AuthenticationError: model.auth_json is not logged in"
    assert state.activity[-1].severity == "error"
    assert "model.auth_json is not logged in" in state.activity[-1].message


def test_dashboard_exposes_verification_backpressure_and_queue_depth() -> None:
    state = reduce_dashboard_event(
        DashboardState(),
        Event(
            "mforge.experiment.events.v3",
            "2026-01-01T00:00:00+00:00",
            "run",
            "verification_backpressure_started",
            {"verification_queue_depth": 16},
        ),
    )
    assert state.verification_queue_depth == 16
    assert state.verification_backpressure_active
    state = reduce_dashboard_event(
        state,
        Event(
            "mforge.experiment.events.v3",
            "2026-01-01T00:00:01+00:00",
            "run",
            "verification_backpressure_ended",
            {"verification_queue_depth": 0, "idle_ns": 1_000_000_000},
        ),
    )
    assert state.verification_queue_depth == 0
    assert not state.verification_backpressure_active
