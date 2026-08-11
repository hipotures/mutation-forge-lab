from __future__ import annotations

import io
import os
import pty
import re
import termios
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.panel import Panel

from mutation_forge import cli
from mutation_forge.events import Event
from mutation_forge.experiment.json_io import write_json
from mutation_forge.experiment.state import ExperimentStateStore
from mutation_forge.output import interactive_dashboard as dashboard
from mutation_forge.output.display_ids import compact_display_ids
from mutation_forge.output.interactive_dashboard import (
    DashboardCapabilities,
    DashboardSlot,
    DashboardState,
    GenerationSlots,
    InteractiveDashboardSink,
    TokenUsage,
    _decode_keys,
    _responsive_mode,
    _TerminalInput,
    reduce_dashboard_event,
    reduce_dashboard_key,
)


def _event(event_type: str, **payload: object) -> Event:
    return Event(
        schema_version="1.0",
        timestamp="2026-08-03T12:34:56+00:00",
        run_id="dashboard-run",
        event_type=event_type,
        payload=payload,
    )


def _running_state() -> DashboardState:
    state = DashboardState()
    state = reduce_dashboard_event(
        state,
        _event(
            "session_started",
            experiment_id="dashboard-run",
            session_id="session-000001",
            run_mode="fresh",
            graph_mode="unrestricted_min_degree_3",
            configured_wall_seconds=7200.0,
            model="gpt-test",
            effort="high",
            configured_concurrency=2,
            worker_count=8,
            active_workers=0,
            population_size=8,
            generation_limit=4,
            max_model_turns=64,
            model_turns_used=13,
            cumulative_provider_turns=12,
            profiling_enabled=True,
            usage={
                "inputTokens": 100,
                "cachedInputTokens": 20,
                "outputTokens": 50,
                "reasoningOutputTokens": 10,
                "totalTokens": 160,
                "quality": "exact",
            },
            session_usage={
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
                "totalTokens": 0,
                "quality": "unknown",
            },
        ),
        monotonic=100.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "generation_started",
            generation=1,
            generation_limit=4,
            population_size=8,
            configured_concurrency=2,
            effective_concurrency=2,
            max_model_turns=64,
            phase="initial",
        ),
        monotonic=101.0,
    )
    for index in range(8):
        state = reduce_dashboard_event(
            state,
            _event(
                "slot_queued",
                generation=1,
                slot=f"slot-{index:02d}",
                parent_id="root",
                phase="initial",
                status="queued",
                completed_slots=0,
                population_size=8,
            ),
            monotonic=102.0 + index,
        )
    state = reduce_dashboard_event(
        state,
        _event(
            "provider_turn_started",
            generation=1,
            slot="slot-00",
            parent_id="root",
            phase="initial",
            timeout_seconds=120.0,
            provider_turn_id="turn-0001",
        ),
        monotonic=110.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "provider_turn_failed",
            generation=1,
            slot="slot-00",
            phase="initial",
            provider_turn_id="turn-0001",
            idempotency_key="turn-0001",
            status="infrastructure",
            accepted=False,
            error="app-server EOF",
            charged=False,
            content=False,
            uncharged=True,
            usage={
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
                "totalTokens": 0,
            },
            usage_quality="exact",
        ),
        monotonic=111.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "validation_started",
            generation=1,
            slot="slot-01",
            phase="validation",
        ),
        monotonic=111.5,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "validation_completed",
            generation=1,
            slot="slot-01",
            phase="validation",
            valid=False,
            validation_codes=["forbidden_input_field"],
            errors=[
                {
                    "code": "forbidden_input_field",
                    "message": "policy reads a forbidden input field",
                }
            ],
        ),
        monotonic=112.0,
    )
    return state


def _adaptive_evaluation_config() -> dict[str, object]:
    return {
        "evaluation": {
            "graph_mode": "unrestricted_min_degree_3",
            "order_schedule": "adaptive",
            "min_order": 22,
            "max_order": 128,
            "orders_per_generation": 5,
            "graph_seeds": [401, 402, 403, 404],
            "policy_seeds": list(range(4001, 4017)),
        }
    }


@pytest.mark.parametrize(
    ("raw", "displayed"),
    (
        ("slot-00", "s00"),
        ("parent-0-slot-02", "p0-s02"),
        ("g0000-slot-00", "g0000-s00"),
        (
            "failed while evaluating g0007-slot-12",
            "failed while evaluating g0007-s12",
        ),
    ),
)
def test_compact_display_ids(raw: str, displayed: str) -> None:
    assert compact_display_ids(raw) == displayed


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    (
        (150, 55, "full"),
        (140, 48, "full"),
        (150, 47, "compact"),
        (120, 35, "compact"),
        (109, 55, "minimal"),
        (150, 31, "minimal"),
        (90, 24, "minimal"),
    ),
)
def test_responsive_mode_uses_the_stricter_dimension(
    width: int, height: int, expected: str
) -> None:
    assert _responsive_mode(width, height) == expected


def test_header_zebra_and_horizontal_progress_keep_parameter_groups_together() -> None:
    header = dashboard._parameter_line(
        (
            ("Run", "demo", None),
            ("State", "RUNNING", "bold cyan"),
            ("Gen", "1/4", None),
            ("Phase", "development", None),
        )
    )
    assert header.plain == "Run demo  State RUNNING  Gen 1/4  Phase development"
    assert [str(span.style) for span in header.spans] == [
        "grey62",
        "bold cyan",
        "grey62",
    ]

    output = io.StringIO()
    Console(file=output, width=40, force_terminal=False, color_system=None).print(
        dashboard._progress_bar(
            "Generation",
            1,
            4,
            width=8,
            stacked=True,
        )
    )
    lines = output.getvalue().splitlines()
    assert lines[0].strip() == "Generation"
    assert "1/4" in lines[1]
    assert len(lines[1].rstrip()) < 25


def test_performance_times_use_whole_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 1047.4)
    monkeypatch.setattr(
        dashboard.resource,
        "getrusage",
        lambda _scope: SimpleNamespace(ru_utime=296.7, ru_stime=245.4),
    )
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        start_live=False,
    )
    sink.state = _running_state()
    output = io.StringIO()
    Console(
        file=output,
        width=60,
        force_terminal=False,
        color_system=None,
    ).print(sink._performance_panel())
    assert "947/296/245s" in output.getvalue()
    sink.close()


def test_token_accounting_groups_rows_without_extra_separator_lines() -> None:
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        start_live=False,
    )
    sink.state = _running_state()
    output = io.StringIO()
    Console(
        file=output,
        width=60,
        force_terminal=False,
        color_system=None,
    ).print(sink._tokens_panel())
    rendered = output.getvalue()
    assert rendered.count("experiment") == 1
    assert rendered.count("session") == 1
    assert rendered.count("usage") == 1
    assert "total" in rendered
    assert "input" in rendered
    assert "quality" in rendered
    assert "reasoning" in rendered
    assert "reasoning (in output)" not in rendered
    assert len(rendered.splitlines()) == 13
    sink.close()


def test_event_reducer_keeps_authoritative_counts_and_deduplicates_tokens() -> None:
    state = _running_state()
    assert state.active_provider_turns == 0
    assert state.evaluation_workers_active == 0
    assert state.evaluation_workers_configured == 8
    assert state.provider_turns_attempted == 14
    assert state.provider_turns_completed == 12
    assert state.cumulative_usage.total == 160
    assert state.session_usage.total == 0
    slot_00 = state.generations[0].slots[0]
    slot_01 = state.generations[0].slots[1]
    assert slot_00.state == "failed"
    assert slot_00.retryable is True
    assert slot_00.error == "app-server EOF"
    assert slot_01.state == "invalid"
    assert slot_01.validation == "forbidden_input_field"
    assert "forbidden input field" in slot_01.validation_message

    checkpointed = reduce_dashboard_event(
        state,
        _event(
            "checkpoint_written",
            generation=1,
            completed_slots=44,
        ),
        monotonic=119.0,
    )
    assert checkpointed.completed_slots == 0
    state = reduce_dashboard_event(
        checkpointed,
        _event(
            "slot_queued",
            generation=1,
            slot="slot-02",
            status="accepted",
            completed_slots=3,
            population_size=8,
        ),
        monotonic=119.5,
    )
    assert state.completed_slots == 3

    completed = _event(
        "provider_turn_completed",
        generation=1,
        slot="slot-02",
        phase="initial",
        idempotency_key="deduplicated-turn",
        usage={
            "inputTokens": 10,
            "cachedInputTokens": 2,
            "outputTokens": 5,
            "reasoningOutputTokens": 3,
            "totalTokens": 18,
        },
        usage_quality="exact",
        provider_duration_ms=412859,
    )
    once = reduce_dashboard_event(state, completed, monotonic=120.0)
    twice = reduce_dashboard_event(once, completed, monotonic=121.0)
    assert once.cumulative_usage.total == 178
    assert twice.cumulative_usage.total == 178
    assert once.session_usage.total == 18
    assert once.generations[0].slots[2].elapsed_seconds == pytest.approx(16.0)
    assert twice.session_usage.total == 18

    archived = _event(
        "candidate_archived",
        generation=1,
        slot="slot-03",
        status="invalid",
        archive_size=1,
    )
    once_archived = reduce_dashboard_event(twice, archived, monotonic=122.0)
    twice_archived = reduce_dashboard_event(once_archived, archived, monotonic=123.0)
    assert once_archived.invalid_candidates == 1
    assert twice_archived.invalid_candidates == 1

    evaluated = reduce_dashboard_event(
        twice_archived,
        _event(
            "evaluation_completed",
            evaluations_completed=7,
            current_objective=0.5,
            best_objective=0.5,
        ),
        monotonic=124.0,
    )
    assert evaluated.evaluations_completed == 7


def test_validation_diagnostics_require_started_work_and_clear_for_new_turn() -> None:
    state = reduce_dashboard_event(
        DashboardState(),
        _event(
            "slot_queued",
            generation=0,
            slot="slot-00",
            phase="initial",
            status="queued",
        ),
        monotonic=1.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "validation_completed",
            generation=0,
            slot="slot-00",
            phase="initial",
            valid=False,
            validation_codes=["turn_provenance", "invalid_output"],
        ),
        monotonic=2.0,
    )
    untouched = state.generations[0].slots[0]
    assert untouched.state == "queued"
    assert untouched.validation == "—"
    assert untouched.error == ""

    state = reduce_dashboard_event(
        state,
        _event(
            "validation_started",
            generation=0,
            slot="slot-00",
            phase="initial",
        ),
        monotonic=3.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "validation_completed",
            generation=0,
            slot="slot-00",
            phase="initial",
            valid=False,
            validation_codes=["invalid_output"],
        ),
        monotonic=4.0,
    )
    invalid = state.generations[0].slots[0]
    assert invalid.state == "invalid"
    assert invalid.validation == "invalid_output"

    state = reduce_dashboard_event(
        state,
        _event(
            "provider_turn_started",
            generation=0,
            slot="slot-00",
            phase="initial",
        ),
        monotonic=5.0,
    )
    model = state.generations[0].slots[0]
    assert model.state == "model"
    assert model.validation == "—"
    assert model.validation_message == ""
    assert model.error == ""


def test_counterexample_lifecycle_is_idempotent_and_terminal() -> None:
    state = DashboardState()
    candidate = _event(
        "counterexample_candidate_found",
        candidate_id="cx-" + "a" * 64,
        order=30,
        target_forbidden_lengths=[4, 8, 16],
        idempotency_key="candidate",
    )
    state = reduce_dashboard_event(state, candidate)
    repeated = reduce_dashboard_event(state, candidate)
    assert repeated == state
    assert state.counterexample_state == "candidate"
    assert state.counterexample_lengths == (4, 8, 16)

    state = reduce_dashboard_event(
        state,
        _event(
            "counterexample_primary_verification_completed",
            status="VERIFIED",
            complete=True,
            idempotency_key="primary",
        ),
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "counterexample_independent_verification_completed",
            status="VERIFIED",
            complete=True,
            idempotency_key="independent",
        ),
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "counterexample_verified",
            candidate_id="cx-" + "a" * 64,
            certificate="/tmp/certificate.json",
            idempotency_key="verified",
        ),
    )
    assert state.counterexample_state == "verified"
    assert state.experiment_state == "completed"


def test_recovered_slot_hydrates_usage_without_changing_aggregate_totals() -> None:
    state = _running_state()
    recovered = reduce_dashboard_event(
        state,
        _event(
            "slot_queued",
            generation=1,
            slot="slot-02",
            parent_id="g0000-slot-01",
            phase="initial",
            status="recovered",
            recovered=True,
            recovered_status="accepted",
            usage={
                "inputTokens": 101,
                "cachedInputTokens": 11,
                "outputTokens": 53,
                "reasoningOutputTokens": 47,
                "totalTokens": 154,
                "quality": "exact",
            },
            usage_quality="exact",
            candidate_id="g0001-slot-02",
            validation_status="passed",
            probe_status="passed",
            charged=False,
        ),
        monotonic=124.0,
    )

    slot = recovered.generations[0].slots[2]
    assert slot.state == "accepted"
    assert slot.candidate == "g0001-slot-02"
    assert slot.validation == "pass"
    assert slot.probe == "pass"
    assert slot.charged is False
    assert slot.usage == TokenUsage(
        input=101,
        cached=11,
        output=53,
        reasoning=47,
        total=154,
        quality="exact",
    )
    assert recovered.cumulative_usage == state.cumulative_usage
    assert recovered.session_usage == state.session_usage


def test_successful_repair_clears_superseded_error() -> None:
    state = _running_state()
    state = reduce_dashboard_event(
        state,
        _event(
            "repair_started",
            generation=1,
            slot="slot-01",
            phase="repair",
            repair_attempt=1,
        ),
        monotonic=120.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "provider_turn_completed",
            generation=1,
            slot="slot-01",
            phase="repair",
            accepted=True,
            status="completed",
        ),
        monotonic=121.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "validation_completed",
            generation=1,
            slot="slot-01",
            phase="repair",
            valid=True,
        ),
        monotonic=122.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "behavior_probe_completed",
            generation=1,
            slot="slot-01",
            phase="repair",
            valid=True,
        ),
        monotonic=123.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "repair_completed",
            generation=1,
            slot="slot-01",
            phase="repair",
            status="accepted",
            repairs=1,
        ),
        monotonic=124.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "candidate_archived",
            generation=1,
            slot="slot-01",
            candidate_id="g0001-slot-01",
            status="accepted",
        ),
        monotonic=125.0,
    )

    slot = state.generations[0].slots[1]
    assert slot.state == "accepted"
    assert slot.error == ""
    assert slot.candidate == "g0001-slot-01"
    assert slot.validation == "pass"
    assert slot.probe == "pass"


def test_evaluation_elapsed_is_per_slot_and_does_not_replace_run_elapsed() -> None:
    state = replace(_running_state(), elapsed_seconds=360.0)
    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_started",
            generation=1,
            slot="slot-02",
            phase="development",
            evaluation_id="g0001-slot-02:development",
            evaluation_total=128,
        ),
        monotonic=120.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_progress",
            generation=1,
            slot="slot-02",
            phase="replay",
            evaluation_id="g0001-slot-02:development",
            completed=29,
            total=128,
            order=12,
            graph_seed=403,
            policy_seed=4009,
            evaluations_per_second=2.5,
            elapsed_seconds=11.0,
            current_objective=0.42,
            **{"pass": "replay"},
        ),
        monotonic=131.0,
    )
    active = state.generations[0].slots[2]
    assert state.elapsed_seconds == 360.0
    assert active.started_monotonic == 104.0
    assert active.evaluation_completed == 29
    assert active.evaluation_total == 128
    assert active.evaluation_pass == "replay"
    assert active.evaluation_order == 12
    assert active.graph_seed == 403
    assert active.policy_seed == 4009
    assert active.evaluation_rate == 2.5
    assert active.objective == pytest.approx(0.42)

    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        start_live=False,
    )
    sink.state = state
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._slot_matrix(150, "full"))
    assert "eval 23%" in output.getvalue()
    assert "evaluating" not in output.getvalue()
    sink.close()

    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_completed",
            generation=1,
            slot="slot-02",
            phase="development",
            evaluation_id="g0001-slot-02:development",
            elapsed_seconds=12.5,
        ),
        monotonic=132.5,
    )
    completed = state.generations[0].slots[2]
    assert state.elapsed_seconds == 360.0
    assert completed.started_monotonic is None
    assert completed.elapsed_seconds == 28.5


def test_slot_matrix_shows_independent_parallel_evaluation_progress() -> None:
    state = _running_state()
    for slot, completed in (("slot-02", 176), ("slot-03", 90), ("slot-04", 320)):
        state = reduce_dashboard_event(
            state,
            _event(
                "evaluation_started",
                generation=1,
                slot=slot,
                phase="development",
                evaluation_total=320,
            ),
            monotonic=120.0,
        )
        state = reduce_dashboard_event(
            state,
            _event(
                "evaluation_progress",
                generation=1,
                slot=slot,
                phase="development",
                completed=completed,
                total=320,
            ),
            monotonic=125.0,
        )

    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        start_live=False,
    )
    sink.state = state
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._slot_matrix(150, "full"))
    rendered = output.getvalue()

    assert "eval 55%" in rendered
    assert "eval 28%" in rendered

    sink.state = replace(state, slot_icon_mode=True)
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._slot_matrix(150, "full"))
    rendered = output.getvalue()

    assert "55%" in rendered
    assert "28%" in rendered
    assert "100%" in rendered
    assert "▲" not in rendered
    assert "eval " not in rendered
    sink.close()


def test_slot_elapsed_covers_every_phase_from_provider_start_to_evaluation_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = DashboardState()
    state = reduce_dashboard_event(
        state,
        _event(
            "slot_queued",
            generation=0,
            slot="slot-00",
            phase="queued",
        ),
        monotonic=0.0,
    )
    queued = state.generations[0].slots[0]
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 0.5)
    assert InteractiveDashboardSink._slot_elapsed(queued) == pytest.approx(0.5)
    events = (
        (1.0, "provider_turn_started", {}),
        (6.0, "provider_turn_completed", {"accepted": True}),
        (7.0, "validation_started", {}),
        (9.0, "validation_completed", {"valid": True}),
        (10.0, "behavior_probe_started", {}),
        (13.0, "behavior_probe_completed", {"valid": True}),
        (
            14.0,
            "evaluation_started",
            {"evaluation_id": "g0000-slot-00:development"},
        ),
    )
    for monotonic, event_type, extra in events:
        state = reduce_dashboard_event(
            state,
            _event(
                event_type,
                generation=0,
                slot="slot-00",
                phase="development",
                **extra,
            ),
            monotonic=monotonic,
        )

    active = state.generations[0].slots[0]
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 20.0)
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        initial_state=state,
        start_live=False,
    )
    assert sink._slot_elapsed(active) == pytest.approx(20.0)  # noqa: SLF001
    sink.close()

    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_completed",
            generation=0,
            slot="slot-00",
            phase="development",
            evaluation_id="g0000-slot-00:development",
            elapsed_seconds=10.0,
        ),
        monotonic=24.0,
    )
    completed = state.generations[0].slots[0]
    assert completed.elapsed_seconds == pytest.approx(24.0)
    assert completed.started_monotonic is None
    assert completed.phase_started_monotonic is None
    lifecycle = {step.phase: step.elapsed_seconds for step in completed.lifecycle}
    assert lifecycle["response"] == pytest.approx(5.0)
    assert lifecycle["schema"] == pytest.approx(2.0)
    assert lifecycle["probe"] == pytest.approx(3.0)
    assert lifecycle["evaluation"] == pytest.approx(10.0)
    archived_state = reduce_dashboard_event(
        state,
        _event(
            "candidate_archived",
            generation=0,
            slot="slot-00",
            status="accepted",
        ),
        monotonic=25.0,
    )
    archived = archived_state.generations[0].slots[0]
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 30.0)
    assert archived.phase == "archived"
    assert archived.elapsed_seconds == pytest.approx(24.0)
    assert InteractiveDashboardSink._slot_elapsed(archived) == pytest.approx(24.0)


def test_evaluation_rate_sums_active_worker_rates() -> None:
    state = _running_state()
    for slot in ("slot-02", "slot-03"):
        state = reduce_dashboard_event(
            state,
            _event(
                "evaluation_started",
                generation=1,
                slot=slot,
                phase="development",
                evaluation_id=f"g0001-{slot}:development",
                evaluation_total=128,
            ),
            monotonic=120.0,
        )

    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_progress",
            generation=1,
            slot="slot-02",
            completed=10,
            total=128,
            evaluations_per_second=2.94,
        ),
        monotonic=121.0,
    )
    assert state.evaluation_rate == pytest.approx(2.94)

    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_progress",
            generation=1,
            slot="slot-03",
            completed=12,
            total=128,
            evaluations_per_second=1.75,
        ),
        monotonic=122.0,
    )
    assert state.evaluation_rate == pytest.approx(4.69)

    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_completed",
            generation=1,
            slot="slot-03",
            elapsed_seconds=20.0,
        ),
        monotonic=123.0,
    )
    assert state.evaluation_rate == pytest.approx(2.94)

    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_completed",
            generation=1,
            slot="slot-02",
            elapsed_seconds=21.0,
        ),
        monotonic=124.0,
    )
    assert state.evaluation_rate is None


def test_key_reducer_navigation_details_generations_and_retry_confirmation() -> None:
    current = GenerationSlots(
        1,
        (
            DashboardSlot("slot-00", 1),
            DashboardSlot("slot-01", 1, state="failed", retryable=True),
        ),
    )
    previous = GenerationSlots(0, (DashboardSlot("slot-00", 0),))
    state = DashboardState(
        generation=1,
        displayed_generation=1,
        generations=(previous, current),
    )
    state, action = reduce_dashboard_key(state, "DOWN")
    assert action is None
    assert state.selected_index == 1
    state, _ = reduce_dashboard_key(state, "DOWN")
    assert state.selected_index == 1
    state, _ = reduce_dashboard_key(state, "ENTER")
    assert state.view == "details"
    state, _ = reduce_dashboard_key(state, "TAB")
    assert state.detail_tab == 1
    state, _ = reduce_dashboard_key(state, "SHIFT_TAB")
    assert state.detail_tab == 0
    state, _ = reduce_dashboard_key(state, "ESC")
    assert state.view == "matrix"

    state, _ = reduce_dashboard_key(state, "r", retry_supported=True)
    assert state.retry_confirmation is True
    state, action = reduce_dashboard_key(state, "y", retry_supported=True)
    assert action is not None
    assert action.kind == "retry"
    assert action.slot == "slot-01"

    state, _ = reduce_dashboard_key(state, "LEFT")
    assert state.displayed_generation == 0
    assert state.generation == 1
    assert state.status_message == "Viewing generation 1"
    state, _ = reduce_dashboard_key(state, "LEFT")
    assert state.displayed_generation == 0
    state, _ = reduce_dashboard_key(state, "RIGHT")
    assert state.displayed_generation == 1
    assert state.generation == 1
    state, _ = reduce_dashboard_key(state, "RIGHT")
    assert state.displayed_generation == 1

    state, _ = reduce_dashboard_key(state, "LEFT")
    state, _ = reduce_dashboard_key(state, "ENTER")
    assert state.view == "details"
    state, _ = reduce_dashboard_key(state, "RIGHT")
    assert state.displayed_generation == 1
    assert state.view == "details"
    state, _ = reduce_dashboard_key(state, "ESC")
    assert state.view == "matrix"
    unchanged, _ = reduce_dashboard_key(state, "n")
    assert unchanged == state
    unchanged, _ = reduce_dashboard_key(state, "N")
    assert unchanged == state


def test_q_arms_graceful_stop_then_requests_immediate_interrupt() -> None:
    state = DashboardState(
        experiment_state="running",
        generation=1,
        displayed_generation=1,
        generations=(
            GenerationSlots(
                1,
                (
                    DashboardSlot("slot-00", 1, state="evaluating"),
                    DashboardSlot("slot-01", 1, state="accepted"),
                    DashboardSlot("slot-02", 1, state="queued"),
                ),
            ),
        ),
    )

    state, action = reduce_dashboard_key(state, "q")

    assert action is not None and action.kind == "quit"
    assert action.immediate is False
    assert state.experiment_state == "stopping"
    assert state.generations[0].slots[0].state == "evaluating"
    assert state.generations[0].slots[1].state == "accepted"
    assert state.generations[0].slots[2].state == "stopping"
    assert "press q again" in state.status_message

    state, action = reduce_dashboard_key(state, "q")
    assert action is not None and action.kind == "quit"
    assert action.immediate is True

    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=True),
        start_live=False,
    )
    sink.state = state
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=True,
        color_system="standard",
    ).print(sink._slot_matrix(150, "full"))
    rendered = output.getvalue()
    assert "stoppi" in rendered
    assert re.search(r"\x1b\[(?:\d+;)*5(?:;\d+)*m", rendered)
    sink.close()

    state, action = reduce_dashboard_key(state, "q")

    assert action is not None and action.kind == "quit"
    assert state.status_message == "Immediate interrupt requested"


def test_dashboard_q_dispatches_second_press_to_immediate_interrupt() -> None:
    graceful = Mock()
    interrupt = Mock()
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=True),
        capabilities=DashboardCapabilities(
            quit=graceful,
            interrupt=interrupt,
        ),
        start_live=False,
    )
    try:
        sink.handle_key("q")
        sink.handle_key("q")
    finally:
        sink.close()

    graceful.assert_called_once_with()
    interrupt.assert_called_once_with()


def test_persisted_dashboard_hydrates_previous_generations_and_objectives(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    state_path = root / "state.sqlite3"
    ExperimentStateStore.initialize(
        state_path,
        exp_id="dashboard-run",
        lock_hash="0" * 64,
        root=root,
    )
    with ExperimentStateStore(state_path) as store:
        store.record_candidate(
            "g0000-slot-00",
            generation=0,
            slot="slot-00",
            status="created",
        )
        store.record_evaluation(
            "g0000-slot-00:development",
            candidate_id="g0000-slot-00",
            kind="development",
            state="completed",
            result={
                "summary": {"mean_auc": 0.75},
                "runtime": {"elapsed_seconds": 12.5},
            },
        )
        store.write_event(
            "provider_turn_started",
            {"generation": 0, "slot": "slot-00"},
            idempotency_key="slot-runtime-start",
        )
        store.write_event(
            "evaluation_completed",
            {"generation": 0, "slot": "slot-00"},
            idempotency_key="slot-runtime-end",
        )
        store.connection.execute(
            "UPDATE events SET timestamp=? WHERE idempotency_key=?",
            ("2026-08-03T12:00:00+00:00", "slot-runtime-start"),
        )
        store.connection.execute(
            "UPDATE events SET timestamp=? WHERE idempotency_key=?",
            ("2026-08-03T12:01:00+00:00", "slot-runtime-end"),
        )
        store.connection.commit()

    checkpoint = root / "artifacts" / "native-generation-checkpoint.json.gz"
    checkpoint.parent.mkdir(parents=True)
    write_json(
        checkpoint,
        {
            "schema_version": "mforge.experiment.generation.v2",
            "generation": 1,
            "slots": {},
        },
    )
    persisted = dashboard.load_persisted_dashboard_state(
        root,
        run_id="dashboard-run",
        population_size=2,
    )

    assert [item.generation for item in persisted.generations] == [0, 1]
    assert persisted.generations[0].slots[0].objective == pytest.approx(0.75)
    assert persisted.generations[0].slots[0].elapsed_seconds == pytest.approx(60.0)
    assert persisted.best_objective == pytest.approx(0.75)
    assert persisted.best_candidate == "g0000-slot-00"

    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=120, force_terminal=False),
        initial_state=DashboardState(
            generation=1,
            displayed_generation=1,
            generations=(GenerationSlots(1, (DashboardSlot("slot-00", 1),)),),
        ),
        persisted_loader=lambda _generation_before: persisted,
        start_live=False,
    )
    sink.handle_key("LEFT")
    assert sink.state.displayed_generation == 0
    assert sink.state.generations[0].slots[0].objective == pytest.approx(0.75)
    sink.close()


def test_persisted_dashboard_uses_newest_retained_slot_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    checkpoint = root / "artifacts" / "native-generation-checkpoint.json.gz"
    checkpoint.parent.mkdir(parents=True)
    write_json(
        checkpoint,
        {
            "schema_version": "mforge.experiment.generation.v2",
            "generation": 1,
            "next_generation": 2,
            "slots": {
                "in-progress": {
                    "generation": 3,
                    "slot": "slot-00",
                    "parent_id": "g0002-slot-04",
                    "status": "pending",
                },
            },
        },
    )

    persisted = dashboard.load_persisted_dashboard_state(
        root,
        run_id="dashboard-run",
        population_size=2,
    )

    assert persisted.generation == 3
    assert persisted.displayed_generation == 3


def test_persisted_dashboard_loads_generations_in_pages_of_ten(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    checkpoint = root / "artifacts" / "native-generation-checkpoint.json.gz"
    checkpoint.parent.mkdir(parents=True)
    write_json(
        checkpoint,
        {
            "schema_version": "mforge.experiment.generation.v2",
            "generation": 14,
            "slots": {
                f"generation-{generation}": {
                    "generation": generation,
                    "slot": "slot-00",
                    "parent_id": "root",
                    "status": "accepted",
                    "candidate": {"source": "retained"},
                }
                for generation in range(15)
            },
        },
    )

    latest = dashboard.load_persisted_dashboard_state(
        root,
        run_id="dashboard-run",
    )
    previous = dashboard.load_persisted_dashboard_state(
        root,
        run_id="dashboard-run",
        generation_before=5,
    )

    assert [group.generation for group in latest.generations] == list(range(5, 15))
    assert latest.displayed_generation == 14
    assert [group.generation for group in previous.generations] == list(range(5))
    assert previous.generation == 14
    assert previous.displayed_generation == 4


def test_generation_navigation_loads_only_at_the_oldest_cached_page() -> None:
    calls: list[int | None] = []

    def load(generation_before: int | None) -> DashboardState:
        calls.append(generation_before)
        assert generation_before is not None
        page_start = max(0, generation_before - 10)
        return DashboardState(
            generation=19,
            displayed_generation=generation_before - 1,
            generations=tuple(
                GenerationSlots(
                    generation,
                    (DashboardSlot("slot-00", generation),),
                )
                for generation in range(page_start, generation_before)
            ),
        )

    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=120, force_terminal=False),
        initial_state=DashboardState(
            generation=19,
            displayed_generation=19,
            generations=tuple(
                GenerationSlots(
                    generation,
                    (DashboardSlot("slot-00", generation),),
                )
                for generation in range(10, 20)
            ),
        ),
        persisted_loader=load,
        start_live=False,
    )

    sink.handle_key("LEFT")
    assert sink.state.displayed_generation == 18
    assert calls == []

    sink.state = replace(sink.state, displayed_generation=10)
    sink.handle_key("LEFT")
    assert sink.state.displayed_generation == 9
    assert calls == [10]

    sink.handle_key("LEFT")
    sink.handle_key("RIGHT")
    assert calls == [10]
    sink.close()


def test_live_generation_cannot_lower_historical_best_objective() -> None:
    state = replace(
        _running_state(),
        best_objective=0.286023644324426,
        best_candidate="g0000-slot-06",
    )

    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_completed",
            generation=2,
            slot="slot-00",
            current_objective=0.00355113636,
            best_objective=0.00355113636,
            best_candidate_id="g0002-slot-00",
        ),
    )

    assert state.current_objective == pytest.approx(0.00355113636)
    assert state.best_objective == pytest.approx(0.286023644324426)
    assert state.best_candidate == "g0000-slot-06"


def test_resumed_generation_start_preserves_persisted_slot_objectives() -> None:
    retained = GenerationSlots(
        3,
        (
            DashboardSlot(
                "slot-00",
                3,
                phase="development",
                state="accepted",
                candidate="g0003-slot-00",
                objective=0.252683555,
            ),
        ),
    )
    state = DashboardState(
        generation=3,
        displayed_generation=3,
        population_size=8,
        generations=(retained,),
    )

    resumed = reduce_dashboard_event(
        state,
        _event("generation_started", generation=3, population_size=8),
    )

    slot = resumed.generations[0].slots[0]
    assert slot.state == "accepted"
    assert slot.phase == "development"
    assert slot.objective == pytest.approx(0.252683555)


def test_newer_slot_event_advances_current_generation_when_start_event_was_missed() -> None:
    state = DashboardState(
        generation=3,
        displayed_generation=3,
        population_size=8,
        generations=(
            GenerationSlots(
                3,
                tuple(DashboardSlot(f"slot-{index:02d}", 3) for index in range(8)),
            ),
        ),
    )

    updated = reduce_dashboard_event(
        state,
        _event(
            "evaluation_progress",
            generation=4,
            slot="slot-01",
            candidate_id="g0004-slot-01",
            phase="replay",
            completed=24,
            total=128,
        ),
    )

    assert updated.generation == 4
    assert updated.displayed_generation == 4
    generation = next(item for item in updated.generations if item.generation == 4)
    assert generation.slots[1].candidate == "g0004-slot-01"


def test_newer_slot_event_keeps_explicit_historical_generation_view() -> None:
    state = DashboardState(
        generation=3,
        displayed_generation=2,
        population_size=8,
        generations=(
            GenerationSlots(2, (DashboardSlot("slot-00", 2),)),
            GenerationSlots(3, (DashboardSlot("slot-00", 3),)),
        ),
    )

    updated = reduce_dashboard_event(
        state,
        _event(
            "evaluation_progress",
            generation=4,
            slot="slot-00",
            candidate_id="g0004-slot-00",
            completed=1,
            total=128,
        ),
    )

    assert updated.generation == 4
    assert updated.displayed_generation == 2


def test_key_reducer_overlays_search_and_disabled_scheduler_actions() -> None:
    state = _running_state()
    state, _ = reduce_dashboard_key(state, "c")
    assert state.view == "config"
    assert state.status_message == "Read-only locked configuration"
    state, _ = reduce_dashboard_key(state, "ESC")
    assert state.view == "matrix"
    assert state.status_message == ""
    state, _ = reduce_dashboard_key(state, "l")
    assert state.view == "logs"
    state, _ = reduce_dashboard_key(state, "h")
    assert state.view == "help"
    state, _ = reduce_dashboard_key(state, "/")
    assert state.search_editing is True
    for key in "slot-01":
        state, _ = reduce_dashboard_key(state, key)
    state, _ = reduce_dashboard_key(state, "ENTER")
    assert state.search_query == "slot-01"
    assert state.search_editing is False

    state = replace(state, search_query="accidental terminal text")
    state, _ = reduce_dashboard_key(state, "ESC")
    assert state.search_query == ""
    assert state.status_message == "Filter cleared"

    state, action = reduce_dashboard_key(state, "p")
    assert action is None
    assert "unavailable" in state.status_message
    state, action = reduce_dashboard_key(state, "r")
    assert action is None
    assert "unavailable" in state.status_message


def test_i_key_toggles_slot_phase_and_state_icons_without_mutating_data() -> None:
    state = _running_state()
    original_slot = state.generations[0].slots[0]

    icon_state, action = reduce_dashboard_key(state, "i")

    assert action is None
    assert icon_state.slot_icon_mode is True
    assert icon_state.status_message == "Slot phase/state: icons"
    assert icon_state.generations[0].slots[0].phase == original_slot.phase
    assert icon_state.generations[0].slots[0].state == original_slot.state

    text_state, action = reduce_dashboard_key(icon_state, "i")

    assert action is None
    assert text_state.slot_icon_mode is False
    assert text_state.status_message == "Slot phase/state: text"


def test_slot_icon_mode_uses_narrow_headers_but_copy_remains_text() -> None:
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        start_live=False,
    )
    state = _running_state()
    slots = list(state.generations[0].slots)
    slots[0] = replace(slots[0], phase="development", state="accepted")
    sink.state = replace(
        state,
        slot_icon_mode=True,
        generations=(GenerationSlots(1, tuple(slots)),),
    )

    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._slot_matrix(150, "full"))
    rendered = output.getvalue()

    assert " P " in rendered
    assert " S " in rendered
    assert "⋆" in rendered
    assert "✓" in rendered
    assert "development" not in rendered
    assert "accepted" not in rendered

    title, matrix = sink._panel_copy_source("slots")
    copied = dashboard.render_panel_copy_text(
        title,
        matrix,
        width=dashboard.PANEL_COPY_WIDTHS["slots"],
    )
    assert "phase" in copied
    assert "state" in copied
    assert "development" in copied
    assert "accepted" in copied
    sink.close()


def test_number_keys_request_stable_panel_copies_but_remain_search_text() -> None:
    state = _running_state()
    for key, panel in dashboard.PANEL_COPY_KEYS.items():
        state, action = reduce_dashboard_key(state, key)
        assert action == dashboard.DashboardAction("copy", panel=panel)

    state = replace(state, search_editing=True, search_query="")
    state, action = reduce_dashboard_key(state, "5")
    assert action is None
    assert state.search_query == "5"


def test_profiling_panel_is_numbered_and_copyable() -> None:
    state = replace(
        _running_state(),
        timing_profile={
            "phase_seconds": {"provider.generate": 2.0},
            "phase_calls": {"provider.generate": 1},
            "unattributed_fraction": 0.0,
        },
    )
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, height=55, force_terminal=False),
        start_live=False,
    )
    sink.state = state

    output = io.StringIO()
    Console(file=output, width=150, height=55, force_terminal=False).print(sink.render())
    profile_title = next(
        line for line in output.getvalue().splitlines() if "Profiling · top-N" in line
    )
    assert profile_title.endswith(" 7 ─╮")

    title, renderable = sink._panel_copy_source("profiling")
    copied = dashboard.render_panel_copy_text(
        title,
        renderable,
        width=dashboard.PANEL_COPY_WIDTHS["profiling"],
    )
    assert title == "Profiling · top-N"
    assert "provider.generate" in copied
    assert "component" in copied
    assert "share" in copied
    assert "calls" in copied
    assert "calls=1" not in copied
    assert "calls=—" not in copied
    assert re.search(r"provider\.generate\s+100\.0%\s+1", copied)
    sink.close()


def test_python_status_separates_baseline_and_candidate_progress_and_errors() -> None:
    status = {
        "state": "running",
        "generation_index": 0,
        "counts": {"planned": 8, "terminal": 0},
        "provider": {
            "program_turns_reserved": 1,
            "active": 1,
            "configured_concurrency": 2,
        },
        "evaluators": {"configured": 12, "active": 12},
        "evaluation_cases": {
            "active_completed": 587,
            "active_total": 640,
            "completed": 587,
            "total": 640,
            "baseline": {
                "active_completed": 587,
                "active_total": 640,
                "completed": 587,
                "total": 640,
            },
            "candidate": {
                "active_completed": 0,
                "active_total": 0,
                "completed": 0,
                "total": 0,
            },
        },
        "evaluation_progress": {
            "baseline:random": {
                "completed": 320,
                "total": 320,
                "queued": 0,
                "running": 0,
                "state": "terminal",
            },
            "baseline:structural": {
                "completed": 267,
                "total": 320,
                "queued": 41,
                "running": 12,
                "state": "running",
            },
        },
        "slots": [
            {
                "candidate_id": "g0000-slot-00",
                "generation": 0,
                "slot": "slot-00",
                "phase": "response",
                "state": "persisting",
            }
        ],
        "last_error": "M5InfrastructureError: provider turn exceeded 300s",
    }
    state = dashboard.dashboard_state_from_python_status(
        status,
        run_id="status-fixture",
        model="fixture",
        effort="high",
        generation_limit=1,
        wall_seconds=3600.0,
    )

    assert state.baseline_cases_completed == 587
    assert state.baseline_cases_total == 640
    assert state.candidate_cases_completed == 0
    assert state.candidate_cases_total == 0
    assert state.generations[0].slots[0].phase == "response"
    assert state.generations[0].slots[0].state == "persisting"
    assert state.activity == ()

    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        start_live=False,
    )
    sink.state = state
    output = io.StringIO()
    Console(file=output, width=150, force_terminal=False).print(
        sink._progress(150, horizontal=False)
    )
    rendered = output.getvalue()
    assert "Baselines" in rendered
    assert "587/640" in rendered
    assert "Candidates" in rendered
    assert "0/0" in rendered
    sink.close()


def test_python_progress_uses_matching_cumulative_scopes_across_generations() -> None:
    snapshots = (
        (960, 1280, 2560, 2560),
        (1280, 1280, 3840, 3840),
    )
    states = []
    for baseline_completed, baseline_total, candidate_completed, candidate_total in snapshots:
        status = {
            "state": "running",
            "generation_index": 1,
            "counts": {"planned": 16, "terminal": 11},
            "provider": {},
            "evaluators": {},
            "evaluation_cases": {
                "active_completed": 960,
                "active_total": 1280,
                "completed": baseline_completed + candidate_completed,
                "total": baseline_total + candidate_total,
                "baseline": {
                    "active_completed": 320,
                    "active_total": 640,
                    "completed": baseline_completed,
                    "total": baseline_total,
                },
                "candidate": {
                    "active_completed": 640,
                    "active_total": 640,
                    "completed": candidate_completed,
                    "total": candidate_total,
                },
            },
            "throughput": {},
            "phase_timings": {},
            "best": {},
            "exact_verification": {},
            "slots": [],
        }
        states.append(
            dashboard.dashboard_state_from_python_status(
                status,
                run_id="two-generation",
                model="fixture",
                effort="high",
                generation_limit=2,
                wall_seconds=3600.0,
            )
        )

    for state in states:
        assert state.baseline_cases_completed <= (state.baseline_cases_total or 0)
        assert state.candidate_cases_completed <= (state.candidate_cases_total or 0)
    assert (
        states[-1].baseline_cases_completed,
        states[-1].baseline_cases_total,
    ) == (
        1280,
        1280,
    )
    assert (
        states[-1].candidate_cases_completed,
        states[-1].candidate_cases_total,
    ) == (
        3840,
        3840,
    )


def test_python_dashboard_auto_follows_current_generation_but_preserves_history() -> None:
    generation_zero = GenerationSlots(0, (DashboardSlot("g0000-slot-00", 0),))
    generation_one = GenerationSlots(1, (DashboardSlot("g0001-slot-00", 1),))
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        initial_state=DashboardState(
            generation=0,
            displayed_generation=0,
            generations=(generation_zero,),
        ),
        start_live=False,
    )

    current = DashboardState(
        generation=1,
        displayed_generation=1,
        generations=(generation_zero, generation_one),
        elapsed_seconds=358.0,
    )
    sink.update_canonical_state(current)
    assert sink.state.displayed_generation == 1

    sink.state = replace(sink.state, displayed_generation=0)
    sink.update_canonical_state(replace(current, elapsed_seconds=359.0))
    assert sink.state.displayed_generation == 0
    sink.close()


def test_python_recent_activity_contains_transitions_not_live_rows() -> None:
    model_slot = DashboardSlot(
        "g0000-slot-00",
        0,
        phase="provider",
        state="model",
    )
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        initial_state=DashboardState(
            generation=0,
            displayed_generation=0,
            generations=(GenerationSlots(0, (model_slot,)),),
        ),
        start_live=False,
    )
    persisting = replace(model_slot, phase="response", state="persisting")
    sink.update_canonical_state(
        DashboardState(
            generation=0,
            displayed_generation=0,
            generations=(GenerationSlots(0, (persisting,)),),
            elapsed_seconds=61.0,
        )
    )
    assert sink.state.activity[0].timestamp == "01:01"
    assert sink.state.activity[0].message == "model completed · persisting"

    accepted = replace(
        persisting,
        phase="archived",
        state="accepted",
        objective=0.379,
    )
    sink.update_canonical_state(
        replace(
            sink.state,
            generations=(GenerationSlots(0, (accepted,)),),
            elapsed_seconds=3672.0,
            generation_objectives=(
                dashboard.GenerationObjectiveSummary(
                    generation=0,
                    candidate_id="g0000-slot-00",
                    objective=0.379,
                    random_objective=0.35,
                    structural_objective=0.36,
                ),
            ),
        )
    )
    messages = [item.message for item in sink.state.activity]
    assert sink.state.activity[0].timestamp == "1:01:12"
    assert any("accepted" in message for message in messages)
    assert any("random completed" in message for message in messages)
    assert any("structural completed" in message for message in messages)
    assert not any("evaluating ·" in message for message in messages)
    assert not any("status program" in message for message in messages)
    sink.close()


def test_objective_panel_uses_generation_local_values_and_full_copy() -> None:
    candidate = "g0001-slot-04-full-candidate"
    program_hash = "58e31933" + "a" * 56
    exact = {
        "lower": {"numerator": 1234567890123456789, "denominator": 10**19},
        "upper": {"numerator": 1234567890123456789, "denominator": 10**19},
    }
    state = DashboardState(
        generation=1,
        displayed_generation=1,
        best_candidate="g0000-slot-01",
        best_objective=0.379,
        best_fitness="379/1000",
        generation_objectives=(
            dashboard.GenerationObjectiveSummary(
                generation=1,
                candidate_id=candidate,
                program_hash=program_hash,
                fitness_interval=exact,
                objective=0.1234,
                structural_interval=exact,
                structural_objective=0.12,
                structural_delta_interval={
                    "lower": {"numerator": 1, "denominator": 100},
                    "upper": {"numerator": 1, "denominator": 100},
                },
                random_interval=exact,
                random_objective=0.11,
                random_delta_interval={
                    "lower": {"numerator": 2, "denominator": 100},
                    "upper": {"numerator": 2, "denominator": 100},
                },
                archive_valid=5,
                archive_failed=3,
            ),
        ),
    )
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        initial_state=state,
        start_live=False,
    )
    live_output = io.StringIO()
    Console(file=live_output, width=80, force_terminal=False).print(
        sink._objective_panel()
    )
    live = live_output.getvalue()
    assert "Objective · G2" in live
    assert "generation best" in live
    assert "candidate-ref" in live
    assert "lower bound" in live
    assert "58e31933" in live
    assert program_hash not in live
    assert "current" not in live

    title, renderable = sink._panel_copy_source("objective")
    copied = dashboard.render_panel_copy_text(
        title,
        renderable,
        width=dashboard.PANEL_COPY_WIDTHS["objective"],
    )
    assert candidate in copied
    assert program_hash in copied
    assert "1234567890123456789/10000000000000000000" in copied
    sink.close()


def test_completed_python_projection_retains_wall_activity_and_objective_history() -> None:
    interval = {
        "lower": {"numerator": 3633, "denominator": 10000},
        "upper": {"numerator": 2009, "denominator": 5000},
    }
    status = {
        "state": "blocked",
        "terminal_reason": "generation_budget",
        "generation_index": 0,
        "counts": {"planned": 8, "terminal": 8, "valid": 8, "evaluated": 8},
        "provider": {
            "turns": 9,
            "program_turns_reserved": 8,
            "completed_turns": 8,
        },
        "evaluators": {"configured": 12, "active": 0, "completed": 3200},
        "evaluation_cases": {
            "completed": 3200,
            "total": 3200,
            "baseline": {"completed": 640, "total": 640},
            "candidate": {"completed": 2560, "total": 2560},
        },
        "throughput": {
            "elapsed_seconds": 392.0,
            "current_run_elapsed_seconds": 392.0,
        },
        "best": {
            "candidate_id": "g0000-slot-05",
            "fitness_interval": interval,
            "program": {"program_hash": "a" * 64},
        },
        "generation_objectives": [
            {
                "generation": 0,
                "best": {
                    "candidate_id": "g0000-slot-05",
                    "program_hash": "a" * 64,
                    "fitness_interval": interval,
                },
                "baselines": {},
                "archive": {"valid": 8},
            }
        ],
        "slots": [
            {
                "candidate_id": "g0000-slot-05",
                "generation": 0,
                "slot": "slot-05",
                "phase": "archived",
                "state": "evaluated",
                "elapsed_seconds": 390.0,
            }
        ],
        "exact_verification": {"verified": False},
    }
    projected = dashboard.dashboard_state_from_python_status(
        status,
        run_id="completed-fixture",
        model="fixture",
        effort="high",
        generation_limit=1,
        wall_seconds=3600.0,
    )

    assert projected.elapsed_seconds == pytest.approx(392.0)
    assert projected.generations[0].slots[0].elapsed_seconds == pytest.approx(390.0)
    assert projected.current_objective == pytest.approx(0.3633)
    assert projected.objective_history == pytest.approx((0.3633,))

    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        initial_state=DashboardState(run_id="completed-fixture"),
        start_live=False,
    )
    sink.update_canonical_state(projected)
    assert sink.state.activity[0].timestamp == "06:32"
    assert sink.state.activity[0].message == "generation_budget"

    performance_output = io.StringIO()
    Console(file=performance_output, width=80, force_terminal=False).print(
        sink._performance_panel()  # noqa: SLF001
    )
    assert "392/" in performance_output.getvalue()

    quick_output = io.StringIO()
    Console(file=quick_output, width=100, force_terminal=False).print(
        sink._quick_view_panel("full", content_width=96)  # noqa: SLF001
    )
    assert "No evaluated objective history yet" not in quick_output.getvalue()
    assert "0.3633" in quick_output.getvalue()
    sink.close()


def test_profiling_panel_hides_zero_value_rows_until_data_arrives() -> None:
    state = replace(
        _running_state(),
        timing_profile={
            "phase_seconds": {
                "provider": 0.0,
                "evaluator/scorer": 0.0,
                "sandbox": 0.0,
            },
            "phase_calls": {"provider": 2},
            "unattributed_fraction": 0.0,
        },
    )
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        start_live=False,
    )
    sink.state = state
    output = io.StringIO()
    Console(file=output, width=60, force_terminal=False).print(sink._profiling_panel())
    rendered = output.getvalue()
    assert "Waiting for profile data" in rendered
    assert "0.0%" not in rendered
    sink.close()


def test_numbered_panel_keeps_centered_title_and_number_in_top_right_corner() -> None:
    output = io.StringIO()
    Console(
        file=output,
        width=40,
        force_terminal=False,
        color_system=None,
    ).print(dashboard._numbered_panel(Panel("body", title="Tokens"), "5"))
    top = output.getvalue().splitlines()[0]
    assert len(top) == 40
    assert "Tokens" in top
    assert top.endswith(" 5 ─╮")
    assert top.startswith("╭──")


@pytest.mark.parametrize("width", (110, 120, 140, 150))
def test_progress_panel_uses_historical_layout_with_current_metrics(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 392.0)
    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=width,
            force_terminal=False,
            color_system=None,
        ),
        locked_config={"model": {"name": "gpt-test"}},
        start_live=False,
    )
    sink.state = replace(
        _running_state(),
        completed_slots=8,
        provider_turns_attempted=49,
        hourly_token_limit=1_000_000,
        hourly_tokens_used=84_200,
    )
    output = io.StringIO()
    Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
    ).print(sink._progress(width, horizontal=True))
    rendered = output.getvalue()
    for label in (
        "Generation",
        "Slots Complete",
        "Model Turn Budget",
        "Token Budget",
        "Wall-time Budget",
    ):
        assert label in rendered
    assert "Evaluation Progress" not in rendered
    for ratio in ("2/4", "8/8", "49/64", "84.2k/1.0M", "292/7.2k"):
        assert ratio in rendered
    assert sum(line.count("%") for line in rendered.splitlines()) == 5
    assert all(len(line) <= width for line in rendered.splitlines())
    sink.close()


def test_progress_panel_omits_metrics_without_real_progress_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 392.0)
    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=150,
            force_terminal=False,
            color_system=None,
        ),
        locked_config=_adaptive_evaluation_config(),
        start_live=False,
    )
    sink.state = replace(
        _running_state(),
        generation=6,
        generation_limit=None,
        provider_turns_attempted=126,
        max_model_turns=None,
        hourly_token_limit=1_000_000,
        hourly_tokens_used=27_501,
    )
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._progress(150, horizontal=True))
    rendered = output.getvalue()
    assert "Generation" not in rendered
    assert "Model Turn Budget" not in rendered
    assert "Slots Complete" in rendered
    assert "Token Budget" in rendered
    assert "Evaluation Progress" not in rendered
    assert "Wall-time Budget" in rendered
    assert "27.5k/1.0M" in rendered
    assert "━" in rendered
    assert "—/—" not in rendered
    content_lines = [
        line for line in rendered.splitlines() if "Slots Complete" in line or "27.5k/1.0M" in line
    ]
    assert content_lines
    assert all(line[1:-1].count("│") == 2 for line in content_lines)
    panel_rows = rendered.splitlines()[1:-1]
    assert len(panel_rows) == 3
    assert all(line[1:-1].count("│") == 2 for line in panel_rows)
    sink.close()


def test_performance_panel_shows_episode_throughput_and_improvement_rate() -> None:
    state = _running_state()
    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_started",
            generation=1,
            slot="slot-02",
            evaluation_total=320,
        ),
        monotonic=120.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_progress",
            generation=1,
            slot="slot-02",
            evaluations_per_second=2.5,
            completed=10,
            total=320,
        ),
        monotonic=121.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_completed",
            generation=1,
            slot="slot-03",
            ir=0.375,
        ),
        monotonic=122.0,
    )
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        initial_state=state,
        start_live=False,
    )
    output = io.StringIO()
    Console(file=output, width=80, force_terminal=False, color_system=None).print(
        sink._performance_panel()  # noqa: SLF001
    )
    rendered = output.getvalue()
    assert "episodes/s" in rendered
    assert "2.50" in rendered
    assert "eval/s" not in rendered
    assert "IR" in rendered
    assert "0.3750" in rendered
    sink.close()


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    (
        (150, 55, ("SLOT MATRIX", "Performance & IR", "Quick View")),
        (120, 35, ("SLOT MATRIX", "Token Accounting", "Recent Activity")),
        (90, 24, ("SELECTED SLOT", "Slots", "Recent Activity")),
    ),
)
def test_dashboard_render_fits_viewport_and_exposes_mode_sections(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    expected: tuple[str, ...],
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 130.0)
    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=width,
            height=height,
            force_terminal=False,
            color_system=None,
        ),
        locked_config={"model": {"name": "gpt-test"}},
        start_live=False,
    )
    sink.state = _running_state()
    output = io.StringIO()
    Console(
        file=output,
        width=width,
        height=height,
        force_terminal=False,
        color_system=None,
    ).print(sink.render())
    rendered = output.getvalue()
    assert len(rendered.splitlines()) <= height
    for value in expected:
        assert value in rendered
    assert "No evaluated objective history yet" in rendered or width < 110
    if width >= 110:
        assert "forbidden_input_f…" not in rendered
        assert "validati…" not in rendered
    if width >= 140:
        matrix_output = io.StringIO()
        Console(
            file=matrix_output,
            width=width,
            force_terminal=False,
            color_system=None,
        ).print(sink._slot_matrix(width, "full"))
        assert "provider turn" not in matrix_output.getvalue().lower()
        assert "/home/user/" not in rendered
        assert "unrestricted_min_degree_3" in rendered
        assert "native-generation-checkpoint.json.gz" not in rendered
        assert "experiment" in rendered
        assert "session" in rendered
        assert "usage" in rendered
        assert "[1–9] copy" in rendered
    elif width >= 110:
        assert "session" in rendered
        assert "[1–9] copy" in rendered
    sink.close()


def test_detail_view_and_profiling_disabled_are_truthful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 130.0)
    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=150,
            height=55,
            force_terminal=False,
            color_system=None,
        ),
        start_live=False,
    )
    state = _running_state()
    state = replace(state, selected_index=1, view="details", detail_tab=2)
    sink.state = state
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        height=55,
        force_terminal=False,
        color_system=None,
    ).print(sink.render())
    rendered = output.getvalue()
    assert "SLOT DETAILS · s01" in rendered
    assert "forbidden_input_field" in rendered
    assert "policy reads a forbidden input field" in rendered

    sink.state = replace(state, view="matrix", profiling_enabled=False)
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        height=55,
        force_terminal=False,
        color_system=None,
    ).print(sink.render())
    assert "Profiling" not in output.getvalue()
    sink.close()


def test_evaluation_overview_uses_per_slot_fields_and_hides_provider_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 130.0)
    state = _running_state()
    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_started",
            generation=1,
            slot="slot-02",
            phase="development",
            evaluation_id="g0001-slot-02:development",
            evaluation_total=128,
        ),
        monotonic=120.0,
    )
    state = reduce_dashboard_event(
        state,
        _event(
            "evaluation_progress",
            generation=1,
            slot="slot-02",
            phase="replay",
            evaluation_id="g0001-slot-02:development",
            completed=29,
            total=128,
            order=12,
            graph_seed=403,
            policy_seed=4009,
            evaluations_per_second=2.5,
            **{"pass": "replay"},
        ),
        monotonic=125.0,
    )
    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=150,
            height=55,
            force_terminal=False,
        ),
        start_live=False,
    )
    sink.state = replace(state, selected_index=2, view="details")
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        height=55,
        force_terminal=False,
        color_system=None,
    ).print(sink.render())
    rendered = output.getvalue()
    assert "evaluation ID" in rendered
    assert "29 / 128" in rendered
    assert "replay" in rendered
    assert "403" in rendered
    assert "4009" in rendered
    assert "2.500/s" in rendered
    assert "provider request" not in rendered
    assert "provider thread" not in rendered
    sink.close()


def test_human_generation_numbers_and_truthful_footer_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 130.0)
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        locked_config=_adaptive_evaluation_config(),
        start_live=False,
    )
    sink.state = replace(_running_state(), profiling_enabled=True, timing_profile=None)

    output = io.StringIO()
    console = Console(file=output, width=150, force_terminal=False, color_system=None)
    console.print(sink._header(150))
    console.print(sink._progress(150, horizontal=True))
    console.print(sink._slot_matrix(150, "full"))
    console.print(sink._quick_view_panel("full"))
    rendered = output.getvalue()
    assert "Gen 2/4" in rendered
    assert "generation 2" in rendered
    assert "2 / 14 / 8" in rendered
    assert "Orders" in rendered
    expected_orders = dashboard.orders_for_generation(
        _adaptive_evaluation_config()["evaluation"],
        1,
    )
    assert ", ".join(map(str, expected_orders)) in rendered
    assert "Model gpt-test:high" in rendered
    assert "gpt-test/high" not in rendered
    assert "Slots" in rendered

    footer = sink._footer(150)
    assert "[←/→] gen" in footer.plain
    assert "[n/N]" not in footer.plain
    assert "[q] stop" in footer.plain
    assert "[p] pause" in footer.plain
    assert "[p] resume" not in footer.plain
    sink.state = replace(sink.state, paused=True)
    assert "[p] resume" in sink._footer(150).plain
    assert "[p] pause" not in sink._footer(150).plain
    sink.state = replace(sink.state, experiment_state="stopping")
    assert "[q] interrupt" in sink._footer(150).plain
    top_span = next(
        span for span in footer.spans if footer.plain[span.start : span.end] == "[t] top"
    )
    assert "dim" in str(top_span.style)
    sink.close()


def test_slot_matrix_integrates_selection_marker_into_slot_column() -> None:
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        start_live=False,
    )
    sink.state = _running_state()

    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._slot_matrix(150, "full"))
    rendered = output.getvalue()

    assert "▶s00" in rendered
    assert "▶  │ s00" not in rendered
    sink.close()


def test_slot_matrix_uses_score_header_and_four_decimal_values() -> None:
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        start_live=False,
    )
    state = _running_state()
    slots = list(state.generations[0].slots)
    slots[0] = replace(slots[0], objective=0.076991)
    sink.state = replace(
        state,
        generations=(GenerationSlots(1, tuple(slots)),),
    )

    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._slot_matrix(150, "full"))
    rendered = output.getvalue()

    assert "score ↑" in rendered
    assert "goal ↑" not in rendered
    assert "fitness ↑" not in rendered
    assert "objective ↑" not in rendered
    assert "0.0769" in rendered
    assert "0.076991" not in rendered
    assert "green better" in rendered
    sink.close()


def test_score_color_is_conservative_and_zero_remains_an_absolute_score() -> None:
    root = DashboardSlot(
        slot="g0000-slot-00",
        generation=0,
        objective=0.0,
    )
    better = replace(root, parent="g0000-slot-01", score_effect="proven_better")
    worse = replace(root, parent="g0000-slot-01", score_effect="proven_worse")

    root_text = dashboard._score_text(root)
    better_text = dashboard._score_text(better)
    worse_text = dashboard._score_text(worse)

    assert root_text.plain == "0.0000"
    assert str(root_text.style) == ""
    assert better_text.plain == "0.0000"
    assert str(better_text.style) == "green"
    assert str(worse_text.style) == "red"


def test_provider_active_waiting_and_queued_are_distinct() -> None:
    state = dashboard.dashboard_state_from_python_status(
        {
            "state": "running",
            "provider": {
                "active": 1,
                "waiting": 2,
                "queued": 4,
                "configured_concurrency": 2,
            },
            "slots": [
                {
                    "candidate_id": "g0000-slot-00",
                    "generation": 0,
                    "state": "model",
                },
                {
                    "candidate_id": "g0000-slot-01",
                    "generation": 0,
                    "state": "waiting",
                },
                {
                    "candidate_id": "g0000-slot-02",
                    "generation": 0,
                    "state": "queued",
                },
            ],
        },
        run_id="provider-counters",
        model="fixture",
        effort="high",
        generation_limit=1,
        wall_seconds=60.0,
    )
    assert state.active_provider_turns == 1
    assert state.waiting_provider_turns == 2
    assert state.queued_provider_turns == 4
    assert [slot.state for slot in state.generations[0].slots] == [
        "model",
        "waiting",
        "queued",
    ]


def test_exact_verification_visual_states_are_distinct() -> None:
    neutral = cli._exact_verification_text({"verified": False})
    verified = cli._exact_verification_text({"verified": True})
    failed = cli._exact_verification_text({"error": "verifier unavailable"})

    assert neutral.plain == "no"
    assert neutral.spans == []
    assert verified.plain == "yes"
    assert verified.spans[0].style == "green"
    assert failed.plain == "error"
    assert failed.spans[0].style == "red"


def test_slot_matrix_uses_available_width_for_full_parent_id() -> None:
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), width=150, force_terminal=False),
        start_live=False,
    )
    state = _running_state()
    slots = list(state.generations[0].slots)
    slots[0] = replace(slots[0], parent="g0004-slot-07")
    sink.state = replace(
        state,
        slot_icon_mode=True,
        generations=(GenerationSlots(1, tuple(slots)),),
    )

    output = io.StringIO()
    Console(
        file=output,
        width=150,
        force_terminal=False,
        color_system=None,
    ).print(sink._slot_matrix(150, "full"))

    assert "g0004-s07" in output.getvalue()
    assert "g0004-slot-07" not in output.getvalue()
    sink.close()


def test_quick_view_explains_objective_sparkline_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 130.0)
    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=150,
            height=55,
            force_terminal=False,
            color_system=None,
        ),
        start_live=False,
    )
    sink.state = replace(_running_state(), objective_history=(0.25, 0.75))
    output = io.StringIO()
    Console(
        file=output,
        width=150,
        height=55,
        force_terminal=False,
        color_system=None,
    ).print(sink.render())
    rendered = output.getvalue()
    assert "Objective history" in rendered
    assert "oldest → latest" not in rendered
    assert "n=2" not in rendered
    objective_line = next(line for line in rendered.splitlines() if "min 0.2500" in line)
    assert "Objective history" in objective_line
    assert "max 0.7500" in objective_line
    sink.close()


@pytest.mark.parametrize("width", (110, 120, 140, 150))
def test_quick_view_sparkline_stays_on_one_adaptive_line(width: int) -> None:
    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=width,
            height=60,
            force_terminal=False,
            color_system=None,
        ),
        locked_config=_adaptive_evaluation_config(),
        start_live=False,
    )
    history = tuple(0.2222 + index * (0.5556 - 0.2222) / 58 for index in range(59))
    sink.state = replace(_running_state(), objective_history=history)

    output = io.StringIO()
    Console(
        file=output,
        width=width,
        height=60,
        force_terminal=False,
        color_system=None,
    ).print(sink.render())
    objective_lines = [line for line in output.getvalue().splitlines() if "min 0.2222" in line]

    assert len(objective_lines) == 1
    assert "max 0.5556" in objective_lines[0]
    rendered = output.getvalue()
    assert "Orders" in rendered
    expected_orders = dashboard.orders_for_generation(
        _adaptive_evaluation_config()["evaluation"],
        1,
    )
    assert ", ".join(map(str, expected_orders)) in rendered
    layout = sink._render_full_or_compact(  # noqa: SLF001
        width,
        60,
        dashboard._responsive_mode(width, 60),  # noqa: SLF001
    )
    assert [child.ratio for child in layout["bottom"].children] == [1, 1]
    sink.close()


def test_key_decoder_handles_navigation_and_detail_keys() -> None:
    keys, remaining = _decode_keys(b"\x1b[A\x1b[B\x1b[C\x1b[D\x1b[H\x1b[F\x1b[Z\r\t\x7f")
    assert remaining == b""
    assert keys == [
        "UP",
        "DOWN",
        "RIGHT",
        "LEFT",
        "HOME",
        "END",
        "SHIFT_TAB",
        "ENTER",
        "TAB",
        "BACKSPACE",
    ]
    assert _decode_keys(b"12345678") == (list("12345678"), b"")


def test_terminal_input_restores_terminal_mode() -> None:
    master_fd, slave_fd = pty.openpty()
    stream = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
    try:
        original = termios.tcgetattr(slave_fd)
        reader = _TerminalInput(stream, lambda _key: None)
        reader.start()
        assert termios.tcgetattr(slave_fd) != original
        reader.close()
        assert termios.tcgetattr(slave_fd) == original
    finally:
        stream.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_terminal_input_survives_callback_error() -> None:
    master_fd, slave_fd = pty.openpty()
    stream = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8", buffering=1)
    callback_seen = threading.Event()

    def callback(_key: str) -> None:
        callback_seen.set()
        raise RuntimeError("simulated startup race")

    reader = _TerminalInput(stream, callback)
    try:
        reader.start()
        os.write(master_fd, b"q")
        assert callback_seen.wait(timeout=1.0)
        assert reader._thread is not None  # noqa: SLF001
        assert reader._thread.is_alive()  # noqa: SLF001
    finally:
        reader.close()
        stream.close()
        os.close(master_fd)
        os.close(slave_fd)


def test_dashboard_switch_is_opt_in_and_old_rich_sink_stays_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "experiment_protocol", lambda _path: "mforge.experiment.v2")
    parsed_default = cli.build_parser().parse_args(["experiment", "run"])
    parsed_dashboard = cli.build_parser().parse_args(["experiment", "run", "--dashboard"])
    parsed_status = cli.build_parser().parse_args(
        [
            "experiment",
            "status",
            "--dashboard",
            "--pause-record",
            "paused-for-budget.json",
        ]
    )
    parsed_resume = cli.build_parser().parse_args(
        [
            "experiment",
            "run",
            "--resume-current-generation",
            "7",
            "--max-new-repair-turns",
            "2",
        ]
    )
    assert parsed_default.dashboard is False
    assert parsed_dashboard.dashboard is True
    assert parsed_status.dashboard is True
    assert parsed_status.pause_record == Path("paused-for-budget.json")
    assert parsed_resume.resume_current_generation == 7
    assert parsed_resume.max_new_repair_turns == 2

    class _TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    config = SimpleNamespace(
        run=SimpleNamespace(output="rich"),
        config_path=Path("experiment.toml"),
        resolved_dict=lambda: {"exp_id": "dashboard-run"},
        immutable_config_sha256=lambda: "abc123",
    )
    old_sink = Mock()
    new_sink = Mock()
    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(cli, "RichLiveSink", Mock(return_value=old_sink))
    monkeypatch.setattr(cli, "InteractiveDashboardSink", Mock(return_value=new_sink))
    monkeypatch.setattr(cli, "run_experiment", lambda *_args, **_kwargs: {"status": "completed"})
    monkeypatch.setattr(cli, "experiment_status", lambda _path: {"state": "completed"})
    monkeypatch.setattr(cli, "render_status", lambda _summary: "completed")
    monkeypatch.setattr(cli.sys, "stdout", _TTY())

    assert cli._experiment_run(Path("experiment.toml"), json_output=False) == 0
    cli.RichLiveSink.assert_called_once()
    cli.InteractiveDashboardSink.assert_not_called()
    old_sink.close.assert_called_once()

    cli.RichLiveSink.reset_mock()
    assert (
        cli._experiment_run(
            Path("experiment.toml"),
            json_output=False,
            dashboard=True,
        )
        == 0
    )
    cli.RichLiveSink.assert_not_called()
    cli.InteractiveDashboardSink.assert_called_once()
    new_sink.close.assert_called_once()


def test_current_generation_resume_flags_route_transient_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def fake_run(
        _path: Path,
        *,
        resume_budget: object,
    ) -> dict[str, object]:
        captured.append(resume_budget)
        return {
            "state": "blocked",
            "terminal_reason": "resume_generation_complete",
        }

    monkeypatch.setattr(
        cli,
        "experiment_protocol",
        lambda _path: "native-v3-python-v1",
    )
    monkeypatch.setattr(cli, "run_python_preview", fake_run)

    assert (
        cli._experiment_run(
            Path("m10.toml"),
            json_output=True,
            resume_current_generation=7,
            max_new_repair_turns=2,
        )
        == 0
    )
    budget = captured[0]
    assert isinstance(budget, cli.ScientificResumeBudgetV1)
    assert budget.expected_pending_primary_slots == 7
    assert budget.max_new_repair_turns == 2

    with pytest.raises(ValueError, match="must be supplied together"):
        cli._experiment_run(
            Path("m10.toml"),
            json_output=True,
            resume_current_generation=7,
        )


def test_standard_dashboard_routes_explicit_python_through_existing_sink(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scientific = SimpleNamespace(
        generation_limit=12,
        current_primary_program_slots=96,
        wall_seconds=36000.0,
        as_dict=lambda: {"provider_concurrency": 4},
    )
    config = SimpleNamespace(
        protocol="native-v3-python-v1",
        source_path=Path("m10.toml"),
        experiment_root=Path("/durable/m10"),
        exp_id="m10-dashboard",
        model="gpt-fixture",
        effort="medium",
        scientific_search=scientific,
    )
    status = {
        "state": "blocked",
        "terminal_reason": "generation_budget",
        "generation_index": 0,
        "counts": {
            "planned": 8,
            "terminal": 8,
            "valid": 1,
            "evaluated": 1,
            "contract_invalid": 7,
            "provider_failed": 0,
            "evaluation_infrastructure_failure": 0,
            "duplicate": 0,
        },
        "provider": {
            "turns": 9,
            "program_turns_reserved": 8,
            "active": 0,
            "configured_concurrency": 4,
            "usage": {"totalTokens": 10},
        },
        "evaluators": {
            "completed": 1,
            "active": 0,
            "configured": 12,
        },
        "throughput": {"elapsed_seconds": 1.0},
        "scientific_activity": {},
        "phase_timings": {},
        "best": {},
        "exact_verification": {},
        "slots": [],
        "recovery": {},
    }
    sink = Mock()
    dashboard_factory = Mock(return_value=sink)
    monkeypatch.setattr(
        cli,
        "experiment_protocol",
        lambda _path: "native-v3-python-v1",
    )
    monkeypatch.setattr(
        cli,
        "load_python_preview_config",
        lambda _path: config,
    )
    worker_started = False

    def bootstrap(_path: Path) -> dict[str, str]:
        assert worker_started is False
        return {"state": "ready"}

    def poll(_path: Path) -> dict[str, object]:
        assert worker_started is True
        return status

    def run_preview(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal worker_started
        worker_started = True
        return status

    monkeypatch.setattr(cli, "python_preview_bootstrap_status", bootstrap)
    monkeypatch.setattr(cli, "python_preview_status", poll)
    run = Mock(side_effect=run_preview)
    monkeypatch.setattr(cli, "run_python_preview", run)
    monkeypatch.setattr(
        cli,
        "InteractiveDashboardSink",
        dashboard_factory,
    )

    assert (
        cli._experiment_run(
            Path("m10.toml"),
            json_output=False,
            dashboard=True,
        )
        == 0
    )
    run.assert_called_once()
    assert run.call_args.args == (Path("m10.toml"),)
    assert callable(run.call_args.kwargs["force_stop"])
    dashboard_factory.assert_called_once()
    initial = dashboard_factory.call_args.kwargs["initial_state"]
    assert initial.completed_slots == 0
    assert initial.provider_turns_attempted == 0
    assert initial.evaluations_completed == 0
    assert sink.update_canonical_state.call_count == 2
    sink.close.assert_called_once()
    final_line = capsys.readouterr().out
    assert "provider turns" in final_line
    assert "total 9" in final_line
    assert "generation_budget (12" in final_line
    assert "generations / 96 primary slots)" in final_line


def test_generation_limit_dashboard_stays_read_only_until_q(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "retained.json").write_bytes(b'{"retained":true}\n')
    before = {
        path.relative_to(workspace): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    scientific = SimpleNamespace(
        generation_limit=2,
        current_primary_program_slots=16,
        wall_seconds=3600.0,
        as_dict=lambda: {"provider_concurrency": 2},
    )
    config = SimpleNamespace(
        protocol="native-v3-python-v1",
        source_path=tmp_path / "experiment.toml",
        experiment_root=workspace,
        exp_id="idle-dashboard",
        model="gpt-fixture",
        effort="medium",
        scientific_search=scientific,
    )
    bootstrap = {
        "state": "blocked",
        "terminal_reason": "generation_budget",
        "generation_index": 1,
    }
    status = {
        **bootstrap,
        "counts": {"planned": 16, "terminal": 16},
        "provider": {"active": 0, "configured_concurrency": 2},
        "evaluators": {"active": 0, "configured": 12},
        "throughput": {},
        "scientific_activity": {},
        "phase_timings": {},
        "best": {},
        "exact_verification": {},
        "slots": [
            {
                "candidate_id": f"g{generation:04d}-slot-00",
                "generation": generation,
                "slot": "slot-00",
                "state": "evaluated",
                "phase": "archived",
            }
            for generation in (0, 1)
        ],
        "recovery": {"last_boundary": "report_persisted"},
    }
    ready = threading.Event()
    captured_sink: list[InteractiveDashboardSink] = []

    class IdleSink(InteractiveDashboardSink):
        def update_canonical_state(self, state: DashboardState) -> None:
            super().update_canonical_state(state)
            ready.set()

    def dashboard_factory(**kwargs: Any) -> InteractiveDashboardSink:
        kwargs.pop("console", None)
        sink = IdleSink(
            **kwargs,
            console=Console(file=io.StringIO(), force_terminal=False),
            start_live=False,
        )
        captured_sink.append(sink)
        return sink

    run = Mock()
    request_stop = Mock()
    status_reader = Mock(return_value=status)
    monkeypatch.setattr(cli, "experiment_protocol", lambda _path: "native-v3-python-v1")
    monkeypatch.setattr(cli, "load_python_preview_config", lambda _path: config)
    monkeypatch.setattr(cli, "python_preview_bootstrap_status", lambda _path: bootstrap)
    monkeypatch.setattr(cli, "python_preview_status", status_reader)
    monkeypatch.setattr(cli, "run_python_preview", run)
    monkeypatch.setattr(cli, "request_python_preview_stop", request_stop)
    monkeypatch.setattr(cli, "InteractiveDashboardSink", dashboard_factory)

    outcome: list[int] = []
    command = threading.Thread(
        target=lambda: outcome.append(
            cli._experiment_run(
                config.source_path,
                json_output=False,
                dashboard=True,
            )
        )
    )
    command.start()
    assert ready.wait(timeout=1)
    assert command.is_alive()
    assert run.call_count == 0
    assert status_reader.call_count == 1
    assert request_stop.call_count == 0

    sink = captured_sink[0]
    assert sink.state.displayed_generation == 1
    sink.handle_key("LEFT")
    assert sink.state.displayed_generation == 0

    started = time.monotonic()
    sink.handle_key("q")
    command.join(timeout=1)
    assert not command.is_alive()
    assert time.monotonic() - started < 0.5
    assert outcome == [0]
    assert status_reader.call_count == 1
    assert request_stop.call_count == 0
    after = {
        path.relative_to(workspace): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("result", "expected"),
    (
        (
            {
                "state": "blocked",
                "resumable": True,
                "terminal_reason": "generation_budget",
            },
            0,
        ),
        (
            {
                "state": "blocked",
                "resumable": True,
                "terminal_reason": "generation_budget",
                "no_op": True,
            },
            0,
        ),
        (
            {
                "state": "blocked",
                "resumable": False,
                "terminal_reason": "infrastructure_failure",
            },
            1,
        ),
        (
            {
                "state": "blocked",
                "resumable": False,
                "terminal_reason": "provider_runtime_missing",
            },
            1,
        ),
    ),
)
def test_python_run_exit_codes_distinguish_requested_bound_from_failures(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, object],
    expected: int,
) -> None:
    monkeypatch.setattr(
        cli,
        "experiment_protocol",
        lambda _path: "native-v3-python-v1",
    )
    monkeypatch.setattr(cli, "run_python_preview", lambda _path: result)

    assert (
        cli._experiment_run(
            Path("m10.toml"),
            json_output=True,
        )
        == expected
    )


def test_python_dashboard_retries_first_q_after_workspace_startup_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scientific = SimpleNamespace(
        generation_limit=1,
        wall_seconds=60.0,
        as_dict=lambda: {"provider_concurrency": 2},
    )
    config = SimpleNamespace(
        protocol="native-v3-python-v1",
        source_path=Path("m10.toml"),
        experiment_root=Path("/durable/m10"),
        exp_id="m10-dashboard",
        model="gpt-fixture",
        effort="medium",
        scientific_search=scientific,
    )
    status = {
        "state": "completed",
        "counts": {},
        "provider": {},
        "evaluators": {},
        "throughput": {},
        "scientific_activity": {},
        "phase_timings": {},
        "best": {},
        "exact_verification": {},
        "slots": [],
        "recovery": {},
    }
    sink = Mock()
    request_stop = Mock(side_effect=[RuntimeError("workspace not ready"), status])
    release = threading.Event()

    def dashboard_factory(**kwargs: Any) -> Mock:
        kwargs["capabilities"].quit()
        return sink

    def fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        release.wait(timeout=0.6)
        return status

    monkeypatch.setattr(cli, "experiment_protocol", lambda _path: "native-v3-python-v1")
    monkeypatch.setattr(cli, "load_python_preview_config", lambda _path: config)
    monkeypatch.setattr(cli, "python_preview_bootstrap_status", lambda _path: status)
    monkeypatch.setattr(cli, "python_preview_status", lambda _path: status)
    monkeypatch.setattr(cli, "request_python_preview_stop", request_stop)
    monkeypatch.setattr(cli, "run_python_preview", fake_run)
    monkeypatch.setattr(cli, "InteractiveDashboardSink", dashboard_factory)

    assert (
        cli._experiment_run(
            Path("m10.toml"),
            json_output=False,
            dashboard=True,
        )
        == 0
    )
    assert request_stop.call_count == 2
    sink.close.assert_called_once()


def test_python_dashboard_second_q_does_not_wait_for_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scientific = SimpleNamespace(
        generation_limit=1,
        wall_seconds=60.0,
        as_dict=lambda: {"provider_concurrency": 2},
    )
    config = SimpleNamespace(
        protocol="native-v3-python-v1",
        source_path=Path("m10.toml"),
        experiment_root=Path("/durable/m10"),
        exp_id="m10-dashboard",
        model="gpt-fixture",
        effort="medium",
        scientific_search=scientific,
    )
    status = {
        "state": "running",
        "counts": {},
        "provider": {},
        "evaluators": {},
        "throughput": {},
        "scientific_activity": {},
        "phase_timings": {},
        "best": {},
        "exact_verification": {},
        "slots": [],
        "recovery": {},
    }
    sink = Mock()
    release = threading.Event()

    def dashboard_factory(**kwargs: Any) -> Mock:
        kwargs["capabilities"].quit()
        kwargs["capabilities"].interrupt()
        return sink

    def fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        release.wait(timeout=5.0)
        return status

    monkeypatch.setattr(cli, "experiment_protocol", lambda _path: "native-v3-python-v1")
    monkeypatch.setattr(cli, "load_python_preview_config", lambda _path: config)
    monkeypatch.setattr(cli, "python_preview_bootstrap_status", lambda _path: status)
    monkeypatch.setattr(cli, "python_preview_status", lambda _path: status)
    monkeypatch.setattr(cli, "request_python_preview_stop", Mock())
    monkeypatch.setattr(cli, "run_python_preview", fake_run)
    monkeypatch.setattr(cli, "InteractiveDashboardSink", dashboard_factory)

    started = time.monotonic()
    try:
        assert (
            cli._experiment_run(
                Path("m10.toml"),
                json_output=False,
                dashboard=True,
            )
            == 1
        )
        assert time.monotonic() - started < 2.0
    finally:
        release.set()
    sink.close.assert_called_once()


def test_read_only_python_status_dashboard_does_not_run_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        protocol="native-v3-python-v1",
        source_path=Path("m10.toml"),
        experiment_root=Path("/durable/m10"),
        exp_id="m10-dashboard",
        model="gpt-fixture",
        effort="medium",
        scientific_search=SimpleNamespace(
            generation_limit=12,
            wall_seconds=36000.0,
        ),
    )
    status = {"state": "PAUSED_FOR_BUDGET"}
    projected = DashboardState(
        run_id=config.exp_id,
        experiment_state="PAUSED_FOR_BUDGET",
        paused=True,
    )
    sink = Mock()
    sink.render.return_value = "read-only dashboard"
    console = Mock()
    pause_record = Path("paused-for-budget.json")
    run = Mock()
    status_loader = Mock(return_value=status)
    monkeypatch.setattr(
        cli,
        "experiment_protocol",
        lambda _path: "native-v3-python-v1",
    )
    monkeypatch.setattr(cli, "load_python_preview_config", lambda _path: config)
    monkeypatch.setattr(cli, "python_preview_status", status_loader)
    monkeypatch.setattr(cli, "dashboard_state_from_python_status", Mock(return_value=projected))
    monkeypatch.setattr(cli, "InteractiveDashboardSink", Mock(return_value=sink))
    monkeypatch.setattr(cli, "Console", Mock(return_value=console))
    monkeypatch.setattr(cli, "run_python_preview", run)

    assert (
        cli._experiment_status(
            Path("m10.toml"),
            json_output=False,
            dashboard=True,
            pause_record_path=pause_record,
        )
        == 0
    )
    status_loader.assert_called_once_with(
        Path("m10.toml"),
        pause_record_path=pause_record,
    )
    run.assert_not_called()
    cli.InteractiveDashboardSink.assert_called_once_with(
        console=console,
        locked_config={
            "protocol": config.protocol,
            "config_path": str(config.source_path),
            "workspace": str(config.experiment_root),
            "read_only": True,
        },
        initial_state=projected,
        start_live=False,
    )
    console.print.assert_called_once_with(sink.render.return_value)
    sink.close.assert_called_once()


def test_dashboard_q_stops_gracefully_then_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "experiment_protocol", lambda _path: "mforge.experiment.v2")

    class _TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    config = SimpleNamespace(
        run=SimpleNamespace(output="rich"),
        config_path=Path("experiment.toml"),
        resolved_dict=lambda: {"exp_id": "dashboard-run"},
        immutable_config_sha256=lambda: "abc123",
    )
    sink = Mock()
    interrupt = Mock()

    def fake_run(*_args: object, control: object, **_kwargs: object) -> dict[str, object]:
        capabilities = cli.InteractiveDashboardSink.call_args.kwargs["capabilities"]
        capabilities.quit()
        assert control.graceful_stop_requested
        interrupt.assert_not_called()
        capabilities.quit()
        return {"status": "completed"}

    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(cli, "InteractiveDashboardSink", Mock(return_value=sink))
    monkeypatch.setattr(cli, "run_experiment", fake_run)
    monkeypatch.setattr(cli, "experiment_status", lambda _path: {"state": "completed"})
    monkeypatch.setattr(cli, "render_status", lambda _summary: "completed")
    monkeypatch.setattr(cli._thread, "interrupt_main", interrupt)
    monkeypatch.setattr(cli.sys, "stdout", _TTY())

    assert (
        cli._experiment_run(
            Path("experiment.toml"),
            json_output=False,
            dashboard=True,
        )
        == 0
    )
    interrupt.assert_called_once_with()
    sink.close.assert_called_once()


def test_until_complete_continues_after_wall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "experiment_protocol", lambda _path: "mforge.experiment.v2")
    config = SimpleNamespace(
        run=SimpleNamespace(output="json"),
        config_path=Path("experiment.toml"),
        resolved_dict=lambda: {"exp_id": "overnight-run"},
        immutable_config_sha256=lambda: "abc123",
    )
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return (
            {"status": "completed", "state": "completed"}
            if calls == 2
            else {
                "status": "idle",
                "state": "idle",
                "stop_reason": "session_wall_seconds",
            }
        )

    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(cli, "run_experiment", fake_run)
    monkeypatch.setattr(cli, "experiment_status", lambda _path: {"state": "completed"})
    monkeypatch.setattr(cli, "render_status", lambda _summary: "completed")
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    assert (
        cli._experiment_run(
            Path("experiment.toml"),
            json_output=True,
            until_complete=True,
        )
        == 0
    )
    assert calls == 2


def test_until_complete_stops_at_hourly_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "experiment_protocol", lambda _path: "mforge.experiment.v2")
    config = SimpleNamespace(
        run=SimpleNamespace(output="json"),
        config_path=Path("experiment.toml"),
        resolved_dict=lambda: {"exp_id": "overnight-run"},
        immutable_config_sha256=lambda: "abc123",
    )
    calls = 0

    def fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "status": "idle",
            "state": "idle",
            "stop_reason": "hourly_token_limit",
        }

    monkeypatch.setattr(cli, "load_experiment_config", lambda _path: config)
    monkeypatch.setattr(cli, "run_experiment", fake_run)
    monkeypatch.setattr(
        cli,
        "experiment_status",
        lambda _path: {
            "state": "idle",
            "hourly_token_limit": 1_000_000,
            "hourly_tokens_used": 1_000_000,
        },
    )
    monkeypatch.setattr(cli, "render_status", lambda _summary: "idle")

    assert (
        cli._experiment_run(
            Path("experiment.toml"),
            json_output=True,
            until_complete=True,
        )
        == 0
    )
    assert calls == 1


def test_sink_dispatches_supported_pause_and_retry_without_execution_side_effects() -> None:
    pause = Mock()
    retry = Mock()
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        capabilities=DashboardCapabilities(pause=pause, retry=retry, quit=Mock()),
        start_live=False,
    )
    sink.state = DashboardState(
        generations=(
            GenerationSlots(
                0,
                (DashboardSlot("slot-00", 0, state="failed", retryable=True),),
            ),
        )
    )
    sink.handle_key("p")
    pause.assert_called_once_with(True)
    sink.handle_key("r")
    sink.handle_key("y")
    retry.assert_called_once_with("slot-00")
    sink.close()


def test_pending_panel_copy_writes_fallback_and_expires_notice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = [130.0]
    clipboard = Mock(return_value=False)
    monkeypatch.setattr(dashboard, "PANEL_COPY_TMP_DIR", tmp_path)
    monkeypatch.setattr(dashboard, "copy_text_to_clipboard_osc52", clipboard)
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: now[0])
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        start_live=False,
    )
    sink.state = _running_state()

    sink.handle_key("5")
    assert sink._pending_copy_action == "tokens"
    with sink._lock:
        sink._handle_pending_panel_copy_unlocked()

    path = tmp_path / "panel-tokens-dashboard-run.txt"
    copied = path.read_text(encoding="utf-8")
    assert copied.startswith("# Token Accounting\n\n")
    assert "experiment" in copied
    assert "session" in copied
    assert "usage" in copied
    assert "input" in copied
    assert "\x1b" not in copied
    assert not any(character in copied for character in "╭╮╰╯│")
    clipboard.assert_called_once_with(copied)
    assert sink.state.status_message == f"OSC 52 unavailable · saved {path}"

    now[0] += dashboard.COPY_NOTICE_SECONDS
    with sink._lock:
        assert sink._expire_copy_notice_unlocked()
    assert sink.state.status_message == ""
    sink.close()


def test_slot_detail_copy_includes_full_prompt_preview() -> None:
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
        start_live=False,
    )
    state = _running_state()
    matrix_slots = list(state.generations[0].slots)
    full_parent = "g0001-slot-00-candidate-full-identifier"
    matrix_slots[0] = replace(matrix_slots[0], parent=full_parent)
    sink.state = replace(
        state,
        generations=(GenerationSlots(1, tuple(matrix_slots)),),
    )
    matrix_title, matrix = sink._panel_copy_source("slots")
    matrix_copy = dashboard.render_panel_copy_text(
        matrix_title,
        matrix,
        width=dashboard.PANEL_COPY_WIDTHS["slots"],
    )
    assert "s00" in matrix_copy
    assert "g0001-s00-candidate-full-identifier" in matrix_copy
    assert full_parent not in matrix_copy
    assert "…" not in matrix_copy
    assert not any(character in matrix_copy for character in "╭╮╰╯│")

    slots = list(state.generations[0].slots)
    prompt = "\n".join(f"line {index}" for index in range(12))
    slots[0] = replace(slots[0], prompt_preview=prompt)
    sink.state = replace(
        state,
        generations=(GenerationSlots(1, tuple(slots)),),
        view="details",
        detail_tab=6,
    )
    title, renderable = sink._panel_copy_source("slots")
    copied = dashboard.render_panel_copy_text(title, renderable, width=150)
    assert "line 0" in copied
    assert "line 11" in copied
    assert "\n…\n" not in copied
    sink.close()


def test_live_updates_immediately_on_events_and_heartbeats_while_active() -> None:
    sink = InteractiveDashboardSink(
        console=Console(file=io.StringIO(), force_terminal=False),
    )
    updated = threading.Event()
    sink.live.update = Mock(side_effect=lambda *_args, **_kwargs: updated.set())
    sink.write(_event("session_started", experiment_id="dashboard-run"))
    assert updated.wait(timeout=0.5)
    updated.clear()
    assert updated.wait(timeout=1.5)
    sink.write(_event("experiment_completed", state="completed"))
    assert updated.wait(timeout=0.5)
    sink.close()


def test_dashboard_prints_final_render_after_live_shutdown() -> None:
    order: list[str] = []
    console = Console(file=io.StringIO(), force_terminal=False)
    sink = InteractiveDashboardSink(
        console=console,
        start_live=False,
    )
    sink._live_started = True
    sink.live.update = Mock(side_effect=lambda *_args, **_kwargs: order.append("update"))
    sink.live.stop = Mock(side_effect=lambda: order.append("stop"))
    sink.console.print = Mock(side_effect=lambda *_args, **_kwargs: order.append("print"))

    sink.close()

    assert order[-3:] == ["update", "stop", "print"]
    sink.console.print.assert_called_once()


@pytest.mark.parametrize(
    ("scenario", "width", "height"),
    (
        ("running_provider_profiled", 150, 55),
        ("evaluation_active", 150, 55),
        ("validation_details", 150, 55),
        ("completed", 150, 55),
        ("profiling_disabled", 150, 55),
        ("compact", 120, 35),
        ("minimal", 90, 24),
    ),
)
def test_dashboard_scenarios_fit_viewport(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    width: int,
    height: int,
) -> None:
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: 130.0)
    monkeypatch.setattr(
        dashboard.resource,
        "getrusage",
        lambda _scope: SimpleNamespace(ru_utime=12.5, ru_stime=1.25),
    )
    state = _running_state()
    if scenario == "running_provider_profiled":
        state = replace(
            state,
            profiling_enabled=True,
            timing_profile={
                "enabled": True,
                "phase_seconds": {"provider.generate": 2.0, "validator.schema": 0.5},
                "phase_calls": {"provider.generate": 1, "validator.schema": 2},
                "unattributed_fraction": 0.1,
            },
        )
    elif scenario == "evaluation_active":
        slots = list(state.generations[0].slots)
        slots[2] = replace(
            slots[2],
            state="evaluating",
            phase="evaluation",
            evaluation_completed=24,
            evaluation_total=64,
        )
        state = replace(
            state,
            generations=(GenerationSlots(1, tuple(slots)),),
            evaluation_workers_active=3,
            evaluation_workers_configured=4,
        )
    elif scenario == "validation_details":
        state = replace(state, selected_index=1, view="details", detail_tab=2)
    elif scenario == "completed":
        state = reduce_dashboard_event(
            state,
            _event(
                "experiment_completed",
                state="completed",
                elapsed_seconds=123.0,
                stop_reason="generation_limit",
            ),
            monotonic=130.0,
        )
    elif scenario == "profiling_disabled":
        state = replace(state, profiling_enabled=False, timing_profile=None)

    sink = InteractiveDashboardSink(
        console=Console(
            file=io.StringIO(),
            width=width,
            height=height,
            force_terminal=False,
            color_system=None,
        ),
        locked_config={"model": {"name": "gpt-test"}},
        start_live=False,
    )
    sink.state = state
    output = io.StringIO()
    Console(
        file=output,
        width=width,
        height=height,
        force_terminal=False,
        color_system=None,
    ).print(sink.render())
    rendered = output.getvalue()
    assert len(rendered.splitlines()) <= height
    assert all(len(line) <= width for line in rendered.splitlines())
    assert "Mutation Forge Lab" in rendered
    if scenario == "completed":
        assert "COMPLETED" in rendered
    elif scenario == "validation_details":
        assert "SLOT DETAILS" in rendered
        assert "outcome/code" in rendered
    elif scenario == "minimal":
        assert "SELECTED SLOT" in rendered
    else:
        assert "SLOT MATRIX" in rendered
    sink.close()
