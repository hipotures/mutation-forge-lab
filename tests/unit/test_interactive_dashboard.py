from __future__ import annotations

import hashlib
import io
import os
import pty
import termios
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.panel import Panel

from mutation_forge import cli
from mutation_forge.events import Event
from mutation_forge.output import interactive_dashboard as dashboard
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
            checkpoint=(
                "/home/user/DEV/mutation-forge-lab/workspace/dashboard-run/"
                "artifacts/native-generation-checkpoint.json"
            ),
            configured_wall_seconds=7200.0,
            model="gpt-test",
            effort="high",
            configured_concurrency=2,
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
    assert "25%" in lines[1] and "1/4" in lines[1]
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
    assert len(rendered.splitlines()) == 13
    sink.close()


def test_event_reducer_keeps_authoritative_counts_and_deduplicates_tokens() -> None:
    state = _running_state()
    assert state.active_provider_turns == 0
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
    )
    once = reduce_dashboard_event(state, completed, monotonic=120.0)
    twice = reduce_dashboard_event(once, completed, monotonic=121.0)
    assert once.cumulative_usage.total == 178
    assert twice.cumulative_usage.total == 178
    assert once.session_usage.total == 18
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
        ),
        monotonic=124.0,
    )

    slot = recovered.generations[0].slots[2]
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

    state, _ = reduce_dashboard_key(state, "N")
    assert state.displayed_generation == 0
    assert state.generation == 1
    state, _ = reduce_dashboard_key(state, "n")
    assert state.displayed_generation == 1
    assert state.generation == 1


def test_key_reducer_overlays_search_and_disabled_scheduler_actions() -> None:
    state = _running_state()
    state, _ = reduce_dashboard_key(state, "c")
    assert state.view == "config"
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

    state, action = reduce_dashboard_key(state, "p")
    assert action is None
    assert "unavailable" in state.status_message
    state, action = reduce_dashboard_key(state, "r")
    assert action is None
    assert "unavailable" in state.status_message


def test_number_keys_request_stable_panel_copies_but_remain_search_text() -> None:
    state = _running_state()
    for key, panel in dashboard.PANEL_COPY_KEYS.items():
        state, action = reduce_dashboard_key(state, key)
        assert action == dashboard.DashboardAction("copy", panel=panel)

    state = replace(state, search_editing=True, search_query="")
    state, action = reduce_dashboard_key(state, "5")
    assert action is None
    assert state.search_query == "5"


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


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    (
        (150, 55, ("SLOT MATRIX", "Performance & IR", "Quick View")),
        (120, 35, ("SLOT MATRIX", "Token Accounting", "Recent Activity")),
        (90, 24, ("SELECTED SLOT", "Slots Complete", "Recent Activity")),
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
        assert (
            "workspace/dashboard-run/artifacts/native-generation-checkpoint.json"
            in rendered
        )
        assert "experiment" in rendered
        assert "session" in rendered
        assert "usage" in rendered
        assert "[1–8] copy" in rendered
    elif width >= 110:
        assert "session" in rendered
        assert "[1–8] copy" in rendered
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
    assert "SLOT DETAILS · slot-01" in rendered
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
    assert "Objective history · oldest → latest · n=2" in rendered
    assert "min 0.250000" in rendered
    assert "max 0.750000" in rendered
    sink.close()


def test_key_decoder_handles_navigation_and_detail_keys() -> None:
    keys, remaining = _decode_keys(
        b"\x1b[A\x1b[B\x1b[C\x1b[D\x1b[H\x1b[F\x1b[Z\r\t\x7f"
    )
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


def test_dashboard_switch_is_opt_in_and_old_rich_sink_stays_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_default = cli.build_parser().parse_args(["experiment", "run"])
    parsed_dashboard = cli.build_parser().parse_args(
        ["experiment", "run", "--dashboard"]
    )
    assert parsed_default.dashboard is False
    assert parsed_dashboard.dashboard is True

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
    assert "slot-00" in matrix_copy
    assert full_parent in matrix_copy
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


GOLDEN_RENDER_HASHES = {
    "running_provider_profiled": (
        "62319fec6348979caff63898fb76bfb8b27273e148cd5cf7ed5a8fcf35de670d"
    ),
    "evaluation_active": (
        "0ae52c16719b68dd481605903c62139c2ccaeac7efe93519528c135ddb4c6bad"
    ),
    "validation_details": (
        "06c042e6db830da0254a4607bb49d3d9c945039b735dff25ecd117a9898c728c"
    ),
    "completed": (
        "b78dc065870fda4b6cb8e72107cbbb973eb4c26e2d7ed86fd2b4f9b2c40bf982"
    ),
    "profiling_disabled": (
        "374fd5d0bac1aa2aeed1a3577c96345fddb71bffb277c27422b4eb090ec0f117"
    ),
    "compact": (
        "0cf6c00f0d71a2cb148cecc0203fc12f1b77a8b402e1b7d184121f8517ca3a5d"
    ),
    "minimal": (
        "d2be7297ac8ee3a90122ab42cd3e2273b706655b228f64dd954f79eec4d051d6"
    ),
}


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
def test_dashboard_golden_render(
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
        slots[2] = replace(slots[2], state="evaluating", phase="evaluation")
        state = replace(
            state,
            generations=(GenerationSlots(1, tuple(slots)),),
            evaluation_episodes_completed=24,
            evaluation_episodes_total=64,
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
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    assert len(rendered.splitlines()) <= height
    assert digest == GOLDEN_RENDER_HASHES[scenario], f"{scenario}: {digest}"
    sink.close()
