from __future__ import annotations

import io
from unittest.mock import patch

from rich import box
from rich.console import Console

from mutation_forge.events import Event
from mutation_forge.output.rich_live import RichLiveSink


def test_rich_live_formats_evaluations_per_second() -> None:
    assert RichLiveSink._rate(1467.452267324598) == "1467.45"


def test_rich_live_places_timing_on_last_row() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        output = io.StringIO()
        Console(file=output, force_terminal=False, width=160).print(sink._render())
        rendered = output.getvalue()
        assert rendered.index("Profile top / unattributed") < rendered.index(
            "Latest event"
        )
        assert rendered.index("Latest event") < rendered.index("Time real/user/sys")
    finally:
        sink.close()


def test_rich_live_formats_profile_summary() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        sink.state["timing_profile"] = {
            "enabled": True,
            "dominant_phase": "proposal_generation",
            "dominant_seconds": 12.34567,
            "unattributed_fraction": 0.042,
        }
        assert sink._profile_summary() == "proposal_generation 12.346s / 4.2%"
        sink.state["timing_profile"] = {"enabled": False}
        assert sink._profile_summary() == "disabled"
    finally:
        sink.close()


def test_rich_live_renders_full_runtime_profile_table() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        sink.state["timing_profile"] = {
            "enabled": True,
            "profiled_episodes": 2,
            "phase_seconds": {
                "scoring": 2.0,
                "proposal_generation": 5.0,
                "exact_verification": 0.0,
            },
            "phase_children_seconds": {
                "proposal_generation": {
                    "rng_setup": 0.5,
                    "graph_materialization": 1.0,
                    "operator_search": 3.0,
                    "proposal_packaging": 0.4,
                    "other": 0.1,
                }
            },
            "phase_calls": {"proposal_generation": 20},
            "phase_children_calls": {
                "proposal_generation": {
                    "rng_setup": 20,
                    "graph_materialization": 20,
                    "operator_search": 20,
                    "proposal_packaging": 20,
                    "other": None,
                }
            },
            "phase_grandchildren_seconds": {
                "proposal_generation": {
                    "operator_search": {
                        "heg_uniform_two_switch": 1.0,
                        "heg_forbidden_cycle_break": 2.0,
                    }
                }
            },
            "phase_grandchildren_calls": {
                "proposal_generation": {
                    "operator_search": {
                        "heg_uniform_two_switch": 8,
                        "heg_forbidden_cycle_break": 12,
                    }
                }
            },
            "measured_total_seconds": 8.0,
            "accounted_seconds": 7.0,
            "unattributed_seconds": 1.0,
            "unattributed_fraction": 0.125,
            "dominant_phase": "proposal_generation",
            "dominant_seconds": 5.0,
        }
        sink.state["real_seconds"] = 8.5
        sink.state["user_seconds"] = 7.25
        sink.state["system_seconds"] = 0.5
        output = io.StringIO()
        Console(file=output, force_terminal=False, width=80).print(sink._render())
        rendered = output.getvalue()
        assert "Runtime profile · 2 episodes" in rendered
        assert rendered.index("proposal_generation") < rendered.index("scoring")
        assert "Calls" in rendered
        assert "Wall [s]" in rendered
        assert "Of parent" in rendered
        assert "Of episode" in rendered
        assert "├─ rng_setup" in rendered
        assert "└─ other" in rendered
        operator_line = next(
            line for line in rendered.splitlines() if "operator_search" in line
        )
        assert "60.0%" in operator_line
        assert "37.5%" in operator_line
        assert "20" in operator_line
        wide_output = io.StringIO()
        profile_table = sink._profile_table()
        assert profile_table is not None
        Console(file=wide_output, force_terminal=False, width=160).print(profile_table)
        wide_rendered = wide_output.getvalue()
        uniform_line = next(
            line
            for line in wide_rendered.splitlines()
            if "heg_uniform_two_switch" in line
        )
        forbidden_line = next(
            line
            for line in wide_rendered.splitlines()
            if "heg_forbidden_cycle_break" in line
        )
        assert "│  ├─ heg_uniform_two_switch" in uniform_line
        assert "8" in uniform_line
        assert "33.3%" in uniform_line
        assert "12.5%" in uniform_line
        assert "│  └─ heg_forbidden_cycle_break" in forbidden_line
        assert "12" in forbidden_line
        assert "66.7%" in forbidden_line
        assert "25.0%" in forbidden_line
        other_line = next(
            line for line in rendered.splitlines() if "└─ other" in line
        )
        assert "—" in other_line
        assert "phases subtotal" in rendered
        assert "other in episodes" in rendered
        assert "episode wall total" in rendered
        assert "62.5%" in rendered
        assert rendered.index("episode wall total") < rendered.index(
            "Run real/user/sys"
        )
        assert "8.500 / 7.250 / 0.500" in rendered
        assert rendered.count("Run real/user/sys") == 1
        process_line = next(
            line for line in rendered.splitlines() if "Run real/user/sys" in line
        )
        assert process_line.count("│") == 2
        assert "│" in rendered
        assert "┼" in rendered
        assert profile_table.box is box.MINIMAL
        assert profile_table.border_style == "grey37"
        assert profile_table.rows[-1].style == "bright_cyan"

        profile = sink.state["timing_profile"]
        assert isinstance(profile, dict)
        del profile["phase_grandchildren_seconds"]
        del profile["phase_grandchildren_calls"]
        assert sink._profile_table() is not None
    finally:
        sink.close()


def test_rich_live_renders_separate_deep_operator_profile() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        sink.state["timing_profile"] = {"enabled": False}
        sink.state["deep_operator_profile"] = {
            "enabled": True,
            "profiled_episodes": 1,
            "operators": {
                "heg_forbidden_cycle_break": {
                    "seconds": 1.0,
                    "calls": 10,
                    "children": {
                        "witness_search": {
                            "seconds": 0.4,
                            "calls": 10,
                        },
                        "witness_edge_materialization": {
                            "seconds": 0.1,
                            "calls": None,
                        },
                        "switch_attempts": {
                            "seconds": 0.35,
                            "calls": 40,
                            "counters": {
                                "timing_scope": "measured children"
                            },
                            "children": {
                                "partner_edge_sampling": {
                                    "seconds": 0.05,
                                    "calls": None,
                                },
                                "candidate_construction": {
                                    "seconds": 0.2,
                                    "calls": None,
                                },
                                "connectivity_validation": {
                                    "seconds": 0.1,
                                    "calls": None,
                                },
                                "graph_family_validation": {
                                    "seconds": 0.0,
                                    "calls": None,
                                },
                            },
                        },
                        "other": {"seconds": 0.15, "calls": None},
                    },
                }
            },
        }
        output = io.StringIO()
        Console(file=output, force_terminal=False, width=180).print(sink._render())
        rendered = output.getvalue()
        assert "Deep operator profile · 1 episodes" in rendered
        assert "Runtime profile" not in rendered
        assert "heg_forbidden_cycle_break" in rendered
        assert "├─ witness_search" in rendered
        assert "├─ switch_attempts" in rendered
        assert "│  ├─ partner_edge_sampling" in rendered
        assert "│  └─ graph_family_validation" in rendered
        assert "measured children" in rendered
        assert "40.0%" in rendered
        assert "Time real/user/sys" in rendered
        table = sink._deep_profile_table()
        assert table is not None
        assert table.box is box.MINIMAL
        assert table.border_style == "grey37"

        sink.state["deep_operator_profile"] = {"enabled": False}
        assert sink._deep_profile_table() is None
    finally:
        sink.close()


def test_rich_live_renders_separate_deep_score_profile() -> None:
    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    try:
        sink.state["timing_profile"] = {"enabled": False}
        sink.state["deep_score_profile"] = {
            "enabled": True,
            "profiled_episodes": 2,
            "counters": {
                "score_result_full_results": 20,
                "score_result_dominated_results": 70,
                "score_result_failures": 1,
                "score_cache_hits": 9,
                "score_cache_misses": 81,
                "score_cache_lookups": 90,
                "worker_failure_calls": 1,
                "worker_restart_successes": 1,
                "python_fallback_calls": 0,
            },
            "prepared_graph": {
                "materialization": {"seconds": 0.4, "calls": 80},
                "validation": {"seconds": 0.2, "calls": 80},
            },
            "worker": {
                "seconds": 4.0,
                "calls": 81,
                "protocol_overhead_seconds": 0.5,
                "children": {
                    "request_packing": {"seconds": 0.1, "calls": 0},
                    "request_write": {"seconds": 0.1, "calls": 0},
                    "worker_wait_and_read": {"seconds": 0.2, "calls": 0},
                    "response_parsing": {"seconds": 0.1, "calls": 0},
                    "cycle_4": {"seconds": 0.5, "calls": 81},
                    "cycle_8": {"seconds": 1.0, "calls": 81},
                    "cycle_16": {"seconds": 1.5, "calls": 81},
                    "other": {"seconds": 0.5, "calls": 0},
                },
            },
            "score_assembly": {"seconds": 0.3, "calls": 20},
        }

        output = io.StringIO()
        Console(file=output, force_terminal=False, width=180).print(sink._render())
        rendered = output.getvalue()
        assert "Deep score profile · 2 episodes" in rendered
        assert "worker_roundtrip" in rendered
        assert "├─ cycle_4" in rendered
        assert "├─ cycle_8" in rendered
        assert "├─ cycle_16" in rendered
        assert "graph_materialization" in rendered
        assert "validation" in rendered
        assert "score_assembly" in rendered
        assert "hits / misses / lookups  9/81/90" in rendered
        assert "full 20 · dominated 70 · failures 1" in rendered
        assert "failures 1 · restarts 1 · fallbacks 0" in rendered
        table = sink._deep_score_profile_table()
        assert table is not None
        assert table.box is box.MINIMAL
        assert table.border_style == "grey37"

        sink.state["deep_score_profile"] = {"enabled": False}
        assert sink._deep_score_profile_table() is None
    finally:
        sink.close()


def test_rich_live_renders_at_most_once_per_second_and_on_terminal_event() -> None:
    with patch("mutation_forge.output.rich_live.time.monotonic") as monotonic:
        monotonic.return_value = 0.0
        sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
        try:
            with (
                patch.object(sink.live, "update") as update,
                patch.object(sink, "_render", wraps=sink._render) as render,
            ):
                monotonic.return_value = 0.2
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:00+00:00",
                        run_id="run-1",
                        event_type="episode_progress",
                        payload={"evaluations": 50},
                    )
                )
                monotonic.return_value = 1.0
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:01+00:00",
                        run_id="run-1",
                        event_type="episode_progress",
                        payload={"evaluations": 100},
                    )
                )
                monotonic.return_value = 1.2
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:01.2+00:00",
                        run_id="run-1",
                        event_type="episode_progress",
                        payload={"evaluations": 150},
                    )
                )
                monotonic.return_value = 1.3
                sink.write(
                    Event(
                        schema_version="1.0",
                        timestamp="2026-07-28T00:00:01+00:00",
                        run_id="run-1",
                        event_type="run_completed",
                        payload={
                            "real_seconds": 1.0,
                            "user_seconds": 0.8,
                            "system_seconds": 0.1,
                        },
                    )
                )
            assert update.call_count == 2
            assert render.call_count == 2
            assert all(call.kwargs == {"refresh": True} for call in update.call_args_list)
        finally:
            sink.close()
