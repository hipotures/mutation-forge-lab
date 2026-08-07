from __future__ import annotations

from mutation_forge.native_v3_python.benchmark import (
    REPORT_SCHEMA_VERSION,
    run_benchmark,
)


def test_runtime_benchmark_reports_all_provisional_limit_headroom() -> None:
    report = run_benchmark(startup_runs=2, call_runs=7, failure_runs=2)
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["sample_counts"] == {
        "startup": 2,
        "normal_calls": 7,
        "selector_action_calls": 7,
        "failure_paths": 2,
    }
    limits = report["provisional_limits"]
    assert limits["propose_wall_seconds"] == 1.0
    assert limits["address_space_bytes"] == 256 * 1024 * 1024
    assert limits["worker_lifetime_seconds"] == 60.0
    assert limits["graph_order"] == 128
    measurements = report["measurements"]
    for name in (
        "startup",
        "normal_call",
        "selector_action_call",
        "invalid_return_failure_total",
    ):
        assert set(measurements[name]) == {"p50_ms", "p95_ms", "max_ms"}
        assert 0 <= measurements[name]["p50_ms"] <= measurements[name]["max_ms"]
        assert 0 <= measurements[name]["p95_ms"] <= measurements[name]["max_ms"]
    matrix = measurements["selector_action_matrix"]
    assert sum(case["calls"] for case in matrix.values()) == 7
    assert {case["graph_order"] for case in matrix.values()} == {30, 64, 128}
    assert report["headroom"]["graph_order"] == 0
    assert all(
        value > 0
        for name, value in report["headroom"].items()
        if name != "graph_order"
    )
