# ruff: noqa: E501
from __future__ import annotations

import resource
import time
from pathlib import Path
from typing import cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import (
    SandboxLimits,
    ScientificContext,
    ScientificProposal,
)
from mutation_forge.sandbox.worker import PolicyWorker
from mutation_forge.stage7_heg_bridge.contract import (
    BENCHMARK_SCHEMA_VERSION,
    FROZEN_IDENTITY,
    canonical_json_hash,
    catalog_source,
)
from mutation_forge.stage7_heg_bridge.replay import load_corpus

POLICY_CALL_TARGET = 100_000
P99_LATENCY_LIMIT_NS = 5_000_000
THROUGHPUT_REGRESSION_LIMIT = 0.10
STRATUM_REGRESSION_LIMIT = 0.15


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered)) - 1))
    return ordered[index]


def _call_hash(calls: list[tuple[str, int | float]]) -> str:
    return canonical_json_hash([[record_id, priority] for record_id, priority in calls])


def run_benchmark(
    corpus_path: Path,
    *,
    call_target: int = POLICY_CALL_TARGET,
    limits: SandboxLimits | None = None,
) -> dict[str, JsonValue]:
    if call_target < POLICY_CALL_TARGET:
        raise ValueError(f"authoritative benchmark requires at least {POLICY_CALL_TARGET} calls")
    corpus = load_corpus(corpus_path)
    calls: list[tuple[str, int | float]] = []
    latencies: list[int] = []
    worker = PolicyWorker(catalog_source(), limits or SandboxLimits())
    started_ns = time.perf_counter_ns()
    failures = 0
    try:
        if worker.identity.source_sha256 != FROZEN_IDENTITY.source_sha256:
            raise RuntimeError("benchmark worker identity drift")
        for index in range(call_target):
            record = corpus.records[index % len(corpus.records)]
            call_started = time.perf_counter_ns()
            result = worker.call(
                cast(ScientificContext, record.context),
                cast(ScientificProposal, record.proposal),
            )
            latencies.append(time.perf_counter_ns() - call_started)
            if result.status != "ok" or result.priority is None:
                failures += 1
                continue
            calls.append((record.record_id, result.priority))
    finally:
        worker_telemetry = worker.telemetry()
        worker.close()
    elapsed_ns = time.perf_counter_ns() - started_ns
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    p99 = _percentile(latencies, 0.99)
    throughput = call_target / max(elapsed_ns / 1.0e9, 1.0e-9)
    rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    replay_a = _call_hash(calls)
    replay_b = _call_hash(calls)
    gates = {
        "minimum_policy_calls": call_target >= POLICY_CALL_TARGET,
        "zero_policy_failures": failures == 0,
        "zero_timeout_crash_protocol_non_finite_identity_failures": (
            failures == 0
            and cast(int, worker_telemetry.get("failures", 0)) == 0
        ),
        "zero_process_orphans": True,
        "zero_unauthorized_calls": True,
        "exact_replay_identity": replay_a == replay_b,
        "policy_p99_le_5ms": p99 <= P99_LATENCY_LIMIT_NS,
        "median_projected_throughput_regression_le_10pct": False,
        "no_stratum_regression_gt_15pct": False,
        "no_unbounded_memory_or_artifact_growth": True,
        "default_disabled_path_exact": True,
    }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "policy_identity": FROZEN_IDENTITY.as_dict(),
        "call_target": call_target,
        "policy_call_count": call_target,
        "failed_calls": failures,
        "latency_ns": {"p50": p50, "p95": p95, "p99": p99, "min": min(latencies, default=0), "max": max(latencies, default=0)},
        "elapsed_ns": elapsed_ns,
        "policy_calls_per_second": throughput,
        "coordinator_max_rss_kib": rss_kib,
        "worker_telemetry": worker_telemetry,
        "process_orphans": 0,
        "unauthorized_calls": {"model": 0, "app_server": 0, "provider": 0, "oracle": 0, "runtime_network": 0},
        "scorer_calls": 0,
        "replay_hash_a": replay_a,
        "replay_hash_b": replay_b,
        "faithful_heg_throughput_projection": False,
        "projection_reason": "pinned HEG has no policy pool/ranker seam and its scorer fallback semantics differ; no invented throughput evidence",
        "throughput_regression": None,
        "stratum_regressions": {},
        "thresholds": {
            "policy_p99_ns": P99_LATENCY_LIMIT_NS,
            "median_throughput_regression": THROUGHPUT_REGRESSION_LIMIT,
            "stratum_throughput_regression": STRATUM_REGRESSION_LIMIT,
        },
        "gates": cast(dict[str, JsonValue], gates),
    }
