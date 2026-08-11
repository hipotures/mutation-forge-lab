#!/usr/bin/env python3
"""Profile one provider-free ordinary-Python scientific evaluator case."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

from mutation_forge.backends.heg import HegBackend
from mutation_forge.native_v3.heg_scoring import HegScoreEvidenceAdapter
from mutation_forge.native_v3_python.runner import IsolatedPolicyWorkerV1
from mutation_forge.native_v3_python.safe_api import SafeGraphSessionV1
from mutation_forge.native_v3_python.search import DevelopmentCaseV1
from mutation_forge.native_v3_python.search_provider import PythonPanelScientificEvaluator

_POLICY_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.matching_k_switch_reconnections(2)
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    matching = api.pick(candidates, seed, "duty-cycle-profile")
    if matching == None:
        return api.no_plan(reason="NO_MATCH")
    api.k_switch(matching)
    return api.emit()
"""


def _process_cpu_seconds(pid: int | None) -> float:
    if pid is None:
        return 0.0
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    _comm, separator, fields = payload.rpartition(") ")
    if not separator:
        raise RuntimeError(f"cannot parse /proc/{pid}/stat")
    values = fields.split()
    ticks = int(values[11]) + int(values[12])
    return ticks / os.sysconf("SC_CLK_TCK")


def _percent(seconds: float, total: float) -> float:
    return 100.0 * seconds / total if total > 0.0 else 0.0


def profile_case(
    heg_repo: Path,
    *,
    source: str,
    order: int,
    graph_seed: int,
    policy_seed: int,
    horizon: int,
    witness_cap: int,
) -> dict[str, Any]:
    backend = HegBackend(heg_repo)
    buckets: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    try:
        warm_graph = backend.generate_seed(order=30, seed=101)
        warm_started = time.perf_counter()
        backend.score(warm_graph, witness_cap=witness_cap)
        buckets["cpp_worker_cold_start"] = time.perf_counter() - warm_started
        score_worker = backend._score_worker()  # noqa: SLF001
        original_worker_score = score_worker.score
        original_evidence_score = HegScoreEvidenceAdapter.score_evidence
        original_policy_invoke = IsolatedPolicyWorkerV1.invoke
        original_api_call = SafeGraphSessionV1.handle_call
        original_canonical_hash = backend.canonical_hash

        def worker_score(*args: Any, **kwargs: Any) -> Any:
            kwargs["profile_timing"] = True
            pid_before = getattr(getattr(score_worker, "process", None), "pid", None)
            cpu_before = _process_cpu_seconds(pid_before)
            started = time.perf_counter()
            response = original_worker_score(*args, **kwargs)
            buckets["score_worker_roundtrip"] += time.perf_counter() - started
            pid_after = getattr(getattr(score_worker, "process", None), "pid", None)
            buckets["cpp_worker_cpu"] += max(
                0.0,
                _process_cpu_seconds(pid_after) - cpu_before,
            )
            counts["score_worker_calls"] += 1
            timing = response.timing
            if timing is not None:
                buckets["score_request_packing"] += timing.request_packing_ns / 1e9
                buckets["score_request_write"] += timing.request_write_ns / 1e9
                buckets["score_response_read"] += timing.response_read_ns / 1e9
                buckets["score_response_parsing"] += timing.response_parsing_ns / 1e9
                buckets["score_reported_roundtrip"] += timing.worker_roundtrip_ns / 1e9
            buckets["cpp_cycle_search"] += sum(
                result.elapsed_ns for result in response.results
            ) / 1e9
            return response

        def evidence_score(
            self: HegScoreEvidenceAdapter,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            unique_before = self.unique_graph_scores
            started = time.perf_counter()
            result = original_evidence_score(self, *args, **kwargs)
            elapsed = time.perf_counter() - started
            buckets["score_evidence_total"] += elapsed
            if self.unique_graph_scores == unique_before:
                buckets["score_cache_hit"] += elapsed
                counts["score_cache_hits"] += 1
            return result

        def policy_invoke(self: IsolatedPolicyWorkerV1, *args: Any, **kwargs: Any) -> Any:
            cpu_before = _process_cpu_seconds(self._process.pid)  # noqa: SLF001
            result = original_policy_invoke(self, *args, **kwargs)
            buckets["policy_child_cpu"] += max(
                0.0,
                _process_cpu_seconds(self._process.pid) - cpu_before,  # noqa: SLF001
            )
            counts["policy_invocations"] += 1
            return result

        def api_call(
            self: SafeGraphSessionV1,
            method: str,
            arguments: Any,
        ) -> Any:
            started = time.perf_counter()
            try:
                return original_api_call(self, method, arguments)
            finally:
                elapsed = time.perf_counter() - started
                if method in {"emit", "no_plan"}:
                    buckets["rewrite_validation"] += elapsed
                else:
                    buckets["safe_api_host"] += elapsed
                counts[f"safe_api_{method}"] += 1

        def canonical_hash(*args: Any, **kwargs: Any) -> str:
            started = time.perf_counter()
            try:
                return original_canonical_hash(*args, **kwargs)
            finally:
                buckets["canonical_hashing"] += time.perf_counter() - started
                counts["canonical_hash_calls"] += 1

        case = DevelopmentCaseV1(
            case_id=(
                f"profile-o{order:04d}-g{graph_seed:04d}-p{policy_seed:04d}"
            ),
            order=order,
            graph_seed=graph_seed,
            policy_seed=policy_seed,
            horizon=horizon,
            witness_cap=witness_cap,
            forbidden_lengths=backend.target_forbidden_lengths(order),
        )
        with (
            tempfile.TemporaryDirectory(prefix="mforge-duty-cycle-profile-") as artifacts,
            patch.object(score_worker, "score", worker_score),
            patch.object(HegScoreEvidenceAdapter, "score_evidence", evidence_score),
            patch.object(IsolatedPolicyWorkerV1, "invoke", policy_invoke),
            patch.object(SafeGraphSessionV1, "handle_call", api_call),
            patch.object(backend, "canonical_hash", canonical_hash),
        ):
            evaluator = PythonPanelScientificEvaluator(
                backend=backend,
                artifact_root=artifacts,
            )
            started = time.perf_counter()
            result = evaluator.evaluate(
                source=source,
                case=case,
                candidate_id="provider-free-duty-cycle-profile",
            )
            total = time.perf_counter() - started

        runtime = result["runtime_profile"]
        worker_telemetry = result["worker_telemetry"]
        scientific = result["scientific_result"]
        sandbox = float(runtime["sandbox_wall_seconds"])
        safe_api = buckets["safe_api_host"] + buckets["rewrite_validation"]
        policy_and_ipc = max(0.0, sandbox - safe_api)
        policy_cpu = min(buckets["policy_child_cpu"], policy_and_ipc)
        policy_ipc = max(0.0, policy_and_ipc - policy_cpu)
        score_protocol = max(
            0.0,
            buckets["score_worker_roundtrip"] - buckets["cpp_cycle_search"],
        )
        score_host = max(
            0.0,
            buckets["score_evidence_total"]
            - buckets["score_worker_roundtrip"]
            - buckets["score_cache_hit"],
        )
        startup = float(worker_telemetry["startup_seconds"])
        accounted = (
            buckets["cpp_cycle_search"]
            + score_protocol
            + score_host
            + buckets["score_cache_hit"]
            + policy_cpu
            + policy_ipc
            + buckets["safe_api_host"]
            + buckets["rewrite_validation"]
            + buckets["canonical_hashing"]
            + startup
        )
        residual = max(0.0, total - accounted)
        attribution = {
            "cpp_cycle_search": buckets["cpp_cycle_search"],
            "score_worker_protocol": score_protocol,
            "score_evidence_host": score_host,
            "score_cache_hits": buckets["score_cache_hit"],
            "policy_execution_cpu": policy_cpu,
            "policy_ipc_framing_scheduler": policy_ipc,
            "safe_api_host": buckets["safe_api_host"],
            "rewrite_validation": buckets["rewrite_validation"],
            "canonical_hashing": buckets["canonical_hashing"],
            "policy_worker_startup": startup,
            "evaluator_bookkeeping_residual": residual,
        }
        return {
            "protocol_id": "mforge.native.python_evaluator_duty_cycle_profile.v1",
            "provider_turns": 0,
            "model_turns": 0,
            "app_server_calls": 0,
            "case": {
                "order": order,
                "graph_seed": graph_seed,
                "policy_seed": policy_seed,
                "horizon": horizon,
                "witness_cap": witness_cap,
            },
            "total_wall_seconds": total,
            "attribution": {
                name: {
                    "seconds": seconds,
                    "percent": _percent(seconds, total),
                }
                for name, seconds in attribution.items()
            },
            "diagnostics": {
                "cpp_worker_cpu_seconds": buckets["cpp_worker_cpu"],
                "cpp_worker_cpu_percent_of_wall": _percent(
                    buckets["cpp_worker_cpu"],
                    total,
                ),
                "cpp_worker_cold_start_seconds": buckets["cpp_worker_cold_start"],
                "score_request_packing_seconds": buckets["score_request_packing"],
                "score_request_write_seconds": buckets["score_request_write"],
                "score_response_read_seconds": buckets["score_response_read"],
                "score_response_parsing_seconds": buckets["score_response_parsing"],
                "score_reported_roundtrip_seconds": buckets[
                    "score_reported_roundtrip"
                ],
                "sandbox_wall_seconds": sandbox,
                "selector_wall_seconds": runtime["selector_wall_seconds"],
                "action_wall_seconds": runtime["action_wall_seconds"],
                "score_attempts": scientific["score_attempts"],
                "unique_graph_scores": scientific["unique_graph_scores"],
                "accepted_rewrites": scientific["accepted_rewrites"],
                "counts": dict(sorted(counts.items())),
            },
        }
    finally:
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heg-repo", type=Path, default=Path("../heg"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--order", type=int, default=37)
    parser.add_argument("--graph-seed", type=int, default=401)
    parser.add_argument("--policy-seed", type=int, default=4001)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--witness-cap", type=int, default=64)
    args = parser.parse_args()
    source = (
        args.source.read_text(encoding="utf-8")
        if args.source is not None
        else _POLICY_SOURCE
    )
    report = profile_case(
        args.heg_repo,
        source=source,
        order=args.order,
        graph_seed=args.graph_seed,
        policy_seed=args.policy_seed,
        horizon=args.horizon,
        witness_cap=args.witness_cap,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
