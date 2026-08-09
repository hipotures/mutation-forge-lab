"""Provider-free benchmark for ordinary-Python scientific evaluation profiles."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import cast

from mutation_forge.backends.heg import HegBackend
from mutation_forge.models import JsonValue

from .runtime_contracts import PolicyRuntimeLimitsV1
from .scientific_evaluation import ScientificEvaluationOptionsV1
from .search import DevelopmentCaseV1
from .search_provider import PythonPanelScientificEvaluator

_NO_PLAN_SOURCE = """\
def propose(ctx, graph, api, seed):
    return api.no_plan()
"""


def _parity_options() -> ScientificEvaluationOptionsV1:
    return ScientificEvaluationOptionsV1(
        graph_mode="unrestricted_min_degree_3",
        order_schedule="adaptive",
        min_order=22,
        max_order=128,
        orders_per_generation=5,
        graph_seeds=(401, 402, 403, 404),
        policy_seeds=tuple(range(4001, 4017)),
        horizon=32,
        witness_cap=64,
        baselines=("random", "structural"),
        replay=False,
    )


def benchmark_evaluation_profiles(
    heg_repo: str | Path,
    *,
    sample_cases: int = 2,
) -> dict[str, JsonValue]:
    """Compare sampled evaluation cost and project the complete workloads."""

    if sample_cases < 1:
        raise ValueError("sample_cases must be positive")

    backend = HegBackend(
        Path(heg_repo),
        graph_mode="unrestricted_min_degree_3",
    )
    try:
        tiny_panel = (
            DevelopmentCaseV1(
                case_id="tiny-order-30-seed-101",
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                forbidden_lengths=backend.target_forbidden_lengths(30),
            ),
            DevelopmentCaseV1(
                case_id="tiny-order-30-seed-103",
                order=30,
                graph_seed=103,
                policy_seed=19,
                horizon=1,
                witness_cap=64,
                forbidden_lengths=backend.target_forbidden_lengths(30),
            ),
        )
        parity = _parity_options()
        parity_panel = parity.panel_for_generation(
            generation=0,
            backend=backend,
        )
        with tempfile.TemporaryDirectory(prefix="mforge-evaluation-parity-") as artifact_root:
            evaluator = PythonPanelScientificEvaluator(
                backend=backend,
                artifact_root=artifact_root,
                runtime_limits=PolicyRuntimeLimitsV1(),
            )
            profiles: dict[str, JsonValue] = {}
            for name, panel in (
                ("removed_tiny_panel", tiny_panel),
                ("native_v2_parity_profile", parity_panel),
            ):
                measured_panel = panel[:sample_cases]
                started = time.perf_counter()
                for case in measured_panel:
                    evaluator.evaluate(
                        source=_NO_PLAN_SOURCE,
                        case=case,
                        candidate_id=f"benchmark-{name}",
                    )
                candidate_seconds = time.perf_counter() - started
                baseline_seconds: dict[str, float] = {}
                for baseline in parity.baselines:
                    started = time.perf_counter()
                    for case in measured_panel:
                        evaluator.evaluate_baseline(
                            baseline=baseline,
                            case=case,
                            generation=0,
                        )
                    baseline_seconds[baseline] = time.perf_counter() - started
                profiles[name] = {
                    "case_count": len(panel),
                    "measured_case_count": len(measured_panel),
                    "horizon": panel[0].horizon,
                    "candidate_policy_invocations": (len(panel) * panel[0].horizon),
                    "measured_candidate_policy_invocations": (
                        len(measured_panel) * panel[0].horizon
                    ),
                    "candidate_seconds": candidate_seconds,
                    "baseline_seconds": cast(JsonValue, baseline_seconds),
                    "total_seconds": candidate_seconds + sum(baseline_seconds.values()),
                    "projected_total_seconds": (
                        (candidate_seconds + sum(baseline_seconds.values()))
                        * len(panel)
                        / len(measured_panel)
                    ),
                }
    finally:
        backend.close()
    return {
        "protocol_id": "mforge.native.python_evaluation_parity_benchmark.v1",
        "provider_turns": 0,
        "model_turns": 0,
        "app_server_calls": 0,
        "sample_cases": sample_cases,
        "profiles": profiles,
    }


__all__ = ["benchmark_evaluation_profiles"]
