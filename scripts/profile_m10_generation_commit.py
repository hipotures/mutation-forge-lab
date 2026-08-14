"""Provider-free commit/memory fixture for one realistic M10 generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutation_forge.models import JsonValue
from mutation_forge.native_v3_python import scientific_search as search
from mutation_forge.native_v3_python.scientific_evaluation import (
    ScientificEvaluationOptionsV1,
)
from mutation_forge.native_v3_python.search import DevelopmentCaseV1


def _panel() -> tuple[DevelopmentCaseV1, ...]:
    return tuple(
        DevelopmentCaseV1(
            f"case-{index:04d}",
            22 + index % 5,
            400 + index % 4,
            4000 + index % 16,
            32,
            64,
            (4, 8),
        )
        for index in range(320)
    )


def _options() -> search.ScientificSearchOptionsV2:
    return search.ScientificSearchOptionsV2(
        generation_limit=1,
        evaluator_workers=12,
        provider_concurrency=2,
        wall_seconds=3600.0,
        primary_program_slots=8,
        repair_turn_limit=0,
        provider_total_turn_limit=8,
        stop_on_verified=True,
        resume_enabled=True,
        replace_terminal_slots=False,
        evaluation=ScientificEvaluationOptionsV1(
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
        ),
    )


class _Evaluator:
    def __init__(self, payload_kib: int) -> None:
        self._padding = "x" * (payload_kib * 1024)

    def _result(self, label: str, case: DevelopmentCaseV1) -> Mapping[str, JsonValue]:
        digest = hashlib.sha256(f"{label}:{case.case_id}".encode()).hexdigest()
        return {
            "behavior_identity": {"behavior_signature": digest},
            "scientific_result": {
                "status": "COMPLETE",
                "fitness_interval": {
                    "lower": {"numerator": 1, "denominator": 3},
                    "upper": {"numerator": 2, "denominator": 5},
                },
                "semantic_trace_hash": digest,
                "initial_counterexample": None,
                "initial_evidence": {"components": []},
                "terminal_evidence": {"components": []},
                "steps": [],
            },
            "worker_telemetry": {"worker_rss_kib": 1},
            "runtime_profile": {},
            "fixture_padding": f"{case.case_id}:{self._padding}",
        }

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        del source
        return self._result(candidate_id, case)

    def evaluate_baseline(
        self,
        *,
        baseline: str,
        case: DevelopmentCaseV1,
        generation: int,
    ) -> Mapping[str, JsonValue]:
        return self._result(f"{baseline}:{generation}", case)

    def close(self) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-kib", type=int, default=256)
    args = parser.parse_args()
    panel = _panel()
    options = _options()
    commit_wall = 0.0
    commit_cpu = 0.0
    evaluation_reads = 0
    original_load = search.core._load_mapping

    def counted_load(path: Path) -> dict[str, Any]:
        nonlocal evaluation_reads
        if path.parent.name == "evaluations":
            evaluation_reads += 1
        return original_load(path)

    search.core._load_mapping = counted_load
    with tempfile.TemporaryDirectory(prefix="mforge-m10-profile-") as raw_root:
        root = Path(raw_root)
        telemetry = search._RuntimeTelemetry(root, options)
        pool = search._ConcurrentEvaluatorPool(
            workers=12,
            queue_capacity=48,
            evaluator_factory=lambda: _Evaluator(args.payload_kib),
            telemetry=telemetry,
        )
        for baseline in options.evaluation.baselines:
            pool.submit_baseline(
                baseline=baseline,
                panel=panel,
                generation=0,
                generation_dir=root / "generations" / "generation-0000",
            ).result()
        for slot_index in range(8):
            candidate_id = f"g0000-slot-{slot_index:02d}"
            slot_dir = root / "generations" / "generation-0000" / f"slot-{slot_index:02d}"
            source = "def propose(ctx, graph, api, seed):\n    return api.no_plan('fixture')\n"
            future = pool.submit(
                source=source,
                panel=panel,
                candidate_id=candidate_id,
                slot_dir=slot_dir,
            )
            future.result()
            pending = search._PendingCommit(
                slot_plan=search.core.SlotPlanV1(
                    slot=f"slot-{slot_index:02d}",
                    kind="root",
                    parent_candidate_id=None,
                    panel_hash=search.core.panel_hash(panel),
                    request_key=f"request-{slot_index:02d}",
                ),
                candidate_id=candidate_id,
                slot_dir=slot_dir,
                prepared={
                    "protocol_id": search.M10_PREPARED_CANDIDATE_PROTOCOL_ID,
                    "status": "evaluation_pending",
                    "candidate_id": candidate_id,
                    "generation": 0,
                    "slot": f"slot-{slot_index:02d}",
                    "kind": "root",
                    "parent_candidate_id": None,
                    "program_hash": f"{slot_index:064x}",
                    "source": source,
                },
                future=future,
            )
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            search._commit_pending(
                pending=pending,
                root=root,
                panel=panel,
                telemetry=telemetry,
                block=True,
                boundary_hook=None,
            )
            commit_wall += time.perf_counter() - wall_started
            commit_cpu += time.process_time() - cpu_started
        pool.close()

    print(
        json.dumps(
            {
                "panels": 10,
                "cases_per_panel": 320,
                "evaluations": 3200,
                "payload_kib": args.payload_kib,
                "candidate_commit_wall_seconds": commit_wall,
                "candidate_commit_cpu_seconds": commit_cpu,
                "candidate_commit_evaluation_reads": evaluation_reads,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
