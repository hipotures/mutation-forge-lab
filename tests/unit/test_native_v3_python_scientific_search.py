from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge.experiment.config import orders_for_generation
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3_python import preview as preview_module
from mutation_forge.native_v3_python import scientific_search as search_module
from mutation_forge.native_v3_python import search_provider as provider_module
from mutation_forge.native_v3_python.contracts import (
    PYTHON_EXPERIMENT_PROTOCOL_ID,
)
from mutation_forge.native_v3_python.preview import (
    PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION,
    load_python_preview_config,
    python_preview_status,
    run_python_preview,
)
from mutation_forge.native_v3_python.scientific_evaluation import (
    ScientificEvaluationOptionsV1,
)
from mutation_forge.native_v3_python.scientific_search import (
    M10_REPORT_PROTOCOL_ID,
    M10_SEARCH_PROTOCOL_ID,
    M10_STOP_FILENAME,
    ScientificResumeBudgetV1,
    ScientificSearchOptionsV2,
    run_sustained_search,
)
from mutation_forge.native_v3_python.search import (
    DevelopmentCaseV1,
    M5InfrastructureError,
    M5OperatorStop,
    M5ProviderContextV1,
    M5ProviderResultV1,
)
from mutation_forge.native_v3_python.search_provider import (
    M10_PROVIDER_MAX_EVENTS,
    M10_PROVIDER_STDOUT_BYTES,
    M10_PROVIDER_TRANSCRIPT_BYTES,
    CodexM5SearchProvider,
    CodexM10SearchProvider,
    specification_ack_schema,
)
from mutation_forge.output.interactive_dashboard import (
    _objective,
    dashboard_state_from_python_status,
)

_PANEL = (
    DevelopmentCaseV1("case-00", 8, 101, 17, 1, 64, (4, 8)),
    DevelopmentCaseV1("case-01", 8, 103, 19, 1, 64, (4, 8)),
)
_POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "source"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": "mforge.native.python_policy_response.v1"},
        "source": {"type": "string"},
    },
}


def _source(label: str) -> str:
    return (
        "def propose(ctx, graph, api, seed):\n"
        "    candidates = api.non_edges_legal()\n"
        f'    candidate = api.pick(candidates, seed, "{label}")\n'
        "    if not candidate:\n"
        '        return api.no_plan("NO_MATCH")\n'
        "    api.add_edge(candidate)\n"
        "    return api.emit()\n"
    )


def _result(
    *,
    source: str,
    case: DevelopmentCaseV1,
) -> dict[str, JsonValue]:
    digest = hashlib.sha256(f"{source}:{case.case_id}".encode()).hexdigest()
    return {
        "behavior_identity": {
            "behavior_signature": digest,
            "probe_manifest_sha256": "a" * 64,
            "protocol_id": "fixture",
        },
        "scientific_result": {
            "status": "COMPLETE",
            "fitness_interval": {
                "lower": {"numerator": 1, "denominator": 2},
                "upper": {"numerator": 1, "denominator": 2},
            },
            "semantic_trace_hash": digest,
            "initial_counterexample": None,
            "initial_evidence": {
                "components": [
                    {
                        "forbidden_length": 4,
                        "lower_bound": 2,
                        "upper_bound": 2,
                    }
                ]
            },
            "terminal_evidence": {
                "components": [
                    {
                        "forbidden_length": 4,
                        "lower_bound": 1,
                        "upper_bound": 1,
                    }
                ]
            },
            "steps": [
                {
                    "outcome": "rewrite",
                    "accepted": True,
                    "no_plan_reason": None,
                    "counterexample": None,
                    "interpreter_trace": [
                        {"method": "non_edges_legal", "ordinal": 0},
                        {"method": "pick", "ordinal": 1},
                        {"method": "add_edge", "ordinal": 2},
                        {"method": "emit", "ordinal": 3},
                    ],
                }
            ],
        },
        "worker_telemetry": {
            "rotations": 0,
            "failures": 0,
            "worker_rss_kib": 1,
        },
        "runtime_profile": {
            "sandbox_wall_seconds": 0.001,
            "selector_wall_seconds": 0.0005,
            "action_wall_seconds": 0.0001,
        },
    }


def _usage() -> dict[str, JsonValue]:
    return {
        "inputTokens": 1,
        "cachedInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "outputTokens": 1,
        "reasoningOutputTokens": 0,
        "totalTokens": 2,
        "final": True,
        "partial": False,
    }


class _Provider:
    model = "fixture-model"
    effort = "medium"
    provider_concurrency = 4

    def __init__(
        self,
        durable: dict[str, M5ProviderResultV1] | None = None,
    ) -> None:
        self.durable = durable if durable is not None else {}
        self.calls: list[tuple[int, str]] = []
        self.prompts: dict[tuple[int, str], str] = {}
        self.anchor = M5ProviderContextV1(
            "anchor-thread",
            "anchor-turn",
            None,
            ("anchor-turn",),
        )
        self.snapshots: list[Mapping[str, Any]] = []
        self.released: list[tuple[int, str]] = []
        self.release_condition = threading.Condition()

    def prepare_generation(
        self,
        *,
        snapshot: Mapping[str, Any],
        **_: Any,
    ) -> None:
        self.snapshots.append(snapshot)

    def primary_lane(self, *, generation: int, slot: str) -> int:
        del generation
        return int(slot.removeprefix("slot-")) % self.provider_concurrency

    def await_primary_slot(self, *, generation: int, slot: str) -> None:
        slot_index = int(slot.removeprefix("slot-"))
        predecessor = (
            generation,
            f"slot-{slot_index - self.provider_concurrency:02d}",
        )
        if slot_index < self.provider_concurrency:
            return
        with self.release_condition:
            assert self.release_condition.wait_for(
                lambda: predecessor in self.released,
                timeout=1,
            )

    def release_primary_slot(self, *, generation: int, slot: str) -> None:
        with self.release_condition:
            self.released.append((generation, slot))
            self.release_condition.notify_all()

    def ensure_specification_anchor(
        self,
        **_: Any,
    ) -> M5ProviderResultV1:
        return M5ProviderResultV1(
            response_text=json.dumps(
                {
                    "schema_version": ("mforge.native.python_m5_specification_ack.v1"),
                    "ack": "specification-retained",
                }
            ),
            context=self.anchor,
            usage=_usage(),
            duration_ms=0,
            warnings=0,
        )

    def _program(
        self,
        *,
        parent: M5ProviderContextV1,
        generation: int,
        slot: str,
        idempotency_key: str,
        artifact_dir: Path,
        prompt: str,
    ) -> M5ProviderResultV1:
        if idempotency_key in self.durable:
            result = self.durable[idempotency_key]
            write_json(
                artifact_dir / "m5-provider-result.json.gz",
                result.as_dict(),
            )
            return result
        turn = f"turn-{generation}-{slot}"
        result = M5ProviderResultV1(
            response_text=json.dumps(
                {
                    "schema_version": ("mforge.native.python_policy_response.v1"),
                    "source": _source(f"{generation}-{slot}"),
                },
                separators=(",", ":"),
            ),
            context=M5ProviderContextV1(
                f"thread-{generation}-{slot}",
                turn,
                None,
                parent.included_turn_ids + (turn,),
            ),
            usage=_usage(),
            duration_ms=1,
            warnings=0,
        )
        self.durable[idempotency_key] = result
        self.calls.append((generation, slot))
        self.prompts[(generation, slot)] = prompt
        write_json(
            artifact_dir / "m5-provider-result.json.gz",
            result.as_dict(),
        )
        return result

    def generate_root(
        self,
        *,
        anchor: M5ProviderContextV1,
        generation: int,
        slot: str,
        idempotency_key: str,
        artifact_dir: Path,
        prompt: str,
        **_: Any,
    ) -> M5ProviderResultV1:
        return self._program(
            parent=anchor,
            generation=generation,
            slot=slot,
            idempotency_key=idempotency_key,
            artifact_dir=artifact_dir,
            prompt=prompt,
        )

    def generate_child(
        self,
        *,
        parent: M5ProviderContextV1,
        generation: int,
        slot: str,
        idempotency_key: str,
        artifact_dir: Path,
        prompt: str,
        **_: Any,
    ) -> M5ProviderResultV1:
        return self._program(
            parent=parent,
            generation=generation,
            slot=slot,
            idempotency_key=idempotency_key,
            artifact_dir=artifact_dir,
            prompt=prompt,
        )

    def repair(self, **_: Any) -> M5ProviderResultV1:
        raise AssertionError("fixture policies are valid")

    def close(self) -> None:
        pass


class _Backend:
    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        assert order >= 4
        return (4, 8)

    def close(self) -> None:
        pass


class _Concurrency:
    def __init__(self, parties: int = 2) -> None:
        self.barrier = threading.Barrier(parties, timeout=1)
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls: list[tuple[str, str]] = []
        self.barrier_candidates: set[str] = set()


class _Evaluator:
    def __init__(self, concurrency: _Concurrency) -> None:
        self.concurrency = concurrency

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        with self.concurrency.lock:
            self.concurrency.active += 1
            self.concurrency.peak = max(
                self.concurrency.peak,
                self.concurrency.active,
            )
            self.concurrency.calls.append((candidate_id, case.case_id))
            use_barrier = (
                candidate_id.endswith(("slot-00", "slot-01"))
                and candidate_id not in self.concurrency.barrier_candidates
            )
            if use_barrier:
                self.concurrency.barrier_candidates.add(candidate_id)
        try:
            if use_barrier:
                self.concurrency.barrier.wait()
            return _result(source=source, case=case)
        finally:
            with self.concurrency.lock:
                self.concurrency.active -= 1

    def evaluate_baseline(
        self,
        *,
        baseline: str,
        case: DevelopmentCaseV1,
        generation: int,
    ) -> Mapping[str, JsonValue]:
        return _result(
            source=f"{baseline}:{generation}",
            case=case,
        )


def _options(
    *,
    workers: int = 2,
    generations: int = 1,
    concurrency: int = 4,
    repairs: int = 0,
) -> ScientificSearchOptionsV2:
    return ScientificSearchOptionsV2(
        generation_limit=generations,
        evaluator_workers=workers,
        provider_concurrency=concurrency,
        wall_seconds=60.0,
        primary_program_slots=generations * 8,
        repair_turn_limit=repairs,
        provider_total_turn_limit=generations * 8 + repairs,
        validated_queue_target=workers * 2,
        validated_queue_capacity=workers * 4,
        stop_on_verified=True,
        resume_enabled=True,
        replace_terminal_slots=False,
        evaluation=ScientificEvaluationOptionsV1(
            graph_mode="unrestricted_min_degree_3",
            order_schedule="adaptive",
            min_order=8,
            max_order=8,
            orders_per_generation=1,
            graph_seeds=(101, 103),
            policy_seeds=(17, 19),
            horizon=1,
            witness_cap=64,
            baselines=("random", "structural"),
            replay=False,
        ),
    )


def _run(
    root: Path,
    *,
    provider: _Provider,
    concurrency: _Concurrency,
    options: ScientificSearchOptionsV2,
    operator_stop: Any = None,
    boundary_hook: Any = None,
    resume_budget: ScientificResumeBudgetV1 | None = None,
) -> dict[str, Any]:
    return run_sustained_search(
        provider=provider,
        evaluator_factory=lambda: _Evaluator(concurrency),
        workspace=root,
        panel_factory=lambda _generation: _PANEL,
        system_prompt="system",
        specification_prompt="specification",
        specification_ack_schema=specification_ack_schema(),
        policy_schema=_POLICY_SCHEMA,
        options=options,
        provider_turn_timeout_seconds=1,
        resume_budget=resume_budget,
        operator_stop=operator_stop,
        boundary_hook=boundary_hook,
    )


def _scientific_config(
    tmp_path: Path,
    *,
    exp_id: str,
    generations: int = 1,
    repairs: int = 0,
    workers: int = 2,
    timeout_seconds: float = 30,
    wall_seconds: float = 60,
) -> Path:
    heg = tmp_path / "heg"
    (heg / "src" / "sglab").mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / f"{exp_id}.toml"
    config_path.write_text(
        f'''schema_version = "{PYTHON_SCIENTIFIC_SEARCH_CONFIG_SCHEMA_VERSION}"
protocol = "{PYTHON_EXPERIMENT_PROTOCOL_ID}"
exp_id = "{exp_id}"
workspace = "{(tmp_path / "workspaces").as_posix()}"

[python_preview]
model = "fixture-model"
effort = "medium"
timeout_seconds = {timeout_seconds}
heg_repo = "{heg.as_posix()}"

[python_preview.scientific_search]
generation_limit = {generations}
evaluator_workers = {workers}
provider_concurrency = 4
wall_seconds = {wall_seconds}
primary_program_slots = {generations * 8}
repair_turn_limit = {repairs}
provider_total_turn_limit = {generations * 8 + repairs}
validated_queue_target = {workers * 2}
validated_queue_capacity = {workers * 4}
stop_on_verified = true
resume_enabled = true
replace_terminal_slots = false

[python_preview.scientific_search.evaluation]
graph_mode = "unrestricted_min_degree_3"
order_schedule = "adaptive"
min_order = 8
max_order = 8
orders_per_generation = 1
graph_seeds = [101, 103]
policy_seeds = [17, 19]
horizon = 1
witness_cap = 64
baselines = ["random", "structural"]
replay = false
''',
        encoding="utf-8",
    )
    return config_path


def test_native_v2_reference_profile_derives_the_same_adaptive_orders() -> None:
    evaluation = ScientificEvaluationOptionsV1(
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
    native_v2 = evaluation.as_dict()

    for generation in range(4):
        assert evaluation.orders_for_generation(generation) == orders_for_generation(
            native_v2, generation
        )
    panel = evaluation.panel_for_generation(
        generation=0,
        backend=_Backend(),
    )
    assert len(panel) == 5 * 4 * 16
    assert {case.horizon for case in panel} == {32}
    assert {case.graph_mode for case in panel} == {"unrestricted_min_degree_3"}


def test_evaluation_graph_modes_keep_native_v2_order_domains() -> None:
    common = {
        "order_schedule": "adaptive",
        "orders_per_generation": 1,
        "graph_seeds": (401,),
        "policy_seeds": (4001,),
        "horizon": 32,
        "witness_cap": 64,
        "baselines": ("random", "structural"),
        "replay": False,
    }
    with pytest.raises(ValueError, match="even min_order and max_order"):
        ScientificEvaluationOptionsV1(
            graph_mode="cubic_first",
            min_order=21,
            max_order=128,
            **common,
        )
    with pytest.raises(ValueError, match="min_order >= 5"):
        ScientificEvaluationOptionsV1(
            graph_mode="minimal_structure_mixed_degree",
            min_order=4,
            max_order=8,
            **common,
        )


def test_sustained_search_overlaps_evaluations_and_commits_canonically(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    concurrency = _Concurrency()
    report = _run(
        tmp_path,
        provider=provider,
        concurrency=concurrency,
        options=_options(),
    )

    assert report["protocol_id"] == M10_REPORT_PROTOCOL_ID
    assert report["candidate_count"] == 8
    assert report["candidate_program_turns"] == 8
    assert report["provider_order"] == [f"g0000-slot-{index:02d}" for index in range(8)]
    assert concurrency.peak == 2
    assert report["runtime"]["peak_active_evaluators"] == 2
    assert report["acceptance_checks"]["provider_program_turn_budget_respected"] is True
    assert report["acceptance_checks"]["generation_baselines_complete"] is True
    baselines = read_json(
        tmp_path / "generations" / "generation-0000" / search_module.M10_BASELINE_FILENAME
    )
    assert set(cast(Mapping[str, Any], baselines["baselines"])) == {
        "random",
        "structural",
    }


def test_case_scheduler_uses_all_configured_workers(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    release = threading.Event()
    active = 0
    peak = 0

    class ConcurrentCaseEvaluator:
        def _evaluate(
            self,
            *,
            source: str,
            case: DevelopmentCaseV1,
        ) -> Mapping[str, JsonValue]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if peak == 12:
                    release.set()
            try:
                assert release.wait(timeout=3)
                return _result(source=source, case=case)
            finally:
                with lock:
                    active -= 1

        def evaluate(
            self,
            *,
            source: str,
            case: DevelopmentCaseV1,
            candidate_id: str,
        ) -> Mapping[str, JsonValue]:
            del candidate_id
            return self._evaluate(source=source, case=case)

        def evaluate_baseline(
            self,
            *,
            baseline: str,
            case: DevelopmentCaseV1,
            generation: int,
        ) -> Mapping[str, JsonValue]:
            return self._evaluate(source=f"{baseline}:{generation}", case=case)

    report = run_sustained_search(
        provider=_Provider(),
        evaluator_factory=ConcurrentCaseEvaluator,
        workspace=tmp_path,
        panel_factory=lambda _: _PANEL,
        system_prompt="system",
        specification_prompt="specification",
        specification_ack_schema=specification_ack_schema(),
        policy_schema=_POLICY_SCHEMA,
        options=_options(workers=12),
        provider_turn_timeout_seconds=1,
    )

    assert peak == 12
    assert report["runtime"]["peak_active_evaluators"] == 12


def test_candidates_and_both_baselines_use_identical_frozen_cases(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    candidate_cases: dict[str, list[dict[str, JsonValue]]] = {}
    baseline_cases: dict[str, list[dict[str, JsonValue]]] = {}

    class RecordingEvaluator:
        def evaluate(
            self,
            *,
            source: str,
            case: DevelopmentCaseV1,
            candidate_id: str,
        ) -> Mapping[str, JsonValue]:
            with lock:
                candidate_cases.setdefault(candidate_id, []).append(case.as_dict())
            return _result(source=source, case=case)

        def evaluate_baseline(
            self,
            *,
            baseline: str,
            case: DevelopmentCaseV1,
            generation: int,
        ) -> Mapping[str, JsonValue]:
            with lock:
                baseline_cases.setdefault(baseline, []).append(case.as_dict())
            return _result(source=f"{baseline}:{generation}", case=case)

    run_sustained_search(
        provider=_Provider(),
        evaluator_factory=RecordingEvaluator,
        workspace=tmp_path,
        panel_factory=lambda _: _PANEL,
        system_prompt="system",
        specification_prompt="specification",
        specification_ack_schema=specification_ack_schema(),
        policy_schema=_POLICY_SCHEMA,
        options=_options(workers=4),
        provider_turn_timeout_seconds=1,
    )

    expected = sorted((case.as_dict() for case in _PANEL), key=lambda item: str(item["case_id"]))
    assert set(baseline_cases) == {"random", "structural"}
    for cases in (*candidate_cases.values(), *baseline_cases.values()):
        assert sorted(cases, key=lambda item: str(item["case_id"])) == expected


def test_completed_baseline_is_visible_while_candidates_are_blocked(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="baseline-visible-before-candidates",
        workers=4,
    )
    candidate_started = threading.Event()
    release_candidates = threading.Event()

    class BlockingCandidateEvaluator(_Evaluator):
        def evaluate(
            self,
            *,
            source: str,
            case: DevelopmentCaseV1,
            candidate_id: str,
        ) -> Mapping[str, JsonValue]:
            candidate_started.set()
            assert release_candidates.wait(timeout=5)
            return _result(source=source, case=case)

    result: dict[str, Any] = {}

    def run() -> None:
        result.update(
            run_python_preview(
                config_path,
                provider_factory=lambda *_: _Provider(),
                backend_factory=lambda _: _Backend(),
                evaluator_factory=lambda *_: BlockingCandidateEvaluator(
                    _Concurrency(parties=1)
                ),
                provenance_guard=lambda **_: {},
                auth_available=lambda _: True,
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert candidate_started.wait(timeout=3)
    root = load_python_preview_config(config_path).experiment_root
    deadline = time.monotonic() + 3
    status: Mapping[str, Any] = {}
    while time.monotonic() < deadline:
        status = python_preview_status(config_path)
        if all(status["baselines"][name] is not None for name in ("random", "structural")):
            break
        time.sleep(0.01)

    assert status["baselines"] == {"random": 0.5, "structural": 0.5}
    assert status["baseline_details"]["random"]["status"] == "complete"
    assert status["baseline_details"]["structural"]["status"] == "complete"
    assert len(list(root.glob("generations/generation-0000/slot-*/candidate.json.gz"))) < 8
    assert len(
        list(
            root.glob(
                "generations/generation-0000/baselines/*/baseline-result.json.gz"
            )
        )
    ) == 2

    release_candidates.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result["state"] == "completed"


def test_prepared_candidate_waiting_for_worker_is_evaluation_queued(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="prepared-evaluation-queued",
        workers=1,
    )
    baseline_started = threading.Event()
    release_baseline = threading.Event()

    class BlockingBaselineEvaluator(_Evaluator):
        def evaluate_baseline(
            self,
            *,
            baseline: str,
            case: DevelopmentCaseV1,
            generation: int,
        ) -> Mapping[str, JsonValue]:
            baseline_started.set()
            assert release_baseline.wait(timeout=5)
            return _result(source=f"{baseline}:{generation}", case=case)

    thread = threading.Thread(
        target=lambda: run_python_preview(
            config_path,
            provider_factory=lambda *_: _Provider(),
            backend_factory=lambda _: _Backend(),
            evaluator_factory=lambda *_: BlockingBaselineEvaluator(
                _Concurrency(parties=1)
            ),
            provenance_guard=lambda **_: {},
            auth_available=lambda _: True,
        )
    )
    thread.start()
    assert baseline_started.wait(timeout=3)
    deadline = time.monotonic() + 3
    status: Mapping[str, Any] = {}
    while time.monotonic() < deadline:
        status = python_preview_status(config_path)
        if any(slot["state"] == "evaluation_queued" for slot in status["slots"]):
            break
        time.sleep(0.01)

    assert status["evaluators"]["active"] == 1
    assert any(slot["state"] == "evaluation_queued" for slot in status["slots"])
    assert all(slot["state"] != "evaluation_running" for slot in status["slots"])
    assert status["evaluators"]["active_work"][0]["owner"].startswith("baseline:")

    release_baseline.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_partial_baseline_cases_are_durable_and_skipped_on_resume(
    tmp_path: Path,
) -> None:
    options = _options(workers=1)
    first_started_second = threading.Event()
    first_closed = threading.Event()
    first_calls: list[str] = []

    class InterruptibleEvaluator:
        def evaluate(self, **_: Any) -> Mapping[str, JsonValue]:
            raise AssertionError("candidate evaluation is not used")

        def evaluate_baseline(
            self,
            *,
            baseline: str,
            case: DevelopmentCaseV1,
            generation: int,
        ) -> Mapping[str, JsonValue]:
            first_calls.append(case.case_id)
            if len(first_calls) == 2:
                first_started_second.set()
                assert first_closed.wait(timeout=3)
                raise RuntimeError("closed")
            return _result(source=f"{baseline}:{generation}", case=case)

        def close(self) -> None:
            first_closed.set()

    telemetry = search_module._RuntimeTelemetry(tmp_path, options)
    pool = search_module._ConcurrentEvaluatorPool(
        workers=1,
        queue_capacity=4,
        evaluator_factory=InterruptibleEvaluator,
        telemetry=telemetry,
    )
    pool.submit_baseline(
        baseline="random",
        panel=_PANEL,
        generation=0,
        generation_dir=tmp_path,
    )
    assert first_started_second.wait(timeout=3)
    pool.close(force=True)

    durable = list(
        tmp_path.glob("baselines/random/evaluations/*.json.gz")
    )
    assert len(durable) == 1
    durable_bytes = durable[0].read_bytes()

    resumed_calls: list[str] = []

    class ResumedEvaluator:
        def evaluate(self, **_: Any) -> Mapping[str, JsonValue]:
            raise AssertionError("candidate evaluation is not used")

        def evaluate_baseline(
            self,
            *,
            baseline: str,
            case: DevelopmentCaseV1,
            generation: int,
        ) -> Mapping[str, JsonValue]:
            resumed_calls.append(case.case_id)
            return _result(source=f"{baseline}:{generation}", case=case)

        def close(self) -> None:
            pass

    resumed_telemetry = search_module._RuntimeTelemetry(tmp_path, options)
    resumed_pool = search_module._ConcurrentEvaluatorPool(
        workers=1,
        queue_capacity=4,
        evaluator_factory=ResumedEvaluator,
        telemetry=resumed_telemetry,
    )
    outcome = resumed_pool.submit_baseline(
        baseline="random",
        panel=_PANEL,
        generation=0,
        generation_dir=tmp_path,
    ).result(timeout=3)
    resumed_pool.close()

    assert outcome.failure_type is None
    assert len(resumed_calls) == 1
    assert durable[0].read_bytes() == durable_bytes
    assert (
        tmp_path / "baselines/random" / search_module.M10_BASELINE_RESULT_FILENAME
    ).is_file()


def test_provider_supply_reaches_four_and_uses_one_frozen_snapshot(
    tmp_path: Path,
) -> None:
    class ConcurrentProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.barrier = threading.Barrier(4, timeout=2)
            self.lock = threading.Lock()
            self.active = 0
            self.peak = 0

        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                self.barrier.wait()
                if kwargs["slot"] == "slot-00":
                    time.sleep(0.02)
                return super().generate_root(**kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    provider = ConcurrentProvider()
    report = _run(
        tmp_path,
        provider=provider,
        concurrency=_Concurrency(parties=1),
        options=_options(workers=2),
    )

    assert provider.peak == 4
    assert report["runtime"]["peak_active_provider_turns"] == 4
    assert report["provider_order"] == [f"g0000-slot-{index:02d}" for index in range(8)]
    assert len(provider.snapshots) == 1
    snapshot = provider.snapshots[0]
    assert snapshot["generation"] == 0
    assert len(snapshot["slots"]) == 8
    assert len(set(provider.prompts.values())) == 1
    assert "active_parent=null" in next(iter(provider.prompts.values()))
    retained = read_json(
        tmp_path / "generations" / "generation-0000" / "generation-snapshot.json.gz"
    )
    assert retained == snapshot


def test_repair_budget_is_separate_and_allocated_in_canonical_slot_order(
    tmp_path: Path,
) -> None:
    class RepairProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.repair_slots: list[str] = []
            self.turn_order: list[str] = []
            self.slot_zero_repaired = threading.Event()

        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            if kwargs["slot"] == "slot-04":
                assert self.slot_zero_repaired.wait(timeout=1)
            self.turn_order.append(f"primary-{kwargs['slot']}")
            result = super().generate_root(**kwargs)
            slot = str(kwargs["slot"])
            if slot not in {"slot-00", "slot-01"}:
                return result
            invalid = M5ProviderResultV1(
                response_text=json.dumps(
                    {
                        "schema_version": ("mforge.native.python_policy_response.v1"),
                        "source": "import os\n",
                    },
                    separators=(",", ":"),
                ),
                context=result.context,
                usage=result.usage,
                duration_ms=result.duration_ms,
                warnings=result.warnings,
            )
            key = str(kwargs["idempotency_key"])
            self.durable[key] = invalid
            write_json(
                Path(kwargs["artifact_dir"]) / "m5-provider-result.json.gz",
                invalid.as_dict(),
            )
            return invalid

        def repair(
            self,
            *,
            previous: M5ProviderResultV1,
            generation: int,
            slot: str,
            idempotency_key: str,
            artifact_dir: Path,
            **_: Any,
        ) -> M5ProviderResultV1:
            self.repair_slots.append(slot)
            self.turn_order.append(f"repair-{slot}")
            if slot == "slot-00":
                self.slot_zero_repaired.set()
            turn = f"repair-{generation}-{slot}"
            result = M5ProviderResultV1(
                response_text=json.dumps(
                    {
                        "schema_version": ("mforge.native.python_policy_response.v1"),
                        "source": _source(f"repair-{slot}"),
                    },
                    separators=(",", ":"),
                ),
                context=M5ProviderContextV1(
                    previous.context.thread_id,
                    turn,
                    None,
                    previous.context.included_turn_ids + (turn,),
                ),
                usage=_usage(),
                duration_ms=1,
                warnings=0,
            )
            self.durable[idempotency_key] = result
            write_json(
                artifact_dir / "m5-provider-result.json.gz",
                result.as_dict(),
            )
            return result

    provider = RepairProvider()
    report = _run(
        tmp_path,
        provider=provider,
        concurrency=_Concurrency(parties=1),
        options=_options(workers=2, repairs=1),
    )

    assert provider.repair_slots == ["slot-00"]
    assert provider.turn_order.index("repair-slot-00") < (
        provider.turn_order.index("primary-slot-04")
    )
    assert report["candidate_status_counts"] == {
        "contract_invalid": 1,
        "evaluated": 7,
    }
    assert report["repaired_valid_count"] == 1
    assert report["provider_accounting"]["primary_turns_submitted"] == 8
    assert report["provider_accounting"]["repair_turns_submitted"] == 1
    assert report["candidate_program_turns"] == 9


def test_all_later_generations_keep_four_children_and_four_roots(
    tmp_path: Path,
) -> None:
    report = _run(
        tmp_path,
        provider=_Provider(),
        concurrency=_Concurrency(parties=1),
        options=_options(workers=2, generations=3),
    )

    assert report["candidate_count"] == 24
    assert report["generation_allocations"] == {
        "0": {"children": 0, "roots": 8},
        "1": {"children": 4, "roots": 4},
        "2": {"children": 4, "roots": 4},
    }
    assert report["acceptance_checks"]["later_generations_four_children_four_roots"] is True


def test_operator_stop_resume_repeats_no_terminal_work(
    tmp_path: Path,
) -> None:
    durable: dict[str, M5ProviderResultV1] = {}
    first_provider = _Provider(durable)
    first_concurrency = _Concurrency(parties=1)

    with pytest.raises(M5OperatorStop):
        _run(
            tmp_path,
            provider=first_provider,
            concurrency=first_concurrency,
            options=_options(generations=2),
            operator_stop=lambda: len(first_provider.calls) >= 8,
        )

    retained = {
        path: path.read_bytes()
        for path in sorted(tmp_path.glob("generations/generation-*/slot-*/candidate.json.gz"))
    }
    assert len(retained) == 8
    resumed_provider = _Provider(durable)
    resumed_concurrency = _Concurrency(parties=1)
    report = _run(
        tmp_path,
        provider=resumed_provider,
        concurrency=resumed_concurrency,
        options=_options(generations=2),
    )

    assert report["candidate_count"] == 16
    assert sorted(resumed_provider.calls) == [(1, f"slot-{index:02d}") for index in range(8)]
    assert all(path.read_bytes() == content for path, content in retained.items())
    assert len(first_concurrency.calls) + len(resumed_concurrency.calls) == 32


def test_crash_after_prepared_boundary_repeats_no_provider_turn(
    tmp_path: Path,
) -> None:
    class Crash(BaseException):
        pass

    durable: dict[str, M5ProviderResultV1] = {}
    first_provider = _Provider(durable)

    def crash(boundary: str) -> None:
        if boundary == "g0000-slot-00_evaluation_queued":
            raise Crash

    with pytest.raises(Crash):
        _run(
            tmp_path,
            provider=first_provider,
            concurrency=_Concurrency(parties=1),
            options=_options(),
            boundary_hook=crash,
        )

    assert sorted(first_provider.calls) == [(0, f"slot-{index:02d}") for index in range(4)]
    resumed = _Provider(durable)
    report = _run(
        tmp_path,
        provider=resumed,
        concurrency=_Concurrency(parties=1),
        options=_options(),
    )
    assert report["candidate_count"] == 8
    assert resumed.calls == [(0, f"slot-{index:02d}") for index in range(4, 8)]
    assert len(durable) == 8


def test_interrupted_provider_turn_is_consumed_without_external_repeat(
    tmp_path: Path,
) -> None:
    class Crash(BaseException):
        pass

    class CrashingProvider(_Provider):
        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            del kwargs
            raise Crash

    with pytest.raises(Crash):
        _run(
            tmp_path,
            provider=CrashingProvider(),
            concurrency=_Concurrency(parties=1),
            options=_options(),
        )

    resumed = _Provider()
    with pytest.raises(
        M5InfrastructureError,
        match="will not repeat",
    ):
        _run(
            tmp_path,
            provider=resumed,
            concurrency=_Concurrency(parties=1),
            options=_options(),
        )
    assert resumed.calls == []


def test_current_generation_resume_retries_pending_and_caps_new_repairs(
    tmp_path: Path,
) -> None:
    durable: dict[str, M5ProviderResultV1] = {}
    options = _options(generations=4, repairs=24)
    initial_provider = _Provider(durable)
    with pytest.raises(M5OperatorStop):
        _run(
            tmp_path,
            provider=initial_provider,
            concurrency=_Concurrency(parties=1),
            options=options,
            operator_stop=lambda: len(initial_provider.calls) >= 24,
        )

    manifest = cast(
        Mapping[str, Any],
        read_json(tmp_path / "generations" / "generation-0002" / "manifest.json.gz"),
    )
    slots = cast(list[Mapping[str, Any]], manifest["slots"])
    pending_slots = {f"slot-{index:02d}" for index in range(1, 8)}
    request_keys: dict[str, str] = {}
    for slot in slots:
        slot_name = str(slot["slot"])
        if slot_name not in pending_slots:
            continue
        request_key = str(slot["request_key"])
        request_keys[slot_name] = request_key
        slot_dir = tmp_path / "generations" / "generation-0002" / slot_name
        (slot_dir / "candidate.json.gz").unlink()
        (slot_dir / "prepared-candidate.json.gz").unlink()
        shutil.rmtree(slot_dir / "provider-initial")
        durable.pop(request_key)

    runtime_path = tmp_path / "m10-runtime.json.gz"
    runtime = cast(dict[str, Any], read_json(runtime_path))
    unstarted = {f"{request_keys[slot]}-initial" for slot in ("slot-03", "slot-04", "slot-05")}
    started = [
        str(key)
        for key in cast(list[object], runtime["provider_started_keys"])
        if str(key) not in unstarted
    ]
    started.append("historical-repair")
    runtime.update(
        {
            "provider_started_keys": started,
            "provider_turns_submitted": 22,
            "primary_turns_submitted": 21,
            "repair_turn_keys": ["historical-repair"],
            "repair_turns_submitted": 1,
            "active_provider_turns": 0,
        }
    )
    write_json(runtime_path, runtime)

    class ResumeProvider(_Provider):
        def __init__(self) -> None:
            super().__init__(durable)
            self.repair_slots: list[str] = []

        def _program(self, **kwargs: Any) -> M5ProviderResultV1:
            result = super()._program(**kwargs)
            if kwargs["slot"] not in {
                "slot-01",
                "slot-02",
                "slot-03",
            }:
                return result
            invalid = M5ProviderResultV1(
                response_text=json.dumps(
                    {
                        "schema_version": ("mforge.native.python_policy_response.v1"),
                        "source": "import os\n",
                    },
                    separators=(",", ":"),
                ),
                context=result.context,
                usage=result.usage,
                duration_ms=result.duration_ms,
                warnings=result.warnings,
            )
            key = str(kwargs["idempotency_key"])
            self.durable[key] = invalid
            write_json(
                Path(kwargs["artifact_dir"]) / "m5-provider-result.json.gz",
                invalid.as_dict(),
            )
            return invalid

        def repair(
            self,
            *,
            previous: M5ProviderResultV1,
            generation: int,
            slot: str,
            idempotency_key: str,
            artifact_dir: Path,
            **_: Any,
        ) -> M5ProviderResultV1:
            self.repair_slots.append(slot)
            turn = f"repair-{generation}-{slot}"
            result = M5ProviderResultV1(
                response_text=json.dumps(
                    {
                        "schema_version": ("mforge.native.python_policy_response.v1"),
                        "source": _source(f"repair-{slot}"),
                    },
                    separators=(",", ":"),
                ),
                context=M5ProviderContextV1(
                    previous.context.thread_id,
                    turn,
                    None,
                    previous.context.included_turn_ids + (turn,),
                ),
                usage=_usage(),
                duration_ms=1,
                warnings=0,
            )
            self.durable[idempotency_key] = result
            write_json(
                artifact_dir / "m5-provider-result.json.gz",
                result.as_dict(),
            )
            return result

    resumed = ResumeProvider()
    report = _run(
        tmp_path,
        provider=resumed,
        concurrency=_Concurrency(parties=1),
        options=options,
        resume_budget=ScientificResumeBudgetV1(
            expected_pending_primary_slots=7,
            max_new_repair_turns=2,
        ),
    )

    assert sorted(resumed.calls) == [(2, f"slot-{index:02d}") for index in range(1, 8)]
    assert resumed.repair_slots == ["slot-01", "slot-02"]
    assert report["stop_reason"] == "resume_generation_complete"
    assert report["candidate_count"] == 24
    assert report["pending_candidate_count"] == 0
    assert not (tmp_path / "generations" / "generation-0003" / "manifest.json.gz").exists()
    final_runtime = cast(dict[str, Any], read_json(runtime_path))
    assert final_runtime["primary_turns_submitted"] == 28
    assert final_runtime["repair_turns_submitted"] == 3
    assert final_runtime["provider_turns_submitted"] == 31
    assert (
        len(
            cast(
                Mapping[str, object],
                final_runtime["interrupted_primary_retries"],
            )
        )
        == 4
    )


def test_resume_budget_remains_cumulative_across_process_restarts(
    tmp_path: Path,
) -> None:
    options = _options(generations=4, repairs=24)
    primary_keys = [f"primary-{index}" for index in range(8)]
    initial = search_module._RuntimeTelemetry(tmp_path, options)
    initial.reserve_primary_generation(
        primary_keys,
        limit=options.primary_program_slots,
    )
    assert initial.provider_started(primary_keys[0], kind="primary")
    initial.provider_finished(0.0, key=primary_keys[0], failed=True)

    budget = ScientificResumeBudgetV1(
        expected_pending_primary_slots=7,
        max_new_repair_turns=2,
    )
    first_resume = search_module._RuntimeTelemetry(
        tmp_path,
        options,
        budget,
    )
    retry_key, attempt = first_resume.admit_primary_retry(
        primary_keys[0],
        limit=options.primary_program_slots,
        durable_result_exists=lambda _: False,
    )
    assert (retry_key, attempt) == ("primary-0-resume-01", 1)
    assert first_resume.provider_started(retry_key, kind="primary")
    first_resume.provider_finished(0.0, key=retry_key, failed=False)

    restarted = search_module._RuntimeTelemetry(tmp_path, options, budget)
    assert restarted.admit_primary_retry(
        primary_keys[0],
        limit=options.primary_program_slots,
        durable_result_exists=lambda retry_attempt: retry_attempt == 1,
    ) == (retry_key, 1)
    assert not restarted.provider_started(retry_key, kind="primary")
    for key in primary_keys[1:7]:
        assert restarted.provider_started(key, kind="primary")
        restarted.provider_finished(0.0, key=key, failed=False)
    with pytest.raises(
        search_module._ProviderTurnBudgetExhausted,
        match="resume primary turn budget is exhausted",
    ):
        restarted.provider_started(primary_keys[7], kind="primary")

    for repair_key in ("repair-a", "repair-b"):
        assert restarted.reserve_repair(
            repair_key,
            limit=options.repair_turn_limit,
        )
        assert restarted.provider_started(repair_key, kind="repair")
        restarted.provider_finished(0.0, key=repair_key, failed=False)
    assert not restarted.reserve_repair(
        "repair-c",
        limit=options.repair_turn_limit,
    )
    restarted_again = search_module._RuntimeTelemetry(
        tmp_path,
        options,
        budget,
    )
    assert not restarted_again.reserve_repair(
        "repair-c",
        limit=options.repair_turn_limit,
    )


def test_one_provider_failure_consumes_its_slot_and_search_continues(
    tmp_path: Path,
) -> None:
    class FailFirstProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            if not self.failed:
                self.failed = True
                raise RuntimeError("bounded provider failure")
            return super().generate_root(**kwargs)

    provider = FailFirstProvider()
    report = _run(
        tmp_path,
        provider=provider,
        concurrency=_Concurrency(parties=1),
        options=_options(),
    )
    assert report["candidate_count"] == 8
    assert report["candidate_status_counts"] == {
        "evaluated": 7,
        "provider_failed": 1,
    }
    assert report["provider_turns"] == 9
    assert report["candidate_program_turns"] == 8


def test_explicit_scientific_profile_routes_status_with_live_metrics(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="scientific-fixture",
    )
    loaded = load_python_preview_config(config_path)
    assert loaded.scientific_search == _options()
    assert isinstance(loaded.scientific_search.wall_seconds, float)
    concurrency = _Concurrency()
    provider = _Provider()

    status = run_python_preview(
        config_path,
        provider_factory=lambda *_: provider,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(concurrency),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert status["state"] == "completed"
    assert status["search_protocol"] == M10_SEARCH_PROTOCOL_ID
    assert status["safe_api_expanded"] is True
    assert status["counts"]["terminal"] == 8
    assert status["counts"]["pending"] == 0
    assert status["counts"]["program_failed"] == 0
    assert status["evaluators"]["configured"] == 2
    assert status["evaluators"]["peak_active"] == 2
    assert status["provider"]["turns"] == 9
    assert status["provider"]["candidate_turns"] == 8
    assert status["provider"]["program_turns_reserved"] == 8
    assert status["throughput"]["policy_invocations_per_second"] > 0
    assert status["phase_timings"]["dominant_bottleneck"] in {
        "provider",
        "evaluator/scorer",
        "persistence",
        "balanced",
    }
    assert status["exact_verification"]["authority"] == "exact_verifier_only"
    assert status["evaluation_workload"] == {
        "generation": 0,
        "panel_hash": status["equal_development_panel"],
        "case_count": 4,
        "orders": [8],
        "graph_mode": "unrestricted_min_degree_3",
        "order_schedule": "adaptive",
        "graph_seed_count": 2,
        "policy_seed_count": 2,
        "horizon": 1,
        "witness_cap": 64,
        "baselines": ["random", "structural"],
        "replay": False,
    }
    for baseline in ("random", "structural"):
        assert status["baselines"][baseline] == 0.5
        assert status["baseline_details"][baseline]["status"] == "complete"
        assert status["baseline_details"][baseline]["fitness_interval"] == {
            "lower": {"numerator": 1, "denominator": 2},
            "upper": {"numerator": 1, "denominator": 2},
        }


def test_fresh_dashboard_keeps_unknown_baselines_and_tokens_unknown(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="fresh-unknown-values",
    )
    status = python_preview_status(config_path)
    config = load_python_preview_config(config_path)
    assert config.scientific_search is not None
    rich = dashboard_state_from_python_status(
        status,
        run_id=config.exp_id,
        model=config.model,
        effort=config.effort,
        generation_limit=config.scientific_search.generation_limit,
        wall_seconds=config.scientific_search.wall_seconds,
    )

    assert status["baselines"] == {
        "random": None,
        "structural": None,
    }
    assert rich.baseline_random is None
    assert rich.baseline_structural is None
    assert rich.cumulative_usage.total is None
    assert rich.cumulative_usage.quality == "unknown"

    zero_status = {**status, "baselines": {"random": 0, "structural": 0}}
    zero_rich = dashboard_state_from_python_status(
        zero_status,
        run_id=config.exp_id,
        model=config.model,
        effort=config.effort,
        generation_limit=config.scientific_search.generation_limit,
        wall_seconds=config.scientific_search.wall_seconds,
    )
    assert zero_rich.baseline_random == 0
    assert zero_rich.baseline_structural == 0
    assert _objective(zero_rich.baseline_random) == "0"


def test_same_exp_id_rejects_changed_frozen_scientific_config_before_work(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(tmp_path, exp_id="frozen-config")
    run_python_preview(
        config_path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("horizon = 1", "horizon = 2"),
        encoding="utf-8",
    )
    calls = {"provider": 0, "backend": 0}

    with pytest.raises(
        preview_module.PythonPreviewWorkspaceError,
        match="scientific configuration differs",
    ):
        run_python_preview(
            config_path,
            provider_factory=lambda *_: calls.__setitem__(
                "provider", calls["provider"] + 1
            ),
            backend_factory=lambda _: calls.__setitem__(
                "backend", calls["backend"] + 1
            ),
            provenance_guard=lambda **_: {},
            auth_available=lambda _: True,
        )

    assert calls == {"provider": 0, "backend": 0}


def test_new_exp_id_accepts_changed_scientific_config(
    tmp_path: Path,
) -> None:
    first_path = _scientific_config(tmp_path, exp_id="config-a")
    second_path = _scientific_config(tmp_path, exp_id="config-b")
    second_path.write_text(
        second_path.read_text(encoding="utf-8").replace("horizon = 1", "horizon = 2"),
        encoding="utf-8",
    )

    for path in (first_path, second_path):
        status = run_python_preview(
            path,
            provider_factory=lambda *_: _Provider(),
            backend_factory=lambda _: _Backend(),
            evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
            provenance_guard=lambda **_: {},
            auth_available=lambda _: True,
        )
        assert status["state"] == "completed"

    assert load_python_preview_config(first_path).scientific_config_sha256 != (
        load_python_preview_config(second_path).scientific_config_sha256
    )


def test_wall_budget_is_per_invocation_and_not_frozen(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="per-invocation-wall",
        generations=2,
        timeout_seconds=0.2,
        wall_seconds=0.3,
    )

    class SlowProvider(_Provider):
        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            time.sleep(0.12)
            return super().generate_root(**kwargs)

    first = run_python_preview(
        config_path,
        provider_factory=lambda *_: SlowProvider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )
    assert first["terminal_reason"] == "wall_clock_budget"
    assert first["resumable"] is True

    resumed = run_python_preview(
        config_path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert resumed["state"] == "completed"
    assert resumed["counts"]["terminal"] == 16
    assert resumed["throughput"]["current_run_elapsed_seconds"] < 0.3


def test_invocation_controls_do_not_change_scientific_config_identity(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="invocation-controls",
        workers=2,
        timeout_seconds=1,
        wall_seconds=2,
    )
    original = load_python_preview_config(config_path).scientific_config_sha256
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("evaluator_workers = 2", "evaluator_workers = 12")
        .replace("validated_queue_target = 4", "validated_queue_target = 24")
        .replace("validated_queue_capacity = 8", "validated_queue_capacity = 48")
        .replace("provider_concurrency = 2", "provider_concurrency = 4")
        .replace("wall_seconds = 2", "wall_seconds = 60")
        .replace("timeout_seconds = 1", "timeout_seconds = 300"),
        encoding="utf-8",
    )

    assert load_python_preview_config(config_path).scientific_config_sha256 == original


def test_rich_projection_uses_the_same_canonical_status_counters(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="rich-canonical-status",
    )
    status = run_python_preview(
        config_path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )
    config = load_python_preview_config(config_path)
    assert config.scientific_search is not None
    rich = dashboard_state_from_python_status(
        status,
        run_id=config.exp_id,
        model=config.model,
        effort=config.effort,
        generation_limit=config.scientific_search.generation_limit,
        wall_seconds=config.scientific_search.wall_seconds,
    )

    assert rich.completed_slots == status["counts"]["terminal"]
    assert rich.provider_turns_attempted == status["provider"]["program_turns_reserved"]
    assert rich.active_provider_turns == status["provider"]["active"]
    assert rich.configured_provider_concurrency == status["provider"]["configured_concurrency"]
    assert rich.evaluations_completed == status["evaluators"]["completed"]
    assert rich.evaluation_workers_active == status["evaluators"]["active"]
    assert sum(len(group.slots) for group in rich.generations) == status["counts"]["planned"]
    assert rich.baseline_random == 0.5
    assert rich.baseline_structural == 0.5
    assert rich.evaluation_orders == (8,)
    assert rich.evaluation_case_count == 4
    assert rich.evaluation_horizon == 1


def test_budget_pause_record_projects_read_only_rich_and_json_status(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="paused-budget-status",
    )
    status = run_python_preview(
        config_path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )
    config = load_python_preview_config(config_path)
    workspace = config.experiment_root
    before = {
        path.relative_to(workspace): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    pause_record = tmp_path / "paused-for-budget.json"
    pause_record.write_text(
        json.dumps(
            {
                "schema_version": ("mforge.native.python_m10_emergency_stop_evidence.v1"),
                "state": "PAUSED_FOR_BUDGET",
                "experiment": config.exp_id,
                "slots": {
                    "terminal_total": 8,
                    "in_flight_cancelled_at_stop": 0,
                    "pending_total": 0,
                    "in_flight_slots": [],
                    "pending_unstarted_slots": [],
                },
                "provider_turns": {
                    "started_reservations": 8,
                    "completed_turns": 8,
                    "in_flight_started_without_finished": 0,
                    "primary_turns_submitted": 8,
                    "repair_turns_submitted": 0,
                    "persisted_usage_including_specification_anchor": {
                        "input_tokens": 100,
                        "cached_input_tokens": 10,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 3,
                        "total_tokens": 120,
                    },
                },
                "best": {
                    "candidate_id": "g0000-slot-00",
                    "program_hash": "b" * 64,
                    "fitness_interval": {
                        "lower": {"numerator": 2, "denominator": 3},
                        "upper": {"numerator": 2, "denominator": 3},
                    },
                },
                "exact_verifier": {
                    "candidate_submissions": 0,
                    "candidate_results": 0,
                    "all_candidate_exact_verified": False,
                },
            }
        ),
        encoding="utf-8",
    )

    paused = python_preview_status(
        config_path,
        pause_record_path=pause_record,
    )
    after = {
        path.relative_to(workspace): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in workspace.rglob("*")
        if path.is_file()
    }

    assert status["counts"]["terminal"] == 8
    assert paused["state"] == "PAUSED_FOR_BUDGET"
    assert paused["resumable"] is True
    assert paused["run_terminal"] is False
    assert paused["counts"]["terminal"] == 8
    assert paused["counts"]["pending"] == 0
    assert paused["provider"]["active"] == 0
    assert paused["provider"]["completed_turns"] == 8
    assert paused["provider"]["usage"]["totalTokens"] == 120
    assert paused["exact_verification"]["verified"] is False
    assert before == after

    assert config.scientific_search is not None
    rich = dashboard_state_from_python_status(
        paused,
        run_id=config.exp_id,
        model=config.model,
        effort=config.effort,
        generation_limit=config.scientific_search.generation_limit,
        wall_seconds=config.scientific_search.wall_seconds,
    )
    assert rich.experiment_state == "PAUSED_FOR_BUDGET"
    assert rich.paused is False
    assert rich.completed_slots == paused["counts"]["terminal"]
    assert rich.provider_turns_attempted == 8
    assert rich.provider_turns_completed == 8
    assert rich.cumulative_usage.total == 120
    assert rich.best_candidate == "g0000-slot-00"
    assert rich.best_program_hash == "b" * 64
    assert rich.best_fitness == "2/3"
    assert rich.best_objective == 2 / 3


def test_status_counts_failed_provider_reservations(
    tmp_path: Path,
) -> None:
    class FailFirstProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            if not self.failed:
                self.failed = True
                raise RuntimeError("bounded provider failure")
            return super().generate_root(**kwargs)

    status = run_python_preview(
        _scientific_config(tmp_path, exp_id="failed-provider-status"),
        provider_factory=lambda *_: FailFirstProvider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert status["counts"]["provider_failed"] == 1
    assert status["provider"]["program_turns_reserved"] == 8
    assert status["provider"]["candidate_turns"] == 8
    assert status["provider"]["turns"] == 9


def test_evaluator_factory_failure_closes_every_owned_backend(
    tmp_path: Path,
) -> None:
    class TrackingBackend(_Backend):
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    backends: list[TrackingBackend] = []

    def backend_factory(_: object) -> TrackingBackend:
        backend = TrackingBackend()
        backends.append(backend)
        return backend

    def fail_evaluator(*_: object) -> _Evaluator:
        raise RuntimeError("evaluator construction failed")

    status = run_python_preview(
        _scientific_config(tmp_path, exp_id="evaluator-factory-failure"),
        provider_factory=lambda *_: _Provider(),
        backend_factory=backend_factory,
        evaluator_factory=fail_evaluator,
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert status["state"] == "blocked"
    assert len(backends) >= 2
    assert all(backend.close_calls == 1 for backend in backends)


def test_terminal_m10_report_wins_after_state_write_interruption(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="report-state-interruption",
    )
    run_python_preview(
        config_path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )
    root = load_python_preview_config(config_path).experiment_root
    state_path = root / "python-preview-state.json.gz"
    retained_state = read_json(state_path)
    retained_state.update(
        {
            "state": "running",
            "resumable": True,
            "run_terminal": False,
            "terminal_reason": None,
        }
    )
    write_json(state_path, retained_state)
    write_json(
        root / M10_STOP_FILENAME,
        {
            "protocol_id": M10_REPORT_PROTOCOL_ID,
            "status": "operator_stop",
            "resumable": True,
        },
    )

    status = python_preview_status(config_path)
    assert status["state"] == "completed"
    assert status["run_terminal"] is True
    assert status["resumable"] is False


def test_sustained_provider_transport_is_bounded_for_one_hundred_twenty_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Capsule:
        root = tmp_path / "capsule"

    captured: dict[str, Any] = {}

    class Adapter:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        provider_module.IsolatedCapsule,
        "create",
        lambda *_args, **_kwargs: Capsule(),
    )
    monkeypatch.setattr(
        provider_module,
        "secure_capsule_parent",
        lambda: tmp_path,
    )
    monkeypatch.setattr(provider_module, "CodexAppServerAdapter", Adapter)

    CodexM5SearchProvider(
        workspace=tmp_path / "provider",
        model="fixture",
        effort="medium",
        base_instructions="system",
        program_turn_limit=120,
    )

    limits = captured["limits"]
    assert limits.max_turns == 121
    assert limits.max_campaigns == 121
    assert limits.max_events == M10_PROVIDER_MAX_EVENTS
    assert limits.stdout_bytes == M10_PROVIDER_STDOUT_BYTES
    assert limits.transcript_bytes == M10_PROVIDER_TRANSCRIPT_BYTES

    captured.clear()
    CodexM5SearchProvider(
        workspace=tmp_path / "m5-provider",
        model="fixture",
        effort="medium",
        base_instructions="system",
    )
    m5_limits = captured["limits"]
    assert m5_limits.max_turns is None
    assert m5_limits.max_campaigns is None
    assert m5_limits.max_events == 10_000


def test_m10_provider_releases_specification_process_before_worker_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    providers: list[object] = []

    class FakeProvider:
        capsule = object()

        def __init__(self, *, coordinator: bool) -> None:
            self.coordinator = coordinator

        def fork_root_worker_from_active_anchor(
            self,
            *,
            anchor: M5ProviderContextV1,
            worker: int,
            artifact_dir: Path,
        ) -> M5ProviderContextV1:
            assert self.coordinator
            events.append(f"coordinator-fork-{worker}")
            return M5ProviderContextV1(
                f"worker-{worker}",
                anchor.turn_id,
                None,
                anchor.included_turn_ids,
            )

        def ensure_anchor_context(self, context: M5ProviderContextV1) -> None:
            assert not self.coordinator
            assert "coordinator-close-False" in events
            events.append(f"worker-{context.thread_id.removeprefix('worker-')}-resume")

        def _increment_telemetry(self, _: str, amount: int = 1) -> None:
            assert amount == 1

        def close(self, *, cleanup_capsule: bool = True) -> None:
            kind = "coordinator" if self.coordinator else "worker"
            events.append(f"{kind}-close-{cleanup_capsule}")

    def provider_factory(**_: Any) -> FakeProvider:
        provider = FakeProvider(coordinator=not providers)
        providers.append(provider)
        return provider

    monkeypatch.setattr(provider_module, "CodexM5SearchProvider", provider_factory)
    monkeypatch.setattr(provider_module, "CodexAppServerAdapter", lambda **_: object())
    provider = CodexM10SearchProvider(
        workspace=tmp_path / "provider",
        model="fixture",
        effort="medium",
        base_instructions="system",
        auth_json=tmp_path / "auth.json",
        turn_timeout_seconds=60,
        provider_concurrency=4,
        provider_total_turn_limit=16,
    )
    anchor = M5ProviderContextV1("anchor-thread", "anchor-turn", None, ("anchor-turn",))
    provider._anchor = anchor
    provider.prepare_generation(
        snapshot={
            "generation": 0,
            "slots": [{"slot": f"slot-{slot:02d}", "kind": "root"} for slot in range(8)],
        },
        anchor=anchor,
        artifact_dir=tmp_path / "generation-provider",
    )
    assert events[:9] == [
        "coordinator-fork-0",
        "coordinator-fork-1",
        "coordinator-fork-2",
        "coordinator-fork-3",
        "coordinator-close-False",
        "worker-0-resume",
        "worker-1-resume",
        "worker-2-resume",
        "worker-3-resume",
    ]
    provider.close(cleanup_capsule=False)
    assert events[-1] == "coordinator-close-False"


def test_m10_provider_retries_only_transient_worker_resume_setup_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: dict[str, int] = {}
    restarts: list[str] = []
    providers: list[object] = []

    class FakeProvider:
        capsule = object()

        def __init__(self, *, workspace: Path, coordinator: bool) -> None:
            self.workspace = workspace
            self.coordinator = coordinator

        def fork_root_worker_from_active_anchor(
            self,
            *,
            anchor: M5ProviderContextV1,
            worker: int,
            artifact_dir: Path,
        ) -> M5ProviderContextV1:
            assert self.coordinator
            return M5ProviderContextV1(
                f"thread-{worker}",
                anchor.turn_id,
                None,
                anchor.included_turn_ids,
            )

        def ensure_anchor_context(self, context: M5ProviderContextV1) -> None:
            assert not self.coordinator
            key = self.workspace.name
            attempts[key] = attempts.get(key, 0) + 1
            if key == "worker-00" and attempts[key] == 1:
                raise provider_module.ProtocolError("request thread/resume failed")

        def _increment_telemetry(self, field: str, amount: int = 1) -> None:
            assert amount == 1
            restarts.append(field)

        def close(self, *, cleanup_capsule: bool = True) -> None:
            return None

    def provider_factory(**kwargs: Any) -> FakeProvider:
        provider = FakeProvider(
            workspace=Path(kwargs["workspace"]),
            coordinator=not providers,
        )
        providers.append(provider)
        return provider

    monkeypatch.setattr(provider_module, "CodexM5SearchProvider", provider_factory)
    monkeypatch.setattr(provider_module, "CodexAppServerAdapter", lambda **_: object())
    provider = CodexM10SearchProvider(
        workspace=tmp_path / "provider",
        model="fixture",
        effort="medium",
        base_instructions="system",
        auth_json=tmp_path / "auth.json",
        turn_timeout_seconds=60,
        provider_concurrency=4,
        provider_total_turn_limit=16,
    )
    anchor = M5ProviderContextV1("anchor-thread", "anchor-turn", None, ("anchor-turn",))
    provider._anchor = anchor
    provider.prepare_generation(
        snapshot={
            "generation": 0,
            "slots": [{"slot": f"slot-{slot:02d}", "kind": "root"} for slot in range(8)],
        },
        anchor=anchor,
        artifact_dir=tmp_path / "generation-provider",
    )
    assert attempts["worker-00"] == 2
    assert restarts == ["process_restarts"]


def test_m10_root_repair_advances_the_persistent_lane_context(
    tmp_path: Path,
) -> None:
    previous_context = M5ProviderContextV1(
        "root-thread",
        "primary-turn",
        None,
        ("anchor-turn", "primary-turn"),
    )
    repaired_context = M5ProviderContextV1(
        "root-thread",
        "repair-turn",
        None,
        ("anchor-turn", "primary-turn", "repair-turn"),
    )
    previous = M5ProviderResultV1(
        response_text="{}",
        context=previous_context,
        usage=_usage(),
        duration_ms=1,
        warnings=0,
    )
    repaired = M5ProviderResultV1(
        response_text="{}",
        context=repaired_context,
        usage=_usage(),
        duration_ms=1,
        warnings=0,
    )

    class Worker:
        def repair(self, **_: Any) -> M5ProviderResultV1:
            return repaired

    provider = object.__new__(CodexM10SearchProvider)
    provider.workspace = tmp_path
    provider.model = "fixture"
    provider.effort = "medium"
    provider.provider_concurrency = 1
    provider._state_path = tmp_path / "provider-pool-state.json.gz"
    provider._state_lock = threading.RLock()
    provider._root_workers = {0: previous_context}
    provider._thread_owners = {"root-thread": 0}
    provider._primary_slot_owners = {}
    provider._completed_primary_slots = set()
    provider._released_primary_slots = set()
    provider._anchor = None
    provider._worker_locks = [threading.Lock()]
    provider._workers = [cast(Any, Worker())]

    result = provider.repair(
        previous=previous,
        generation=0,
        slot="slot-00",
    )

    assert result == repaired
    assert provider._root_workers[0] == repaired_context
