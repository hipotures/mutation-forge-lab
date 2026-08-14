from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge import cli
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
            "config": {
                "order": case.order,
                "horizon": case.horizon,
            },
            "utility_trajectory": [
                {
                    "lower": {"numerator": 2, "denominator": 5},
                    "upper": {"numerator": 2, "denominator": 5},
                }
                for _ in range(case.horizon + 1)
            ],
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


def _panel_of_size(size: int) -> tuple[DevelopmentCaseV1, ...]:
    return tuple(
        DevelopmentCaseV1(
            f"case-{index:04d}",
            8,
            1000 + index,
            2000 + index,
            1,
            64,
            (4, 8),
        )
        for index in range(size)
    )


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
        *,
        program_barrier: threading.Barrier | None = None,
        result_hook: Callable[[int, str], None] | None = None,
    ) -> None:
        self.durable = durable if durable is not None else {}
        self.program_barrier = program_barrier
        self.result_hook = result_hook
        self.program_lock = threading.Lock()
        self.active_program_calls = 0
        self.peak_program_calls = 0
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
        self.parents: dict[tuple[int, str], M5ProviderContextV1] = {}
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
        self.parents[(generation, slot)] = parent
        if idempotency_key in self.durable:
            result = self.durable[idempotency_key]
            write_json(
                artifact_dir / "m5-provider-result.json.gz",
                result.as_dict(),
            )
            if self.result_hook is not None:
                self.result_hook(generation, slot)
            return result
        with self.program_lock:
            self.active_program_calls += 1
            self.peak_program_calls = max(
                self.peak_program_calls,
                self.active_program_calls,
            )
        try:
            if self.program_barrier is not None:
                self.program_barrier.wait(timeout=2)
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
            if self.result_hook is not None:
                self.result_hook(generation, slot)
            return result
        finally:
            with self.program_lock:
                self.active_program_calls -= 1

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
        self.baseline_calls: list[tuple[int, str, str]] = []
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
        with self.concurrency.lock:
            self.concurrency.baseline_calls.append(
                (generation, baseline, case.case_id)
            )
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
    force_stop: Any = None,
    boundary_hook: Any = None,
    resume_budget: ScientificResumeBudgetV1 | None = None,
    evaluator_factory: Callable[[], _Evaluator] | None = None,
) -> dict[str, Any]:
    return run_sustained_search(
        provider=provider,
        evaluator_factory=evaluator_factory or (lambda: _Evaluator(concurrency)),
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
        force_stop=force_stop,
        boundary_hook=boundary_hook,
    )


def test_immediate_stop_force_closes_provider_without_deleting_capsule(
    tmp_path: Path,
) -> None:
    close_calls: list[tuple[bool, bool]] = []

    class RecordingProvider(_Provider):
        def close(
            self,
            *,
            cleanup_capsule: bool = True,
            force: bool = False,
        ) -> None:
            close_calls.append((cleanup_capsule, force))

    with pytest.raises(M5OperatorStop, match="immediate operator stop requested"):
        _run(
            tmp_path,
            provider=RecordingProvider(),
            concurrency=_Concurrency(parties=1),
            options=_options(workers=1),
            force_stop=lambda: True,
        )

    assert close_calls == [(False, True)]


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


def _change_generation_limit(
    config_path: Path,
    *,
    old: int,
    new: int,
) -> None:
    before = f"generation_limit = {old}\n"
    text = config_path.read_text(encoding="utf-8")
    assert text.count(before) == 1
    config_path.write_text(
        text.replace(before, f"generation_limit = {new}\n"),
        encoding="utf-8",
    )


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
    assert result["state"] == "blocked"
    assert result["resumable"] is True


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


def test_fresh_commit_uses_compact_outcome_and_resume_loads_durable_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel_of_size(320)
    options = _options(workers=4)
    slot_dir = tmp_path / "generations" / "generation-0000" / "slot-00"

    class Evaluator:
        def evaluate(
            self,
            *,
            source: str,
            case: DevelopmentCaseV1,
            candidate_id: str,
        ) -> Mapping[str, JsonValue]:
            del candidate_id
            return _result(source=source, case=case)

        def close(self) -> None:
            pass

    telemetry = search_module._RuntimeTelemetry(tmp_path, options)
    pool = search_module._ConcurrentEvaluatorPool(
        workers=4,
        queue_capacity=4,
        evaluator_factory=Evaluator,
        telemetry=telemetry,
    )
    source = _source("fresh-320")
    future = pool.submit(
        source=source,
        panel=panel,
        candidate_id="g0000-slot-00",
        slot_dir=slot_dir,
    )
    outcome = future.result(timeout=30)
    pool.close()

    assert outcome.evaluation_case_count == 320
    assert outcome.behavior_profile is not None
    assert not hasattr(outcome, "payloads")

    prepared = {
        "protocol_id": search_module.M10_PREPARED_CANDIDATE_PROTOCOL_ID,
        "status": "evaluation_pending",
        "candidate_id": "g0000-slot-00",
        "generation": 0,
        "slot": "slot-00",
        "kind": "root",
        "parent_candidate_id": None,
        "program_hash": "a" * 64,
        "source": source,
    }
    pending = search_module._PendingCommit(
        slot_plan=search_module.core.SlotPlanV1(
            slot="slot-00",
            kind="root",
            parent_candidate_id=None,
            panel_hash=search_module.core.panel_hash(panel),
            request_key="request-00",
        ),
        candidate_id="g0000-slot-00",
        slot_dir=slot_dir,
        prepared=prepared,
        future=future,
    )
    loads = 0
    original_load = search_module.core._load_mapping

    def counted_load(path: Path) -> dict[str, Any]:
        nonlocal loads
        if path.parent.name == "evaluations":
            loads += 1
        return original_load(path)

    monkeypatch.setattr(search_module.core, "_load_mapping", counted_load)
    committed = search_module._commit_pending(
        pending=pending,
        root=tmp_path,
        panel=panel,
        telemetry=telemetry,
        block=True,
        boundary_hook=None,
    )

    assert committed == (False, None)
    assert loads == 0

    resume_root = tmp_path / "resume"
    resume_slot = resume_root / "generations" / "generation-0000" / "slot-00"
    resume_telemetry = search_module._RuntimeTelemetry(resume_root, options)
    first_pool = search_module._ConcurrentEvaluatorPool(
        workers=4,
        queue_capacity=4,
        evaluator_factory=Evaluator,
        telemetry=resume_telemetry,
    )
    first_pool.submit(
        source=source,
        panel=panel,
        candidate_id="g0000-slot-00",
        slot_dir=resume_slot,
    ).result(timeout=30)
    first_pool.close()

    loads = 0
    resumed_telemetry = search_module._RuntimeTelemetry(resume_root, options)
    resumed_pool = search_module._ConcurrentEvaluatorPool(
        workers=4,
        queue_capacity=4,
        evaluator_factory=Evaluator,
        telemetry=resumed_telemetry,
    )
    resumed = resumed_pool.submit(
        source=source,
        panel=panel,
        candidate_id="g0000-slot-00",
        slot_dir=resume_slot,
    ).result(timeout=30)
    resumed_pool.close()

    assert resumed.evaluation_case_count == 320
    assert resumed.behavior_profile == outcome.behavior_profile
    assert loads == 320


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


def test_two_provider_lanes_advance_independently_without_duplicate_turns(
    tmp_path: Path,
) -> None:
    class TwoLaneProvider(_Provider):
        provider_concurrency = 2

        def __init__(self) -> None:
            super().__init__()
            self.initial = threading.Barrier(2, timeout=2)
            self.slot_two_started = threading.Event()
            self.lock = threading.Lock()
            self.starts: list[str] = []
            self.active_by_lane = [0, 0]
            self.peak_by_lane = [0, 0]
            self.active = 0
            self.peak = 0

        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            slot = str(kwargs["slot"])
            lane = self.primary_lane(generation=int(kwargs["generation"]), slot=slot)
            with self.lock:
                self.starts.append(slot)
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.active_by_lane[lane] += 1
                self.peak_by_lane[lane] = max(
                    self.peak_by_lane[lane],
                    self.active_by_lane[lane],
                )
            try:
                if slot in {"slot-00", "slot-01"}:
                    self.initial.wait()
                if slot == "slot-01":
                    assert self.slot_two_started.wait(timeout=2)
                elif slot == "slot-02":
                    self.slot_two_started.set()
                return super().generate_root(**kwargs)
            finally:
                with self.lock:
                    self.active -= 1
                    self.active_by_lane[lane] -= 1

    provider = TwoLaneProvider()
    report = _run(
        tmp_path,
        provider=provider,
        concurrency=_Concurrency(parties=1),
        options=_options(workers=2, concurrency=2),
    )

    assert set(provider.starts[:2]) == {"slot-00", "slot-01"}
    assert provider.starts.index("slot-02") < provider.starts.index("slot-03")
    assert provider.peak == 2
    assert provider.peak_by_lane == [1, 1]
    assert sorted(provider.calls) == [
        (0, f"slot-{index:02d}") for index in range(8)
    ]
    assert len(set(provider.calls)) == 8
    assert report["runtime"]["peak_active_provider_turns"] == 2


def test_provider_telemetry_distinguishes_queued_waiting_and_active(
    tmp_path: Path,
) -> None:
    telemetry = search_module._RuntimeTelemetry(
        tmp_path,
        _options(concurrency=2),
    )
    keys = [f"g0000-slot-{index:02d}-initial" for index in range(8)]
    telemetry.reserve_primary_generation(keys, limit=8)
    telemetry.provider_tasks_queued(keys)
    telemetry.provider_task_waiting(keys[0])
    waiting = telemetry.snapshot()
    assert len(cast(list[object], waiting["queued_provider_keys"])) == 7
    assert waiting["waiting_provider_keys"] == [keys[0]]
    assert waiting["active_provider_turns"] == 0

    assert telemetry.provider_started(keys[0], kind="primary")
    active = telemetry.snapshot()
    assert len(cast(list[object], active["queued_provider_keys"])) == 7
    assert active["waiting_provider_keys"] == []
    assert active["active_provider_turns"] == 1
    telemetry.provider_finished(0.1, key=keys[0], failed=False)


def test_parent_references_are_deduplicated_persisted_and_reused(
    tmp_path: Path,
) -> None:
    concurrency = _Concurrency(parties=1)
    reference_calls: list[tuple[str, str]] = []

    class CountingEvaluator(_Evaluator):
        def evaluate(
            self,
            *,
            source: str,
            case: DevelopmentCaseV1,
            candidate_id: str,
        ) -> Mapping[str, JsonValue]:
            if candidate_id.startswith("parent-reference-"):
                reference_calls.append((candidate_id, case.case_id))
            return super().evaluate(
                source=source,
                case=case,
                candidate_id=candidate_id,
            )

    provider = _Provider()
    _run(
        tmp_path,
        provider=provider,
        concurrency=concurrency,
        options=_options(workers=2, generations=2),
        evaluator_factory=lambda: CountingEvaluator(concurrency),
    )

    reference_root = (
        tmp_path / "generations" / "generation-0001" / "parent-references"
    )
    results = sorted(reference_root.glob("*/parent-reference-result.json.gz"))
    assert results
    assert len(reference_calls) == len(results) * len(_PANEL)
    assert len({candidate_id for candidate_id, _case_id in reference_calls}) == len(results)
    assert len(provider.calls) == 16
    assert not list(reference_root.glob("*/candidate.json.gz"))
    for path in results:
        retained = read_json(path)
        assert retained["authoritative"] is False
        assert retained["purpose"] == "same_panel_mutation_display"

    completed_calls = list(reference_calls)
    _run(
        tmp_path,
        provider=provider,
        concurrency=concurrency,
        options=_options(workers=2, generations=2),
        evaluator_factory=lambda: CountingEvaluator(concurrency),
    )
    assert reference_calls == completed_calls
    assert len(provider.calls) == 16


def test_parent_reference_resume_reuses_completed_cases_without_provider(
    tmp_path: Path,
) -> None:
    concurrency = _Concurrency(parties=1)
    calls: list[str] = []

    class CountingEvaluator(_Evaluator):
        def evaluate(
            self,
            *,
            source: str,
            case: DevelopmentCaseV1,
            candidate_id: str,
        ) -> Mapping[str, JsonValue]:
            calls.append(case.case_id)
            return super().evaluate(
                source=source,
                case=case,
                candidate_id=candidate_id,
            )

    generation_dir = tmp_path / "generations" / "generation-0001"
    parent = {
        "candidate_id": "g0000-slot-00",
        "program_hash": "parent-program",
        "source": _source("parent"),
    }
    first_case_path = (
        generation_dir
        / "parent-references"
        / "g0000-slot-00"
        / "evaluations"
        / f"{_PANEL[0].case_id}.json.gz"
    )
    write_json(
        first_case_path,
        _result(source=str(parent["source"]), case=_PANEL[0]),
    )
    telemetry = search_module._RuntimeTelemetry(tmp_path, _options())
    pool = search_module._ConcurrentEvaluatorPool(
        workers=2,
        queue_capacity=4,
        evaluator_factory=lambda: CountingEvaluator(concurrency),
        telemetry=telemetry,
    )
    outcome = pool.submit_parent_reference(
        parent=parent,
        panel=_PANEL,
        generation=1,
        generation_dir=generation_dir,
    ).result(timeout=30)
    pool.close()
    assert outcome.evaluation_case_count == len(_PANEL)
    assert calls == [_PANEL[1].case_id]

    resumed_telemetry = search_module._RuntimeTelemetry(tmp_path, _options())
    resumed_pool = search_module._ConcurrentEvaluatorPool(
        workers=2,
        queue_capacity=4,
        evaluator_factory=lambda: CountingEvaluator(concurrency),
        telemetry=resumed_telemetry,
    )
    resumed_pool.submit_parent_reference(
        parent=parent,
        panel=_PANEL,
        generation=1,
        generation_dir=generation_dir,
    ).result(timeout=30)
    resumed_pool.close()
    assert calls == [_PANEL[1].case_id]


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
    all_evaluations = [*first_concurrency.calls, *resumed_concurrency.calls]
    authoritative = [
        item
        for item in all_evaluations
        if not item[0].startswith("parent-reference-")
    ]
    parent_references = [
        item
        for item in all_evaluations
        if item[0].startswith("parent-reference-")
    ]
    assert len(authoritative) == 32
    assert len(parent_references) == 8


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


def test_interrupted_provider_resume_budget_is_derived_from_workspace(
    tmp_path: Path,
) -> None:
    class Crash(BaseException):
        pass

    class CrashingProvider(_Provider):
        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            del kwargs
            raise Crash

    options = _options(repairs=24)
    with pytest.raises(Crash):
        _run(
            tmp_path,
            provider=CrashingProvider(),
            concurrency=_Concurrency(parties=1),
            options=options,
        )

    budget = search_module.automatic_resume_budget(
        root=tmp_path,
        options=options,
    )

    assert budget == ScientificResumeBudgetV1(
        expected_pending_primary_slots=8,
        max_new_repair_turns=8,
    )
    search_module._RuntimeTelemetry(
        tmp_path,
        options,
        budget,
    )
    runtime = read_json(tmp_path / search_module.M10_RUNTIME_FILENAME)
    assert isinstance(runtime, Mapping)
    assert runtime["resume_budget_guard"] == {
        "protocol_id": "mforge.native.python.resume_budget.v1",
        "expected_pending_primary_slots": 8,
        "max_new_repair_turns": 8,
        "provider_started_baseline": runtime["provider_started_keys"],
        "repair_turn_baseline": runtime["repair_turn_keys"],
    }


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


def test_runtime_telemetry_throttles_case_updates_and_forces_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    writes: list[dict[str, Any]] = []

    monkeypatch.setattr(search_module.time, "monotonic", lambda: now[0])

    def record_runtime(_path: Path, payload: Mapping[str, Any]) -> None:
        writes.append(json.loads(json.dumps(payload)))

    monkeypatch.setattr(search_module, "write_json", record_runtime)
    telemetry = search_module._RuntimeTelemetry(tmp_path, _options())
    telemetry.evaluation_started(
        key="candidate:g0000-slot-00",
        total=1000,
        completed=0,
        queued=1000,
    )

    for completed in range(1, 1001):
        now[0] = completed / 1000
        telemetry.evaluation_case_completed(
            key="candidate:g0000-slot-00",
            completed=completed,
        )

    assert len(writes) < 10
    assert telemetry.snapshot()["candidate_evaluation_cases_completed"] == 1000
    writes_before_boundary = len(writes)

    telemetry.boundary("generation_0_committed")

    assert len(writes) == writes_before_boundary + 1
    assert writes[-1]["candidate_evaluation_cases_completed"] == 1000
    assert writes[-1]["last_boundary"] == "generation_0_committed"

    telemetry.finish("generation_budget")

    assert writes[-1]["candidate_evaluation_cases_completed"] == 1000
    assert writes[-1]["terminal_reason"] == "generation_budget"
    assert writes[-1]["current_run_elapsed_seconds"] == pytest.approx(1.0)
    now[0] = 10.0
    assert telemetry.snapshot()["current_run_elapsed_seconds"] == pytest.approx(1.0)


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

    assert status["state"] == "blocked"
    assert status["resumable"] is True
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
    assert status["evaluation_cases"]["baseline"] == {
        "active_completed": 0,
        "active_total": 0,
        "completed": 8,
        "total": 8,
    }
    assert status["evaluation_cases"]["candidate"] == {
        "active_completed": 0,
        "active_total": 0,
        "completed": 32,
        "total": 32,
    }
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


def test_terminal_provider_turn_projects_persisting_before_result_artifact(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(tmp_path, exp_id="provider-persisting")
    config = load_python_preview_config(config_path)
    state = preview_module._initialize_workspace(config)
    state.update(
        {
            "state": "running",
            "resumable": True,
            "run_terminal": False,
            "last_boundary": "generation_0_snapshot",
        }
    )
    root = config.experiment_root
    slot_dir = root / "generations" / "generation-0000" / "slot-00"
    write_json(
        slot_dir.parent / "manifest.json.gz",
        {
            "generation": 0,
            "slots": [
                {
                    "slot": "slot-00",
                    "kind": "root",
                    "parent_candidate_id": None,
                    "request_key": "g0000-slot-00",
                }
            ],
        },
    )
    write_json(
        root / search_module.M10_RUNTIME_FILENAME,
        {
            "protocol_id": search_module.M10_RUNTIME_PROTOCOL_ID,
            "resume_started_epoch_seconds": time.time() - 1,
            "active_provider_turns": 1,
            "provider_turns_submitted": 1,
            "provider_concurrency_timeline": [
                {
                    "key": "g0000-slot-00-initial",
                    "kind": "primary",
                    "started_epoch_seconds": time.time() - 0.5,
                }
            ],
            "evaluation_progress": {},
        },
    )
    write_json(
        slot_dir
        / "provider-initial"
        / "g0000-slot-00-root.turn-terminal.json.gz",
        {
            "status": "completed",
            "turn_id": "turn-slot-00",
            "final_item_received": True,
            "usage_observed": True,
        },
    )

    status = preview_module._progress(config, state)
    assert not (slot_dir / "provider-initial" / "m5-provider-result.json.gz").exists()
    assert status["slots"][0]["state"] == "persisting"
    assert status["slots"][0]["phase"] == "response"
    rich = dashboard_state_from_python_status(
        status,
        run_id=config.exp_id,
        model=config.model,
        effort=config.effort,
        generation_limit=1,
        wall_seconds=60,
    )
    assert rich.generations[0].slots[0].state == "persisting"
    assert rich.generations[0].slots[0].phase == "response"

    state.update(
        {
            "state": "blocked",
            "terminal_reason": "infrastructure_failure",
            "last_error": "M5InfrastructureError: provider turn exceeded 300s",
        }
    )
    write_json(
        root / search_module.M10_RUNTIME_FILENAME,
        {
            "protocol_id": search_module.M10_RUNTIME_PROTOCOL_ID,
            "resume_started_epoch_seconds": time.time() - 1,
            "active_provider_turns": 0,
            "provider_turns_submitted": 1,
            "provider_concurrency_timeline": [
                {
                    "key": "g0000-slot-00-initial",
                    "kind": "primary",
                    "started_epoch_seconds": time.time() - 0.5,
                    "finished_epoch_seconds": time.time(),
                    "failed": True,
                    "error": "M5InfrastructureError: provider turn exceeded 300s",
                }
            ],
            "evaluation_progress": {},
        },
    )
    failed = preview_module._progress(config, state)
    assert failed["last_error"] == "M5InfrastructureError: provider turn exceeded 300s"
    assert failed["slots"][0]["state"] == "failed"
    assert failed["slots"][0]["error"] == (
        "M5InfrastructureError: provider turn exceeded 300s"
    )


def test_slot_usage_is_identical_from_provider_terminal_through_archive(
    tmp_path: Path,
) -> None:
    config_path = _scientific_config(tmp_path, exp_id="live-slot-usage")
    config = load_python_preview_config(config_path)
    state = preview_module._initialize_workspace(config)
    state.update(
        {
            "state": "running",
            "resumable": True,
            "run_terminal": False,
            "last_boundary": "generation_0_snapshot",
        }
    )
    root = config.experiment_root
    slot_dir = root / "generations" / "generation-0000" / "slot-00"
    expected_usage = {
        "inputTokens": 101,
        "cachedInputTokens": 11,
        "cacheWriteInputTokens": 7,
        "outputTokens": 23,
        "reasoningOutputTokens": 5,
        "totalTokens": 131,
    }
    attempt = {
        "usage": {
            **expected_usage,
            "final": True,
            "partial": False,
        },
        "warnings": 0,
    }
    write_json(
        slot_dir.parent / "manifest.json.gz",
        {
            "generation": 0,
            "slots": [
                {
                    "slot": "slot-00",
                    "kind": "root",
                    "parent_candidate_id": None,
                    "request_key": "g0000-slot-00",
                }
            ],
        },
    )
    write_json(
        root / search_module.M10_RUNTIME_FILENAME,
        {
            "protocol_id": search_module.M10_RUNTIME_PROTOCOL_ID,
            "resume_started_epoch_seconds": time.time() - 1,
            "active_provider_turns": 1,
            "provider_turns_submitted": 1,
            "provider_concurrency_timeline": [
                {
                    "key": "g0000-slot-00-initial",
                    "kind": "primary",
                    "started_epoch_seconds": time.time() - 0.5,
                }
            ],
            "evaluation_progress": {},
        },
    )
    write_json(slot_dir / "provider-initial" / "m5-provider-result.json.gz", attempt)
    write_json(
        slot_dir / "provider-initial" / "turn.turn-terminal.json.gz",
        {"status": "completed", "usage_observed": True},
    )

    provider_terminal = preview_module._progress(config, state)
    assert provider_terminal["slots"][0]["state"] == "persisting"
    assert provider_terminal["slots"][0]["usage"] == expected_usage
    assert provider_terminal["provider"]["usage"] == {
        key: value for key, value in expected_usage.items()
    }

    write_json(
        slot_dir / search_module.M10_PREPARED_FILENAME,
        {
            "provider_attempts": [attempt],
            "usage": expected_usage,
            "repairs": 0,
        },
    )
    runtime = cast(
        dict[str, Any],
        read_json(root / search_module.M10_RUNTIME_FILENAME),
    )
    runtime["active_provider_turns"] = 0
    runtime["evaluation_progress"] = {
        "candidate:g0000-slot-00": {
            "state": "running",
            "completed": 1,
            "total": 4,
        }
    }
    write_json(root / search_module.M10_RUNTIME_FILENAME, runtime)

    evaluating = preview_module._progress(config, state)
    assert evaluating["slots"][0]["state"] == "evaluation_running"
    assert evaluating["slots"][0]["usage"] == expected_usage
    assert evaluating["provider"]["usage"] == expected_usage

    write_json(
        slot_dir / "candidate.json.gz",
        {
            "candidate_id": "g0000-slot-00",
            "generation": 0,
            "slot": "slot-00",
            "kind": "root",
            "status": "evaluated",
            "parent_candidate_id": None,
            "provider_attempts": [attempt],
            "usage": expected_usage,
            "behavior_profile": {
                "fitness_interval": {
                    "lower": {"numerator": 1, "denominator": 2},
                    "upper": {"numerator": 1, "denominator": 2},
                }
            },
        },
    )
    write_json(
        slot_dir / search_module.M10_PREPARED_FILENAME,
        {
            "provider_attempts": [],
            "usage": {},
            "repairs": 0,
        },
    )

    archived = preview_module._progress(config, state)
    assert archived["slots"][0]["state"] == "evaluated"
    assert archived["slots"][0]["phase"] == "archived"
    assert archived["slots"][0]["usage"] == expected_usage
    assert archived["provider"]["usage"] == expected_usage
    assert [
        snapshot["slots"][0]["usage"]
        for snapshot in (provider_terminal, evaluating, archived)
    ] == [expected_usage] * 3


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


def test_provider_free_multi_generation_lifecycle_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="mutable-generation-limit",
        generations=2,
        workers=1,
    )
    initial_identity = load_python_preview_config(
        config_path
    ).scientific_config_sha256
    durable: dict[str, M5ProviderResultV1] = {}
    providers: list[_Provider] = []
    evaluator_runs: list[_Concurrency] = []
    live_usage_before_evaluation: list[Mapping[str, Any]] = []

    def record_live_usage(generation: int, slot: str) -> None:
        if generation != 0 or slot != "slot-00" or live_usage_before_evaluation:
            return
        live_status = python_preview_status(config_path)
        live_slot = next(
            item
            for item in live_status["slots"]
            if item["candidate_id"] == "g0000-slot-00"
        )
        slot_dir = (
            load_python_preview_config(config_path).experiment_root
            / "generations"
            / "generation-0000"
            / "slot-00"
        )
        assert not list((slot_dir / "evaluations").glob("*.json.gz"))
        live_usage_before_evaluation.append(cast(Mapping[str, Any], live_slot["usage"]))

    def provider_factory(*_: Any) -> _Provider:
        provider = _Provider(
            durable,
            program_barrier=threading.Barrier(4, timeout=2),
            result_hook=record_live_usage,
        )
        providers.append(provider)
        return provider

    def evaluator_factory(*_: Any) -> _Evaluator:
        instrumentation = _Concurrency(parties=1)
        evaluator_runs.append(instrumentation)
        return _Evaluator(instrumentation)

    # Phase A: create two generations in one stable experiment.
    first = run_python_preview(
        config_path,
        provider_factory=provider_factory,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=evaluator_factory,
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )
    config = load_python_preview_config(config_path)
    root = config.experiment_root
    retained_generation_bytes = {
        path.relative_to(root): path.read_bytes()
        for generation in (0, 1)
        for path in (
            root / "generations" / f"generation-{generation:04d}"
        ).rglob("*")
        if path.is_file()
    }
    all_bytes_at_limit = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    assert first["state"] == "blocked"
    assert first["resumable"] is True
    assert first["terminal_reason"] == "generation_budget"
    assert first["generation_index"] == 1
    assert first["counts"]["terminal"] == 16
    assert first["provider"]["candidate_turns"] == 16
    assert first["provider"]["usage"]["totalTokens"] == 34
    assert live_usage_before_evaluation == [
        {
            key: value
            for key, value in _usage().items()
            if key not in {"final", "partial"}
        }
    ]
    assert providers[0].peak_program_calls == 4
    assert len(providers[0].calls) == 16
    assert len(evaluator_runs) == 1
    panel_case_count = first["evaluation_workload"]["case_count"]
    assert len(
        [
            call
            for call in evaluator_runs[0].calls
            if call[0].startswith(("g0000-", "g0001-"))
        ]
    ) == 16 * panel_case_count
    assert len(evaluator_runs[0].baseline_calls) == 2 * 2 * panel_case_count
    assert all(
        isinstance(slot["elapsed_seconds"], int | float)
        and slot["elapsed_seconds"] > 0
        for slot in first["slots"]
    )
    assert {generation for generation, _ in providers[0].calls} == {0, 1}

    for generation in (0, 1):
        manifest = cast(
            Mapping[str, Any],
            read_json(
                root
                / "generations"
                / f"generation-{generation:04d}"
                / "manifest.json.gz"
            ),
        )
        slots = cast(list[Mapping[str, Any]], manifest["slots"])
        expected_kinds = ["root"] * 8 if generation == 0 else ["child"] * 4 + ["root"] * 4
        assert [slot["kind"] for slot in slots] == expected_kinds
        for slot in slots:
            slot_name = str(slot["slot"])
            parent_id = slot.get("parent_candidate_id")
            provider_parent = providers[0].parents[(generation, slot_name)]
            if slot["kind"] == "root":
                assert parent_id is None
                assert provider_parent == providers[0].anchor
                continue
            assert isinstance(parent_id, str)
            assert parent_id.startswith(f"g{generation - 1:04d}-")
            parent_slot = parent_id.split("-", 1)[1]
            parent = cast(
                Mapping[str, Any],
                read_json(
                    root
                    / "generations"
                    / f"generation-{generation - 1:04d}"
                    / parent_slot
                    / "candidate.json.gz"
                ),
            )
            assert provider_parent.as_dict() == parent["provider_context"]
            assert str(parent["source"]) in providers[0].prompts[(generation, slot_name)]

    status_evaluation_reads = 0
    original_status_load = preview_module._load_mapping

    def counted_status_load(path: Path) -> dict[str, Any] | None:
        nonlocal status_evaluation_reads
        if path.parent.name == "evaluations":
            status_evaluation_reads += 1
        return original_status_load(path)

    with monkeypatch.context() as status_probe:
        status_probe.setattr(
            preview_module,
            "_load_mapping",
            counted_status_load,
        )
        python_preview_status(config_path)
    assert status_evaluation_reads == 0

    # At the bound, the ordinary run path is a read-only no-op.
    with monkeypatch.context() as no_work:
        no_work.setattr(
            preview_module,
            "_progress",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("no-op built the full status projection")
            ),
        )
        same_limit = run_python_preview(
            config_path,
            provider_factory=lambda *_: (_ for _ in ()).throw(
                AssertionError("no-op constructed provider")
            ),
            backend_factory=lambda _: (_ for _ in ()).throw(
                AssertionError("no-op constructed backend")
            ),
            evaluator_factory=lambda *_: (_ for _ in ()).throw(
                AssertionError("no-op constructed evaluator")
            ),
            provenance_guard=lambda **_: {},
            auth_available=lambda _: True,
        )
    assert same_limit["state"] == "blocked"
    assert same_limit["resumable"] is True
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == all_bytes_at_limit

    # Phase B: extend only generation_limit and resume the same campaign to G4.
    protocol = cast(dict[str, Any], read_json(root / "protocol.json.gz"))
    protocol["generation_limit"] = 2
    write_json(root / "protocol.json.gz", protocol)
    _change_generation_limit(config_path, old=2, new=5)
    assert load_python_preview_config(config_path).scientific_config_sha256 == (
        initial_identity
    )
    to_five = run_python_preview(
        config_path,
        provider_factory=provider_factory,
        backend_factory=lambda _: _Backend(),
        evaluator_factory=evaluator_factory,
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert to_five["state"] == "blocked"
    assert to_five["resumable"] is True
    assert to_five["terminal_reason"] == "generation_budget"
    assert to_five["generation_index"] == 4
    assert to_five["counts"]["planned"] == 40
    assert to_five["counts"]["terminal"] == 40
    assert to_five["provider"]["candidate_turns"] == 40
    assert to_five["provider"]["turns"] == 41
    assert to_five["provider"]["usage"] == {
        "inputTokens": 41,
        "cachedInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "outputTokens": 41,
        "reasoningOutputTokens": 0,
        "totalTokens": 82,
    }
    assert {generation for generation, _ in providers[1].calls} == {2, 3, 4}
    assert len(providers[1].calls) == 24
    assert providers[1].peak_program_calls == 4
    assert len(evaluator_runs) == 2
    resumed_candidate_calls = [
        call for call in evaluator_runs[1].calls if call[0].startswith("g")
    ]
    assert len(resumed_candidate_calls) == 24 * panel_case_count
    assert all(
        candidate_id.startswith(("g0002-", "g0003-", "g0004-"))
        for candidate_id, _case_id in resumed_candidate_calls
    )
    assert len(evaluator_runs[1].baseline_calls) == 3 * 2 * panel_case_count
    assert {
        generation for generation, _baseline, _case_id in evaluator_runs[1].baseline_calls
    } == {2, 3, 4}
    assert {
        path.relative_to(root): path.read_bytes()
        for generation in (0, 1)
        for path in (
            root / "generations" / f"generation-{generation:04d}"
        ).rglob("*")
        if path.is_file()
    } == retained_generation_bytes

    assert sorted(
        path.name
        for path in (root / "generations").glob("generation-*")
    ) == [f"generation-{generation:04d}" for generation in range(5)]

    for generation in (2, 3, 4):
        manifest = cast(
            Mapping[str, Any],
            read_json(
                root
                / "generations"
                / f"generation-{generation:04d}"
                / "manifest.json.gz"
            ),
        )
        slots = cast(list[Mapping[str, Any]], manifest["slots"])
        assert [slot["kind"] for slot in slots] == ["child"] * 4 + ["root"] * 4
        for slot in slots:
            slot_name = str(slot["slot"])
            parent_id = slot.get("parent_candidate_id")
            if slot["kind"] == "root":
                assert parent_id is None
                assert providers[1].parents[(generation, slot_name)] == providers[1].anchor
                continue
            assert isinstance(parent_id, str)
            assert parent_id.startswith(f"g{generation - 1:04d}-")
            parent_slot = parent_id.split("-", 1)[1]
            parent = cast(
                Mapping[str, Any],
                read_json(
                    root
                    / "generations"
                    / f"generation-{generation - 1:04d}"
                    / parent_slot
                    / "candidate.json.gz"
                ),
            )
            assert (
                providers[1].parents[(generation, slot_name)].as_dict()
                == parent["provider_context"]
            )
            assert str(parent["source"]) in providers[1].prompts[(generation, slot_name)]

    current_slots = [
        slot for slot in to_five["slots"] if slot["generation"] == 4
    ]
    assert len(current_slots) == 8
    assert all(slot["usage"]["totalTokens"] == 2 for slot in current_slots)
    assert all(slot["gain"] == pytest.approx(0.1) for slot in current_slots)

    # Phase C: cold status reads only compact summaries, preserves browsing,
    # and the idle dashboard exits on q without starting worker work.
    cold_evaluation_reads = 0

    def counted_cold_load(path: Path) -> dict[str, Any] | None:
        nonlocal cold_evaluation_reads
        if path.parent.name == "evaluations":
            cold_evaluation_reads += 1
        return original_status_load(path)

    with monkeypatch.context() as cold_probe:
        cold_probe.setattr(preview_module, "_load_mapping", counted_cold_load)
        cold = python_preview_status(config_path)
    assert cold_evaluation_reads == 0
    assert cold["terminal_reason"] == "generation_budget"
    rich = dashboard_state_from_python_status(
        cold,
        run_id=config.exp_id,
        model=config.model,
        effort=config.effort,
        generation_limit=5,
        wall_seconds=60,
    )
    assert [group.generation for group in rich.generations] == [0, 1, 2, 3, 4]

    dashboard_states: list[Any] = []

    class AutoQuitDashboard:
        def __init__(self, **kwargs: Any) -> None:
            dashboard_states.append(kwargs["initial_state"])
            kwargs["capabilities"].quit()

        def update_canonical_state(self, status: Any) -> None:
            dashboard_states.append(status)

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "InteractiveDashboardSink", AutoQuitDashboard)
    monkeypatch.setattr(
        cli,
        "run_python_preview",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("idle dashboard started worker work")
        ),
    )
    started = time.monotonic()
    assert cli._experiment_run(config_path, json_output=False, dashboard=True) == 0
    assert time.monotonic() - started < 0.5
    assert dashboard_states[-1].terminal_reason == "generation_budget"


def test_generation_limit_stop_preserves_provider_capsule_but_exact_stop_cleans_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_flags: list[bool] = []

    class RecordingProvider(_Provider):
        def close(self, *, cleanup_capsule: bool = True) -> None:
            cleanup_flags.append(cleanup_capsule)

    monkeypatch.setattr(
        preview_module,
        "CodexM10SearchProvider",
        RecordingProvider,
    )
    limit_path = _scientific_config(
        tmp_path,
        exp_id="generation-limit-cleanup",
        workers=1,
    )
    result = run_python_preview(
        limit_path,
        provider_factory=lambda *_: RecordingProvider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert result["terminal_reason"] == "generation_budget"
    assert cleanup_flags == [False]

    terminal_path = _scientific_config(
        tmp_path,
        exp_id="exact-stop-cleanup",
        workers=1,
    )
    monkeypatch.setattr(
        preview_module,
        "run_sustained_search",
        lambda **_: {
            "stop_reason": "exact_verified_counterexample",
            "exact_verified": True,
        },
    )
    terminal = run_python_preview(
        terminal_path,
        provider_factory=lambda *_: RecordingProvider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert terminal["state"] == "completed"
    assert cleanup_flags == [False, True]


def test_legacy_generation_budget_state_and_missing_capsule_are_narrowly_interpreted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _scientific_config(
        tmp_path,
        exp_id="legacy-generation-limit",
        workers=1,
    )
    run_python_preview(
        config_path,
        provider_factory=lambda *_: _Provider(),
        backend_factory=lambda _: _Backend(),
        evaluator_factory=lambda *_: _Evaluator(_Concurrency(parties=1)),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )
    config = load_python_preview_config(config_path)
    root = config.experiment_root
    state_path = root / "python-preview-state.json.gz"
    legacy_state = cast(dict[str, Any], read_json(state_path))
    legacy_state.update(
        {
            "config_sha256": config.legacy_scientific_config_sha256,
            "state": "completed",
            "resumable": False,
            "run_terminal": True,
            "terminal_reason": "generation_budget",
            "last_error": None,
        }
    )
    write_json(state_path, legacy_state)
    missing_capsule = tmp_path / "deleted-provider-capsule"
    write_json(
        root
        / "provider-runtime"
        / "coordinator"
        / "provider-state.json.gz",
        {"capsule_root": str(missing_capsule)},
    )

    status = python_preview_status(config_path)
    assert status["state"] == "blocked"
    assert status["resumable"] is True
    assert status["terminal_reason"] == "generation_budget"

    with monkeypatch.context() as no_work:
        no_work.setattr(
            preview_module,
            "_progress",
            lambda *_: (_ for _ in ()).throw(
                AssertionError("legacy no-op built the full status projection")
            ),
        )
        same_limit = run_python_preview(
            config_path,
            provider_factory=lambda *_: (_ for _ in ()).throw(
                AssertionError("legacy no-op constructed provider")
            ),
            backend_factory=lambda _: (_ for _ in ()).throw(
                AssertionError("legacy no-op constructed backend")
            ),
            provenance_guard=lambda **_: {},
            auth_available=lambda _: True,
        )
    assert same_limit["state"] == "blocked"
    assert same_limit["terminal_reason"] == "generation_budget"

    _change_generation_limit(config_path, old=1, new=2)
    continuation = run_python_preview(
        config_path,
        provider_factory=lambda *_: (_ for _ in ()).throw(
            AssertionError("legacy missing capsule constructed provider")
        ),
        backend_factory=lambda _: (_ for _ in ()).throw(
            AssertionError("legacy missing capsule constructed backend")
        ),
        provenance_guard=lambda **_: {},
        auth_available=lambda _: True,
    )

    assert continuation["state"] == "blocked"
    assert continuation["resumable"] is False
    assert (
        continuation["terminal_reason"]
        == "legacy_generation_limit_provider_runtime_missing"
    )


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
        assert status["state"] == "blocked"
        assert status["resumable"] is True

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

    assert resumed["state"] == "blocked"
    assert resumed["resumable"] is True
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
    assert status["state"] == "blocked"
    assert status["run_terminal"] is False
    assert status["resumable"] is True


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
