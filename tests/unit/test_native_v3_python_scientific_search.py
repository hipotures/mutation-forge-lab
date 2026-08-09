from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
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
from mutation_forge.native_v3_python.scientific_search import (
    M10_REPORT_PROTOCOL_ID,
    M10_SEARCH_PROTOCOL_ID,
    M10_STOP_FILENAME,
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
    M5_PROVIDER_MAX_CAMPAIGNS,
    M5_PROVIDER_MAX_TURNS,
    M10_PROVIDER_MAX_EVENTS,
    M10_PROVIDER_STDOUT_BYTES,
    M10_PROVIDER_TRANSCRIPT_BYTES,
    CodexM5SearchProvider,
    CodexM10SearchProvider,
    specification_ack_schema,
)
from mutation_forge.output.interactive_dashboard import (
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
        "schema_version": {
            "const": "mforge.native.python_policy_response.v1"
        },
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
    digest = hashlib.sha256(
        f"{source}:{case.case_id}".encode()
    ).hexdigest()
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

    def ensure_specification_anchor(
        self,
        **_: Any,
    ) -> M5ProviderResultV1:
        return M5ProviderResultV1(
            response_text=json.dumps(
                {
                    "schema_version": (
                        "mforge.native.python_m5_specification_ack.v1"
                    ),
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
                    "schema_version": (
                        "mforge.native.python_policy_response.v1"
                    ),
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
        assert order == 30
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
    )


def _run(
    root: Path,
    *,
    provider: _Provider,
    concurrency: _Concurrency,
    options: ScientificSearchOptionsV2,
    operator_stop: Any = None,
    boundary_hook: Any = None,
) -> dict[str, Any]:
    return run_sustained_search(
        provider=provider,
        evaluator_factory=lambda: _Evaluator(concurrency),
        workspace=root,
        panel=_PANEL,
        system_prompt="system",
        specification_prompt="specification",
        specification_ack_schema=specification_ack_schema(),
        policy_schema=_POLICY_SCHEMA,
        options=options,
        provider_turn_timeout_seconds=1,
        operator_stop=operator_stop,
        boundary_hook=boundary_hook,
    )


def _scientific_config(
    tmp_path: Path,
    *,
    exp_id: str,
    generations: int = 1,
    repairs: int = 0,
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
timeout_seconds = 30
heg_repo = "{heg.as_posix()}"

[python_preview.scientific_search]
generation_limit = {generations}
evaluator_workers = 2
provider_concurrency = 4
wall_seconds = 60
primary_program_slots = {generations * 8}
repair_turn_limit = {repairs}
provider_total_turn_limit = {generations * 8 + repairs}
validated_queue_target = 4
validated_queue_capacity = 8
stop_on_verified = true
resume_enabled = true
replace_terminal_slots = false
''',
        encoding="utf-8",
    )
    return config_path


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
    assert report["provider_order"] == [
        f"g0000-slot-{index:02d}" for index in range(8)
    ]
    assert concurrency.peak == 2
    assert report["runtime"]["peak_active_evaluators"] == 2
    assert report["acceptance_checks"][
        "provider_program_turn_budget_respected"
    ] is True


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
    assert report["provider_order"] == [
        f"g0000-slot-{index:02d}" for index in range(8)
    ]
    assert len(provider.snapshots) == 1
    snapshot = provider.snapshots[0]
    assert snapshot["generation"] == 0
    assert len(snapshot["slots"]) == 8
    assert len(set(provider.prompts.values())) == 1
    assert "active_parent=null" in next(iter(provider.prompts.values()))
    retained = read_json(
        tmp_path
        / "generations"
        / "generation-0000"
        / "generation-snapshot.json.gz"
    )
    assert retained == snapshot


def test_repair_budget_is_separate_and_allocated_in_canonical_slot_order(
    tmp_path: Path,
) -> None:
    class RepairProvider(_Provider):
        def __init__(self) -> None:
            super().__init__()
            self.repair_slots: list[str] = []

        def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
            result = super().generate_root(**kwargs)
            slot = str(kwargs["slot"])
            if slot not in {"slot-00", "slot-01"}:
                return result
            invalid = M5ProviderResultV1(
                response_text=json.dumps(
                    {
                        "schema_version": (
                            "mforge.native.python_policy_response.v1"
                        ),
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
                Path(kwargs["artifact_dir"])
                / "m5-provider-result.json.gz",
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
                        "schema_version": (
                            "mforge.native.python_policy_response.v1"
                        ),
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
    assert report["acceptance_checks"][
        "later_generations_four_children_four_roots"
    ] is True


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
        for path in sorted(
            tmp_path.glob(
                "generations/generation-*/slot-*/candidate.json.gz"
            )
        )
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
    assert sorted(resumed_provider.calls) == [
        (1, f"slot-{index:02d}") for index in range(8)
    ]
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

    assert len(first_provider.calls) == 8
    resumed = _Provider(durable)
    report = _run(
        tmp_path,
        provider=resumed,
        concurrency=_Concurrency(parties=1),
        options=_options(),
    )
    assert report["candidate_count"] == 8
    assert resumed.calls == []


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
        evaluator_factory=lambda *_: _Evaluator(
            _Concurrency(parties=1)
        ),
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
    assert rich.provider_turns_attempted == status["provider"][
        "program_turns_reserved"
    ]
    assert rich.active_provider_turns == status["provider"]["active"]
    assert rich.configured_provider_concurrency == status["provider"][
        "configured_concurrency"
    ]
    assert rich.evaluations_completed == status["evaluators"]["completed"]
    assert rich.evaluation_workers_active == status["evaluators"]["active"]
    assert sum(len(group.slots) for group in rich.generations) == status[
        "counts"
    ]["planned"]


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
    assert m5_limits.max_turns == M5_PROVIDER_MAX_TURNS
    assert m5_limits.max_campaigns == M5_PROVIDER_MAX_CAMPAIGNS
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

        def prepare_root_worker(
            self,
            *,
            anchor: M5ProviderContextV1,
            worker: int,
            artifact_dir: Path,
        ) -> M5ProviderContextV1:
            assert "coordinator-close-False" in events
            events.append(f"worker-{worker}-resume")
            return M5ProviderContextV1(
                f"worker-{worker}",
                anchor.turn_id,
                None,
                anchor.included_turn_ids,
            )

        def ensure_context(self, _: M5ProviderContextV1) -> None:
            return None

        def close(self, *, cleanup_capsule: bool = True) -> None:
            kind = "coordinator" if self.coordinator else "worker"
            events.append(f"{kind}-close-{cleanup_capsule}")

    def provider_factory(**_: Any) -> FakeProvider:
        provider = FakeProvider(coordinator=not providers)
        providers.append(provider)
        return provider

    monkeypatch.setattr(
        provider_module, "CodexM5SearchProvider", provider_factory
    )
    monkeypatch.setattr(
        provider_module, "CodexAppServerAdapter", lambda **_: object()
    )
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
    anchor = M5ProviderContextV1(
        "anchor-thread", "anchor-turn", None, ("anchor-turn",)
    )
    provider._anchor = anchor
    provider.prepare_generation(
        snapshot={
            "generation": 0,
            "slots": [
                {"slot": f"slot-{slot:02d}", "kind": "root"}
                for slot in range(8)
            ],
        },
        anchor=anchor,
        artifact_dir=tmp_path / "generation-provider",
    )
    assert events[:5] == [
        "coordinator-close-False",
        "worker-0-resume",
        "worker-1-resume",
        "worker-2-resume",
        "worker-3-resume",
    ]
    provider.close(cleanup_capsule=False)
    assert events[-1] == "coordinator-close-False"
