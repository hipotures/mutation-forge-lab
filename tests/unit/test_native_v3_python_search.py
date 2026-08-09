from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3_python import search as search_module
from mutation_forge.native_v3_python import search_provider as search_provider_module
from mutation_forge.native_v3_python.search import (
    M5_TERMINAL_CANDIDATE_STATUSES,
    DevelopmentCaseV1,
    M5InfrastructureError,
    M5OperatorStop,
    M5ProviderContextV1,
    M5ProviderResultV1,
    aggregate_behavior,
    build_child_prompt,
    build_generation_manifest,
    build_root_prompt,
    build_search_memory,
    run_m5_search,
    select_parent_candidates,
)

_SYSTEM = "Follow the supplied request and return only the required JSON object."
_SPEC = "Retain the ordinary-Python policy specification and acknowledge it."
_ACK_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "ack"],
    "properties": {
        "schema_version": {"type": "string"},
        "ack": {"type": "string"},
    },
}
_POLICY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "source"],
    "properties": {
        "schema_version": {
            "type": "string",
            "const": "mforge.native.python_policy_response.v1",
        },
        "source": {"type": "string"},
    },
}
_PANEL = (
    DevelopmentCaseV1("case-00", 8, 101, 17, 1, 64, (4, 8)),
    DevelopmentCaseV1("case-01", 8, 103, 19, 1, 64, (4, 8)),
)


def _source(label: str, *, action: str = "add_edge") -> str:
    selector = "non_edges_legal" if action == "add_edge" else "edges_removable"
    variable = "candidate"
    return (
        "def propose(ctx, graph, api, seed):\n"
        f"    candidates = api.{selector}()\n"
        f"    {variable} = api.pick(candidates, seed, \"{label}\")\n"
        f"    if not {variable}:\n"
        "        return api.no_plan(\"NO_MATCH\")\n"
        f"    api.{action}({variable})\n"
        "    return api.emit()\n"
    )


def _envelope(source: str) -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.python_policy_response.v1",
            "source": source,
        },
        separators=(",", ":"),
    )


def _fraction(value: int, denominator: int = 10) -> dict[str, int]:
    return {"numerator": value, "denominator": denominator}


def _evaluation(
    *,
    source: str,
    case: DevelopmentCaseV1,
    accepted: bool,
) -> dict[str, JsonValue]:
    method = "remove_edge" if "remove_edge" in source else "add_edge"
    digest = hashlib.sha256(f"{source}:{case.case_id}".encode()).hexdigest()
    initial = 10 + case.graph_seed % 3
    terminal = initial - int(accepted)
    return {
        "behavior_identity": {
            "behavior_signature": digest,
            "probe_manifest_sha256": "a" * 64,
            "protocol_id": "fixture",
        },
        "scientific_result": {
            "status": "COMPLETE",
            "fitness_interval": {
                "lower": _fraction(7 if accepted else 4),
                "upper": _fraction(7 if accepted else 4),
            },
            "semantic_trace_hash": digest,
            "initial_counterexample": None,
            "initial_evidence": {
                "components": [
                    {
                        "forbidden_length": 4,
                        "lower_bound": initial,
                        "upper_bound": initial,
                    },
                    {
                        "forbidden_length": 8,
                        "lower_bound": 3,
                        "upper_bound": 3,
                    },
                ]
            },
            "terminal_evidence": {
                "components": [
                    {
                        "forbidden_length": 4,
                        "lower_bound": terminal,
                        "upper_bound": terminal,
                    },
                    {
                        "forbidden_length": 8,
                        "lower_bound": 3,
                        "upper_bound": 3,
                    },
                ]
            },
            "steps": [
                {
                    "outcome": "rewrite",
                    "accepted": accepted,
                    "no_plan_reason": None,
                    "counterexample": None,
                    "interpreter_trace": [
                        {"method": "non_edges_legal", "ordinal": 0},
                        {"method": "pick", "ordinal": 1},
                        {"method": method, "ordinal": 2},
                        {"method": "emit", "ordinal": 3},
                    ],
                }
            ],
        },
    }


class _Evaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        self.calls.append((candidate_id, case.case_id))
        accepted = int(candidate_id[1:5]) == 0 or candidate_id.endswith(
            ("00", "02", "04", "06")
        )
        return _evaluation(source=source, case=case, accepted=accepted)


class _Provider:
    model = "fixture-model"
    effort = "high"

    def __init__(
        self,
        durable: dict[str, M5ProviderResultV1] | None = None,
        *,
        duplicate_child: bool = False,
    ) -> None:
        self.durable = durable if durable is not None else {}
        self.duplicate_child = duplicate_child
        self.calls: list[tuple[str, int, str, str]] = []
        self.closed = False
        self.anchor = M5ProviderContextV1(
            "thread-anchor",
            "turn-anchor",
            "/opaque/anchor",
            ("turn-anchor",),
        )

    def ensure_specification_anchor(
        self,
        *,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, artifact_dir
        return M5ProviderResultV1(
            response_text=json.dumps(
                {
                    "schema_version": "mforge.native.python_m5_specification_ack.v1",
                    "ack": "specification-retained",
                }
            ),
            context=self.anchor,
            usage={
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
                "totalTokens": 0,
                "final": True,
                "partial": False,
            },
            duration_ms=0,
            warnings=0,
        )

    def _result(
        self,
        *,
        kind: str,
        generation: int,
        slot: str,
        prompt: str,
        idempotency_key: str,
        history: tuple[str, ...],
    ) -> M5ProviderResultV1:
        if idempotency_key in self.durable:
            return self.durable[idempotency_key]
        index = int(slot[-2:])
        if self.duplicate_child and generation == 1 and slot == "slot-00":
            source = _source("g0-slot-00")
        else:
            source = _source(
                f"g{generation}-{slot}",
                action="remove_edge" if index % 2 else "add_edge",
            )
        turn = f"turn-g{generation}-{slot}-{kind}"
        context = M5ProviderContextV1(
            thread_id=f"thread-g{generation}-{slot}",
            turn_id=turn,
            thread_path=f"/opaque/g{generation}/{slot}",
            included_turn_ids=history + (turn,),
        )
        result = M5ProviderResultV1(
            response_text=_envelope(source),
            context=context,
            usage={
                "inputTokens": 10,
                "cachedInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 10,
                "reasoningOutputTokens": 0,
                "totalTokens": 20,
                "final": True,
                "partial": False,
            },
            duration_ms=1,
            warnings=0,
        )
        self.calls.append((kind, generation, slot, prompt))
        self.durable[idempotency_key] = result
        return result

    def generate_root(
        self,
        *,
        anchor: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del system_prompt, output_schema, artifact_dir
        return self._result(
            kind="root",
            generation=generation,
            slot=slot,
            prompt=prompt,
            idempotency_key=idempotency_key,
            history=anchor.included_turn_ids,
        )

    def generate_child(
        self,
        *,
        parent: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del system_prompt, output_schema, artifact_dir
        return self._result(
            kind="child",
            generation=generation,
            slot=slot,
            prompt=prompt,
            idempotency_key=idempotency_key,
            history=parent.included_turn_ids,
        )

    def repair(
        self,
        *,
        previous: M5ProviderResultV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del system_prompt, output_schema, artifact_dir
        return self._result(
            kind="repair",
            generation=generation,
            slot=slot,
            prompt=prompt,
            idempotency_key=idempotency_key,
            history=previous.context.included_turn_ids,
        )

    def close(self) -> None:
        self.closed = True


def _run(
    root: Path,
    provider: _Provider,
    evaluator: _Evaluator,
    *,
    boundary_hook: Any = None,
) -> dict[str, Any]:
    return run_m5_search(
        provider=provider,
        evaluator=evaluator,
        workspace=root,
        panel=_PANEL,
        system_prompt=_SYSTEM,
        specification_prompt=_SPEC,
        specification_ack_schema=_ACK_SCHEMA,
        policy_schema=_POLICY_SCHEMA,
        boundary_hook=boundary_hook,
    )


def test_generation_manifest_is_exact_8_then_4_children_4_roots() -> None:
    generation_zero = build_generation_manifest(generation=0, panel=_PANEL)
    assert [item.kind for item in generation_zero.slots] == ["root"] * 8
    previous = [
        {
            "candidate_id": f"g0000-slot-{index:02d}",
            "status": "evaluated",
            "duplicate_of": None,
            "program_hash": f"{index + 1:064x}",
            "behavior_signature": f"{index + 101:064x}",
            "behavior_profile": {
                "fitness_interval": {
                    "lower": _fraction(index + 1),
                    "upper": _fraction(index + 1),
                }
            },
        }
        for index in range(8)
    ]
    generation_one = build_generation_manifest(
        generation=1,
        panel=_PANEL,
        previous_candidates=previous,
    )
    assert [item.kind for item in generation_one.slots] == [
        "child",
        "child",
        "child",
        "child",
        "root",
        "root",
        "root",
        "root",
    ]
    assert all(
        item.parent_candidate_id is not None
        for item in generation_one.slots[:4]
    )
    assert all(
        item.parent_candidate_id is None
        for item in generation_one.slots[4:]
    )


def test_m5_is_standalone_and_does_not_route_through_dsl_or_preview() -> None:
    source = inspect.getsource(search_module) + inspect.getsource(
        search_provider_module
    )
    assert "invoke_program(" not in source
    assert "evaluate_serial_program(" not in source
    assert "native_v3.interpreter" not in source
    assert "experiment run" not in source


def test_no_valid_parent_fallback_is_frozen_as_eight_roots() -> None:
    manifest = build_generation_manifest(
        generation=1,
        panel=_PANEL,
        previous_candidates=[
            {
                "candidate_id": "invalid",
                "status": "contract_invalid",
                "duplicate_of": None,
            }
        ],
    )
    assert manifest.all_root_fallback is True
    assert manifest.fallback_reason == "no_valid_evaluated_parent"
    assert [item.kind for item in manifest.slots] == ["root"] * 8


def test_parent_selection_is_completion_order_invariant_and_repeats() -> None:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "status": "evaluated",
            "duplicate_of": None,
            "program_hash": f"{index + 1:064x}",
            "behavior_signature": f"{index + 11:064x}",
            "behavior_profile": {
                "fitness_interval": {
                    "lower": _fraction(9 - index),
                    "upper": _fraction(9 - index),
                }
            },
        }
        for index in range(3)
    ]
    selected = select_parent_candidates(candidates)
    assert selected == select_parent_candidates(tuple(reversed(candidates)))
    assert len(selected) == 4
    assert selected[-1] in selected[:-1]


def test_parent_diversity_uses_measured_behavior_not_digest_distance() -> None:
    candidates: list[dict[str, Any]] = []
    for index, actions in enumerate(
        ({"add_edge": 1}, {"add_edge": 1}, {"remove_edge": 10})
    ):
        candidates.append(
            {
                "candidate_id": f"candidate-{index}",
                "status": "evaluated",
                "duplicate_of": None,
                "program_hash": f"{index + 1:064x}",
                "behavior_signature": f"{100 - index:064x}",
                "behavior_profile": {
                    "fitness_interval": {
                        "lower": _fraction(5),
                        "upper": _fraction(5),
                    },
                    "action_frequencies": actions,
                },
            }
        )
    selected = select_parent_candidates(candidates)
    assert selected[:2] == ("candidate-0", "candidate-2")


def test_search_memory_is_bounded_source_free_and_active_parent_null() -> None:
    profile = aggregate_behavior(
        [_evaluation(source=_source("memory"), case=_PANEL[0], accepted=True)]
    )
    memory = build_search_memory(
        [
            {
                "candidate_id": "candidate-00",
                "parent_candidate_id": None,
                "generation": 0,
                "slot": "slot-00",
                "status": "evaluated",
                "program_hash": "a" * 64,
                "behavior_signature": profile["behavior_signature"],
                "behavior_profile": profile,
                "control_flow": {"if_count": 1, "for_count": 0},
            }
        ]
    )
    model = memory["model_projection"]
    assert isinstance(model, Mapping)
    assert model["active_parent"] is None
    text = json.dumps(model, sort_keys=True)
    assert "source" not in text
    assert "program_hash" not in text
    assert "behavior_signature" not in text
    assert len(json.dumps(memory, sort_keys=True).encode()) < 16 * 1024
    prompt = build_root_prompt(memory)
    assert _source("memory") not in prompt


def test_search_memory_exact_bounds_and_canonical_order() -> None:
    successful_profile = aggregate_behavior(
        [_evaluation(source=_source("memory"), case=_PANEL[0], accepted=True)]
    )
    tested_profile = aggregate_behavior(
        [_evaluation(source=_source("memory"), case=_PANEL[0], accepted=False)]
    )
    candidates = [
        {
            "candidate_id": f"g{index // 8:04d}-slot-{index % 8:02d}",
            "parent_candidate_id": None,
            "generation": index // 8,
            "slot": f"slot-{index % 8:02d}",
            "status": "evaluated",
            "program_hash": f"{index + 1:064x}",
            "behavior_signature": f"{index + 1001:064x}",
            "behavior_profile": (
                successful_profile if index % 2 == 0 else tested_profile
            ),
            "control_flow": {"if_count": 1, "for_count": 0},
        }
        for index in range(80)
    ]
    memory = build_search_memory(tuple(reversed(candidates)))
    canonical = build_search_memory(candidates)
    assert memory["sha256"] == canonical["sha256"]
    assert len(memory["seen_program_hashes"]) == 64
    assert len(memory["seen_behavior_signatures"]) == 64
    assert len(memory["active_lineages"]) == 16
    assert len(memory["validated_archive_ids"]) == 16
    projection = memory["model_projection"]
    assert isinstance(projection, Mapping)
    assert len(projection["successful_patterns"]) == 8
    assert len(projection["tested_patterns"]) == 8
    assert (
        len(
            json.dumps(
                memory,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        <= 16 * 1024
    )


def test_child_prompt_resends_exact_parent_and_only_host_feedback() -> None:
    source = _source("parent")
    profile = aggregate_behavior(
        [_evaluation(source=source, case=_PANEL[0], accepted=False)]
    )
    prompt = build_child_prompt(parent_source=source, parent_profile=profile)
    assert source in prompt
    assert "fitness_interval" in prompt
    assert "action_frequencies" in prompt
    assert str(profile["behavior_signature"]) not in prompt
    assert "program_hash" not in prompt


def test_two_generation_search_reports_exact_population_lineage_and_profiles(
    tmp_path: Path,
) -> None:
    provider = _Provider(duplicate_child=True)
    evaluator = _Evaluator()
    report = _run(tmp_path / "run", provider, evaluator)
    assert report["generation_allocations"] == {
        "0": {"children": 0, "roots": 8},
        "1": {"children": 4, "roots": 4},
    }
    assert report["candidate_count"] == 16
    assert len(evaluator.calls) == 32
    assert report["duplicates"]
    assert report["usage"]["totalTokens"] == 320
    assert report["provider_turns"] == 17
    assert report["specification_anchor_turns"] == 1
    assert report["candidate_program_turns"] == 16
    assert report["generation_manifest_hashes"].keys() == {"0", "1"}
    assert all(report["acceptance_checks"].values())
    lineage = report["lineage"]
    assert all(item["parent_candidate_id"] is None for item in lineage[:8])
    assert all(item["parent_candidate_id"] is not None for item in lineage[8:12])
    assert all(item["parent_candidate_id"] is None for item in lineage[12:])
    child_calls = [item for item in provider.calls if item[0] == "child"]
    assert len(child_calls) == 4
    assert all("Exact parent source" in item[3] for item in child_calls)
    root_calls = [item for item in provider.calls if item[0] == "root"]
    assert len(root_calls) == 12
    assert all("Exact parent source" not in item[3] for item in root_calls)
    assert all(
        profile is not None
        for profile in report["behavior_profiles"].values()
    )


def test_crash_resume_reuses_provider_turns_and_evaluations_exactly(
    tmp_path: Path,
) -> None:
    baseline_boundaries: list[str] = []
    baseline = _run(
        tmp_path / "baseline",
        _Provider(),
        _Evaluator(),
        boundary_hook=baseline_boundaries.append,
    )
    assert len(baseline_boundaries) == len(set(baseline_boundaries))
    for index, boundary in enumerate(baseline_boundaries):
        crash_root = tmp_path / f"crash-{index:03d}"
        durable: dict[str, M5ProviderResultV1] = {}
        tripped = False
        evaluator = _Evaluator()

        def crash(name: str, expected: str = boundary) -> None:
            nonlocal tripped
            if not tripped and name == expected:
                tripped = True
                raise RuntimeError(f"crash:{expected}")

        with pytest.raises(RuntimeError, match="crash:"):
            _run(crash_root, _Provider(durable), evaluator, boundary_hook=crash)
        assert tripped
        resumed = _run(crash_root, _Provider(durable), evaluator)
        comparable_keys = (
            "stop_reason",
            "generation_count",
            "candidate_count",
            "generation_allocations",
            "generation_manifest_hashes",
            "search_memory_hashes",
            "lineage",
            "provider_order",
            "evaluation_order",
            "behavior_profiles",
            "duplicates",
            "usage",
            "equal_panel_hash",
        )
        assert {key: resumed[key] for key in comparable_keys} == {
            key: baseline[key] for key in comparable_keys
        }
        assert len(durable) == 16
        assert len(evaluator.calls) == 32
        assert read_json(crash_root / "m5-report.json.gz") == resumed


class _RepairProvider(_Provider):
    def _result(
        self,
        *,
        kind: str,
        generation: int,
        slot: str,
        prompt: str,
        idempotency_key: str,
        history: tuple[str, ...],
    ) -> M5ProviderResultV1:
        result = super()._result(
            kind=kind,
            generation=generation,
            slot=slot,
            prompt=prompt,
            idempotency_key=idempotency_key,
            history=history,
        )
        if (
            generation == 0
            and slot in {"slot-00", "slot-01"}
            and (kind != "repair" or slot == "slot-01")
        ):
            return M5ProviderResultV1(
                response_text=_envelope("import os\n"),
                context=result.context,
                usage=result.usage,
                duration_ms=result.duration_ms,
                warnings=result.warnings,
            )
        return result


def test_repair_and_exhausted_invalid_response_consume_planned_slots(
    tmp_path: Path,
) -> None:
    provider = _RepairProvider()
    evaluator = _Evaluator()
    report = _run(tmp_path / "repair", provider, evaluator)
    candidates = {
        item["candidate_id"]: item
        for item in (
            read_json(path)
            for path in sorted(
                (tmp_path / "repair").glob(
                    "generations/generation-*/slot-*/candidate.json.gz"
                )
            )
        )
    }
    assert report["candidate_count"] == 16
    assert report["repair_turns"] == 2
    assert candidates["g0000-slot-00"]["status"] == "evaluated"
    assert candidates["g0000-slot-00"]["repairs"] == 1
    assert candidates["g0000-slot-01"]["status"] == "contract_invalid"
    assert candidates["g0000-slot-01"]["repairs"] == 1
    assert len(evaluator.calls) == 30


class _ProgramFailureEvaluator(_Evaluator):
    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        value = dict(
            super().evaluate(
                source=source,
                case=case,
                candidate_id=candidate_id,
            )
        )
        if candidate_id == "g0000-slot-00":
            scientific_raw = value["scientific_result"]
            assert isinstance(scientific_raw, Mapping)
            scientific = dict(scientific_raw)
            scientific["status"] = "PROGRAM_FAILURE"
            scientific["fitness_interval"] = {
                "lower": _fraction(0),
                "upper": _fraction(0),
            }
            scientific["steps"] = [
                {
                    "outcome": "failure",
                    "accepted": False,
                    "no_plan_reason": None,
                    "counterexample": None,
                    "interpreter_trace": [],
                }
            ]
            value["scientific_result"] = scientific
        return value


def test_program_failure_consumes_full_equal_panel_budget(tmp_path: Path) -> None:
    root = tmp_path / "program-failure"
    report = _run(root, _Provider(), _ProgramFailureEvaluator())
    candidate = read_json(
        root
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "candidate.json.gz"
    )
    assert candidate["status"] == "evaluated"
    assert candidate["evaluation_case_count"] == len(_PANEL)
    assert candidate["behavior_profile"]["program_failure_count"] == len(_PANEL)
    assert report["equal_development_budget"][
        "all_evaluated_candidates_complete"
    ] is True


class _FailProvider(_Provider):
    def generate_root(self, **kwargs: Any) -> M5ProviderResultV1:
        del kwargs
        raise RuntimeError("fixture provider failed")


class _FailFirstGenerationOneProvider(_Provider):
    def generate_child(self, **kwargs: Any) -> M5ProviderResultV1:
        if kwargs["generation"] == 1 and kwargs["slot"] == "slot-00":
            raise RuntimeError("fixture generation-one provider failed")
        return super().generate_child(**kwargs)


def test_provider_failure_consumes_slot_and_stops_as_infrastructure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider-failure"
    provider = _FailProvider()
    with pytest.raises(M5InfrastructureError, match="provider failed"):
        _run(root, provider, _Evaluator())
    candidate = read_json(
        root
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "candidate.json.gz"
    )
    assert candidate["status"] == "provider_failed"
    assert read_json(root / "m5-stop.json.gz")["status"] == "infrastructure_failure"
    assert provider.closed is False


def test_resume_skips_one_terminal_provider_failure_and_runs_seven_pending(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider-failure-resume"
    initial_provider = _FailFirstGenerationOneProvider()
    initial_evaluator = _Evaluator()
    with pytest.raises(M5InfrastructureError, match="provider failed"):
        _run(root, initial_provider, initial_evaluator)
    generation_zero_paths = sorted(
        (root / "generations" / "generation-0000").glob(
            "slot-*/candidate.json.gz"
        )
    )
    generation_zero_bytes = {
        path.relative_to(root): path.read_bytes() for path in generation_zero_paths
    }
    manifest_path = (
        root / "generations" / "generation-0001" / "manifest.json.gz"
    )
    memory_path = (
        root / "generations" / "generation-0001" / "search-memory.json.gz"
    )
    manifest_before = manifest_path.read_bytes()
    memory_before = memory_path.read_bytes()
    failed_path = (
        root
        / "generations"
        / "generation-0001"
        / "slot-00"
        / "candidate.json.gz"
    )
    assert read_json(failed_path)["status"] == "provider_failed"

    resumed_provider = _Provider()
    resumed_evaluator = _Evaluator()
    report = _run(root, resumed_provider, resumed_evaluator)

    assert len(resumed_provider.calls) == 7
    assert [(kind, slot) for kind, _, slot, _ in resumed_provider.calls] == [
        ("child", "slot-01"),
        ("child", "slot-02"),
        ("child", "slot-03"),
        ("root", "slot-04"),
        ("root", "slot-05"),
        ("root", "slot-06"),
        ("root", "slot-07"),
    ]
    assert len(resumed_evaluator.calls) == 7 * len(_PANEL)
    assert read_json(failed_path)["status"] == "provider_failed"
    assert manifest_path.read_bytes() == manifest_before
    assert memory_path.read_bytes() == memory_before
    assert {
        path.relative_to(root): path.read_bytes() for path in generation_zero_paths
    } == generation_zero_bytes
    assert report["candidate_status_counts"]["provider_failed"] == 1
    assert report["generation_status_counts"]["1"] == {
        "evaluated": 7,
        "provider_failed": 1,
    }
    assert report["generation_allocations"]["1"] == {
        "children": 4,
        "roots": 4,
    }


def test_terminal_missing_slot_is_consumed_and_not_submitted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-resume"
    with pytest.raises(M5InfrastructureError, match="provider failed"):
        _run(root, _FailFirstGenerationOneProvider(), _Evaluator())
    missing_path = (
        root
        / "generations"
        / "generation-0001"
        / "slot-00"
        / "candidate.json.gz"
    )
    missing = read_json(missing_path)
    assert isinstance(missing, dict)
    missing["status"] = "missing"
    write_json(missing_path, missing)

    provider = _Provider()
    report = _run(root, provider, _Evaluator())
    assert len(provider.calls) == 7
    assert report["generation_status_counts"]["1"] == {
        "evaluated": 7,
        "missing": 1,
    }
    assert {
        "evaluated",
        "contract_invalid",
        "duplicate",
        "provider_failed",
        "missing",
        "evaluation_infrastructure_failure",
    } == M5_TERMINAL_CANDIDATE_STATUSES


def test_operator_stop_is_distinct_and_resumable(tmp_path: Path) -> None:
    root = tmp_path / "operator-stop"
    provider = _Provider()
    with pytest.raises(M5OperatorStop):
        run_m5_search(
            provider=provider,
            evaluator=_Evaluator(),
            workspace=root,
            panel=_PANEL,
            system_prompt=_SYSTEM,
            specification_prompt=_SPEC,
            specification_ack_schema=_ACK_SCHEMA,
            policy_schema=_POLICY_SCHEMA,
            operator_stop=lambda: True,
        )
    stop = read_json(root / "m5-stop.json.gz")
    assert stop["status"] == "operator_stop"
    assert stop["resumable"] is True
    assert provider.closed is False
    resumed = _run(root, provider, _Evaluator())
    assert resumed["candidate_count"] == 16


def test_offline_resume_revalidates_retained_scientific_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corrupt-evidence"
    _run(root, _Provider(), _Evaluator())
    evaluation_path = (
        root
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "evaluations"
        / "case-00.json.gz"
    )
    evaluation = read_json(evaluation_path)
    assert isinstance(evaluation, dict)
    scientific = evaluation["scientific_result"]
    assert isinstance(scientific, dict)
    scientific["semantic_trace_hash"] = "0" * 64
    write_json(evaluation_path, evaluation)
    with pytest.raises(M5InfrastructureError, match="behavior evidence changed"):
        _run(root, _Provider(), _Evaluator())


def test_heuristic_zero_does_not_stop_without_exact_verified(tmp_path: Path) -> None:
    report = _run(tmp_path / "run", _Provider(), _Evaluator())
    assert report["stop_reason"] == "generation_budget"
    assert report["exact_verified"] is False


class _ExactVerifiedEvaluator(_Evaluator):
    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        value = dict(
            super().evaluate(
                source=source,
                case=case,
                candidate_id=candidate_id,
            )
        )
        scientific_raw = value["scientific_result"]
        assert isinstance(scientific_raw, Mapping)
        scientific = dict(scientific_raw)
        scientific["initial_counterexample"] = {"decision": "stop_verified"}
        value["scientific_result"] = scientific
        return value


def test_only_exact_verified_counterexample_stops_search(tmp_path: Path) -> None:
    report = _run(
        tmp_path / "exact-verified",
        _Provider(),
        _ExactVerifiedEvaluator(),
    )
    assert report["stop_reason"] == "exact_verified_counterexample"
    assert report["exact_verified"] is True
    assert report["candidate_count"] == 1
    assert report["generation_count"] == 1
