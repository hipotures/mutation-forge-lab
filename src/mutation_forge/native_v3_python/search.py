"""Bounded two-generation search over ordinary-Python mutation policies."""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3.canonical import canonical_json_bytes, domain_hash
from mutation_forge.native_v3.serial_evaluator import SerialEvaluationStatus

from .validation import (
    ACTION_METHODS,
    SELECTOR_METHODS,
    normalize_source_newlines,
    validate_python_policy_response,
)

M5_SEARCH_PROTOCOL_ID = "mforge.native.python_search.v1"
M5_MANIFEST_PROTOCOL_ID = "mforge.native.python_generation_manifest.v1"
M5_CANDIDATE_PROTOCOL_ID = "mforge.native.python_search_candidate.v1"
M5_SEARCH_MEMORY_PROTOCOL_ID = "mforge.native.python_search_memory.v1"
M5_MODEL_MEMORY_PROTOCOL_ID = "mforge.native.python_search_memory.model.v1"
M5_REPORT_PROTOCOL_ID = "mforge.native.python_m5_search_report.v1"
M5_TERMINAL_CANDIDATE_STATUSES = frozenset(
    {
        "evaluated",
        "contract_invalid",
        "duplicate",
        "provider_failed",
        "missing",
        "evaluation_infrastructure_failure",
    }
)

POPULATION_SIZE = 8
CHILD_SLOTS = 4
ROOT_SLOTS = 4
MAX_GENERATIONS = 2
MAX_REPAIRS = 1
MAX_MEMORY_IDENTITIES = 64
MAX_MEMORY_PATTERNS = 8
MAX_ACTIVE_LINEAGES = 16
MAX_ARCHIVE_IDS = 16
MAX_MEMORY_BYTES = 16 * 1024
MAX_TRACE_METHODS = 24

_MANIFEST_DOMAIN = b"mforge-native-v3-python-m5-manifest-v1\0"
_PANEL_DOMAIN = b"mforge-native-v3-python-m5-panel-v1\0"
_BEHAVIOR_DOMAIN = b"mforge-native-v3-python-m5-behavior-v1\0"
_MEMORY_DOMAIN = b"mforge-native-v3-python-m5-memory-v1\0"
_REQUEST_DOMAIN = b"mforge-native-v3-python-m5-request-v1\0"
_HEX_DIGEST = re.compile(r"(?i)\b[0-9a-f]{64}\b")
_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_PRIVATE_PATH = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)")


def _evaluation_telemetry_summary(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, JsonValue]:
    totals: dict[str, int | float] = {
        "starts": 0,
        "rotations": 0,
        "failures": 0,
        "timeouts": 0,
        "maximum_rss_kib": 0,
        "policy_invocations": 0,
        "graph_score_attempts": 0,
        "unique_graph_scores": 0,
        "sandbox_wall_seconds": 0.0,
        "selector_wall_seconds": 0.0,
        "action_wall_seconds": 0.0,
        "scoring_wall_seconds": 0.0,
    }

    def nonnegative_int(value: object) -> int:
        return (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            else 0
        )

    for evaluation in evaluations:
        runtime_profile = evaluation.get("runtime_profile")
        if isinstance(runtime_profile, Mapping):
            for key in (
                "sandbox_wall_seconds",
                "selector_wall_seconds",
                "action_wall_seconds",
            ):
                raw = runtime_profile.get(key)
                if (
                    isinstance(raw, int | float)
                    and not isinstance(raw, bool)
                    and raw >= 0
                ):
                    totals[key] += float(raw)
        worker = evaluation.get("worker_telemetry")
        if isinstance(worker, Mapping):
            totals["starts"] += 1
            totals["rotations"] += nonnegative_int(worker.get("rotations"))
            totals["failures"] += nonnegative_int(worker.get("failures"))
            totals["maximum_rss_kib"] = max(
                totals["maximum_rss_kib"],
                nonnegative_int(worker.get("worker_rss_kib")),
            )
        scientific = evaluation.get("scientific_result")
        if not isinstance(scientific, Mapping):
            continue
        steps = scientific.get("steps")
        if isinstance(steps, Sequence) and not isinstance(steps, str | bytes):
            totals["policy_invocations"] += len(steps)
        totals["graph_score_attempts"] += nonnegative_int(
            scientific.get("score_attempts")
        )
        totals["unique_graph_scores"] += nonnegative_int(
            scientific.get("unique_graph_scores")
        )
        stack: list[object] = [scientific]
        while stack:
            item = stack.pop()
            if isinstance(item, Mapping):
                wall_time_ns = item.get("wall_time_ns")
                if (
                    isinstance(wall_time_ns, int)
                    and not isinstance(wall_time_ns, bool)
                    and wall_time_ns >= 0
                    and isinstance(item.get("forbidden_length"), int)
                ):
                    totals["scoring_wall_seconds"] += (
                        wall_time_ns / 1_000_000_000
                    )
                stack.extend(item.values())
            elif isinstance(item, Sequence) and not isinstance(
                item, str | bytes
            ):
                stack.extend(item)
        failure = scientific.get("failure")
        if (
            isinstance(failure, Mapping)
            and failure.get("code") == "PROPOSE_TIMEOUT"
        ):
            totals["timeouts"] += 1
    return cast(dict[str, JsonValue], totals)


class M5SearchError(RuntimeError):
    """The bounded search contract was violated."""


class M5InfrastructureError(M5SearchError):
    """Provider, sandbox, backend, scorer, or persistence failed."""


class M5OperatorStop(M5SearchError):
    """The operator requested a resumable stop between immutable boundaries."""


@dataclass(frozen=True, slots=True)
class DevelopmentCaseV1:
    """One immutable development-panel episode."""

    case_id: str
    order: int
    graph_seed: int
    policy_seed: int
    horizon: int
    witness_cap: int
    forbidden_lengths: tuple[int, ...]
    graph_mode: str = "unrestricted_min_degree_3"

    def __post_init__(self) -> None:
        if (
            not self.case_id
            or self.order < 1
            or self.graph_seed < 0
            or self.policy_seed < 0
            or self.horizon < 0
            or self.witness_cap < 1
            or not self.forbidden_lengths
            or not self.graph_mode
            or tuple(sorted(set(self.forbidden_lengths))) != self.forbidden_lengths
        ):
            raise ValueError("invalid development case")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "case_id": self.case_id,
            "order": self.order,
            "graph_seed": self.graph_seed,
            "policy_seed": self.policy_seed,
            "horizon": self.horizon,
            "witness_cap": self.witness_cap,
            "forbidden_lengths": list(self.forbidden_lengths),
            "graph_mode": self.graph_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DevelopmentCaseV1:
        forbidden = value.get("forbidden_lengths")
        if not isinstance(forbidden, Sequence) or isinstance(forbidden, str | bytes):
            raise ValueError("development case forbidden_lengths are malformed")
        return cls(
            case_id=str(value["case_id"]),
            order=int(value["order"]),
            graph_seed=int(value["graph_seed"]),
            policy_seed=int(value["policy_seed"]),
            horizon=int(value["horizon"]),
            witness_cap=int(value["witness_cap"]),
            forbidden_lengths=tuple(int(item) for item in forbidden),
            graph_mode=str(value["graph_mode"]),
        )


@dataclass(frozen=True, slots=True)
class M5ProviderContextV1:
    """Host-only durable provider boundary for one generated program."""

    thread_id: str
    turn_id: str
    thread_path: str | None
    included_turn_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.thread_id
            or not self.turn_id
            or not self.included_turn_ids
            or self.included_turn_ids[-1] != self.turn_id
            or len(set(self.included_turn_ids)) != len(self.included_turn_ids)
        ):
            raise ValueError("invalid durable provider context")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "thread_path": self.thread_path,
            "included_turn_ids": list(self.included_turn_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> M5ProviderContextV1:
        return cls(
            thread_id=str(value["thread_id"]),
            turn_id=str(value["turn_id"]),
            thread_path=(
                str(value["thread_path"]) if value.get("thread_path") is not None else None
            ),
            included_turn_ids=tuple(
                str(item) for item in cast(Sequence[object], value["included_turn_ids"])
            ),
        )


@dataclass(frozen=True, slots=True)
class M5ProviderResultV1:
    """One terminal model turn, including exact fork evidence."""

    response_text: str
    context: M5ProviderContextV1
    usage: Mapping[str, JsonValue]
    duration_ms: int
    warnings: int
    attempts: int = 1

    def __post_init__(self) -> None:
        if (
            not self.response_text
            or self.duration_ms < 0
            or self.warnings < 0
            or self.attempts != 1
            or self.usage.get("final") is not True
            or self.usage.get("partial") is not False
        ):
            raise ValueError("invalid terminal provider result")
        for key in (
            "inputTokens",
            "cachedInputTokens",
            "cacheWriteInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        ):
            value = self.usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid terminal provider usage {key}")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "response_text": self.response_text,
            "context": self.context.as_dict(),
            "usage": dict(self.usage),
            "duration_ms": self.duration_ms,
            "warnings": self.warnings,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> M5ProviderResultV1:
        context = value.get("context")
        usage = value.get("usage")
        if not isinstance(context, Mapping) or not isinstance(usage, Mapping):
            raise ValueError("invalid retained provider result")
        return cls(
            response_text=str(value["response_text"]),
            context=M5ProviderContextV1.from_dict(context),
            usage=cast(Mapping[str, JsonValue], dict(usage)),
            duration_ms=int(value.get("duration_ms", 0)),
            warnings=int(value.get("warnings", 0)),
            attempts=int(value.get("attempts", 1)),
        )


class M5SearchProvider(Protocol):
    """Persistent one-program provider used only by the standalone M5 runner."""

    model: str
    effort: str

    def ensure_specification_anchor(
        self,
        *,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        artifact_dir: Path,
    ) -> M5ProviderResultV1: ...

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
    ) -> M5ProviderResultV1: ...

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
    ) -> M5ProviderResultV1: ...

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
    ) -> M5ProviderResultV1: ...

    def close(self) -> None: ...


class M10SearchProvider(M5SearchProvider, Protocol):
    """Concurrent provider pool for one frozen complete generation."""

    provider_concurrency: int

    def prepare_generation(
        self,
        *,
        snapshot: Mapping[str, Any],
        anchor: M5ProviderContextV1,
        artifact_dir: Path,
    ) -> None: ...

    def primary_lane(self, *, generation: int, slot: str) -> int: ...

    def await_primary_slot(self, *, generation: int, slot: str) -> None: ...

    def release_primary_slot(self, *, generation: int, slot: str) -> None: ...


class M5ScientificEvaluator(Protocol):
    """Evaluate one source on exactly one immutable development case."""

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]: ...


@dataclass(frozen=True, slots=True)
class SlotPlanV1:
    slot: str
    kind: Literal["child", "root"]
    parent_candidate_id: str | None
    panel_hash: str
    request_key: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "slot": self.slot,
            "kind": self.kind,
            "parent_candidate_id": self.parent_candidate_id,
            "panel_hash": self.panel_hash,
            "request_key": self.request_key,
        }


@dataclass(frozen=True, slots=True)
class GenerationManifestV1:
    generation: int
    slots: tuple[SlotPlanV1, ...]
    all_root_fallback: bool
    fallback_reason: str | None
    panel: tuple[DevelopmentCaseV1, ...]
    protocol_id: str = M5_MANIFEST_PROTOCOL_ID

    def __post_init__(self) -> None:
        if len(self.slots) != POPULATION_SIZE:
            raise ValueError("generation manifest must contain exactly eight slots")
        if tuple(item.slot for item in self.slots) != tuple(
            f"slot-{index:02d}" for index in range(POPULATION_SIZE)
        ):
            raise ValueError("generation slots are not canonical")
        expected_panel_hash = panel_hash(self.panel)
        if any(item.panel_hash != expected_panel_hash for item in self.slots):
            raise ValueError("generation slots do not share one panel")
        kinds = tuple(item.kind for item in self.slots)
        if self.generation == 0 and kinds != ("root",) * POPULATION_SIZE:
            raise ValueError("generation zero must contain eight roots")
        if (
            self.generation > 0
            and not self.all_root_fallback
            and kinds != ("child",) * CHILD_SLOTS + ("root",) * ROOT_SLOTS
        ):
            raise ValueError("later generation must contain four children then four roots")
        if self.all_root_fallback and kinds != ("root",) * POPULATION_SIZE:
            raise ValueError("fallback generation must contain eight roots")

    def _payload(self) -> dict[str, JsonValue]:
        return {
            "protocol_id": self.protocol_id,
            "generation": self.generation,
            "slots": [item.as_dict() for item in self.slots],
            "all_root_fallback": self.all_root_fallback,
            "fallback_reason": self.fallback_reason,
            "panel": [item.as_dict() for item in self.panel],
        }

    def as_dict(self) -> dict[str, JsonValue]:
        return {**self._payload(), "sha256": self.sha256}

    @property
    def sha256(self) -> str:
        return domain_hash(_MANIFEST_DOMAIN, canonical_json_bytes(self._payload()))


def panel_hash(panel: Sequence[DevelopmentCaseV1]) -> str:
    return domain_hash(
        _PANEL_DOMAIN,
        canonical_json_bytes([item.as_dict() for item in panel]),
    )


def _fraction(value: Mapping[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _scientific_result(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = evaluation.get("scientific_result")
    if not isinstance(value, Mapping):
        raise M5InfrastructureError("evaluation omitted scientific_result")
    return value


def _event_method(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    method = value.get("method")
    if isinstance(method, str):
        return method
    kind = value.get("kind")
    return kind if isinstance(kind, str) else None


def _component_map(evidence: object) -> dict[int, Mapping[str, Any]]:
    if not isinstance(evidence, Mapping):
        return {}
    raw = evidence.get("components")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return {}
    result: dict[int, Mapping[str, Any]] = {}
    for item in raw:
        if isinstance(item, Mapping) and isinstance(item.get("forbidden_length"), int):
            result[int(item["forbidden_length"])] = item
    return result


def _initial_only_fitness_lower(
    evaluations: Sequence[Mapping[str, Any]],
) -> Fraction | None:
    """Project the frozen panel's no-policy lower bound from in-memory results."""

    values_by_order: dict[int, list[Fraction]] = {}
    for evaluation in evaluations:
        scientific = _scientific_result(evaluation)
        config = scientific.get("config")
        trajectory = scientific.get("utility_trajectory")
        if (
            not isinstance(config, Mapping)
            or not isinstance(config.get("order"), int)
            or isinstance(config.get("order"), bool)
            or not isinstance(trajectory, Sequence)
            or isinstance(trajectory, str | bytes)
            or not trajectory
            or not isinstance(trajectory[0], Mapping)
        ):
            return None
        initial = cast(Mapping[str, Any], trajectory[0]).get("lower")
        if not isinstance(initial, Mapping):
            return None
        values_by_order.setdefault(int(config["order"]), []).append(_fraction(initial))
    if not values_by_order:
        return None
    per_order = [
        sum(values_by_order[order], Fraction()) / len(values_by_order[order])
        for order in sorted(values_by_order)
    ]
    return sum(per_order, Fraction()) / len(per_order)


def aggregate_behavior(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, JsonValue]:
    """Aggregate only measured panel behavior and scientific outcomes."""

    selector_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    no_plan_reasons: Counter[str] = Counter()
    behavior_signatures: list[str] = []
    semantic_hashes: list[str] = []
    witness_totals: dict[int, list[int]] = {}
    accepted = rejected = illegal = failures = propose_calls = api_calls = 0
    lower = Fraction()
    upper = Fraction()
    verified = False
    verifier_submissions = 0
    verifier_records = 0
    external_activity: Counter[str] = Counter()
    statuses: list[str] = []

    for evaluation in evaluations:
        raw_external = evaluation.get("external_activity")
        if isinstance(raw_external, Mapping):
            for key, value in raw_external.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    external_activity[str(key)] += value
        behavior = evaluation.get("behavior_identity")
        if isinstance(behavior, Mapping):
            signature = behavior.get("behavior_signature")
            if isinstance(signature, str):
                behavior_signatures.append(signature)
        scientific = _scientific_result(evaluation)
        status = str(scientific.get("status", ""))
        statuses.append(status)
        fitness = scientific.get("fitness_interval")
        if not isinstance(fitness, Mapping):
            raise M5InfrastructureError("evaluation omitted fitness interval")
        raw_lower = fitness.get("lower")
        raw_upper = fitness.get("upper")
        if not isinstance(raw_lower, Mapping) or not isinstance(raw_upper, Mapping):
            raise M5InfrastructureError("evaluation fitness interval is malformed")
        lower += _fraction(raw_lower)
        upper += _fraction(raw_upper)
        semantic = scientific.get("semantic_trace_hash")
        if isinstance(semantic, str):
            semantic_hashes.append(semantic)
        if status == SerialEvaluationStatus.PROGRAM_FAILURE.value:
            failures += 1
        steps = scientific.get("steps", ())
        if not isinstance(steps, Sequence) or isinstance(steps, str | bytes):
            raise M5InfrastructureError("evaluation steps are malformed")
        propose_calls += len(steps)
        for step in steps:
            if not isinstance(step, Mapping):
                raise M5InfrastructureError("evaluation step is malformed")
            outcome = str(step.get("outcome", "UNKNOWN"))
            outcome_counts[outcome] += 1
            if step.get("accepted") is True:
                accepted += 1
            elif outcome == "rewrite":
                rejected += 1
            reason = step.get("no_plan_reason")
            if isinstance(reason, str):
                no_plan_reasons[reason] += 1
                if reason == "ILLEGAL_FINAL_STATE":
                    illegal += 1
            events = step.get("interpreter_trace", ())
            if not isinstance(events, Sequence) or isinstance(events, str | bytes):
                raise M5InfrastructureError("semantic API trace is malformed")
            api_calls += len(events)
            for event in events:
                method = _event_method(event)
                if method in SELECTOR_METHODS or method == "pick":
                    selector_counts[method] += 1
                if method in ACTION_METHODS - {"emit", "no_plan"}:
                    action_counts[method] += 1
            counterexample = step.get("counterexample")
            if isinstance(counterexample, Mapping):
                verifier_submissions += 1
                records = counterexample.get("records")
                verifier_records += (
                    len(records)
                    if isinstance(records, Sequence) and not isinstance(records, str | bytes)
                    else 1
                )
                verified |= str(counterexample.get("decision", "")).lower() == "stop_verified"
        initial_counterexample = scientific.get("initial_counterexample")
        if isinstance(initial_counterexample, Mapping):
            verifier_submissions += 1
            records = initial_counterexample.get("records")
            verifier_records += (
                len(records)
                if isinstance(records, Sequence) and not isinstance(records, str | bytes)
                else 1
            )
            verified |= str(initial_counterexample.get("decision", "")).lower() == "stop_verified"
        initial = _component_map(scientific.get("initial_evidence"))
        terminal = _component_map(scientific.get("terminal_evidence"))
        for length in sorted(initial.keys() | terminal.keys()):
            before = initial.get(length, {})
            after = terminal.get(length, {})
            values = witness_totals.setdefault(length, [0, 0, 0, 0])
            values[0] += int(before.get("lower_bound", 0))
            values[1] += int(before.get("upper_bound", 0))
            values[2] += int(after.get("lower_bound", 0))
            values[3] += int(after.get("upper_bound", 0))

    count = len(evaluations)
    if count == 0:
        raise M5InfrastructureError("candidate received no development evaluations")
    initial_only_lower = _initial_only_fitness_lower(evaluations)
    aggregate_signature = domain_hash(
        _BEHAVIOR_DOMAIN,
        canonical_json_bytes(
            {
                "episode_behavior_signatures": behavior_signatures,
                "semantic_trace_hashes": semantic_hashes,
                "selectors": dict(sorted(selector_counts.items())),
                "actions": dict(sorted(action_counts.items())),
                "outcomes": dict(sorted(outcome_counts.items())),
            }
        ),
    )
    return {
        "behavior_signature": aggregate_signature,
        "fitness_interval": cast(
            JsonValue,
            {
                "lower": _fraction_dict(lower / count),
                "upper": _fraction_dict(upper / count),
            },
        ),
        **(
            {
                "initial_only_fitness_lower": cast(
                    JsonValue,
                    _fraction_dict(initial_only_lower),
                )
            }
            if initial_only_lower is not None
            else {}
        ),
        "episode_statuses": cast(JsonValue, statuses),
        "propose_calls": propose_calls,
        "rewrite_plan_count": outcome_counts["rewrite"],
        "no_plan_count": outcome_counts["no_plan"],
        "program_failure_count": failures,
        "accepted_rewrite_count": accepted,
        "rejected_rewrite_count": rejected,
        "illegal_final_state_count": illegal,
        "selector_frequencies": dict(sorted(selector_counts.items())),
        "action_frequencies": dict(sorted(action_counts.items())),
        "no_plan_reasons": dict(sorted(no_plan_reasons.items())),
        "mean_api_calls": {
            "numerator": api_calls,
            "denominator": max(1, propose_calls),
        },
        "witness_deltas": {
            str(length): {
                "initial_lower": values[0],
                "initial_upper": values[1],
                "terminal_lower": values[2],
                "terminal_upper": values[3],
                "proved_reduction": values[0] - values[3],
            }
            for length, values in sorted(witness_totals.items())
        },
        "semantic_trace_summary": cast(
            JsonValue,
            {
                "methods": [
                    [name, item_count]
                    for name, item_count in (
                        sorted(
                            (selector_counts + action_counts).items(),
                            key=lambda item: (-item[1], item[0]),
                        )[:MAX_TRACE_METHODS]
                    )
                ],
                "semantic_trace_hashes": semantic_hashes,
            },
        ),
        "exact_verified": verified,
        "exact_verifier_submissions": verifier_submissions,
        "exact_verifier_records": verifier_records,
        "scientific_external_activity": dict(sorted(external_activity.items())),
    }


def python_control_flow_summary(source: str) -> dict[str, JsonValue]:
    tree = ast.parse(source, mode="exec", type_comments=True)
    counts = Counter(type(node).__name__ for node in ast.walk(tree))
    helpers = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name != "propose"
    ]
    return {
        "if_count": counts["If"],
        "for_count": counts["For"],
        "helper_count": len(helpers),
        "return_count": counts["Return"],
        "boolean_operation_count": counts["BoolOp"],
    }


def _feedback(profile: Mapping[str, Any]) -> dict[str, JsonValue]:
    fitness = cast(Mapping[str, JsonValue], profile["fitness_interval"])
    actions = cast(Mapping[str, JsonValue], profile["action_frequencies"])
    selectors = cast(Mapping[str, JsonValue], profile["selector_frequencies"])
    strengths: list[str] = []
    weaknesses: list[str] = []
    if int(profile["accepted_rewrite_count"]) > 0:
        strengths.append("Produced strictly accepted rewrites on the development panel.")
    if int(profile["program_failure_count"]) == 0:
        strengths.append("Completed without program failures.")
    if int(profile["no_plan_count"]) > int(profile["accepted_rewrite_count"]):
        weaknesses.append("NoPlan outcomes outnumbered accepted rewrites.")
    if int(profile["illegal_final_state_count"]) > 0:
        weaknesses.append("Some proposals failed final graph validation.")
    if len(actions) <= 1 and actions:
        weaknesses.append("Observed action behavior collapsed to one action family.")
    if not strengths:
        strengths.append("The host observed no proved development-panel strength.")
    if not weaknesses:
        weaknesses.append("Improve the conservative fitness lower bound without failures.")
    trace = profile["semantic_trace_summary"]
    if not isinstance(trace, Mapping):
        raise M5SearchError("parent semantic trace summary is malformed")
    return {
        "fitness_interval": dict(fitness),
        "witness_deltas": cast(JsonValue, profile["witness_deltas"]),
        "accepted_rewrites": int(profile["accepted_rewrite_count"]),
        "rejected_rewrites": int(profile["rejected_rewrite_count"]),
        "no_plan": int(profile["no_plan_count"]),
        "program_failures": int(profile["program_failure_count"]),
        "illegal_final_states": int(profile["illegal_final_state_count"]),
        "action_frequencies": dict(actions),
        "selector_frequencies": dict(selectors),
        "mean_api_calls": cast(JsonValue, profile["mean_api_calls"]),
        "semantic_api_summary": {
            "methods": cast(JsonValue, trace.get("methods", [])),
        },
        "observed_strengths": cast(JsonValue, strengths),
        "observed_weaknesses": cast(JsonValue, weaknesses),
    }


def _fitness_key(candidate: Mapping[str, Any]) -> tuple[Fraction, Fraction, str]:
    profile = candidate.get("behavior_profile")
    if not isinstance(profile, Mapping):
        return Fraction(), Fraction(), str(candidate.get("program_hash", ""))
    interval = profile.get("fitness_interval")
    if not isinstance(interval, Mapping):
        return Fraction(), Fraction(), str(candidate.get("program_hash", ""))
    lower = interval.get("lower")
    upper = interval.get("upper")
    if not isinstance(lower, Mapping) or not isinstance(upper, Mapping):
        return Fraction(), Fraction(), str(candidate.get("program_hash", ""))
    return _fraction(lower), _fraction(upper), str(candidate.get("program_hash", ""))


def behavior_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Fraction:
    """Return deterministic distance over measured behavior, never digest bits."""

    scalar_fields = (
        "propose_calls",
        "rewrite_plan_count",
        "no_plan_count",
        "program_failure_count",
        "accepted_rewrite_count",
        "rejected_rewrite_count",
        "illegal_final_state_count",
    )
    distance = sum(
        (
            abs(Fraction(int(left.get(field, 0))) - Fraction(int(right.get(field, 0))))
            for field in scalar_fields
        ),
        Fraction(),
    )
    for field in (
        "selector_frequencies",
        "action_frequencies",
        "no_plan_reasons",
    ):
        left_counts = left.get(field, {})
        right_counts = right.get(field, {})
        if not isinstance(left_counts, Mapping) or not isinstance(right_counts, Mapping):
            raise M5SearchError(f"behavior profile {field} is malformed")
        distance += sum(
            (
                abs(
                    Fraction(int(left_counts.get(key, 0))) - Fraction(int(right_counts.get(key, 0)))
                )
                for key in sorted(
                    {str(item) for item in left_counts} | {str(item) for item in right_counts}
                )
            ),
            Fraction(),
        )
    left_witnesses = left.get("witness_deltas", {})
    right_witnesses = right.get("witness_deltas", {})
    if not isinstance(left_witnesses, Mapping) or not isinstance(right_witnesses, Mapping):
        raise M5SearchError("behavior witness deltas are malformed")
    for length in sorted(
        {str(item) for item in left_witnesses} | {str(item) for item in right_witnesses}
    ):
        left_delta = left_witnesses.get(length, {})
        right_delta = right_witnesses.get(length, {})
        if not isinstance(left_delta, Mapping) or not isinstance(right_delta, Mapping):
            raise M5SearchError("behavior witness delta is malformed")
        for bound in (
            "initial_lower",
            "initial_upper",
            "terminal_lower",
            "terminal_upper",
            "proved_reduction",
        ):
            distance += abs(
                Fraction(int(left_delta.get(bound, 0))) - Fraction(int(right_delta.get(bound, 0)))
            )
    return distance


def select_parent_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Choose four deterministic fitness/diversity parents, repeating if needed."""

    eligible = [
        item
        for item in candidates
        if item.get("status") == "evaluated"
        and item.get("duplicate_of") is None
        and isinstance(item.get("behavior_signature"), str)
        and isinstance(item.get("behavior_profile"), Mapping)
        and int(cast(Mapping[str, Any], item["behavior_profile"]).get("program_failure_count", 0))
        == 0
    ]
    eligible.sort(
        key=lambda item: (
            -_fitness_key(item)[0],
            -_fitness_key(item)[1],
            _fitness_key(item)[2],
            str(item["candidate_id"]),
        )
    )
    if not eligible:
        return ()
    selected: list[Mapping[str, Any]] = [eligible[0]]
    remaining = eligible[1:]
    while remaining and len(selected) < CHILD_SLOTS:
        chosen = min(
            remaining,
            key=lambda item: (
                -min(
                    behavior_distance(
                        cast(Mapping[str, Any], item["behavior_profile"]),
                        cast(Mapping[str, Any], parent["behavior_profile"]),
                    )
                    for parent in selected
                ),
                -_fitness_key(item)[0],
                -_fitness_key(item)[1],
                _fitness_key(item)[2],
                str(item["candidate_id"]),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    parent_ids = [str(item["candidate_id"]) for item in selected]
    return tuple(parent_ids[index % len(parent_ids)] for index in range(CHILD_SLOTS))


def _request_key(
    generation: int,
    slot: str,
    kind: str,
    parent_candidate_id: str | None,
    panel_digest: str,
) -> str:
    return domain_hash(
        _REQUEST_DOMAIN,
        canonical_json_bytes(
            {
                "generation": generation,
                "slot": slot,
                "kind": kind,
                "parent_candidate_id": parent_candidate_id,
                "panel_hash": panel_digest,
            }
        ),
    )


def build_generation_manifest(
    *,
    generation: int,
    panel: tuple[DevelopmentCaseV1, ...],
    previous_candidates: Sequence[Mapping[str, Any]] = (),
) -> GenerationManifestV1:
    digest = panel_hash(panel)
    if generation == 0:
        parents: tuple[str, ...] = ()
        kinds = ("root",) * POPULATION_SIZE
        fallback = False
        reason = None
    else:
        parents = select_parent_candidates(previous_candidates)
        fallback = not parents
        reason = "no_valid_evaluated_parent" if fallback else None
        kinds = (
            ("root",) * POPULATION_SIZE
            if fallback
            else ("child",) * CHILD_SLOTS + ("root",) * ROOT_SLOTS
        )
    slots: list[SlotPlanV1] = []
    for index, kind in enumerate(kinds):
        slot = f"slot-{index:02d}"
        parent = parents[index] if kind == "child" else None
        slots.append(
            SlotPlanV1(
                slot=slot,
                kind=cast(Literal["child", "root"], kind),
                parent_candidate_id=parent,
                panel_hash=digest,
                request_key=_request_key(generation, slot, kind, parent, digest),
            )
        )
    return GenerationManifestV1(
        generation=generation,
        slots=tuple(slots),
        all_root_fallback=fallback,
        fallback_reason=reason,
        panel=panel,
    )


def _memory_pattern(candidate: Mapping[str, Any]) -> dict[str, JsonValue]:
    profile = cast(Mapping[str, Any], candidate["behavior_profile"])
    witness_deltas = cast(Mapping[str, Any], profile["witness_deltas"])
    return {
        "outcome": ("successful" if int(profile["accepted_rewrite_count"]) > 0 else "tested"),
        "fitness_interval": cast(JsonValue, profile["fitness_interval"]),
        "selectors": cast(
            JsonValue,
            sorted(
                str(key)
                for key, value in cast(Mapping[str, Any], profile["selector_frequencies"]).items()
                if int(value) > 0
            ),
        ),
        "actions": cast(
            JsonValue,
            sorted(
                str(key)
                for key, value in cast(Mapping[str, Any], profile["action_frequencies"]).items()
                if int(value) > 0
            ),
        ),
        "proved_witness_reductions": {
            str(length): int(cast(Mapping[str, Any], delta).get("proved_reduction", 0))
            for length, delta in sorted(witness_deltas.items())
            if isinstance(delta, Mapping)
        },
        "outcome_counts": {
            "accepted": int(profile["accepted_rewrite_count"]),
            "rejected": int(profile["rejected_rewrite_count"]),
            "no_plan": int(profile["no_plan_count"]),
            "program_failure": int(profile["program_failure_count"]),
            "illegal_final_state": int(profile["illegal_final_state_count"]),
        },
        "control_flow": cast(JsonValue, candidate["control_flow"]),
    }


def build_search_memory(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, JsonValue]:
    """Build bounded host memory and its source-free model projection."""

    retained = sorted(
        [
            item
            for item in candidates
            if item.get("status") == "evaluated"
            and isinstance(item.get("program_hash"), str)
            and isinstance(item.get("behavior_signature"), str)
            and isinstance(item.get("behavior_profile"), Mapping)
        ],
        key=lambda item: (
            int(item.get("generation", -1)),
            str(item.get("slot", "")),
            str(item.get("candidate_id", "")),
        ),
    )
    identity_retained = retained[-MAX_MEMORY_IDENTITIES:]
    lineage_retained = retained[-MAX_ACTIVE_LINEAGES:]
    successful = [
        _memory_pattern(item)
        for item in identity_retained
        if int(cast(Mapping[str, Any], item["behavior_profile"])["accepted_rewrite_count"]) > 0
    ][-MAX_MEMORY_PATTERNS:]
    tested = [
        _memory_pattern(item)
        for item in identity_retained
        if int(cast(Mapping[str, Any], item["behavior_profile"])["accepted_rewrite_count"]) == 0
    ][-MAX_MEMORY_PATTERNS:]
    host: dict[str, JsonValue] = {
        "protocol_id": M5_SEARCH_MEMORY_PROTOCOL_ID,
        "seen_program_hashes": [str(item["program_hash"]) for item in identity_retained],
        "seen_behavior_signatures": [str(item["behavior_signature"]) for item in identity_retained],
        "active_lineages": [
            {
                "candidate_id": str(item["candidate_id"]),
                "parent_candidate_id": (
                    str(item["parent_candidate_id"])
                    if item.get("parent_candidate_id") is not None
                    else None
                ),
            }
            for item in lineage_retained
        ],
        "validated_archive_ids": [
            str(item["candidate_id"]) for item in retained[-MAX_ARCHIVE_IDS:]
        ],
        "active_parent": None,
    }
    model: dict[str, JsonValue] = {
        "protocol_id": M5_MODEL_MEMORY_PROTOCOL_ID,
        "successful_patterns": cast(JsonValue, successful),
        "tested_patterns": cast(JsonValue, tested),
        "active_parent": None,
    }
    host["model_projection"] = model
    while True:
        host.pop("sha256", None)
        host["sha256"] = domain_hash(_MEMORY_DOMAIN, canonical_json_bytes(host))
        if len(canonical_json_bytes(host)) <= MAX_MEMORY_BYTES:
            break
        program_hashes = cast(list[JsonValue], host["seen_program_hashes"])
        behavior_signatures = cast(list[JsonValue], host["seen_behavior_signatures"])
        active_lineages = cast(list[JsonValue], host["active_lineages"])
        archive_ids = cast(list[JsonValue], host["validated_archive_ids"])
        successful_patterns = cast(list[JsonValue], model["successful_patterns"])
        tested_patterns = cast(list[JsonValue], model["tested_patterns"])
        if len(program_hashes) > MAX_ARCHIVE_IDS:
            program_hashes.pop(0)
            behavior_signatures.pop(0)
        elif successful_patterns or tested_patterns:
            target = (
                successful_patterns
                if len(successful_patterns) >= len(tested_patterns)
                else tested_patterns
            )
            target.pop(0)
        elif active_lineages:
            active_lineages.pop(0)
        elif archive_ids:
            archive_ids.pop(0)
        elif program_hashes:
            program_hashes.pop(0)
            behavior_signatures.pop(0)
        else:
            raise M5SearchError("minimal Search Memory exceeds 16 KiB")
    model_text = json.dumps(model, sort_keys=True, separators=(",", ":"))
    if any(
        token in model_text
        for token in ("program_hash", "behavior_signature", "candidate_id", "source")
    ):
        raise M5SearchError("model Search Memory contains a host identity or source")
    return host


def build_root_prompt(memory: Mapping[str, Any]) -> str:
    model = memory.get("model_projection")
    if not isinstance(model, Mapping) or model.get("active_parent") is not None:
        raise M5SearchError("fresh root requires source-free memory and active_parent=null")
    return (
        "Generate one complete fresh ordinary-Python policy using the retained "
        "specification. Return the exact two-field response envelope. The host "
        "will evaluate actual behavior; do not describe or predict fitness.\n\n"
        "Host-derived source-free Search Memory:\n"
        + json.dumps(model, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )


def build_child_prompt(
    *,
    parent_source: str,
    parent_profile: Mapping[str, Any],
) -> str:
    feedback = _feedback(parent_profile)
    return (
        "Return one complete replacement ordinary-Python policy that mutates the "
        "exact parent below. Do not return a patch or diff. Make measured behavior, "
        "not source appearance, address the host-derived weaknesses.\n\n"
        "Exact parent source:\n```python\n"
        + parent_source
        + "\n```\n\nHost-derived development feedback:\n"
        + json.dumps(feedback, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )


def _assert_model_prompt_hygiene(prompt: str) -> None:
    lowered = prompt.lower()
    forbidden_terms = (
        "program_hash",
        "behavior_signature",
        "source_sha256",
        "canonical_ast_sha256",
        "thread_id",
        "turn_id",
        "workspace path",
        "provider state",
        "held-out",
    )
    if (
        _HEX_DIGEST.search(prompt)
        or _UUID.search(prompt)
        or _PRIVATE_PATH.search(prompt)
        or any(term in lowered for term in forbidden_terms)
        or "priority(ctx, proposal)" in prompt
    ):
        raise M5SearchError(
            "model-facing prompt contains a prohibited host identity or legacy ranker contract"
        )


def build_repair_prompt(diagnostics: Sequence[Mapping[str, Any]]) -> str:
    bounded = [
        {
            "code": str(item.get("code", "INVALID"))[:128],
            "path": str(item.get("path", "/"))[:512],
            "message": str(item.get("message", "invalid response"))[:512],
            "line": item.get("line") if isinstance(item.get("line"), int) else None,
            "column": (item.get("column") if isinstance(item.get("column"), int) else None),
        }
        for item in diagnostics[:32]
    ]
    return (
        "Return one complete replacement ordinary-Python policy. Repair only the "
        "following host validator diagnostics; do not return a patch or diff:\n"
        + json.dumps(bounded, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )


def _candidate_id(generation: int, slot: str) -> str:
    return f"g{generation:04d}-{slot}"


def _candidate_path(root: Path, generation: int, slot: str) -> Path:
    return root / "generations" / f"generation-{generation:04d}" / slot


def _load_mapping(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise M5InfrastructureError(f"artifact is not a JSON object: {path}")
    return dict(value)


def _verify_retained_candidate(
    *,
    root: Path,
    path: Path,
    panel: tuple[DevelopmentCaseV1, ...],
    slot_plan: SlotPlanV1,
    search_memory_sha256: str,
) -> dict[str, Any]:
    candidate = _load_mapping(path)
    expected_slot = path.parent.name
    expected_generation = int(path.parent.parent.name.removeprefix("generation-"))
    expected_id = _candidate_id(expected_generation, expected_slot)
    if (
        candidate.get("protocol_id") != M5_CANDIDATE_PROTOCOL_ID
        or candidate.get("candidate_id") != expected_id
        or candidate.get("generation") != expected_generation
        or candidate.get("slot") != expected_slot
        or candidate.get("kind") != slot_plan.kind
        or candidate.get("parent_candidate_id") != slot_plan.parent_candidate_id
        or candidate.get("panel_hash") != panel_hash(panel)
        or candidate.get("panel_case_ids") != [item.case_id for item in panel]
        or candidate.get("search_memory_sha256") != search_memory_sha256
    ):
        raise M5InfrastructureError(f"retained candidate identity changed: {path}")
    candidates = _all_candidates(root)
    by_id = {str(item.get("candidate_id")): item for item in candidates}
    parent = (
        by_id.get(slot_plan.parent_candidate_id)
        if slot_plan.parent_candidate_id is not None
        else None
    )
    if slot_plan.kind == "child" and parent is None:
        raise M5InfrastructureError("retained child parent is unavailable")
    expected_parent_program_hash = parent.get("program_hash") if parent is not None else None
    expected_parent_behavior_signature = (
        parent.get("behavior_signature") if parent is not None else None
    )
    if (
        candidate.get("parent_program_hash") != expected_parent_program_hash
        or candidate.get("parent_behavior_signature") != expected_parent_behavior_signature
    ):
        raise M5InfrastructureError("retained parent identity changed")
    raw_attempts = candidate.get("provider_attempts")
    if not isinstance(raw_attempts, Sequence) or isinstance(raw_attempts, str | bytes):
        raise M5InfrastructureError(f"retained provider attempts are malformed: {path}")
    attempts = [
        M5ProviderResultV1.from_dict(item) for item in raw_attempts if isinstance(item, Mapping)
    ]
    if len(attempts) != len(raw_attempts):
        raise M5InfrastructureError(f"retained provider attempt changed: {path}")
    status = candidate.get("status")
    if status == "missing":
        if (
            attempts
            or candidate.get("evaluation_case_count") != 0
            or candidate.get("provider_context") is not None
        ):
            raise M5InfrastructureError(
                "terminal missing slot gained provider or scientific evidence"
            )
        return candidate
    if status == "provider_failed":
        if candidate.get("evaluation_case_count") != 0:
            raise M5InfrastructureError("provider failure gained scientific evidence")
        return candidate
    if not attempts:
        raise M5InfrastructureError(f"retained candidate omitted provider evidence: {path}")
    context = candidate.get("provider_context")
    if (
        not isinstance(context, Mapping)
        or M5ProviderContextV1.from_dict(context) != attempts[-1].context
    ):
        raise M5InfrastructureError(f"retained provider context changed: {path}")
    retained_validation = candidate.get("validation")
    replayed_validation = validate_python_policy_response(attempts[-1].response_text)
    if (
        not isinstance(retained_validation, Mapping)
        or dict(retained_validation) != replayed_validation.as_dict()
    ):
        raise M5InfrastructureError(f"retained validation changed: {path}")
    if status == "contract_invalid":
        if replayed_validation.valid or candidate.get("evaluation_case_count") != 0:
            raise M5InfrastructureError("invalid candidate gained scientific evidence")
        return candidate
    if status not in {
        "evaluated",
        "duplicate",
        "evaluation_infrastructure_failure",
    }:
        raise M5InfrastructureError(f"unknown retained candidate status: {status}")
    if (
        not replayed_validation.valid
        or replayed_validation.response is None
        or replayed_validation.identity is None
        or replayed_validation.identity.program_hash is None
    ):
        raise M5InfrastructureError("retained valid candidate no longer validates")
    source = normalize_source_newlines(replayed_validation.response.source)
    source_path_value = candidate.get("source_path")
    if not isinstance(source_path_value, str):
        raise M5InfrastructureError("retained valid candidate omitted its source path")
    source_path = (root / source_path_value).resolve()
    if (
        not source_path.is_relative_to(root.resolve())
        or not source_path.is_file()
        or source_path.read_text(encoding="utf-8") != source
        or candidate.get("source") != source
        or candidate.get("program_hash") != replayed_validation.identity.program_hash
        or candidate.get("source_sha256") != replayed_validation.identity.source_sha256
        or candidate.get("canonical_ast_sha256")
        != replayed_validation.identity.canonical_ast_sha256
    ):
        raise M5InfrastructureError(f"retained source identity changed: {path}")
    if status == "evaluation_infrastructure_failure":
        evaluation_count = candidate.get("evaluation_case_count")
        if (
            isinstance(evaluation_count, bool)
            or not isinstance(evaluation_count, int)
            or not 0 <= evaluation_count < len(panel)
            or not isinstance(candidate.get("failure"), Mapping)
            or candidate.get("behavior_profile") is not None
            or candidate.get("behavior_signature") is not None
            or candidate.get("duplicate_of") is not None
        ):
            raise M5InfrastructureError("retained evaluation infrastructure failure changed")
        for index, case in enumerate(panel):
            evaluation_path = path.parent / "evaluations" / f"{case.case_id}.json.gz"
            if evaluation_path.is_file() != (index < evaluation_count):
                raise M5InfrastructureError("retained partial evaluation boundary changed")
            if index < evaluation_count:
                _load_mapping(evaluation_path)
        return candidate
    evaluations = [
        _load_mapping(path.parent / "evaluations" / f"{case.case_id}.json.gz") for case in panel
    ]
    if candidate.get("evaluation_case_count") != len(evaluations):
        raise M5InfrastructureError("retained candidate evaluation budget changed")
    replayed_profile = aggregate_behavior(evaluations)
    if (
        canonical_json_bytes(candidate.get("behavior_profile"))
        != canonical_json_bytes(replayed_profile)
        or candidate.get("behavior_signature") != replayed_profile["behavior_signature"]
    ):
        raise M5InfrastructureError(f"retained behavior evidence changed: {path}")
    duplicate_of = candidate.get("duplicate_of")
    if status == "evaluated" and duplicate_of is not None:
        raise M5InfrastructureError("retained duplicate classification changed")
    if status == "duplicate":
        if not isinstance(duplicate_of, str):
            raise M5InfrastructureError("retained duplicate classification changed")
        candidate_key = (expected_generation, expected_slot)
        duplicate = next(
            (
                item
                for item in candidates
                if item.get("candidate_id") == duplicate_of
                and (
                    int(item.get("generation", -1)),
                    str(item.get("slot", "")),
                )
                < candidate_key
            ),
            None,
        )
        if duplicate is None or (
            duplicate.get("program_hash") != candidate.get("program_hash")
            and duplicate.get("behavior_signature") != candidate.get("behavior_signature")
        ):
            raise M5InfrastructureError("retained duplicate target changed")
    return candidate


def _write_exclusive_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        retained = _load_mapping(path)
        if canonical_json_bytes(retained) != canonical_json_bytes(value):
            raise M5InfrastructureError(f"immutable artifact changed: {path}")
        return
    write_json(path, value, exclusive=True)


def _write_source_exclusive_or_verify(path: Path, source: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != source:
            raise M5InfrastructureError(f"program identity collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != source:
                raise M5InfrastructureError(f"program identity collision: {path}") from None
    finally:
        temporary_path.unlink(missing_ok=True)


def _json_nonnegative_int(value: JsonValue | None, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M5InfrastructureError(f"invalid retained provider {field}")
    return value


def _all_candidates(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.glob("generations/generation-*/slot-*/candidate.json.gz")):
        result.append(_load_mapping(path))
    return result


def _generation_candidates(root: Path, generation: int) -> list[dict[str, Any]]:
    directory = root / "generations" / f"generation-{generation:04d}"
    return [_load_mapping(path) for path in sorted(directory.glob("slot-*/candidate.json.gz"))]


def _seen_duplicates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    program_hash: str,
    behavior_signature: str,
) -> str | None:
    for candidate in candidates:
        if candidate.get("program_hash") == program_hash:
            return str(candidate["candidate_id"])
        if candidate.get("behavior_signature") == behavior_signature:
            return str(candidate["candidate_id"])
    return None


def _provider_context(candidate: Mapping[str, Any]) -> M5ProviderContextV1:
    value = candidate.get("provider_context")
    if not isinstance(value, Mapping):
        raise M5InfrastructureError("parent has no durable provider context")
    return M5ProviderContextV1.from_dict(value)


def _assert_provider_turn_boundary(
    result: M5ProviderResultV1,
    *,
    expected_history: tuple[str, ...],
    expected_thread_id: str | None = None,
) -> None:
    if result.context.included_turn_ids[:-1] != expected_history:
        raise M5InfrastructureError("provider result crossed its exact inclusive fork boundary")
    if expected_thread_id is not None and result.context.thread_id != expected_thread_id:
        raise M5InfrastructureError("provider repair changed thread")


def _usage_total(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "inputTokens",
        "cachedInputTokens",
        "cacheWriteInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    return {
        key: sum(
            int(cast(Mapping[str, Any], item.get("usage", {})).get(key, 0)) for item in candidates
        )
        for key in keys
    }


def _usage_with_anchor(
    candidates: Sequence[Mapping[str, Any]],
    anchor_usage: Mapping[str, JsonValue],
) -> dict[str, int]:
    candidate_usage = _usage_total(candidates)
    return {
        key: candidate_usage[key]
        + _json_nonnegative_int(anchor_usage.get(key), field=f"anchor usage {key}")
        for key in candidate_usage
    }


def _attempt_usage_total(
    attempts: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    keys = (
        "inputTokens",
        "cachedInputTokens",
        "cacheWriteInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    usage: dict[str, JsonValue] = {
        key: sum(
            int(cast(Mapping[str, Any], attempt.get("usage", {})).get(key, 0))
            for attempt in attempts
        )
        for key in keys
    }
    usage["final"] = all(
        cast(Mapping[str, Any], attempt.get("usage", {})).get("final") is True
        for attempt in attempts
    )
    usage["partial"] = any(
        cast(Mapping[str, Any], attempt.get("usage", {})).get("partial") is True
        for attempt in attempts
    )
    return usage


def run_m5_search(
    *,
    provider: M5SearchProvider,
    evaluator: M5ScientificEvaluator,
    workspace: str | Path,
    panel: tuple[DevelopmentCaseV1, ...],
    system_prompt: str,
    specification_prompt: str,
    specification_ack_schema: Mapping[str, Any],
    policy_schema: Mapping[str, Any],
    preview_active: bool = False,
    close_provider: bool = True,
    operator_stop: Callable[[], bool] | None = None,
    boundary_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run or resume the exact sequential M5 population contract."""

    if not panel:
        raise ValueError("development panel must not be empty")
    _assert_model_prompt_hygiene(system_prompt)
    _assert_model_prompt_hygiene(specification_prompt)
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json.gz"
    protocol = {
        "protocol_id": M5_SEARCH_PROTOCOL_ID,
        "population_size": POPULATION_SIZE,
        "generation_count": MAX_GENERATIONS,
        "generation_zero_roots": POPULATION_SIZE,
        "later_child_slots": CHILD_SLOTS,
        "later_root_slots": ROOT_SLOTS,
        "panel_hash": panel_hash(panel),
        "panel": [item.as_dict() for item in panel],
        "model": provider.model,
        "effort": provider.effort,
        "sequential": True,
        "safe_api_expanded": False,
        "preview_active": preview_active,
        "dsl_runtime_used": False,
    }
    _write_exclusive_or_verify(protocol_path, protocol)
    if boundary_hook is not None:
        boundary_hook("protocol_persisted")
    anchor_dir = root / "provider" / "specification-anchor"
    try:
        anchor_result = provider.ensure_specification_anchor(
            prompt=specification_prompt,
            system_prompt=system_prompt,
            output_schema=specification_ack_schema,
            artifact_dir=anchor_dir,
        )
        _assert_provider_turn_boundary(anchor_result, expected_history=())
        anchor = anchor_result.context
        _write_exclusive_or_verify(
            root / "anchor.json.gz",
            anchor_result.as_dict(),
        )
        if boundary_hook is not None:
            boundary_hook("anchor_persisted")
        exact_verified = False
        stopped_reason = "generation_budget"
        for generation in range(MAX_GENERATIONS):
            previous = _generation_candidates(root, generation - 1) if generation > 0 else []
            manifest = build_generation_manifest(
                generation=generation,
                panel=panel,
                previous_candidates=previous,
            )
            generation_dir = root / "generations" / f"generation-{generation:04d}"
            manifest_path = generation_dir / "manifest.json.gz"
            _write_exclusive_or_verify(manifest_path, manifest.as_dict())
            if boundary_hook is not None:
                boundary_hook(f"generation_{generation}_manifest")
            memory_path = generation_dir / "search-memory.json.gz"
            memory_candidates = [
                item
                for item in _all_candidates(root)
                if int(item.get("generation", -1)) < generation
            ]
            memory = build_search_memory(memory_candidates)
            _write_exclusive_or_verify(memory_path, memory)
            if boundary_hook is not None:
                boundary_hook(f"generation_{generation}_search_memory")
            prior_candidates = _all_candidates(root)
            by_id = {str(item["candidate_id"]): item for item in prior_candidates}
            for slot_plan in manifest.slots:
                slot_dir = _candidate_path(root, generation, slot_plan.slot)
                candidate_path = slot_dir / "candidate.json.gz"
                if candidate_path.exists():
                    _verify_retained_candidate(
                        root=root,
                        path=candidate_path,
                        panel=panel,
                        slot_plan=slot_plan,
                        search_memory_sha256=str(memory["sha256"]),
                    )
                    continue
                if operator_stop is not None and operator_stop():
                    stopped_reason = "operator_stop"
                    raise M5OperatorStop("operator stop requested")
                candidate_id = _candidate_id(generation, slot_plan.slot)
                prompt: str
                parent: Mapping[str, Any] | None = None
                try:
                    if slot_plan.kind == "root":
                        prompt = build_root_prompt(memory)
                        _assert_model_prompt_hygiene(prompt)
                        parent = None
                        provider_result = provider.generate_root(
                            anchor=anchor,
                            generation=generation,
                            slot=slot_plan.slot,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            output_schema=policy_schema,
                            idempotency_key=slot_plan.request_key,
                            artifact_dir=slot_dir / "provider-initial",
                        )
                        _assert_provider_turn_boundary(
                            provider_result,
                            expected_history=anchor.included_turn_ids,
                        )
                    else:
                        parent_id = slot_plan.parent_candidate_id
                        if parent_id is None or parent_id not in by_id:
                            raise M5InfrastructureError("frozen child parent is unavailable")
                        parent = by_id[parent_id]
                        parent_source = parent.get("source")
                        parent_profile = parent.get("behavior_profile")
                        if not isinstance(parent_source, str) or not isinstance(
                            parent_profile, Mapping
                        ):
                            raise M5InfrastructureError("selected parent evidence is incomplete")
                        prompt = build_child_prompt(
                            parent_source=parent_source,
                            parent_profile=parent_profile,
                        )
                        _assert_model_prompt_hygiene(prompt)
                        provider_result = provider.generate_child(
                            parent=_provider_context(parent),
                            generation=generation,
                            slot=slot_plan.slot,
                            prompt=prompt,
                            system_prompt=system_prompt,
                            output_schema=policy_schema,
                            idempotency_key=slot_plan.request_key,
                            artifact_dir=slot_dir / "provider-initial",
                        )
                        _assert_provider_turn_boundary(
                            provider_result,
                            expected_history=_provider_context(parent).included_turn_ids,
                        )
                except Exception as error:
                    failure = {
                        "protocol_id": M5_CANDIDATE_PROTOCOL_ID,
                        "candidate_id": candidate_id,
                        "generation": generation,
                        "slot": slot_plan.slot,
                        "kind": slot_plan.kind,
                        "parent_candidate_id": slot_plan.parent_candidate_id,
                        "parent_program_hash": (
                            parent.get("program_hash") if parent is not None else None
                        ),
                        "parent_behavior_signature": (
                            parent.get("behavior_signature") if parent is not None else None
                        ),
                        "panel_hash": slot_plan.panel_hash,
                        "panel_case_ids": [item.case_id for item in panel],
                        "search_memory_sha256": memory["sha256"],
                        "status": "provider_failed",
                        "provider_context": None,
                        "provider_attempts": [],
                        "repairs": 0,
                        "usage": {},
                        "program_hash": None,
                        "behavior_signature": None,
                        "behavior_profile": None,
                        "duplicate_of": None,
                        "evaluation_case_count": 0,
                        "failure": {
                            "type": type(error).__name__,
                            "message": str(error)[:1024],
                        },
                    }
                    _write_exclusive_or_verify(candidate_path, failure)
                    if boundary_hook is not None:
                        boundary_hook(f"{candidate_id}_committed")
                    raise M5InfrastructureError(
                        f"provider failed for {candidate_id}: {type(error).__name__}: {error}"
                    ) from error
                if boundary_hook is not None:
                    boundary_hook(f"{candidate_id}_provider")
                validation = validate_python_policy_response(provider_result.response_text)
                repairs = 0
                attempts = [provider_result.as_dict()]
                if not validation.valid:
                    diagnostics = [item.as_dict() for item in validation.diagnostics[:32]]
                    previous_result = provider_result
                    repair_prompt = build_repair_prompt(diagnostics)
                    _assert_model_prompt_hygiene(repair_prompt)
                    try:
                        provider_result = provider.repair(
                            previous=previous_result,
                            generation=generation,
                            slot=slot_plan.slot,
                            prompt=repair_prompt,
                            system_prompt=system_prompt,
                            output_schema=policy_schema,
                            idempotency_key=slot_plan.request_key + "-repair-01",
                            artifact_dir=slot_dir / "provider-repair-01",
                        )
                        _assert_provider_turn_boundary(
                            provider_result,
                            expected_history=(previous_result.context.included_turn_ids),
                            expected_thread_id=previous_result.context.thread_id,
                        )
                    except Exception as error:
                        failure = {
                            "protocol_id": M5_CANDIDATE_PROTOCOL_ID,
                            "candidate_id": candidate_id,
                            "generation": generation,
                            "slot": slot_plan.slot,
                            "kind": slot_plan.kind,
                            "parent_candidate_id": slot_plan.parent_candidate_id,
                            "parent_program_hash": (
                                parent.get("program_hash") if parent is not None else None
                            ),
                            "parent_behavior_signature": (
                                parent.get("behavior_signature") if parent is not None else None
                            ),
                            "panel_hash": slot_plan.panel_hash,
                            "panel_case_ids": [item.case_id for item in panel],
                            "search_memory_sha256": memory["sha256"],
                            "status": "provider_failed",
                            "provider_context": provider_result.context.as_dict(),
                            "provider_attempts": attempts,
                            "repairs": 1,
                            "usage": _attempt_usage_total(attempts),
                            "program_hash": None,
                            "behavior_signature": None,
                            "behavior_profile": None,
                            "duplicate_of": None,
                            "evaluation_case_count": 0,
                            "failure": {
                                "type": type(error).__name__,
                                "message": str(error)[:1024],
                            },
                        }
                        _write_exclusive_or_verify(candidate_path, failure)
                        if boundary_hook is not None:
                            boundary_hook(f"{candidate_id}_committed")
                        raise M5InfrastructureError(
                            f"provider repair failed for {candidate_id}: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    attempts.append(provider_result.as_dict())
                    validation = validate_python_policy_response(provider_result.response_text)
                    repairs = 1
                base: dict[str, Any] = {
                    "protocol_id": M5_CANDIDATE_PROTOCOL_ID,
                    "candidate_id": candidate_id,
                    "generation": generation,
                    "slot": slot_plan.slot,
                    "kind": slot_plan.kind,
                    "parent_candidate_id": slot_plan.parent_candidate_id,
                    "parent_program_hash": (
                        parent.get("program_hash") if parent is not None else None
                    ),
                    "parent_behavior_signature": (
                        parent.get("behavior_signature") if parent is not None else None
                    ),
                    "panel_hash": slot_plan.panel_hash,
                    "panel_case_ids": [item.case_id for item in panel],
                    "search_memory_sha256": memory["sha256"],
                    "provider_context": provider_result.context.as_dict(),
                    "provider_attempts": attempts,
                    "repairs": repairs,
                    "usage": _attempt_usage_total(attempts),
                    "duration_ms": sum(
                        _json_nonnegative_int(attempt.get("duration_ms"), field="duration_ms")
                        for attempt in attempts
                    ),
                    "warnings": sum(
                        _json_nonnegative_int(attempt.get("warnings"), field="warnings")
                        for attempt in attempts
                    ),
                    "request_prompt_bytes": len(prompt.encode("utf-8")),
                }
                if (
                    not validation.valid
                    or validation.response is None
                    or validation.identity is None
                    or validation.identity.program_hash is None
                ):
                    base.update(
                        {
                            "status": "contract_invalid",
                            "validation": validation.as_dict(),
                            "program_hash": None,
                            "behavior_signature": None,
                            "behavior_profile": None,
                            "duplicate_of": None,
                            "evaluation_case_count": 0,
                        }
                    )
                    _write_exclusive_or_verify(candidate_path, base)
                    if boundary_hook is not None:
                        boundary_hook(f"{candidate_id}_committed")
                    continue
                source = normalize_source_newlines(validation.response.source)
                program_hash = validation.identity.program_hash
                source_path = root / "sources" / f"{program_hash}.py"
                _write_source_exclusive_or_verify(source_path, source)
                if boundary_hook is not None:
                    boundary_hook(f"{candidate_id}_source_persisted")
                evaluation_payloads: list[dict[str, Any]] = []
                for case in panel:
                    evaluation_path = slot_dir / "evaluations" / f"{case.case_id}.json.gz"
                    if evaluation_path.exists():
                        evaluation = _load_mapping(evaluation_path)
                    else:
                        try:
                            evaluation = dict(
                                evaluator.evaluate(
                                    source=source,
                                    case=case,
                                    candidate_id=candidate_id,
                                )
                            )
                        except Exception as error:
                            base.update(
                                {
                                    "status": "evaluation_infrastructure_failure",
                                    "validation": validation.as_dict(),
                                    "source": source,
                                    "source_path": str(source_path.relative_to(root)),
                                    "source_sha256": validation.identity.source_sha256,
                                    "canonical_ast_sha256": (
                                        validation.identity.canonical_ast_sha256
                                    ),
                                    "program_hash": program_hash,
                                    "behavior_signature": None,
                                    "behavior_profile": None,
                                    "duplicate_of": None,
                                    "evaluation_case_count": len(evaluation_payloads),
                                    "evaluation_telemetry": _evaluation_telemetry_summary(
                                        evaluation_payloads
                                    ),
                                    "failure": {
                                        "type": type(error).__name__,
                                        "message": str(error)[:1024],
                                        "case_id": case.case_id,
                                    },
                                }
                            )
                            _write_exclusive_or_verify(candidate_path, base)
                            if boundary_hook is not None:
                                boundary_hook(f"{candidate_id}_committed")
                            raise M5InfrastructureError(
                                f"development evaluation failed for {candidate_id}/"
                                f"{case.case_id}: {type(error).__name__}: {error}"
                            ) from error
                        _write_exclusive_or_verify(evaluation_path, evaluation)
                    evaluation_payloads.append(evaluation)
                    if boundary_hook is not None:
                        boundary_hook(f"{candidate_id}_evaluation_{case.case_id}")
                behavior_profile = aggregate_behavior(evaluation_payloads)
                behavior_signature = str(behavior_profile["behavior_signature"])
                duplicate_of = _seen_duplicates(
                    _all_candidates(root),
                    program_hash=program_hash,
                    behavior_signature=behavior_signature,
                )
                base.update(
                    {
                        "status": "duplicate" if duplicate_of is not None else "evaluated",
                        "validation": validation.as_dict(),
                        "source": source,
                        "source_path": str(source_path.relative_to(root)),
                        "source_sha256": validation.identity.source_sha256,
                        "canonical_ast_sha256": (validation.identity.canonical_ast_sha256),
                        "program_hash": program_hash,
                        "behavior_signature": behavior_signature,
                        "behavior_profile": behavior_profile,
                        "control_flow": python_control_flow_summary(source),
                        "duplicate_of": duplicate_of,
                        "evaluation_case_count": len(evaluation_payloads),
                        "evaluation_telemetry": _evaluation_telemetry_summary(
                            evaluation_payloads
                        ),
                        "exact_verified": behavior_profile["exact_verified"],
                    }
                )
                _write_exclusive_or_verify(candidate_path, base)
                if boundary_hook is not None:
                    boundary_hook(f"{candidate_id}_committed")
                exact_verified |= behavior_profile["exact_verified"] is True
                if exact_verified:
                    stopped_reason = "exact_verified_counterexample"
                    break
            if exact_verified:
                break
        candidates = _all_candidates(root)
        generations = sorted({int(item["generation"]) for item in candidates})
        candidates_by_id = {str(item["candidate_id"]): item for item in candidates}
        child_mutation_proofs = [
            {
                "candidate_id": item["candidate_id"],
                "parent_candidate_id": item["parent_candidate_id"],
                "source_changed": (
                    isinstance(item.get("source"), str)
                    and isinstance(
                        candidates_by_id.get(str(item["parent_candidate_id"]), {}).get("source"),
                        str,
                    )
                    and item.get("source")
                    != candidates_by_id.get(str(item["parent_candidate_id"]), {}).get("source")
                ),
                "program_changed": (
                    isinstance(item.get("program_hash"), str)
                    and isinstance(
                        candidates_by_id.get(str(item["parent_candidate_id"]), {}).get(
                            "program_hash"
                        ),
                        str,
                    )
                    and item.get("program_hash")
                    != candidates_by_id.get(str(item["parent_candidate_id"]), {}).get(
                        "program_hash"
                    )
                ),
                "semantic_behavior_changed": (
                    isinstance(item.get("behavior_signature"), str)
                    and isinstance(
                        candidates_by_id.get(str(item["parent_candidate_id"]), {}).get(
                            "behavior_signature"
                        ),
                        str,
                    )
                    and item.get("behavior_signature")
                    != candidates_by_id.get(str(item["parent_candidate_id"]), {}).get(
                        "behavior_signature"
                    )
                ),
            }
            for item in candidates
            if item["kind"] == "child"
        ]
        complete_behavior_profiles = all(
            isinstance(item.get("behavior_profile"), Mapping)
            for item in candidates
            if item.get("status") in {"evaluated", "duplicate"}
        )
        scientific_external_activity_zero = all(
            all(
                int(value) == 0
                for value in cast(
                    Mapping[str, Any],
                    cast(Mapping[str, Any], item["behavior_profile"]).get(
                        "scientific_external_activity", {}
                    ),
                ).values()
            )
            for item in candidates
            if isinstance(item.get("behavior_profile"), Mapping)
        )
        report = {
            "protocol_id": M5_REPORT_PROTOCOL_ID,
            "status": "completed",
            "stop_reason": stopped_reason,
            "generation_count": len(generations),
            "population_size": POPULATION_SIZE,
            "candidate_count": len(candidates),
            "candidate_status_counts": dict(
                sorted(Counter(str(item.get("status")) for item in candidates).items())
            ),
            "generation_status_counts": {
                str(generation): dict(
                    sorted(
                        Counter(
                            str(item.get("status"))
                            for item in candidates
                            if int(item["generation"]) == generation
                        ).items()
                    )
                )
                for generation in generations
            },
            "generation_allocations": {
                str(generation): {
                    "children": sum(
                        item["kind"] == "child"
                        for item in candidates
                        if int(item["generation"]) == generation
                    ),
                    "roots": sum(
                        item["kind"] == "root"
                        for item in candidates
                        if int(item["generation"]) == generation
                    ),
                }
                for generation in generations
            },
            "generation_manifest_hashes": {
                str(generation): _load_mapping(
                    root / "generations" / f"generation-{generation:04d}" / "manifest.json.gz"
                )["sha256"]
                for generation in generations
            },
            "search_memory_hashes": {
                str(generation): _load_mapping(
                    root / "generations" / f"generation-{generation:04d}" / "search-memory.json.gz"
                )["sha256"]
                for generation in generations
            },
            "lineage": [
                {
                    "candidate_id": item["candidate_id"],
                    "parent_candidate_id": item["parent_candidate_id"],
                    "parent_program_hash": item.get("parent_program_hash"),
                    "parent_behavior_signature": item.get("parent_behavior_signature"),
                    "generation": item["generation"],
                    "slot": item["slot"],
                    "kind": item["kind"],
                    "program_hash": item["program_hash"],
                    "behavior_signature": item["behavior_signature"],
                    "provider_context": item["provider_context"],
                }
                for item in candidates
            ],
            "provider_order": [item["candidate_id"] for item in candidates],
            "evaluation_order": [
                {
                    "candidate_id": item["candidate_id"],
                    "case_id": case.case_id,
                }
                for item in candidates
                for case in panel
                if (
                    root
                    / "generations"
                    / f"generation-{int(item['generation']):04d}"
                    / str(item["slot"])
                    / "evaluations"
                    / f"{case.case_id}.json.gz"
                ).exists()
            ],
            "child_mutation_proofs": child_mutation_proofs,
            "behavior_profiles": {
                str(item["candidate_id"]): item["behavior_profile"] for item in candidates
            },
            "duplicates": [
                {
                    "candidate_id": item["candidate_id"],
                    "duplicate_of": item["duplicate_of"],
                }
                for item in candidates
                if item.get("duplicate_of") is not None
            ],
            "usage": _usage_with_anchor(candidates, anchor_result.usage),
            "provider_turns": 1
            + sum(len(cast(Sequence[object], item["provider_attempts"])) for item in candidates),
            "specification_anchor_turns": 1,
            "candidate_program_turns": sum(
                len(cast(Sequence[object], item["provider_attempts"])) for item in candidates
            ),
            "repair_turns": sum(int(item["repairs"]) for item in candidates),
            "provider_accounting": {
                "model": provider.model,
                "effort": provider.effort,
                "specification_anchor_turns": 1,
                "candidate_program_turns": sum(
                    len(cast(Sequence[object], item["provider_attempts"])) for item in candidates
                ),
                "warnings": anchor_result.warnings
                + sum(int(item.get("warnings", 0)) for item in candidates),
                "duration_ms": anchor_result.duration_ms
                + sum(int(item.get("duration_ms", 0)) for item in candidates),
                "usage_final_exact": (
                    anchor_result.usage.get("final") is True
                    and anchor_result.usage.get("partial") is False
                    and all(
                        item.get("status") not in {"provider_failed", "missing"}
                        for item in candidates
                    )
                    and all(
                        cast(Mapping[str, Any], attempt.get("usage", {})).get("final") is True
                        and cast(Mapping[str, Any], attempt.get("usage", {})).get("partial")
                        is False
                        for item in candidates
                        for attempt in cast(
                            Sequence[Mapping[str, Any]],
                            item["provider_attempts"],
                        )
                    )
                ),
                "system_prompt_bytes": len(system_prompt.encode("utf-8")),
                "specification_prompt_bytes": len(specification_prompt.encode("utf-8")),
                "specification_schema_bytes": len(canonical_json_bytes(specification_ack_schema)),
                "policy_schema_bytes": len(canonical_json_bytes(policy_schema)),
            },
            "exact_verified": exact_verified,
            "exact_verification": {
                "authority": "exact_verifier_only",
                "submissions": sum(
                    int(
                        cast(Mapping[str, Any], item["behavior_profile"]).get(
                            "exact_verifier_submissions", 0
                        )
                    )
                    for item in candidates
                    if isinstance(item.get("behavior_profile"), Mapping)
                ),
                "records": sum(
                    int(
                        cast(Mapping[str, Any], item["behavior_profile"]).get(
                            "exact_verifier_records", 0
                        )
                    )
                    for item in candidates
                    if isinstance(item.get("behavior_profile"), Mapping)
                ),
                "verified": exact_verified,
            },
            "equal_panel_hash": panel_hash(panel),
            "equal_development_budget": {
                "case_count": len(panel),
                "case_ids": [item.case_id for item in panel],
                "all_evaluated_candidates_complete": all(
                    int(item.get("evaluation_case_count", 0)) == len(panel)
                    for item in candidates
                    if item.get("status") in {"evaluated", "duplicate"}
                ),
                "held_out_evidence_used": False,
            },
            "sequential": True,
            "preview_active": preview_active,
            "dsl_runtime_used": False,
            "safe_api_expanded": False,
            "api_expressiveness": {
                "witness_scoped_k_switch_selector": False,
                "witness_scoped_fanout_selector": False,
                "witness_scoped_relocation_selector": False,
                "decision": "report_only_no_m5_api_expansion",
            },
            "acceptance_checks": {
                "generation_zero_eight_roots": sum(
                    int(item["generation"]) == 0 and item["kind"] == "root" for item in candidates
                )
                == 8,
                "generation_one_four_children_four_roots": (
                    sum(
                        int(item["generation"]) == 1 and item["kind"] == "child"
                        for item in candidates
                    )
                    == 4
                    and sum(
                        int(item["generation"]) == 1 and item["kind"] == "root"
                        for item in candidates
                    )
                    == 4
                ),
                "child_source_changed": any(
                    item["source_changed"] is True for item in child_mutation_proofs
                ),
                "child_program_or_behavior_changed": any(
                    item["program_changed"] is True or item["semantic_behavior_changed"] is True
                    for item in child_mutation_proofs
                ),
                "equal_development_panel_and_budget": all(
                    int(item.get("evaluation_case_count", 0)) == len(panel)
                    for item in candidates
                    if item.get("status") in {"evaluated", "duplicate"}
                ),
                "behavior_profile_for_every_evaluated_program": (complete_behavior_profiles),
                "scientific_evaluator_model_provider_activity_zero": (
                    scientific_external_activity_zero
                ),
            },
        }
        write_json(root / "m5-report.json.gz", report)
        if boundary_hook is not None:
            boundary_hook("report_persisted")
        if close_provider:
            provider.close()
        return report
    except (M5InfrastructureError, M5OperatorStop) as error:
        write_json(
            root / "m5-stop.json.gz",
            {
                "protocol_id": M5_REPORT_PROTOCOL_ID,
                "status": (
                    "operator_stop"
                    if isinstance(error, M5OperatorStop)
                    else "infrastructure_failure"
                ),
                "error_type": type(error).__name__,
                "error": str(error)[:2048],
                "candidate_count": len(_all_candidates(root)),
                "panel_hash": panel_hash(panel),
                "resumable": True,
            },
        )
        raise


__all__ = [
    "CHILD_SLOTS",
    "DevelopmentCaseV1",
    "GenerationManifestV1",
    "M5InfrastructureError",
    "M5OperatorStop",
    "M5ProviderContextV1",
    "M5ProviderResultV1",
    "M5ScientificEvaluator",
    "M5SearchError",
    "M5_TERMINAL_CANDIDATE_STATUSES",
    "M5SearchProvider",
    "M10SearchProvider",
    "M5_REPORT_PROTOCOL_ID",
    "M5_SEARCH_PROTOCOL_ID",
    "POPULATION_SIZE",
    "ROOT_SLOTS",
    "SlotPlanV1",
    "aggregate_behavior",
    "behavior_distance",
    "build_child_prompt",
    "build_generation_manifest",
    "build_root_prompt",
    "build_search_memory",
    "python_control_flow_summary",
    "run_m5_search",
    "select_parent_candidates",
]
