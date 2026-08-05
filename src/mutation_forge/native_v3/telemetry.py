"""Derived scheduler telemetry and bottleneck classification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from mutation_forge.models import JsonValue

from .scheduler import TelemetryEvent

TELEMETRY_PROTOCOL_ID = "native_v3_scheduler_telemetry_v1"


class Bottleneck(StrEnum):
    PROVIDER_BOUND = "PROVIDER_BOUND"
    EVALUATOR_BOUND = "EVALUATOR_BOUND"
    PERSISTENCE_BOUND = "PERSISTENCE_BOUND"
    BALANCED = "BALANCED"


@dataclass(frozen=True, slots=True)
class SchedulerTelemetry:
    epoch_wall_ns: int
    provider_calls: int
    provider_latency_ns: int
    programs_returned: int
    valid_programs: int
    provider_utilization: Fraction
    maximum_provider_calls_in_flight: int
    maximum_candidate_queue_depth: int
    maximum_evaluation_shard_queue_depth: int
    evaluator_utilization: Fraction
    cpu_idle_provider_starvation_ns: int
    provider_idle_backpressure_ns: int
    generation_wall_share: Fraction
    validation_wall_share: Fraction
    evaluation_wall_share: Fraction
    persistence_wall_share: Fraction
    time_to_first_evaluation_ns: int | None
    first_valid_ast_to_first_worker_ns: int | None
    first_valid_ast_to_50_percent_workers_ns: int | None
    first_valid_ast_to_all_workers_ns: int | None
    bottleneck: Bottleneck

    def as_dict(self) -> dict[str, JsonValue]:
        def rational(value: Fraction) -> dict[str, JsonValue]:
            return {
                "numerator": value.numerator,
                "denominator": value.denominator,
            }

        provider_minutes = Fraction(max(1, self.provider_latency_ns), 60_000_000_000)
        return {
            "protocol_id": TELEMETRY_PROTOCOL_ID,
            "epoch_wall_ns": self.epoch_wall_ns,
            "provider_calls": self.provider_calls,
            "provider_response_latency_ns": self.provider_latency_ns,
            "programs_returned": self.programs_returned,
            "programs_returned_per_call": rational(
                Fraction(self.programs_returned, max(1, self.provider_calls))
            ),
            "valid_programs_per_provider_minute": rational(
                Fraction(self.valid_programs, 1) / provider_minutes
            ),
            "provider_utilization": rational(self.provider_utilization),
            "provider_calls_in_flight": self.maximum_provider_calls_in_flight,
            "candidate_queue_depth": self.maximum_candidate_queue_depth,
            "evaluation_shard_queue_depth": self.maximum_evaluation_shard_queue_depth,
            "evaluator_utilization": rational(self.evaluator_utilization),
            "cpu_idle_time_caused_by_provider_starvation_ns": (
                self.cpu_idle_provider_starvation_ns
            ),
            "provider_idle_time_caused_by_evaluation_backpressure_ns": (
                self.provider_idle_backpressure_ns
            ),
            "generation_wall_share": rational(self.generation_wall_share),
            "validation_wall_share": rational(self.validation_wall_share),
            "evaluation_wall_share": rational(self.evaluation_wall_share),
            "persistence_wall_share": rational(self.persistence_wall_share),
            "time_to_first_evaluation_ns": self.time_to_first_evaluation_ns,
            "first_valid_ast_to_first_worker_ns": self.first_valid_ast_to_first_worker_ns,
            "first_valid_ast_to_50_percent_workers_ns": (
                self.first_valid_ast_to_50_percent_workers_ns
            ),
            "first_valid_ast_to_all_workers_ns": self.first_valid_ast_to_all_workers_ns,
            "bottleneck": self.bottleneck.value,
        }


def summarize_scheduler_telemetry(
    events: tuple[TelemetryEvent, ...],
    *,
    provider_concurrency: int,
    evaluator_workers: int,
    validation_wall_ns: int = 0,
    persistence_wall_ns: int = 0,
) -> SchedulerTelemetry:
    if not events:
        raise ValueError("scheduler telemetry requires events")
    scheduler_wall_ns = max(1, events[-1].monotonic_ns - events[0].monotonic_ns)
    epoch_wall_ns = max(1, scheduler_wall_ns + max(0, validation_wall_ns))
    provider_calls = 0
    provider_latency_ns = 0
    programs_returned = 0
    valid_programs = 0
    max_provider = 0
    max_candidate_queue = 0
    max_evaluation_queue = 0
    provider_backpressure_ns = 0
    starvation_ns = 0
    first_evaluation_ns: int | None = None
    first_candidate_worker_ns: int | None = None
    first_candidate_half_ns: int | None = None
    first_candidate_all_ns: int | None = None
    evaluator_starts: dict[str, int] = {}
    evaluator_active_ns = 0
    provider_intervals: list[tuple[int, int]] = []
    evaluator_intervals: list[tuple[int, int]] = []

    def integer(fields: object, key: str) -> int:
        if not isinstance(fields, Mapping):
            return 0
        value = fields.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    for event in events:
        fields = event.fields
        max_provider = max(
            max_provider,
            integer(fields, "provider_calls_in_flight"),
        )
        max_candidate_queue = max(
            max_candidate_queue,
            integer(fields, "candidate_queue_depth"),
        )
        max_evaluation_queue = max(
            max_evaluation_queue,
            integer(fields, "evaluation_shard_queue_depth"),
        )
        if event.name == "provider_call_completed":
            provider_calls += 1
            latency_ns = integer(fields, "latency_ns")
            provider_latency_ns += latency_ns
            provider_intervals.append(
                (max(events[0].monotonic_ns, event.monotonic_ns - latency_ns), event.monotonic_ns)
            )
            programs_returned += integer(fields, "programs_returned")
            valid_programs += integer(fields, "valid_programs")
        elif event.name == "provider_call_failed":
            provider_calls += 1
            latency_ns = integer(fields, "latency_ns")
            provider_latency_ns += latency_ns
            provider_intervals.append(
                (max(events[0].monotonic_ns, event.monotonic_ns - latency_ns), event.monotonic_ns)
            )
        elif event.name == "provider_backpressure_ended":
            provider_backpressure_ns += integer(fields, "idle_ns")
        elif event.name == "cpu_idle_provider_starvation_ended":
            starvation_ns += integer(fields, "idle_ns")
        elif event.name == "evaluation_shard_started":
            shard_id = str(fields.get("shard_id", ""))
            evaluator_starts[shard_id] = event.monotonic_ns
            observed = fields.get("time_to_first_evaluation_ns")
            if isinstance(observed, int) and first_evaluation_ns is None:
                first_evaluation_ns = observed
            observed = fields.get("first_valid_ast_to_first_worker_ns")
            if isinstance(observed, int) and first_candidate_worker_ns is None:
                first_candidate_worker_ns = observed
            observed = fields.get("first_valid_ast_to_50_percent_workers_ns")
            if isinstance(observed, int) and first_candidate_half_ns is None:
                first_candidate_half_ns = observed
            observed = fields.get("first_valid_ast_to_all_workers_ns")
            if isinstance(observed, int) and first_candidate_all_ns is None:
                first_candidate_all_ns = observed
        elif event.name in {
            "evaluation_shard_completed",
            "evaluation_shard_failed",
            "evaluation_shard_rescheduled",
        }:
            shard_id = str(fields.get("shard_id", ""))
            started = evaluator_starts.pop(shard_id, None)
            if started is not None:
                duration = max(0, event.monotonic_ns - started)
                evaluator_active_ns += duration
                evaluator_intervals.append((started, event.monotonic_ns))

    def union_duration(intervals: list[tuple[int, int]]) -> int:
        total = 0
        end = -1
        for start, stop in sorted(intervals):
            if stop <= start:
                continue
            if start >= end:
                total += stop - start
            elif stop > end:
                total += stop - end
            end = max(end, stop)
        return total

    provider_wall_ns = union_duration(provider_intervals)
    evaluator_wall_ns = union_duration(evaluator_intervals)
    provider_utilization = min(
        Fraction(1),
        Fraction(provider_latency_ns, epoch_wall_ns * max(1, provider_concurrency)),
    )
    evaluator_utilization = min(
        Fraction(1),
        Fraction(evaluator_active_ns, epoch_wall_ns * max(1, evaluator_workers)),
    )
    generation_share = min(Fraction(1), Fraction(provider_wall_ns, epoch_wall_ns))
    validation_share = min(Fraction(1), Fraction(validation_wall_ns, epoch_wall_ns))
    evaluation_share = min(Fraction(1), Fraction(evaluator_wall_ns, epoch_wall_ns))
    persistence_share = min(Fraction(1), Fraction(persistence_wall_ns, epoch_wall_ns))
    if persistence_share >= max(generation_share, evaluation_share, Fraction(1, 4)):
        bottleneck = Bottleneck.PERSISTENCE_BOUND
    elif provider_backpressure_ns > starvation_ns and evaluator_utilization >= Fraction(3, 4):
        bottleneck = Bottleneck.EVALUATOR_BOUND
    elif starvation_ns > 0 or provider_utilization >= evaluator_utilization:
        bottleneck = Bottleneck.PROVIDER_BOUND
    else:
        bottleneck = Bottleneck.BALANCED
    return SchedulerTelemetry(
        epoch_wall_ns,
        provider_calls,
        provider_latency_ns,
        programs_returned,
        valid_programs,
        provider_utilization,
        max_provider,
        max_candidate_queue,
        max_evaluation_queue,
        evaluator_utilization,
        starvation_ns,
        provider_backpressure_ns,
        generation_share,
        validation_share,
        evaluation_share,
        persistence_share,
        first_evaluation_ns,
        first_candidate_worker_ns,
        first_candidate_half_ns,
        first_candidate_all_ns,
        bottleneck,
    )
