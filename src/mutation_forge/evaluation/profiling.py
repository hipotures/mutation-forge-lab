from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from mutation_forge.models import (
    DeepOperatorTimingProfile,
    EpisodeTimingProfile,
    JsonValue,
)

DEEP_COUNTER_FIELDS = (
    "witness_cache_lookups",
    "witness_cache_hits",
    "witness_cache_misses",
    "witness_searches",
    "witness_search_ns",
    "witness_edge_materialization_ns",
    "switch_attempts",
    "partner_edge_sampling_ns",
    "candidate_construction_ns",
    "connectivity_validation_ns",
    "graph_family_validation_ns",
)


@dataclass(slots=True)
class TimingAccumulator:
    scoring_ns: int = 0
    proposal_generation_ns: int = 0
    rewrite_application_ns: int = 0
    duplicate_detection_ns: int = 0
    controller_ns: int = 0
    exact_verification_ns: int = 0
    progress_reporting_ns: int = 0
    finalization_ns: int = 0
    proposal_rng_setup_ns: int = 0
    proposal_graph_materialization_ns: int = 0
    proposal_operator_search_ns: int = 0
    proposal_packaging_ns: int = 0
    proposal_generation_calls: int = 0
    proposal_rng_setup_calls: int = 0
    proposal_graph_materialization_calls: int = 0
    proposal_operator_search_calls: int = 0
    proposal_packaging_calls: int = 0

    def record_proposal_phase(self, phase: str, elapsed_ns: int) -> None:
        match phase:
            case "rng_setup":
                self.proposal_rng_setup_ns += elapsed_ns
                self.proposal_rng_setup_calls += 1
            case "graph_materialization":
                self.proposal_graph_materialization_ns += elapsed_ns
                self.proposal_graph_materialization_calls += 1
            case "operator_search":
                self.proposal_operator_search_ns += elapsed_ns
                self.proposal_operator_search_calls += 1
            case "proposal_packaging":
                self.proposal_packaging_ns += elapsed_ns
                self.proposal_packaging_calls += 1
            case _:
                raise ValueError(f"unknown proposal timing phase: {phase}")

    def finish(self, measured_total_ns: int) -> EpisodeTimingProfile:
        return EpisodeTimingProfile(
            measured_total_ns=measured_total_ns,
            scoring_ns=self.scoring_ns,
            proposal_generation_ns=self.proposal_generation_ns,
            rewrite_application_ns=self.rewrite_application_ns,
            duplicate_detection_ns=self.duplicate_detection_ns,
            controller_ns=self.controller_ns,
            exact_verification_ns=self.exact_verification_ns,
            progress_reporting_ns=self.progress_reporting_ns,
            finalization_ns=self.finalization_ns,
            proposal_rng_setup_ns=self.proposal_rng_setup_ns,
            proposal_graph_materialization_ns=self.proposal_graph_materialization_ns,
            proposal_operator_search_ns=self.proposal_operator_search_ns,
            proposal_packaging_ns=self.proposal_packaging_ns,
            proposal_generation_calls=self.proposal_generation_calls,
            proposal_rng_setup_calls=self.proposal_rng_setup_calls,
            proposal_graph_materialization_calls=(
                self.proposal_graph_materialization_calls
            ),
            proposal_operator_search_calls=self.proposal_operator_search_calls,
            proposal_packaging_calls=self.proposal_packaging_calls,
        )


@dataclass(slots=True)
class DeepOperatorTimingAccumulator:
    operator_counters: dict[str, dict[str, int]]

    def __init__(self) -> None:
        self.operator_counters = {}

    def record(
        self,
        operator: str,
        payload: Mapping[str, int | float | bool],
    ) -> None:
        counters = self.operator_counters.setdefault(operator, {})
        if operator == "heg_uniform_two_switch":
            operator_ns = payload.get("uniform_ns", 0)
            operator_calls = payload.get("uniform_evaluations", 0)
        else:
            operator_ns = payload.get("targeted_ns", 0)
            operator_calls = payload.get("targeted_evaluations", 0)
        if isinstance(operator_ns, int) and not isinstance(operator_ns, bool):
            counters["operator_search_ns"] = (
                counters.get("operator_search_ns", 0) + operator_ns
            )
        if isinstance(operator_calls, int) and not isinstance(
            operator_calls, bool
        ):
            counters["operator_search_calls"] = (
                counters.get("operator_search_calls", 0) + operator_calls
            )
        for field in DEEP_COUNTER_FIELDS:
            value = payload.get(field, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                counters[field] = counters.get(field, 0) + value

    def finish(self) -> DeepOperatorTimingProfile:
        return DeepOperatorTimingProfile(
            operator_counters={
                operator: dict(counters)
                for operator, counters in self.operator_counters.items()
            }
        )


def aggregate_timing_profiles(
    profiles: Iterable[tuple[str, EpisodeTimingProfile | None]],
    *,
    enabled: bool,
) -> dict[str, JsonValue]:
    collected = [
        (baseline, profile)
        for baseline, profile in profiles
        if profile is not None
    ]
    phase_totals: dict[str, int] = {}
    proposal_phase_totals: dict[str, int] = {}
    proposal_phase_calls: dict[str, int] = {}
    operator_seconds_by_baseline: dict[str, int] = {}
    operator_calls_by_baseline: dict[str, int] = {}
    proposal_generation_calls = 0
    measured_total_ns = 0
    for baseline, profile in collected:
        measured_total_ns += profile.measured_total_ns
        proposal_generation_calls += profile.proposal_generation_calls
        operator_seconds_by_baseline[baseline] = (
            operator_seconds_by_baseline.get(baseline, 0)
            + profile.proposal_operator_search_ns
        )
        operator_calls_by_baseline[baseline] = (
            operator_calls_by_baseline.get(baseline, 0)
            + profile.proposal_operator_search_calls
        )
        for phase, nanoseconds in profile.phase_nanoseconds().items():
            phase_totals[phase] = phase_totals.get(phase, 0) + nanoseconds
        for phase, nanoseconds in profile.proposal_breakdown_nanoseconds().items():
            proposal_phase_totals[phase] = (
                proposal_phase_totals.get(phase, 0) + nanoseconds
            )
        for phase, calls in profile.proposal_phase_calls().items():
            proposal_phase_calls[phase] = proposal_phase_calls.get(phase, 0) + calls

    accounted_ns = sum(phase_totals.values())
    unattributed_ns = max(0, measured_total_ns - accounted_ns)
    dominant_phase: str | None = None
    dominant_ns = 0
    if phase_totals:
        dominant_phase, dominant_ns = max(phase_totals.items(), key=lambda item: item[1])
    operator_calls_json: dict[str, JsonValue] = {
        baseline: calls for baseline, calls in operator_calls_by_baseline.items()
    }
    return {
        "enabled": enabled,
        "profiled_episodes": len(collected),
        "phase_seconds": {
            phase: nanoseconds / 1_000_000_000 for phase, nanoseconds in phase_totals.items()
        },
        "phase_children_seconds": {
            "proposal_generation": {
                phase: nanoseconds / 1_000_000_000
                for phase, nanoseconds in proposal_phase_totals.items()
            }
        },
        "phase_calls": {"proposal_generation": proposal_generation_calls},
        "phase_children_calls": {
            "proposal_generation": {
                **proposal_phase_calls,
                "other": None,
            }
        },
        "phase_grandchildren_seconds": {
            "proposal_generation": {
                "operator_search": {
                    baseline: nanoseconds / 1_000_000_000
                    for baseline, nanoseconds in operator_seconds_by_baseline.items()
                }
            }
        },
        "phase_grandchildren_calls": {
            "proposal_generation": {
                "operator_search": operator_calls_json,
            }
        },
        "measured_total_seconds": measured_total_ns / 1_000_000_000,
        "accounted_seconds": accounted_ns / 1_000_000_000,
        "unattributed_seconds": unattributed_ns / 1_000_000_000,
        "unattributed_fraction": unattributed_ns / max(1, measured_total_ns),
        "dominant_phase": dominant_phase,
        "dominant_seconds": dominant_ns / 1_000_000_000,
    }


def aggregate_deep_operator_profiles(
    profiles: Iterable[DeepOperatorTimingProfile | None],
    *,
    enabled: bool,
) -> dict[str, JsonValue]:
    collected = [profile for profile in profiles if profile is not None]
    merged: dict[str, dict[str, int]] = {}
    for profile in collected:
        for operator, counters in profile.operator_counters.items():
            aggregate = merged.setdefault(operator, {})
            for field, value in counters.items():
                aggregate[field] = aggregate.get(field, 0) + value
    serialized = DeepOperatorTimingProfile(merged).as_dict()
    return {
        "enabled": enabled,
        "profiled_episodes": len(collected),
        **serialized,
    }
