from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mutation_forge.models import EpisodeTimingProfile, JsonValue


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


def aggregate_timing_profiles(
    profiles: Iterable[EpisodeTimingProfile | None],
    *,
    enabled: bool,
) -> dict[str, JsonValue]:
    collected = [profile for profile in profiles if profile is not None]
    phase_totals: dict[str, int] = {}
    proposal_phase_totals: dict[str, int] = {}
    proposal_phase_calls: dict[str, int] = {}
    proposal_generation_calls = 0
    measured_total_ns = 0
    for profile in collected:
        measured_total_ns += profile.measured_total_ns
        proposal_generation_calls += profile.proposal_generation_calls
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
        "measured_total_seconds": measured_total_ns / 1_000_000_000,
        "accounted_seconds": accounted_ns / 1_000_000_000,
        "unattributed_seconds": unattributed_ns / 1_000_000_000,
        "unattributed_fraction": unattributed_ns / max(1, measured_total_ns),
        "dominant_phase": dominant_phase,
        "dominant_seconds": dominant_ns / 1_000_000_000,
    }
