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
        )


def aggregate_timing_profiles(
    profiles: Iterable[EpisodeTimingProfile | None],
    *,
    enabled: bool,
) -> dict[str, JsonValue]:
    collected = [profile for profile in profiles if profile is not None]
    phase_totals: dict[str, int] = {}
    measured_total_ns = 0
    for profile in collected:
        measured_total_ns += profile.measured_total_ns
        for phase, nanoseconds in profile.phase_nanoseconds().items():
            phase_totals[phase] = phase_totals.get(phase, 0) + nanoseconds

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
        "measured_total_seconds": measured_total_ns / 1_000_000_000,
        "accounted_seconds": accounted_ns / 1_000_000_000,
        "unattributed_seconds": unattributed_ns / 1_000_000_000,
        "unattributed_fraction": unattributed_ns / max(1, measured_total_ns),
        "dominant_phase": dominant_phase,
        "dominant_seconds": dominant_ns / 1_000_000_000,
    }
