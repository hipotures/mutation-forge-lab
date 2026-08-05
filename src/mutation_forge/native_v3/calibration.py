"""Provider batch calibration using independent-call, wall-clock metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from time import monotonic_ns

BATCH_SIZES = (1, 2, 4, 8)
CALIBRATION_PROTOCOL_ID = "native_v3_batch_calibration_16_slots_v1"
PRACTICAL_IMPROVEMENT = Fraction(11, 10)


@dataclass(frozen=True, slots=True)
class CalibrationCallResult:
    valid_program_hashes: tuple[str, ...]
    correctness_passed: bool
    diversity_passed: bool
    provider_active_ns: int


@dataclass(frozen=True, slots=True)
class BatchCalibrationResult:
    batch_size: int
    independent_calls: int
    unique_valid_programs: int
    wall_time_to_four_ns: int | None
    wall_time_to_terminal_eight_ns: int | None
    epoch_wall_ns: int
    provider_active_ns: int
    correctness_passed: bool
    diversity_passed: bool

    @property
    def unique_per_epoch_wall_minute(self) -> Fraction:
        return Fraction(
            self.unique_valid_programs * 60_000_000_000,
            max(1, self.epoch_wall_ns),
        )

    @property
    def valid_per_provider_worker_minute(self) -> Fraction:
        return Fraction(
            self.unique_valid_programs * 60_000_000_000,
            max(1, self.provider_active_ns),
        )


def calibrate_batch_size(
    *,
    batch_size: int,
    model_concurrency: int,
    call: Callable[[int, int], CalibrationCallResult],
    slot_count: int = 16,
) -> BatchCalibrationResult:
    if batch_size not in BATCH_SIZES:
        raise ValueError("calibration batch size must be 1, 2, 4, or 8")
    if model_concurrency <= 0 or slot_count <= 0:
        raise ValueError("calibration concurrency and slot count must be positive")
    call_sizes = tuple(
        min(batch_size, slot_count - offset) for offset in range(0, slot_count, batch_size)
    )
    started = monotonic_ns()
    hashes: set[str] = set()
    time_to_four: int | None = None
    time_to_eight: int | None = None
    active_ns = 0
    correctness = True
    diversity = True
    with ThreadPoolExecutor(max_workers=model_concurrency) as executor:
        futures = {
            executor.submit(call, ordinal, size): ordinal for ordinal, size in enumerate(call_sizes)
        }
        for future in as_completed(futures):
            result = future.result()
            active_ns += result.provider_active_ns
            correctness = correctness and result.correctness_passed
            diversity = diversity and result.diversity_passed
            hashes.update(result.valid_program_hashes)
            elapsed = monotonic_ns() - started
            if len(hashes) >= 4 and time_to_four is None:
                time_to_four = elapsed
            if len(hashes) >= 8 and time_to_eight is None:
                time_to_eight = elapsed
    return BatchCalibrationResult(
        batch_size,
        len(call_sizes),
        len(hashes),
        time_to_four,
        time_to_eight,
        max(1, monotonic_ns() - started),
        active_ns,
        correctness,
        diversity,
    )


def select_batch_size(
    results: Iterable[BatchCalibrationResult],
) -> BatchCalibrationResult:
    by_size = {result.batch_size: result for result in results}
    if set(by_size) != set(BATCH_SIZES):
        raise ValueError("calibration requires batch sizes 1, 2, 4, and 8")
    selected = by_size[1]
    for size in BATCH_SIZES[1:]:
        challenger = by_size[size]
        gates = challenger.correctness_passed and challenger.diversity_passed
        practical_gain = (
            challenger.unique_per_epoch_wall_minute
            > selected.unique_per_epoch_wall_minute * PRACTICAL_IMPROVEMENT
        )
        faster_four = challenger.wall_time_to_four_ns is not None and (
            selected.wall_time_to_four_ns is None
            or challenger.wall_time_to_four_ns < selected.wall_time_to_four_ns
        )
        faster_cohort = challenger.wall_time_to_terminal_eight_ns is not None and (
            selected.wall_time_to_terminal_eight_ns is None
            or challenger.wall_time_to_terminal_eight_ns < selected.wall_time_to_terminal_eight_ns
        )
        if gates and practical_gain and faster_four and faster_cohort:
            selected = challenger
    return selected
