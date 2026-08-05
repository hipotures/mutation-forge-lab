from __future__ import annotations

from mutation_forge.native_v3.calibration import (
    BATCH_SIZES,
    BatchCalibrationResult,
    CalibrationCallResult,
    calibrate_batch_size,
    select_batch_size,
)


def test_sixteen_slot_calibration_counts_independent_provider_calls() -> None:
    observed: dict[int, int] = {}
    for batch_size in BATCH_SIZES:
        result = calibrate_batch_size(
            batch_size=batch_size,
            model_concurrency=2,
            call=lambda ordinal, size, batch_size=batch_size: CalibrationCallResult(
                tuple(f"{batch_size}:{ordinal}:{index}" for index in range(size)),
                True,
                True,
                1,
            ),
        )
        observed[batch_size] = result.independent_calls
    assert observed == {1: 16, 2: 8, 4: 4, 8: 2}


def _result(size: int, *, wall: int, valid: int = 8) -> BatchCalibrationResult:
    return BatchCalibrationResult(
        size,
        16 // size,
        valid,
        wall // 2,
        wall,
        wall,
        wall,
        True,
        True,
    )


def test_larger_batch_requires_more_than_ten_percent_practical_wall_gain() -> None:
    selected = select_batch_size(
        (
            _result(1, wall=100),
            _result(2, wall=95),
            _result(4, wall=80),
            _result(8, wall=79),
        )
    )
    assert selected.batch_size == 4


def test_correctness_or_diversity_gate_blocks_larger_batch() -> None:
    invalid = BatchCalibrationResult(
        8,
        2,
        8,
        1,
        2,
        2,
        2,
        True,
        False,
    )
    selected = select_batch_size(
        (_result(1, wall=100), _result(2, wall=90), _result(4, wall=80), invalid)
    )
    assert selected.batch_size == 4
