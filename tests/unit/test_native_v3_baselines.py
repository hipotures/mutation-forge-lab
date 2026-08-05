from __future__ import annotations

from pathlib import Path

from mutation_forge.native_v3.baselines import load_baseline_programs


def test_all_four_fixed_baselines_are_valid_native_v3_programs() -> None:
    root = Path(__file__).resolve().parents[2]
    baselines = load_baseline_programs(
        root / "configs" / "native" / "native-v3-baseline-programs.json"
    )
    assert len(baselines) == 4
    assert len({baseline.program.program_hash for baseline in baselines}) == 4
    assert {baseline.baseline_id for baseline in baselines} == {
        "add-low-local-cycle-risk",
        "remove-low-bridge-risk",
        "random-valid",
        "degree-fanout",
    }
    random_valid = next(
        baseline for baseline in baselines if baseline.baseline_id == "random-valid"
    )
    assert random_valid.operator_family_weights == {"add_edge": 1, "remove_edge": 1}
