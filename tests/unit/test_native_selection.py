from __future__ import annotations

import hashlib

from mutation_forge.experiment.generation import Candidate
from mutation_forge.experiment.native import NativeExperimentAdapter


def _candidate(index: int, score: float) -> Candidate:
    source = f"def priority(ctx, proposal):\n    return {index}\n"
    return Candidate(
        source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        normalized_ast_sha256=hashlib.sha256(f"ast-{index}".encode()).hexdigest(),
        generation=4,
        slot=f"slot-{index:02d}",
        parent_id="previous-parent",
        behavior_signature={"score": score},
    )


def test_persistent_elite_selection_keeps_global_best_and_weights_top_half() -> None:
    candidates = tuple(_candidate(index, (index + 1) / 10) for index in range(8))

    selected = NativeExperimentAdapter._select_parents(
        generation=4,
        candidates=candidates,
        slots=8,
        selection="persistent-elite-weighted-diversity",
        global_best_id="g0003-slot-07",
    )

    assert tuple(selected) == tuple(f"slot-{index:02d}" for index in range(8))
    assert tuple(selected.values())[:3] == ("g0003-slot-07",) * 3
    assert tuple(selected.values())[3:5] == ("g0004-slot-07",) * 2
    assert set(tuple(selected.values())[5:]) <= {
        "g0004-slot-04",
        "g0004-slot-05",
        "g0004-slot-06",
        "g0004-slot-07",
    }
    assert selected == NativeExperimentAdapter._select_parents(
        generation=4,
        candidates=candidates,
        slots=8,
        selection="persistent-elite-weighted-diversity",
        global_best_id="g0003-slot-07",
    )


def test_elite_diversity_retains_existing_one_parent_per_candidate_behavior() -> None:
    candidates = tuple(_candidate(index, (index + 1) / 10) for index in range(8))

    selected = NativeExperimentAdapter._select_parents(
        generation=4,
        candidates=candidates,
        slots=8,
        selection="elite-diversity",
        global_best_id="g0003-slot-07",
    )

    assert len(set(selected.values())) == 8
    assert "g0003-slot-07" not in selected.values()
    assert set(selected.values()) == {f"g0004-slot-{index:02d}" for index in range(8)}
