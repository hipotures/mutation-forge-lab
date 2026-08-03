from __future__ import annotations

import json
from pathlib import Path

from mutation_forge.experiment.native import (
    _NativeArchive,
    _unfinished_generation_program_ids,
)


def _candidate(generation: int, slot: str) -> dict[str, object]:
    return {
        "generation": generation,
        "slot": slot,
    }


def test_unfinished_generation_does_not_deduplicate_its_own_archive(
    tmp_path: Path,
) -> None:
    archive = _NativeArchive(tmp_path / "archive")
    archive.append(
        {
            "program_id": "g0000-slot-00",
            "source": "def priority(ctx, proposal):\n    return 0\n",
        }
    )
    archive.append(
        {
            "program_id": "g0001-slot-00",
            "source": "def priority(ctx, proposal):\n    return 1\n",
        }
    )
    checkpoint_path = tmp_path / "native-generation-checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": "mforge.experiment.generation.v2",
                "campaign_id": "native-test",
                "slots": {
                    "generation-0": {"candidate": _candidate(0, "slot-00")},
                    "generation-1": {"candidate": _candidate(1, "slot-00")},
                },
                "callbacks": {"0": {"status": "completed"}},
            }
        ),
        encoding="utf-8",
    )

    unfinished = _unfinished_generation_program_ids(checkpoint_path)

    assert unfinished == {"g0001-slot-00"}
    assert archive.existing_sources(exclude_program_ids=unfinished) == (
        "def priority(ctx, proposal):\n    return 0\n",
    )
