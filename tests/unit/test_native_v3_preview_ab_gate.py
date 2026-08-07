from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from mutation_forge.experiment.json_io import write_json
from mutation_forge.native_v3.persistent_experiment import BRIEF_IDS

_SCRIPT_PATH = Path("scripts/native_v3_preview_ab_gate.py").resolve()
_SPEC = importlib.util.spec_from_file_location("native_v3_preview_ab_gate", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _selected_root(tmp_path: Path, *, briefs: tuple[str, ...] = BRIEF_IDS) -> Path:
    root = tmp_path / "selected"
    output = root / "native-v3-output" / "epoch-0000"
    write_json(
        output / "cohort-report.json.gz",
        {
            "provider_mode": "persistent_single_ast",
            "output_contract": "slot_specific",
            "program_turns": 4,
            "valid_slots": 4,
            "unique_valid_programs": 4,
            "time_to_first_valid_ast_ms": 1234,
            "usage": {"totalTokens": 100},
        },
    )
    write_json(
        output / "communication-state.json.gz",
        {
            "status": "completed",
            "provider_retries": 0,
            "provider_warnings": 0,
            "slot_reports": [
                {
                    "entry": {"program_hash": str(index) * 64},
                    "attempts": [{"artifact_complete": True}],
                }
                for index in range(1, 5)
            ],
        },
    )
    write_json(
        output / "epoch-manifest.json.gz",
        {"slots": [{"brief_id": brief_id} for brief_id in briefs]},
    )
    return root


def test_selected_summary_requires_the_integrated_contract(tmp_path: Path) -> None:
    summary = _MODULE._selected_summary(_selected_root(tmp_path))

    assert summary == {
        "provider_mode": "persistent_single_ast",
        "output_contract": "slot_specific",
        "brief_ids": list(BRIEF_IDS),
        "program_turns": 4,
        "valid_slots": 4,
        "unique_valid_programs": 4,
        "program_hashes": [str(index) * 64 for index in range(1, 5)],
        "time_to_first_valid_ast_ms": 1234,
        "provider_retries": 0,
        "provider_warnings": 0,
        "artifact_parity": True,
        "usage": {"totalTokens": 100},
    }


def test_selected_summary_rejects_a_different_brief_set(tmp_path: Path) -> None:
    root = _selected_root(tmp_path, briefs=("add-edge",))

    with pytest.raises(ValueError, match="briefs do not match"):
        _MODULE._selected_summary(root)
