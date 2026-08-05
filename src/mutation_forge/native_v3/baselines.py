"""Fixed Native v3 baseline programs executed through the same DSL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mutation_forge.models import JsonValue

from .contracts import ValidatedProgram, validate_program

BASELINE_SCHEMA_VERSION = "mforge.native.baseline_programs.v3"


@dataclass(frozen=True, slots=True)
class BaselineProgram:
    baseline_id: str
    program: ValidatedProgram
    operator_family_weights: dict[str, int]


def load_baseline_programs(path: Path) -> tuple[BaselineProgram, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported Native v3 baseline manifest")
    entries = payload.get("baselines")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("Native v3 requires exactly four fixed baselines")
    output: list[BaselineProgram] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("baseline entry must be an object")
        baseline_id = entry.get("baseline_id")
        if not isinstance(baseline_id, str) or not baseline_id or baseline_id in seen:
            raise ValueError("baseline IDs must be non-empty and unique")
        seen.add(baseline_id)
        weights = entry.get("operator_family_weights")
        if (
            not isinstance(weights, dict)
            or not weights
            or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for key, value in weights.items()
            )
        ):
            raise ValueError("baseline operator distribution must have positive weights")
        program_ast = entry.get("program_ast")
        raw = json.dumps(
            program_ast,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        validation = validate_program(raw)
        if validation.program is None:
            diagnostics = "; ".join(
                f"{item.code}@{item.path}: {item.message}" for item in validation.diagnostics
            )
            raise ValueError(f"invalid baseline {baseline_id}: {diagnostics}")
        output.append(
            BaselineProgram(
                baseline_id,
                validation.program,
                cast(dict[str, int], weights),
            )
        )
    return tuple(output)


def baseline_summary(baseline: BaselineProgram) -> dict[str, JsonValue]:
    return {
        "baseline_id": baseline.baseline_id,
        "program_hash": baseline.program.program_hash,
        "operator_family_weights": cast(
            dict[str, JsonValue],
            baseline.operator_family_weights,
        ),
    }
