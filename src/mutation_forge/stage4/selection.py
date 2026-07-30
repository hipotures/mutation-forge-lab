"""Deterministic parent selection for the eight Stage 4 slots."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .archive import ProgramRecord


def behavior_distance(left: str | Mapping[str, Any], right: str | Mapping[str, Any]) -> int:
    """Bitwise Hamming distance between two 256-bit (64 hex character) signatures."""

    def normal(value: str | Mapping[str, Any]) -> str:
        if isinstance(value, Mapping):
            candidate = value.get(
                "signature_sha256",
                value.get("behavior_signature_sha256", value.get("signature", "")),
            )
            value = candidate if isinstance(candidate, str) else ""
        text = value.lower()
        if len(text) != 64:
            raise ValueError("behavior signature must be a 64-hex-character SHA")
        try:
            int(text, 16)
        except ValueError as exc:
            raise ValueError("behavior signature must be hexadecimal") from exc
        return text

    a, b = normal(left), normal(right)
    return sum((int(x, 16) ^ int(y, 16)).bit_count() for x, y in zip(a, b, strict=True))


hamming_distance = behavior_distance


def _metric(record: ProgramRecord, *names: str) -> float | None:
    for name in names:
        value = record.search_metrics.get(name, record.metrics.get(name))
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    return None


def _median(record: ProgramRecord, *names: str) -> float | None:
    for name in names:
        value = record.search_metrics.get(name, record.metrics.get(name))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = sorted(
                float(item)
                for item in value
                if isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            )
            if values:
                mid = len(values) // 2
                return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return _metric(record, *names)


def _eligible(record: ProgramRecord) -> bool:
    if not record.unique or record.fitness_status.lower() not in {
        "complete",
        "completed",
        "ok",
        "pass",
        "passed",
        "evaluated",
        "verified",
    }:
        return False
    if record.validation_status.lower() not in {
        "valid",
        "pass",
        "passed",
        "ok",
        "complete",
        "completed",
        "verified",
    }:
        return False
    mode = (record.generation_mode or "").lower()
    identifier = record.program_id.lower()
    return not any(
        token in mode or token in identifier for token in ("random", "structural", "baseline")
    )


def _fitness_key(record: ProgramRecord) -> tuple[float, float, float, str]:
    pooled = _median(record, "pooled_median_auc", "pooled_auc", "auc_pooled", "median_auc")
    order10 = _median(
        record, "order10_median_auc", "order_10_median_auc", "order10_auc", "auc_order10"
    )
    witness = _median(record, "median_best_total_witness", "best_total_witness", "total_witness")
    # NaN-like/missing values sort after measured values.  Sorting descending on AUC
    # and ascending on witnesses is expressed directly by this key at call sites.
    return (
        pooled if pooled is not None else float("-inf"),
        order10 if order10 is not None else float("-inf"),
        witness if witness is not None else float("inf"),
        record.normalized_ast_sha256,
    )


@dataclass(frozen=True, slots=True)
class ParentSelection:
    parents: tuple[str, ...]
    slots: tuple[tuple[str, str], ...]
    fitness_elites: tuple[str, ...]
    diversity_parents: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "parents": list(self.parents),
            "slots": [[slot, parent] for slot, parent in self.slots],
            "fitness_elites": list(self.fitness_elites),
            "diversity_parents": list(self.diversity_parents),
        }


def select_parents(records: Iterable[ProgramRecord], *, slot_count: int = 8) -> ParentSelection:
    candidates = sorted(
        (record for record in records if _eligible(record)),
        key=lambda item: (
            -(_fitness_key(item)[0]),
            -(_fitness_key(item)[1]),
            _fitness_key(item)[2],
            _fitness_key(item)[3],
            item.program_id,
        ),
    )
    # AST dedup is normally done while writing.  Be defensive for callers passing
    # records directly and retain the first deterministic representative.
    unique: list[ProgramRecord] = []
    seen_ast: set[str] = set()
    for item in candidates:
        if item.normalized_ast_sha256 not in seen_ast:
            seen_ast.add(item.normalized_ast_sha256)
            unique.append(item)
    elites = unique[:4]
    selected = list(elites)
    diversity: list[ProgramRecord] = []
    remaining = [
        item for item in unique if item.program_id not in {entry.program_id for entry in selected}
    ]
    while len(selected) < min(slot_count, 8) and remaining:
        if len(selected) < 4:
            break

        def key(item: ProgramRecord) -> tuple[int, float, float, float, str, str]:
            distances = [
                behavior_distance(item.behavior_signature, parent.behavior_signature)
                for parent in selected
            ]
            fit = _fitness_key(item)
            return (
                -(min(distances) if distances else 0),
                -fit[0],
                -fit[1],
                fit[2],
                fit[3],
                item.program_id,
            )

        # max diversity; ties resolve by fitness, then AST SHA and id.
        chosen = min(remaining, key=key)
        diversity.append(chosen)
        selected.append(chosen)
        remaining.remove(chosen)
    # If fewer than four diversity candidates exist, continue in fitness order.
    for item in unique:
        if len(selected) >= min(slot_count, 8):
            break
        if item.program_id not in {entry.program_id for entry in selected}:
            selected.append(item)
    selected = selected[:slot_count]
    pairs = tuple((f"slot-{index:02d}", item.program_id) for index, item in enumerate(selected))
    return ParentSelection(
        tuple(item.program_id for item in selected),
        pairs,
        tuple(item.program_id for item in elites),
        tuple(item.program_id for item in diversity),
    )


choose_parents = select_parents
rank_parents = select_parents
