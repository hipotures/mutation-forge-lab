"""Deterministic cross-panel selection and promotion contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from .scoring import RationalInterval

PROMOTION_PROTOCOL_ID = "native_v3_frozen_promotion_shortlist_v1"


class IncomparableFitness(ValueError):
    """Development results do not share an identical panel and protocol."""


@dataclass(frozen=True, slots=True)
class ProgramFitness:
    program_hash: str
    manifest_hash: str
    protocol_bundle_hash: str
    interval: RationalInterval
    exact_episode_count: int
    total_episode_count: int
    behavior_signature: str
    archive_eligible: bool = True

    @property
    def exactness(self) -> Fraction:
        if self.total_episode_count <= 0:
            return Fraction()
        return Fraction(self.exact_episode_count, self.total_episode_count)

    @property
    def width(self) -> Fraction:
        return self.interval.upper - self.interval.lower

    @property
    def cache_key(self) -> tuple[str, str, str]:
        return self.program_hash, self.manifest_hash, self.protocol_bundle_hash


@dataclass(frozen=True, slots=True)
class PromotionShortlist:
    epoch_id: str
    development_manifest_hash: str
    protocol_bundle_hash: str
    program_hashes: tuple[str, ...]
    protocol_id: str = PROMOTION_PROTOCOL_ID

    def __post_init__(self) -> None:
        if not 1 <= len(self.program_hashes) <= 4:
            raise ValueError("promotion shortlist must contain one to four programs")
        if len(set(self.program_hashes)) != len(self.program_hashes):
            raise ValueError("promotion shortlist programs must be unique")


def require_comparable(values: Iterable[ProgramFitness]) -> tuple[ProgramFitness, ...]:
    materialized = tuple(values)
    identities = {(value.manifest_hash, value.protocol_bundle_hash) for value in materialized}
    if len(identities) > 1:
        raise IncomparableFitness(
            "development fitness may only be compared under one manifest and protocol"
        )
    return materialized


def development_order(values: Iterable[ProgramFitness]) -> tuple[ProgramFitness, ...]:
    comparable = require_comparable(values)
    return tuple(
        sorted(
            comparable,
            key=lambda value: (
                -value.interval.lower,
                -value.exactness,
                value.width,
                -value.interval.upper,
                value.program_hash,
            ),
        )
    )


def freeze_promotion_shortlist(
    *,
    epoch_id: str,
    values: Iterable[ProgramFitness],
    maximum: int = 4,
) -> PromotionShortlist:
    if not 1 <= maximum <= 4:
        raise ValueError("promotion maximum must be in 1..4")
    ordered = [value for value in development_order(values) if value.archive_eligible]
    if not ordered:
        raise ValueError("no archive-eligible program can be promoted")
    selected: list[ProgramFitness] = [ordered.pop(0)]
    while ordered and len(selected) < maximum:
        signatures = {item.behavior_signature for item in selected}
        diverse = [item for item in ordered if item.behavior_signature not in signatures]
        chosen = (diverse or ordered)[0]
        selected.append(chosen)
        ordered.remove(chosen)
    first = selected[0]
    return PromotionShortlist(
        epoch_id=epoch_id,
        development_manifest_hash=first.manifest_hash,
        protocol_bundle_hash=first.protocol_bundle_hash,
        program_hashes=tuple(value.program_hash for value in selected),
    )


def validated_global_best(
    values: Iterable[ProgramFitness],
    *,
    validation_manifest_hash: str,
    protocol_bundle_hash: str,
) -> ProgramFitness:
    completed = [
        value
        for value in values
        if value.manifest_hash == validation_manifest_hash
        and value.protocol_bundle_hash == protocol_bundle_hash
    ]
    if not completed:
        raise ValueError("no program completed the locked validation panel")
    return development_order(completed)[0]


def missing_current_manifest_evaluations(
    *,
    program_hashes: Iterable[str],
    manifest_hash: str,
    protocol_bundle_hash: str,
    cache: Mapping[tuple[str, str, str], ProgramFitness],
) -> tuple[str, ...]:
    """Plan retained-parent and baseline work at epoch start."""

    return tuple(
        sorted(
            program_hash
            for program_hash in set(program_hashes)
            if (program_hash, manifest_hash, protocol_bundle_hash) not in cache
        )
    )
