"""Native v3 component evidence, exact energy intervals, and fitness."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum
from fractions import Fraction

from mutation_forge.models import JsonValue

from .randomness import UINT64_RANGE, splitmix64

SCORE_EVIDENCE_SCHEMA_VERSION = "mforge.native.score_evidence.v3"
SCORE_PROTOCOL_ID = "native_v3_score_50k_200k_v1"
FITNESS_PROTOCOL_ID = "native_v3_interval_utility_auc_v1"
ACCEPTANCE_PROTOCOL_ID = "native_v3_decimal_metropolis_v1"


class ScoreTimeoutWithoutPartial(TimeoutError):
    """A bounded scientific search timed out without sound partial evidence."""


class EvidenceStatus(StrEnum):
    EXACT = "EXACT"
    SATURATED_AT_CAP = "SATURATED_AT_CAP"
    SEARCH_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"
    SEARCH_TIMEOUT_WITH_SAFE_PARTIAL = "SEARCH_TIMEOUT_WITH_SAFE_PARTIAL"
    SEARCH_TIMEOUT_WITHOUT_PARTIAL = "SEARCH_TIMEOUT_WITHOUT_PARTIAL"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    PROGRAM_FAILURE = "PROGRAM_FAILURE"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"


class AttemptKind(StrEnum):
    INITIAL = "INITIAL"
    EXPANDED = "EXPANDED"


@dataclass(frozen=True, slots=True)
class IntegerInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval lower bound exceeds upper bound")

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    def as_dict(self) -> dict[str, JsonValue]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class RationalInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval lower bound exceeds upper bound")

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "lower": {
                "numerator": self.lower.numerator,
                "denominator": self.lower.denominator,
            },
            "upper": {
                "numerator": self.upper.numerator,
                "denominator": self.upper.denominator,
            },
        }


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    backend_id: str
    heg_commit: str
    source_tree_sha256: str
    binary_sha256: str
    compiler_identity: str
    build_flags: tuple[str, ...]
    platform: str
    architecture: str
    score_protocol_id: str = SCORE_PROTOCOL_ID

    def canonical_key(self) -> tuple[object, ...]:
        return (
            self.backend_id,
            self.heg_commit,
            self.source_tree_sha256,
            self.binary_sha256,
            self.compiler_identity,
            self.build_flags,
            self.platform,
            self.architecture,
            self.score_protocol_id,
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "backend_id": self.backend_id,
            "heg_commit": self.heg_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "binary_sha256": self.binary_sha256,
            "compiler_identity": self.compiler_identity,
            "build_flags": list(self.build_flags),
            "platform": self.platform,
            "architecture": self.architecture,
            "score_protocol_id": self.score_protocol_id,
        }


@dataclass(frozen=True, slots=True)
class CycleComponentEvidence:
    forbidden_length: int
    observed_count: int
    lower_bound: int
    upper_bound: int
    status: EvidenceStatus
    node_budget: int
    nodes_visited: int
    wall_time_ns: int
    attempt_kind: AttemptKind
    backend_identity: BackendIdentity

    def __post_init__(self) -> None:
        if self.forbidden_length < 3:
            raise ValueError("forbidden length must be at least three")
        if not 0 <= self.lower_bound <= self.observed_count <= self.upper_bound:
            raise ValueError("component evidence bounds are inconsistent")
        if self.node_budget <= 0 or self.nodes_visited < 0 or self.wall_time_ns < 0:
            raise ValueError("component resource counters are invalid")
        if (
            self.status in {EvidenceStatus.EXACT, EvidenceStatus.SATURATED_AT_CAP}
            and self.lower_bound != self.upper_bound
        ):
            raise ValueError("exact component evidence must have a point bound")

    @property
    def interval(self) -> IntegerInterval:
        return IntegerInterval(self.lower_bound, self.upper_bound)

    @property
    def scientifically_bounded(self) -> bool:
        return self.status in {
            EvidenceStatus.EXACT,
            EvidenceStatus.SATURATED_AT_CAP,
            EvidenceStatus.SEARCH_BUDGET_EXHAUSTED,
            EvidenceStatus.SEARCH_TIMEOUT_WITH_SAFE_PARTIAL,
        }

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "forbidden_length": self.forbidden_length,
            "observed_count": self.observed_count,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "status": self.status.value,
            "node_budget": self.node_budget,
            "nodes_visited": self.nodes_visited,
            "wall_time_ns": self.wall_time_ns,
            "attempt_kind": self.attempt_kind.value,
            "backend_identity": self.backend_identity.as_dict(),
        }


def witness_weight(forbidden_length: int) -> int:
    return max(1, 64 // forbidden_length)


@dataclass(frozen=True, slots=True)
class ScoreEvidence:
    graph_content_hash: str
    order: int
    edge_count: int
    witness_cap: int
    components: tuple[CycleComponentEvidence, ...]

    def __post_init__(self) -> None:
        lengths = tuple(component.forbidden_length for component in self.components)
        if tuple(sorted(set(lengths))) != lengths:
            raise ValueError("score evidence lengths must be sorted and unique")
        if self.order <= 0 or self.edge_count < 0 or self.witness_cap <= 0:
            raise ValueError("score evidence graph fields are invalid")
        identities = {component.backend_identity.canonical_key() for component in self.components}
        if len(identities) > 1:
            raise ValueError("score evidence mixes backend identities")

    @property
    def total_witness_interval(self) -> IntegerInterval:
        return IntegerInterval(
            sum(component.lower_bound for component in self.components),
            sum(component.upper_bound for component in self.components),
        )

    @property
    def weighted_penalty_interval(self) -> IntegerInterval:
        return IntegerInterval(
            sum(
                witness_weight(component.forbidden_length) * component.lower_bound
                for component in self.components
            ),
            sum(
                witness_weight(component.forbidden_length) * component.upper_bound
                for component in self.components
            ),
        )

    @property
    def complete_under_cap(self) -> bool:
        return all(
            component.status in {EvidenceStatus.EXACT, EvidenceStatus.SATURATED_AT_CAP}
            for component in self.components
        )

    @property
    def scientifically_bounded(self) -> bool:
        return all(component.scientifically_bounded for component in self.components)

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": SCORE_EVIDENCE_SCHEMA_VERSION,
            "graph_content_hash": self.graph_content_hash,
            "order": self.order,
            "edge_count": self.edge_count,
            "witness_cap": self.witness_cap,
            "components": [component.as_dict() for component in self.components],
            "total_witness_interval": self.total_witness_interval.as_dict(),
            "weighted_penalty_interval": self.weighted_penalty_interval.as_dict(),
            "complete_under_cap": self.complete_under_cap,
            "scientifically_bounded": self.scientifically_bounded,
        }


@dataclass(frozen=True, slots=True)
class EnergyScale:
    order: int
    forbidden_lengths: tuple[int, ...]
    witness_cap: int
    total_max: int
    weighted_max: int
    edge_min: int
    edge_max: int
    energy_min: int
    energy_max: int

    @classmethod
    def build(
        cls,
        *,
        order: int,
        forbidden_lengths: tuple[int, ...],
        witness_cap: int,
    ) -> EnergyScale:
        if order < 4 or witness_cap <= 0:
            raise ValueError("energy scale requires order >= 4 and positive cap")
        lengths = tuple(sorted(set(forbidden_lengths)))
        if lengths != forbidden_lengths or not lengths:
            raise ValueError("forbidden lengths must be non-empty, sorted, and unique")
        total_max = len(lengths) * witness_cap
        weighted_max = sum(witness_weight(length) * witness_cap for length in lengths)
        edge_min = (3 * order + 1) // 2
        edge_max = order * (order - 1) // 2
        edge_span = edge_max - edge_min
        energy_max = (total_max * (weighted_max + 1) + weighted_max) * (edge_span + 1) + edge_span
        return cls(
            order,
            lengths,
            witness_cap,
            total_max,
            weighted_max,
            edge_min,
            edge_max,
            0,
            energy_max,
        )

    def encode(self, *, total: int, weighted: int, edge_count: int) -> int:
        if not 0 <= total <= self.total_max:
            raise ValueError("total witnesses outside energy scale")
        if not 0 <= weighted <= self.weighted_max:
            raise ValueError("weighted penalty outside energy scale")
        if not self.edge_min <= edge_count <= self.edge_max:
            raise ValueError("edge count outside valid-graph energy scale")
        return (
            (total * (self.weighted_max + 1) + weighted) * (self.edge_max - self.edge_min + 1)
            + edge_count
            - self.edge_min
        )

    def interval(self, evidence: ScoreEvidence) -> IntegerInterval:
        if (
            evidence.order != self.order
            or evidence.witness_cap != self.witness_cap
            or tuple(component.forbidden_length for component in evidence.components)
            != self.forbidden_lengths
        ):
            raise ValueError("score evidence does not match energy scale")
        total = evidence.total_witness_interval
        weighted = evidence.weighted_penalty_interval
        return IntegerInterval(
            self.encode(
                total=total.lower,
                weighted=weighted.lower,
                edge_count=evidence.edge_count,
            ),
            self.encode(
                total=total.upper,
                weighted=weighted.upper,
                edge_count=evidence.edge_count,
            ),
        )

    def utility(self, energy: IntegerInterval) -> RationalInterval:
        span = self.energy_max - self.energy_min
        if span <= 0:
            raise ValueError("energy scale has no positive span")

        def value(item: int) -> Fraction:
            return Fraction(1, 1) - Fraction(item - self.energy_min, span)

        return RationalInterval(value(energy.upper), value(energy.lower))


def best_so_far_curve(values: Iterable[RationalInterval]) -> tuple[RationalInterval, ...]:
    curve: list[RationalInterval] = []
    lower: Fraction | None = None
    upper: Fraction | None = None
    for value in values:
        lower = value.lower if lower is None else max(lower, value.lower)
        upper = value.upper if upper is None else max(upper, value.upper)
        curve.append(RationalInterval(lower, upper))
    if not curve:
        raise ValueError("best-so-far curve cannot be empty")
    return tuple(curve)


def interval_mean(values: Iterable[RationalInterval]) -> RationalInterval:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("interval mean requires at least one value")
    count = len(materialized)
    return RationalInterval(
        sum((value.lower for value in materialized), Fraction()) / count,
        sum((value.upper for value in materialized), Fraction()) / count,
    )


def episode_auc(values: Iterable[RationalInterval], *, horizon: int) -> RationalInterval:
    materialized = tuple(values)
    if len(materialized) != horizon + 1:
        raise ValueError("episode trajectory must contain initial state plus horizon steps")
    return interval_mean(best_so_far_curve(materialized))


def aggregate_order_balanced(
    values: Mapping[int, Iterable[RationalInterval]],
) -> RationalInterval:
    if not values:
        raise ValueError("program fitness requires at least one graph order")
    per_order = [interval_mean(tuple(values[order])) for order in sorted(values)]
    return interval_mean(per_order)


@dataclass(frozen=True, slots=True)
class ScoreEvidenceCacheKey:
    labeled_graph_content_hash: str
    forbidden_lengths: tuple[int, ...]
    witness_cap: int
    node_budget: int
    attempt_kind: AttemptKind
    score_protocol_id: str
    backend_identity: tuple[object, ...]


class ScoreEvidenceCache:
    def __init__(self) -> None:
        self._values: dict[ScoreEvidenceCacheKey, ScoreEvidence] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: ScoreEvidenceCacheKey) -> ScoreEvidence | None:
        value = self._values.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: ScoreEvidenceCacheKey, value: ScoreEvidence) -> None:
        existing = self._values.get(key)
        if existing is not None and existing != value:
            raise ValueError("conflicting score evidence for one cache key")
        self._values[key] = value

    @property
    def hit_rate(self) -> Fraction:
        total = self.hits + self.misses
        return Fraction(self.hits, total) if total else Fraction()


_DECIMAL_CONTEXT = Context(
    prec=80,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
)


def metropolis_threshold(delta: Fraction, temperature: Fraction) -> int:
    if delta < 0:
        raise ValueError("Metropolis threshold requires non-negative delta")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if delta == 0:
        return UINT64_RANGE
    with localcontext(_DECIMAL_CONTEXT):
        ratio = (
            Decimal(delta.numerator)
            / Decimal(delta.denominator)
            / Decimal(temperature.numerator)
            * Decimal(temperature.denominator)
        )
        probability = (-ratio).exp()
        threshold = int(
            (probability * Decimal(UINT64_RANGE)).to_integral_value(rounding=ROUND_FLOOR)
        )
    return max(0, min(UINT64_RANGE, threshold))


def acceptance_seed64(
    *,
    protocol_id: str,
    program_hash: str,
    episode_id: str,
    policy_seed: int,
    step_index: int,
    ast_path: str,
    repeat_indices: tuple[int, ...],
    invocation_ordinal: int,
    draw_ordinal: int,
) -> int:
    digest = hashlib.sha256()
    digest.update(b"mforge-native-v3-acceptance\0")
    parts = (
        protocol_id,
        program_hash,
        episode_id,
        str(policy_seed),
        str(step_index),
        ast_path,
        ",".join(str(index) for index in repeat_indices),
        str(invocation_ordinal),
        str(draw_ordinal),
    )
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def metropolis_accepts(
    *,
    delta: Fraction,
    temperature: Fraction,
    seed64: int,
) -> tuple[bool, int, int]:
    threshold = metropolis_threshold(delta, temperature)
    draw = splitmix64(seed64)
    return threshold == UINT64_RANGE or draw < threshold, threshold, draw
