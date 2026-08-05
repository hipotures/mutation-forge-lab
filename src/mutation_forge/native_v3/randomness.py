"""Versioned deterministic random primitives used by Native v3."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

MASK64 = (1 << 64) - 1
UINT64_RANGE = 1 << 64
RANDOM_PROTOCOL_ID = "native_v3_splitmix64_v1"


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def derive_seed64(*parts: str | int) -> int:
    digest = hashlib.sha256()
    digest.update(b"mforge-native-v3-random\0")
    for part in parts:
        if isinstance(part, int):
            sign = b"-" if part < 0 else b"+"
            payload = sign + abs(part).to_bytes(max(1, (abs(part).bit_length() + 7) // 8), "big")
        else:
            payload = part.encode("utf-8")
        digest.update(_length_prefixed(payload))
    return int.from_bytes(digest.digest()[:8], "big")


def splitmix64(state: int) -> int:
    value = (state + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def draw64(seed: int, ordinal: int) -> int:
    if ordinal < 0:
        raise ValueError("draw ordinal must be non-negative")
    return splitmix64((seed + ordinal * 0x9E3779B97F4A7C15) & MASK64)


def uniform_below(seed: int, ordinal: int, upper: int) -> tuple[int, int]:
    """Return an unbiased draw in ``range(upper)`` and the next ordinal."""

    if upper <= 0 or upper > UINT64_RANGE:
        raise ValueError("uniform upper bound must be in [1, 2**64]")
    if upper == UINT64_RANGE:
        return draw64(seed, ordinal), ordinal + 1
    limit = UINT64_RANGE - (UINT64_RANGE % upper)
    while True:
        value = draw64(seed, ordinal)
        ordinal += 1
        if value < limit:
            return value % upper, ordinal


def weighted_index(seed: int, ordinal: int, weights: Iterable[int]) -> tuple[int, int]:
    normalized = tuple(weights)
    if not normalized or any(weight <= 0 for weight in normalized):
        raise ValueError("weights must be a non-empty sequence of positive integers")
    total = sum(normalized)
    if total > (1 << 63) - 1:
        raise ValueError("cumulative weight exceeds signed 64-bit")
    target, next_ordinal = uniform_below(seed, ordinal, total)
    cumulative = 0
    for index, weight in enumerate(normalized):
        cumulative += weight
        if target < cumulative:
            return index, next_ordinal
    raise AssertionError("weighted selection did not resolve")
