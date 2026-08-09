"""Deterministic, platform-independent randomness for Native v3 programs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

RANDOM_PROTOCOL_ID = "native_v3_splitmix64_v1"
_RANDOM_DOMAIN = b"mforge-native-v3-random\0"
_MASK64 = (1 << 64) - 1
_UINT64_RANGE = 1 << 64


def derive_seed64(*parts: str | int) -> int:
    """Derive a stable unsigned 64-bit seed from length-prefixed components."""

    digest = hashlib.sha256()
    digest.update(_RANDOM_DOMAIN)
    for part in parts:
        if isinstance(part, int):
            sign = b"-" if part < 0 else b"+"
            magnitude = abs(part)
            width = max(1, (magnitude.bit_length() + 7) // 8)
            encoded = b"i" + sign + magnitude.to_bytes(width, "big")
        else:
            encoded = b"s" + part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def splitmix64(value: int) -> int:
    """Return the SplitMix64 output for one unsigned 64-bit input."""

    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def draw64(seed: int, ordinal: int) -> int:
    """Draw one deterministic unsigned 64-bit value."""

    if ordinal < 0:
        raise ValueError("draw ordinal must be non-negative")
    return splitmix64((seed + ordinal) & _MASK64)


def uniform_below(seed: int, upper_bound: int) -> tuple[int, int]:
    """Draw uniformly from ``range(upper_bound)`` and return draws consumed."""

    if not 1 <= upper_bound <= _UINT64_RANGE:
        raise ValueError("upper_bound must be in 1..2**64")
    rejection_limit = _UINT64_RANGE - (_UINT64_RANGE % upper_bound)
    ordinal = 0
    while True:
        value = draw64(seed, ordinal)
        ordinal += 1
        if value < rejection_limit:
            return value % upper_bound, ordinal


def weighted_index(seed: int, weights: Sequence[int]) -> tuple[int, int]:
    """Choose an index using positive integer weights."""

    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("weights must be a non-empty sequence of positive integers")
    total = sum(weights)
    if total > (1 << 63) - 1:
        raise ValueError("weight sum exceeds signed 64-bit range")
    target, draws = uniform_below(seed, total)
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index, draws
    raise AssertionError("weighted choice did not select an index")
