from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

IMPLEMENTATION_ID = "mforge-independent-cycle-mitm-v2"


class Graph6Error(ValueError):
    pass


def _six_bits(value: int) -> int:
    result = value - 63
    if not 0 <= result <= 63:
        raise Graph6Error("graph6 contains a byte outside the 6-bit alphabet")
    return result


def _decode_order(data: bytes) -> tuple[int, int]:
    if not data:
        raise Graph6Error("graph6 input is empty")
    first = _six_bits(data[0])
    if first < 63:
        return first, 1
    if len(data) < 4:
        raise Graph6Error("truncated graph6 order")
    second = _six_bits(data[1])
    if second < 63:
        return (
            (second << 12) | (_six_bits(data[2]) << 6) | _six_bits(data[3]),
            4,
        )
    if len(data) < 8:
        raise Graph6Error("truncated large graph6 order")
    order = 0
    for byte in data[2:8]:
        order = (order << 6) | _six_bits(byte)
    return order, 8


def decode_graph6(value: bytes) -> tuple[tuple[int, ...], ...]:
    data = value.strip()
    header = b">>graph6<<"
    if data.startswith(header):
        data = data[len(header) :]
    order, offset = _decode_order(data)
    needed_bits = order * (order - 1) // 2
    encoded = data[offset:]
    if len(encoded) * 6 < needed_bits:
        raise Graph6Error("truncated graph6 adjacency payload")
    adjacency = [set[int]() for _ in range(order)]
    bit_index = 0
    for high in range(1, order):
        for low in range(high):
            byte = _six_bits(encoded[bit_index // 6])
            bit = (byte >> (5 - bit_index % 6)) & 1
            bit_index += 1
            if bit:
                adjacency[high].add(low)
                adjacency[low].add(high)
    padding = len(encoded) * 6 - needed_bits
    if padding >= 6:
        raise Graph6Error("graph6 contains trailing adjacency bytes")
    if padding and encoded:
        mask = (1 << padding) - 1
        if _six_bits(encoded[-1]) & mask:
            raise Graph6Error("graph6 contains non-zero padding bits")
    return tuple(tuple(sorted(neighbors)) for neighbors in adjacency)


def _connected(adjacency: tuple[tuple[int, ...], ...]) -> bool:
    if not adjacency:
        return False
    seen = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return len(seen) == len(adjacency)


def _find_cycle(
    adjacency: tuple[tuple[int, ...], ...],
    length: int,
) -> tuple[int, ...] | None:
    order = len(adjacency)
    if length > order:
        return None
    half_length = length // 2
    for start in range(order):
        paths_by_endpoint: dict[int, list[tuple[tuple[int, ...], int]]] = {}
        stack: list[tuple[int, tuple[int, ...], int]] = [
            (start, (start,), 1 << start)
        ]
        while stack:
            vertex, path, mask = stack.pop()
            if len(path) == half_length + 1:
                allowed_overlap = (1 << start) | (1 << vertex)
                for other_path, other_mask in paths_by_endpoint.get(vertex, ()):
                    if mask & other_mask == allowed_overlap:
                        return path + tuple(reversed(other_path[1:-1]))
                paths_by_endpoint.setdefault(vertex, []).append((path, mask))
                continue
            for neighbor in adjacency[vertex]:
                bit = 1 << neighbor
                if neighbor <= start or mask & bit:
                    continue
                stack.append((neighbor, (*path, neighbor), mask | bit))
    return None


def verify(path: Path) -> dict[str, Any]:
    started = time.monotonic()
    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        adjacency = decode_graph6(path.read_bytes())
    except (OSError, Graph6Error) as exc:
        return {
            "status": "INVALID",
            "complete": True,
            "message": str(exc),
            "implementation": IMPLEMENTATION_ID,
            "implementation_sha256": implementation_sha256,
            "witnesses": [],
            "elapsed_seconds": time.monotonic() - started,
        }
    order = len(adjacency)
    lengths: list[int] = []
    length = 4
    while length <= order:
        lengths.append(length)
        length *= 2
    if order == 0:
        message = "graph is empty"
    elif not _connected(adjacency):
        message = "graph is disconnected"
    elif min(map(len, adjacency)) < 3:
        message = "minimum degree is below 3"
    else:
        message = ""
    if message:
        return {
            "status": "INVALID",
            "complete": True,
            "message": message,
            "implementation": IMPLEMENTATION_ID,
            "implementation_sha256": implementation_sha256,
            "target_forbidden_lengths": lengths,
            "witnesses": [],
            "elapsed_seconds": time.monotonic() - started,
        }
    for forbidden_length in lengths:
        witness = _find_cycle(adjacency, forbidden_length)
        if witness is not None:
            return {
                "status": "REJECTED",
                "complete": True,
                "message": f"found a cycle of length {forbidden_length}",
                "implementation": IMPLEMENTATION_ID,
                "implementation_sha256": implementation_sha256,
                "target_forbidden_lengths": lengths,
                "witnesses": [[f"C{forbidden_length}", list(witness)]],
                "elapsed_seconds": time.monotonic() - started,
            }
    return {
        "status": "VERIFIED",
        "complete": True,
        "message": "all target cycle lengths are absent",
        "implementation": IMPLEMENTATION_ID,
        "implementation_sha256": implementation_sha256,
        "target_forbidden_lengths": lengths,
        "witnesses": [],
        "elapsed_seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print(
            json.dumps(
                {
                    "status": "UNKNOWN",
                    "complete": False,
                    "message": "expected one candidate.graph6 path",
                    "implementation": IMPLEMENTATION_ID,
                    "witnesses": [],
                },
                sort_keys=True,
            )
        )
        return 2
    result = verify(Path(arguments[0]))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] in {"VERIFIED", "REJECTED", "INVALID"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
