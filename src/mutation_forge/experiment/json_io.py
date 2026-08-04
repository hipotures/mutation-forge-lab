"""Deterministic gzip JSON I/O for experiment workspace artifacts."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any

JSON_GZIP_SUFFIX = ".json.gz"


def _require_json_gzip(path: Path) -> None:
    if not path.name.endswith(JSON_GZIP_SUFFIX):
        raise ValueError(f"compressed JSON path must end with {JSON_GZIP_SUFFIX}: {path}")


def compress_json_bytes(payload: bytes) -> bytes:
    """Return deterministic gzip bytes for an encoded JSON document."""

    return gzip.compress(payload, compresslevel=6, mtime=0)


def read_json_bytes(path: str | Path) -> bytes:
    source = Path(path)
    _require_json_gzip(source)
    return gzip.decompress(source.read_bytes())


def read_json(path: str | Path) -> Any:
    return json.loads(read_json_bytes(path).decode("utf-8"))


def write_json_bytes(
    path: str | Path,
    payload: bytes,
    *,
    exclusive: bool = False,
) -> None:
    destination = Path(path)
    _require_json_gzip(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(compress_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        try:
            directory = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            pass
        else:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(
    path: str | Path,
    value: object,
    *,
    indent: int | None = None,
    exclusive: bool = False,
) -> None:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if indent is None:
        options["separators"] = (",", ":")
    else:
        options["indent"] = indent
    payload = json.dumps(value, **options).encode("utf-8") + b"\n"
    write_json_bytes(path, payload, exclusive=exclusive)


__all__ = [
    "JSON_GZIP_SUFFIX",
    "compress_json_bytes",
    "read_json",
    "read_json_bytes",
    "write_json",
    "write_json_bytes",
]
