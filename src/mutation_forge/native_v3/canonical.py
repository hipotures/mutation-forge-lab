"""Strict parsing and canonical serialization for Native v3 programs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import NoReturn

from mutation_forge.models import JsonValue

CANONICAL_PROTOCOL_ID = "native_v3_canonical_json_v1"
PROGRAM_HASH_DOMAIN = b"mforge-native-v3-program\0"
_PRINTABLE_ASCII = re.compile(r"^[\x20-\x7e]*$")


class CanonicalJsonError(ValueError):
    """The value cannot be represented by the Native v3 canonical format."""


def _reject_float(value: str) -> NoReturn:
    raise CanonicalJsonError(f"JSON floating-point values are forbidden: {value!r}")


def _reject_constant(value: str) -> NoReturn:
    raise CanonicalJsonError(f"non-finite JSON values are forbidden: {value!r}")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJsonError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def parse_strict_json(raw: str, *, maximum_bytes: int) -> object:
    """Parse JSON while rejecting duplicate keys, floats, and oversized input."""

    encoded = raw.encode("utf-8")
    if len(encoded) > maximum_bytes:
        raise CanonicalJsonError(f"decoded JSON exceeds {maximum_bytes} UTF-8 bytes")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_object_from_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalJsonError("invalid JSON") from exc


def _validate_string(value: str) -> None:
    if not _PRINTABLE_ASCII.fullmatch(value):
        raise CanonicalJsonError("Native v3 AST strings must be printable ASCII")


def _canonical_string(value: str) -> str:
    _validate_string(value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _canonical(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        raise CanonicalJsonError("Native v3 canonical JSON forbids floats")
    if isinstance(value, str):
        return _canonical_string(value)
    if isinstance(value, Mapping):
        items: list[str] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise CanonicalJsonError("object keys must be strings")
            items.append(f"{_canonical_string(key)}:{_canonical(value[key])}")
        return "{" + ",".join(items) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    raise CanonicalJsonError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return the normative ASCII/UTF-8 representation."""

    return _canonical(value).encode("utf-8")


def domain_hash(domain: bytes, payload: bytes) -> str:
    """Hash a canonical payload under an explicit non-empty protocol domain."""

    if not domain or not domain.endswith(b"\0"):
        raise CanonicalJsonError("hash domain must be non-empty and NUL-terminated")
    return hashlib.sha256(domain + payload).hexdigest()


def program_hash(*, schema_version: str, canonical_program: bytes) -> str:
    _validate_string(schema_version)
    digest = hashlib.sha256()
    digest.update(PROGRAM_HASH_DOMAIN)
    digest.update(schema_version.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_program)
    return digest.hexdigest()


def json_value(value: object) -> JsonValue:
    """Narrow a validated canonical value for artifact serialization."""

    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    raise CanonicalJsonError(f"value is not strict JSON: {type(value).__name__}")
