from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TypedDict, cast

from mutation_forge.models import JsonValue

PROBE_SCHEMA_VERSION = "stage2a.probe.v1"
VALIDATOR_VERSION = "stage2a.validator.v1"
BEHAVIOR_SCHEMA_VERSION = "stage2a.behavior.v1"
ARTIFACT_SCHEMA_VERSION = "stage2a.artifact.v1"

MAX_DEPTH = 8
MAX_MAPPING_ENTRIES = 64
MAX_SEQUENCE_ITEMS = 256
MAX_STRING_BYTES = 4096
MAX_KEY_BYTES = 128
MAX_INTEGER_BITS = 4096
MAX_ABS_FLOAT = 1.0e100


class ProbeContext(TypedDict):
    probe_id: str
    step: int
    budget_remaining: int
    features: dict[str, JsonValue]


class ProbeProposal(TypedDict):
    proposal_id: str
    kind: str
    features: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    max_source_bytes: int = 12 * 1024
    max_ast_nodes: int = 500
    max_static_loop_bound: int = 256
    address_space_bytes: int = 128 * 1024 * 1024
    per_call_wall_seconds: float = 0.025
    total_wall_seconds: float = 60.0
    request_bytes: int = 64 * 1024
    response_bytes: int = 16 * 1024
    captured_output_bytes: int = 64 * 1024
    open_files: int = 16
    process_count: int = 1

    def as_dict(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


class ContractError(ValueError):
    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.path = path

    def as_dict(self) -> dict[str, JsonValue]:
        return {"code": self.code, "message": str(self), "path": self.path}


def _validate_plain_data(value: object, *, path: str, depth: int) -> JsonValue:
    if depth > MAX_DEPTH:
        raise ContractError("max_depth", f"input exceeds depth {MAX_DEPTH}", path)
    if value is None or isinstance(value, bool | str):
        if isinstance(value, str) and len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise ContractError(
                "string_too_large",
                f"string exceeds {MAX_STRING_BYTES} UTF-8 bytes",
                path,
            )
        return value
    if isinstance(value, int):
        if value.bit_length() > MAX_INTEGER_BITS:
            raise ContractError(
                "integer_too_large",
                f"integer exceeds {MAX_INTEGER_BITS} bits",
                path,
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > MAX_ABS_FLOAT:
            raise ContractError("invalid_float", "float must be finite and bounded", path)
        return value
    if isinstance(value, list | tuple):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise ContractError(
                "sequence_too_large",
                f"sequence exceeds {MAX_SEQUENCE_ITEMS} items",
                path,
            )
        return [
            _validate_plain_data(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if len(value) > MAX_MAPPING_ENTRIES:
            raise ContractError(
                "mapping_too_large",
                f"mapping exceeds {MAX_MAPPING_ENTRIES} entries",
                path,
            )
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("invalid_key", "mapping keys must be strings", path)
            if len(key.encode("utf-8")) > MAX_KEY_BYTES:
                raise ContractError(
                    "key_too_large",
                    f"mapping key exceeds {MAX_KEY_BYTES} UTF-8 bytes",
                    path,
                )
            normalized[key] = _validate_plain_data(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return normalized
    raise ContractError(
        "invalid_type",
        f"unsupported input type: {type(value).__name__}",
        path,
    )


def canonical_plain_data(value: object) -> JsonValue:
    """Validate JSON-compatible plain data and canonicalize tuples to lists."""
    return _validate_plain_data(value, path="$", depth=0)


def canonical_json_bytes(value: object) -> bytes:
    normalized = canonical_plain_data(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def validate_probe_inputs(
    ctx: object,
    proposal: object,
    *,
    max_request_bytes: int,
) -> tuple[ProbeContext, ProbeProposal]:
    normalized_ctx = canonical_plain_data(ctx)
    normalized_proposal = canonical_plain_data(proposal)
    if not isinstance(normalized_ctx, dict):
        raise ContractError("invalid_ctx", "ctx must be a mapping", "$.ctx")
    if not isinstance(normalized_proposal, dict):
        raise ContractError(
            "invalid_proposal",
            "proposal must be a mapping",
            "$.proposal",
        )
    _validate_typed_mapping(
        normalized_ctx,
        required={
            "probe_id": str,
            "step": int,
            "budget_remaining": int,
            "features": dict,
        },
        name="ctx",
    )
    _validate_typed_mapping(
        normalized_proposal,
        required={"proposal_id": str, "kind": str, "features": dict},
        name="proposal",
    )
    step = normalized_ctx["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ContractError("invalid_ctx", "ctx.step must be a non-negative integer")
    budget = normalized_ctx["budget_remaining"]
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ContractError(
            "invalid_ctx",
            "ctx.budget_remaining must be a non-negative integer",
        )
    payload = {"ctx": normalized_ctx, "proposal": normalized_proposal}
    if len(canonical_json_bytes(payload)) > max_request_bytes:
        raise ContractError(
            "request_too_large",
            f"request exceeds {max_request_bytes} bytes",
        )
    return (
        cast(ProbeContext, normalized_ctx),
        cast(ProbeProposal, normalized_proposal),
    )


def _validate_typed_mapping(
    value: dict[str, JsonValue],
    *,
    required: dict[str, type[object]],
    name: str,
) -> None:
    if set(value) != set(required):
        raise ContractError(
            f"invalid_{name}",
            f"{name} keys must be exactly {sorted(required)}",
            f"$.{name}",
        )
    for key, expected in required.items():
        item = value[key]
        if not isinstance(item, expected) or (
            expected is int and isinstance(item, bool)
        ):
            raise ContractError(
                f"invalid_{name}",
                f"{name}.{key} must be {expected.__name__}",
                f"$.{name}.{key}",
            )


def freeze_plain_data(value: JsonValue) -> object:
    """Recursively remove mutation APIs before data reaches generated code."""
    if isinstance(value, list):
        return tuple(freeze_plain_data(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: freeze_plain_data(item) for key, item in value.items()}
        )
    return value


def validate_priority(value: object, *, max_response_bytes: int) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(
            "invalid_output_type",
            "priority must return an int or float, not bool or a container",
            "$.priority",
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(
            "non_finite_output",
            "priority must return a finite number",
            "$.priority",
        )
    if isinstance(value, int) and value.bit_length() > MAX_INTEGER_BITS:
        raise ContractError(
            "output_integer_too_large",
            f"priority integer exceeds {MAX_INTEGER_BITS} bits",
            "$.priority",
        )
    try:
        encoded = json.dumps(
            {"priority": value},
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    except (ValueError, OverflowError) as error:
        raise ContractError("invalid_output", str(error), "$.priority") from error
    if len(encoded) > max_response_bytes:
        raise ContractError(
            "response_too_large",
            f"response exceeds {max_response_bytes} bytes",
            "$.priority",
        )
    return value
