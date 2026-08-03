from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TypedDict, cast

from mutation_forge.models import JsonValue

PROBE_SCHEMA_VERSION = "stage2a.probe.v1"
SCIENTIFIC_CONTEXT_SCHEMA_VERSION = "mforge.scientific_context.v2"
SCIENTIFIC_PROPOSAL_SCHEMA_VERSION = "mforge.scientific_proposal.v2"
SCIENTIFIC_SELECTOR_TAGS = frozenset(
    {
        "uniform_random",
        "sampled_forbidden_cycle_anchored",
        "high_sampled_witness_load",
        "remote_from_anchor",
        "pairwise_distant_disjoint",
        "mixed_exploit_explore",
    }
)
VALIDATOR_VERSION = "stage2a.validator.v2"
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


class ScientificContext(TypedDict):
    schema_version: str
    order: int
    forbidden_lengths: list[int]
    capped_cycle_counts: list[int]
    weighted_penalty: int
    step: int
    remaining_steps: int
    stagnation: int
    recent_best_improvement: float
    recent_acceptance_rate: float
    recent_duplicate_rate: float


class ScientificProposal(TypedDict):
    schema_version: str
    proposal_id: str
    k: int
    operator_family: str
    selector_tags: list[str]
    anchor_forbidden_length: int | None
    broken_sampled_witnesses_by_length: list[int]
    removed_edge_load_sum_by_length: list[int]
    removed_edge_load_max_by_length: list[int]
    minimum_distance_between_removed_edges: int
    mean_distance_between_removed_edges: float
    minimum_preexisting_distance_for_new_edges: int
    mean_preexisting_distance_for_new_edges: float
    local_triangle_risk: int
    local_c4_risk: int
    reconnection_span: float


type RankerContext = ProbeContext | ScientificContext
type RankerProposal = ProbeProposal | ScientificProposal


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    max_source_bytes: int = 12 * 1024
    max_ast_nodes: int = 1000
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


def validate_ranker_inputs(
    ctx: object,
    proposal: object,
    *,
    max_request_bytes: int,
) -> tuple[RankerContext, RankerProposal]:
    normalized_ctx = canonical_plain_data(ctx)
    normalized_proposal = canonical_plain_data(proposal)
    if not isinstance(normalized_ctx, dict) or not isinstance(
        normalized_proposal,
        dict,
    ):
        raise ContractError(
            "invalid_ranker_input",
            "ctx and proposal must be mappings",
        )
    if normalized_ctx.get("schema_version") != SCIENTIFIC_CONTEXT_SCHEMA_VERSION:
        return validate_probe_inputs(
            normalized_ctx,
            normalized_proposal,
            max_request_bytes=max_request_bytes,
        )
    _validate_scientific_context(normalized_ctx)
    _validate_scientific_proposal(normalized_proposal)
    lengths = cast(list[JsonValue], normalized_ctx["forbidden_lengths"])
    for name in (
        "broken_sampled_witnesses_by_length",
        "removed_edge_load_sum_by_length",
        "removed_edge_load_max_by_length",
    ):
        vector = cast(list[JsonValue], normalized_proposal[name])
        if len(vector) != len(lengths):
            raise ContractError(
                "invalid_proposal",
                f"proposal.{name} must align with ctx.forbidden_lengths",
                f"$.proposal.{name}",
            )
    anchor = normalized_proposal["anchor_forbidden_length"]
    if anchor is not None and anchor not in lengths:
        raise ContractError(
            "invalid_proposal",
            "anchor_forbidden_length must occur in ctx.forbidden_lengths",
            "$.proposal.anchor_forbidden_length",
        )
    if (
        len(canonical_json_bytes({"ctx": normalized_ctx, "proposal": normalized_proposal}))
        > max_request_bytes
    ):
        raise ContractError(
            "request_too_large",
            f"request exceeds {max_request_bytes} bytes",
        )
    return (
        cast(ScientificContext, normalized_ctx),
        cast(ScientificProposal, normalized_proposal),
    )


def _require_exact_keys(
    value: dict[str, JsonValue],
    keys: set[str],
    name: str,
) -> None:
    if set(value) != keys:
        raise ContractError(
            f"invalid_{name}",
            f"{name} keys must be exactly {sorted(keys)}",
            f"$.{name}",
        )


def _nonnegative_int(value: JsonValue, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError("invalid_integer", "must be a non-negative integer", path)
    return value


def _finite_number(value: JsonValue, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ContractError("invalid_number", "must be a finite number", path)
    return value


def _int_list(value: JsonValue, path: str) -> list[int]:
    if not isinstance(value, list) or len(value) > 16:
        raise ContractError("invalid_integer_list", "must be a bounded integer list", path)
    result: list[int] = []
    for index, item in enumerate(value):
        result.append(_nonnegative_int(item, f"{path}[{index}]"))
    return result


def _validate_scientific_context(value: dict[str, JsonValue]) -> None:
    keys = {
        "schema_version",
        "order",
        "forbidden_lengths",
        "capped_cycle_counts",
        "weighted_penalty",
        "step",
        "remaining_steps",
        "stagnation",
        "recent_best_improvement",
        "recent_acceptance_rate",
        "recent_duplicate_rate",
    }
    _require_exact_keys(value, keys, "ctx")
    if value["schema_version"] != SCIENTIFIC_CONTEXT_SCHEMA_VERSION:
        raise ContractError("invalid_ctx", "unsupported scientific context schema")
    order = _nonnegative_int(value["order"], "$.ctx.order")
    if order < 4:
        raise ContractError("invalid_ctx", "ctx.order must be at least four")
    lengths = _int_list(value["forbidden_lengths"], "$.ctx.forbidden_lengths")
    counts = _int_list(value["capped_cycle_counts"], "$.ctx.capped_cycle_counts")
    if (
        not lengths
        or any(length < 1 for length in lengths)
        or len(set(lengths)) != len(lengths)
        or len(lengths) != len(counts)
    ):
        raise ContractError(
            "invalid_ctx",
            "forbidden lengths must be positive and unique; counts must align",
        )
    _nonnegative_int(value["weighted_penalty"], "$.ctx.weighted_penalty")
    _nonnegative_int(value["step"], "$.ctx.step")
    _nonnegative_int(value["remaining_steps"], "$.ctx.remaining_steps")
    _nonnegative_int(value["stagnation"], "$.ctx.stagnation")
    for name in (
        "recent_best_improvement",
        "recent_acceptance_rate",
        "recent_duplicate_rate",
    ):
        number = _finite_number(value[name], f"$.ctx.{name}")
        if name.endswith("_rate") and not 0 <= number <= 1:
            raise ContractError("invalid_ctx", f"ctx.{name} must be in [0, 1]")


def _validate_scientific_proposal(value: dict[str, JsonValue]) -> None:
    keys = {
        "schema_version",
        "proposal_id",
        "k",
        "operator_family",
        "selector_tags",
        "anchor_forbidden_length",
        "broken_sampled_witnesses_by_length",
        "removed_edge_load_sum_by_length",
        "removed_edge_load_max_by_length",
        "minimum_distance_between_removed_edges",
        "mean_distance_between_removed_edges",
        "minimum_preexisting_distance_for_new_edges",
        "mean_preexisting_distance_for_new_edges",
        "local_triangle_risk",
        "local_c4_risk",
        "reconnection_span",
    }
    _require_exact_keys(value, keys, "proposal")
    if value["schema_version"] != SCIENTIFIC_PROPOSAL_SCHEMA_VERSION:
        raise ContractError("invalid_proposal", "unsupported scientific proposal schema")
    proposal_id = value["proposal_id"]
    if (
        not isinstance(proposal_id, str)
        or len(proposal_id) != 64
        or any(character not in "0123456789abcdef" for character in proposal_id)
    ):
        raise ContractError(
            "invalid_proposal",
            "proposal_id must be a lowercase SHA-256 hex digest",
        )
    k = _nonnegative_int(value["k"], "$.proposal.k")
    if k not in {2, 3, 4}:
        raise ContractError("invalid_proposal", "proposal.k must be 2, 3, or 4")
    if value["operator_family"] != f"legal_{k}_switch":
        raise ContractError(
            "invalid_proposal",
            "operator_family must match proposal.k",
        )
    tags = value["selector_tags"]
    if (
        not isinstance(tags, list)
        or not tags
        or len(tags) > 8
        or any(tag not in SCIENTIFIC_SELECTOR_TAGS for tag in tags)
    ):
        raise ContractError(
            "invalid_proposal",
            "selector_tags must contain reviewed selector names",
        )
    anchor = value["anchor_forbidden_length"]
    if anchor is not None:
        _nonnegative_int(anchor, "$.proposal.anchor_forbidden_length")
    vector_names = (
        "broken_sampled_witnesses_by_length",
        "removed_edge_load_sum_by_length",
        "removed_edge_load_max_by_length",
    )
    vector_lengths = {len(_int_list(value[name], f"$.proposal.{name}")) for name in vector_names}
    if len(vector_lengths) != 1 or vector_lengths == {0}:
        raise ContractError("invalid_proposal", "proposal feature vectors must align")
    for name in (
        "minimum_distance_between_removed_edges",
        "minimum_preexisting_distance_for_new_edges",
        "local_triangle_risk",
        "local_c4_risk",
    ):
        _nonnegative_int(value[name], f"$.proposal.{name}")
    for name in (
        "mean_distance_between_removed_edges",
        "mean_preexisting_distance_for_new_edges",
        "reconnection_span",
    ):
        if _finite_number(value[name], f"$.proposal.{name}") < 0:
            raise ContractError(
                "invalid_proposal",
                f"proposal.{name} must be non-negative",
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
        if not isinstance(item, expected) or (expected is int and isinstance(item, bool)):
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
        return MappingProxyType({key: freeze_plain_data(item) for key, item in value.items()})
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
