"""Fail-closed contracts for the frozen Stage 4 search boundary."""
# Frozen contract lines intentionally remain compact.
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.validation import validate_policy

GENERATED_POLICY_SCHEMA_VERSION: Final = "stage4.generated_policy.v1"
MAX_SOURCE_BYTES: Final = 12 * 1024
MAX_TEXT_BYTES: Final = 2048
MAX_ITEM_BYTES: Final = 512
MAX_FIELDS: Final = 32
MAX_ITEMS: Final = 16
MAX_RESPONSE_BYTES: Final = 16 * 1024

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_ROOT = REPO_ROOT / "fixtures" / "stage4-seeds"
STAGE3_ROOT = REPO_ROOT / "runs" / "stage3-development" / "stage3-generation-1f7f0784e37c-attempt-01" / "revalidation" / "slots"
PROVENANCE = {
    "stage3_canonical_sha256": "43dee7e356ccc3f11c3fff326a78d16c70b0524a5b046732f6aca289335ccd73",
    "evidence_manifest_sha256": "b1cacc1340bddd548d9b4b3ed2b358c1a99e7d2b24bfe0c589ddc040c84359dc",
    "archive_sha256": "4acdc374914005681a0792abc84f4efd424e9a59292869278c144f7bf4bd4b4e",
}


class Stage4ContractError(ValueError):
    def __init__(self, code: str, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.code, self.path = code, path

    def as_dict(self) -> dict[str, JsonValue]:
        return {"code": self.code, "message": str(self), "path": self.path}


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage4ContractError("invalid_json", "value is not finite JSON") from exc


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def snapshot_hash(value: object) -> str:
    return canonical_hash(value)


def _text(value: object, name: str, limit: int, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage4ContractError("invalid_string", f"{name} must be non-empty", path)
    if len(value.encode()) > limit:
        raise Stage4ContractError("string_too_large", f"{name} exceeds {limit} bytes", path)
    return value


def _items(value: object, name: str, max_items: int, item_limit: int, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > max_items:
        raise Stage4ContractError("invalid_array", f"{name} must contain at most {max_items} strings", path)
    result = tuple(_text(v, f"{name}[{i}]", item_limit, f"{path}[{i}]") for i, v in enumerate(value))
    if len(set(result)) != len(result):
        raise Stage4ContractError("duplicate_value", f"{name} contains duplicates", path)
    return result


@dataclass(frozen=True, slots=True)
class GeneratedPolicy:
    source: str
    design_summary: str
    change_summary: str
    hypothesis: str
    used_fields: tuple[str, ...]
    assumptions: tuple[str, ...]
    expected_failure_modes: tuple[str, ...]
    schema_version: str = GENERATED_POLICY_SCHEMA_VERSION

    def as_dict(self) -> dict[str, JsonValue]:
        return {"schema_version": self.schema_version, "source": self.source, "design_summary": self.design_summary,
                "change_summary": self.change_summary, "hypothesis": self.hypothesis,
                "used_fields": list(self.used_fields), "assumptions": list(self.assumptions),
                "expected_failure_modes": list(self.expected_failure_modes)}

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.as_dict())

    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()


def _known_fields() -> set[str]:
    result: set[str] = set()
    for prefix, filename in (("ctx", "stage2b-context.schema.json"), ("proposal", "stage2b-proposal.schema.json")):
        try:
            raw = json.loads((REPO_ROOT / "configs" / "schemas" / filename).read_text())
            result.update(f"{prefix}.{name}" for name in raw.get("properties", {}))
        except (OSError, json.JSONDecodeError):
            pass
    return result


def parse_generated_policy(
    value: object,
    *,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    validate_source: bool = True,
) -> GeneratedPolicy:
    if not isinstance(value, dict):
        raise Stage4ContractError("invalid_output", "generated policy must be an object")
    expected = {"schema_version", "source", "design_summary", "change_summary", "hypothesis", "used_fields", "assumptions", "expected_failure_modes"}
    if set(value) != expected:
        raise Stage4ContractError("invalid_keys", f"output keys must be exactly {sorted(expected)}")
    if value["schema_version"] != GENERATED_POLICY_SCHEMA_VERSION:
        raise Stage4ContractError("invalid_schema_version", f"schema_version must be {GENERATED_POLICY_SCHEMA_VERSION!r}")
    source = _text(value["source"], "source", MAX_SOURCE_BYTES, "$.source")
    design = _text(value["design_summary"], "design_summary", MAX_TEXT_BYTES, "$.design_summary")
    change = _text(value["change_summary"], "change_summary", MAX_TEXT_BYTES, "$.change_summary")
    hypothesis = _text(value["hypothesis"], "hypothesis", MAX_TEXT_BYTES, "$.hypothesis")
    fields = _items(value["used_fields"], "used_fields", MAX_FIELDS, 128, "$.used_fields")
    unknown = [f for f in fields if f not in _known_fields()]
    if unknown:
        raise Stage4ContractError("unknown_field", f"unknown used field {unknown[0]!r}", "$.used_fields")
    assumptions = _items(value["assumptions"], "assumptions", MAX_ITEMS, MAX_ITEM_BYTES, "$.assumptions")
    failures = _items(value["expected_failure_modes"], "expected_failure_modes", MAX_ITEMS, MAX_ITEM_BYTES, "$.expected_failure_modes")
    result = GeneratedPolicy(source, design, change, hypothesis, fields, assumptions, failures)
    if len(result.canonical_bytes()) > max_response_bytes:
        raise Stage4ContractError("response_too_large", f"response exceeds {max_response_bytes} bytes")
    if validate_source:
        validation = validate_policy(source)
        if not validation.valid:
            raise Stage4ContractError("invalid_source", "source does not satisfy priority(ctx, proposal) contract")
    return result


def validate_generated_policy(value: object) -> GeneratedPolicy:
    return parse_generated_policy(value)


@dataclass(frozen=True, slots=True)
class SeedRecord:
    candidate_id: str
    slot_id: str
    source: str
    source_sha256: str
    ast_sha256: str
    behavior_signature: str
    design_summary: str
    used_fields: tuple[str, ...]
    assumptions: tuple[str, ...]
    provenance: dict[str, str]

    def as_dict(self) -> dict[str, JsonValue]:
        return {"candidate_id": self.candidate_id, "slot_id": self.slot_id, "source_sha256": self.source_sha256,
                "ast_sha256": self.ast_sha256, "behavior_signature": self.behavior_signature,
                "design_summary": self.design_summary, "used_fields": list(self.used_fields),
                "assumptions": list(self.assumptions), "provenance": dict(self.provenance)}


def _ast_hash(source: str) -> str | None:
    return validate_policy(source).identity.normalized_ast_sha256


def load_seed_capsule(root: str | Path = SEED_ROOT) -> tuple[SeedRecord, ...]:
    base = Path(root)
    metadata_manifest = REPO_ROOT / "configs" / "manifests" / "stage4-seeds-v1.json"
    listed: dict[str, dict[str, Any]] = {}
    if metadata_manifest.is_file():
        try:
            raw_manifest = json.loads(metadata_manifest.read_text(encoding="utf-8"))
            listed = {str(item["slot_id"]): cast(dict[str, Any], item) for item in raw_manifest.get("seeds", []) if isinstance(item, dict) and "slot_id" in item}
        except (OSError, json.JSONDecodeError, KeyError):
            listed = {}
    rows: list[SeedRecord] = []
    for i in range(8):
        slot = f"slot-{i:02d}"
        path = base / f"{slot}.py"
        if not path.is_file():
            path = base / slot / "source.py"
        source = path.read_text(encoding="utf-8")
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        canonical = STAGE3_ROOT / slot
        if (canonical / "identity.json").is_file():
            expected = json.loads((canonical / "identity.json").read_text())
        elif slot in listed:
            expected = listed[slot]
        else:
            raise Stage4ContractError("seed_metadata_missing", f"metadata missing for {slot}")
        expected_ast = expected.get("ast_sha256", expected.get("normalized_ast_sha256"))
        # The normalized AST digest is a frozen Stage-3 artifact; source bytes are
        # rehashed here and must match before that digest is accepted.
        if source_hash != expected["source_sha256"] or not isinstance(expected_ast, str):
            raise Stage4ContractError("seed_hash_mismatch", f"seed {slot} bytes/hash mismatch")
        if (canonical / "behavior.json").is_file():
            behavior = json.loads((canonical / "behavior.json").read_text())
            signature = cast(dict[str, Any], behavior.get("signature", {})).get("signature_sha256")
        else:
            signature = expected.get("behavior_signature")
        if not isinstance(signature, str):
            raise Stage4ContractError("seed_behavior_missing", f"behavior signature missing for {slot}")
        if (canonical / "canonical_response.json").is_file():
            response = json.loads((canonical / "canonical_response.json").read_text())
        else:
            response = expected
        ast_sha = expected.get("ast_sha256", expected.get("normalized_ast_sha256"))
        observed_ast = _ast_hash(source)
        if observed_ast != ast_sha:
            raise Stage4ContractError(
                "seed_ast_mismatch",
                f"seed {slot} normalized AST/hash mismatch",
            )
        rows.append(SeedRecord(f"stage3-{slot}", slot, source, source_hash, str(ast_sha), signature,
                               str(response.get("design_summary", "Hypothesis: retained Stage3 policy")),
                               tuple(response.get("used_fields", [])), tuple(response.get("assumptions", [])), dict(PROVENANCE)))
    if len({r.ast_sha256 for r in rows}) != 8 or len({r.behavior_signature for r in rows}) != 8:
        raise Stage4ContractError("seed_duplicate", "seed capsule must contain eight unique ASTs and behaviors")
    return tuple(rows)


def load_seed_manifest(path: str | Path) -> tuple[SeedRecord, ...]:
    rows = load_seed_capsule()
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != "stage4.seed_capsule.v1":
        raise Stage4ContractError("invalid_seed_manifest", "unexpected seed manifest schema")
    if raw.get("manifest_sha256") != canonical_hash({k: v for k, v in raw.items() if k != "manifest_sha256"}):
        raise Stage4ContractError("invalid_seed_manifest", "seed manifest hash mismatch")
    if raw.get("provenance") != PROVENANCE:
        raise Stage4ContractError("invalid_seed_manifest", "seed provenance mismatch")
    listed = raw.get("seeds")
    if not isinstance(listed, list) or len(listed) != 8:
        raise Stage4ContractError("invalid_seed_manifest", "seed manifest must list eight seeds")
    for expected, item in zip(rows, listed, strict=True):
        if not isinstance(item, dict) or item.get("slot_id") != expected.slot_id or item.get("source_sha256") != expected.source_sha256 or item.get("ast_sha256") != expected.ast_sha256 or item.get("behavior_signature") != expected.behavior_signature:
            raise Stage4ContractError("seed_manifest_mismatch", f"manifest mismatch for {expected.slot_id}")
    return rows


parse_stage4_generated_policy = parse_generated_policy
load_seed_capsule_manifest = load_seed_manifest

__all__ = ["GeneratedPolicy", "SeedRecord", "Stage4ContractError", "GENERATED_POLICY_SCHEMA_VERSION", "canonical_bytes", "canonical_hash", "snapshot_hash", "parse_generated_policy", "parse_stage4_generated_policy", "validate_generated_policy", "load_seed_capsule", "load_seed_manifest", "load_seed_capsule_manifest"]
