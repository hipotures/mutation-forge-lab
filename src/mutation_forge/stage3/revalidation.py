"""Offline revalidation of retained Stage 3 generation responses."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from mutation_forge.sandbox.contracts import VALIDATOR_VERSION
from mutation_forge.sandbox.validation import validate_policy

from .artifacts import canonical_hash
from .config import Stage3GenerationConfig
from .contracts import parse_generated_policy
from .generation import SLOTS, _behavior, _sha_source

REVALIDATION_SCHEMA_VERSION = "stage3.generation_revalidation.v1"
MAX_REVALIDATION_JSON_BYTES = 1_048_576
_USAGE_KEYS = (
    "inputTokens",
    "cachedInputTokens",
    "cacheWriteInputTokens",
    "outputTokens",
    "reasoningOutputTokens",
    "totalTokens",
)


def _atomic_bytes(path: Path, payload: bytes, *, max_bytes: int) -> None:
    if len(payload) > max_bytes:
        raise RuntimeError(f"revalidation artifact exceeds bound: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    _atomic_bytes(path, payload, max_bytes=MAX_REVALIDATION_JSON_BYTES)


def _write_source(path: Path, source: str, *, max_bytes: int) -> None:
    _atomic_bytes(path, source.encode(), max_bytes=max_bytes)


def _final_response_path(root: Path, slot: str) -> Path:
    slot_root = root / "slots" / slot
    repair = slot_root / f"{slot}.repair.response.json"
    initial = slot_root / f"{slot}.response.json"
    path = repair if repair.is_file() else initial
    if not path.is_file():
        raise FileNotFoundError(f"{slot} retained response is missing")
    return path


def _read_candidate(path: Path, *, model: str, effort: str) -> tuple[str, dict[str, Any]]:
    outer = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(outer, dict)
        or outer.get("status") != "completed"
        or outer.get("accepted") is not True
        or outer.get("content") is not True
        or outer.get("model") != model
        or outer.get("effort") != effort
    ):
        raise ValueError("retained final response has incomplete provenance")
    response = outer.get("response")
    if not isinstance(response, str):
        raise ValueError("retained final response has no structured content")
    value = json.loads(response)
    generated = parse_generated_policy(value)
    return generated.source, cast(dict[str, Any], value)


def _baseline_identities(config: Stage3GenerationConfig) -> tuple[set[str], set[str]]:
    source_hashes: set[str] = set()
    ast_hashes: set[str] = set()
    for path in (config.random_policy_path, config.structural_policy_path):
        source = path.read_text(encoding="utf-8")
        validation = validate_policy(source, config.sandbox)
        if not validation.valid or validation.identity.normalized_ast_sha256 is None:
            raise RuntimeError(f"baseline policy no longer validates: {path.name}")
        source_hashes.add(_sha_source(source))
        ast_hashes.add(validation.identity.normalized_ast_sha256)
    return source_hashes, ast_hashes


def _retained_turn_accounting(
    root: Path, *, model: str, effort: str
) -> tuple[int, int, dict[str, int]]:
    initial_count = 0
    repair_count = 0
    totals = {key: 0 for key in _USAGE_KEYS}
    for slot in SLOTS:
        slot_root = root / "slots" / slot
        paths = [slot_root / f"{slot}.response.json"]
        repair = slot_root / f"{slot}.repair.response.json"
        if repair.is_file():
            paths.append(repair)
        initial_count += 1
        repair_count += len(paths) - 1
        for path in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            usage = value.get("usage") if isinstance(value, dict) else None
            if (
                not isinstance(value, dict)
                or value.get("status") != "completed"
                or value.get("accepted") is not True
                or value.get("content") is not True
                or value.get("model") != model
                or value.get("effort") != effort
                or not isinstance(usage, dict)
                or usage.get("final") is not True
                or usage.get("partial") is not False
            ):
                raise ValueError(f"{slot} retained turn provenance is incomplete")
            for key in _USAGE_KEYS:
                amount = usage.get(key)
                if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
                    raise ValueError(f"{slot} retained turn usage is invalid")
                totals[key] += amount
            if bool(value.get("charged")) != (usage["totalTokens"] > 0):
                raise ValueError(f"{slot} retained turn charge accounting is invalid")
    return initial_count, repair_count, totals


def revalidate_saved_generation(
    config: Stage3GenerationConfig,
    run: str | Path,
    *,
    persist: bool,
) -> dict[str, Any]:
    """Revalidate final retained responses without calling a generation provider."""
    root = Path(run).resolve()
    generation_path = root / "generation_summary.json"
    generation_bytes = generation_path.read_bytes()
    generation_summary = json.loads(generation_bytes)
    if not isinstance(generation_summary, dict):
        raise ValueError("generation summary must be an object")
    initial_count, repair_count, usage_totals = _retained_turn_accounting(
        root,
        model=config.model.name,
        effort=config.model.effort,
    )
    source_freeze_path = root / "freeze.json"
    source_freeze = json.loads(source_freeze_path.read_text(encoding="utf-8"))
    if not isinstance(source_freeze, dict):
        raise ValueError("source generation freeze must be an object")

    seen_sources, seen_asts = _baseline_identities(config)
    slots: list[dict[str, Any]] = []
    unique_count = 0
    total_smoke_calls = 0
    for slot in SLOTS:
        response_path = _final_response_path(root, slot)
        relative_response = response_path.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "slot": slot,
            "response_path": relative_response,
            "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
            "status": "failed",
            "valid": False,
            "duplicate": False,
            "errors": [],
            "source_sha256": None,
            "normalized_ast_sha256": None,
            "behavior_signature_sha256": None,
            "smoke_calls": 0,
        }
        source: str | None = None
        identity: dict[str, Any] | None = None
        behavior: dict[str, Any] | None = None
        telemetry: dict[str, Any] | None = None
        canonical_response: dict[str, Any] | None = None
        try:
            source, canonical_response = _read_candidate(
                response_path,
                model=config.model.name,
                effort=config.model.effort,
            )
            validation = validate_policy(source, config.sandbox)
            record["errors"] = [error.as_dict() for error in validation.errors]
            if not validation.valid or validation.identity.normalized_ast_sha256 is None:
                raise ValueError("static validation failed")
            identity = cast(dict[str, Any], validation.identity.as_dict())
            behavior_value, telemetry_value = _behavior(source, config.sandbox, 10_000)
            behavior = dict(behavior_value)
            telemetry = dict(telemetry_value)
            source_sha256 = _sha_source(source)
            normalized_ast_sha256 = validation.identity.normalized_ast_sha256
            duplicate = source_sha256 in seen_sources or normalized_ast_sha256 in seen_asts
            record.update(
                {
                    "status": "duplicate" if duplicate else "accepted",
                    "valid": True,
                    "duplicate": duplicate,
                    "source_sha256": source_sha256,
                    "normalized_ast_sha256": normalized_ast_sha256,
                    "behavior_signature_sha256": behavior.get("signature_sha256"),
                    "smoke_calls": 10_000,
                }
            )
            total_smoke_calls += 10_000
            if not duplicate:
                seen_sources.add(source_sha256)
                seen_asts.add(normalized_ast_sha256)
                unique_count += 1
        except Exception as error:
            if not record["errors"]:
                record["errors"] = [
                    {"code": type(error).__name__, "message": str(error)[:256]}
                ]
        slots.append(record)
        if persist:
            slot_root = root / "revalidation" / "slots" / slot
            _write_json(slot_root / "result.json", record)
            if source is not None and bool(record["valid"]):
                _write_source(
                    slot_root / "source.py",
                    source,
                    max_bytes=config.sandbox.max_source_bytes,
                )
                _write_json(slot_root / "canonical_response.json", canonical_response)
                _write_json(slot_root / "identity.json", identity)
                _write_json(slot_root / "behavior.json", {"signature": behavior})
                _write_json(slot_root / "worker_telemetry.json", telemetry)

    summary: dict[str, Any] = {
        "schema_version": REVALIDATION_SCHEMA_VERSION,
        "status": "completed",
        "decision": "READY_FOR_EVALUATION",
        "source_run": root.name,
        "source_generation_summary_sha256": hashlib.sha256(generation_bytes).hexdigest(),
        "source_generation_freeze_sha256": hashlib.sha256(
            source_freeze_path.read_bytes()
        ).hexdigest(),
        "source_generation_tag": source_freeze.get("preregistration_tag"),
        "source_generation_status": generation_summary.get("status"),
        "validator_version": VALIDATOR_VERSION,
        "provider_calls": 0,
        "model_calls": 0,
        "app_server_calls": 0,
        "initial_turn_count": initial_count,
        "repair_turn_count": repair_count,
        "total_live_turns": initial_count + repair_count,
        "completed_turns": initial_count + repair_count,
        "accepted_model_turns": initial_count + repair_count,
        "source_model_turns": initial_count + repair_count,
        "initial_max_active": generation_summary.get("initial_max_active"),
        "max_active": generation_summary.get("max_active"),
        "exact_usage_complete": True,
        "usage_totals": usage_totals,
        "slots": slots,
        "unique_count": unique_count,
        "all_valid": all(bool(slot["valid"]) for slot in slots),
        "total_smoke_calls": total_smoke_calls,
    }
    summary["canonical_revalidation_sha256"] = canonical_hash(summary)
    if persist:
        _write_json(root / "revalidation_summary.json", summary)
    return summary


def replay_saved_revalidation(
    config: Stage3GenerationConfig, run: str | Path
) -> dict[str, Any]:
    """Reproduce a persisted revalidation without a provider or model call."""
    root = Path(run).resolve()
    expected = json.loads((root / "revalidation_summary.json").read_text(encoding="utf-8"))
    actual = revalidate_saved_generation(config, root, persist=False)
    valid = expected == actual
    return {
        **actual,
        "replay_validated": valid,
        "replay_errors": [] if valid else [{"code": "revalidation_mismatch"}],
    }


__all__ = [
    "REVALIDATION_SCHEMA_VERSION",
    "replay_saved_revalidation",
    "revalidate_saved_generation",
]
