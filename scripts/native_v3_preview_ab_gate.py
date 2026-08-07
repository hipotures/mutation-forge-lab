#!/usr/bin/env python3
"""Compare one completed selected preview with one four-program rollback batch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3.experiment import (
    MULTI_PROGRAM_BATCH,
    V3_DEFAULT_COMMUNICATION_MODE,
)
from mutation_forge.native_v3.persistent_experiment import (
    BRIEF_IDS,
    run_live_batch_reference,
)
from mutation_forge.native_v3.preview import PERSISTENT_SINGLE_AST
from mutation_forge.native_v3.single_program_ir import (
    SLOT_SPECIFIC_OUTPUT_CONTRACT,
)

REPORT_SCHEMA_VERSION = "mforge.native-v3.preview-ab-gate.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--auth-json", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--turn-timeout", type=float, default=600.0)
    return parser


def _object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not an object")
    return {str(key): item for key, item in value.items()}


def _selected_summary(root: Path) -> dict[str, Any]:
    output = root / "native-v3-output" / "epoch-0000"
    report = _object(output / "cohort-report.json.gz")
    state = _object(output / "communication-state.json.gz")
    manifest = _object(output / "epoch-manifest.json.gz")
    if (
        report.get("provider_mode") != PERSISTENT_SINGLE_AST
        or report.get("output_contract") != SLOT_SPECIFIC_OUTPUT_CONTRACT
        or state.get("status") != "completed"
    ):
        raise ValueError("selected preview identity is not accepted")
    briefs = [
        item.get("brief_id")
        for item in manifest.get("slots", [])
        if isinstance(item, Mapping)
    ]
    if briefs != list(BRIEF_IDS):
        raise ValueError("selected preview briefs do not match the rollback gate")
    slot_reports = state.get("slot_reports")
    if not isinstance(slot_reports, list):
        raise ValueError("selected preview slot reports are unavailable")
    hashes = [
        item.get("entry", {}).get("program_hash")
        for item in slot_reports
        if isinstance(item, Mapping) and isinstance(item.get("entry"), Mapping)
    ]
    artifact_parity = all(
        isinstance(item, Mapping)
        and isinstance(item.get("attempts"), list)
        and all(
            isinstance(attempt, Mapping)
            and attempt.get("artifact_complete") is True
            for attempt in item["attempts"]
        )
        for item in slot_reports
    )
    return {
        "provider_mode": report["provider_mode"],
        "output_contract": report["output_contract"],
        "brief_ids": briefs,
        "program_turns": report.get("program_turns"),
        "valid_slots": report.get("valid_slots"),
        "unique_valid_programs": report.get("unique_valid_programs"),
        "program_hashes": hashes,
        "time_to_first_valid_ast_ms": report.get("time_to_first_valid_ast_ms"),
        "provider_retries": state.get("provider_retries"),
        "provider_warnings": state.get("provider_warnings"),
        "artifact_parity": artifact_parity,
        "usage": report.get("usage"),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.workspace.exists():
        raise SystemExit("workspace must not exist")
    if not args.auth_json.is_file():
        raise SystemExit("auth JSON does not exist")
    selected = _selected_summary(args.selected_root)
    provider = LocalCodexAppServerProvider(
        model=args.model,
        effort=args.effort,
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=args.turn_timeout / 2,
        auth_json=args.auth_json,
        persist_artifacts=False,
    )
    try:
        rollback = run_live_batch_reference(args.workspace, provider=provider)
    finally:
        provider.close()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "brief_ids": list(BRIEF_IDS),
        "selected": selected,
        "rollback": rollback,
        "gate": {
            "selected_valid_unique_4_of_4": (
                selected["valid_slots"] == 4
                and selected["unique_valid_programs"] == 4
                and len(set(selected["program_hashes"])) == 4
            ),
            "selected_zero_retries_and_warnings": (
                selected["provider_retries"] == 0
                and selected["provider_warnings"] == 0
            ),
            "selected_artifact_parity": selected["artifact_parity"],
            "rollback_artifact_parity": rollback["turn_artifact_complete"],
            "production_default_changed": (
                V3_DEFAULT_COMMUNICATION_MODE != MULTI_PROGRAM_BATCH
            ),
        },
    }
    write_json(args.workspace / "preview-ab-report.json.gz", report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
