#!/usr/bin/env python3
"""Run the bounded Native v3 Step 12B A/B/C experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import write_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3.persistent_experiment import (
    run_ab_experiment,
    run_live_batch_reference,
)
from mutation_forge.stage3.app_server import AppServerLimits, CodexAppServerAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--auth-json", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--turn-timeout", type=float, default=600.0)
    return parser


def _cost_per_valid(value: dict[str, Any]) -> float | None:
    valid = sum(
        turn["program_hash"] is not None
        for turn in value["turns"]
    )
    return value["usage"]["totalTokens"] / valid if valid else None


def main() -> int:
    args = _parser().parse_args()
    if args.workspace.exists():
        raise SystemExit("workspace must not exist")
    if not args.auth_json.is_file():
        raise SystemExit("auth JSON does not exist")

    def factory(base_instructions: str, prefix: str) -> CodexAppServerAdapter:
        return CodexAppServerAdapter(
            auth_json=args.auth_json,
            limits=AppServerLimits(
                max_turns=5 if prefix == "b-bootstrap" else 1,
                max_campaigns=1,
                turn_timeout=args.turn_timeout,
            ),
            base_instructions=base_instructions,
            compress_json_artifacts=True,
            sandbox_mode="danger-full-access",
            approval_policy="never",
        )

    report = run_ab_experiment(
        args.workspace,
        model=args.model,
        effort=args.effort,
        forbidden_lengths=(4, 8, 16),
        adapter_factory=factory,
    )
    batch_provider = LocalCodexAppServerProvider(
        model=args.model,
        effort=args.effort,
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=args.turn_timeout / 2,
        auth_json=args.auth_json,
        persist_artifacts=False,
    )
    try:
        report["C_existing_batch"] = run_live_batch_reference(
            args.workspace / "c-reference",
            provider=batch_provider,
        )
    finally:
        batch_provider.close()
    a = report["A_fresh_threads"]
    b = report["B_persistent_thread"]
    report["comparison"] = {
        "persistent_improves_time_to_first_valid": (
            b["time_to_first_valid_ast_ms"] is not None
            and a["time_to_first_valid_ast_ms"] is not None
            and b["time_to_first_valid_ast_ms"] < a["time_to_first_valid_ast_ms"]
        ),
        "fresh_cost_per_valid_program": _cost_per_valid(a),
        "persistent_cost_per_valid_program": _cost_per_valid(b),
        "sample_size_warning": "One bounded four-brief sample; do not select a default.",
    }
    write_json(args.workspace / "abc-report.json.gz", report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
