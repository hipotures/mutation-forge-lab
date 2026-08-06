#!/usr/bin/env python3
"""Run the Native v3 Step 12D lineage and Search Memory experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mutation_forge.native_v3.lineage_experiment import run_lineage_experiment
from mutation_forge.stage3.app_server import AppServerLimits, CodexAppServerAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures/native_v3_single_program_responses.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--auth-json", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--turn-timeout", type=float, default=900.0)
    return parser


def _candidate_responses() -> dict[str, dict[str, Any]]:
    value = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, dict)
        for key, item in value.items()
    ):
        raise ValueError("invalid lineage fixture responses")
    return value


def main() -> int:
    args = _parser().parse_args()
    if args.workspace.exists():
        raise SystemExit("workspace must not exist")
    if not args.auth_json.is_file():
        raise SystemExit("auth JSON does not exist")

    def factory(base_instructions: str) -> CodexAppServerAdapter:
        return CodexAppServerAdapter(
            auth_json=args.auth_json,
            limits=AppServerLimits(
                max_turns=8,
                max_campaigns=3,
                turn_timeout=args.turn_timeout,
            ),
            base_instructions=base_instructions,
            compress_json_artifacts=True,
            sandbox_mode="danger-full-access",
            approval_policy="never",
            copy_rollout_artifact=False,
        )

    report = run_lineage_experiment(
        args.workspace,
        model=args.model,
        effort=args.effort,
        forbidden_lengths=(4, 8, 16),
        candidate_responses=_candidate_responses(),
        adapter_factory=factory,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
