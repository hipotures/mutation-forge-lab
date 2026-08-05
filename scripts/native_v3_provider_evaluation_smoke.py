#!/usr/bin/env python3
"""Run one bounded Native v3 provider turn through one HEG evaluation."""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3.provider_evaluation import (
    run_provider_evaluation_smoke,
)
from mutation_forge.native_v3.serial_evaluator import SerialEpisodeConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_workspace() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    return Path("/tmp") / f"mforge-native-v3-provider-evaluation-{stamp}-{suffix}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate one Native v3 AST without public routing"
    )
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--heg-repo", type=Path, default=PROJECT_ROOT.parent / "heg")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--order", type=int, default=30)
    parser.add_argument("--graph-seed", type=int, default=101)
    parser.add_argument("--policy-seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--witness-cap", type=int, default=64)
    args = parser.parse_args(argv)

    workspace = (args.workspace or _default_workspace()).resolve()
    heg_repo = args.heg_repo.resolve()
    provider = LocalCodexAppServerProvider(
        model=args.model,
        effort=args.effort,
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=args.timeout_seconds / 2,
        auth_json=Path.home() / ".codex" / "auth.json",
        persist_artifacts=False,
    )
    try:
        report = run_provider_evaluation_smoke(
            provider,
            workspace,
            backend_factory=lambda: HegBackend(heg_repo),
            config=SerialEpisodeConfig(
                order=args.order,
                graph_seed=args.graph_seed,
                policy_seed=args.policy_seed,
                horizon=args.horizon,
                witness_cap=args.witness_cap,
                episode_id="native-v3-step09-slot-00",
            ),
        )
    finally:
        provider.close()
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
