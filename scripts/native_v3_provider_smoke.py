#!/usr/bin/env python3
"""Run one bounded real Native v3 turn through the Native v2 provider."""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3.provider_smoke import run_provider_smoke


def _default_workspace() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"mforge-native-v3-provider-smoke-{stamp}-{secrets.token_hex(4)}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate one Native v3 AST through the unchanged Native v2 provider"
    )
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    workspace = (args.workspace or _default_workspace()).resolve()
    auth_json = Path.home() / ".codex" / "auth.json"
    provider = LocalCodexAppServerProvider(
        model=args.model,
        effort=args.effort,
        concurrency=1,
        max_repairs=0,
        turn_timeout_base_seconds=args.timeout_seconds / 2,
        auth_json=auth_json,
        persist_artifacts=False,
    )
    try:
        report = run_provider_smoke(provider, workspace)
    finally:
        provider.close()
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
