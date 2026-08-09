"""Run the single authorized M4 Codex-to-Python-to-HEG root smoke."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3_python import (
    PythonSerialEpisodeConfigV1,
)
from mutation_forge.native_v3_python.provider_evaluation import run_m4_single_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "mutation-forge-native-v3-python-m4",
    )
    parser.add_argument("--heg-repo", type=Path, default=PROJECT_ROOT.parent / "heg")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="high")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_id = (
        "native-v3-python-m4-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    root = args.output_root / run_id
    if root.exists():
        raise RuntimeError(f"refusing to reuse M4 smoke root: {root}")
    backend = HegBackend(args.heg_repo)
    provider = LocalCodexAppServerProvider(
        model=args.model,
        effort=args.effort,
        concurrency=1,
        max_repairs=1,
        turn_timeout_base_seconds=300,
        auth_json=Path.home() / ".codex" / "auth.json",
        persist_artifacts=False,
        sandbox_mode="read-only",
        approval_policy="never",
    )
    try:
        report = run_m4_single_root(
            provider,
            root,
            backend_factory=lambda: backend,
            config=PythonSerialEpisodeConfigV1(
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                episode_id="native-v3-python-m4-single-root",
                forbidden_lengths=backend.target_forbidden_lengths(30),
            ),
        )
    finally:
        provider.close()
    print(json.dumps({**report, "workspace": str(root)}, sort_keys=True))
    return 0 if report["status"] in {"completed", "program_failure"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
