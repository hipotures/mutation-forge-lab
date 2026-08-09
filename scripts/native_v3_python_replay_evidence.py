"""Replay a durable ordinary-Python preview evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mutation_forge.native_v3_python.evidence import (
    EvidenceReplayError,
    replay_evidence_bundle,
    write_evidence_report,
)
from mutation_forge.native_v3_python.preview import load_python_preview_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace = load_python_preview_config(args.config).experiment_root.resolve()
    if args.output.resolve().is_relative_to(workspace):
        raise EvidenceReplayError(
            "replay report must be outside the immutable evidence workspace"
        )
    report = replay_evidence_bundle(
        config_path=args.config,
        artifact_root=args.artifact_root,
        expected_report=args.expected,
    )
    write_evidence_report(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
