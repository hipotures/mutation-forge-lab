"""Run the authorized bounded M5 two-generation ordinary-Python search."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mutation_forge.backends.heg import HegBackend
from mutation_forge.native_v3_python import (
    CodexM5SearchProvider,
    DevelopmentCaseV1,
    PythonPanelScientificEvaluator,
    ensure_m5_acceptance_provenance,
    run_m5_search,
    specification_ack_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "mutation-forge-native-v3-python-m5",
    )
    parser.add_argument(
        "--resume-workspace",
        type=Path,
        help="Resume one retained M5 workspace without reallocating its manifests.",
    )
    parser.add_argument("--heg-repo", type=Path, default=PROJECT_ROOT.parent / "heg")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="medium")
    return parser


def _text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    if args.resume_workspace is not None:
        root = args.resume_workspace.resolve(strict=True)
        if not (root / "protocol.json.gz").is_file():
            raise RuntimeError(f"not an M5 workspace: {root}")
    else:
        run_id = "native-v3-python-m5-" + datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        root = args.output_root / run_id
        if root.exists():
            raise RuntimeError(f"refusing to reuse M5 search root: {root}")
    system_prompt = _text("prompts/native-v3-python/m5-system.md").strip()
    request_template = _text("prompts/native-v3-python/m4-request.md")
    specification_prompt = (
        "Retain the complete policy specification below for later root and "
        "mutation turns. Do not generate a policy on this turn. Return only "
        "the required specification acknowledgement.\n\n"
        + request_template
    )
    policy_schema = json.loads(
        _text("configs/native/native-v3-python-policy-response.schema.json")
    )
    ack_schema = specification_ack_schema()
    ensure_m5_acceptance_provenance(
        workspace=root,
        resume=args.resume_workspace is not None,
        repository_root=PROJECT_ROOT,
        heg_root=args.heg_repo,
        experiment_config=PROJECT_ROOT / "experiment.toml",
        model=args.model,
        effort=args.effort,
        system_prompt=system_prompt,
        request_template=request_template,
        specification_prompt=specification_prompt,
        output_schema=policy_schema,
        specification_ack_schema=ack_schema,
    )
    backend = HegBackend(args.heg_repo)
    forbidden_lengths = backend.target_forbidden_lengths(30)
    panel = (
        DevelopmentCaseV1(
            case_id="order-30-seed-101",
            order=30,
            graph_seed=101,
            policy_seed=17,
            horizon=1,
            witness_cap=64,
            forbidden_lengths=forbidden_lengths,
        ),
        DevelopmentCaseV1(
            case_id="order-30-seed-103",
            order=30,
            graph_seed=103,
            policy_seed=19,
            horizon=1,
            witness_cap=64,
            forbidden_lengths=forbidden_lengths,
        ),
    )
    provider = CodexM5SearchProvider(
        workspace=root / "provider-runtime",
        model=args.model,
        effort=args.effort,
        base_instructions=system_prompt,
        auth_json=Path.home() / ".codex" / "auth.json",
        turn_timeout_seconds=300,
    )
    evaluator = PythonPanelScientificEvaluator(
        backend=backend,
        artifact_root=root / "scientific-artifacts",
    )
    try:
        report = run_m5_search(
            provider=provider,
            evaluator=evaluator,
            workspace=root,
            panel=panel,
            system_prompt=system_prompt,
            specification_prompt=specification_prompt,
            specification_ack_schema=ack_schema,
            policy_schema=policy_schema,
        )
    finally:
        backend.close()
    print(json.dumps({**report, "workspace": str(root)}, sort_keys=True))
    checks = report.get("acceptance_checks")
    return (
        0
        if report["status"] == "completed"
        and (
            report.get("exact_verified") is True
            or (
                isinstance(checks, dict)
                and all(value is True for value in checks.values())
            )
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
