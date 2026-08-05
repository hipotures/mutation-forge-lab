"""Run one bounded real-provider Native v2 baseline smoke."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import secrets
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path(tempfile.gettempdir()) / "mutation-forge-native-v2-smoke"


def fresh_experiment_id(now: datetime | None = None) -> str:
    """Return a collision-resistant, human-readable disposable experiment id."""

    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"native-v2-smoke-{timestamp}-{secrets.token_hex(4)}"


def render_config(*, exp_id: str, workspace: Path) -> str:
    """Render the smallest real-provider Native v2 experiment configuration."""

    return f'''schema_version = "mforge.experiment.v2"
exp_id = "{exp_id}"
workspace = "{workspace.as_posix()}"
kind = "heg"
preset = "native"

[run]
wall_seconds = 900
output = "json"
turn_timeout_base_seconds = 300
max_total_tokens_per_hour = 100000

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "high"
concurrency = 1
max_repairs = 1

[search]
population_size = 1
max_generations = 1
max_model_turns = 2
selection = "elite-diversity"

[evaluation]
graph_mode = "unrestricted_min_degree_3"
order_schedule = "static"
orders = [4]
graph_seeds = [401]
policy_seeds = [4001]
horizon = 1
proposal_pool_size = 2
baselines = ["random", "structural"]
replay = false

[resources]
workers = 1
thread_count = 1
'''


def provider_artifact_snapshot(experiment_root: Path) -> dict[str, str]:
    """Return relative paths and SHA-256 hashes for every provider-turn artifact."""

    generations = experiment_root / "artifacts" / "generations"
    if not generations.is_dir():
        return {}
    return {
        path.relative_to(generations).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(generations.rglob("*"))
        if path.is_file()
    }


def verify_appserver_artifact_structure(experiment_root: Path) -> dict[str, Any]:
    """Verify real provider turns against the frozen Native v2 structure."""

    path = PROJECT_ROOT / "scripts" / "appserver_artifact_parity.py"
    spec = importlib.util.spec_from_file_location("appserver_artifact_parity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load App Server parity gate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.verify_real_provider_workspace(experiment_root))


def _run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _run_json(command: Sequence[str]) -> dict[str, Any]:
    raw = _run_text(command)
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise RuntimeError(f"command did not return a JSON object: {shlex.join(command)}")
    return dict(parsed)


def main() -> int:
    exp_id = fresh_experiment_id()
    workspace = DEFAULT_WORKSPACE
    config_dir = workspace / "configs"
    report_dir = workspace / "reports"
    config_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{exp_id}.toml"
    experiment_root = workspace / exp_id
    if config_path.exists() or experiment_root.exists():
        raise RuntimeError(f"refusing to reuse disposable experiment id {exp_id}")
    config_path.write_text(
        render_config(exp_id=exp_id, workspace=workspace),
        encoding="utf-8",
    )

    doctor_command = [
        "uv",
        "run",
        "mforge",
        "doctor",
        "--heg-repo",
        "../heg",
        "--run-root",
        str(workspace / "doctor"),
    ]
    run_command = [
        "uv",
        "run",
        "mforge",
        "experiment",
        "run",
        "--config",
        str(config_path),
        "--json",
    ]
    status_command = [
        "uv",
        "run",
        "mforge",
        "experiment",
        "status",
        "--config",
        str(config_path),
        "--json",
    ]

    subprocess.run(doctor_command, cwd=PROJECT_ROOT, check=True)
    run_result = _run_json(run_command)
    artifacts_before_status = provider_artifact_snapshot(experiment_root)
    status = _run_json(status_command)
    artifacts_after_status = provider_artifact_snapshot(experiment_root)
    artifact_contract = verify_appserver_artifact_structure(experiment_root)

    checks = {
        "model_turns_positive": int(status.get("model_turns_used", 0)) > 0,
        "accepted_candidates_positive": int(status.get("unique_candidate_count", 0)) > 0,
        "evaluations_positive": int(status.get("evaluation_count", 0)) > 0,
        "provider_artifacts_present": bool(artifacts_before_status),
        "provider_artifacts_unchanged_by_status": (
            artifacts_after_status == artifacts_before_status
        ),
        "appserver_artifact_structure": int(artifact_contract.get("turn_count", 0)) > 0,
        "terminal_without_error": (
            status.get("state") in {"completed", "exhausted"}
            and status.get("last_error") is None
        ),
    }
    report = {
        "schema_version": "mforge.native-v2-smoke-report.v1",
        "exp_id": exp_id,
        "workspace": str(experiment_root),
        "config": str(config_path),
        "commands": {
            "doctor": shlex.join(doctor_command),
            "run": shlex.join(run_command),
            "status": shlex.join(status_command),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "uv": _run_text(["uv", "--version"]),
            "codex": _run_text(["codex", "--version"]),
            "project_commit": _run_text(["git", "rev-parse", "HEAD"]),
        },
        "run_result": run_result,
        "status": status,
        "checks": checks,
        "provider_turn_artifact_tree": list(artifacts_before_status),
        "provider_turn_artifact_sha256": artifacts_before_status,
        "appserver_artifact_contract": artifact_contract,
    }
    report_path = report_dir / f"{exp_id}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report": str(report_path)}, sort_keys=True))
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        print(f"Native v2 smoke failed checks: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
