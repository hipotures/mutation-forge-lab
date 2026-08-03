from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mutation_forge.config import LabConfig
from mutation_forge.models import JsonValue


def canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_state(repo: Path) -> dict[str, JsonValue]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--short")),
    }


def environment_record(lock_path: Path) -> dict[str, JsonValue]:
    lock_hash = (
        hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
    )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "uv_lock_sha256": lock_hash,
        "pid": os.getpid(),
    }


class RunArtifacts:
    def __init__(self, config: LabConfig, run_id: str) -> None:
        self.run_id = run_id
        self.path = config.run.run_root / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        for name in ("programs", "prompts", "responses", "graphs"):
            (self.path / "artifacts" / name).mkdir(parents=True)
        shutil.copy2(config.source_path, self.path / "run_config.toml")

    def write_json(self, name: str, value: object) -> Path:
        path = self.path / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return path

    def write_manifest(
        self,
        config: LabConfig,
        *,
        project_state: dict[str, JsonValue],
        heg_state: dict[str, JsonValue],
        environment: dict[str, JsonValue],
        status: str,
    ) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "schema_version": "mforge.experiment.run.v2",
            "run_id": self.run_id,
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
            "mutation_forge": project_state,
            "heg": heg_state,
            "environment": environment,
            "config_hash": config.stable_hash(),
            "resolved_config": config.resolved_dict(),
            "dataset_schema_version": "1.0",
            "score_schema_version": "1.0",
            "proposal_schema_version": "1.0",
            "policy_schema_version": "1.0",
            "event_schema_version": "mforge.experiment.events.v2",
            "resource_limits": {
                "wall_seconds": config.run.wall_seconds,
                "evaluations_per_episode": config.search.evaluations_per_episode,
                "witness_cap": config.score.witness_cap,
            },
        }
        self.write_json("run_manifest.json", manifest)
        return manifest
