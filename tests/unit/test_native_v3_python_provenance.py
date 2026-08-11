from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from mutation_forge.experiment.json_io import read_json
from mutation_forge.native_v3_python.provenance import (
    ABORTED_UNVERIFIABLE_PROVENANCE,
    M5_PROVENANCE_FILENAME,
    GitRepositoryIdentityV1,
    M5ProvenanceError,
    ensure_m5_acceptance_provenance,
)
from mutation_forge.native_v3_python.runtime_contracts import PolicyRuntimeLimitsV1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "native_v3_python_m5_live_search.py"

_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "source"],
    "properties": {
        "schema_version": {"const": "mforge.native.python_policy_response.v1"},
        "source": {"type": "string"},
    },
    "additionalProperties": False,
}
_ACK_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "ack"],
    "properties": {
        "schema_version": {"type": "string"},
        "ack": {"type": "string"},
    },
}


def _identity_loader(
    repository_identity: GitRepositoryIdentityV1 | None = None,
    heg_identity: GitRepositoryIdentityV1 | None = None,
) -> Callable[[Path], GitRepositoryIdentityV1]:
    repository = repository_identity or GitRepositoryIdentityV1(
        "1" * 40, "2" * 40, False
    )
    heg = heg_identity or GitRepositoryIdentityV1("3" * 40, "4" * 40, False)

    def load(path: Path) -> GitRepositoryIdentityV1:
        return heg if path.name == "heg" else repository

    return load


def _inputs(tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "mutation-forge"
    heg = tmp_path / "heg"
    repository.mkdir()
    heg.mkdir()
    config = repository / "experiment.toml"
    config.write_text("schema_version = 1\n", encoding="utf-8")
    return {
        "workspace": tmp_path / "workspace",
        "resume": False,
        "repository_root": repository,
        "heg_root": heg,
        "experiment_config": config,
        "model": "gpt-5.6-luna",
        "effort": "medium",
        "system_prompt": "system prompt",
        "request_template": "request template",
        "specification_prompt": "specification prompt",
        "output_schema": _OUTPUT_SCHEMA,
        "specification_ack_schema": _ACK_SCHEMA,
        "git_identity_loader": _identity_loader(),
    }


def _commit_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@mutation-forge.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Mutation Forge Tests"],
        cwd=path,
        check=True,
    )
    marker = path / ".gitkeep"
    marker.touch()
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)


def test_clean_run_persists_exact_provenance_before_provider_start(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    snapshot = ensure_m5_acceptance_provenance(**inputs)
    retained = read_json(inputs["workspace"] / M5_PROVENANCE_FILENAME)
    assert retained == snapshot
    assert snapshot["mutation_forge"] == {
        "commit_sha": "1" * 40,
        "tree_sha": "2" * 40,
        "dirty": False,
    }
    assert snapshot["heg"] == {
        "commit_sha": "3" * 40,
        "tree_sha": "4" * 40,
        "dirty": False,
    }
    assert snapshot["model"] == "gpt-5.6-luna"
    assert snapshot["reasoning_effort"] == "medium"
    assert snapshot["m2"]["frozen_limits"] == PolicyRuntimeLimitsV1().as_dict()


def test_new_uncommitted_experiment_config_is_accepted(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    repository = cast(Path, inputs["repository_root"])
    heg = cast(Path, inputs["heg_root"])
    _commit_repository(repository)
    _commit_repository(heg)
    config = cast(Path, inputs["experiment_config"])
    config.write_text("schema_version = 2\n", encoding="utf-8")
    inputs.pop("git_identity_loader")

    snapshot = ensure_m5_acceptance_provenance(**inputs)

    assert snapshot["mutation_forge"]["dirty"] is False
    assert snapshot["heg"]["dirty"] is False


def test_dirty_worktree_fails_before_workspace_or_provider_start(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["git_identity_loader"] = _identity_loader(
        GitRepositoryIdentityV1("1" * 40, "2" * 40, True)
    )
    with pytest.raises(M5ProvenanceError, match="clean mutation-forge worktree"):
        ensure_m5_acceptance_provenance(**inputs)
    assert not inputs["workspace"].exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "repository_commit",
        "repository_tree",
        "config",
        "model",
        "effort",
        "system_prompt",
        "request_template",
        "specification_prompt",
        "output_schema",
        "runtime_limits",
        "heg_commit",
        "heg_tree",
        "heg_dirty",
    ),
)
def test_resume_rejects_every_semantic_provenance_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    inputs = _inputs(tmp_path)
    ensure_m5_acceptance_provenance(**inputs)
    inputs["resume"] = True
    if mutation == "repository_commit":
        inputs["git_identity_loader"] = _identity_loader(
            GitRepositoryIdentityV1("5" * 40, "2" * 40, False)
        )
    elif mutation == "repository_tree":
        inputs["git_identity_loader"] = _identity_loader(
            GitRepositoryIdentityV1("1" * 40, "5" * 40, False)
        )
    elif mutation == "config":
        Path(inputs["experiment_config"]).write_text(
            "schema_version = 2\n", encoding="utf-8"
        )
    elif mutation in {
        "model",
        "effort",
        "system_prompt",
        "request_template",
        "specification_prompt",
    }:
        inputs[mutation] = str(inputs[mutation]) + "-changed"
    elif mutation == "output_schema":
        inputs["output_schema"] = {**_OUTPUT_SCHEMA, "title": "changed"}
    elif mutation == "runtime_limits":
        inputs["runtime_limits"] = PolicyRuntimeLimitsV1(
            propose_wall_seconds=0.5
        )
    elif mutation == "heg_commit":
        inputs["git_identity_loader"] = _identity_loader(
            heg_identity=GitRepositoryIdentityV1("5" * 40, "4" * 40, False)
        )
    elif mutation == "heg_tree":
        inputs["git_identity_loader"] = _identity_loader(
            heg_identity=GitRepositoryIdentityV1("3" * 40, "5" * 40, False)
        )
    else:
        inputs["git_identity_loader"] = _identity_loader(
            heg_identity=GitRepositoryIdentityV1("3" * 40, "4" * 40, True)
        )
    with pytest.raises(M5ProvenanceError, match="differs"):
        ensure_m5_acceptance_provenance(**inputs)


def test_matching_provenance_resumes_without_rewriting_snapshot(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    initial = ensure_m5_acceptance_provenance(**inputs)
    path = inputs["workspace"] / M5_PROVENANCE_FILENAME
    before = path.read_bytes()
    inputs["resume"] = True
    assert ensure_m5_acceptance_provenance(**inputs) == initial
    assert path.read_bytes() == before


def test_resume_freezes_supplied_scientific_identity_not_invocation_config(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["experiment_config_sha256"] = "a" * 64
    initial = ensure_m5_acceptance_provenance(**inputs)

    Path(inputs["experiment_config"]).write_text(
        "schema_version = 1\nwall_seconds = 60\n",
        encoding="utf-8",
    )
    inputs["resume"] = True
    assert ensure_m5_acceptance_provenance(**inputs) == initial

    inputs["experiment_config_sha256"] = "b" * 64
    with pytest.raises(M5ProvenanceError, match="differs"):
        ensure_m5_acceptance_provenance(**inputs)


def test_resume_accepts_only_the_declared_legacy_scientific_identity_hash(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["experiment_config_sha256"] = "a" * 64
    ensure_m5_acceptance_provenance(**inputs)
    path = inputs["workspace"] / M5_PROVENANCE_FILENAME
    before = path.read_bytes()

    inputs["resume"] = True
    inputs["experiment_config_sha256"] = "b" * 64
    inputs["legacy_experiment_config_sha256"] = "a" * 64
    resumed = ensure_m5_acceptance_provenance(**inputs)

    assert resumed["experiment_config_sha256"] == "b" * 64
    assert path.read_bytes() == before

    inputs["legacy_experiment_config_sha256"] = "c" * 64
    with pytest.raises(M5ProvenanceError, match="differs"):
        ensure_m5_acceptance_provenance(**inputs)


def test_old_workspace_without_snapshot_is_explicitly_rejected(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["workspace"].mkdir()
    inputs["resume"] = True
    with pytest.raises(
        M5ProvenanceError,
        match=ABORTED_UNVERIFIABLE_PROVENANCE,
    ):
        ensure_m5_acceptance_provenance(**inputs)


def _load_live_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("m5_live_search", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_entrypoint_provenance_failure_precedes_backend_and_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_live_script()
    calls = {"provenance": 0, "backend": 0, "provider": 0}

    def reject(**kwargs: Any) -> None:
        del kwargs
        calls["provenance"] += 1
        raise M5ProvenanceError("fixture preflight rejection")

    class ForbiddenBackend:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            calls["backend"] += 1

    class ForbiddenProvider:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            calls["provider"] += 1

    monkeypatch.setattr(module, "ensure_m5_acceptance_provenance", reject)
    monkeypatch.setattr(module, "HegBackend", ForbiddenBackend)
    monkeypatch.setattr(module, "CodexM5SearchProvider", ForbiddenProvider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--output-root",
            str(tmp_path),
        ],
    )
    with pytest.raises(M5ProvenanceError, match="preflight rejection"):
        module.main()
    assert calls == {"provenance": 1, "backend": 0, "provider": 0}
