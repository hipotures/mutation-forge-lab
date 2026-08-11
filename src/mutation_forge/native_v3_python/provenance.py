"""Fail-closed provenance for official M5 acceptance runs and resumes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3.scoring import (
    FITNESS_PROTOCOL_ID,
    SCORE_EVIDENCE_SCHEMA_VERSION,
    SCORE_PROTOCOL_ID,
)

from .contracts import (
    BEHAVIOR_IDENTITY_PROTOCOL_ID,
    PYTHON_POLICY_PROTOCOL_ID,
    PYTHON_RESPONSE_SCHEMA_VERSION,
)
from .runtime_contracts import (
    RANDOM_PROTOCOL_ID,
    RUNTIME_PROTOCOL_ID,
    SAFE_GRAPH_API_PROTOCOL_ID,
    SEMANTIC_TRACE_PROTOCOL_ID,
    PolicyRuntimeLimitsV1,
)
from .search import (
    M5_CANDIDATE_PROTOCOL_ID,
    M5_MANIFEST_PROTOCOL_ID,
    M5_SEARCH_PROTOCOL_ID,
)
from .serial_evaluator import (
    PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
    PYTHON_SERIAL_RESULT_PROTOCOL_ID,
)
from .validation import (
    IDENTITY_PROTOCOL_VERSION,
    PYTHON_SYNTAX_VERSION,
    VALIDATOR_VERSION,
)

M5_PROVENANCE_PROTOCOL_ID = "mforge.native.python_m5_acceptance_provenance.v1"
M2_SANDBOX_PROTOCOL_ID = (
    "mforge.native.python_policy_sandbox.linux_bwrap_seccomp_rlimit.v1"
)
M5_PROVENANCE_FILENAME = "acceptance-provenance.json.gz"
ABORTED_UNVERIFIABLE_PROVENANCE = "ABORTED_UNVERIFIABLE_PROVENANCE"

_PROVENANCE_DOMAIN = b"mforge-native-v3-python-m5-acceptance-provenance-v1\0"


class M5ProvenanceError(RuntimeError):
    """An official M5 run cannot prove an exact clean execution environment."""


@dataclass(frozen=True, slots=True)
class GitRepositoryIdentityV1:
    """Auditable Git identity without embedding a local filesystem path."""

    commit_sha: str
    tree_sha: str
    dirty: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("commit_sha", self.commit_sha),
            ("tree_sha", self.tree_sha),
        ):
            if (
                len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-1 object ID")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "dirty": self.dirty,
        }


GitIdentityLoader = Callable[[Path], GitRepositoryIdentityV1]


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git failed"
        raise M5ProvenanceError(
            f"cannot identify Git repository: {message[:512]}"
        )
    return result.stdout.strip()


def read_git_repository_identity(
    repository: Path,
    *,
    ignored_path: Path | None = None,
) -> GitRepositoryIdentityV1:
    """Read the exact commit, tree, and dirty state of one Git repository."""

    root = repository.resolve(strict=True)
    status_arguments = [
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ]
    if ignored_path is not None:
        try:
            relative = ignored_path.resolve(strict=True).relative_to(root)
        except ValueError:
            pass
        else:
            status_arguments.extend(
                ("--", ".", f":(exclude){relative.as_posix()}")
            )
    return GitRepositoryIdentityV1(
        commit_sha=_run_git(root, "rev-parse", "HEAD"),
        tree_sha=_run_git(root, "rev-parse", "HEAD^{tree}"),
        dirty=bool(_run_git(root, *status_arguments)),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _snapshot_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_PROVENANCE_DOMAIN + _canonical_bytes(payload))


def build_m5_acceptance_provenance(
    *,
    repository_root: Path,
    heg_root: Path,
    experiment_config: Path,
    model: str,
    effort: str,
    system_prompt: str,
    request_template: str,
    specification_prompt: str,
    output_schema: Mapping[str, Any],
    specification_ack_schema: Mapping[str, Any],
    runtime_limits: PolicyRuntimeLimitsV1 | None = None,
    experiment_config_sha256: str | None = None,
    git_identity_loader: GitIdentityLoader = read_git_repository_identity,
) -> dict[str, JsonValue]:
    """Build the deterministic provenance snapshot for one official run."""

    if not model or not effort or not system_prompt or not request_template:
        raise ValueError("M5 provenance inputs must be non-empty")
    if experiment_config_sha256 is not None and (
        len(experiment_config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in experiment_config_sha256)
    ):
        raise ValueError("experiment_config_sha256 must be lowercase SHA-256")
    repository = (
        read_git_repository_identity(
            repository_root,
            ignored_path=experiment_config,
        )
        if git_identity_loader is read_git_repository_identity
        else git_identity_loader(repository_root)
    )
    heg = git_identity_loader(heg_root)
    limits = runtime_limits or PolicyRuntimeLimitsV1()
    payload: dict[str, JsonValue] = {
        "protocol_id": M5_PROVENANCE_PROTOCOL_ID,
        "mutation_forge": repository.as_dict(),
        "heg": heg.as_dict(),
        "experiment_config_sha256": (
            experiment_config_sha256
            if experiment_config_sha256 is not None
            else _sha256_bytes(experiment_config.resolve(strict=True).read_bytes())
        ),
        "model": model,
        "reasoning_effort": effort,
        "generation_protocol": {
            "search_protocol_id": M5_SEARCH_PROTOCOL_ID,
            "manifest_protocol_id": M5_MANIFEST_PROTOCOL_ID,
            "candidate_protocol_id": M5_CANDIDATE_PROTOCOL_ID,
        },
        "prompt_hashes": {
            "system_prompt_sha256": _sha256_text(system_prompt),
            "request_template_sha256": _sha256_text(request_template),
            "specification_prompt_sha256": _sha256_text(specification_prompt),
        },
        "schema_hashes": {
            "output_schema_sha256": _sha256_bytes(
                _canonical_bytes(output_schema)
            ),
            "specification_ack_schema_sha256": _sha256_bytes(
                _canonical_bytes(specification_ack_schema)
            ),
            "response_schema_version": PYTHON_RESPONSE_SCHEMA_VERSION,
        },
        "m1": {
            "policy_protocol_id": PYTHON_POLICY_PROTOCOL_ID,
            "validator_version": VALIDATOR_VERSION,
            "identity_protocol_version": IDENTITY_PROTOCOL_VERSION,
            "behavior_identity_protocol_id": BEHAVIOR_IDENTITY_PROTOCOL_ID,
            "python_syntax_version": PYTHON_SYNTAX_VERSION,
        },
        "m2": {
            "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
            "sandbox_protocol_id": M2_SANDBOX_PROTOCOL_ID,
            "safe_graph_api_protocol_id": SAFE_GRAPH_API_PROTOCOL_ID,
            "semantic_trace_protocol_id": SEMANTIC_TRACE_PROTOCOL_ID,
            "random_protocol_id": RANDOM_PROTOCOL_ID,
            "frozen_limits": limits.as_dict(),
        },
        "m3": {
            "evaluator_protocol_id": PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
            "result_protocol_id": PYTHON_SERIAL_RESULT_PROTOCOL_ID,
            "score_evidence_schema_version": SCORE_EVIDENCE_SCHEMA_VERSION,
            "score_protocol_id": SCORE_PROTOCOL_ID,
            "fitness_protocol_id": FITNESS_PROTOCOL_ID,
        },
    }
    return {**payload, "sha256": _snapshot_hash(payload)}


def _resume_comparison_payload(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(provenance)
    payload.pop("sha256", None)
    raw_repository = payload.get("mutation_forge")
    if (
        not isinstance(raw_repository, Mapping)
        or set(raw_repository) != {"commit_sha", "tree_sha", "dirty"}
        or not isinstance(raw_repository.get("dirty"), bool)
    ):
        return payload
    try:
        repository = GitRepositoryIdentityV1(
            commit_sha=str(raw_repository["commit_sha"]),
            tree_sha=str(raw_repository["tree_sha"]),
            dirty=raw_repository["dirty"] is True,
        )
    except ValueError:
        return payload
    payload["mutation_forge"] = {"dirty": repository.dirty}
    return payload


def ensure_m5_acceptance_provenance(
    *,
    workspace: Path,
    resume: bool,
    repository_root: Path,
    heg_root: Path,
    experiment_config: Path,
    model: str,
    effort: str,
    system_prompt: str,
    request_template: str,
    specification_prompt: str,
    output_schema: Mapping[str, Any],
    specification_ack_schema: Mapping[str, Any],
    runtime_limits: PolicyRuntimeLimitsV1 | None = None,
    experiment_config_sha256: str | None = None,
    legacy_experiment_config_sha256: str | None = None,
    git_identity_loader: GitIdentityLoader = read_git_repository_identity,
) -> dict[str, JsonValue]:
    """Persist or compare provenance before App Server construction."""

    current = build_m5_acceptance_provenance(
        repository_root=repository_root,
        heg_root=heg_root,
        experiment_config=experiment_config,
        model=model,
        effort=effort,
        system_prompt=system_prompt,
        request_template=request_template,
        specification_prompt=specification_prompt,
        output_schema=output_schema,
        specification_ack_schema=specification_ack_schema,
        runtime_limits=runtime_limits,
        experiment_config_sha256=experiment_config_sha256,
        git_identity_loader=git_identity_loader,
    )
    repository = current["mutation_forge"]
    if not isinstance(repository, Mapping) or repository.get("dirty") is not False:
        raise M5ProvenanceError(
            "official M5 acceptance runs require a clean mutation-forge worktree"
        )

    path = workspace / M5_PROVENANCE_FILENAME
    if resume:
        if not path.is_file():
            raise M5ProvenanceError(
                f"{ABORTED_UNVERIFIABLE_PROVENANCE}: "
                "workspace has no immutable M5 acceptance provenance"
            )
        retained = read_json(path)
        if not isinstance(retained, Mapping):
            raise M5ProvenanceError("retained M5 provenance is not an object")
        retained_dict = dict(retained)
        retained_hash = retained_dict.pop("sha256", None)
        if retained_hash != _snapshot_hash(retained_dict):
            raise M5ProvenanceError("retained M5 provenance hash is invalid")
        retained_payload = _resume_comparison_payload(retained)
        current_payload = _resume_comparison_payload(current)
        if retained_payload != current_payload:
            retained_config_hash = retained_payload.get(
                "experiment_config_sha256"
            )
            if (
                legacy_experiment_config_sha256 is None
                or retained_config_hash != legacy_experiment_config_sha256
            ):
                raise M5ProvenanceError(
                    "current environment differs from immutable M5 provenance"
                )
            retained_payload["experiment_config_sha256"] = (
                current_payload["experiment_config_sha256"]
            )
            if retained_payload != current_payload:
                raise M5ProvenanceError(
                    "current environment differs from immutable M5 provenance"
                )
        return current

    write_json(path, current, exclusive=True)
    return current


__all__ = [
    "ABORTED_UNVERIFIABLE_PROVENANCE",
    "GitRepositoryIdentityV1",
    "M2_SANDBOX_PROTOCOL_ID",
    "M5_PROVENANCE_FILENAME",
    "M5_PROVENANCE_PROTOCOL_ID",
    "M5ProvenanceError",
    "build_m5_acceptance_provenance",
    "ensure_m5_acceptance_provenance",
    "read_git_repository_identity",
]
