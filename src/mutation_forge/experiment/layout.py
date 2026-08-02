"""Filesystem layout and atomic initialization for an experiment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ExperimentConfig


class WorkspaceError(RuntimeError):
    """The selected experiment workspace is invalid or cannot be opened."""


@dataclass(frozen=True, slots=True)
class ExperimentLayout:
    """All paths belonging to one experiment.

    Paths are derived exclusively from the resolved configuration directory,
    workspace, and exact ``exp_id``.  Callers never need to construct a run or
    retained-run path themselves.
    """

    workspace: Path
    exp_id: str

    @property
    def root(self) -> Path:
        return self.workspace / self.exp_id

    @property
    def root_dir(self) -> Path:
        return self.root

    @property
    def experiment_config(self) -> Path:
        return self.root / "experiment.toml"

    @property
    def config_path(self) -> Path:
        return self.experiment_config

    @property
    def lock(self) -> Path:
        return self.root / "experiment.lock.json"

    @property
    def lock_path(self) -> Path:
        return self.lock

    @property
    def state(self) -> Path:
        return self.root / "state.sqlite3"

    @property
    def state_path(self) -> Path:
        return self.state

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def checkpoints_dir(self) -> Path:
        return self.checkpoints

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts

    @property
    def experiment_manifest(self) -> Path:
        return self.artifacts / "experiment-manifest.json"

    @property
    def sessions(self) -> Path:
        return self.artifacts / "sessions"

    @property
    def sessions_dir(self) -> Path:
        return self.sessions

    @property
    def generations(self) -> Path:
        return self.artifacts / "generations"

    @property
    def archive(self) -> Path:
        return self.artifacts / "archive"

    @property
    def archive_dir(self) -> Path:
        return self.archive

    @property
    def evaluations(self) -> Path:
        return self.artifacts / "evaluations"

    @property
    def manifests(self) -> Path:
        return self.artifacts / "manifests"

    @property
    def reports(self) -> Path:
        return self.artifacts / "reports"

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> ExperimentLayout:
        return cls(config.workspace, config.exp_id)

    def ensure_subdirectories(self, root: Path | None = None) -> None:
        base = root or self.root
        for relative in (
            "checkpoints",
            "artifacts",
            "artifacts/sessions",
            "artifacts/generations",
            "artifacts/archive/programs",
            "artifacts/archive/sources",
            "artifacts/evaluations/development",
            "artifacts/evaluations/replay",
            "artifacts/manifests",
            "artifacts/reports",
        ):
            (base / relative).mkdir(parents=True, exist_ok=True)

    def generation_slot_phase(
        self,
        generation: int,
        slot: int | str,
        phase: str = "initial",
        *,
        root: Path | None = None,
    ) -> Path:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        slot_text = str(slot)
        if slot_text.startswith("slot-"):
            slot_name = slot_text
        elif slot_text.isdigit():
            slot_name = f"slot-{int(slot_text):02d}"
        else:
            raise ValueError("slot must be an integer or slot-NN")
        if phase == "initial":
            phase_name = "initial"
        elif (phase.startswith("repair-") and phase[7:].isdigit()) or (
            phase.startswith("retry-") and phase[6:].isdigit()
        ):
            phase_name = phase
        else:
            raise ValueError("phase must be initial, repair-NN, or retry-NN")
        base = root or self.root
        return (
            base
            / "artifacts"
            / "generations"
            / f"generation-{generation:04d}"
            / slot_name
            / phase_name
        )

    def session_dir(self, number: int, *, root: Path | None = None) -> Path:
        if number <= 0:
            raise ValueError("session number must be positive")
        return (root or self.root) / "artifacts" / "sessions" / f"session-{number:06d}"

    def checkpoint_path(self, sequence: int, *, root: Path | None = None) -> Path:
        if sequence <= 0:
            raise ValueError("checkpoint sequence must be positive")
        return (root or self.root) / "checkpoints" / f"checkpoint-{sequence:012d}.json"

    def _validate_root(self) -> None:
        workspace = self.workspace.resolve()
        root = self.root.resolve()
        try:
            root.relative_to(workspace)
        except ValueError as exc:
            raise WorkspaceError("experiment root escapes workspace") from exc
        if root.parent != workspace:
            raise WorkspaceError("experiment root must be exactly one directory below workspace")

    def initialize_atomic(
        self,
        config: ExperimentConfig,
        *,
        lock_payload: dict[str, Any] | None = None,
        state_initializer: Callable[[Path], None] | None = None,
    ) -> None:
        """Create the complete root with a single visible rename.

        ``state_initializer`` is called while the temporary tree is private;
        implementations should close all database handles before returning.
        """

        self._validate_root()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.root.exists():
            raise WorkspaceError(f"experiment already exists: {self.root}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{self.exp_id}.", dir=self.workspace))
        try:
            self.ensure_subdirectories(temporary)
            _atomic_write(temporary / "experiment.toml", config.source_bytes)
            if lock_payload is not None:
                _atomic_write(
                    temporary / "experiment.lock.json",
                    json.dumps(
                        lock_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n",
                )
            else:
                from .lock import build_lock

                lock_payload = build_lock(config, self)
                _atomic_write(
                    temporary / "experiment.lock.json",
                    json.dumps(
                        lock_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n",
                )
            if state_initializer is not None:
                state_initializer(temporary / "state.sqlite3")
            else:
                from .state import ExperimentStateStore

                ExperimentStateStore.initialize(
                    temporary / "state.sqlite3",
                    exp_id=config.exp_id,
                    lock_hash=str(lock_payload["immutable_config_sha256"]),
                    root=self.root,
                )
            manifest = _artifact_manifest(temporary / "artifacts")
            _atomic_write(
                temporary / "artifacts" / "experiment-manifest.json",
                json.dumps(
                    manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                + b"\n",
            )
            _fsync_tree(temporary)
            try:
                os.replace(temporary, self.root)
            except FileExistsError as exc:
                raise WorkspaceError(f"experiment already exists: {self.root}") from exc
            _fsync_directory(self.workspace)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def verify_root(self) -> None:
        self._validate_root()
        if not self.root.is_dir():
            raise WorkspaceError(f"experiment workspace does not exist: {self.root}")
        required = (self.experiment_config, self.lock, self.state, self.checkpoints, self.artifacts)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise WorkspaceError(f"experiment workspace is incomplete: {', '.join(missing)}")

    def write_artifact_manifest(self) -> dict[str, Any]:
        manifest = _artifact_manifest(self.artifacts)
        _atomic_write(
            self.experiment_manifest,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n",
        )
        return manifest

    def verify_artifact_manifest(self) -> bool:
        try:
            value = json.loads(self.experiment_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError("cannot read experiment artifact manifest") from exc
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "mforge.experiment.manifest.v1"
        ):
            raise WorkspaceError("invalid experiment artifact manifest schema")
        expected = value.get("manifest_sha256")
        base = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if (
            expected
            != hashlib.sha256(
                json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
        ):
            raise WorkspaceError("experiment artifact manifest hash mismatch")
        rows = value.get("files")
        if not isinstance(rows, list):
            raise WorkspaceError("experiment artifact manifest files must be an array")
        expected_paths = {str(row.get("path")) for row in rows if isinstance(row, dict)}
        actual_paths = {
            path.relative_to(self.artifacts).as_posix()
            for path in self.artifacts.rglob("*")
            if path.is_file() and path.name != "experiment-manifest.json"
        }
        if expected_paths != actual_paths:
            raise WorkspaceError("experiment artifact manifest file set mismatch")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise WorkspaceError("invalid experiment artifact manifest entry")
            path = (self.artifacts / str(row["path"])).resolve()
            try:
                path.relative_to(self.artifacts.resolve())
            except ValueError as exc:
                raise WorkspaceError("experiment artifact manifest path escapes artifacts") from exc
            data = path.read_bytes()
            if (
                int(row.get("size", -1)) != len(data)
                or row.get("sha256") != hashlib.sha256(data).hexdigest()
            ):
                raise WorkspaceError(f"experiment artifact digest mismatch: {row['path']}")
        return True


WorkspaceLayout = ExperimentLayout


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            try:
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            except OSError:
                pass
        elif path.is_dir():
            _fsync_directory(path)


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "experiment-manifest.json":
            continue
        data = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    base = {"schema_version": "mforge.experiment.manifest.v1", "files": files}
    return {
        **base,
        "manifest_sha256": hashlib.sha256(
            json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


__all__ = ["ExperimentLayout", "WorkspaceError", "WorkspaceLayout"]
