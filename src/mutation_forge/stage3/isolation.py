"""Process and filesystem isolation for the thin Codex app-server adapter.

This module deliberately contains no Codex protocol code.  It creates a private
capsule and the exact, strict argv used to launch app-server so the transport can
be tested with a fake process without touching a user's Codex installation.
"""

from __future__ import annotations

import os
import resource
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

THIN_APP_SERVER_ARGS: tuple[str, ...] = (
    "codex",
    "app-server",
    "--stdio",
    "--strict-config",
    "--disable",
    "apps",
    "--disable",
    "browser_use",
    "--disable",
    "browser_use_external",
    "--disable",
    "browser_use_full_cdp_access",
    "--disable",
    "computer_use",
    "--disable",
    "in_app_browser",
    "--disable",
    "image_generation",
    "--disable",
    "multi_agent",
    "--disable",
    "multi_agent_v2",
    "--disable",
    "plugins",
    "--disable",
    "remote_plugin",
    "--disable",
    "plugin_sharing",
    "--disable",
    "skill_search",
    "--disable",
    "skill_mcp_dependency_install",
    "--disable",
    "shell_tool",
    "--disable",
    "shell_snapshot",
    "--disable",
    "unified_exec",
    "--disable",
    "code_mode_host",
    "--disable",
    "goals",
    "--disable",
    "hooks",
    "--disable",
    "memories",
    "--disable",
    "tool_suggest",
    "--disable",
    "workspace_dependencies",
    "-c",
    "project_doc_max_bytes=0",
    "-c",
    "project_doc_fallback_filenames=[]",
    "-c",
    'web_search="disabled"',
    "-c",
    "mcp_servers={}",
)

MAX_AUTH_JSON_BYTES = 65_536


def _install_authorized_auth(source: str | Path, codex_home: Path) -> None:
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        raise IsolationError("authorized auth file path must be absolute")
    try:
        source_stat = source_path.lstat()
    except OSError as error:
        raise IsolationError("authorized auth file is unavailable") from error
    if (
        source_path.is_symlink()
        or not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_uid != os.getuid()
        or source_stat.st_mode & 0o077
        or not 0 < source_stat.st_size <= MAX_AUTH_JSON_BYTES
    ):
        raise IsolationError("authorized auth file failed security validation")

    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise IsolationError("authorized auth copy requires O_NOFOLLOW")
    try:
        source_fd = os.open(source_path, source_flags | no_follow)
    except OSError as error:
        raise IsolationError("authorized auth file could not be opened safely") from error
    temporary = codex_home / ".auth.json.tmp"
    destination_fd: int | None = None
    try:
        opened_stat = os.fstat(source_fd)
        if (
            opened_stat.st_dev != source_stat.st_dev
            or opened_stat.st_ino != source_stat.st_ino
            or opened_stat.st_size != source_stat.st_size
            or not stat.S_ISREG(opened_stat.st_mode)
        ):
            raise IsolationError("authorized auth file changed during validation")
        destination_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        remaining = opened_stat.st_size
        while remaining:
            chunk = os.read(source_fd, min(remaining, 16_384))
            if not chunk:
                raise IsolationError("authorized auth file changed during copy")
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise IsolationError("authorized auth file could not be copied")
                view = view[written:]
            remaining -= len(chunk)
        if os.read(source_fd, 1):
            raise IsolationError("authorized auth file exceeds validated size")
        os.fchmod(destination_fd, 0o600)
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = None
        os.replace(temporary, codex_home / "auth.json")
        directory_fd = os.open(codex_home, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class IsolatedCapsule:
    """Private app-server home, sqlite home, and empty working directory."""

    root: Path
    codex_home: Path
    sqlite_home: Path
    workdir: Path
    env: Mapping[str, str]
    codex_executable: str

    @classmethod
    def create(
        cls,
        root: str | Path | None = None,
        *,
        auth_json: str | Path | None = None,
    ) -> IsolatedCapsule:
        if root is None:
            parent = None
        else:
            parent = Path(root)
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("capsule parent must be an existing real directory")
            parent = parent.resolve()
        base = Path(tempfile.mkdtemp(prefix="mutation-forge-codex-", dir=parent))
        for path in (base,):
            path.chmod(0o700)
        homes = (base / "codex-home", base / "codex-sqlite", base / "codex-work")
        for path in homes:
            path.mkdir(mode=0o700, exist_ok=True)
            path.chmod(0o700)
        if auth_json is not None:
            try:
                _install_authorized_auth(auth_json, homes[0])
            except Exception:
                shutil.rmtree(base, ignore_errors=True)
                raise
        # Do not inherit arbitrary user configuration, proxy credentials, or
        # token-bearing variables.  PATH is retained solely to locate codex.
        env: dict[str, str] = {"PATH": os.environ.get("PATH", "")}
        for key in ("LANG", "LC_ALL", "TZ"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        env.update({"CODEX_HOME": str(homes[0]), "CODEX_SQLITE_HOME": str(homes[1])})
        executable = shutil.which("codex")
        if executable is None:
            raise IsolationError("trusted codex executable was not found")
        executable_path = Path(executable).resolve()
        if not executable_path.is_file() or not (executable_path.stat().st_mode & stat.S_IXUSR):
            raise IsolationError("trusted codex executable is not executable")
        return cls(base, homes[0], homes[1], homes[2], env, str(executable_path))

    def cleanup(self) -> None:
        """Best-effort cleanup of the capsule (safe only for this exact root)."""
        if self.root.name.startswith("mutation-forge-codex-") and self.root.is_dir():
            shutil.rmtree(self.root, ignore_errors=True)


def sanitized_environment(capsule: IsolatedCapsule) -> dict[str, str]:
    """Return a fresh environment mapping suitable for ``Popen``."""
    return dict(capsule.env)


class IsolationError(RuntimeError):
    pass


def linux_resource_preexec(
    *,
    cpu_seconds: int = 120,
    address_space_bytes: int = 2 * 1024 * 1024 * 1024,
    file_bytes: int = 8 * 1024 * 1024,
    open_files: int = 256,
    processes: int = 1024,
) -> None:
    """Apply conservative child limits before exec; unsupported systems fail closed."""
    if not sys.platform.startswith("linux"):
        raise IsolationError("resource limits require Linux")
    limits = (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_AS, (address_space_bytes, address_space_bytes)),
        (resource.RLIMIT_FSIZE, (file_bytes, file_bytes)),
        (resource.RLIMIT_NOFILE, (open_files, open_files)),
    )
    for kind, values in limits:
        resource.setrlimit(kind, values)
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
