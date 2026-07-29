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
    def create(cls, root: str | Path | None = None) -> IsolatedCapsule:
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
