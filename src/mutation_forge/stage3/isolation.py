"""Private process/filesystem isolation for Codex app-server."""

from __future__ import annotations

import os
import resource
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
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
    "--enable",
    "use_linux_sandbox_bwrap",
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
APP_SERVER_SANDBOX_MODES = frozenset({"read-only", "danger-full-access"})
APP_SERVER_APPROVAL_POLICIES = frozenset({"never"})


def secure_capsule_parent() -> Path:
    """Return a private cache parent outside the repository and system temp dir."""
    configured = os.environ.get("XDG_CACHE_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".cache"
    if not base.is_absolute() or base.is_symlink():
        raise IsolationError("cache parent must be an absolute real directory")
    base_resolved = base.resolve(strict=False)
    if base_resolved == Path("/tmp") or str(base_resolved).startswith("/tmp/"):
        raise IsolationError("capsule cache parent must not be a temporary directory")
    with suppress(ValueError):
        base_resolved.relative_to(Path.cwd().resolve(strict=True))
        raise IsolationError("capsule cache parent must not be inside the repository")
    parent = base / "mutation-forge-lab" / "capsules"
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent.parent.chmod(0o700)
    parent.chmod(0o700)
    resolved = parent.resolve(strict=True)
    if resolved in {Path("/"), Path("/tmp")} or str(resolved).startswith("/tmp/"):
        raise IsolationError("capsule cache parent must not be a temporary directory")
    return resolved


class IsolationError(RuntimeError):
    pass


def isolated_config(*, sandbox_mode: str, approval_policy: str) -> bytes:
    if sandbox_mode not in APP_SERVER_SANDBOX_MODES:
        raise IsolationError("unsupported app-server sandbox mode")
    if approval_policy not in APP_SERVER_APPROVAL_POLICIES:
        raise IsolationError("unsupported app-server approval policy")
    return (
        f'sandbox_mode = "{sandbox_mode}"\n'
        f'approval_policy = "{approval_policy}"\n\n'
        "[features]\n"
        "use_linux_sandbox_bwrap = true\n"
    ).encode()


def _install_authorized_auth(source: str | Path, codex_home: Path) -> None:
    path = Path(source).expanduser()
    if not path.is_absolute():
        raise IsolationError("authorized auth file path must be absolute")
    try:
        st = path.lstat()
    except OSError as exc:
        raise IsolationError("authorized auth file is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(st.st_mode)
        or st.st_uid != os.getuid()
        or st.st_mode & 0o077
        or not 0 < st.st_size <= MAX_AUTH_JSON_BYTES
    ):
        raise IsolationError("authorized auth file failed security validation")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise IsolationError("authorized auth copy requires O_NOFOLLOW")
    fd = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    tmp = codex_home / ".auth.json.tmp"
    out: int | None = None
    try:
        ost = os.fstat(fd)
        if (ost.st_dev, ost.st_ino, ost.st_size) != (
            st.st_dev,
            st.st_ino,
            st.st_size,
        ) or not stat.S_ISREG(ost.st_mode):
            raise IsolationError("authorized auth file changed during validation")
        out = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600
        )
        remaining = ost.st_size
        while remaining:
            data = os.read(fd, min(remaining, 16384))
            if not data:
                raise IsolationError("authorized auth file changed during copy")
            os.write(out, data)
            remaining -= len(data)
        if os.read(fd, 1):
            raise IsolationError("authorized auth file exceeds validated size")
        os.fchmod(out, 0o600)
        os.fsync(out)
        os.close(out)
        out = None
        os.replace(tmp, codex_home / "auth.json")
        dfd = os.open(codex_home, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        os.close(fd)
        if out is not None:
            os.close(out)
        with suppress(FileNotFoundError):
            tmp.unlink()


@dataclass(frozen=True, slots=True)
class IsolatedCapsule:
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
        workdir: str | Path | None = None,
        sandbox_mode: str = "danger-full-access",
        approval_policy: str = "never",
    ) -> IsolatedCapsule:
        parent: str | None = None
        if root is not None:
            p = Path(root)
            if p.is_symlink() or not p.is_dir():
                raise ValueError("capsule parent must be an existing real directory")
            parent = str(p.resolve())
        base = Path(tempfile.mkdtemp(prefix="mutation-forge-codex-", dir=parent))
        base.chmod(0o700)
        homes = (base / "codex-home", base / "codex-sqlite", base / "codex-work")
        try:
            for p in homes:
                p.mkdir(mode=0o700)
                p.chmod(0o700)
            config_path = homes[0] / "config.toml"
            config_fd = os.open(
                config_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                os.write(
                    config_fd,
                    isolated_config(
                        sandbox_mode=sandbox_mode,
                        approval_policy=approval_policy,
                    ),
                )
                os.fchmod(config_fd, 0o600)
                os.fsync(config_fd)
            finally:
                os.close(config_fd)
            if auth_json is not None:
                _install_authorized_auth(auth_json, homes[0])
            exe = shutil.which("codex")
            if exe is None:
                raise IsolationError("trusted codex executable was not found")
            ep = Path(exe).resolve()
            if not ep.is_file() or not ep.stat().st_mode & stat.S_IXUSR:
                raise IsolationError("trusted codex executable is not executable")
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(base),
                "CODEX_HOME": str(homes[0]),
                "CODEX_SQLITE_HOME": str(homes[1]),
            }
            managed_root = os.environ.get("CODEX_MANAGED_PACKAGE_ROOT")
            if managed_root:
                managed_path = Path(managed_root)
                if not managed_path.is_absolute() or not managed_path.is_dir():
                    raise IsolationError("Codex managed package root is invalid")
                env["CODEX_MANAGED_PACKAGE_ROOT"] = str(managed_path.resolve(strict=True))
            if os.environ.get("CODEX_MANAGED_BY_NPM") == "1":
                env["CODEX_MANAGED_BY_NPM"] = "1"
            if os.environ.get("CODEX_CI") == "1":
                env["CODEX_CI"] = "1"
            for k in ("LANG", "LC_ALL", "TZ"):
                if os.environ.get(k):
                    env[k] = os.environ[k]
            runtime_workdir = Path(workdir) if workdir is not None else homes[2]
            if (
                not runtime_workdir.is_absolute()
                or runtime_workdir.is_symlink()
                or not runtime_workdir.is_dir()
            ):
                raise IsolationError(
                    "runtime work directory must be an existing absolute directory"
                )
            return cls(base, homes[0], homes[1], runtime_workdir, env, str(ep))
        except Exception:
            shutil.rmtree(base, ignore_errors=True)
            raise

    def cleanup(self) -> None:
        if self.root.name.startswith("mutation-forge-codex-") and self.root.is_dir():
            shutil.rmtree(self.root, ignore_errors=True)


def sanitized_environment(capsule: IsolatedCapsule) -> dict[str, str]:
    return dict(capsule.env)


def linux_resource_preexec(
    *,
    cpu_seconds: int = 120,
    address_space_bytes: int = 2 * 1024 * 1024 * 1024,
    file_bytes: int = 8 * 1024 * 1024,
    open_files: int = 256,
    processes: int = 1024,
) -> None:
    if not sys.platform.startswith("linux"):
        raise IsolationError("resource limits require Linux")
    for kind, vals in (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_AS, (address_space_bytes, address_space_bytes)),
        (resource.RLIMIT_FSIZE, (file_bytes, file_bytes)),
        (resource.RLIMIT_NOFILE, (open_files, open_files)),
    ):
        resource.setrlimit(kind, vals)
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
