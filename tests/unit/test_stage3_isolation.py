from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mutation_forge.stage3.isolation import (
    THIN_APP_SERVER_ARGS,
    IsolatedCapsule,
    IsolationError,
    isolated_config,
    sanitized_environment,
)


def test_strict_app_server_args_disable_agent_capabilities() -> None:
    assert THIN_APP_SERVER_ARGS[:4] == ("codex", "app-server", "--stdio", "--strict-config")
    assert "shell_tool" in THIN_APP_SERVER_ARGS
    assert "plugins" in THIN_APP_SERVER_ARGS
    assert 'web_search="disabled"' in THIN_APP_SERVER_ARGS
    assert "mcp_servers={}" in THIN_APP_SERVER_ARGS


@pytest.mark.parametrize("sandbox_mode", ["read-only", "danger-full-access"])
def test_private_config_supports_both_sandbox_modes(sandbox_mode: str) -> None:
    config = isolated_config(sandbox_mode=sandbox_mode, approval_policy="never").decode()
    assert f'sandbox_mode = "{sandbox_mode}"' in config
    assert 'approval_policy = "never"' in config


def test_private_config_rejects_unapproved_modes() -> None:
    with pytest.raises(IsolationError, match="sandbox mode"):
        isolated_config(sandbox_mode="workspace-write", approval_policy="never")
    with pytest.raises(IsolationError, match="approval policy"):
        isolated_config(sandbox_mode="read-only", approval_policy="on-request")


def test_capsule_copies_only_secure_explicit_auth_and_sanitizes_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth = tmp_path / "auth.json"
    secret = '{"access_token":"Bearer private-token-value"}'
    auth.write_text(secret)
    auth.chmod(0o600)
    monkeypatch.setenv("HOME", "/private/home")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")
    capsule = IsolatedCapsule.create(tmp_path, auth_json=auth)
    try:
        copied = capsule.codex_home / "auth.json"
        assert copied.read_text() == secret
        assert stat.S_IMODE(copied.stat().st_mode) == 0o600
        assert stat.S_IMODE(capsule.root.stat().st_mode) == 0o700
        assert set(sanitized_environment(capsule)).issuperset(
            {"PATH", "CODEX_HOME", "CODEX_SQLITE_HOME"}
        )
        assert capsule.env["HOME"] == str(capsule.root)
        assert "UNRELATED_SECRET" not in capsule.env
        assert str(capsule.workdir) not in str(copied)
    finally:
        capsule.cleanup()
    assert not capsule.root.exists()


def test_capsule_rejects_insecure_or_symlinked_auth(tmp_path: Path) -> None:
    insecure = tmp_path / "insecure-auth.json"
    insecure.write_text("{}")
    insecure.chmod(0o644)
    with pytest.raises(IsolationError, match="security validation"):
        IsolatedCapsule.create(tmp_path, auth_json=insecure)

    secure = tmp_path / "secure-auth.json"
    secure.write_text("{}")
    secure.chmod(0o600)
    symlink = tmp_path / "auth-link.json"
    os.symlink(secure, symlink)
    with pytest.raises(IsolationError, match="security validation"):
        IsolatedCapsule.create(tmp_path, auth_json=symlink)


def test_capsule_requires_existing_real_parent_and_absolute_auth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing real directory"):
        IsolatedCapsule.create(tmp_path / "absent")
    with pytest.raises(IsolationError, match="absolute"):
        IsolatedCapsule.create(tmp_path, auth_json="relative-auth.json")
