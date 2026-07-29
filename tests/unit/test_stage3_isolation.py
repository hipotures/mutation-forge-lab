from __future__ import annotations

import stat
from pathlib import Path

import pytest

from mutation_forge.stage3.isolation import (
    MAX_AUTH_JSON_BYTES,
    IsolatedCapsule,
    IsolationError,
)


def _auth_file(path: Path, content: bytes = b'{"fixture":"credential"}\n') -> Path:
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def test_authorized_auth_is_copied_only_to_private_codex_home(tmp_path: Path) -> None:
    source = _auth_file(tmp_path / "authorized-auth.json")
    parent = tmp_path / "capsules"
    parent.mkdir()

    capsule = IsolatedCapsule.create(parent, auth_json=source)
    try:
        copied = capsule.codex_home / "auth.json"
        assert copied.read_bytes() == source.read_bytes()
        assert stat.S_IMODE(copied.stat().st_mode) == 0o600
        assert sorted(path.name for path in capsule.codex_home.iterdir()) == ["auth.json"]
        assert stat.S_IMODE(capsule.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(capsule.codex_home.stat().st_mode) == 0o700
    finally:
        capsule.cleanup()


def test_authorized_auth_rejects_symlink(tmp_path: Path) -> None:
    source = _auth_file(tmp_path / "authorized-auth.json")
    link = tmp_path / "auth-link.json"
    link.symlink_to(source)
    parent = tmp_path / "capsules"
    parent.mkdir()

    with pytest.raises(IsolationError, match="security validation"):
        IsolatedCapsule.create(parent, auth_json=link)
    assert not tuple(parent.iterdir())


def test_authorized_auth_rejects_permissive_mode(tmp_path: Path) -> None:
    source = _auth_file(tmp_path / "authorized-auth.json")
    source.chmod(0o640)
    parent = tmp_path / "capsules"
    parent.mkdir()

    with pytest.raises(IsolationError, match="security validation"):
        IsolatedCapsule.create(parent, auth_json=source)
    assert not tuple(parent.iterdir())


def test_authorized_auth_rejects_oversized_file(tmp_path: Path) -> None:
    source = _auth_file(tmp_path / "authorized-auth.json", b"x" * (MAX_AUTH_JSON_BYTES + 1))
    parent = tmp_path / "capsules"
    parent.mkdir()

    with pytest.raises(IsolationError, match="security validation"):
        IsolatedCapsule.create(parent, auth_json=source)
    assert not tuple(parent.iterdir())
