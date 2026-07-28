from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def heg_repo(project_root: Path) -> Path:
    repo = project_root.parent / "heg"
    if not (repo / "src" / "sglab").is_dir():
        pytest.fail(f"mandatory sibling HEG repository is unavailable: {repo}")
    return repo
