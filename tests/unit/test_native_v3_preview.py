from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge import cli
from mutation_forge.native_v3.preview import (
    V2_PROTOCOL,
    V3_PREVIEW_SELECTOR,
    experiment_protocol,
    load_v3_preview_config,
)


def _config(tmp_path: Path, *, extra: str = "") -> str:
    return f'''schema_version = "mforge.experiment.v3-preview.v1"
protocol = "native-v3-preview"
exp_id = "preview"
workspace = "{(tmp_path / "workspace").as_posix()}"
{extra}

[native_v3_preview]
model = "gpt-5.6-luna"
effort = "high"
timeout_seconds = 30
heg_repo = "{(tmp_path / "heg").as_posix()}"
'''


def test_protocol_omission_keeps_native_v2_default(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text('schema_version = "mforge.experiment.v2"\n', encoding="utf-8")

    assert experiment_protocol(path) == V2_PROTOCOL


def test_preview_config_is_explicit_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(tmp_path), encoding="utf-8")

    config = load_v3_preview_config(path)

    assert experiment_protocol(path) == V3_PREVIEW_SELECTOR
    assert config.exp_id == "preview"
    assert config.workspace == tmp_path / "workspace"
    assert config.heg_repo == tmp_path / "heg"


@pytest.mark.parametrize("field", ["model", "search", "evaluation", "resources"])
def test_preview_rejects_native_v2_sections(tmp_path: Path, field: str) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(tmp_path, extra=f"\n[{field}]\n"), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot contain Native v2 fields"):
        load_v3_preview_config(path)

    assert not (tmp_path / "workspace").exists()


def test_preview_rejects_unknown_selector_and_fields(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        _config(tmp_path).replace(
            'protocol = "native-v3-preview"',
            'protocol = "native-v3"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported experiment protocol selector"):
        experiment_protocol(path)

    path.write_text(
        _config(tmp_path).replace(
            'heg_repo = "',
            'population_size = 8\nheg_repo = "',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unsupported \[native_v3_preview\]"):
        load_v3_preview_config(path)


def test_public_cli_routes_preview_run_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(tmp_path), encoding="utf-8")
    calls: list[str] = []

    def run(config_path: str | Path) -> dict[str, object]:
        calls.append(f"run:{Path(config_path).name}")
        return {"state": "completed", "protocol": V3_PREVIEW_SELECTOR}

    def status(config_path: str | Path) -> dict[str, object]:
        calls.append(f"status:{Path(config_path).name}")
        return {"state": "completed", "protocol": V3_PREVIEW_SELECTOR}

    monkeypatch.setattr(cli, "run_v3_preview", run)
    monkeypatch.setattr(cli, "v3_preview_status", status)

    assert (
        cli.main(["experiment", "run", "--config", str(path), "--json"])
        == 0
    )
    assert (
        cli.main(["experiment", "status", "--config", str(path), "--json"])
        == 0
    )
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [payload["protocol"] for payload in payloads] == [
        V3_PREVIEW_SELECTOR,
        V3_PREVIEW_SELECTOR,
    ]
    assert calls == ["run:experiment.toml", "status:experiment.toml"]


def test_preview_rejects_v2_only_run_options_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(tmp_path), encoding="utf-8")
    called = False

    def run(_: str | Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {"state": "completed"}

    monkeypatch.setattr(cli, "run_v3_preview", run)

    assert (
        cli.main(
            [
                "experiment",
                "run",
                "--config",
                str(path),
                "--dashboard",
                "--json",
            ]
        )
        == 1
    )
    assert called is False
