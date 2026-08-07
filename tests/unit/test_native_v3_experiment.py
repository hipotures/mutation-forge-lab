from __future__ import annotations

import json
from pathlib import Path

import pytest

from mutation_forge import cli
from mutation_forge.native_v3.experiment import (
    MULTI_PROGRAM_BATCH,
    PERSISTENT_SINGLE_AST,
    SLOT_SPECIFIC_OUTPUT_CONTRACT,
    V2_PROTOCOL,
    V3_SELECTOR,
    experiment_protocol,
    load_v3_config,
)


def _config(tmp_path: Path, *, extra: str = "") -> str:
    return f'''schema_version = "mforge.experiment.v3"
protocol = "v3"
exp_id = "v3-run"
workspace = "{(tmp_path / "workspace").as_posix()}"
{extra}

[v3]
model = "gpt-5.6-luna"
effort = "high"
timeout_seconds = 30
heg_repo = "{(tmp_path / "heg").as_posix()}"
'''


def test_protocol_omission_keeps_native_v2_default(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text('schema_version = "mforge.experiment.v2"\n', encoding="utf-8")

    assert experiment_protocol(path) == V2_PROTOCOL


def test_v3_config_is_explicit_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(tmp_path), encoding="utf-8")

    config = load_v3_config(path)

    assert experiment_protocol(path) == V3_SELECTOR
    assert config.exp_id == "v3-run"
    assert config.workspace == tmp_path / "workspace"
    assert config.heg_repo == tmp_path / "heg"
    assert config.communication_mode == MULTI_PROGRAM_BATCH
    assert config.output_contract is None


def test_v3_communication_mode_is_explicitly_selectable(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        _config(tmp_path).replace(
            'heg_repo = "',
            f'communication_mode = "{MULTI_PROGRAM_BATCH}"\nheg_repo = "',
        ),
        encoding="utf-8",
    )

    assert load_v3_config(path).communication_mode == MULTI_PROGRAM_BATCH

    path.write_text(
        _config(tmp_path).replace(
            'heg_repo = "',
            (
                f'communication_mode = "{PERSISTENT_SINGLE_AST}"\n'
                f'output_contract = "{SLOT_SPECIFIC_OUTPUT_CONTRACT}"\n'
                'heg_repo = "'
            ),
        ),
        encoding="utf-8",
    )
    config = load_v3_config(path)
    assert config.communication_mode == PERSISTENT_SINGLE_AST
    assert config.output_contract == SLOT_SPECIFIC_OUTPUT_CONTRACT

    path.write_text(
        _config(tmp_path).replace(
            'heg_repo = "',
            f'communication_mode = "{PERSISTENT_SINGLE_AST}"\nheg_repo = "',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires explicit output_contract"):
        load_v3_config(path)

    path.write_text(
        _config(tmp_path).replace(
            'heg_repo = "',
            'communication_mode = "unknown"\nheg_repo = "',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="communication_mode must be one of"):
        load_v3_config(path)

    path.write_text(
        _config(tmp_path).replace(
            'heg_repo = "',
            'output_contract = "slot_specific"\nheg_repo = "',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only valid for persistent_single_ast"):
        load_v3_config(path)


@pytest.mark.parametrize("field", ["model", "search", "evaluation", "resources"])
def test_v3_rejects_native_v2_sections(tmp_path: Path, field: str) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(tmp_path, extra=f"\n[{field}]\n"), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot contain Native v2 fields"):
        load_v3_config(path)

    assert not (tmp_path / "workspace").exists()


def test_v3_rejects_old_names_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(
        _config(tmp_path).replace(
            'protocol = "v3"',
            'protocol = "native-v3-preview"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported experiment protocol selector"):
        experiment_protocol(path)

    path.write_text(
        _config(tmp_path).replace(
            'schema_version = "mforge.experiment.v3"',
            'schema_version = "mforge.experiment.v3-preview.v2"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="v3 requires schema_version"):
        load_v3_config(path)

    path.write_text(
        _config(tmp_path).replace(
            'heg_repo = "',
            'population_size = 8\nheg_repo = "',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unsupported \[v3\]"):
        load_v3_config(path)


def test_public_cli_routes_v3_run_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "experiment.toml"
    path.write_text(_config(tmp_path), encoding="utf-8")
    calls: list[str] = []

    def run(config_path: str | Path) -> dict[str, object]:
        calls.append(f"run:{Path(config_path).name}")
        return {"state": "completed", "protocol": V3_SELECTOR}

    def status(config_path: str | Path) -> dict[str, object]:
        calls.append(f"status:{Path(config_path).name}")
        return {"state": "completed", "protocol": V3_SELECTOR}

    monkeypatch.setattr(cli, "run_v3", run)
    monkeypatch.setattr(cli, "v3_status", status)

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
        V3_SELECTOR,
        V3_SELECTOR,
    ]
    assert calls == ["run:experiment.toml", "status:experiment.toml"]


def test_v3_rejects_v2_only_run_options_before_execution(
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

    monkeypatch.setattr(cli, "run_v3", run)

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
