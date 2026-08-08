"""Run or resume one explicit guarded ordinary-Python preview campaign."""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mutation_forge.cli import main as mforge_main
from mutation_forge.native_v3_python.preview import (
    PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(tempfile.gettempdir())
        / "mutation-forge-native-v3-python-m6-preview",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Resume the exact retained preview configuration.",
    )
    parser.add_argument("--heg-repo", type=Path, default=PROJECT_ROOT.parent / "heg")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--effort", default="medium")
    return parser


def _new_config(args: argparse.Namespace) -> Path:
    run_id = "native-v3-python-m6-" + datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    config_dir = args.output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / f"{run_id}.toml"
    if config.exists():
        raise RuntimeError(f"refusing to reuse preview configuration: {config}")
    config.write_text(
        f'''schema_version = "{PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION}"
protocol = "native-v3-python-v1"
exp_id = "{run_id}"
workspace = "{(args.output_root / "workspaces").resolve().as_posix()}"

[python_preview]
model = "{args.model}"
effort = "{args.effort}"
timeout_seconds = 300
heg_repo = "{args.heg_repo.resolve().as_posix()}"
''',
        encoding="utf-8",
    )
    return config


def main() -> int:
    args = _parser().parse_args()
    config = (
        args.config.resolve(strict=True)
        if args.config is not None
        else _new_config(args)
    )
    return mforge_main(
        [
            "experiment",
            "run",
            "--config",
            str(config),
            "--json",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
