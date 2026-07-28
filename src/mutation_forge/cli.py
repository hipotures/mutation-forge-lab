from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.table import Table

from mutation_forge import __version__
from mutation_forge.backends.heg import HegBackend
from mutation_forge.config import LabConfig, load_config
from mutation_forge.evaluation.benchmark import load_summary, run_benchmark
from mutation_forge.evaluation.dataset import build_dataset
from mutation_forge.models import JsonValue


def _doctor(heg_repo: Path, run_root: Path) -> int:
    checks: list[dict[str, JsonValue]] = []

    def check(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append(
            {"name": name, "ok": ok, "detail": detail, "required": required}
        )

    check(
        "python",
        sys.version_info >= (3, 12),
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    check("package", bool(__version__), f"mutation-forge-lab {__version__}")
    check("sqlite", sqlite3.sqlite_version_info >= (3, 35), sqlite3.sqlite_version)
    try:
        rich = importlib.import_module("rich")
        rich_version = getattr(rich, "__version__", "installed")
        check("rich", True, str(rich_version))
    except ImportError as error:
        check("rich", False, str(error))
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=run_root):
            pass
        check("run_directory", True, str(run_root.resolve()))
    except OSError as error:
        check("run_directory", False, str(error))

    backend: HegBackend | None = None
    try:
        backend = HegBackend(heg_repo)
        required_methods = (
            "generate_seed",
            "validate",
            "score",
            "exact_verify",
            "canonical_hash",
            "state_hash",
            "apply_rewrite",
            "propose_rewrite",
        )
        missing = [name for name in required_methods if not hasattr(backend, name)]
        check("heg_repository", True, str(backend.repo))
        check("heg_commit", len(backend.commit) == 40, backend.commit)
        check("heg_clean", not backend.dirty, "clean" if not backend.dirty else "dirty")
        check(
            "heg_interfaces",
            not missing,
            "all expected interfaces present" if not missing else f"missing: {missing}",
        )
        seed = backend.generate_seed(order=10, seed=101)
        validation = backend.validate(seed)
        check("heg_seed_validation", validation.valid, "; ".join(validation.errors) or "valid")
        score = backend.score(seed, witness_cap=8)
        check(
            "heg_scorer",
            score.valid,
            f"{backend.score_implementation}; counts={score.capped_cycle_counts}",
        )
    except Exception as error:
        check("heg_repository", False, f"{type(error).__name__}: {error}")
    finally:
        if backend is not None:
            backend.close()

    skill_path = Path.home() / ".codex" / "skills" / "codex-app-server" / "SKILL.md"
    check(
        "app_server_skill",
        skill_path.is_file(),
        str(skill_path) if skill_path.is_file() else "not discovered; informational in Stage 1",
        required=False,
    )
    required_ok = all(
        bool(item["ok"]) for item in checks if bool(item["required"])
    )
    console = Console()
    table = Table(title="Mutation Forge doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for item in checks:
        status = "PASS" if item["ok"] else ("INFO" if not item["required"] else "FAIL")
        table.add_row(str(item["name"]), status, str(item["detail"]))
    console.print(table)
    return 0 if required_ok else 1


def _build_dataset(config: LabConfig, *, json_output: bool) -> int:
    backend = HegBackend(config.heg.repo)
    try:
        manifest = build_dataset(config, backend, heg_commit=backend.commit)
        dataset_root = config.run.run_root / "datasets"
        dataset_root.mkdir(parents=True, exist_ok=True)
        destination = dataset_root / f"{manifest['manifest_hash']}.json"
        serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if destination.exists() and destination.read_text() != serialized:
            raise RuntimeError("immutable dataset manifest hash collision")
        destination.write_text(serialized)
        result = {
            "status": "completed",
            "dataset_manifest": str(destination),
            "manifest_hash": manifest["manifest_hash"],
            "entries": len(cast(list[object], manifest["entries"])),
            "heg_commit": backend.commit,
        }
        if json_output:
            print(json.dumps(result, sort_keys=True))
        else:
            Console().print(
                f"Dataset manifest: {destination}\n"
                f"Entries: {result['entries']}\n"
                f"Hash: {result['manifest_hash']}"
            )
        return 0
    finally:
        backend.close()


def _inspect(path: Path, *, json_output: bool) -> int:
    summary = load_summary(path)
    if json_output:
        print(json.dumps(summary, sort_keys=True))
        return 0
    console = Console()
    table = Table(title=f"Mutation Forge run: {path}")
    table.add_column("Field")
    table.add_column("Value")
    for field in (
        "status",
        "run_id",
        "summary_hash",
        "dataset_manifest_hash",
        "evaluations",
        "evaluations_per_second",
        "elapsed_seconds",
    ):
        table.add_row(field, str(summary.get(field, "-")))
    console.print(table)
    fitness = summary.get("fitness")
    if isinstance(fitness, dict):
        console.print_json(json.dumps(fitness))
    return 0


def _compare(left_path: Path, right_path: Path, *, json_output: bool) -> int:
    left = load_summary(left_path)
    right = load_summary(right_path)
    result: dict[str, Any] = {
        "status": "completed",
        "run_a": str(left_path),
        "run_b": str(right_path),
        "same_dataset": left.get("dataset_manifest_hash")
        == right.get("dataset_manifest_hash"),
        "same_summary_hash": left.get("summary_hash") == right.get("summary_hash"),
        "summary_hash_a": left.get("summary_hash"),
        "summary_hash_b": right.get("summary_hash"),
        "fitness_a": left.get("fitness"),
        "fitness_b": right.get("fitness"),
    }
    if json_output:
        print(json.dumps(result, sort_keys=True))
    else:
        Console().print_json(json.dumps(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mforge")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--heg-repo", type=Path, default=Path("../heg"))
    doctor.add_argument("--run-root", type=Path, default=Path("./runs"))

    dataset = commands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_commands.add_parser("build")
    dataset_build.add_argument("--config", type=Path, required=True)
    dataset_build.add_argument("--json", action="store_true")

    baseline = commands.add_parser("baseline")
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_run = baseline_commands.add_parser("run")
    baseline_run.add_argument("--config", type=Path, required=True)
    baseline_run.add_argument("--json", action="store_true")

    inspect_command = commands.add_parser("inspect")
    inspect_command.add_argument("run", type=Path)
    inspect_command.add_argument("--json", action="store_true")

    compare = commands.add_parser("compare")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args.heg_repo, args.run_root)
        if args.command == "dataset" and args.dataset_command == "build":
            return _build_dataset(load_config(args.config), json_output=args.json)
        if args.command == "baseline" and args.baseline_command == "run":
            config = load_config(args.config)
            result = run_benchmark(config, output="json" if args.json else None)
            if not args.json:
                Console().print(f"Run artifacts: {result.run_path}")
            return 0
        if args.command == "inspect":
            return _inspect(args.run, json_output=args.json)
        if args.command == "compare":
            return _compare(args.run_a, args.run_b, json_output=args.json)
    except Exception as error:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "event_type": "run_failed",
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                )
            )
        else:
            Console(stderr=True).print(f"[red]{type(error).__name__}: {error}[/red]")
        return 1
    parser.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
