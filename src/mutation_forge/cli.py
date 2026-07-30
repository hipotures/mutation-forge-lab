from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import sys
import tempfile
import tomllib
from collections.abc import Mapping
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
from mutation_forge.sandbox.config import load_policy_config
from mutation_forge.sandbox.policy import evaluate_policy, probe_policy
from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage2b.config import Stage2BConfig, load_stage2b_config
from mutation_forge.stage2b.evaluation import (
    evaluate_source_policy,
    inspect_proposals,
    run_stage2b_compare,
)
from mutation_forge.stage2c.config import load_stage2c_config
from mutation_forge.stage2c.evaluation import (
    run_pool_oracle,
    run_stage2c_control,
    run_stage2c_matrix,
)
from mutation_forge.stage2d.config import load_stage2d_config
from mutation_forge.stage2d.evaluation import (
    plan_stage2d,
    reduce_stage2d,
    run_stage2d_shard,
    verify_stage2d_replay,
)
from mutation_forge.stage3 import commands as stage3_commands
from mutation_forge.stage4 import commands as stage4_commands


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
        if score is None:
            raise RuntimeError("initial HEG score cannot be cutoff-dominated")
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
    timing_profile = summary.get("timing_profile")
    if isinstance(timing_profile, dict):
        console.print_json(json.dumps(timing_profile))
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


def _emit_policy_result(result: object, *, json_output: bool) -> None:
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if json_output:
        print(canonical)
    else:
        Console().print_json(canonical)


def _stage2d(args: argparse.Namespace) -> int:
    if args.stage2d_command == "verify-replay":
        result = verify_stage2d_replay(
            args.primary,
            args.replay,
            args.output,
        )
    else:
        config = load_stage2d_config(args.config)
        if args.stage2d_command == "plan":
            result = plan_stage2d(config)
        elif args.stage2d_command == "run-shard":
            result = run_stage2d_shard(
                config,
                args.shard,
                args.output_dir,
            )
        elif args.stage2d_command == "reduce":
            result = reduce_stage2d(
                config,
                args.input_root,
                args.output_dir,
                bootstrap_workers=args.workers,
            )
        else:
            raise ValueError(f"unknown Stage 2D command {args.stage2d_command!r}")
    _emit_policy_result(result, json_output=args.json)
    return 0


def _emit_stage3(result: Mapping[str, Any], *, json_output: bool) -> None:
    """Render one canonical Stage 3 result in JSON or Rich mode.

    Both modes consume exactly the same compact, sorted JSON representation.
    This keeps machine output single-line and prevents Rich-only fields.
    """
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if json_output:
        print(payload)
    else:
        Console().print_json(payload)


def _stage3(args: argparse.Namespace) -> int:
    command = args.stage3_command
    if command == "appserver-doctor":
        result = stage3_commands.appserver_doctor(args.config, auth_json=args.auth_json)
    elif command == "freeze":
        result = stage3_commands.freeze(args.config)
    elif command == "generate":
        result = stage3_commands.generate(
            args.config,
            concurrency=args.concurrency,
            auth_json=args.auth_json,
        )
    elif command == "validate":
        result = stage3_commands.validate(args.run)
    elif command == "revalidate":
        result = stage3_commands.revalidate(args.config, args.run)
    elif command == "evaluate":
        result = stage3_commands.evaluate(args.config, args.run, workers=args.workers)
    elif command == "verify-replay":
        result = stage3_commands.verify_replay(args.primary, args.replay)
    elif command == "replay-generation":
        replay = getattr(stage3_commands, "replay_generation", None)
        if replay is None:
            raise ValueError("offline replay-generation is unavailable")
        result = replay(args.run)
    else:
        raise ValueError(f"unknown Stage 3 command {command!r}")
    _emit_stage3(result, json_output=args.json)
    return 0 if result.get("status") in {"completed", "ok"} else 1


def _stage4_observer(*, json_output: bool) -> Any:
    def emit(event: Mapping[str, Any]) -> None:
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if json_output:
            print(payload)
        else:
            label = event.get("event", "stage4")
            detail = ", ".join(
                f"{key}={value}" for key, value in event.items() if key != "event"
            )
            Console().print(f"[cyan]{label}[/cyan] {detail}")

    return emit


def _stage4(args: argparse.Namespace) -> int:
    command = args.stage4_command
    observer = _stage4_observer(json_output=args.json)
    if command == "doctor":
        result = stage4_commands.doctor(args.config, auth_json=args.auth_json)
    elif command == "freeze":
        result = stage4_commands.freeze(args.config, auth_json=args.auth_json)
    elif command == "evolve":
        result = stage4_commands.evolve(
            args.config,
            concurrency=args.concurrency,
            auth_json=args.auth_json,
            observer=observer,
        )
    elif command == "resume":
        result = stage4_commands.resume(
            args.run,
            config_path=args.config,
            auth_json=args.auth_json,
            observer=observer,
        )
    elif command == "archive":
        if args.stage4_archive_command == "inspect":
            result = stage4_commands.archive_inspect(args.run)
        elif args.stage4_archive_command == "reindex":
            result = stage4_commands.archive_reindex(args.run)
        else:
            raise ValueError(f"unknown Stage 4 archive command {args.stage4_archive_command!r}")
    elif command == "evaluate-candidate":
        result = stage4_commands.evaluate_candidate(
            args.run,
            args.program_id,
            pass_name=args.pass_name,
            workers=args.workers,
            config_path=args.config,
        )
    elif command == "freeze-validation":
        result = stage4_commands.freeze_validation(args.run, config_path=args.config)
    elif command == "validate":
        result = stage4_commands.validate(
            args.run,
            workers=args.workers,
            config_path=args.config,
            observer=observer,
        )
    elif command == "verify-replay":
        result = stage4_commands.verify_replay(args.run)
    else:
        raise ValueError(f"unknown Stage 4 command {command!r}")
    _emit_stage3(result, json_output=args.json)
    return 0 if result.get("status") in {"completed", "ok"} else 1


def _policy_validate(path: Path, *, json_output: bool) -> int:
    result = validate_policy(path.read_text()).as_dict()
    _emit_policy_result(result, json_output=json_output)
    return 0 if result["valid"] else 1


def _policy_probe(path: Path, *, json_output: bool) -> int:
    result = probe_policy(path.read_text())
    _emit_policy_result(result, json_output=json_output)
    return 0 if result["status"] == "completed" else 1


def _policy_evaluate(
    path: Path,
    config_path: Path,
    *,
    force_json: bool,
) -> int:
    with config_path.open("rb") as handle:
        schema_version = tomllib.load(handle).get("schema_version")
    if schema_version == "stage2b.1":
        result = evaluate_source_policy(path, load_stage2b_config(config_path))
        _emit_policy_result(result, json_output=force_json)
        return 0 if result["status"] == "completed" else 1
    config = load_policy_config(config_path)
    result = evaluate_policy(path, config)
    _emit_policy_result(
        result,
        json_output=force_json or config.output == "json",
    )
    return 0 if result["status"] == "completed" else 1


def _stage2b_policy_path(value: str, config: Stage2BConfig) -> Path:
    builtins = {
        "random": "fixtures/rankers/stage2b_random.py",
        "structural": "fixtures/rankers/stage2b_structural.py",
    }
    relative = builtins.get(value.lower())
    if relative is not None:
        return config.repositories.project_repo / relative
    return Path(value).resolve()


def _policy_compare(
    random_policy: str,
    structural_policy: str,
    config_path: Path,
    *,
    json_output: bool,
) -> int:
    config = load_stage2b_config(config_path)
    result = run_stage2b_compare(
        _stage2b_policy_path(random_policy, config),
        _stage2b_policy_path(structural_policy, config),
        config,
    )
    _emit_policy_result(
        result,
        json_output=json_output or config.run.output == "json",
    )
    return 0 if result["status"] == "completed" else 1


def _proposals_inspect(config_path: Path, *, json_output: bool) -> int:
    config = load_stage2b_config(config_path)
    result = inspect_proposals(config)
    _emit_policy_result(result, json_output=json_output)
    return 0


def _stage2c_diagnostic(
    command: str,
    config_path: Path,
    *,
    json_output: bool,
) -> int:
    resolved_config_path = config_path
    supplied_stage2b_control = False
    if command == "stage2c-control":
        with config_path.open("rb") as handle:
            schema_version = tomllib.load(handle).get("schema_version")
        if schema_version == "stage2b.1":
            supplied_stage2b_control = True
            resolved_config_path = config_path.parent / "stage2c-diagnostic.toml"
    config = load_stage2c_config(resolved_config_path)
    if (
        supplied_stage2b_control
        and config.control.stage2b_config != config_path.resolve()
    ):
        raise ValueError(
            "Stage 2C control command must reference its frozen Stage 2B config"
        )
    if command == "stage2c-control":
        result = run_stage2c_control(config)
    elif command == "pool-oracle":
        result = run_pool_oracle(config)
    elif command == "stage2c-matrix":
        result = run_stage2c_matrix(config)
    else:
        raise ValueError(f"unsupported diagnostic command: {command}")
    _emit_policy_result(
        result,
        json_output=json_output or config.run.output == "json",
    )
    return 0 if result["status"] == "completed" else 1


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

    proposals = commands.add_parser("proposals")
    proposal_commands = proposals.add_subparsers(
        dest="proposals_command",
        required=True,
    )
    proposals_inspect = proposal_commands.add_parser("inspect")
    proposals_inspect.add_argument("--config", type=Path, required=True)
    proposals_inspect.add_argument("--json", action="store_true")

    policy = commands.add_parser("policy")
    policy_commands = policy.add_subparsers(dest="policy_command", required=True)
    policy_validate = policy_commands.add_parser("validate")
    policy_validate.add_argument("policy", type=Path)
    policy_validate.add_argument("--json", action="store_true")
    policy_probe = policy_commands.add_parser("probe")
    policy_probe.add_argument("policy", type=Path)
    policy_probe.add_argument("--json", action="store_true")
    policy_evaluate = policy_commands.add_parser("evaluate")
    policy_evaluate.add_argument("policy", type=Path)
    policy_evaluate.add_argument("--config", type=Path, required=True)
    policy_evaluate.add_argument("--json", action="store_true")
    policy_compare = policy_commands.add_parser("compare")
    policy_compare.add_argument("random_policy")
    policy_compare.add_argument("structural_policy")
    policy_compare.add_argument("--config", type=Path, required=True)
    policy_compare.add_argument("--json", action="store_true")

    diagnostics = commands.add_parser("diagnostics")
    diagnostic_commands = diagnostics.add_subparsers(
        dest="diagnostics_command",
        required=True,
    )
    for name in ("stage2c-control", "pool-oracle", "stage2c-matrix"):
        diagnostic = diagnostic_commands.add_parser(name)
        diagnostic.add_argument("--config", type=Path, required=True)
        diagnostic.add_argument("--json", action="store_true")

    stage2d = commands.add_parser("stage2d")
    stage2d_commands = stage2d.add_subparsers(
        dest="stage2d_command",
        required=True,
    )
    stage2d_plan = stage2d_commands.add_parser("plan")
    stage2d_plan.add_argument("--config", type=Path, required=True)
    stage2d_plan.add_argument("--json", action="store_true")
    stage2d_shard = stage2d_commands.add_parser("run-shard")
    stage2d_shard.add_argument("--config", type=Path, required=True)
    stage2d_shard.add_argument("--shard", required=True)
    stage2d_shard.add_argument("--output-dir", type=Path, required=True)
    stage2d_shard.add_argument("--json", action="store_true")
    stage2d_reduce = stage2d_commands.add_parser("reduce")
    stage2d_reduce.add_argument("--config", type=Path, required=True)
    stage2d_reduce.add_argument("--input-root", type=Path, required=True)
    stage2d_reduce.add_argument("--output-dir", type=Path, required=True)
    stage2d_reduce.add_argument("--workers", type=int, default=1)
    stage2d_reduce.add_argument("--json", action="store_true")
    stage2d_replay = stage2d_commands.add_parser("verify-replay")
    stage2d_replay.add_argument("--primary", type=Path, required=True)
    stage2d_replay.add_argument("--replay", type=Path, required=True)
    stage2d_replay.add_argument("--output", type=Path, required=True)
    stage2d_replay.add_argument("--json", action="store_true")

    stage3 = commands.add_parser("stage3")
    stage3_commands_parser = stage3.add_subparsers(dest="stage3_command", required=True)
    stage3_doctor = stage3_commands_parser.add_parser("appserver-doctor")
    stage3_doctor.add_argument("--config", type=Path, required=True)
    stage3_doctor.add_argument("--auth-json", type=Path)
    stage3_doctor.add_argument("--json", action="store_true")
    stage3_freeze = stage3_commands_parser.add_parser("freeze")
    stage3_freeze.add_argument("--config", type=Path, required=True)
    stage3_freeze.add_argument("--json", action="store_true")
    stage3_generate = stage3_commands_parser.add_parser("generate")
    stage3_generate.add_argument("--config", type=Path, required=True)
    stage3_generate.add_argument("--auth-json", type=Path)
    stage3_generate.add_argument("--concurrency", type=int, default=8)
    stage3_generate.add_argument("--json", action="store_true")
    stage3_validate = stage3_commands_parser.add_parser("validate")
    stage3_validate.add_argument("run", type=Path)
    stage3_validate.add_argument("--json", action="store_true")
    stage3_revalidate = stage3_commands_parser.add_parser("revalidate")
    stage3_revalidate.add_argument("--config", type=Path, required=True)
    stage3_revalidate.add_argument("--run", type=Path, required=True)
    stage3_revalidate.add_argument("--json", action="store_true")
    stage3_evaluate = stage3_commands_parser.add_parser("evaluate")
    stage3_evaluate.add_argument("--config", type=Path, required=True)
    stage3_evaluate.add_argument("--run", type=Path, required=True)
    stage3_evaluate.add_argument("--workers", type=int, default=8)
    stage3_evaluate.add_argument("--json", action="store_true")
    stage3_replay = stage3_commands_parser.add_parser("verify-replay")
    stage3_replay.add_argument("primary", type=Path)
    stage3_replay.add_argument("replay", type=Path)
    stage3_replay.add_argument("--json", action="store_true")
    stage3_replay_generation = stage3_commands_parser.add_parser("replay-generation")
    stage3_replay_generation.add_argument("run", type=Path)
    stage3_replay_generation.add_argument("--json", action="store_true")

    stage4 = commands.add_parser("stage4")
    stage4_commands_parser = stage4.add_subparsers(dest="stage4_command", required=True)
    stage4_doctor = stage4_commands_parser.add_parser("doctor")
    stage4_doctor.add_argument("--config", type=Path, required=True)
    stage4_doctor.add_argument("--auth-json", type=Path)
    stage4_doctor.add_argument("--json", action="store_true")
    stage4_freeze = stage4_commands_parser.add_parser("freeze")
    stage4_freeze.add_argument("--config", type=Path, required=True)
    stage4_freeze.add_argument("--auth-json", type=Path)
    stage4_freeze.add_argument("--json", action="store_true")
    stage4_evolve = stage4_commands_parser.add_parser("evolve")
    stage4_evolve.add_argument("--config", type=Path, required=True)
    stage4_evolve.add_argument("--auth-json", type=Path)
    stage4_evolve.add_argument("--concurrency", type=int, default=8)
    stage4_evolve.add_argument("--json", action="store_true")
    stage4_resume = stage4_commands_parser.add_parser("resume")
    stage4_resume.add_argument("run", type=Path)
    stage4_resume.add_argument("--config", type=Path, default=Path("configs/stage4-search.toml"))
    stage4_resume.add_argument("--auth-json", type=Path)
    stage4_resume.add_argument("--json", action="store_true")
    stage4_archive = stage4_commands_parser.add_parser("archive")
    stage4_archive_commands = stage4_archive.add_subparsers(
        dest="stage4_archive_command",
        required=True,
    )
    for name in ("inspect", "reindex"):
        stage4_archive_command = stage4_archive_commands.add_parser(name)
        stage4_archive_command.add_argument("run", type=Path)
        stage4_archive_command.add_argument("--json", action="store_true")
    stage4_evaluate = stage4_commands_parser.add_parser("evaluate-candidate")
    stage4_evaluate.add_argument("run", type=Path)
    stage4_evaluate.add_argument("program_id")
    stage4_evaluate.add_argument(
        "--pass",
        dest="pass_name",
        choices=("primary", "replay"),
        required=True,
    )
    stage4_evaluate.add_argument("--workers", type=int, default=8)
    stage4_evaluate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage4-search.toml"),
    )
    stage4_evaluate.add_argument("--json", action="store_true")
    stage4_validation_freeze = stage4_commands_parser.add_parser("freeze-validation")
    stage4_validation_freeze.add_argument("run", type=Path)
    stage4_validation_freeze.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage4-search.toml"),
    )
    stage4_validation_freeze.add_argument("--json", action="store_true")
    stage4_validate = stage4_commands_parser.add_parser("validate")
    stage4_validate.add_argument("run", type=Path)
    stage4_validate.add_argument("--workers", type=int, default=8)
    stage4_validate.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage4-search.toml"),
    )
    stage4_validate.add_argument("--json", action="store_true")
    stage4_replay = stage4_commands_parser.add_parser("verify-replay")
    stage4_replay.add_argument("run", type=Path)
    stage4_replay.add_argument("--json", action="store_true")
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
        if args.command == "proposals" and args.proposals_command == "inspect":
            return _proposals_inspect(args.config, json_output=args.json)
        if args.command == "policy" and args.policy_command == "validate":
            return _policy_validate(args.policy, json_output=args.json)
        if args.command == "policy" and args.policy_command == "probe":
            return _policy_probe(args.policy, json_output=args.json)
        if args.command == "policy" and args.policy_command == "evaluate":
            return _policy_evaluate(
                args.policy,
                args.config,
                force_json=args.json,
            )
        if args.command == "policy" and args.policy_command == "compare":
            return _policy_compare(
                args.random_policy,
                args.structural_policy,
                args.config,
                json_output=args.json,
            )
        if args.command == "diagnostics":
            return _stage2c_diagnostic(
                args.diagnostics_command,
                args.config,
                json_output=args.json,
            )
        if args.command == "stage2d":
            return _stage2d(args)
        if args.command == "stage3":
            return _stage3(args)
        if args.command == "stage4":
            return _stage4(args)
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
