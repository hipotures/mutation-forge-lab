"""Synthetic and boundary checks for the independent Stage 6 package."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from mutation_forge.stage6_independent.config import (
    REQUIRED_ENTRY_COMMIT,
    build_manifest,
    load_config,
    validate_exact_project_entry,
    validate_manifest,
)
from mutation_forge.stage6_independent.metrics import (
    POLICY_IDS,
    bootstrap,
    parse_metrics_episodes,
    summarize,
)
from mutation_forge.stage6_independent.persistence import timing_stripped
from mutation_forge.stage6_independent.redteam import make_fixture, run_redteam
from mutation_forge.stage6_independent.runner import plan, run_shard, verify_replay


def test_frozen_config_and_manifest_are_exact() -> None:
    config = load_config()
    manifest = build_manifest(config)
    assert validate_manifest(manifest, config)["orders_disjoint"]
    assert manifest["episode_count"] == 768
    assert len(manifest["shards"]) == 12
    assert all(item["episode_count"] == 64 for item in manifest["shards"])


def test_exact_entry_guard_rejects_older_ancestor() -> None:
    repo = Path(__file__).parents[2]
    older_ancestor = "cc2f7b7254705d47fd4995a4b8a2bd45d545795c"
    assert subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            older_ancestor,
            REQUIRED_ENTRY_COMMIT,
        ],
        capture_output=True,
        timeout=30,
    ).returncode == 0
    with pytest.raises(ValueError, match="exactly match"):
        validate_exact_project_entry(repo, older_ancestor, current_commit=REQUIRED_ENTRY_COMMIT)


def test_provenance_amendment_hash_and_freeze_tag_are_immutable() -> None:
    repo = Path(__file__).parents[2]
    amendment_path = repo / "configs/stage6-verification-provenance-amendment-v1.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in amendment.items() if key != "amendment_sha256"}
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    assert amendment["amendment_sha256"] == hashlib.sha256(canonical).hexdigest()
    freeze_path = repo / "configs/stage6-verification-freeze-v1.json"
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == (
        "39eaaef22707a03c4db70b7f8ee75d39ee23da527ff165234fd4e7e372cd66a3"
    )
    assert subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "stage6-verification-frozen-v1^{}"],
        text=True,
    ).strip() == "6eaf9a446668751706239e6c1d8d10a26e32fde2"


def test_dry_run_shard_and_replay_are_deterministic() -> None:
    stage_plan = plan(
        orders=(20,),
        graph_seeds=(701,),
        relabeling_seeds=(7101,),
        policy_seeds=(7001,),
        strict_layout=False,
    )
    primary = run_shard(stage_plan, 0, dry_run=True)
    replay = run_shard(stage_plan, 0, dry_run=True)
    assert verify_replay(primary, replay)["exact"]


def test_metrics_use_exact_hierarchical_fractions() -> None:
    rows = []
    for order in (20, 24, 28):
        for graph_seed in (701,):
            for relabeling_seed in (7101, 7102):
                for policy_seed in (7001, 7002):
                    rows.append(
                        {
                            "episode_id": (
                                f"o{order}-g{graph_seed}-r{relabeling_seed}-p{policy_seed}"
                            ),
                            "order": order,
                            "graph_seed": graph_seed,
                            "relabeling_seed": relabeling_seed,
                            "policy_seed": policy_seed,
                            "policies": {
                                policy: {"normalized_best_so_far_curve": [0, 1]}
                                for policy in POLICY_IDS
                            },
                        }
                    )
    summary = summarize(parse_metrics_episodes(rows))
    draws = bootstrap(summary, samples=4, seed=2026080104)
    assert summary.policy_means[POLICY_IDS[0]].numerator == 1
    assert summary.policy_means[POLICY_IDS[0]].denominator == 2
    assert set(summary.relative_improvements) == {
        "C_vs_stage3",
        "C_vs_random",
        "C_vs_structural",
    }
    assert all(value == 0 for value in summary.relative_improvements.values())
    assert draws.samples == 4


def test_timing_projection_does_not_strip_scientific_fields() -> None:
    value = {"timing_ns": 1, "selected_score_calls": 2, "scientific_timing_ns": 3}
    projected = timing_stripped(value)
    assert projected == {"selected_score_calls": 2, "scientific_timing_ns": 3}


def test_redteam_fixture_is_complete() -> None:
    assert run_redteam(evidence=make_fixture())["status"] == "passed"


def test_independent_package_has_no_stage5_imports() -> None:
    root = Path(__file__).parents[2] / "src" / "mutation_forge" / "stage6_independent"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name == "mutation_forge.stage5"
                or name.startswith("mutation_forge.stage5.")
                for name in names
            ), path
