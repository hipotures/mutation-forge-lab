"""Stage 4E frozen provider-free confirmation orchestration."""
# The result schema deliberately keeps long, human-readable gate labels.
# ruff: noqa: E501

from __future__ import annotations

import json
import statistics
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from mutation_forge.stage3.manifest import canonical_bytes

from .stage4e_config import (
    CHAMPION_ID,
    COMPARATOR_ID,
    HEG_COMMIT,
    START_COMMIT,
    Stage4EConfig,
    load_manifest,
    load_stage4e_config,
    sha256_bytes,
    sha256_value,
    write_manifest,
)
from .stage4e_execution import (
    execute_stage4e_confirmation,
    verify_stage4e_pass,
    verify_stage4e_replay,
)
from .stage4e_metrics import (
    BootstrapSummary,
    PairedAreaEpisode,
    PairedAreaSummary,
    bootstrap_paired_theta,
    fraction_text,
    summarize_paired_areas,
    terminal_gate_checks,
)

FREEZE_PATH_NAME = "stage4e-confirmation-freeze-v1.json"
TERMINAL_SCHEMA = "stage4e.confirmation.terminal.v1"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(dict(value)) + b"\n")
    temporary.replace(path)


def _fraction_payload(value: Fraction) -> dict[str, float | str]:
    return {"value": float(value), "fraction": fraction_text(value)}


def _median(values: list[Fraction]) -> Fraction:
    if not values:
        raise ValueError("median requires values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _policy_source_hashes(config: Stage4EConfig) -> dict[str, str]:
    return {
        config.champion_id: config.champion_source_sha256,
        config.comparator_id: config.comparator_source_sha256,
    }


def _freeze_path(config: Stage4EConfig) -> Path:
    return config.source_path.parent / FREEZE_PATH_NAME


def _metric_hash() -> str:
    return sha256_bytes(Path(__file__).with_name("stage4e_metrics.py").read_bytes())


def freeze_stage4e(config_path: str | Path = "configs/stage4e-confirmation.toml") -> dict[str, Any]:
    """Validate and persist the preregistration freeze before any outcomes."""
    config = load_stage4e_config(config_path)
    manifest = write_manifest(config)
    freeze_path = _freeze_path(config)
    if config.run_root.exists() and any(config.run_root.rglob("*-summary.json")):
        raise RuntimeError("Stage 4E outcomes already exist; freeze cannot be amended")
    implementation_commit = _git(config.project_repo, "rev-parse", "HEAD")
    freeze = {
        "schema_version": "stage4e.confirmation.freeze.v1",
        "stage4e_results_observed": False,
        "implementation_commit": implementation_commit,
        "start_commit": START_COMMIT,
        "branch": _git(config.project_repo, "branch", "--show-current"),
        "config_path": str(config.source_path),
        "config_sha256": sha256_bytes(config.source_path.read_bytes()),
        "config_stable_hash": config.stable_hash(),
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "metric_path": str(Path(__file__).with_name("stage4e_metrics.py")),
        "metric_sha256": _metric_hash(),
        "policy_ids": [config.champion_id, config.comparator_id],
        "policy_source_sha256": _policy_source_hashes(config),
        "policy_ast_sha256": {
            config.champion_id: config.champion_ast_sha256,
            config.comparator_id: config.comparator_ast_sha256,
        },
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "seed": config.bootstrap_seed,
            "confidence_level": config.confidence_level,
            "percentile_rule": "linear_interpolation_at_p_times_n_minus_1",
        },
        "gate": {
            "relative_improvement_at_least": config.relative_improvement_threshold,
            "bootstrap_lower_bound_strictly_positive": True,
            "order_effects_nonnegative": True,
        },
        "shards": {
            "count": config.experiment.shard_count,
            "episodes_per_shard": config.experiment.episodes_per_shard,
            "workers": config.resources.workers,
            "reserved_physical_cores": config.resources.reserved_physical_cores,
            "thread_count": config.resources.thread_count,
        },
        "heg_commit": HEG_COMMIT,
        "provider_calls_allowed": False,
    }
    freeze["freeze_sha256"] = sha256_value(freeze)
    _write_json(freeze_path, freeze)
    return {**freeze, "status": "completed", "freeze_path": str(freeze_path)}


def _load_and_verify_freeze(config: Stage4EConfig) -> dict[str, Any]:
    freeze = _read_json(_freeze_path(config))
    if freeze.get("schema_version") != "stage4e.confirmation.freeze.v1":
        raise ValueError("unexpected Stage 4E freeze schema")
    if freeze.get("stage4e_results_observed") is not False:
        raise ValueError("Stage 4E freeze is already marked as observed")
    current_commit = _git(config.project_repo, "rev-parse", "HEAD")
    frozen_commit = freeze.get("implementation_commit")
    if not isinstance(frozen_commit, str):
        raise ValueError("freeze implementation commit is missing")
    descendant = subprocess.run(
        [
            "git",
            "-C",
            str(config.project_repo),
            "merge-base",
            "--is-ancestor",
            frozen_commit,
            current_commit,
        ],
        capture_output=True,
        timeout=30,
    )
    if descendant.returncode != 0:
        raise ValueError("current implementation does not descend from the frozen commit")
    if freeze.get("manifest_sha256") != json.loads(config.manifest_path.read_text())["manifest_sha256"]:
        raise ValueError("current manifest does not match the preregistration freeze")
    if freeze.get("metric_sha256") != _metric_hash():
        raise ValueError("metric implementation changed after preregistration freeze")
    return freeze


def _records_by_id(records: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {str(record.get("episode_id")): record for record in records}
    if len(result) != len(records):
        raise ValueError("duplicate Stage 4E episode record")
    return result


def _paired_episodes(
    records: list[Mapping[str, Any]], config: Stage4EConfig
) -> list[PairedAreaEpisode]:
    result: list[PairedAreaEpisode] = []
    for record in records:
        policies = record.get("policies")
        if not isinstance(policies, Mapping):
            raise ValueError("Stage 4E record is missing policy curves")
        candidate = policies.get(config.champion_id)
        comparator = policies.get(config.comparator_id)
        if not isinstance(candidate, Mapping) or not isinstance(comparator, Mapping):
            raise ValueError("Stage 4E record policy roster mismatch")
        candidate_curve = candidate.get("normalized_best_so_far_curve")
        comparator_curve = comparator.get("normalized_best_so_far_curve")
        if not isinstance(candidate_curve, list | tuple) or not isinstance(comparator_curve, list | tuple):
            raise ValueError("Stage 4E record is missing normalized curves")
        result.append(
            PairedAreaEpisode(
                order=int(record["order"]),
                graph_seed=int(record["graph_seed"]),
                episode_id=str(record["episode_id"]),
                candidate_curve=cast(list[float], candidate_curve),
                comparator_curve=cast(list[float], comparator_curve),
            )
        )
    return result


def _secondary_metrics(records: list[Mapping[str, Any]], config: Stage4EConfig) -> dict[str, Any]:
    policy_aucs: dict[str, list[Fraction]] = {config.champion_id: [], config.comparator_id: []}
    paired_deltas: list[Fraction] = []
    first_steps: dict[str, list[int]] = {config.champion_id: [], config.comparator_id: []}
    witnesses: dict[str, list[int]] = {config.champion_id: [], config.comparator_id: []}
    for record in records:
        policies = cast(Mapping[str, Mapping[str, Any]], record["policies"])
        for policy in policy_aucs:
            summary = policies[policy]
            policy_aucs[policy].append(Fraction(str(summary["auc"])))
            witnesses[policy].append(int(summary.get("best_total_witnesses", 0)))
            step = summary.get("first_improvement_step")
            if isinstance(step, int):
                first_steps[policy].append(step)
        candidate_curve = cast(list[float], policies[config.champion_id]["normalized_best_so_far_curve"])
        comparator_curve = cast(list[float], policies[config.comparator_id]["normalized_best_so_far_curve"])
        paired_deltas.append(
            sum(
                (Fraction(str(candidate)) - Fraction(str(baseline)) for candidate, baseline in zip(candidate_curve, comparator_curve, strict=True)),
                Fraction(0),
            )
            / len(candidate_curve)
        )
    medians = {policy: _median(values) for policy, values in policy_aucs.items()}
    return {
        "normalized_auc_medians": {policy: _fraction_payload(value) for policy, value in medians.items()},
        "difference_of_separate_medians": _fraction_payload(medians[config.champion_id] - medians[config.comparator_id]),
        "median_paired_episode_delta": _fraction_payload(_median(paired_deltas)),
        "witness_medians": {policy: statistics.median(values) for policy, values in witnesses.items()},
        "first_improvement_step_medians": {policy: statistics.median(values) if values else None for policy, values in first_steps.items()},
        "evaluations_to_first_improvement_medians": {
            policy: statistics.median([value + 1 for value in first_steps[policy]]) if first_steps[policy] else None
            for policy in first_steps
        },
    }


def _summary_payload(summary: PairedAreaSummary) -> dict[str, Any]:
    return {
        "estimand": "transition-aware paired area; graph mean, equal-order mean",
        "episode_count": len(summary.episodes),
        "graph_count": len(summary.graphs),
        "order_count": len(summary.orders),
        "theta": _fraction_payload(summary.theta),
        "mu_B": _fraction_payload(summary.mu_B),
        "relative_improvement": _fraction_payload(summary.relative_improvement),
        "episodes": [
            {
                "order": item.order,
                "graph_seed": item.graph_seed,
                "episode_id": item.episode_id,
                "candidate_area": _fraction_payload(item.candidate_area),
                "comparator_area": _fraction_payload(item.comparator_area),
                "delta": _fraction_payload(item.delta),
            }
            for item in summary.episodes
        ],
        "graphs": [
            {
                "order": item.order,
                "graph_seed": item.graph_seed,
                "episode_count": item.episode_count,
                "candidate_mean": _fraction_payload(item.candidate_mean),
                "comparator_mean": _fraction_payload(item.comparator_mean),
                "delta_mean": _fraction_payload(item.delta_mean),
            }
            for item in summary.graphs
        ],
        "orders": [
            {
                "order": item.order,
                "graph_count": item.graph_count,
                "episode_count": item.episode_count,
                "candidate_mean": _fraction_payload(item.candidate_mean),
                "comparator_mean": _fraction_payload(item.comparator_mean),
                "delta_mean": _fraction_payload(item.delta_mean),
            }
            for item in summary.orders
        ],
        "sign_counts": {
            "episode": {
                "negative": sum(item.delta < 0 for item in summary.episodes),
                "zero": sum(item.delta == 0 for item in summary.episodes),
                "positive": sum(item.delta > 0 for item in summary.episodes),
            },
            "graph": {
                "negative": sum(item.delta_mean < 0 for item in summary.graphs),
                "zero": sum(item.delta_mean == 0 for item in summary.graphs),
                "positive": sum(item.delta_mean > 0 for item in summary.graphs),
            },
        },
    }


def _bootstrap_payload(bootstrap: BootstrapSummary) -> dict[str, Any]:
    payload = bootstrap.as_dict()
    payload["observed_theta"] = _fraction_payload(bootstrap.observed_theta)
    payload["interval"] = [_fraction_payload(value) for value in bootstrap.interval]
    return cast(dict[str, Any], payload)


def _gate(
    primary: Mapping[str, Any],
    replay: Mapping[str, Any],
    replay_check: Mapping[str, Any],
    summary: PairedAreaSummary,
    bootstrap: BootstrapSummary,
    config: Stage4EConfig,
) -> dict[str, Any]:
    records = cast(list[Mapping[str, Any]], primary["records"])
    gate: dict[str, bool] = {
        "frozen_policy_identities_exact": primary.get("policy_source_sha256") == _policy_source_hashes(config),
        "all_primary_episodes_complete": len(records) == config.experiment.episode_count and all(record.get("terminal_status") == "completed" for record in records),
        "all_replay_episodes_complete": int(replay.get("record_count", 0)) == config.experiment.episode_count,
        "primary_replay_exact": bool(replay_check.get("exact")),
        "graph_validity_100_percent": all(int(record.get("invalid_graphs", 0)) == 0 for record in records),
        "worker_failures_crashes_timeouts_protocol_zero": all(int(record.get("policy_failures", 0)) == 0 for record in records),
        "selected_plan_only_and_oracle_zero": all(int(record.get("oracle_score_calls", 0)) == 0 for record in records),
        "model_app_server_calls_zero": all(int(record.get("model_calls", 0)) == 0 and int(record.get("app_server_calls", 0)) == 0 for record in records),
    }
    gate.update(terminal_gate_checks(summary, bootstrap, minimum_relative_improvement=config.relative_improvement_threshold, minimum_bootstrap_lower_bound=0))
    gate["order_effects_nonnegative"] = all(item.delta_mean >= 0 for item in summary.orders)
    return {"checks": gate, "all_pass": all(gate.values()), "decision": "GO_TO_STAGE_5" if all(gate.values()) else "NO_GO"}


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    summary = cast(Mapping[str, Any], result["paired_area"])
    bootstrap = cast(Mapping[str, Any], result["bootstrap"])
    gate = cast(Mapping[str, Any], result["terminal_gate"])
    theta = cast(Mapping[str, Any], summary["theta"])
    relative = cast(Mapping[str, Any], summary["relative_improvement"])
    interval = cast(list[Mapping[str, Any]], bootstrap["interval"])
    report = f"""# Stage 4E Confirmation Report

Decision: **{result['decision']}**

This provider-free confirmation evaluated the byte-identical frozen Stage 4R
champion against the frozen Stage 3 comparator on 1,536 unseen paired episodes.
The primary estimand is the transition-aware paired area: per-step normalized
best-so-far differences averaged within each episode, then graph means, equal
graph-weighted order means, and finally equal order weighting.

## Frozen design

- Orders: 10, 12, 16; graph seeds 501–516 per order; policy seeds 5001–5032.
- Horizon: 32; 24 shards × 64 paired episodes; eight workers with eight
  reserved physical cores and one numerical thread.
- Policies: `{CHAMPION_ID}` and `{COMPARATOR_ID}`; no model, App Server, oracle,
  Stage 5, or HEG work.
- Bootstrap: 10,000 graph-then-policy draws, seed 2026080102, 95% linear
  percentile interval.

## Primary result

- theta: `{theta['fraction']}` ({theta['value']}).
- comparator hierarchical mean AUC (mu_B): `{summary['mu_B']['fraction']}` ({summary['mu_B']['value']}).
- relative improvement: `{relative['fraction']}` ({relative['value']}).
- bootstrap interval: `[{interval[0]['fraction']}, {interval[1]['fraction']}]`.
- bootstrap sign counts: {bootstrap['sign_counts']}.

## Gate

```json
{json.dumps(gate, sort_keys=True, indent=2)}
```

Primary and replay rows, shard hashes, reductions, metric summaries, bootstrap
support, and the terminal gate were compared after timing-only fields were
removed. The historical Stage 4R `NO_GO` and Stage 4D diagnosis remain
unchanged. This issue does not authorize or start Stage 5.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def confirm_stage4e(config_path: str | Path = "configs/stage4e-confirmation.toml") -> dict[str, Any]:
    """Run exactly one primary and one deterministic replay, then gate them."""
    try:
        config = load_stage4e_config(config_path)
        _load_and_verify_freeze(config)
        manifest = load_manifest(config)
        policies = {
            config.champion_id: config.champion_source_path.read_text(encoding="utf-8"),
            config.comparator_id: config.comparator_source_path.read_text(encoding="utf-8"),
        }
        execution = execute_stage4e_confirmation(
            config,
            manifest,
            policies,
            config.run_root,
            workers=config.resources.workers,
            resume=True,
        )
        primary = cast(dict[str, Any], execution["primary"])
        replay = cast(dict[str, Any], execution["replay"])
        primary_verify = verify_stage4e_pass(primary, manifest)
        replay_verify = verify_stage4e_pass(replay, manifest)
        replay_check = verify_stage4e_replay(primary, replay)
        if not primary_verify.get("exact") or not replay_verify.get("exact") or not replay_check.get("exact"):
            raise RuntimeError("primary/replay artifact identity mismatch")
        primary_records = cast(list[Mapping[str, Any]], primary["records"])
        replay_records = cast(list[Mapping[str, Any]], replay["records"])
        if [dict(record) for record in primary_records] != [dict(record) for record in replay_records]:
            raise RuntimeError("timing-stripped primary/replay episode rows differ")
        summary = summarize_paired_areas(_paired_episodes(primary_records, config))
        replay_summary = summarize_paired_areas(_paired_episodes(replay_records, config))
        if _summary_payload(summary) != _summary_payload(replay_summary):
            raise RuntimeError("primary/replay paired-area reductions differ")
        bootstrap = bootstrap_paired_theta(
            summary,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed,
            confidence_level=config.confidence_level,
        )
        secondary = _secondary_metrics(primary_records, config)
        terminal_gate = _gate(primary, replay, replay_check, summary, bootstrap, config)
        result: dict[str, Any] = {
            "schema_version": TERMINAL_SCHEMA,
            "status": "completed",
            "decision": terminal_gate["decision"],
            "historical_stage4r_decision": "NO_GO",
            "stage5_started": False,
            "heg_modified": False,
            "provider_calls": 0,
            "manifest_sha256": manifest["manifest_sha256"],
            "config_sha256": config.stable_hash(),
            "freeze": _read_json(_freeze_path(config)),
            "primary": {key: value for key, value in primary.items() if key != "records"},
            "replay": {key: value for key, value in replay.items() if key != "records"},
            "primary_verification": primary_verify,
            "replay_verification": replay_verify,
            "replay_identity": replay_check,
            "paired_area": _summary_payload(summary),
            "bootstrap": _bootstrap_payload(bootstrap),
            "secondary_metrics": secondary,
            "terminal_gate": terminal_gate,
        }
        root = config.run_root
        _write_json(root / "paired-area-summary.json", cast(Mapping[str, Any], result["paired_area"]))
        _write_json(root / "bootstrap-support.json", cast(Mapping[str, Any], result["bootstrap"]))
        _write_json(root / "cluster-summary.json", {"graphs": result["paired_area"]["graphs"], "orders": result["paired_area"]["orders"], "sign_counts": result["paired_area"]["sign_counts"]})
        _write_json(root / "replay-verification.json", replay_check)
        _write_json(root / "terminal-gate.json", terminal_gate)
        _write_json(root / "stage4e-summary.json", result)
        _write_report(Path("docs/reports/STAGE4E_CONFIRMATION_REPORT.md"), result)
        return result
    except Exception as error:
        return {
            "schema_version": TERMINAL_SCHEMA,
            "status": "infrastructure_failure",
            "decision": "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
            "error": f"{type(error).__name__}: {error}",
            "stage5_started": False,
            "heg_modified": False,
            "provider_calls": 0,
        }


__all__ = ["confirm_stage4e", "freeze_stage4e"]
