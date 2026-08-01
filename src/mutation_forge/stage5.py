"""Stage 5 held-out generalization freeze, execution, and terminal reduction."""
# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mutation_forge.stage3.manifest import canonical_bytes

from .stage5_config import (
    HEG_COMMIT,
    POLICY_IDS,
    STAGE5_FREEZE_VERSION,
    START_COMMIT,
    Stage5Config,
    load_manifest,
    load_stage5_config,
    sha256_bytes,
    sha256_value,
    validate_manifest,
    write_manifest,
)
from .stage5_execution import (
    execute_stage5_pass,
    verify_stage5_pass,
    verify_stage5_replay,
)
from .stage5_metrics import (
    EFFECTS,
    PolicyAreaEpisode,
    bootstrap_stage5,
    curve_area,
    fraction_payload,
    gate_checks,
    summarize_stage5,
)
from .stage5_relabel import relabel_contract_digest

FREEZE_PATH_NAME = "stage5-generalization-freeze-v1.json"
TERMINAL_PATH_NAME = "stage5-terminal.json"
SUMMARY_PATH_NAME = "stage5-summary.json"
REPORT_PATH = Path("docs/reports/STAGE5_GENERALIZATION_REPORT.md")


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


def _module_hash(name: str) -> str:
    return sha256_bytes(Path(__file__).with_name(name).read_bytes())


def _freeze_path(config: Stage5Config) -> Path:
    return config.source_path.parent / FREEZE_PATH_NAME


def _policy_payload(config: Stage5Config) -> dict[str, Any]:
    return {
        policy_id: {
            "source_path": str(config.policy_paths[policy_id]),
            "source_sha256": config.policy_source_hashes[policy_id],
            "normalized_ast_sha256": config.policy_ast_hashes[policy_id],
            "behavior_signature_sha256": config.policy_behavior_hashes[policy_id],
        }
        for policy_id in POLICY_IDS
    }


def freeze_stage5(config_path: str | Path = "configs/stage5-generalization.toml") -> dict[str, Any]:
    """Create the immutable Stage 5 freeze before any official episode."""
    config = load_stage5_config(config_path)
    manifest = write_manifest(config)
    if config.run_root.exists() and any(config.run_root.rglob("*-summary.json")):
        raise RuntimeError("Stage 5 outcomes already exist; freeze cannot be amended")
    implementation_commit = _git(config.project_repo, "rev-parse", "HEAD")
    disjointness = validate_manifest(manifest, config)
    freeze: dict[str, Any] = {
        "schema_version": STAGE5_FREEZE_VERSION,
        "stage5_results_observed": False,
        "stage6_started": False,
        "implementation_commit": implementation_commit,
        "start_commit": START_COMMIT,
        "branch": _git(config.project_repo, "branch", "--show-current"),
        "config_path": str(config.source_path),
        "config_sha256": sha256_bytes(config.source_path.read_bytes()),
        "config_stable_hash": config.stable_hash(),
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_disjointness": disjointness,
        "policy_ids": list(POLICY_IDS),
        "policies": _policy_payload(config),
        "relabeling": {
            "algorithm": config.relabel_algorithm,
            "module_sha256": _module_hash("stage5_relabel.py"),
            "contract_sha256": relabel_contract_digest(
                config.experiment.orders,
                config.experiment.graph_seeds,
                config.experiment.relabeling_seeds,
            ),
        },
        "execution": {
            "module_sha256": _module_hash("stage5_execution.py"),
            "schema_version": "stage5.generalization.execution.v1",
        },
        "metrics": {
            "module_sha256": _module_hash("stage5_metrics.py"),
            "estimand": "stage5_hierarchical_paired_area_theta",
            "hierarchy": "16 policy seeds -> 2 relabelings -> 16 graphs/order -> equal orders",
        },
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "seed": config.bootstrap_seed,
            "confidence_level": config.confidence_level,
            "percentile_rule": "linear_interpolation_at_p_times_n_minus_1",
        },
        "gates": {
            "champion_stage3_relative_improvement_at_least": config.champion_stage3_threshold,
            "champion_random_relative_improvement_at_least": config.champion_random_threshold,
            "structural_retention_at_least": config.structural_retention_threshold,
            "bootstrap_lower_bound_strictly_positive": True,
            "order_effects_nonnegative": True,
            "six_order_relabel_strata_nonnegative": True,
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
    _write_json(_freeze_path(config), freeze)
    # Keep an identical run-local copy for evidence; the tracked config freeze
    # remains the authoritative preregistration object.
    _write_json(config.run_root / FREEZE_PATH_NAME, freeze)
    return {**freeze, "status": "completed", "freeze_path": str(_freeze_path(config))}


def _load_and_verify_freeze(config: Stage5Config) -> dict[str, Any]:
    freeze = _read_json(_freeze_path(config))
    if freeze.get("schema_version") != STAGE5_FREEZE_VERSION:
        raise ValueError("unexpected Stage 5 freeze schema")
    if freeze.get("stage5_results_observed") is not False or freeze.get("stage6_started") is not False:
        raise ValueError("Stage 5 freeze is already observed or Stage 6 was started")
    freeze_hash = freeze.get("freeze_sha256")
    if not isinstance(freeze_hash, str) or sha256_value({key: value for key, value in freeze.items() if key != "freeze_sha256"}) != freeze_hash:
        raise ValueError("Stage 5 freeze hash mismatch")
    current = _git(config.project_repo, "rev-parse", "HEAD")
    if _git(config.project_repo, "status", "--short"):
        raise ValueError("Stage 5 official execution requires a clean project repository")
    if _git(config.heg_repo, "rev-parse", "HEAD") != HEG_COMMIT or _git(config.heg_repo, "status", "--short"):
        raise ValueError("Stage 5 official execution requires the clean pinned HEG repository")
    frozen = freeze.get("implementation_commit")
    if not isinstance(frozen, str):
        raise ValueError("Stage 5 freeze implementation commit is missing")
    descendant = subprocess.run(
        ["git", "-C", str(config.project_repo), "merge-base", "--is-ancestor", frozen, current],
        capture_output=True,
        timeout=30,
    )
    if descendant.returncode != 0:
        raise ValueError("current implementation does not descend from the frozen commit")
    manifest = load_manifest(config)
    if freeze.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Stage 5 manifest differs from the preregistration freeze")
    if freeze.get("config_sha256") != sha256_bytes(config.source_path.read_bytes()):
        raise ValueError("Stage 5 config bytes differ from the preregistration freeze")
    if freeze.get("config_stable_hash") != config.stable_hash():
        raise ValueError("Stage 5 stable config hash differs from the preregistration freeze")
    metrics = freeze.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("module_sha256") != _module_hash("stage5_metrics.py"):
        raise ValueError("Stage 5 metric implementation changed after freeze")
    relabel = freeze.get("relabeling")
    if not isinstance(relabel, Mapping) or relabel.get("module_sha256") != _module_hash("stage5_relabel.py"):
        raise ValueError("Stage 5 relabel implementation changed after freeze")
    if not isinstance(relabel, Mapping) or relabel.get("contract_sha256") != relabel_contract_digest(
        config.experiment.orders,
        config.experiment.graph_seeds,
        config.experiment.relabeling_seeds,
    ):
        raise ValueError("Stage 5 relabel contract differs from the preregistration freeze")
    execution = freeze.get("execution")
    if not isinstance(execution, Mapping) or execution.get("module_sha256") != _module_hash("stage5_execution.py"):
        raise ValueError("Stage 5 execution implementation changed after freeze")
    if freeze.get("policy_ids") != list(POLICY_IDS) or freeze.get("policies") != _policy_payload(config):
        raise ValueError("Stage 5 policy provenance differs from the preregistration freeze")
    return freeze


def _records_to_episodes(records: list[Mapping[str, Any]]) -> list[PolicyAreaEpisode]:
    episodes: list[PolicyAreaEpisode] = []
    for row in records:
        policies = row.get("policies")
        if not isinstance(policies, Mapping):
            raise ValueError("Stage 5 compact record has no policy rows")
        areas = {
            policy_id: curve_area(cast(Sequence[float], cast(Mapping[str, Any], policies[policy_id])["normalized_best_so_far_curve"]), f"{policy_id} curve")
            for policy_id in POLICY_IDS
        }
        episodes.append(
            PolicyAreaEpisode(
                order=int(row["order"]),
                graph_seed=int(row["graph_seed"]),
                relabeling_seed=int(row["relabeling_seed"]),
                policy_seed=int(row["policy_seed"]),
                episode_id=str(row["episode_id"]),
                areas=areas,
            )
        )
    return episodes


def _scientific_gates(
    config: Stage5Config,
    freeze: Mapping[str, Any],
    manifest: Mapping[str, Any],
    primary: Mapping[str, Any],
    replay: Mapping[str, Any],
    replay_check: Mapping[str, Any],
    summary: Any,
    bootstrap: Any,
    *,
    preservation_verified: bool,
) -> dict[str, bool]:
    records = cast(list[Mapping[str, Any]], primary.get("records", []))
    metric_checks = gate_checks(
        summary,
        bootstrap,
        champion_stage3_threshold=config.champion_stage3_threshold,
        champion_random_threshold=config.champion_random_threshold,
        structural_retention_threshold=config.structural_retention_threshold,
    )
    primary_complete = primary.get("status") == "completed" and int(primary.get("record_count", -1)) == 1536
    replay_complete = replay.get("status") == "completed" and int(replay.get("record_count", -1)) == 1536
    validity = all(int(row.get("invalid_graphs", 1)) == 0 for row in records)
    ranker_flags_clear = all(
        not bool(trace.get("ranker_flags", {}).get(flag, False))
        for row in records
        for step in cast(Sequence[Mapping[str, Any]], row.get("steps", []))
        for trace in cast(Mapping[str, Mapping[str, Any]], step.get("policies", {})).values()
        for flag in ("exception", "timeout", "crash", "protocol")
    )
    failures = all(int(row.get("policy_failures", 1)) == 0 for row in records) and ranker_flags_clear
    complete_curves = all(
        all(
            isinstance(policy.get("normalized_best_so_far_curve"), Sequence)
            and len(cast(Sequence[Any], policy["normalized_best_so_far_curve"])) == config.experiment.horizon
            for policy in cast(Mapping[str, Mapping[str, Any]], row.get("policies", {})).values()
        )
        for row in records
    )
    selected_only = all(
        int(row.get("oracle_score_calls", 1)) == 0
        and int(row.get("selected_score_calls", -1)) == int(row.get("horizon", -2)) * 4
        and int(row.get("evaluation_count", -1)) == int(row.get("horizon", -2)) * 4
        and complete_curves
        for row in records
    )
    provider_free = all(all(int(row.get(counter, 1)) == 0 for counter in ("model_calls", "app_server_calls", "runtime_network_calls")) for row in records)
    provenance = (
        freeze.get("policy_ids") == list(POLICY_IDS)
        and freeze.get("manifest_sha256") == manifest.get("manifest_sha256")
        and freeze.get("stage5_results_observed") is False
    )
    artifact = preservation_verified and bool(primary.get("canonical_reduction_sha256")) and bool(replay.get("canonical_reduction_sha256"))
    return {
        "1_policy_provenance_exact": provenance,
        "2_manifest_complete_and_disjoint": primary_complete and replay_complete and bool(freeze.get("manifest_disjointness", {}).get("orders_disjoint", False)) and bool(freeze.get("manifest_disjointness", {}).get("base_identities_disjoint", False)) and bool(freeze.get("manifest_disjointness", {}).get("complete_identities_disjoint", False)),
        "3_primary_and_replay_complete_equal_budgets": primary_complete and replay_complete,
        "4_timing_stripped_replay_identity_exact": bool(replay_check.get("exact", False)),
        "5_graph_validity_100_percent": validity,
        "6_zero_worker_failures_crashes_timeouts_protocol_violations": failures,
        "7_selected_plan_only_zero_oracle": selected_only,
        "8_zero_model_app_server_runtime_network_calls": provider_free,
        "9_C_vs_stage3_relative_improvement_ge_2_percent": metric_checks["relative_improvement_C_vs_stage3_at_least_threshold"],
        "10_C_vs_stage3_bootstrap_lower_bound_positive": metric_checks["bootstrap_C_vs_stage3_lower_bound_positive"],
        "11_C_vs_stage3_nonnegative_each_order": metric_checks["C_vs_stage3_nonnegative_each_order"],
        "12_C_vs_stage3_nonnegative_all_six_order_relabel_strata": metric_checks["C_vs_stage3_nonnegative_all_six_order_relabel_strata"],
        "13_C_vs_random_threshold_and_lower_bound": metric_checks["relative_improvement_C_vs_random_at_least_threshold"] and metric_checks["bootstrap_C_vs_random_lower_bound_positive"],
        "14_structural_retention_ge_99_percent": metric_checks["structural_retention_at_least_threshold"],
        "15_artifact_provenance_preservation_repository_verified": artifact,
    }


def _decision(gates: Mapping[str, bool], *, infrastructure_failure: bool = False) -> str:
    if infrastructure_failure:
        return "INCONCLUSIVE_INFRASTRUCTURE_FAILURE"
    return "GO_TO_STAGE_6" if all(gates.values()) else "NO_GO"


def generalize_stage5(config_path: str | Path = "configs/stage5-generalization.toml", *, preservation_verified: bool = False) -> dict[str, Any]:
    """Run the canary, complete primary, and exactly one deterministic replay."""
    config = load_stage5_config(config_path)
    freeze = _load_and_verify_freeze(config)
    manifest = load_manifest(config)
    policies = {policy_id: config.policy_paths[policy_id].read_text(encoding="utf-8") for policy_id in POLICY_IDS}
    root = config.run_root
    # The official canary is a distinct first invocation and is retained in the
    # primary state; a completed valid shard is never rerun.
    execute_stage5_pass(config, manifest, policies, root / "primary", "primary", workers=1, shard_indices=[0], resume=True)
    primary = execute_stage5_pass(config, manifest, policies, root / "primary", "primary", workers=config.resources.workers, resume=True)
    primary_verify = verify_stage5_pass(root / "primary" / next(root.joinpath("primary").glob("*-primary-summary.json")).name, manifest)
    replay = execute_stage5_pass(config, manifest, policies, root / "replay", "replay", workers=config.resources.workers, resume=True)
    replay_verify = verify_stage5_pass(root / "replay" / next(root.joinpath("replay").glob("*-replay-summary.json")).name, manifest)
    replay_check = verify_stage5_replay(
        root / "primary" / next(root.joinpath("primary").glob("*-primary-summary.json")).name,
        root / "replay" / next(root.joinpath("replay").glob("*-replay-summary.json")).name,
    )
    primary_records = cast(list[Mapping[str, Any]], primary["records"])
    summary = summarize_stage5(_records_to_episodes(primary_records), POLICY_IDS)
    bootstrap = bootstrap_stage5(summary, samples=config.bootstrap_samples, seed=config.bootstrap_seed, confidence_level=config.confidence_level)
    gates = _scientific_gates(config, freeze, manifest, primary, replay, replay_check, summary, bootstrap, preservation_verified=preservation_verified)
    decision = _decision(gates)
    result: dict[str, Any] = {
        "schema_version": "stage5.generalization.result.v1",
        "stage5_results_observed": True,
        "decision": decision,
        "freeze_sha256": freeze["freeze_sha256"],
        "config_sha256": freeze["config_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "policy_sha256": sha256_value(freeze["policies"]),
        "metric_sha256": str(cast(Mapping[str, Any], freeze["metrics"])["module_sha256"]),
        "primary": {
            "summary_sha256": sha256_bytes((root / "primary" / next(root.joinpath("primary").glob("*-primary-summary.json")).name).read_bytes()),
            "verification": primary_verify,
            "canonical_reduction_sha256": primary.get("canonical_reduction_sha256"),
            "timing_stripped_reduction_sha256": primary.get("timing_stripped_reduction_sha256"),
        },
        "replay": {
            "summary_sha256": sha256_bytes((root / "replay" / next(root.joinpath("replay").glob("*-replay-summary.json")).name).read_bytes()),
            "verification": replay_verify,
            "canonical_reduction_sha256": replay.get("canonical_reduction_sha256"),
            "timing_stripped_reduction_sha256": replay.get("timing_stripped_reduction_sha256"),
        },
        "replay_verification": replay_check,
        "metrics": {
            "policy_means": {policy: fraction_payload(value) for policy, value in summary.policy_means.items()},
            "effects": {
                effect: {
                    "theta": fraction_payload(summary.effects[effect].theta),
                    "relative_improvement": fraction_payload(summary.relative_improvements[effect]),
                    "order_deltas": {str(order): fraction_payload(value) for order, value in summary.effects[effect].order_deltas.items()},
                    "relabel_deltas": {f"{order}-{graph}-{relabel}": fraction_payload(value) for (order, graph, relabel), value in summary.effects[effect].relabel_deltas.items()},
                    "stratum_deltas": {f"{order}-{relabel}": fraction_payload(value) for (order, relabel), value in summary.effects[effect].stratum_deltas.items()},
                    "sign_counts": summary.effects[effect].sign_counts,
                }
                for effect in EFFECTS
            },
            "structural_retention": fraction_payload(summary.structural_retention),
            "bootstrap": bootstrap.as_dict(),
        },
        "gates": gates,
        "gate_count": len(gates),
        "bootstrap_sha256": sha256_value(bootstrap.as_dict()),
        "gate_sha256": sha256_value(gates),
        "primary_hash": replay_check.get("primary_sha256"),
        "replay_hash": replay_check.get("replay_sha256"),
        "provider_calls": 0,
        "heg_commit": HEG_COMMIT,
        "stage6_started": False,
    }
    result["result_sha256"] = sha256_value(result)
    _write_json(root / SUMMARY_PATH_NAME, result)
    _write_json(root / TERMINAL_PATH_NAME, result)
    return result


def finalize_stage5(
    config_path: str | Path = "configs/stage5-generalization.toml",
    *,
    preserved_evidence_path: str | Path,
    evidence_manifest_sha256: str,
    report_path: str | Path = REPORT_PATH,
) -> dict[str, Any]:
    """Bind preserved-evidence verification and write the final report."""
    config = load_stage5_config(config_path)
    result = _read_json(config.run_root / TERMINAL_PATH_NAME)
    preserved = Path(preserved_evidence_path)
    if not preserved.is_dir() or len(evidence_manifest_sha256) != 64:
        raise ValueError("preserved evidence path or manifest hash is invalid")
    manifest_candidates = (
        preserved / "evidence-manifest.sha256",
        preserved / "SHA256SUMS",
    )
    manifest_file = next((path for path in manifest_candidates if path.is_file()), None)
    if manifest_file is None or sha256_bytes(manifest_file.read_bytes()) != evidence_manifest_sha256:
        raise ValueError("preserved evidence manifest is missing or has the wrong hash")
    # Preservation coordinates belong in the report and issue handoff, not in
    # the hashed run artifact itself; this avoids a self-referential evidence
    # manifest while keeping the terminal result hash stable.
    gates = cast(dict[str, bool], result.get("gates", {}))
    gates["15_artifact_provenance_preservation_repository_verified"] = True
    result["gates"] = gates
    result["decision"] = _decision(gates)
    result["result_sha256"] = sha256_value({key: value for key, value in result.items() if key != "result_sha256"})
    _write_json(config.run_root / TERMINAL_PATH_NAME, result)
    _write_json(config.run_root / SUMMARY_PATH_NAME, result)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    metrics = cast(Mapping[str, Any], result.get("metrics", {}))
    lines = [
        "# Stage 5 held-out generalization report",
        "",
        f"Terminal decision: **{result['decision']}**",
        "",
        "Stage 5 used the frozen four-policy roster, held-out relabeled graph manifest, and provider-free Stage 4E evaluation contract. Stage 6 was not started.",
        "",
        f"- Freeze SHA-256: `{result['freeze_sha256']}`",
        f"- Manifest SHA-256: `{result['manifest_sha256']}`",
        f"- Policy provenance SHA-256: `{result['policy_sha256']}`",
        f"- Metric implementation SHA-256: `{result['metric_sha256']}`",
        f"- Bootstrap SHA-256: `{result['bootstrap_sha256']}`",
        f"- Gate result SHA-256: `{result['gate_sha256']}`",
        f"- Primary replay-comparison hash: `{result['primary_hash']}`",
        f"- Replay replay-comparison hash: `{result['replay_hash']}`",
        f"- Primary summary hash: `{result['primary']['summary_sha256']}`",
        f"- Replay summary hash: `{result['replay']['summary_sha256']}`",
        f"- Preserved evidence: `{preserved}`",
        f"- Evidence manifest SHA-256: `{evidence_manifest_sha256}`",
        "- Provider/model/App Server/oracle/runtime-network calls: **0**",
        f"- HEG commit: `{HEG_COMMIT}` (read-only and clean)",
        "",
        "## Principal effects",
        "",
    ]
    effects = cast(Mapping[str, Mapping[str, Any]], metrics.get("effects", {}))
    for effect in EFFECTS:
        item = effects[effect]
        interval = cast(Mapping[str, Any], cast(Mapping[str, Any], metrics.get("bootstrap", {})).get("intervals", {})).get(effect)
        lines.append(f"- {effect}: theta `{item['theta']['fraction']}`, relative improvement `{item['relative_improvement']['fraction']}`, 95% interval `{interval}`.")
    lines.extend(["", "## Gates", ""])
    for name, passed in gates.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(["", "The issue remains open for review; no automatic merge was performed.", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    return result


__all__ = ["finalize_stage5", "freeze_stage5", "generalize_stage5"]
