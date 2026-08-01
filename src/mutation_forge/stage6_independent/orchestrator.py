"""Coordinator for the frozen Stage 6 audit, replication, and decision.

The coordinator is intentionally small: all scientific arithmetic lives in
``metrics`` and all raw-artifact checks live in ``audit``/``recompute``.  It
does not import a Stage 5 implementation or expose a policy-search path.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .audit import audit_stage5
from .config import (
    HEG_COMMIT,
    POLICY_IDS,
    PROJECT_COMMIT,
    REQUIRED_ENTRY_COMMIT,
    STAGE5_EVIDENCE_MANIFEST_SHA256,
    Stage6Config,
    load_config,
    manifest_hash,
    sha256_bytes,
    validate_manifest,
)
from .metrics import (
    EFFECTS,
    bootstrap,
    parse_metrics_episodes,
    summarize,
)
from .metrics import (
    gates as metric_gates,
)
from .persistence import (
    canonical_bytes,
    canonical_record_hash,
    read_json,
    read_shard,
    reduction_hash,
    write_json,
)
from .recompute import recompute_stage5
from .redteam import run_redteam, write_fixture_set
from .runner import POLICY_IDS as RUNNER_POLICY_IDS
from .runner import plan as runner_plan
from .runner import run_pass, verify_replay

DECISIONS = (
    "GO_TO_STAGE_7",
    "NO_GO",
    "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
)
FREEZE_SCHEMA = "stage6.verification.freeze.v1"
FREEZE_TAG = "stage6-verification-frozen-v1"
FREEZE_PATH = Path("configs/stage6-verification-freeze-v1.json")
PROVENANCE_AMENDMENT_PATH = Path("configs/stage6-verification-provenance-amendment-v1.json")
REPORT_SCHEMA = "stage6.verification.report.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _config_manifest(config: Stage6Config) -> dict[str, Any]:
    if not config.manifest_path.is_file():
        raise FileNotFoundError(config.manifest_path)
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Stage 6 manifest must be an object")
    validate_manifest(manifest, config)
    return cast(dict[str, Any], dict(manifest))


def _freeze_file(config: Stage6Config) -> Path:
    return (config.project_repo / FREEZE_PATH).resolve()


def _historical_freeze_pin_is_amended(
    config: Stage6Config,
    freeze: Mapping[str, Any],
) -> bool:
    """Accept only the retained old pin covered by the immutable amendment."""

    path = (config.project_repo / PROVENANCE_AMENDMENT_PATH).resolve()
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, Mapping):
        return False
    expected_hash = hashlib.sha256(
        canonical_bytes({key: item for key, item in value.items() if key != "amendment_sha256"})
    ).hexdigest()
    original = value.get("original_freeze")
    scientific = value.get("scientific_impact")
    return bool(
        value.get("amendment_sha256") == expected_hash
        and value.get("issue_required_base_commit") == REQUIRED_ENTRY_COMMIT
        and isinstance(original, Mapping)
        and original.get("freeze_tag") == FREEZE_TAG
        and original.get("freeze_payload_sha256") == freeze.get("freeze_sha256")
        and original.get("incorrectly_recorded_required_project_commit") == PROJECT_COMMIT
        and isinstance(scientific, Mapping)
        and scientific.get("scientific_inputs_changed") is False
    )


def _load_freeze(config: Stage6Config) -> dict[str, Any]:
    path = _freeze_file(config)
    if not path.is_file():
        raise FileNotFoundError(f"Stage 6 freeze is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("invalid Stage 6 freeze")
    freeze = cast(dict[str, Any], dict(value))
    expected_hash = hashlib.sha256(
        canonical_bytes({key: item for key, item in freeze.items() if key != "freeze_sha256"})
    ).hexdigest()
    if freeze.get("freeze_sha256") != expected_hash:
        raise ValueError("Stage 6 freeze payload hash mismatch")
    if freeze.get("freeze_tag") != FREEZE_TAG:
        raise ValueError("Stage 6 freeze tag identity mismatch")
    recorded_project_commit = freeze.get("required_project_commit")
    project_pin_ok = recorded_project_commit == REQUIRED_ENTRY_COMMIT or (
        recorded_project_commit == PROJECT_COMMIT and _historical_freeze_pin_is_amended(config, freeze)
    )
    if not project_pin_ok or freeze.get("required_heg_commit") != HEG_COMMIT:
        raise ValueError("Stage 6 freeze repository pins differ")
    if freeze.get("heg_commit") != HEG_COMMIT or freeze.get("heg_dirty") is not False:
        raise ValueError("Stage 6 freeze HEG provenance is not clean")
    tag_commit = _git(config.project_repo, "rev-list", "-n", "1", FREEZE_TAG)
    frozen_commit = str(freeze.get("project_commit", ""))
    if not frozen_commit or not tag_commit:
        raise ValueError("Stage 6 freeze commit or tag is missing")
    ancestry = subprocess.run(
        ["git", "-C", str(config.project_repo), "merge-base", "--is-ancestor", frozen_commit, tag_commit],
        capture_output=True,
        timeout=30,
    )
    if ancestry.returncode != 0:
        raise ValueError("Stage 6 freeze tag does not descend from its frozen commit")
    if freeze.get("official_verification_started") is not False or freeze.get("stage6_results_observed") is not False or freeze.get("stage7_started") is not False:
        raise ValueError("Stage 6 freeze is marked as observed or started")
    return freeze


def freeze(config_path: str | Path = "configs/stage6-verification.toml") -> dict[str, Any]:
    """Create the immutable preregistration envelope.

    Existing bytes are never replaced with different bytes.  This makes a
    second invocation a harmless verification rather than a mutable freeze.
    """

    config = load_config(config_path)
    manifest = _config_manifest(config)
    freeze_path = _freeze_file(config)
    project_commit = _git(config.project_repo, "rev-parse", "HEAD")
    heg_commit = _git(config.heg_repo, "rev-parse", "HEAD")
    heg_dirty = bool(_git(config.heg_repo, "status", "--short"))
    payload: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "freeze_tag": FREEZE_TAG,
        "config_path": str(config.source_path),
        "config_sha256": _sha(config.source_path),
        "config_stable_hash": config.stable_hash(),
        "manifest_path": str(config.manifest_path),
        "manifest_sha256": manifest_hash(manifest),
        "stage5_manifest_sha256": config.expected_stage5_manifest_sha256,
        "stage5_evidence_path": str(config.stage5_evidence_path),
        "stage5_evidence_manifest_path": str(config.stage5_evidence_manifest_path),
        "stage5_evidence_manifest_sha256": config.expected_stage5_evidence_manifest_sha256,
        "stage5_freeze_sha256": _sha(config.stage5_freeze_path),
        "project_commit": project_commit,
        "required_project_commit": REQUIRED_ENTRY_COMMIT,
        "heg_commit": heg_commit,
        "required_heg_commit": HEG_COMMIT,
        "heg_dirty": heg_dirty,
        "policy_ids": list(POLICY_IDS),
        "policies": {
            policy_id: {
                "path": str(config.policy_paths[policy_id]),
                "source_sha256": config.policy_source_hashes[policy_id],
                "normalized_ast_sha256": config.policy_ast_hashes[policy_id],
                "behavior_signature_sha256": config.policy_behavior_hashes[policy_id],
            }
            for policy_id in POLICY_IDS
        },
        "experiment": config.experiment.__dict__ if hasattr(config.experiment, "__dict__") else {
            "orders": list(config.experiment.orders),
            "graph_seeds": list(config.experiment.graph_seeds),
            "relabeling_seeds": list(config.experiment.relabeling_seeds),
            "policy_seeds": list(config.experiment.policy_seeds),
            "horizon": config.experiment.horizon,
            "identity_count": config.experiment.identity_count,
            "shard_count": config.experiment.shard_count,
            "episodes_per_shard": config.experiment.episodes_per_shard,
        },
        "resources": {
            "workers": config.resources.workers,
            "reserved_physical_cores": config.resources.reserved_physical_cores,
            "thread_count": config.resources.thread_count,
        },
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "seed": config.bootstrap_seed,
            "confidence_level": config.confidence_level,
            "percentile_rule": "linear_interpolation_at_p_times_n_minus_1",
        },
        "gates": {
            "champion_stage3_threshold": config.champion_stage3_threshold,
            "champion_random_threshold": config.champion_random_threshold,
            "structural_retention_threshold": config.structural_retention_threshold,
        },
        "relabeling_algorithm": config.relabel_algorithm,
        "official_verification_started": False,
        "stage6_results_observed": False,
        "provider_calls_allowed": False,
        "stage7_started": False,
    }
    payload["freeze_sha256"] = sha256_bytes(canonical_bytes(payload))
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(payload) + b"\n"
    if freeze_path.exists() and freeze_path.read_bytes() != data:
        raise ValueError("Stage 6 freeze already exists with different bytes")
    if not freeze_path.exists():
        freeze_path.write_bytes(data)
    return {"status": "completed", "freeze_path": str(freeze_path), **payload}


def audit(config_path: str | Path = "configs/stage6-verification.toml") -> dict[str, Any]:
    config = load_config(config_path)
    _load_freeze(config)
    destination = config.run_root / "stage5-audit-copy"
    report = audit_stage5(
        config.stage5_evidence_path,
        destination,
        config.project_repo,
        config.heg_repo,
        expected_manifest_sha256=STAGE5_EVIDENCE_MANIFEST_SHA256,
    )
    report["schema_version"] = "stage6.verification.audit.v1"
    report["stage5_evidence_manifest_sha256"] = config.expected_stage5_evidence_manifest_sha256
    report["analysis_path"] = str(destination)
    write_json(config.run_root / "stage5-audit.json", report, overwrite=True)
    return report


def redteam(config_path: str | Path = "configs/stage6-verification.toml") -> dict[str, Any]:
    config = load_config(config_path)
    _load_freeze(config)
    fixture_path = config.run_root / "redteam" / "fixtures"
    write_fixture_set(fixture_path)
    result = run_redteam(fixture_root=fixture_path)
    result["fixture_root"] = str(fixture_path)
    write_json(config.run_root / "redteam" / "findings.json", result, overwrite=True)
    return result


def plan_fresh(config_path: str | Path = "configs/stage6-verification.toml") -> dict[str, Any]:
    config = load_config(config_path)
    _load_freeze(config)
    manifest = _config_manifest(config)
    prepared = runner_plan(manifest, policies={policy_id: True for policy_id in RUNNER_POLICY_IDS})
    prepared["manifest_sha256"] = manifest_hash(manifest)
    prepared["config_sha256"] = _sha(config.source_path)
    prepared["stage6_manifest_sha256"] = manifest.get("manifest_sha256")
    write_json(config.run_root / "fresh-plan.json", prepared, overwrite=True)
    return prepared


def _source_policies(config: Stage6Config) -> dict[str, str]:
    return {
        policy_id: config.policy_paths[policy_id].read_text(encoding="utf-8")
        for policy_id in POLICY_IDS
    }


def run_fresh(
    config_path: str | Path = "configs/stage6-verification.toml",
    *,
    workers: int | None = None,
    replay: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    freeze_value = _load_freeze(config)
    if freeze_value.get("stage7_started") is True:
        raise ValueError("Stage 7 must not be started by Stage 6")
    prepared = plan_fresh(config_path)
    result = run_pass(
        prepared,
        config.run_root,
        policies=_source_policies(config),
        workers=config.resources.workers if workers is None else workers,
        replay=replay,
        config=config,
    )
    result["schema_version"] = "stage6.verification.fresh-run.v1"
    result["config_sha256"] = _sha(config.source_path)
    result["manifest_sha256"] = prepared.get("stage6_manifest_sha256")
    result["bootstrap_seed"] = config.bootstrap_seed
    write_json(config.run_root / "fresh-run.json", result, overwrite=True)
    return result


def _load_summary(root: Path, name: str = "summary.json") -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def _records(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    inline = summary.get("records")
    if isinstance(inline, list):
        return [cast(dict[str, Any], row) for row in inline if isinstance(row, Mapping)]
    root = Path(str(summary.get("artifact_dir", ".")))
    rows: list[dict[str, Any]] = []
    for entry in cast(list[Mapping[str, Any]], summary.get("shards", [])):
        rows.extend(read_shard(root, entry))
    return rows


def _validate_fresh_records(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    config: Stage6Config,
) -> dict[str, Any]:
    expected_rows = cast(list[Mapping[str, Any]], manifest.get("episodes", []))
    expected = {str(row["episode_id"]): row for row in expected_rows}
    seen: set[str] = set()
    failures: list[dict[str, Any]] = []
    traces_ok = True
    graph_ok = True
    budgets_ok = True
    counters_ok = True
    identity_ok = True
    for row in records:
        eid = str(row.get("episode_id", ""))
        if eid in seen or eid not in expected:
            failures.append({"episode_id": eid, "reason": "missing_or_duplicate_or_extra"})
        seen.add(eid)
        try:
            if row.get("canonical_episode_sha256") != canonical_record_hash(row):
                failures.append({"episode_id": eid, "reason": "canonical_hash"})
        except Exception as error:
            failures.append({"episode_id": eid, "reason": f"canonical_hash:{type(error).__name__}"})
        if set(row.get("policies", {})) != set(POLICY_IDS):
            failures.append({"episode_id": eid, "reason": "policy_roster"})
        expected_row = expected.get(eid, {})
        for key in ("order", "graph_seed", "relabeling_seed", "policy_seed", "horizon"):
            if row.get(key) != expected_row.get(key):
                failures.append({"episode_id": eid, "reason": f"identity:{key}"})
        budgets_ok &= (
            row.get("horizon") == config.experiment.horizon
            and row.get("initial_score_calls") == 1
            and row.get("selected_score_calls") == config.experiment.horizon * 4
            and row.get("evaluation_count") == config.experiment.horizon * 4
            and row.get("oracle_score_calls") == 0
        )
        counters_ok &= all(row.get(key) == 0 for key in ("model_calls", "app_server_calls", "oracle_score_calls", "runtime_network_calls"))
        identities = row.get("policy_identities", {})
        for policy_id in POLICY_IDS:
            identity = identities.get(policy_id, {}) if isinstance(identities, Mapping) else {}
            identity_ok &= (
                identity.get("source_sha256") == config.policy_source_hashes[policy_id]
                and identity.get("normalized_ast_sha256") == config.policy_ast_hashes[policy_id]
                and identity.get("behavior_signature_sha256") == config.policy_behavior_hashes[policy_id]
            )
        proof = row.get("relabel_proof")
        if not isinstance(proof, Mapping):
            graph_ok = False
        else:
            permutation = proof.get("permutation")
            graph_ok &= (
                isinstance(permutation, list)
                and sorted(permutation) == list(range(int(row.get("order", 0))))
                and proof.get("algorithm") == config.relabel_algorithm
                and proof.get("base_graph_hash") == row.get("base_graph_hash")
                and proof.get("relabeled_graph_hash") == row.get("relabeled_graph_hash")
            )
        steps = row.get("steps")
        if not isinstance(steps, list) or len(steps) != config.experiment.horizon:
            traces_ok = False
        else:
            for step in steps:
                traces = step.get("policies", {}) if isinstance(step, Mapping) else {}
                if set(traces) != set(POLICY_IDS):
                    traces_ok = False
                for trace in cast(Mapping[str, Any], traces).values():
                    flags = trace.get("ranker_flags", {}) if isinstance(trace, Mapping) else {}
                    traces_ok &= not any(bool(flags.get(flag)) for flag in ("exception", "timeout", "crash", "protocol"))
    expected_ids = set(expected)
    complete = seen == expected_ids and len(records) == len(expected_ids)
    return {
        "complete": complete,
        "record_count": len(records),
        "expected_count": len(expected_ids),
        "failures": failures,
        "traces_ok": traces_ok,
        "graph_ok": graph_ok,
        "budgets_ok": budgets_ok,
        "counters_ok": counters_ok,
        "identity_ok": identity_ok,
        "timing_stripped_reduction_sha256": reduction_hash(records, timing_only=True),
    }


def _stage6_gates(
    config: Stage6Config,
    manifest: Mapping[str, Any],
    primary: Mapping[str, Any],
    replay: Mapping[str, Any],
    replay_check: Mapping[str, Any],
    validation: Mapping[str, Any],
    summary: Any,
    bootstrap_summary: Any,
    *,
    audit_result: Mapping[str, Any],
    redteam_result: Mapping[str, Any],
    stage5_recomputation: Mapping[str, Any],
    preservation_verified: bool,
) -> dict[str, bool]:
    metric = metric_gates(
        summary,
        bootstrap_summary,
        champion_stage3_threshold=config.champion_stage3_threshold,
        champion_random_threshold=config.champion_random_threshold,
        structural_retention_threshold=config.structural_retention_threshold,
    )
    findings = cast(list[Mapping[str, Any]], redteam_result.get("findings", []))
    redteam_clear = redteam_result.get("status") == "passed" and all(bool(item.get("passed")) for item in findings)
    return {
        "1_stage5_audit_complete_and_manifest_verified": audit_result.get("ok") is True,
        "2_stage5_independent_recomputation_exact": stage5_recomputation.get("status") == "passed" and stage5_recomputation.get("comparison", {}).get("exact") is True,
        "3_redteam_no_unresolved_critical_high_or_material_medium": redteam_clear,
        "4_manifest_complete_and_disjoint": bool(validation.get("complete")) and len(cast(list[Any], manifest.get("episodes", []))) == config.experiment.episode_count,
        "5_primary_and_replay_complete_equal_budgets": primary.get("status") == "completed" and replay.get("status") == "completed" and validation.get("budgets_ok") is True,
        "6_timing_stripped_replay_identity_exact": replay_check.get("exact") is True,
        "7_graph_validity_100_percent": validation.get("graph_ok") is True,
        "8_zero_worker_failures_crashes_timeouts_protocol_violations": validation.get("traces_ok") is True,
        "9_selected_plan_only_zero_oracle": validation.get("counters_ok") is True,
        "10_zero_model_app_server_provider_runtime_network_calls": validation.get("counters_ok") is True,
        "11_C_vs_stage3_relative_improvement_ge_2_percent": metric["relative_improvement_C_vs_stage3_at_least_threshold"],
        "12_C_vs_stage3_bootstrap_lower_bound_positive": metric["bootstrap_C_vs_stage3_lower_bound_positive"],
        "13_C_vs_stage3_nonnegative_each_order_and_six_strata": metric["C_vs_stage3_nonnegative_each_order"] and metric["C_vs_stage3_nonnegative_all_six_order_relabel_strata"],
        "14_C_vs_random_and_structural_retention_thresholds": metric["relative_improvement_C_vs_random_at_least_threshold"] and metric["bootstrap_C_vs_random_lower_bound_positive"] and metric["structural_retention_at_least_threshold"],
        "15_artifact_provenance_preservation_repository_verified": preservation_verified and bool(primary.get("canonical_reduction_sha256")) and bool(replay.get("canonical_reduction_sha256")) and validation.get("identity_ok") is True,
    }


def reduce(
    config_path: str | Path = "configs/stage6-verification.toml",
    *,
    preservation_verified: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    freeze_value = _load_freeze(config)
    manifest = _config_manifest(config)
    root = config.run_root
    primary = _load_summary(root / "primary")
    replay = _load_summary(root / "replay")
    primary_records = _records(primary)
    replay_records = _records(replay)
    validation = _validate_fresh_records(primary_records, manifest, config)
    replay_validation = _validate_fresh_records(replay_records, manifest, config)
    validation["complete"] = bool(validation["complete"] and replay_validation["complete"])
    validation["budgets_ok"] = bool(validation["budgets_ok"] and replay_validation["budgets_ok"])
    validation["graph_ok"] = bool(validation["graph_ok"] and replay_validation["graph_ok"])
    validation["traces_ok"] = bool(validation["traces_ok"] and replay_validation["traces_ok"])
    validation["counters_ok"] = bool(validation["counters_ok"] and replay_validation["counters_ok"])
    validation["identity_ok"] = bool(validation["identity_ok"] and replay_validation["identity_ok"])
    replay_check = verify_replay(primary, replay)
    episodes = parse_metrics_episodes(primary_records, POLICY_IDS)
    summary = summarize(episodes, POLICY_IDS)
    bootstrap_summary = bootstrap(
        summary,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
    )
    relative_improvements = summary.relative_improvements
    effect_payloads: dict[str, dict[str, Any]] = {}
    for effect in EFFECTS:
        payload = dict(summary.effects[effect].as_dict())
        payload["relative_improvement"] = {
            "fraction": str(relative_improvements[effect].numerator)
            + (
                f"/{relative_improvements[effect].denominator}"
                if relative_improvements[effect].denominator != 1
                else ""
            ),
            "value": float(relative_improvements[effect]),
        }
        effect_payloads[effect] = payload
    independent_metrics = {
        "policy_means": {key: {"fraction": str(value.numerator) + (f"/{value.denominator}" if value.denominator != 1 else ""), "value": float(value)} for key, value in summary.policy_means.items()},
        "effects": effect_payloads,
        "relative_improvements": {
            effect: {
                "fraction": str(value.numerator) + (f"/{value.denominator}" if value.denominator != 1 else ""),
                "value": float(value),
            }
            for effect, value in relative_improvements.items()
        },
        "structural_retention": {"fraction": str(summary.structural_retention.numerator) + (f"/{summary.structural_retention.denominator}" if summary.structural_retention.denominator != 1 else ""), "value": float(summary.structural_retention)},
        "bootstrap": bootstrap_summary.as_dict(),
    }
    audit_path = root / "stage5-audit.json"
    redteam_path = root / "redteam" / "findings.json"
    audit_result = read_json(audit_path) if audit_path.is_file() else {"ok": False}
    redteam_result = read_json(redteam_path) if redteam_path.is_file() else {"status": "failed", "findings": []}
    recomputation_path = root / "stage5-recomputation.json"
    if recomputation_path.is_file():
        stage5_recomputation = read_json(recomputation_path)
    else:
        stage5_recomputation = recompute_stage5(config.stage5_evidence_path)
        write_json(recomputation_path, stage5_recomputation, overwrite=True)
    gates = _stage6_gates(
        config,
        manifest,
        primary,
        replay,
        replay_check,
        validation,
        summary,
        bootstrap_summary,
        audit_result=audit_result,
        redteam_result=redteam_result,
        stage5_recomputation=stage5_recomputation,
        preservation_verified=preservation_verified,
    )
    decision = "GO_TO_STAGE_7" if all(gates.values()) else "NO_GO"
    result: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "decision": decision,
        "terminal_decision": decision,
        "stage7_started": False,
        "freeze_tag": FREEZE_TAG,
        "freeze": freeze_value,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "audit": audit_result,
        "stage5_recomputation": stage5_recomputation,
        "redteam": redteam_result,
        "primary": {key: value for key, value in primary.items() if key != "records"},
        "replay": {key: value for key, value in replay.items() if key != "records"},
        "replay_verification": replay_check,
        "validation": validation,
        "metrics": independent_metrics,
        "gates": gates,
        "gate_count": len(gates),
        "all_gates_pass": all(gates.values()),
        "preservation_verified": preservation_verified,
        "provider_calls": 0,
    }
    write_json(root / "stage6-terminal.json", result, overwrite=True)
    return result


__all__ = [
    "DECISIONS",
    "FREEZE_PATH",
    "FREEZE_SCHEMA",
    "FREEZE_TAG",
    "audit",
    "freeze",
    "plan_fresh",
    "redteam",
    "reduce",
    "run_fresh",
]
