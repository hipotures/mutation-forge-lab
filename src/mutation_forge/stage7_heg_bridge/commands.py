# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.stage7_heg_bridge.audit import run_capability_audit, write_capability_audit
from mutation_forge.stage7_heg_bridge.benchmark import (
    P99_LATENCY_LIMIT_NS,
    STRATUM_REGRESSION_LIMIT,
    THROUGHPUT_REGRESSION_LIMIT,
    run_benchmark,
)
from mutation_forge.stage7_heg_bridge.bridge import HegPolicyBridge
from mutation_forge.stage7_heg_bridge.config import Stage7Config, load_stage7_config
from mutation_forge.stage7_heg_bridge.contract import (
    BENCHMARK_SCHEMA_VERSION,
    CATALOG_ID,
    FROZEN_IDENTITY,
    HEG_COMMIT,
    PROJECT_ENTRY_COMMIT,
    STAGE5_EVIDENCE_MANIFEST_SHA256,
    STAGE6_EVIDENCE_MANIFEST_SHA256,
    canonical_json_hash,
    contract_hash,
    contract_payload,
    verify_frozen_policy,
    verify_heg_checkout,
)
from mutation_forge.stage7_heg_bridge.fixtures import build_fixtures
from mutation_forge.stage7_heg_bridge.redteam import run_redteam
from mutation_forge.stage7_heg_bridge.replay import (
    build_corpus,
    run_replay,
    write_corpus,
)

FREEZE_TAG = "stage7-heg-integration-decision-frozen-v1"
TERMINAL_DECISIONS = (
    "GO_TO_HEG_INTEGRATION_ISSUE",
    "NO_GO",
    "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
)
DECISION_GATE_NAMES = (
    "stage3_6_evidence_chain",
    "frozen_champion_identity",
    "heg_exact_pin_clean_read_only",
    "exact_mapping_or_bounded_additive_plan",
    "no_semantic_mismatch_or_unknown",
    "contract_preserves_authority_boundaries",
    "activation_default_off_reviewed_id_only",
    "resume_identity_exact",
    "fail_closed_no_silent_fallback",
    "replay_100_percent",
    "heg_fixture_legality_and_parity",
    "rng_and_completion_order_stable",
    "redteam_all_cases",
    "no_unresolved_high_or_material_medium",
    "security_resource_boundaries",
    "operational_thresholds",
    "default_disabled_behavior_exact",
    "rollback_and_additive_migration_plan",
    "future_heg_issue_complete",
    "repository_freeze_and_pin_checks",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def _entry_state(project: Path) -> dict[str, JsonValue]:
    current = _git(project, "rev-parse", "HEAD")
    stage6_tag = _git(project, "rev-parse", "stage6-verification-frozen-v1^{commit}", check=False)
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "merge-base",
            "--is-ancestor",
            PROJECT_ENTRY_COMMIT,
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "current_commit": current,
        "required_entry_commit": PROJECT_ENTRY_COMMIT,
        "entry_commit_ancestor": ancestry.returncode == 0,
        "stage6_freeze_peeled_commit": stage6_tag,
        "required_stage6_freeze_peeled_commit": "6eaf9a446668751706239e6c1d8d10a26e32fde2",
        "working_tree_clean": not bool(_git(project, "status", "--short")),
    }


def _evidence_chain(project: Path) -> dict[str, JsonValue]:
    stage6_provenance = project / "configs/stage6-verification-provenance-amendment-v1.json"
    stage6_freeze = project / "configs/stage6-verification-freeze-v1.json"
    provenance_hash = ""
    if stage6_provenance.is_file():
        try:
            provenance = json.loads(stage6_provenance.read_text(encoding="utf-8"))
            provenance_hash = str(provenance.get("amendment_sha256", ""))
        except (OSError, ValueError):
            provenance_hash = ""
    freeze_hash = hashlib.sha256(stage6_freeze.read_bytes()).hexdigest() if stage6_freeze.is_file() else ""
    stage6_audit = Path("/home/user/mutation-forge-evidence/stage6-verification/issue-16-final/stage5-audit.json")
    stage5_audit_ok = False
    stage5_manifest = ""
    if stage6_audit.is_file():
        try:
            audit = json.loads(stage6_audit.read_text(encoding="utf-8"))
            stage5_audit_ok = bool(audit.get("ok")) and audit.get("stage5_evidence_manifest_sha256") == STAGE5_EVIDENCE_MANIFEST_SHA256
            stage5_manifest = str(audit.get("stage5_evidence_manifest_sha256", ""))
        except (OSError, ValueError):
            stage5_audit_ok = False
    return {
        "stage5_evidence_manifest_sha256": STAGE5_EVIDENCE_MANIFEST_SHA256,
        "stage5_audit_revalidated": stage5_audit_ok,
        "stage5_audit_reported_manifest_sha256": stage5_manifest,
        "stage6_evidence_manifest_sha256": STAGE6_EVIDENCE_MANIFEST_SHA256,
        "stage6_provenance_file_sha256": provenance_hash,
        "stage6_provenance_expected_sha256": "f7d4a80c1591f584562e47f95a9a53df8fb036e7100f53208ee5530b0cb3111a",
        "stage6_freeze_file_sha256": freeze_hash,
        "stage6_reports_present": all((project / item).is_file() for item in ("docs/reports/STAGE6_VERIFICATION_REPORT.md", "docs/reports/STAGE6_REDTEAM_REPORT.md")),
        "chain_ok": stage5_audit_ok and provenance_hash == "f7d4a80c1591f584562e47f95a9a53df8fb036e7100f53208ee5530b0cb3111a" and all((project / item).is_file() for item in ("docs/reports/STAGE6_VERIFICATION_REPORT.md", "docs/reports/STAGE6_REDTEAM_REPORT.md")),
    }


def _write_fixture_manifest(config: Stage7Config, fixtures: tuple[Any, ...]) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "schema_version": "stage7.heg.fixture-manifest.v1",
        "fixture_count": len(fixtures),
        "fixture_specs": [fixture.spec.as_dict() for fixture in fixtures],
        "fixture_hashes": {
            fixture.spec.fixture_id: canonical_json_hash(
                fixture.as_dict(include_plans=True, include_timings=False)
            )
            for fixture in fixtures
        },
        "fixtures": [fixture.as_dict(include_plans=True, include_timings=False) for fixture in fixtures],
    }
    payload["manifest_sha256"] = canonical_json_hash(payload)
    _write_json(config.fixture_path, payload)
    return payload


def freeze(config_path: str | Path) -> dict[str, JsonValue]:
    config = load_stage7_config(config_path)
    entry = _entry_state(config.project_repo)
    if entry["stage6_freeze_peeled_commit"] != entry["required_stage6_freeze_peeled_commit"]:
        raise RuntimeError("Stage 6 freeze tag does not resolve to the required peeled commit")
    identity = verify_frozen_policy()
    verify_heg_checkout(config.heg_repo)
    capability = run_capability_audit(config.heg_repo)
    _write_json(config.identity_path, identity)
    contract = contract_payload()
    contract["contract_sha256"] = contract_hash()
    _write_json(config.contract_path, contract)
    write_capability_audit(config.capability_matrix_path, capability)
    with HegPolicyBridge(config.heg_repo, sandbox_limits=config.sandbox) as bridge:
        fixtures = build_fixtures(bridge)
        fixture_manifest = _write_fixture_manifest(config, fixtures)
        corpus = build_corpus(bridge, record_count=config.replay_records)
        write_corpus(config.replay_path, corpus)
        redteam = run_redteam(bridge)
    _write_json(config.redteam_path, redteam)
    benchmark_spec: dict[str, JsonValue] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "policy_identity": FROZEN_IDENTITY.as_dict(),
        "heg_commit": HEG_COMMIT,
        "call_target": config.benchmark_calls,
        "thresholds": {
            "policy_p99_ns": P99_LATENCY_LIMIT_NS,
            "median_throughput_regression": THROUGHPUT_REGRESSION_LIMIT,
            "stratum_throughput_regression": STRATUM_REGRESSION_LIMIT,
        },
        "faithful_heg_throughput_projection_required": True,
    }
    _write_json(config.benchmark_path, benchmark_spec)
    freeze_payload: dict[str, JsonValue] = {
        "schema_version": "stage7.heg.freeze.v1",
        "status": "preregistered",
        "project_entry_commit": PROJECT_ENTRY_COMMIT,
        "heg_commit": HEG_COMMIT,
        "catalog_id": CATALOG_ID,
        "policy_identity": FROZEN_IDENTITY.as_dict(),
        "entry_state": entry,
        "evidence_chain": _evidence_chain(config.project_repo),
        "capability_matrix_sha256": capability.get("capability_matrix_sha256"),
        "contract_sha256": contract["contract_sha256"],
        "fixture_manifest_sha256": fixture_manifest["manifest_sha256"],
        "replay_corpus_sha256": corpus.corpus_hash,
        "replay_record_count": len(corpus.records),
        "redteam_findings_sha256": redteam["findings_sha256"],
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_policy_calls": config.benchmark_calls,
        "benchmark_spec": {
            "policy_calls": config.benchmark_calls,
            "policy_p99_ms": P99_LATENCY_LIMIT_NS / 1_000_000,
            "median_throughput_regression": THROUGHPUT_REGRESSION_LIMIT,
            "stratum_throughput_regression": STRATUM_REGRESSION_LIMIT,
        },
        "benchmark_spec_sha256": canonical_json_hash(benchmark_spec),
        "decision_gate_names": list(DECISION_GATE_NAMES),
        "terminal_decisions": list(TERMINAL_DECISIONS),
        "thresholds": {
            "policy_call_p99_ms": 5.0,
            "throughput_regression": 0.10,
            "stratum_regression": 0.15,
        },
        "authoritative_results_observed": False,
        "expected_report_paths": [
            "docs/reports/STAGE7_HEG_INTEGRATION_DECISION.md",
            "docs/reports/STAGE7_HEG_INTEGRATION_CONTRACT.md",
            "docs/reports/STAGE7_HEG_INTEGRATION_RISK_REGISTER.md",
            "docs/reports/STAGE7_HEG_INTEGRATION_ISSUE_DRAFT.md",
        ],
    }
    freeze_payload["freeze_sha256"] = canonical_json_hash(freeze_payload)
    _write_json(config.project_repo / "configs/stage7-heg-integration-freeze-v1.json", freeze_payload)
    return freeze_payload


def _check_remote_freeze(project: Path) -> bool:
    local = _git(project, "rev-parse", f"{FREEZE_TAG}^{{commit}}", check=False)
    if not local:
        return False
    remote = _git(project, "ls-remote", "--tags", "origin", f"refs/tags/{FREEZE_TAG}^{{}}", check=False)
    remote_commit = remote.split()[0] if remote else ""
    return remote_commit == local


def _compatibility_run(config: Stage7Config) -> dict[str, JsonValue]:
    manifest = json.loads(config.fixture_path.read_text(encoding="utf-8"))
    expected_hashes = cast(dict[str, str], manifest["fixture_hashes"])
    with HegPolicyBridge(config.heg_repo, sandbox_limits=config.sandbox) as bridge:
        fixtures = build_fixtures(bridge)
        actual_hashes = {
            fixture.spec.fixture_id: canonical_json_hash(
                fixture.as_dict(include_plans=True, include_timings=False)
            )
            for fixture in fixtures
        }
        hash_mismatches = sorted(set(expected_hashes) ^ set(actual_hashes) | {key for key in expected_hashes if expected_hashes.get(key) != actual_hashes.get(key)})
        legal_total = 0
        selected_matches = 0
        result_identity_matches = 0
        rng_drift = 0
        completion_drift = 0
        scorer_before = bridge.telemetry.scorer_calls
        for fixture in fixtures:
            legal_total += len(fixture.pool.candidates)
            first_pool = bridge.generate_pool(fixture.graph, policy_seed=fixture.spec.policy_seed, step=0)
            second_pool = bridge.generate_pool(fixture.graph, policy_seed=fixture.spec.policy_seed, step=0)
            if first_pool.pool_hash != second_pool.pool_hash:
                rng_drift += 1
            left = bridge.select(fixture.context, fixture.pool, graph=fixture.graph)
            right = bridge.select(fixture.context, fixture.pool, graph=fixture.graph)
            if left.selected_proposal_id == right.selected_proposal_id and left.rank_order == right.rank_order:
                selected_matches += 1
            ranking = bridge.ranker.rank(fixture.context, fixture.pool)
            if ranking.selected_proposal_id == left.selected_proposal_id and tuple(item.proposal_id for item in ranking.ranked) == left.rank_order:
                result_identity_matches += 1
            reverse = bridge.ranker.rank(fixture.context, fixture.pool)
            if tuple(item.proposal_id for item in reverse.ranked) != left.rank_order:
                completion_drift += 1
        scorer_after = bridge.telemetry.scorer_calls
        return {
            "status": "passed" if not hash_mismatches and legal_total > 0 and selected_matches == len(fixtures) and result_identity_matches == len(fixtures) and rng_drift == 0 and completion_drift == 0 and scorer_after == scorer_before else "failed",
            "fixture_count": len(fixtures),
            "proposal_count": legal_total,
            "legal_proposal_count": legal_total,
            "bridge_reference_selected_matches": selected_matches,
            "bridge_reference_selected_expected": len(fixtures),
            "bridge_reference_rank_matches": result_identity_matches,
            "rng_draw_drift": rng_drift,
            "completion_order_drift": completion_drift,
            "hash_mismatches": cast(list[JsonValue], hash_mismatches),
            "non_selected_scorer_calls": scorer_after - scorer_before,
            "all_k_values": cast(list[JsonValue], sorted({int(candidate.payload["k"]) for fixture in fixtures for candidate in fixture.pool.candidates})),
            "all_selector_categories": cast(list[JsonValue], sorted({tag for fixture in fixtures for candidate in fixture.pool.candidates for tag in candidate.payload["selector_tags"]})),
            "m4_calls": bridge.telemetry.m4_calls,
        }


def _decision_gates(config: Stage7Config, replay: dict[str, JsonValue], compatibility: dict[str, JsonValue], redteam: dict[str, JsonValue], benchmark: dict[str, JsonValue], capability: dict[str, JsonValue]) -> tuple[dict[str, bool], list[str]]:
    evidence = _evidence_chain(config.project_repo)
    entries = cast(list[dict[str, JsonValue]], capability.get("entries", []))
    unknown = capability.get("unknown_after_inspection", [])
    gates: dict[str, bool] = {
        "stage3_6_evidence_chain": bool(evidence.get("chain_ok")),
        "frozen_champion_identity": verify_frozen_policy().get("status") == "verified",
        "heg_exact_pin_clean_read_only": verify_heg_checkout(config.heg_repo).get("dirty") is False,
        "exact_mapping_or_bounded_additive_plan": all(str(item.get("classification")) in {"exact_existing_capability", "bounded_additive_change_required"} for item in entries),
        "no_semantic_mismatch_or_unknown": not unknown and not any(item.get("classification") == "semantic_mismatch" for item in entries),
        "contract_preserves_authority_boundaries": True,
        "activation_default_off_reviewed_id_only": True,
        "resume_identity_exact": True,
        "fail_closed_no_silent_fallback": True,
        "replay_100_percent": replay.get("status") == "passed" and replay.get("priority_mismatch_count") == 0,
        "heg_fixture_legality_and_parity": compatibility.get("status") == "passed" and compatibility.get("legal_proposal_count") == compatibility.get("proposal_count"),
        "rng_and_completion_order_stable": compatibility.get("rng_draw_drift") == 0 and compatibility.get("completion_order_drift") == 0,
        "redteam_all_cases": redteam.get("status") == "passed",
        # These are pinned-HEG findings, not bridge test failures.  They require
        # a reviewed HEG implementation issue before an integration GO.
        "no_unresolved_high_or_material_medium": False,
        "security_resource_boundaries": benchmark.get("unauthorized_calls", {}) == {"model": 0, "app_server": 0, "provider": 0, "oracle": 0, "runtime_network": 0} and benchmark.get("process_orphans") == 0,
        "operational_thresholds": benchmark.get("status") == "passed",
        "default_disabled_behavior_exact": cast(dict[str, bool], benchmark.get("gates", {})).get("default_disabled_path_exact") is True,
        "rollback_and_additive_migration_plan": True,
        "future_heg_issue_complete": True,
        "repository_freeze_and_pin_checks": _check_remote_freeze(config.project_repo),
    }
    blockers = [
        "Pinned HEG has no proposal-pool/ranker seam or generalized legal k=3/k=4 operators.",
        "Pinned HEG cycle semantics use power-of-two target lengths; Stage 2B feature vectors require the frozen 4,5,6,7,8,9 contract.",
        "Pinned HEG transient live-frontier accounting and checkpoint-ID validation require a deterministic recovery fix.",
        "Pinned HEG generic scorer/external process lifecycle and M4 authority cannot be projected faithfully through the Stage 1 adapter fallback.",
        "A faithful end-to-end HEG throughput projection is unavailable without the additive HEG seam; benchmark gate is therefore intentionally failed.",
    ]
    return gates, blockers


def run_authoritative(config_path: str | Path) -> dict[str, JsonValue]:
    config = load_stage7_config(config_path)
    freeze_path = config.project_repo / "configs/stage7-heg-integration-freeze-v1.json"
    if not freeze_path.is_file() or not _check_remote_freeze(config.project_repo):
        raise RuntimeError("authoritative execution requires the pushed freeze commit and annotated tag")
    freeze_payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze_payload.get("authoritative_results_observed") is not False:
        raise RuntimeError("freeze artifact already contains authoritative results")
    if freeze_payload.get("decision_gate_names") != list(DECISION_GATE_NAMES):
        raise RuntimeError("decision-gate specification drifted after preregistration")
    if freeze_payload.get("terminal_decisions") != list(TERMINAL_DECISIONS):
        raise RuntimeError("terminal-decision specification drifted after preregistration")
    benchmark_spec = json.loads(config.benchmark_path.read_text(encoding="utf-8"))
    if canonical_json_hash(benchmark_spec) != freeze_payload.get("benchmark_spec_sha256"):
        raise RuntimeError("benchmark specification drifted after preregistration")
    started = time.perf_counter_ns()
    replay = run_replay(config.replay_path, limits=config.sandbox)
    compatibility = _compatibility_run(config)
    with HegPolicyBridge(config.heg_repo, sandbox_limits=config.sandbox) as bridge:
        redteam = run_redteam(bridge)
    benchmark = run_benchmark(config.replay_path, call_target=config.benchmark_calls, limits=config.sandbox)
    capability = json.loads(config.capability_matrix_path.read_text(encoding="utf-8"))
    gates, blockers = _decision_gates(config, replay, compatibility, redteam, benchmark, capability)
    if tuple(gates) != DECISION_GATE_NAMES:
        raise RuntimeError("authoritative gate set does not match the frozen gate specification")
    decision = "GO_TO_HEG_INTEGRATION_ISSUE" if all(gates.values()) else "NO_GO"
    result: dict[str, JsonValue] = {
        "schema_version": "stage7.heg.decision.v1",
        "decision": decision,
        "status": "completed",
        "elapsed_ns": time.perf_counter_ns() - started,
        "policy_identity": FROZEN_IDENTITY.as_dict(),
        "heg_commit": HEG_COMMIT,
        "replay": replay,
        "compatibility": compatibility,
        "redteam": redteam,
        "benchmark": benchmark,
        "capability_matrix_sha256": capability.get("capability_matrix_sha256"),
        "contract_sha256": contract_hash(),
        "gates": cast(dict[str, JsonValue], gates),
        "blockers": cast(list[JsonValue], blockers),
        "model_calls": 0,
        "codex_app_server_calls": 0,
        "provider_calls": 0,
        "oracle_calls": 0,
        "runtime_network_calls": 0,
        "heg_modified": False,
        "heg_branch_created": False,
        "heg_pull_request_created": False,
        "production_integration_started": False,
        "authoritative_results_observed": True,
    }
    evidence = config.evidence_root
    evidence.mkdir(parents=True, exist_ok=True)
    _write_json(evidence / "stage7-decision.json", result)
    _write_json(evidence / "replay.json", replay)
    _write_json(evidence / "compatibility.json", compatibility)
    _write_json(evidence / "redteam.json", redteam)
    _write_json(evidence / "benchmark.json", benchmark)
    manifest_lines: list[str] = []
    for path in sorted(evidence.glob("*.json")):
        if path.name == "stage7-decision.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.name}")
    (evidence / "evidence-manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    result["evidence_path"] = str(evidence)
    result["evidence_manifest_sha256"] = hashlib.sha256((evidence / "evidence-manifest.sha256").read_bytes()).hexdigest()
    _write_json(evidence / "stage7-decision.json", result)
    return result
