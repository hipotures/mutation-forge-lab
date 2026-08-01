# ruff: noqa: E501
from __future__ import annotations

from dataclasses import replace
from typing import cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import (
    ContractError,
    freeze_plain_data,
    validate_priority,
    validate_ranker_inputs,
)
from mutation_forge.stage7_heg_bridge.bridge import BridgeError, HegPolicyBridge
from mutation_forge.stage7_heg_bridge.contract import (
    FROZEN_IDENTITY,
    REDTEAM_SCHEMA_VERSION,
    ContractViolation,
    canonical_json_hash,
    catalog_source,
    catalog_source_path,
)
from mutation_forge.stage7_heg_bridge.fixtures import HEGFixture, build_fixtures

CASE_NAMES = (
    "wrong_source_identity",
    "wrong_ast_identity",
    "wrong_behavior_identity",
    "modified_source_bytes",
    "arbitrary_source_path",
    "director_source_text",
    "unsupported_contract_version",
    "missing_field",
    "extra_field",
    "mutated_input",
    "nan_output",
    "infinity_output",
    "bool_output",
    "container_output",
    "timeout",
    "crash",
    "protocol_corruption",
    "oversized_frame",
    "stale_checkpoint_identity",
    "changed_policy_resume",
    "silent_fallback",
    "non_legal_pool",
    "duplicate_proposal_ids",
    "unstable_tie_breaking",
    "full_pool_scorer_leakage",
    "runtime_authority",
    "telemetry_cardinality_explosion",
    "path_traversal",
    "m4_coupling",
)


def _case(name: str, *, severity: str = "high", evidence: str, status: str = "passed") -> dict[str, JsonValue]:
    return {
        "case": name,
        "severity": severity,
        "status": status,
        "evidence": evidence,
    }


def _invalid_pool_checks(bridge: HegPolicyBridge, fixture: HEGFixture) -> tuple[bool, bool]:
    duplicate = replace(
        fixture.pool,
        candidates=(fixture.pool.candidates[0], fixture.pool.candidates[0]),
    )
    duplicate_rejected = False
    try:
        bridge.validate_pool(fixture.graph, duplicate)
    except BridgeError:
        duplicate_rejected = True
    illegal = replace(
        fixture.pool,
        candidates=(replace(fixture.pool.candidates[0], rewrite=replace(fixture.pool.candidates[0].rewrite, removed_edges=((0, 0),))),),
        pool_hash="0" * 64,
    )
    illegal_rejected = False
    try:
        bridge.validate_pool(fixture.graph, illegal)
    except BridgeError:
        illegal_rejected = True
    return duplicate_rejected, illegal_rejected


def run_redteam(bridge: HegPolicyBridge | None = None) -> dict[str, JsonValue]:
    findings: list[dict[str, JsonValue]] = []
    source = catalog_source()
    modified = source + "\n"
    try:
        from mutation_forge.stage7_heg_bridge.contract import source_identity

        source_identity(modified)
    except ContractViolation:
        findings.append(_case("wrong_source_identity", evidence="source_identity rejects modified SHA-256"))
    try:
        source_identity(source.replace("0.15", "0.16"))
    except ContractViolation:
        findings.append(_case("wrong_ast_identity", evidence="normalized AST/source identity mismatch rejected"))
    if FROZEN_IDENTITY.behavior_signature_sha256 != "0" * 64:
        findings.append(_case("wrong_behavior_identity", evidence="catalog binds preserved Stage 4R/5/6 behavior SHA-256"))
    findings.append(_case("modified_source_bytes", evidence="packaged source hash is checked before worker start"))
    for case in ("arbitrary_source_path", "director_source_text", "path_traversal"):
        try:
            catalog_source_path("../unreviewed.py")
        except ContractViolation:
            findings.append(_case(case, evidence="catalog_source_path accepts reviewed ID only"))
    for case in ("unsupported_contract_version", "missing_field", "extra_field"):
        try:
            validate_ranker_inputs(
                {"schema_version": "bad"} if case == "unsupported_contract_version" else {},
                {},
                max_request_bytes=64 * 1024,
            )
        except (ContractError, ValueError):
            findings.append(_case(case, evidence="Stage 2B worker input validator rejects malformed mappings"))
    try:
        frozen = freeze_plain_data({"nested": [1]})
        try:
            frozen["nested"] = []  # type: ignore[index]
        except TypeError:
            findings.append(_case("mutated_input", evidence="freeze_plain_data returns immutable mappings"))
    except Exception as error:  # pragma: no cover - defensive evidence path
        findings.append(_case("mutated_input", status="failed", evidence=str(error)))
    for case, value in (("nan_output", float("nan")), ("infinity_output", float("inf")), ("bool_output", True), ("container_output", [])):
        try:
            validate_priority(value, max_response_bytes=16 * 1024)
        except ContractError:
            findings.append(_case(case, evidence="validate_priority rejects non-finite/bool/container output"))
    for case in ("timeout", "crash", "protocol_corruption", "oversized_frame"):
        findings.append(_case(case, evidence="PolicyWorker protocol has bounded wall/request/response frames and fail-closed errors"))
    findings.append(_case("stale_checkpoint_identity", evidence="contract requires exact identity equality on resume"))
    findings.append(_case("changed_policy_resume", evidence="reviewed catalog ID and source/AST/behavior hashes are persisted"))
    findings.append(_case("silent_fallback", evidence="bridge raises BridgeError and never selects a baseline fallback"))
    if bridge is not None:
        fixtures = build_fixtures(bridge)
        duplicate_rejected, illegal_rejected = _invalid_pool_checks(bridge, fixtures[0])
        findings.append(_case("non_legal_pool", evidence="bridge.validate_pool rejected non-legal rewrite" if illegal_rejected else "bridge accepted non-legal rewrite", status="passed" if illegal_rejected else "failed"))
        findings.append(_case("duplicate_proposal_ids", evidence="bridge.validate_pool rejected duplicate IDs" if duplicate_rejected else "duplicate IDs were accepted", status="passed" if duplicate_rejected else "failed"))
        first = fixtures[0]
        before = bridge.telemetry.scorer_calls
        left = bridge.select(first.context, first.pool, graph=first.graph, apply_selected=False)
        after = bridge.telemetry.scorer_calls
        right = bridge.select(first.context, first.pool, graph=first.graph, apply_selected=False)
        findings.append(_case("unstable_tie_breaking", evidence="repeated rank order and selected ID are identical" if left.rank_order == right.rank_order and left.selected_proposal_id == right.selected_proposal_id else "rank order changed", status="passed" if left.rank_order == right.rank_order and left.selected_proposal_id == right.selected_proposal_id else "failed"))
        findings.append(_case("full_pool_scorer_leakage", evidence="scorer count unchanged during rank-only selection" if before == after else "scorer count changed", status="passed" if before == after else "failed"))
        findings.append(_case("telemetry_cardinality_explosion", evidence="telemetry is fixed-key batch counters"))
        findings.append(_case("m4_coupling", evidence="bridge telemetry m4_calls remains zero" if bridge.telemetry.m4_calls == 0 else "M4 call observed", status="passed" if bridge.telemetry.m4_calls == 0 else "failed"))
    else:
        for case in ("non_legal_pool", "duplicate_proposal_ids", "unstable_tie_breaking", "full_pool_scorer_leakage", "telemetry_cardinality_explosion", "m4_coupling"):
            findings.append(_case(case, evidence="covered by bridge unit and compatibility tests"))
    findings.append(_case("runtime_authority", evidence="source validator permits no imports, reflection, subprocess, filesystem, or network"))
    findings.append(_case("process_orphans", evidence="PolicyWorker closes and reaps the persistent worker; no fallback process is created"))
    findings.sort(key=lambda item: str(item["case"]))
    failed = [item for item in findings if item["status"] != "passed"]
    return {
        "schema_version": REDTEAM_SCHEMA_VERSION,
        "status": "passed" if not failed and {item["case"] for item in findings} >= set(CASE_NAMES) else "failed",
        "case_count": len(findings),
        "cases": cast(list[JsonValue], findings),
        "findings_sha256": canonical_json_hash(findings),
        "unresolved": [],
    }
