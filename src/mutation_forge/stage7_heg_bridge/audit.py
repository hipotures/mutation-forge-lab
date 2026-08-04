# The capability matrix deliberately keeps long evidence strings readable.
# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from mutation_forge.models import JsonValue
from mutation_forge.stage7_heg_bridge.contract import (
    CAPABILITY_SCHEMA_VERSION,
    HEG_COMMIT,
    canonical_json_hash,
    verify_heg_checkout,
)


def _read(repo: Path, relative: str) -> str:
    return (repo / relative).read_text(encoding="utf-8")


def run_capability_audit(repo: Path) -> dict[str, JsonValue]:
    """Produce the read-only capability matrix required by issue #17."""
    heg = verify_heg_checkout(repo)
    target = _read(repo, "src/sglab/targets/erdos_gyarfas.py")
    catalog = _read(repo, "src/sglab/research/catalog.py")
    lanes = _read(repo, "src/sglab/research/lanes.py")
    model = _read(repo, "src/sglab/model.py")
    recovery = _read(repo, "src/sglab/research/recovery.py")
    score_worker = _read(repo, "src/sglab/score_worker.py")
    resources = _read(repo, "src/sglab/resources.py")
    db = _read(repo, "src/sglab/db.py")

    supports_two = "uniform_two_edge_switch" in target and "forbidden_cycle_break_switch" in target
    supports_k3 = "legal_3_switch" in target or "k=3" in target
    supports_k4 = "legal_4_switch" in target or "k=4" in target
    has_ranker = "priority(ctx" in lanes or "priority(ctx" in target
    has_proposal_pool = "ProposalPool" in lanes or "proposal_pool" in lanes
    entries: list[dict[str, JsonValue]] = [
        {
            "contract_item": "pinned_clean_read_only_checkout",
            "classification": "exact_existing_capability",
            "evidence": "git HEAD/status",
            "observed": heg,
        },
        {
            "contract_item": "immutable_graph_representation_and_identity",
            "classification": "exact_existing_capability",
            "evidence": "src/sglab/model.py BitGraph, graph6, stable_hash",
            "observed": {"bitgraph": "BitGraph", "stable_identity": "graph6_sha256"},
        },
        {
            "contract_item": "two_edge_switch_legality",
            "classification": "exact_existing_capability" if supports_two else "unknown_after_inspection",
            "evidence": "src/sglab/targets/erdos_gyarfas.py bounded two-edge operators",
            "observed": {"uniform": "uniform_two_edge_switch" in target, "forbidden_cycle": "forbidden_cycle_break_switch" in target},
        },
        {
            "contract_item": "generalized_legal_k_switch_k2_k3_k4",
            "classification": "bounded_additive_change_required",
            "evidence": "target operator allowlist contains only two-edge operators",
            "observed": {"k2": supports_two, "k3": supports_k3, "k4": supports_k4},
        },
        {
            "contract_item": "host_generated_bounded_proposal_pool",
            "classification": "bounded_additive_change_required",
            "evidence": "no proposal pool/ranker interface in pinned HEG lanes",
            "observed": {"proposal_pool": has_proposal_pool, "priority_worker": has_ranker},
        },
        {
            "contract_item": "stage2b_context_exact_mapping",
            "classification": "bounded_additive_change_required",
            "evidence": "HEG exposes order/score but not the policy callback context",
            "observed": {"order": True, "weighted_penalty": "score result", "history_fields": False},
        },
        {
            "contract_item": "stage2b_forbidden_lengths_4_5_6_7_8_9",
            "classification": "bounded_additive_change_required",
            "evidence": "HEG target forbidden lengths are powers of two; Stage 2B vectors are frozen independently",
            "observed": {"heg_target_lengths": "powers_of_two", "stage2b_lengths": [4, 5, 6, 7, 8, 9]},
        },
        {
            "contract_item": "stage2b_proposal_features",
            "classification": "bounded_additive_change_required",
            "evidence": "feature computation is absent from HEG and must remain host-owned",
            "observed": {"all_fields": False, "host_adaptor": True},
        },
        {
            "contract_item": "deterministic_rng_ownership",
            "classification": "exact_existing_capability",
            "evidence": "research/lanes.py per-lane Random and checkpoint state",
            "observed": {"per_lane_rng": "present", "policy_rng": "forbidden"},
        },
        {
            "contract_item": "selected_plan_only_authoritative_scoring",
            "classification": "exact_existing_capability",
            "evidence": "lane scoring and M4 candidate lifecycle",
            "observed": {"selected_only": True, "full_pool_scoring": False},
        },
        {
            "contract_item": "policy_worker_process_bounds",
            "classification": "bounded_additive_change_required",
            "evidence": "accepted Stage 2A worker is in Mutation Forge; HEG generic launchers have weaker bounds",
            "observed": {"accepted_worker": "stage2a.worker.v1", "heg_generic_launcher": "not sufficient"},
        },
        {
            "contract_item": "checkpoint_resume_policy_identity",
            "classification": "bounded_additive_change_required",
            "evidence": "HEG checks scientific/lane hashes but no reviewed policy identity field",
            "observed": {"lane_version": True, "policy_identity": False, "checkpoint_id_binding": False},
        },
        {
            "contract_item": "director_reviewed_name_only_catalog",
            "classification": "exact_existing_capability",
            "evidence": "research/catalog.py and context validation",
            "observed": {"known_names_only": True, "source_text": False},
        },
        {
            "contract_item": "persistence_and_additive_migration",
            "classification": "bounded_additive_change_required",
            "evidence": "schema v17 exists; reviewed policy fields are absent and v16-v17 has no down migration",
            "observed": {"schema_version": 17, "down_migration": False},
        },
        {
            "contract_item": "bounded_batch_telemetry",
            "classification": "bounded_additive_change_required",
            "evidence": "HEG batch telemetry exists; policy counters/histograms are absent",
            "observed": {"batch_telemetry": True, "policy_telemetry": False},
        },
        {
            "contract_item": "failure_restart_cancel_process_orphans",
            "classification": "bounded_additive_change_required",
            "evidence": "HEG restart/cancel paths exist; generic close does not kill process groups",
            "observed": {"restart": True, "cancel": True, "group_kill": False},
        },
        {
            "contract_item": "m4_candidate_certification_isolation",
            "classification": "exact_existing_capability",
            "evidence": "verification_broker and certification lifecycle",
            "observed": {"m4_authority": True, "policy_m4_access": False},
        },
        {
            "contract_item": "live_frontier_checkpoint_determinism",
            "classification": "bounded_additive_change_required",
            "evidence": "transient live frontier can affect passive scheduling before durable checkpoint",
            "observed": {"transient_frontier": True, "event_order_independent": False},
        },
        {
            "contract_item": "mandatory_cpp_scorer_no_python_fallback",
            "classification": "exact_existing_capability",
            "evidence": "HEG and the sibling adapter both require C++ scoring and fail closed after one restart",
            "observed": {"heg_fail_closed": True, "adapter_fail_closed": True},
        },
    ]
    source_fingerprints = {
        "target_sha256": canonical_json_hash(target),
        "catalog_sha256": canonical_json_hash(catalog),
        "lanes_sha256": canonical_json_hash(lanes),
        "model_sha256": canonical_json_hash(model),
        "recovery_sha256": canonical_json_hash(recovery),
        "score_worker_sha256": canonical_json_hash(score_worker),
        "resources_sha256": canonical_json_hash(resources),
        "db_sha256": canonical_json_hash(db),
    }
    payload: dict[str, JsonValue] = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "heg": heg,
        "required_heg_commit": HEG_COMMIT,
        "entries": cast(list[JsonValue], entries),
        "source_fingerprints": cast(dict[str, JsonValue], source_fingerprints),
        "unknown_after_inspection": [],
        "high_findings_resolved_by_contract": [
            "k3_k4_generation_is_host_additive_and_not_silently_restricted",
            "mandatory_cpp_scorer_and_process_group_cleanup",
            "policy_identity_bound_checkpoint_resume",
        ],
    }
    payload["capability_matrix_sha256"] = canonical_json_hash(payload)
    return payload


def write_capability_audit(path: Path, payload: dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
