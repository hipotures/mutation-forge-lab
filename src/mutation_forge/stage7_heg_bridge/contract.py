from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.validation import ProgramIdentity, validate_policy

CONTRACT_SCHEMA_VERSION = "stage7.heg.integration.contract.v1"
CAPABILITY_SCHEMA_VERSION = "stage7.heg.capability-matrix.v1"
REPLAY_SCHEMA_VERSION = "stage7.heg.replay.v1"
FIXTURE_SCHEMA_VERSION = "stage7.heg.fixture.v1"
REDTEAM_SCHEMA_VERSION = "stage7.heg.redteam.v1"
BENCHMARK_SCHEMA_VERSION = "stage7.heg.benchmark.v1"

PROJECT_ENTRY_COMMIT = "a6f0da20fa5a3e1c8b58cbc77a0d613c54d9f051"
HEG_COMMIT = "fd97451b0f3d87400d1d955a2c6b1b18303344ff"
STAGE6_PEELED_COMMIT = "6eaf9a446668751706239e6c1d8d10a26e32fde2"
STAGE6_PROVENANCE_SHA256 = (
    "f7d4a80c1591f584562e47f95a9a53df8fb036e7100f53208ee5530b0cb3111a"
)
STAGE5_EVIDENCE_MANIFEST_SHA256 = (
    "e996563c145ac12bc7e7ae9bb284ae98d14a2990aaac9bce17e9992486780cce"
)
STAGE6_EVIDENCE_MANIFEST_SHA256 = (
    "66064a1b9a7583da588d64cab2e3e4a79be6a5f77997be3df0e4fbbfd3677e87"
)

CATALOG_ID = "mutation_forge_stage4r_v1"
FROZEN_POLICY_ID = "program-d5ad1c8203e0d9f25f03aabd"
POLICY_SOURCE_SHA256 = "e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b"
POLICY_AST_SHA256 = "2243214df58c805e9a9343dc31ed082279e1c2ac31b21243bf889dbc9a19e165"
POLICY_BEHAVIOR_SHA256 = "8c2bdaa213f11b253d3ffcae1653bd01536879bb5c254a1586ded9ae522a868e"
VALIDATOR_VERSION = "stage2a.validator.v2"
RUNTIME_PROTOCOL_VERSION = "stage2a.worker.v1"
FEATURE_CONTRACT_VERSION = "stage2b.context.v1+stage2b.proposal.v1"
PROPOSAL_POOL_CONTRACT_VERSION = "stage2b.pool.v1"
TIE_BREAKING_RULE = "descending_priority_then_lexicographic_proposal_id"
FAILURE_POLICY = "fail_closed_no_silent_fallback"
FROZEN_FORBIDDEN_LENGTHS = (4, 5, 6, 7, 8, 9)
FROZEN_WITNESS_SAMPLE_CAP = 32
FROZEN_CYCLE_NODE_BUDGET = 20_000
FROZEN_DISTANCE_QUERY_BUDGET = 256
FROZEN_LOCAL_RISK_BUDGET = 2_048

_SOURCE_RELATIVE_PATH = Path("fixtures/stage7/mutation_policy_stage4r_v1.py")


class ContractViolation(ValueError):
    """Raised when a frozen Stage 7 identity or boundary is violated."""


@dataclass(frozen=True, slots=True)
class FrozenPolicyIdentity:
    catalog_id: str
    policy_id: str
    source_sha256: str
    normalized_ast_sha256: str
    behavior_signature_sha256: str
    validator_version: str
    runtime_protocol_version: str

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "catalog_id": self.catalog_id,
            "policy_id": self.policy_id,
            "source_sha256": self.source_sha256,
            "normalized_ast_sha256": self.normalized_ast_sha256,
            "behavior_signature_sha256": self.behavior_signature_sha256,
            "validator_version": self.validator_version,
            "runtime_protocol_version": self.runtime_protocol_version,
        }


FROZEN_IDENTITY = FrozenPolicyIdentity(
    catalog_id=CATALOG_ID,
    policy_id=FROZEN_POLICY_ID,
    source_sha256=POLICY_SOURCE_SHA256,
    normalized_ast_sha256=POLICY_AST_SHA256,
    behavior_signature_sha256=POLICY_BEHAVIOR_SHA256,
    validator_version=VALIDATOR_VERSION,
    runtime_protocol_version=RUNTIME_PROTOCOL_VERSION,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def catalog_source_path(catalog_id: str = CATALOG_ID) -> Path:
    """Resolve only the reviewed catalog ID; paths and source text are rejected."""
    if catalog_id != CATALOG_ID:
        raise ContractViolation(f"unknown reviewed policy catalog ID: {catalog_id!r}")
    return project_root() / _SOURCE_RELATIVE_PATH


def catalog_source(catalog_id: str = CATALOG_ID) -> str:
    path = catalog_source_path(catalog_id)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ContractViolation(f"reviewed policy source is unavailable: {path}") from error


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def source_identity(source: str) -> ProgramIdentity:
    validation = validate_policy(source)
    if not validation.valid:
        raise ContractViolation(
            "frozen source fails the accepted Stage 2A validator: "
            + json.dumps(validation.as_dict(), sort_keys=True)
        )
    identity = validation.identity
    if identity.source_sha256 != POLICY_SOURCE_SHA256:
        raise ContractViolation("frozen source SHA-256 does not match the catalog")
    if identity.normalized_ast_sha256 != POLICY_AST_SHA256:
        raise ContractViolation("frozen normalized AST SHA-256 does not match the catalog")
    if identity.validator_version != VALIDATOR_VERSION:
        raise ContractViolation("validator version drifted from the frozen contract")
    return identity


def verify_frozen_policy() -> dict[str, JsonValue]:
    """Verify the packaged bytes and the externally frozen behavior identity.

    The behavior signature is intentionally not regenerated here: its identity
    is the Stage 4R/Stage 5/Stage 6 evidence identity and regenerating it with a
    different probe corpus would silently redefine the scientific contract.
    """
    path = catalog_source_path()
    source = catalog_source()
    identity = source_identity(source)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {
        "status": "verified",
        "catalog_source_path": str(path.relative_to(project_root())),
        "catalog_id": CATALOG_ID,
        "policy_id": FROZEN_POLICY_ID,
        "source_sha256": source_hash,
        "normalized_ast_sha256": identity.normalized_ast_sha256,
        "behavior_signature_sha256": POLICY_BEHAVIOR_SHA256,
        "behavior_identity_authority": "stage4r_stage5_stage6_preserved_evidence",
        "validator_version": identity.validator_version,
        "runtime_protocol_version": RUNTIME_PROTOCOL_VERSION,
    }


def verify_heg_checkout(repo: Path) -> dict[str, JsonValue]:
    repo = repo.resolve()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty_output = git("status", "--short")
    if commit != HEG_COMMIT:
        raise ContractViolation(f"HEG must be pinned to {HEG_COMMIT}, got {commit}")
    if dirty_output:
        raise ContractViolation("HEG checkout is dirty")
    return {
        "repo": str(repo),
        "commit": commit,
        "dirty": False,
        "read_only": True,
    }


def contract_payload() -> dict[str, JsonValue]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "catalog": FROZEN_IDENTITY.as_dict(),
        "entry": {
            "mutation_forge_commit": PROJECT_ENTRY_COMMIT,
            "heg_commit": HEG_COMMIT,
            "stage6_freeze_commit": STAGE6_PEELED_COMMIT,
            "stage6_provenance_sha256": STAGE6_PROVENANCE_SHA256,
            "stage5_evidence_manifest_sha256": STAGE5_EVIDENCE_MANIFEST_SHA256,
            "stage6_evidence_manifest_sha256": STAGE6_EVIDENCE_MANIFEST_SHA256,
        },
        "activation": {
            "default_enabled": False,
            "reviewed_id_only": True,
            "explicit_lane_parameter_required": True,
            "silent_resume_activation": False,
        },
        "determinism": {
            "validator_version": VALIDATOR_VERSION,
            "runtime_protocol_version": RUNTIME_PROTOCOL_VERSION,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "proposal_pool_contract_version": PROPOSAL_POOL_CONTRACT_VERSION,
            "feature_limits": {
                "forbidden_lengths": list(FROZEN_FORBIDDEN_LENGTHS),
                "witness_sample_cap": FROZEN_WITNESS_SAMPLE_CAP,
                "cycle_node_budget": FROZEN_CYCLE_NODE_BUDGET,
                "distance_query_budget": FROZEN_DISTANCE_QUERY_BUDGET,
                "local_risk_budget": FROZEN_LOCAL_RISK_BUDGET,
            },
            "tie_breaking_rule": TIE_BREAKING_RULE,
            "failure_policy": FAILURE_POLICY,
            "resume_requires_exact_identity": True,
        },
        "authority": {
            "host_owns_legal_pool": True,
            "policy_calls_scorer": False,
            "policy_calls_m4": False,
            "selected_plan_only_scoring": True,
            "m4_is_only_certification_authority": True,
            "no_runtime_fallback": True,
        },
        "security": {
            "accepted_worker": RUNTIME_PROTOCOL_VERSION,
            "filesystem": False,
            "environment": False,
            "subprocess": False,
            "network": False,
            "database": False,
            "dynamic_code": False,
            "inherited_stdin": False,
            "source_path_from_director": False,
        },
        "telemetry": {
            "scope": "bounded_micro_batch",
            "per_proposal_history": False,
            "fields": [
                "policy_call_count",
                "invalid_result_count",
                "timeout_crash_protocol_count",
                "selection_latency_ns_sum",
                "selected_k_counts",
                "selector_counts",
                "tie_count",
                "pool_generation_ns",
                "feature_computation_ns",
            ],
        },
        "rollback": {
            "new_lanes_default_disabled": True,
            "historical_evidence_rewritten": False,
            "checkpoint_readable": True,
            "migration": "additive_or_online_backup_restore; no_downgrade",
        },
        "change_surface": {
            "heg": [
                "src/sglab/targets/erdos_gyarfas.py",
                "src/sglab/targets/base.py",
                "src/sglab/research/catalog.py",
                "src/sglab/research/validation.py",
                "src/sglab/research/lanes.py",
                "src/sglab/research/recovery.py",
                "src/sglab/research/store.py",
                "src/sglab/db.py",
                "tests/test_search.py",
                "tests/test_lanes.py",
                "tests/test_score_worker.py",
                "tests/test_certification.py",
            ],
            "mutation_forge_reference_only": [
                "src/mutation_forge/stage7_heg_bridge/",
                "fixtures/stage7/",
            ],
        },
    }


def contract_hash() -> str:
    return canonical_json_hash(contract_payload())


def as_json(value: object) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], value)
