"""Read-only, dependency-free forensic audit of preserved Stage 5 evidence.

The auditor deliberately does not import any Stage 5 implementation.  It treats
the evidence directory as untrusted bytes and returns a machine-readable report.
"""
# ruff: noqa: E501, E702
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

POLICIES = (
    "program-d5ad1c8203e0d9f25f03aabd",
    "candidate-slot-04",
    "random",
    "structural",
)
COUNTERS = ("model_calls", "app_server_calls", "oracle_score_calls", "runtime_network_calls")
TIMING_KEYS = {"timing_ns", "first_improvement_ns", "ranker_elapsed_ns", "selected_scoring_ns", "pool_legality_ns", "pool_feature_ns", "elapsed_ns", "started_at", "finished_at", "timing", "timing_profile", "elapsed_seconds"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _tree_sha256(root: Path) -> str:
    """Hash paths and bytes, including the evidence manifest itself."""
    digest = hashlib.sha256()
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def timing_stripped_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): timing_stripped_projection(v) for k, v in value.items() if str(k) not in TIMING_KEYS}
    if isinstance(value, (list, tuple)):
        return [timing_stripped_projection(item) for item in value]
    return value


def build_sha256_manifest(root: str | Path, *, exclude: Iterable[str] = ("evidence-manifest.sha256",)) -> list[str]:
    """Return deterministic ``sha256  relative/path`` lines for regular files."""
    base = Path(root).resolve()
    excluded = {str(item).replace("\\", "/") for item in exclude}
    lines: list[str] = []
    for path in sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        lines.append(f"{_sha256(path)}  {relative}")
    return lines


def manifest_bytes(root: str | Path, *, exclude: Iterable[str] = ("evidence-manifest.sha256",)) -> bytes:
    return ("\n".join(build_sha256_manifest(root, exclude=exclude)) + "\n").encode("utf-8")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assertion(report: dict[str, Any], name: str, ok: bool, evidence: Iterable[str], details: Any = None) -> None:
    record: dict[str, Any] = {"assertion": name, "ok": bool(ok), "evidence": sorted({str(x) for x in evidence})}
    if details is not None:
        record["details"] = details
    report["assertions"][name] = record
    if not ok:
        report["findings"].append({"assertion": name, "evidence": record["evidence"], "details": details})


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _copy_audit(source: Path, destination: Path) -> tuple[bool, str | None]:
    if source.resolve() == destination.resolve():
        return False, "audit copy must differ from source"
    if destination.exists():
        identical = _tree_sha256(source) == _tree_sha256(destination)
        return identical, None if identical else "existing audit copy differs from source"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
    try:
        shutil.copytree(source, temporary / source.name)
        os.replace(temporary / source.name, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return True, None


def _load_pass(root: Path, pass_name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    summaries = sorted(root.glob(f"*-{pass_name}-summary.json"))
    states = sorted(root.glob(f"*-{pass_name}-state.json"))
    summary = _json(summaries[0]) if len(summaries) == 1 else None
    state = _json(states[0]) if len(states) == 1 else None
    records: list[dict[str, Any]] = []
    if summary is None:
        errors.append(f"expected one {pass_name} summary")
        return summary, state, records, errors
    entries = summary.get("shards", [])
    if not isinstance(entries, list) or len(entries) != 24:
        errors.append("summary does not list 24 shards")
        return summary, state, records, errors
    expected = summary.get("manifest_episode_ids", [])
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"shard {index} entry is not an object"); continue
        path = root / str(entry.get("path", ""))
        if not path.is_file():
            errors.append(f"missing shard {index}: {path}"); continue
        if entry.get("file_sha256") != _sha256(path):
            errors.append(f"shard {index} hash mismatch")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                shard_rows = [json.loads(line) for line in handle if line.strip()]
            if len(shard_rows) != int(entry.get("record_count", -1)):
                errors.append(f"shard {index} record count mismatch")
            records.extend(row for row in shard_rows if isinstance(row, dict))
        except Exception as exc:  # malformed compressed evidence is a finding, not a crash
            errors.append(f"shard {index} unreadable: {type(exc).__name__}")
    if isinstance(expected, list) and sorted(str(x) for x in expected) != sorted(str(r.get("episode_id")) for r in records):
        errors.append("manifest roster differs from shard rows")
    return summary, state, records, errors


def audit_stage5(
    evidence: str | Path,
    audit_copy: str | Path | None = None,
    project_repo: str | Path | None = None,
    heg_repo: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit a preserved Stage 5 evidence tree without mutating it."""
    source = Path(evidence).resolve()
    report: dict[str, Any] = {"status": "failed", "ok": False, "source_path": str(source), "source_sha256": None, "audit_path": None, "audit_sha256": None, "inventory": [], "assertions": {}, "findings": [], "errors": []}
    if not source.is_dir():
        report["errors"].append(f"evidence directory does not exist: {source}")
        return report
    report["source_sha256"] = _tree_sha256(source)
    report["inventory"] = [
        {
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(
            (item for item in source.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(source).as_posix(),
        )
    ]
    manifest_path = source / "evidence-manifest.sha256"
    declared_lines: list[str] = []
    if manifest_path.is_file():
        try:
            declared_lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            parsed: dict[str, str] = {}
            for line in declared_lines:
                digest, relative = line.split(None, 1); parsed[relative.strip()] = digest
            actual = {line.split(None, 1)[1]: line.split(None, 1)[0] for line in build_sha256_manifest(source)}
            _assertion(report, "evidence_manifest_hashes", parsed == actual, [str(manifest_path)], {"missing": sorted(set(actual) - set(parsed)), "extra": sorted(set(parsed) - set(actual))})
            _assertion(report, "evidence_manifest_sorted", declared_lines == sorted(declared_lines, key=lambda line: line.split(None, 1)[1]), [str(manifest_path)])
            if expected_manifest_sha256 is not None:
                _assertion(
                    report,
                    "evidence_manifest_sha256_expected",
                    _sha256(manifest_path) == expected_manifest_sha256,
                    [str(manifest_path)],
                    {"expected": expected_manifest_sha256, "actual": _sha256(manifest_path)},
                )
        except Exception as exc:
            _assertion(report, "evidence_manifest_hashes", False, [str(manifest_path)], f"{type(exc).__name__}: {exc}")
    else:
        _assertion(report, "evidence_manifest_hashes", False, [str(source)], "manifest missing")
    if audit_copy is not None:
        destination = Path(audit_copy).resolve()
        copied, copy_error = _copy_audit(source, destination)
        report["audit_path"] = str(destination)
        report["audit_sha256"] = _tree_sha256(destination) if destination.is_dir() else None
        _assertion(report, "byte_identical_audit_copy", copied and copy_error is None and report["source_sha256"] == report["audit_sha256"], [str(source), str(destination)], copy_error)
        _assertion(
            report,
            "audit_copy_sorted_manifest_matches_source",
            destination.is_dir() and build_sha256_manifest(source) == build_sha256_manifest(destination),
            [str(source), str(destination)],
        )

    analysis_root = Path(str(report["audit_path"])).resolve() if report.get("audit_path") else source

    freeze_path = analysis_root / "stage5-generalization-freeze-v1.json"
    terminal_path = analysis_root / "stage5-terminal.json"
    summary_path = analysis_root / "stage5-summary.json"
    freeze = _json(freeze_path) if freeze_path.is_file() else {}
    terminal = _json(terminal_path) if terminal_path.is_file() else {}
    final_summary = _json(summary_path) if summary_path.is_file() else {}
    _assertion(report, "freeze_integrity", bool(freeze) and freeze.get("schema_version") == "stage5.generalization.freeze.v1" and freeze.get("stage5_results_observed") is False and freeze.get("stage6_started") is False and freeze.get("provider_calls_allowed") is False and freeze.get("freeze_sha256") == hashlib.sha256(_canonical({k: v for k, v in freeze.items() if k != "freeze_sha256"})).hexdigest(), [str(freeze_path)])
    _assertion(report, "terminal_and_summary_integrity", bool(terminal) and terminal == final_summary and terminal.get("decision") == "GO_TO_STAGE_6", [str(terminal_path), str(summary_path)])

    passes: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[str]]] = {}
    for name in ("primary", "replay"):
        passes[name] = _load_pass(analysis_root / name, name)
        summary, state, records, errors = passes[name]
        _assertion(report, f"{name}_24_shards_and_roster", summary is not None and not errors and len(records) == 1536 and len({str(r.get("episode_id")) for r in records}) == 1536, [str(analysis_root / name)], errors or {"rows": len(records)})
        _assertion(report, f"{name}_state_complete", state is not None and state.get("status") == "completed" and len(state.get("completed_shards", {})) == 24, [str(analysis_root / name / "*-state.json")])
        if summary:
            _assertion(report, f"{name}_summary_schema", summary.get("status") == "completed" and summary.get("record_count") == 1536 and summary.get("shard_count") == 24 and summary.get("episodes_per_shard") == 64 and summary.get("policy_ids") == list(POLICIES), [str(analysis_root / name)])

    primary_records, replay_records = passes["primary"][2], passes["replay"][2]
    primary_ids, replay_ids = {str(r.get("episode_id")) for r in primary_records}, {str(r.get("episode_id")) for r in replay_records}
    _assertion(report, "cross_pass_pairing", primary_ids == replay_ids and len(primary_ids) == 1536, [str(source / "primary"), str(source / "replay")], {"primary_only": sorted(primary_ids - replay_ids), "replay_only": sorted(replay_ids - primary_ids)})
    policy_ok = True; metrics_ok = True; counters_ok = True; budgets_ok = True; graph_ok = True; relabel_ok = True
    for row in primary_records + replay_records:
        policy_rows = row.get("policies")
        policy_ok &= isinstance(policy_rows, Mapping) and set(policy_rows) == set(POLICIES)
        metrics = row.get("metrics_input")
        metrics_ok &= isinstance(metrics, Mapping) and set(metrics.get("policies", {})) == set(POLICIES) and all(k in metrics for k in ("episode_id", "order", "graph_seed", "relabeling_seed", "policy_seed", "horizon"))
        counters_ok &= all(row.get(counter, 1) == 0 for counter in COUNTERS) and row.get("policy_failures", 1) == 0
        budgets_ok &= row.get("evaluation_count") == 128 and row.get("selected_score_calls") == 128 and row.get("horizon") == 32
        proof = row.get("relabel_proof")
        graph_ok &= row.get("invalid_graphs") == 0 and isinstance(proof, Mapping) and proof.get("base_graph_hash") == row.get("base_graph_hash") and proof.get("relabeled_graph_hash") == row.get("relabeled_graph_hash") and proof.get("canonical_unlabeled_hash") == row.get("canonical_unlabeled_hash")
        perm = proof.get("permutation") if isinstance(proof, Mapping) else None
        relabel_ok &= (
            isinstance(perm, list)
            and len(perm) == int(row.get("order", 0))
            and sorted(perm) == list(range(int(row.get("order", 0))))
            and isinstance(proof, Mapping)
            and proof.get("algorithm") == "fisher-yates-sha256-v1"
        )
    refs = [str(analysis_root / "primary"), str(analysis_root / "replay")]
    _assertion(report, "four_policy_roster_and_metrics_input", policy_ok and metrics_ok, refs)
    _assertion(report, "zero_counters_and_equal_budgets", counters_ok and budgets_ok, refs)
    _assertion(report, "graph_validity_and_relabel_proofs", graph_ok and relabel_ok, refs)
    _assertion(report, "provider_free_execution", counters_ok and freeze.get("provider_calls_allowed") is False, [str(freeze_path), *refs])
    if primary_records and replay_records:
        by_p = {str(r.get("episode_id")): timing_stripped_projection(r) for r in primary_records}
        by_r = {str(r.get("episode_id")): timing_stripped_projection(r) for r in replay_records}
        _assertion(report, "timing_only_replay_projection", by_p == by_r, refs)

    frozen_policies = freeze.get("policies", {}) if isinstance(freeze, Mapping) else {}
    provenance_ok = True; provenance_details: list[str] = []
    for policy in POLICIES:
        item = frozen_policies.get(policy, {}) if isinstance(frozen_policies, Mapping) else {}
        path = Path(str(item.get("source_path", "")))
        if not path.is_file() or item.get("source_sha256") != _sha256(path):
            provenance_ok = False; provenance_details.append(policy)
    # Defaults are intentionally local and read-only: the caller's checkout is
    # the project repository, with a sibling ``heg`` checkout when present.
    project = Path(project_repo).resolve() if project_repo else Path.cwd().resolve()
    impl_commit = freeze.get("implementation_commit")
    if impl_commit and _git(project, "cat-file", "-e", f"{impl_commit}^{{commit}}") is None:
        provenance_ok = False; provenance_details.append("implementation_commit")
    heg = Path(heg_repo).resolve() if heg_repo else project.parent / "heg"
    heg_commit = freeze.get("heg_commit")
    if heg_commit and _git(heg, "cat-file", "-e", f"{heg_commit}^{{commit}}") is None:
        provenance_details.append("heg_commit_unavailable")
    _assertion(report, "provenance_metadata", provenance_ok, [str(freeze_path)], provenance_details)
    report["ok"] = all(item["ok"] for item in report["assertions"].values())
    report["status"] = "passed" if report["ok"] else "failed"
    return report


# Friendly aliases keep the helper usable from small audit scripts while the
# canonical name remains explicit about SHA-256 output.
build_manifest = build_sha256_manifest
generate_manifest = build_sha256_manifest

__all__ = [
    "audit_stage5",
    "build_sha256_manifest",
    "build_manifest",
    "generate_manifest",
    "manifest_bytes",
    "timing_stripped_projection",
]
