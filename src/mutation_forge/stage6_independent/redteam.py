# ruff: noqa: E501, E701, F841, UP034
"""Small, deterministic red-team corpus for the Stage 6 evidence verifier.

The red-team corpus deliberately does not import a Stage 5 analysis module.  It
uses a tiny paired data set and the same rules expected of an independent
evidence reader: exact roster and shard accounting, paired weights, canonical
metric/bootstrap values, and an explicit provenance envelope.  Timing and
record order are transport details and are therefore excluded from the
canonical projection.

``run_redteam`` is useful both in tests and from an audit driver.  It returns
JSON-compatible dictionaries only; no random state, wall clock, or external
files are consulted unless a fixture path is supplied.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "stage6.redteam.v1"
FIXTURE_SCHEMA = "stage6.synthetic.evidence.v1"
CASE_NAMES: tuple[str, ...] = (
    "missing_record",
    "duplicate_record",
    "extra_record",
    "modified_record",
    "unpaired_record",
    "misweighted_record",
    "relabeled_record",
    "shard_missing",
    "shard_duplicate",
    "shard_wrong_roster",
    "record_hash",
    "manifest_hash",
    "metrics_hash",
    "metrics_value",
    "bootstrap_seed",
    "bootstrap_samples",
    "bootstrap_interval",
    "bootstrap_values",
    "provenance_commit",
    "provenance_dataset",
    "provenance_config",
    "provenance_dirty",
    "provenance_schema",
    "label_sensitive_relabeling",
    "fraction_float_drift",
)

# Metamorphic transformations are intentionally outside CASE_NAMES: accepting
# them is itself an assertion made by the red-team harness.
METAMORPHIC_CASES: tuple[str, ...] = (
    "shard_permutation",
    "record_permutation",
    "harmless_timing_change",
    "equivalent_relabeling",
)

_TIMING_FIELDS = frozenset(
    {
        "timing_ns",
        "elapsed_ns",
        "started_at",
        "finished_at",
        "timing",
        "timing_profile",
        "elapsed_seconds",
    }
)
_EXPECTED_PROVENANCE = {
    "implementation_commit": "stage6-impl-0001",
    "heg_commit": "heg-0001",
    "dataset_manifest_sha256": "dataset-0001",
    "config_sha256": "config-0001",
    "dirty": False,
}


def _fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        # Float values are accepted only as exact decimal text.  This makes a
        # 0.3333333 substitution distinguishable from the frozen 1/3 value.
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise ValueError(f"unsupported number {value!r}")


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _canonical(value: Any, *, strip_timing: bool = False, strip_labels: bool = False) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted((str(k) for k in value)):
            if strip_timing and key in _TIMING_FIELDS:
                continue
            if strip_labels and key in {"vertex_labels", "label_names"}:
                continue
            result[key] = _canonical(value[key], strip_timing=strip_timing, strip_labels=strip_labels)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical(item, strip_timing=strip_timing, strip_labels=strip_labels) for item in value]
    if isinstance(value, Fraction):
        return {"__fraction__": _fraction_text(value)}
    return value


def _digest(value: Any, **kwargs: Any) -> str:
    encoded = json.dumps(_canonical(value, **kwargs), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _record_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove self-referential integrity metadata before hashing a record."""
    return {str(key): item for key, item in value.items() if str(key) != "record_sha256"}


def _records_projection(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_record_projection(row) for row in sorted(values, key=lambda row: str(row.get("record_id", "")))]


def _shards_projection(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for shard in values:
        row = dict(shard)
        row["record_ids"] = sorted(str(item) for item in row.get("record_ids", []))
        projected.append(row)
    return sorted(projected, key=lambda row: str(row.get("shard_id", "")))


def make_fixture() -> dict[str, Any]:
    """Return a fresh, valid synthetic evidence fixture."""
    records = [
        {
            "record_id": "e01-A",
            "episode_id": "e01",
            "pair_id": "e01",
            "policy_id": "A",
            "graph_id": "g01",
            "vertex_labels": ["a", "b", "c"],
            "canonical_graph_id": "triangle:g01",
            "label_sensitive": "role-a",
            "score": "1/2",
            "weight": "1",
            "timing_ns": 100,
        },
        {
            "record_id": "e01-B",
            "episode_id": "e01",
            "pair_id": "e01",
            "policy_id": "B",
            "graph_id": "g01",
            "vertex_labels": ["a", "b", "c"],
            "canonical_graph_id": "triangle:g01",
            "label_sensitive": "role-a",
            "score": "1/4",
            "weight": "1",
            "timing_ns": 120,
        },
        {
            "record_id": "e02-A",
            "episode_id": "e02",
            "pair_id": "e02",
            "policy_id": "A",
            "graph_id": "g02",
            "vertex_labels": ["a", "b", "c"],
            "canonical_graph_id": "triangle:g02",
            "label_sensitive": "role-a",
            "score": "3/4",
            "weight": "1",
            "timing_ns": 140,
        },
        {
            "record_id": "e02-B",
            "episode_id": "e02",
            "pair_id": "e02",
            "policy_id": "B",
            "graph_id": "g02",
            "vertex_labels": ["a", "b", "c"],
            "canonical_graph_id": "triangle:g02",
            "label_sensitive": "role-a",
            "score": "1/2",
            "weight": "1",
            "timing_ns": 160,
        },
    ]
    shards = [
        {"shard_id": "shard-00", "record_ids": ["e01-A", "e01-B"], "weight": "1"},
        {"shard_id": "shard-01", "record_ids": ["e02-A", "e02-B"], "weight": "1"},
    ]
    fixture: dict[str, Any] = {
        "schema_version": FIXTURE_SCHEMA,
        "manifest": {
            "manifest_id": "synthetic-stage6-v1",
            "expected_record_ids": [row["record_id"] for row in records],
            "shard_ids": [row["shard_id"] for row in shards],
            "label_mode": "isomorphism-invariant",
        },
        "records": records,
        "shards": shards,
        "metrics": {
            "policy_means": {"A": "5/8", "B": "3/8"},
            "paired_delta": "1/4",
        },
        "bootstrap": {
            "seed": 1729,
            "samples": 8,
            "values": ["1/4"] * 8,
            "interval": ["1/4", "1/4"],
        },
        "provenance": dict(_EXPECTED_PROVENANCE),
    }
    _attach_integrity(fixture)
    return fixture


def make_label_sensitive_fixture() -> dict[str, Any]:
    """Return the same tiny corpus with labels declared semantically fixed.

    This fixture is useful for proving that the isomorphism-invariant relabeling
    exemption is not accidentally applied to a label-sensitive metric.
    """
    fixture = make_fixture()
    fixture["manifest"]["label_mode"] = "label-sensitive"
    return fixture


def _attach_integrity(fixture: dict[str, Any]) -> None:
    records = fixture["records"]
    for row in records:
        row["record_sha256"] = _digest(_record_projection(row), strip_timing=True, strip_labels=True)
    fixture["manifest"]["records_sha256"] = _digest(_records_projection(records), strip_timing=True, strip_labels=True)
    fixture["manifest"]["shards_sha256"] = _digest(_shards_projection(fixture["shards"]))
    fixture["metrics"]["sha256"] = _digest({k: v for k, v in fixture["metrics"].items() if k != "sha256"})
    fixture["bootstrap"]["sha256"] = _digest({k: v for k, v in fixture["bootstrap"].items() if k != "sha256"})


def _load(value: Any) -> dict[str, Any]:
    if value is None:
        return make_fixture()
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    path = Path(value)
    if path.is_dir():
        for name in ("evidence.json", "base.json", "fixture.json"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
        else:
            # A fixture root may only contain the corpus manifest; generate the
            # canonical tiny fixture in that case.
            return make_fixture()
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _records_by_id(fixture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = fixture.get("records")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("records must be a list")
    return {str(row.get("record_id")): dict(row) for row in rows if isinstance(row, Mapping)}


def verify_fixture(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Verify a fixture and return structured diagnostics (never raise)."""
    errors: list[dict[str, Any]] = []
    try:
        fixture = _load(value)
        if fixture.get("schema_version") != FIXTURE_SCHEMA:
            errors.append({"code": "schema", "path": "$.schema_version"})
        manifest = fixture.get("manifest")
        records = fixture.get("records")
        shards = fixture.get("shards")
        if not isinstance(manifest, Mapping):
            errors.append({"code": "manifest_missing", "path": "$.manifest"})
            manifest = {}
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            errors.append({"code": "records_missing", "path": "$.records"})
            records = []
        if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
            errors.append({"code": "shards_missing", "path": "$.shards"})
            shards = []
        rows = [dict(row) for row in records if isinstance(row, Mapping)]
        ids = [str(row.get("record_id", "")) for row in rows]
        expected_ids = [str(item) for item in manifest.get("expected_record_ids", [])]
        if len(ids) != len(set(ids)):
            errors.append({"code": "duplicate_record", "path": "$.records"})
        if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
            errors.append({"code": "record_roster", "path": "$.manifest.expected_record_ids", "missing": sorted(set(expected_ids) - set(ids)), "extra": sorted(set(ids) - set(expected_ids))})
        by_id = {row.get("record_id"): row for row in rows}
        # Record and aggregate hashes ignore only timing and vertex-label order.
        for row in rows:
            expected_hash = row.get("record_sha256")
            actual_hash = _digest(_record_projection(row), strip_timing=True, strip_labels=True)
            if expected_hash != actual_hash:
                errors.append({"code": "record_modified", "path": f"$.records[{row.get('record_id')}].record_sha256"})
        if manifest.get("records_sha256") != _digest(_records_projection(rows), strip_timing=True, strip_labels=True):
            errors.append({"code": "records_hash", "path": "$.manifest.records_sha256"})
        # Shard identities and record membership are exact, but shard/record
        # ordering is deliberately not significant.
        shard_ids = [str(shard.get("shard_id", "")) for shard in shards if isinstance(shard, Mapping)]
        expected_shards = [str(item) for item in manifest.get("shard_ids", [])]
        if len(shard_ids) != len(set(shard_ids)):
            errors.append({"code": "duplicate_shard", "path": "$.shards"})
        if set(shard_ids) != set(expected_shards) or len(shard_ids) != len(expected_shards):
            errors.append({"code": "shard_roster", "path": "$.manifest.shard_ids"})
        memberships: list[str] = []
        for shard in shards:
            if not isinstance(shard, Mapping):
                continue
            member_ids = [str(item) for item in shard.get("record_ids", [])]
            memberships.extend(member_ids)
            if int(shard.get("weight", 1)) != 1:
                errors.append({"code": "misweighted_shard", "path": f"$.shards[{shard.get('shard_id')}]"})
        if len(memberships) != len(set(memberships)) or set(memberships) != set(ids):
            errors.append({"code": "shard_membership", "path": "$.shards[*].record_ids"})
        if manifest.get("shards_sha256") != _digest(_shards_projection(shards)):
            errors.append({"code": "shards_hash", "path": "$.manifest.shards_sha256"})
        # Pairing, policy weights, graph identity and label-sensitive semantics.
        by_pair: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            pair = str(row.get("pair_id", ""))
            by_pair.setdefault(pair, []).append(row)
            if row.get("label_sensitive") != "role-a":
                errors.append({"code": "label_sensitive_relabel", "path": f"$.records[{row.get('record_id')}].label_sensitive"})
            if manifest.get("label_mode") == "label-sensitive" and row.get("vertex_labels") != ["a", "b", "c"]:
                errors.append({"code": "label_sensitive_fixture", "path": f"$.records[{row.get('record_id')}].vertex_labels"})
            try:
                if _fraction(row.get("weight")) != 1:
                    errors.append({"code": "misweighted_record", "path": f"$.records[{row.get('record_id')}].weight"})
            except Exception:
                errors.append({"code": "invalid_weight", "path": f"$.records[{row.get('record_id')}].weight"})
        for pair, pair_rows in by_pair.items():
            if {str(row.get("policy_id")) for row in pair_rows} != {"A", "B"} or len(pair_rows) != 2:
                errors.append({"code": "unpaired_record", "path": f"$.pair[{pair}]"})
            if len({row.get("graph_id") for row in pair_rows}) != 1:
                errors.append({"code": "pair_graph_mismatch", "path": f"$.pair[{pair}]"})
        metrics = fixture.get("metrics")
        if not isinstance(metrics, Mapping):
            errors.append({"code": "metrics_missing", "path": "$.metrics"})
        else:
            metric_core = {k: v for k, v in metrics.items() if k != "sha256"}
            if metrics.get("sha256") != _digest(metric_core):
                errors.append({"code": "metrics_hash", "path": "$.metrics.sha256"})
            try:
                means: dict[str, Fraction] = {}
                for policy in ("A", "B"):
                    vals = [_fraction(row["score"]) for row in rows if row.get("policy_id") == policy]
                    means[policy] = sum(vals, Fraction(0)) / len(vals)
                if any(_fraction(metrics["policy_means"][key]) != means[key] for key in means):
                    errors.append({"code": "metrics_value", "path": "$.metrics.policy_means"})
                deltas = [
                    next(_fraction(row["score"]) for row in pair_rows if row.get("policy_id") == "A")
                    - next(_fraction(row["score"]) for row in pair_rows if row.get("policy_id") == "B")
                    for pair_rows in by_pair.values() if len(pair_rows) == 2
                ]
                if _fraction(metrics["paired_delta"]) != sum(deltas, Fraction(0)) / len(deltas):
                    errors.append({"code": "paired_metric", "path": "$.metrics.paired_delta"})
            except Exception:
                errors.append({"code": "metrics_invalid", "path": "$.metrics"})
        bootstrap = fixture.get("bootstrap")
        if not isinstance(bootstrap, Mapping):
            errors.append({"code": "bootstrap_missing", "path": "$.bootstrap"})
        else:
            core = {k: v for k, v in bootstrap.items() if k != "sha256"}
            if bootstrap.get("sha256") != _digest(core):
                errors.append({"code": "bootstrap_hash", "path": "$.bootstrap.sha256"})
            try:
                values = [_fraction(v) for v in bootstrap["values"]]
                if int(bootstrap["samples"]) != 8 or len(values) != 8 or any(v != Fraction(1, 4) for v in values) or [_fraction(v) for v in bootstrap["interval"]] != [Fraction(1, 4), Fraction(1, 4)] or int(bootstrap["seed"]) != 1729:
                    errors.append({"code": "bootstrap_invalid", "path": "$.bootstrap"})
            except Exception:
                errors.append({"code": "bootstrap_invalid", "path": "$.bootstrap"})
        provenance = fixture.get("provenance")
        if not isinstance(provenance, Mapping) or dict(provenance) != _EXPECTED_PROVENANCE:
            errors.append({"code": "provenance_invalid", "path": "$.provenance"})
    except Exception as exc:  # malformed input is a failed verification
        errors.append({"code": "malformed", "path": "$", "message": f"{type(exc).__name__}: {exc}"})
    return {"exact": not errors, "status": "verified" if not errors else "rejected", "errors": errors}


def tamper_fixture(value: Mapping[str, Any] | None, case: str) -> dict[str, Any]:
    """Apply one named deterministic mutation to a fixture."""
    fixture = copy.deepcopy(dict(value) if value is not None else make_fixture())
    rows = fixture["records"]
    if case == "missing_record": rows.pop()
    elif case == "duplicate_record": rows.append(copy.deepcopy(rows[0]))
    elif case == "extra_record": rows.append({**copy.deepcopy(rows[0]), "record_id": "e99-X"})
    elif case in {"modified_record", "record_hash"}: rows[0]["score"] = "7/8"
    elif case == "unpaired_record": rows[0]["pair_id"] = "e99"
    elif case == "misweighted_record": rows[0]["weight"] = "2"
    elif case == "relabeled_record": rows[0]["policy_id"] = "C"
    elif case == "shard_missing": fixture["shards"].pop()
    elif case == "shard_duplicate": fixture["shards"].append(copy.deepcopy(fixture["shards"][0]))
    elif case == "shard_wrong_roster": fixture["shards"][0]["record_ids"][0] = "e99-X"
    elif case == "manifest_hash": fixture["manifest"]["records_sha256"] = "0" * 64
    elif case == "metrics_hash": fixture["metrics"]["sha256"] = "0" * 64
    elif case == "metrics_value": fixture["metrics"]["paired_delta"] = "1/3"
    elif case == "bootstrap_seed": fixture["bootstrap"]["seed"] = 1730
    elif case == "bootstrap_samples": fixture["bootstrap"]["samples"] = 9
    elif case == "bootstrap_interval": fixture["bootstrap"]["interval"] = ["0", "1"]
    elif case == "bootstrap_values": fixture["bootstrap"]["values"][0] = "0"
    elif case == "provenance_commit": fixture["provenance"]["implementation_commit"] = "evil"
    elif case == "provenance_dataset": fixture["provenance"]["dataset_manifest_sha256"] = "evil"
    elif case == "provenance_config": fixture["provenance"]["config_sha256"] = "evil"
    elif case == "provenance_dirty": fixture["provenance"]["dirty"] = True
    elif case == "provenance_schema": fixture["schema_version"] = "stage6.synthetic.evidence.v0"
    elif case == "label_sensitive_relabeling": rows[0]["label_sensitive"] = "role-b"
    elif case == "fraction_float_drift": rows[0]["score"] = 0.3333333333
    elif case == "shard_permutation": fixture["shards"] = list(reversed(fixture["shards"]))
    elif case == "record_permutation": fixture["records"] = list(reversed(fixture["records"]))
    elif case == "harmless_timing_change":
        for index, row in enumerate(rows): row["timing_ns"] = 10_000 + index
    elif case == "equivalent_relabeling":
        for row in rows: row["vertex_labels"] = ["x", "y", "z"]
    else:
        raise KeyError(case)
    return fixture


def run_redteam(config: Any = None, evidence: Any = None, fixture_root: str | Path | None = None) -> dict[str, Any]:
    """Run all tamper and metamorphic checks, returning JSON-compatible output."""
    source = evidence if evidence is not None else fixture_root
    if source is None and config is not None:
        source = config.get("evidence") if isinstance(config, Mapping) else getattr(config, "evidence", None)
    baseline = _load(source)
    if not verify_fixture(baseline)["exact"]:
        baseline = make_fixture()
    findings: list[dict[str, Any]] = []
    for case in CASE_NAMES + METAMORPHIC_CASES:
        expected_accept = case in METAMORPHIC_CASES
        result = verify_fixture(tamper_fixture(baseline, case))
        observed_accept = bool(result["exact"])
        passed = observed_accept == expected_accept
        findings.append({
            "case": case,
            "severity": "informational" if expected_accept else ("critical" if case in {"modified_record", "record_hash", "provenance_commit", "provenance_dataset", "provenance_config", "provenance_dirty"} else "high"),
            "evidence": {"expected_accept": expected_accept, "observed_accept": observed_accept, "errors": result["errors"]},
            "impact": "accepted metamorphic change" if expected_accept else "tampered evidence was rejected" if not observed_accept else "tampered evidence bypassed verification",
            "disposition": "allowed" if expected_accept and passed else "rejected" if not expected_accept and passed else "failure",
            "passed": passed,
        })
    return {"schema_version": SCHEMA_VERSION, "status": "passed" if all(item["passed"] for item in findings) else "failed", "findings": findings}


def write_fixture_set(root: str | Path) -> Path:
    """Write the tiny base fixture and all deterministic mutations to ``root``."""
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    base = make_fixture()
    (destination / "base.json").write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for case in CASE_NAMES + METAMORPHIC_CASES:
        (destination / f"{case}.json").write_text(json.dumps(tamper_fixture(base, case), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def generate_tamper_cases(value: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return all named mutations in memory, keyed by deterministic case name."""
    base = copy.deepcopy(dict(value) if value is not None else make_fixture())
    return {case: tamper_fixture(base, case) for case in CASE_NAMES + METAMORPHIC_CASES}


# Friendly aliases used by callers that describe the operation as generation or
# verification rather than fixture construction.
generate_tamper_fixtures = write_fixture_set
verify_tampering = run_redteam


__all__ = [
    "CASE_NAMES",
    "METAMORPHIC_CASES",
    "FIXTURE_SCHEMA",
    "make_fixture",
    "make_label_sensitive_fixture",
    "run_redteam",
    "tamper_fixture",
    "verify_fixture",
    "write_fixture_set",
    "generate_tamper_cases",
    "generate_tamper_fixtures",
    "verify_tampering",
]
