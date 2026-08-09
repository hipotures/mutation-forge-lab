"""Durable offline replay for an accepted ordinary-Python preview bundle."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3.canonical import canonical_json_bytes

from .preview import load_python_preview_config
from .search import DevelopmentCaseV1
from .search_provider import PythonPanelScientificEvaluator
from .validation import validate_python_policy_source

EVIDENCE_REPLAY_PROTOCOL_ID = "mforge.native.python_m7_evidence_replay.v1"


class EvidenceReplayError(RuntimeError):
    """A durable preview bundle cannot be reproduced exactly."""


def _mapping(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise EvidenceReplayError(f"evidence artifact is not an object: {path}")
    return dict(value)


def _case(value: object) -> DevelopmentCaseV1:
    if not isinstance(value, Mapping):
        raise EvidenceReplayError("development panel case is malformed")
    raw_lengths = value.get("forbidden_lengths")
    if not isinstance(raw_lengths, Sequence) or isinstance(
        raw_lengths, str | bytes
    ):
        raise EvidenceReplayError("development forbidden lengths are malformed")
    return DevelopmentCaseV1(
        case_id=str(value["case_id"]),
        order=int(value["order"]),
        graph_seed=int(value["graph_seed"]),
        policy_seed=int(value["policy_seed"]),
        horizon=int(value["horizon"]),
        witness_cap=int(value["witness_cap"]),
        forbidden_lengths=tuple(int(item) for item in raw_lengths),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if key != "wall_time_ns"
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_stable(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise EvidenceReplayError(f"unsupported replay value: {type(value).__name__}")


def _projection(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    scientific = value.get("scientific_result")
    if not isinstance(scientific, Mapping):
        raise EvidenceReplayError("evaluation omitted scientific result")
    return {
        "protocol_id": cast(JsonValue, value.get("protocol_id")),
        "config": _stable(value.get("config")),
        "program_identity": _stable(value.get("program_identity")),
        "behavior_identity": _stable(value.get("behavior_identity")),
        "external_activity": _stable(value.get("external_activity")),
        "scientific_result": _stable(scientific),
    }


def _projection_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(_projection(value)))


def _file_inventory(root: Path) -> list[dict[str, JsonValue]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_bytes(path.read_bytes()),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def replay_evidence_bundle(
    *,
    config_path: str | Path,
    artifact_root: str | Path,
    expected_report: str | Path | None = None,
) -> dict[str, JsonValue]:
    """Replay every evaluated source and return a canonical comparison report."""

    config = load_python_preview_config(config_path)
    root = config.experiment_root.resolve(strict=True)
    replay_root = Path(artifact_root).resolve()
    if replay_root.is_relative_to(root):
        raise EvidenceReplayError(
            "replay artifacts must be outside the immutable evidence workspace"
        )
    if expected_report is not None and Path(expected_report).resolve().is_relative_to(
        root
    ):
        raise EvidenceReplayError(
            "expected replay report must be outside the immutable evidence workspace"
        )
    protocol = _mapping(root / "protocol.json.gz")
    raw_panel = protocol.get("panel")
    if not isinstance(raw_panel, Sequence) or isinstance(raw_panel, str | bytes):
        raise EvidenceReplayError("protocol omitted the development panel")
    panel = tuple(_case(item) for item in raw_panel)
    final_report = _mapping(root / "m5-report.json.gz")
    if (
        final_report.get("status") != "completed"
        or final_report.get("candidate_count") != 16
    ):
        raise EvidenceReplayError("preview bundle is not terminal and complete")

    backend = HegBackend(config.heg_repo)
    evaluator = PythonPanelScientificEvaluator(
        backend=backend,
        artifact_root=replay_root,
    )
    replays: list[dict[str, JsonValue]] = []
    try:
        for candidate_path in sorted(
            root.glob("generations/generation-*/slot-*/candidate.json.gz")
        ):
            candidate = _mapping(candidate_path)
            if candidate.get("status") != "evaluated":
                continue
            candidate_id = candidate.get("candidate_id")
            source_path_value = candidate.get("source_path")
            if not isinstance(candidate_id, str) or not isinstance(
                source_path_value, str
            ):
                raise EvidenceReplayError("evaluated candidate omitted source identity")
            source_path = (root / source_path_value).resolve(strict=True)
            if not source_path.is_relative_to(root):
                raise EvidenceReplayError("candidate source escapes evidence root")
            source = source_path.read_text(encoding="utf-8")
            validation = validate_python_policy_source(source)
            if (
                not validation.valid
                or validation.identity is None
                or validation.identity.program_hash != candidate.get("program_hash")
                or validation.identity.source_sha256
                != candidate.get("source_sha256")
                or validation.identity.canonical_ast_sha256
                != candidate.get("canonical_ast_sha256")
            ):
                raise EvidenceReplayError(
                    f"candidate source identity changed: {candidate_id}"
                )
            case_reports: list[dict[str, JsonValue]] = []
            for case in panel:
                retained = _mapping(
                    candidate_path.parent
                    / "evaluations"
                    / f"{case.case_id}.json.gz"
                )
                replayed = dict(
                    evaluator.evaluate(
                        source=source,
                        case=case,
                        candidate_id=candidate_id,
                    )
                )
                retained_sha = _projection_sha256(retained)
                replayed_sha = _projection_sha256(replayed)
                if retained_sha != replayed_sha:
                    raise EvidenceReplayError(
                        f"semantic replay changed: {candidate_id}/{case.case_id}"
                    )
                case_reports.append(
                    {
                        "case_id": case.case_id,
                        "semantic_projection_sha256": retained_sha,
                        "semantic_trace_hash": cast(
                            Mapping[str, Any],
                            retained["scientific_result"],
                        )["semantic_trace_hash"],
                    }
                )
            replays.append(
                {
                    "candidate_id": candidate_id,
                    "generation": cast(JsonValue, candidate.get("generation")),
                    "slot": cast(JsonValue, candidate.get("slot")),
                    "kind": cast(JsonValue, candidate.get("kind")),
                    "parent_candidate_id": cast(
                        JsonValue, candidate.get("parent_candidate_id")
                    ),
                    "program_hash": validation.identity.program_hash,
                    "behavior_signature": cast(
                        JsonValue, candidate.get("behavior_signature")
                    ),
                    "source_sha256": validation.identity.source_sha256,
                    "canonical_ast_sha256": (
                        validation.identity.canonical_ast_sha256
                    ),
                    "cases": cast(JsonValue, case_reports),
                }
            )
    finally:
        backend.close()

    inventory = _file_inventory(root)
    payload: dict[str, JsonValue] = {
        "protocol_id": EVIDENCE_REPLAY_PROTOCOL_ID,
        "workspace_protocol_id": cast(JsonValue, protocol.get("protocol_id")),
        "panel": [item.as_dict() for item in panel],
        "panel_hash": cast(JsonValue, protocol.get("panel_hash")),
        "generation_manifest_hashes": cast(
            JsonValue, final_report.get("generation_manifest_hashes")
        ),
        "search_memory_hashes": cast(
            JsonValue, final_report.get("search_memory_hashes")
        ),
        "candidate_status_counts": cast(
            JsonValue, final_report.get("candidate_status_counts")
        ),
        "lineage": cast(JsonValue, final_report.get("lineage")),
        "exact_verification": cast(
            JsonValue, final_report.get("exact_verification")
        ),
        "final_status": cast(JsonValue, final_report.get("status")),
        "stop_reason": cast(JsonValue, final_report.get("stop_reason")),
        "evaluated_candidate_count": len(replays),
        "replays": cast(JsonValue, replays),
        "workspace_file_count": len(inventory),
        "workspace_inventory_sha256": _sha256_bytes(
            canonical_json_bytes(inventory)
        ),
        "all_semantic_replays_match": True,
    }
    comparison_sha = _sha256_bytes(canonical_json_bytes(payload))
    report = {**payload, "sha256": comparison_sha}
    if expected_report is not None:
        expected = _mapping(Path(expected_report))
        if expected != report:
            raise EvidenceReplayError("post-cleanup replay differs from baseline")
    return report


def write_evidence_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write one deterministic durable replay report."""

    write_json(path, dict(report))


__all__ = [
    "EVIDENCE_REPLAY_PROTOCOL_ID",
    "EvidenceReplayError",
    "replay_evidence_bundle",
    "write_evidence_report",
]
