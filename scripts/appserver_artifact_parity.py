"""Freeze and verify the Native v2 App Server provider-turn artifact contract."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mutation_forge.experiment.artifacts import (
    TurnArtifactStore,
    render_generated_policy_markdown,
)
from mutation_forge.experiment.json_io import compress_json_bytes, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "native_v2_appserver_artifact_contract.json"
)
CONTRACT_SCHEMA_VERSION = "mforge.native-v2-appserver-artifact-contract.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}\n?$")


class ParityError(RuntimeError):
    """The generated or real provider artifacts do not match the frozen contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compress_json_bytes(_canonical(value) + b"\n"))


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(record) + b"\n" for record in records))


def _response(*, valid_source: bool = True) -> dict[str, Any]:
    return {
        "schema_version": "mforge.native.generated_policy.v1",
        "source": (
            "def priority(ctx, proposal):\n    return 0\n"
            if valid_source
            else "def priority(:\n"
        ),
        "design_summary": "A deterministic fixture policy.",
        "hypothesis": "The artifact contract is reproducible.",
        "used_fields": ["proposal.local_c4_risk"],
        "assumptions": ["The host supplies legal proposals."],
        "expected_failure_modes": ["The fixture policy may underperform."],
    }


def _request(*, phase: str, idempotency_key: str) -> dict[str, Any]:
    output_schema = {
        "$id": "mforge.native.generated-policy.v1",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Native generated policy",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "source",
            "design_summary",
            "hypothesis",
            "used_fields",
            "assumptions",
            "expected_failure_modes",
        ],
        "properties": {
            "schema_version": {"const": "mforge.native.generated_policy.v1"},
            "source": {"type": "string"},
        },
    }
    return {
        "archive_context": [],
        "artifact_dir": "[FIXTURE_PATH]",
        "artifact_prefix": "slot-00",
        "artifact_root": "[FIXTURE_PATH]",
        "brief_id": "native-v2-parity",
        "campaign_id": "native-v2-parity",
        "effort": "high",
        "generation": 0,
        "idempotency_key": idempotency_key,
        "max_repairs": 1,
        "model": "gpt-5.6-luna",
        "output_schema": output_schema,
        "parent_id": "native-baseline",
        "parent_metadata": {},
        "parent_source": "def priority(ctx, proposal):\n    return 0\n",
        "phase": phase,
        "prompt": "Return the deterministic Native v2 fixture policy.",
        "prompt_hash": "a" * 64,
        "reasoning_effort": "high",
        "remaining_repairs": 1 if phase == "initial" else 0,
        "repair_attempt": 0 if phase == "initial" else 1,
        "repair_prompt": "Repair the generated policy using the diagnostics.",
        "request_idempotency_key": idempotency_key,
        "search_feedback": {},
        "slot": "slot-00",
        "system_prompt": "Return one generated policy JSON object.",
    }


def _usage(*, final: bool = True) -> dict[str, Any]:
    return {
        "cacheWriteInputTokens": 0,
        "cachedInputTokens": 0,
        "final": final,
        "inputTokens": 2,
        "outputTokens": 3,
        "partial": not final,
        "reasoningOutputTokens": 1,
        "totalTokens": 5,
    }


def _write_transport(
    directory: Path,
    *,
    prefix: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    usage: Mapping[str, Any],
    terminal_status: str,
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    prompt = str(request["prompt"])
    system_prompt = str(request["system_prompt"])
    output_schema = dict(request["output_schema"])
    profile = {
        "approvalPolicy": "never",
        "artifactPrefix": prefix,
        "effort": "high",
        "ephemeral": True,
        "model": "gpt-5.6-luna",
        "protocolAuditSha256": "b" * 64,
        "sandbox": "danger-full-access",
    }
    rpc = [{"id": 1, "result": {"status": terminal_status}}]
    events = [
        {
            "emittedAtMs": 1,
            "method": "turn/completed",
            "params": {"status": terminal_status},
        }
    ]
    stdout = [rpc[0], events[0]]
    wire = [
        {"direction": "client_to_server", "message": {"id": 1}},
        {"direction": "server_to_client", "message": {"id": 1}},
    ]
    diagnostics = [
        {
            "bytes": 64,
            "event": "turn_completed",
            "method": "turn/completed",
            "request": 1,
        }
    ]
    provider_raw = {
        "request_id": 1,
        "response_projection_valid": response is not None,
        "response_text": (
            json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if response is not None
            else ""
        ),
        "thread_id": "thread-fixture",
        "transport_diagnostics": diagnostics,
        "turn_id": "turn-fixture",
        "usage": dict(usage),
    }

    (directory / f"{prefix}.request.md").write_text(prompt, encoding="utf-8")
    _write_json(directory / f"{prefix}.request.json.gz", request)
    (directory / f"{prefix}.system-prompt.md").write_text(
        system_prompt,
        encoding="utf-8",
    )
    _write_json(directory / f"{prefix}.output-schema.json.gz", output_schema)
    _write_json(directory / f"{prefix}.provider-raw.json.gz", provider_raw)
    _write_json(directory / f"{prefix}.codex-profile.json.gz", profile)
    _write_json(directory / f"{prefix}.usage.json.gz", usage)
    _write_json(directory / f"{prefix}.transport-diagnostics.json.gz", diagnostics)
    _write_jsonl(directory / f"{prefix}.codex-rpc.jsonl", rpc)
    _write_jsonl(directory / f"{prefix}.events.jsonl", events)
    _write_jsonl(directory / f"{prefix}.stdout.jsonl", stdout)
    _write_jsonl(directory / f"{prefix}.wire.jsonl", wire)
    (directory / f"{prefix}.stderr.txt").write_text("", encoding="utf-8")
    transcript = b"".join(
        path.read_bytes()
        for path in (
            directory / f"{prefix}.wire.jsonl",
            directory / f"{prefix}.codex-rpc.jsonl",
            directory / f"{prefix}.events.jsonl",
            directory / f"{prefix}.stdout.jsonl",
            directory / f"{prefix}.stderr.txt",
        )
    )
    (directory / f"{prefix}.transcript.sha256").write_text(
        hashlib.sha256(transcript).hexdigest() + "\n",
        encoding="ascii",
    )

    if response is not None:
        response_text = json.dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        (directory / f"{prefix}.response.raw.txt").write_text(
            response_text,
            encoding="utf-8",
        )
        (directory / f"{prefix}.response.md").write_bytes(
            render_generated_policy_markdown(response)
        )
        _write_json(directory / f"{prefix}.response.json.gz", response)

    return sorted(path.name for path in directory.iterdir() if path.name.startswith(prefix))


def _result(
    *,
    response: Mapping[str, Any] | None,
    usage: Mapping[str, Any],
    artifact_refs: Sequence[str],
    validation_valid: bool | None,
    status: str = "completed",
    charged: bool = True,
    uncharged: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "accepted": True,
        "artifact_refs": list(artifact_refs),
        "charged": charged,
        "content": response is not None,
        "provider_thread_id": "thread-fixture",
        "provider_turn_id": "turn-fixture",
        "response_projection_valid": response is not None,
        "status": status,
        "uncharged": uncharged,
        "usage": dict(usage),
        "validation_completed": validation_valid is not None,
    }
    if response is not None:
        value.update(
            {
                "canonical_response": dict(response),
                "identity": {
                    "ast_node_count": 9,
                    "normalized_ast_sha256": "c" * 64,
                    "source_sha256": "d" * 64,
                    "validator_version": "native-v2-fixture",
                },
                "provenance": {
                    "effort": "high",
                    "model": "gpt-5.6-luna",
                    "provider_request_id": 1,
                    "provider_thread_id": "thread-fixture",
                    "provider_turn_id": "turn-fixture",
                    "transport_sha256": "e" * 64,
                },
                "response": dict(response),
                "response_text": json.dumps(
                    response,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "validation": {
                    "errors": [] if validation_valid else ["invalid syntax"],
                    "identity": "d" * 64,
                    "valid": validation_valid,
                },
            }
        )
    if validation_valid is True:
        value.update(
            {
                "behavior": {
                    "identity": "d" * 64,
                    "input_contract": "mforge.native.behavior-input.v1",
                    "probes": [],
                    "schema_version": "mforge.native.behavior.v1",
                    "signature_sha256": "f" * 64,
                    "terminal_status": "completed",
                },
                "metadata_validation": {
                    "declared_used_fields": ["proposal.local_c4_risk"],
                    "errors": [],
                    "extracted_used_fields": ["proposal.local_c4_risk"],
                    "schema_version": "mforge.native.metadata-validation.v1",
                    "status": "valid",
                },
                "worker_telemetry": {
                    "calls": 1,
                    "captured_stderr_bytes": 0,
                    "controls": {},
                    "failures": 0,
                    "max_policy_elapsed_ns": 1,
                    "pid": 1,
                    "protocol_version": "mforge.sandbox.worker.v1",
                    "total_call_wall_seconds": 0.001,
                    "total_policy_elapsed_ns": 1,
                    "usable": True,
                },
            }
        )
    if status != "completed":
        value["error"] = "FixtureTransportError: deterministic failure"
    return value


def _record_turn(
    store: TurnArtifactStore,
    *,
    phase: str,
    prefix: str,
    response: Mapping[str, Any] | None,
    validation_valid: bool | None,
    status: str = "completed",
    charged: bool = True,
    uncharged: bool = False,
    final_usage: bool = True,
) -> Path:
    request_phase = "repair" if phase.startswith("repair-") else "initial"
    request = _request(
        phase=request_phase,
        idempotency_key=f"native-v2-parity:{phase}:{prefix}",
    )
    directory = store.turn_directory(0, "slot-00", phase)
    usage = _usage(final=final_usage)
    refs = _write_transport(
        directory,
        prefix=prefix,
        request=request,
        response=response,
        usage=usage,
        terminal_status=status,
    )
    result = _result(
        response=response,
        usage=usage,
        artifact_refs=refs,
        validation_valid=validation_valid,
        status=status,
        charged=charged,
        uncharged=uncharged,
    )
    store.record_existing_turn(
        directory,
        generation=0,
        slot="slot-00",
        phase=phase,
        request=request,
        result=result,
    )
    store.verify_turn(directory)
    return directory


def generate_fixture(root: Path) -> None:
    """Create the deterministic initial/repair/retry/success/failure corpus."""

    if root.exists() and any(root.iterdir()):
        raise ParityError(f"fixture destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    initial_store = TurnArtifactStore(root / "initial-success" / "artifacts")
    _record_turn(
        initial_store,
        phase="initial",
        prefix="slot-00",
        response=_response(),
        validation_valid=True,
    )

    repair_store = TurnArtifactStore(root / "repair-success" / "artifacts")
    _record_turn(
        repair_store,
        phase="initial",
        prefix="slot-00",
        response=_response(valid_source=False),
        validation_valid=False,
    )
    _record_turn(
        repair_store,
        phase="repair-01",
        prefix="slot-00",
        response=_response(),
        validation_valid=True,
    )

    retry_store = TurnArtifactStore(root / "retry-success" / "artifacts")
    retry_directory = _record_turn(
        retry_store,
        phase="initial",
        prefix="slot-00",
        response=None,
        validation_valid=None,
        status="failed",
        charged=False,
        uncharged=True,
    )
    retry_store.archive_retryable_manifest(retry_directory)
    retry_request = _request(
        phase="initial",
        idempotency_key="native-v2-parity:initial:slot-00.retry-01",
    )
    retry_usage = _usage()
    retry_response = _response()
    retry_refs = _write_transport(
        retry_directory,
        prefix="slot-00.retry-01",
        request=retry_request,
        response=retry_response,
        usage=retry_usage,
        terminal_status="completed",
    )
    retry_result = _result(
        response=retry_response,
        usage=retry_usage,
        artifact_refs=retry_refs,
        validation_valid=True,
    )
    retry_store.record_existing_turn(
        retry_directory,
        generation=0,
        slot="slot-00",
        phase="initial",
        request=retry_request,
        result=retry_result,
    )
    retry_store.verify_turn(retry_directory)

    failure_store = TurnArtifactStore(root / "terminal-failure" / "artifacts")
    _record_turn(
        failure_store,
        phase="initial",
        prefix="slot-00",
        response=None,
        validation_valid=None,
        status="failed",
        charged=True,
        final_usage=False,
    )


def _json_shape(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        shape: dict[str, Any] = {
            "type": "object",
            "keys": sorted(str(key) for key in value),
        }
        if "schema_version" in value:
            shape["schema_version"] = value["schema_version"]
        return shape
    if isinstance(value, list):
        item_keys = sorted(
            {
                str(key)
                for item in value
                if isinstance(item, Mapping)
                for key in item
            }
        )
        return {"type": "array", "item_keys": item_keys}
    return {"type": type(value).__name__}


def _describe_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    result: dict[str, Any] = {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if path.name.endswith(".json.gz"):
        if raw[:2] != b"\x1f\x8b":
            raise ParityError(f"gzip magic is missing: {path}")
        decoded = gzip.decompress(raw)
        value = json.loads(decoded.decode("utf-8"))
        result.update(
            {
                "kind": "gzip-json",
                "gzip_mtime": int.from_bytes(raw[4:8], "little"),
                "json": _json_shape(value),
            }
        )
        return result
    if path.name.endswith(".jsonl"):
        records = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
        result.update(
            {
                "kind": "jsonl",
                "item_keys": sorted(
                    {
                        str(key)
                        for item in records
                        if isinstance(item, Mapping)
                        for key in item
                    }
                ),
            }
        )
        return result
    text = raw.decode("utf-8")
    if path.name.endswith(".transcript.sha256"):
        if not _SHA256.fullmatch(text):
            raise ParityError(f"invalid transcript digest: {path}")
        result["kind"] = "sha256-text"
    elif path.suffix == ".py":
        result["kind"] = "python-utf8"
    else:
        result["kind"] = "utf8-text"
    return result


def _normalize_name(name: str, slot: str) -> str:
    return "{prefix}" + name[len(slot) :] if name.startswith(slot) else name


def snapshot_fixture(root: Path) -> dict[str, Any]:
    raw_sha256 = {
        path.relative_to(root).as_posix(): _describe_file(path)["raw_sha256"]
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    profiles: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(root.rglob("turn-manifest.json.gz")):
        directory = manifest_path.parent
        manifest = read_json(manifest_path)
        slot = str(manifest["slot"])
        profile = {
            _normalize_name(path.name, slot): {
                key: value
                for key, value in _describe_file(path).items()
                if key != "raw_sha256"
            }
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
        relative = directory.relative_to(root).as_posix()
        profiles[relative] = profile
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "raw_sha256": raw_sha256,
        "structural_profiles": profiles,
    }


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ParityError(f"unsupported artifact contract: {path}")
    return dict(value)


def compare_contract(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    if actual == expected:
        return
    expected_files = expected.get("raw_sha256", {})
    actual_files = actual.get("raw_sha256", {})
    expected_names = set(expected_files) if isinstance(expected_files, Mapping) else set()
    actual_names = set(actual_files) if isinstance(actual_files, Mapping) else set()
    missing = sorted(expected_names - actual_names)
    added = sorted(actual_names - expected_names)
    changed = sorted(
        name
        for name in expected_names & actual_names
        if expected_files[name] != actual_files[name]
    )
    details = []
    if missing:
        details.append(f"missing={missing}")
    if added:
        details.append(f"added={added}")
    if changed:
        details.append(f"changed={changed}")
    if expected.get("structural_profiles") != actual.get("structural_profiles"):
        details.append("structural_profiles changed")
    raise ParityError("Native v2 App Server artifact parity failed: " + "; ".join(details))


def verify_fixture(root: Path, contract: Mapping[str, Any] | None = None) -> None:
    compare_contract(contract or load_contract(), snapshot_fixture(root))


def verify_frozen_fixture() -> dict[str, Any]:
    expected = load_contract()
    with (
        tempfile.TemporaryDirectory(prefix="mforge-appserver-parity-a-") as first_name,
        tempfile.TemporaryDirectory(prefix="mforge-appserver-parity-b-") as second_name,
    ):
        first = Path(first_name)
        second = Path(second_name)
        generate_fixture(first)
        generate_fixture(second)
        first_snapshot = snapshot_fixture(first)
        second_snapshot = snapshot_fixture(second)
        if first_snapshot != second_snapshot:
            raise ParityError("deterministic provider fixture differs across repeated runs")
        compare_contract(expected, first_snapshot)
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "cases": sorted(
            {str(name).split("/", 1)[0] for name in expected["raw_sha256"]}
        ),
        "file_count": len(expected["raw_sha256"]),
    }


def _profile_for_turn(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "turn-manifest.json.gz")
    slot = str(manifest["slot"])
    return {
        _normalize_name(path.name, slot): {
            key: value
            for key, value in _describe_file(path).items()
            if key != "raw_sha256"
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def verify_real_provider_workspace(experiment_root: Path) -> dict[str, Any]:
    """Structurally compare real Native v2 turns with frozen fixture profiles."""

    artifacts = experiment_root / "artifacts"
    generations = artifacts / "generations"
    manifests = sorted(generations.rglob("turn-manifest.json.gz"))
    if not manifests:
        raise ParityError("real-provider workspace has no provider-turn manifest")
    expected_profiles = list(load_contract()["structural_profiles"].values())
    matched: list[str] = []
    store = TurnArtifactStore(artifacts)
    for manifest_path in manifests:
        directory = manifest_path.parent
        store.verify_turn(directory)
        profile = _profile_for_turn(directory)
        if profile not in expected_profiles:
            raise ParityError(
                "real-provider turn does not match any frozen structural profile: "
                f"{directory.relative_to(generations)}"
            )
        matched.append(directory.relative_to(generations).as_posix())
    return {"turn_count": len(manifests), "matched_turns": matched}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the frozen Native v2 App Server artifact contract."
    )
    parser.add_argument(
        "--real-provider-workspace",
        type=Path,
        help="also structurally verify one disposable real-provider experiment workspace",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="print a freshly generated contract without modifying the repository",
    )
    args = parser.parse_args(argv)
    if args.print_contract:
        with tempfile.TemporaryDirectory(prefix="mforge-appserver-contract-") as root_name:
            root = Path(root_name)
            generate_fixture(root)
            print(json.dumps(snapshot_fixture(root), indent=2, sort_keys=True))
        return 0
    result = verify_frozen_fixture()
    if args.real_provider_workspace is not None:
        result["real_provider"] = verify_real_provider_workspace(
            args.real_provider_workspace.resolve()
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
