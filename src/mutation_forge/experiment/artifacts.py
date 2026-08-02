"""Expanded, redacted App Server turn artifacts for experiment workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
ARTIFACT_MANIFEST = "turn-manifest.json"
_SECRET_KEY = re.compile(
    r"(?i)(?:access[_-]?token|refresh[_-]?token|api[_-]?key|authorization|password|secret|cookie|credential|jwt|private[_-]?key|client[_-]?secret)"
)
_SECRET_VALUE = re.compile(
    r"(?ix)(bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"
)
_PRIVATE_PATH = re.compile(r"(?:/home/[^/]+|/Users/[^/]+|[A-Za-z]:\\Users\\[^\\]+)")


class ArtifactIncompleteError(RuntimeError):
    """A turn exceeded a persistence bound or could not retain required evidence."""


def redact(value: object, key: str = "") -> object:
    if _SECRET_KEY.search(key) or key.lower().replace("_", "") in {
        "token",
        "authtoken",
        "accesstoken",
        "refreshtoken",
        "apikey",
        "authorization",
        "jwt",
        "privatekey",
    }:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(name): redact(item, str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, key) for item in value]
    if isinstance(value, str):
        return _PRIVATE_PATH.sub("[PRIVATE_PATH]", _SECRET_VALUE.sub("[REDACTED]", value))
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise ArtifactIncompleteError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise ArtifactIncompleteError(f"refusing to overwrite artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _text(value: object, *, redact_value: bool = True) -> bytes:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = _canonical(redact(value) if redact_value else value) + b"\n"
    return data


def _json_lines(value: object) -> bytes:
    if isinstance(value, (str, bytes)):
        return _text(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return b"".join(_canonical(redact(item)) + b"\n" for item in value)
    return _canonical(redact(value)) + b"\n"


class TurnArtifactStore:
    """Write one immutable initial/repair turn directory.

    The writer keeps human-readable prompt/response files alongside the raw
    transport projections.  A manifest is written even for a failed turn so a
    caller can distinguish an invalid response from missing evidence.
    """

    def __init__(self, root: str | Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes

    def turn_directory(self, generation: int, slot: int | str, phase: str = "initial") -> Path:
        slot_text = str(slot)
        slot_name = slot_text if slot_text.startswith("slot-") else f"slot-{int(slot_text):02d}"
        if phase != "initial" and not re.fullmatch(r"(?:repair|retry)-\d+", phase):
            raise ValueError("phase must be initial, repair-NN, or retry-NN")
        return self.root / "generations" / f"generation-{generation:04d}" / slot_name / phase

    def write_turn(
        self,
        *,
        generation: int,
        slot: int | str,
        phase: str = "initial",
        request: object | None = None,
        request_text: str | bytes | None = None,
        response: object | None = None,
        response_text: str | bytes | None = None,
        source: str | bytes | None = None,
        usage: Mapping[str, Any] | None = None,
        identity: Mapping[str, Any] | None = None,
        behavior: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        worker_telemetry: Mapping[str, Any] | None = None,
        canonical_response: Mapping[str, Any] | None = None,
        provider_raw: object | None = None,
        codex_profile: object | None = None,
        rpc: object | None = None,
        events: object | None = None,
        wire: object | None = None,
        stdout: object | None = None,
        stderr: object | None = None,
        request_idempotency_key: str | None = None,
        provider_thread_id: str | None = None,
        provider_turn_id: str | None = None,
        terminal_status: str = "completed",
        request_accepted: bool = False,
        content_received: bool | None = None,
        validation_completed: bool = False,
        error: str | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        directory = self.turn_directory(generation, slot, phase)
        if directory.exists():
            raise ArtifactIncompleteError(f"turn directory already exists: {directory}")
        limit = self.max_bytes if max_bytes is None else max_bytes
        directory.mkdir(parents=True, exist_ok=False)
        written: list[Path] = []
        missing: dict[str, str] = {}
        complete = True
        slot_text = str(slot) if str(slot).startswith("slot-") else f"slot-{int(slot):02d}"

        def put(name: str, data: bytes, *, required: bool = False) -> None:
            nonlocal complete
            if len(data) > limit:
                complete = False
                missing[name] = f"artifact bound exceeded ({len(data)} > {limit} bytes)"
                return
            path = directory / name
            _atomic_write(path, data, exclusive=True)
            written.append(path)

        def put_json(name: str, value: object, *, required: bool = False) -> None:
            if value is None:
                if required:
                    missing[name] = "not supplied"
                return
            put(name, _canonical(redact(value)) + b"\n", required=required)

        if request_text is not None:
            put(f"{slot_text}.request.md", _text(request_text), required=True)
        elif request is not None:
            put(f"{slot_text}.request.md", _text(request), required=True)
        else:
            missing[f"{slot_text}.request.md"] = "request construction did not occur"
            complete = False
        if response_text is not None:
            put(f"{slot_text}.response.md", _text(response_text), required=True)
        elif isinstance(response, str | bytes):
            put(f"{slot_text}.response.md", _text(response), required=True)
        elif content_received:
            missing[f"{slot_text}.response.md"] = (
                "textual response was marked received but not supplied"
            )
            complete = False
        elif content_received is None:
            missing[f"{slot_text}.response.md"] = "no textual content received"

        put_json(f"{slot_text}.request.json", request)
        put_json(f"{slot_text}.response.json", response)
        put_json("canonical_response.json", canonical_response)
        put_json("usage.json", usage)
        put_json("identity.json", identity)
        put_json("behavior.json", behavior)
        put_json("provenance.json", provenance)
        put_json("validation.json", validation)
        put_json("worker_telemetry.json", worker_telemetry)
        put_json(f"{slot_text}.provider-raw.json", provider_raw)
        put_json(f"{slot_text}.codex-profile.json", codex_profile)
        if source is not None:
            put("source.py", _text(source), required=True)
        if rpc is not None:
            put(f"{slot_text}.codex-rpc.jsonl", _json_lines(rpc))
        if events is not None:
            put(f"{slot_text}.events.jsonl", _json_lines(events))
        if wire is not None:
            wire_records = (
                wire
                if isinstance(wire, Sequence) and not isinstance(wire, (str, bytes))
                else [wire]
            )
            for record in cast(Sequence[object], wire_records):
                if not isinstance(record, Mapping) or record.get("direction") not in {
                    "client_to_server",
                    "server_to_client",
                }:
                    complete = False
                    missing[f"{slot_text}.wire.jsonl"] = (
                        "wire records must declare client_to_server/server_to_client directions"
                    )
                    break
            else:
                put(f"{slot_text}.wire.jsonl", _json_lines(wire_records))
        if stdout is not None:
            put(f"{slot_text}.stdout.jsonl", _json_lines(stdout))
        if stderr is not None:
            put(f"{slot_text}.stderr.txt", _text(stderr))
        transport_values = [wire, rpc, events, stdout, stderr]
        if any(value is not None for value in transport_values):
            transcript = b"".join(
                _json_lines(value) for value in transport_values if value is not None
            )
            put(
                f"{slot_text}.transcript.sha256",
                _sha256(transcript).encode("ascii") + b"\n",
            )

        source_extracted = source is not None
        if not validation_completed and source_extracted:
            missing.setdefault("validation.json", "validation did not complete")
        # Required transport evidence is recorded as missing rather than
        # silently inferred.  A caller may explicitly mark a pre-response turn
        # complete only when all transport streams were retained.
        for required_name, supplied in {
            f"{slot_text}.wire.jsonl": wire is not None,
            f"{slot_text}.codex-rpc.jsonl": rpc is not None,
            f"{slot_text}.events.jsonl": events is not None,
            f"{slot_text}.stdout.jsonl": stdout is not None,
            f"{slot_text}.stderr.txt": stderr is not None,
            f"{slot_text}.codex-profile.json": codex_profile is not None,
            "usage.json": usage is not None,
        }.items():
            if not supplied:
                missing.setdefault(required_name, "not supplied by provider adapter")
                complete = False

        manifest = {
            "schema_version": "mforge.experiment.turn-manifest.v1",
            "generation": generation,
            "slot": str(slot) if str(slot).startswith("slot-") else f"slot-{int(slot):02d}",
            "phase": phase,
            "request_idempotency_key": request_idempotency_key,
            "provider_thread_id": provider_thread_id,
            "provider_turn_id": provider_turn_id,
            "terminal_status": terminal_status,
            "request_accepted": request_accepted,
            "usage_final_exact": usage is not None,
            "content_received": bool(
                content_received
                if content_received is not None
                else response is not None or response_text is not None
            ),
            "source_extraction": source_extracted,
            "validation_completed": validation_completed,
            "artifact_complete": complete and not missing,
            "files": [
                {
                    "path": path.relative_to(directory).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path.read_bytes()),
                }
                for path in sorted(written)
            ],
            "missing_files": missing,
        }
        if error:
            # A provider failure is a valid retained turn when the complete
            # communication boundary was persisted.  Evidence incompleteness
            # is determined by missing/bounded artifacts above, not by the
            # terminal status itself.
            manifest["error"] = error
        _atomic_write(directory / ARTIFACT_MANIFEST, _canonical(manifest) + b"\n", exclusive=True)
        if not bool(manifest["artifact_complete"]):
            raise ArtifactIncompleteError(
                f"incomplete App Server turn artifacts for generation {generation}, slot {slot}: "
                + "; ".join(f"{key}: {value}" for key, value in missing.items())
            )
        return cast(dict[str, Any], manifest)

    def verify_turn(self, directory: str | Path) -> bool:
        root = Path(directory)
        manifest_path = root / ARTIFACT_MANIFEST
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIncompleteError(f"cannot read turn manifest: {manifest_path}") from exc
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema_version") != "mforge.experiment.turn-manifest.v1"
        ):
            raise ArtifactIncompleteError("invalid turn manifest schema")
        if manifest.get("artifact_complete") is not True:
            raise ArtifactIncompleteError("turn manifest marks artifacts incomplete")
        listed = manifest.get("files")
        if not isinstance(listed, list):
            raise ArtifactIncompleteError("turn manifest files must be an array")
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ARTIFACT_MANIFEST
        }
        expected: set[str] = set()
        for row in listed:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                raise ArtifactIncompleteError("invalid turn manifest file entry")
            relative = str(row["path"])
            target = (root / relative).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError as exc:
                raise ArtifactIncompleteError("turn manifest path escapes directory") from exc
            if (
                not target.is_file()
                or target.stat().st_size != int(row.get("size", -1))
                or _sha256(target.read_bytes()) != row.get("sha256")
            ):
                raise ArtifactIncompleteError(f"turn artifact digest mismatch: {relative}")
            expected.add(relative)
        if actual != expected:
            raise ArtifactIncompleteError(
                "turn artifact file set mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        return True


def copy_canonical_source(
    turn_directory: str | Path, archive_root: str | Path, program_id: str
) -> str:
    source = Path(turn_directory) / "source.py"
    if not source.is_file():
        raise ArtifactIncompleteError("cannot archive a turn without source.py")
    data = source.read_bytes()
    destination = Path(archive_root) / "sources" / f"{program_id}.py"
    if destination.exists():
        if destination.read_bytes() != data:
            raise ArtifactIncompleteError(
                "canonical archive source already exists with different bytes"
            )
    else:
        _atomic_write(destination, data, exclusive=True)
    if destination.read_bytes() != data:
        raise ArtifactIncompleteError("turn-local and archive source bytes differ")
    return _sha256(data)


ArtifactStore = TurnArtifactStore


__all__ = [
    "ARTIFACT_MANIFEST",
    "ArtifactIncompleteError",
    "ArtifactStore",
    "MAX_ARTIFACT_BYTES",
    "TurnArtifactStore",
    "copy_canonical_source",
    "redact",
]
