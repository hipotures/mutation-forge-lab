"""Bounded atomic run artifacts and transport log redaction."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_SECRET = re.compile(
    r"(?i)(?:access[_-]?token|refresh[_-]?token|api[_-]?key|authorization|password|secret|cookie|credential|installation[_-]?id|auth[_-]?token|jwt|private[_-]?key|client[_-]?secret)"
)
_SECRET_VALUE = re.compile(
    r"(?ix)(bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|(?:auth(?:entication)?[_-]?token|access[_-]?token|refresh[_-]?token|api[_-]?key|jwt|token|credential|secret)\s*[:=]\s*[\"']?[^,\s\"']+)"
)
_PRIVATE = re.compile(r"(?:/home/[^/]+|/Users/[^/]+|[A-Za-z]:\\Users\\[^\\]+)")
_ARTIFACT_LOCK = threading.Lock()
_AGGREGATE_SIZES: dict[Path, int] = {}
DEFAULT_MAX_RUN_BYTES = 32 * 1024 * 1024


def _contained_path(base: Path, relative: str | Path) -> Path:
    """Resolve a child path while rejecting absolute and traversal escapes."""
    base_resolved = Path(base).resolve()
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("artifact path must be relative")
    resolved = (base_resolved / candidate).resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError("artifact path escapes root") from exc
    return resolved


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def safe_value(value: object, key: str = "") -> object:
    normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
    if _SECRET.search(key) or normalized_key in {
        "token",
        "authtoken",
        "accesstoken",
        "refreshtoken",
        "apikey",
        "authorization",
        "jwt",
        "clientsecret",
        "privatekey",
    }:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): safe_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v, key) for v in value]
    if isinstance(value, str):
        return _PRIVATE.sub("[PRIVATE_PATH]", _SECRET_VALUE.sub("[REDACTED]", value))
    return value


class GenerationArtifacts:
    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        max_file_bytes: int = 1_048_576,
        max_total_bytes: int = DEFAULT_MAX_RUN_BYTES,
    ) -> None:
        self.root = _contained_path(Path(root), run_id)
        self.root.mkdir(parents=True, exist_ok=False)
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self._started = False
        self.telemetry: dict[str, int] = {"write_failures": 0, "truncations": 0}

    def _bounded(self, path: Path, payload: str) -> None:
        size = len(payload.encode())
        if size > self.max_file_bytes:
            self.telemetry["write_failures"] += 1
            raise ValueError("artifact exceeds per-file byte bound")
        current_size = path.stat().st_size if path.is_file() else 0
        stored_size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        if stored_size - current_size + size > self.max_total_bytes:
            self.telemetry["write_failures"] += 1
            raise ValueError("artifact run exceeds total byte bound")

    def _write_payload(self, path: Path, payload: str) -> Path:
        with _ARTIFACT_LOCK:
            self._bounded(path, payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return path

    def _write(self, relative: str, value: object) -> Path:
        return self._write_payload(
            _contained_path(self.root, relative), canonical_json(safe_value(value)) + "\n"
        )

    def start(self, metadata: Mapping[str, object]) -> None:
        if self._started:
            return
        self._started = True
        d = dict(metadata)
        d.setdefault("status", "failed")
        d.setdefault("created_at", datetime.now(UTC).isoformat())
        self._write("generation_summary.json", d)
        self._write("run_summary.json", {"status": "failed", "run_id": self.root.name})

    def write(self, name: str, value: object) -> Path:
        if not self._started:
            self.start({"run_id": self.root.name})
        return self._write(name, value)

    def write_text(self, name: str, value: str) -> Path:
        if not self._started:
            self.start({"run_id": self.root.name})
        return self._write_payload(_contained_path(self.root, name), value)

    def finish(self, status: str, summary: Mapping[str, object]) -> None:
        existing = {}
        p = self.root / "generation_summary.json"
        if p.is_file():
            existing = cast(dict[str, Any], json.loads(p.read_text()))
        final = {**existing, **dict(summary), "status": status}
        final.setdefault("run_id", self.root.name)
        final.setdefault("finished_at", datetime.now(UTC).isoformat())
        self._write("generation_summary.json", final)
        self._write("run_summary.json", {"status": status, "run_id": self.root.name, **final})

    @staticmethod
    def read_summary(root: Path) -> dict[str, Any]:
        return cast(
            dict[str, Any], json.loads((Path(root) / "generation_summary.json").read_text())
        )


class TransportLogger:
    """Bounded transport logs with append-only streaming files."""

    def __init__(
        self,
        directory: Path,
        prefix: str = "",
        *,
        max_bytes: int = 2 * 1024 * 1024,
        max_events: int = 10_000,
        max_line_bytes: int = 256 * 1024,
        aggregate_root: Path | None = None,
        max_aggregate_bytes: int = DEFAULT_MAX_RUN_BYTES,
        compress_json: bool = False,
    ) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix.rstrip(".")
        if self.prefix and (
            Path(self.prefix).is_absolute() or "/" in self.prefix or "\\" in self.prefix
        ):
            raise ValueError("transport log prefix escapes artifact directory")
        self.max_bytes = max_bytes
        self.max_events = max_events
        self.max_line_bytes = max_line_bytes
        self.aggregate_root = (
            Path(aggregate_root).resolve() if aggregate_root is not None else self.directory
        )
        self.max_aggregate_bytes = max_aggregate_bytes
        self.compress_json = compress_json
        self.events = 0
        self.bytes = 0
        self.rpc: list[str] = []
        self.notifications: list[str] = []
        self.wire: list[str] = []
        self.transcript: list[str] = []
        self._transcript_hash = hashlib.sha256()
        self._finalized = False
        self.telemetry: dict[str, int] = {
            "events": 0,
            "bytes": 0,
            "truncations": 0,
            "write_failures": 0,
        }
        for name in (
            "codex-rpc.jsonl",
            "events.jsonl",
            "wire.jsonl",
            "stdout.jsonl",
            "stderr.txt",
        ):
            self._write_lines(name, [])
        self.finalize()

    def _name(self, name: str) -> str:
        if self.compress_json and name.endswith(".json"):
            name = f"{name}.gz"
        return f"{self.prefix}.{name}" if self.prefix else name

    def _write_lines(self, name: str, lines: list[str]) -> None:
        payload = "".join(lines).encode()
        if len(payload) > self.max_bytes:
            self.telemetry["write_failures"] += 1
            raise ValueError("transport log exceeds byte limit")
        target = _contained_path(self.directory, self._name(name))
        stored = gzip.compress(payload, compresslevel=6, mtime=0) if target.name.endswith(
            ".json.gz"
        ) else payload
        with _ARTIFACT_LOCK:
            current_size = target.stat().st_size if target.is_file() else 0
            aggregate_size = self._aggregate_size_locked()
            updated_size = aggregate_size - current_size + len(stored)
            if updated_size > self.max_aggregate_bytes:
                self.telemetry["write_failures"] += 1
                raise ValueError("transport run exceeds aggregate byte limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            # A provider can be interrupted while another adapter is draining
            # its transport.  Retry one missing-temp-file race so a transient
            # cleanup cannot turn an otherwise valid turn into a provider
            # failure.  The second failure remains visible to the caller.
            for attempt in range(2):
                fd, tmp = tempfile.mkstemp(prefix=".log.", dir=self.directory)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(stored)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, target)
                    _AGGREGATE_SIZES[self.aggregate_root] = updated_size
                    break
                except FileNotFoundError:
                    if attempt == 1:
                        raise
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)

    def _aggregate_size_locked(self) -> int:
        retained = _AGGREGATE_SIZES.get(self.aggregate_root)
        if retained is not None:
            return retained
        retained = sum(
            path.stat().st_size
            for path in self.aggregate_root.rglob("*")
            if path.is_file()
        )
        _AGGREGATE_SIZES[self.aggregate_root] = retained
        return retained

    def _append(self, name: str, value: str) -> None:
        payload = value.encode()
        target = _contained_path(self.directory, self._name(name))
        with _ARTIFACT_LOCK:
            current_size = target.stat().st_size if target.is_file() else 0
            if current_size + len(payload) > self.max_bytes:
                self.telemetry["write_failures"] += 1
                raise ValueError("transport log exceeds byte limit")
            aggregate_size = self._aggregate_size_locked()
            if aggregate_size + len(payload) > self.max_aggregate_bytes:
                self.telemetry["write_failures"] += 1
                raise ValueError("transport run exceeds aggregate byte limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("ab") as stream:
                stream.write(payload)
            _AGGREGATE_SIZES[self.aggregate_root] = aggregate_size + len(payload)

    def cleanup_temporary_files(self) -> None:
        """Remove interrupted atomic-write leftovers after logger shutdown."""

        with _ARTIFACT_LOCK:
            for path in self.directory.glob(".log.*"):
                path.unlink(missing_ok=True)

    def _record_wire(
        self,
        direction: str,
        value: Mapping[str, Any],
        raw: bytes | str | None,
    ) -> tuple[dict[str, Any], str]:
        self.events += 1
        rawb = raw.encode() if isinstance(raw, str) else (raw or b"")
        self.bytes += len(rawb)
        self.telemetry["events"] = self.events
        self.telemetry["bytes"] = self.bytes
        if (
            self.events > self.max_events
            or len(rawb) > self.max_line_bytes
            or self.bytes > self.max_bytes
        ):
            self.telemetry["write_failures"] += 1
            raise ValueError("transport message limit exceeded")
        safe = cast(dict[str, Any], safe_value(dict(value)))
        line = canonical_json(safe) + "\n"
        wire_line = canonical_json({"direction": direction, "message": safe}) + "\n"
        self.wire.append(wire_line)
        self.transcript.append(wire_line)
        encoded_wire = wire_line.encode()
        self._transcript_hash.update(encoded_wire)
        self._finalized = False
        self._append("wire.jsonl", wire_line)
        return safe, line

    def sent(self, value: Mapping[str, Any], raw: bytes | str | None = None) -> None:
        self._record_wire("client_to_server", value, raw)

    def message(self, value: Mapping[str, Any], raw: bytes | str | None = None) -> None:
        _, line = self._record_wire("server_to_client", value, raw)
        if "id" in value:
            self.rpc.append(line)
            self._append("codex-rpc.jsonl", line)
        else:
            self.notifications.append(line)
            self._append("events.jsonl", line)

    def append_text(self, name: str, value: str) -> None:
        """Append one redacted streaming text chunk without rewriting prior chunks."""

        self._append(name, cast(str, safe_value(value)))

    def text(self, name: str, value: str) -> None:
        # Markdown artifacts are text projections, not a second JSON
        # serialization.  In particular, a prompt that happens to contain a
        # JSON fragment must remain the exact prompt text here; structured
        # envelopes belong in their separate ``*.json`` files.
        value = cast(str, safe_value(value))
        self._bounded_text(name, value)

    def raw_text(self, name: str, value: str | bytes) -> None:
        """Persist an exact text payload without redaction or JSON wrapping.

        Native response retention uses this for the byte-faithful assistant
        payload.  Callers must opt in explicitly because ordinary transport
        logs continue to redact credentials and private paths.
        """

        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        self._bounded_text(name, text)

    def _bounded_text(self, name: str, value: str) -> None:
        if len(value.encode()) > self.max_bytes:
            original = len(value.encode())
            marker = f"\n[TRUNCATED original_bytes={original}]\n"
            marker_bytes = marker.encode()
            if len(marker_bytes) > self.max_bytes:
                self.telemetry["write_failures"] += 1
                raise ValueError("transport text limit too small for truncation telemetry")
            retained = value.encode()[: self.max_bytes - len(marker_bytes)].decode(
                "utf-8", "ignore"
            )
            value = retained + marker
            self.telemetry["truncations"] += 1
        self._write_lines(name, [value])

    def remove(self, name: str) -> None:
        """Remove one known logger file when a later semantic projection supersedes it."""

        target = _contained_path(self.directory, self._name(name))
        with _ARTIFACT_LOCK:
            current_size = target.stat().st_size if target.is_file() else 0
            target.unlink(missing_ok=True)
            _AGGREGATE_SIZES[self.aggregate_root] = max(
                0,
                self._aggregate_size_locked() - current_size,
            )

    def profile(self, value: Mapping[str, Any]) -> None:
        self._write_lines("codex-profile.json", [canonical_json(safe_value(value)) + "\n"])

    def document(self, name: str, value: object) -> None:
        self._write_lines(name, [canonical_json(safe_value(value)) + "\n"])

    def finalize(self) -> None:
        """Persist the transcript digest once at a meaningful turn boundary."""

        if self._finalized:
            return
        self._write_lines("transcript.sha256", [self.transcript_sha256 + "\n"])
        self._finalized = True

    @property
    def transcript_sha256(self) -> str:
        return self._transcript_hash.copy().hexdigest()


def replay_generation(root: Path) -> dict[str, Any]:
    """Revalidate persisted candidates and reproduce behavior without a provider call."""
    from mutation_forge.sandbox.contracts import SandboxLimits
    from mutation_forge.sandbox.validation import validate_policy

    from .generation import _behavior, _sha_source

    run_root = Path(root).resolve()
    summary = GenerationArtifacts.read_summary(run_root)
    errors: list[dict[str, str]] = []
    replayed: list[dict[str, Any]] = []
    try:
        if not isinstance(summary, dict) or summary.get("status") != "completed":
            raise ValueError("generation campaign is incomplete")

        def strip_generation(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(k): strip_generation(v)
                    for k, v in value.items()
                    if k
                    not in {
                        "started_at",
                        "finished_at",
                        "created_at",
                        "elapsed_seconds",
                        "timing",
                        "canonical_generation_sha256",
                    }
                    and not str(k).endswith("_ns")
                }
            if isinstance(value, (list, tuple)):
                return [strip_generation(v) for v in value]
            return value

        expected_generation_hash = summary.get("canonical_generation_sha256")
        if (
            not isinstance(expected_generation_hash, str)
            or canonical_hash(strip_generation(summary)) != expected_generation_hash
        ):
            raise ValueError("canonical generation hash mismatch")

        generation_config = cast(
            dict[str, Any],
            json.loads((run_root / "generation_config.json").read_text(encoding="utf-8")),
        )
        config_hash = canonical_hash(generation_config)
        for key in ("canonical_sha256", "generation_config_sha256", "config_sha256"):
            declared = generation_config.get(key)
            if declared is not None:
                payload = {k: v for k, v in generation_config.items() if k != key}
                if declared != canonical_hash(payload):
                    raise ValueError(f"generation config {key} mismatch")
        for key in ("generation_config_sha256", "config_sha256"):
            declared = summary.get(key)
            if declared is not None and declared != config_hash:
                raise ValueError(f"{key} mismatch")
        limits_value = generation_config.get("sandbox_limits")
        if not isinstance(limits_value, dict):
            raise ValueError("generation config omits sandbox_limits")
        limits = SandboxLimits(**limits_value)
        smoke_calls = int(generation_config.get("smoke_calls", 0))
        slots = summary.get("slots")
        expected_slots = [f"slot-{i:02d}" for i in range(8)]
        if (
            not isinstance(slots, list)
            or [s.get("slot") for s in slots if isinstance(s, dict)] != expected_slots
        ):
            raise ValueError("generation summary slots are not exactly ordered slot-00..07")
        slots_root = run_root / "slots"
        if not slots_root.is_dir() or {p.name for p in slots_root.iterdir() if p.is_dir()} != set(
            expected_slots
        ):
            raise ValueError("slot artifact set is incomplete or contains unexpected paths")
        if (run_root / "slots.json").is_file() and json.loads(
            (run_root / "slots.json").read_text(encoding="utf-8")
        ) != slots:
            raise ValueError("slots artifact does not match generation summary")
        freeze_path = run_root / "freeze.json"
        if freeze_path.is_file():
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if (
                not isinstance(freeze, dict)
                or freeze.get("slots") != expected_slots
                or freeze.get("max_concurrency") != 8
            ):
                raise ValueError("frozen campaign identity mismatch")
            freeze_hash = canonical_hash(freeze)
            for key in ("freeze_sha256", "freeze_payload_sha256", "freeze_identity_sha256"):
                declared = summary.get(key)
                if declared is not None and declared != freeze_hash:
                    raise ValueError(f"{key} mismatch")

        usage_keys = (
            "inputTokens",
            "cachedInputTokens",
            "cacheWriteInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        )
        usage_totals = {key: 0 for key in usage_keys}
        repair_count = 0
        unique_hashes: set[tuple[str | None, str | None]] = set()
        for slot_value in slots:
            if not isinstance(slot_value, dict) or not isinstance(slot_value.get("slot"), str):
                raise ValueError("malformed generation slot summary")
            slot = cast(str, slot_value["slot"])
            if slot not in expected_slots:
                raise ValueError("invalid slot path")
            if (
                slot_value.get("status") not in {"accepted", "duplicate"}
                or bool(slot_value.get("duplicate")) != (slot_value.get("status") == "duplicate")
                or int(slot_value.get("repairs", -1)) not in {0, 1}
            ):
                raise ValueError(f"{slot} has invalid status or repair count")
            repair_count += int(slot_value["repairs"])
            slot_root = slots_root / slot
            events_value = json.loads((slot_root / "events.json").read_text(encoding="utf-8"))
            if (
                not isinstance(events_value, list)
                or len(events_value) != 1
                or not isinstance(events_value[0], dict)
            ):
                raise ValueError(f"{slot} terminal event missing")
            terminal = events_value[0]
            if terminal.get("status") != slot_value["status"]:
                raise ValueError(f"{slot} status mismatch")
            initial = terminal.get("initial")
            repair = terminal.get("repair")
            if not isinstance(initial, dict):
                raise ValueError(f"{slot} initial turn missing")
            if (
                initial.get("status") != "completed"
                or initial.get("accepted") is not True
                or initial.get("content") is not True
            ):
                raise ValueError(f"{slot} initial turn provenance is incomplete")
            if int(slot_value["repairs"]) == 1 and not isinstance(repair, dict):
                raise ValueError(f"{slot} repair turn missing")
            if int(slot_value["repairs"]) == 0 and repair is not None:
                raise ValueError(f"{slot} unexpected repair turn")
            final_turn = repair if isinstance(repair, dict) else initial
            if (
                final_turn.get("status") != "completed"
                or final_turn.get("accepted") is not True
                or final_turn.get("content") is not True
            ):
                raise ValueError(f"{slot} turn provenance is incomplete")
            usage = final_turn.get("usage")
            if (
                not isinstance(usage, dict)
                or usage.get("final") is not True
                or usage.get("partial") is not False
            ):
                raise ValueError(f"{slot} exact usage is missing")
            for key in usage_keys:
                if (
                    not isinstance(usage.get(key), int)
                    or isinstance(usage.get(key), bool)
                    or usage[key] < 0
                ):
                    raise ValueError(f"{slot} invalid usage")
                usage_totals[key] += usage[key]
            if bool(final_turn.get("charged")) != (usage["totalTokens"] > 0):
                raise ValueError(f"{slot} usage provenance mismatch")
            request_path = slot_root / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if (
                not isinstance(request, dict)
                or request.get("slot") != slot
                or request.get("model") != generation_config.get("model")
                or request.get("effort") != generation_config.get("effort")
                or request.get("protocol_version") != generation_config.get("protocol_version")
            ):
                raise ValueError(f"{slot} request/config identity mismatch")
            source_path = run_root / "slots" / slot / "source.py"
            if not source_path.is_file():
                raise ValueError(f"{slot} is missing persisted source")
            source = source_path.read_text(encoding="utf-8")
            validation = validate_policy(source, limits)
            if not validation.valid:
                raise ValueError(f"{slot} source no longer validates")
            source_sha256 = _sha_source(source)
            if source_sha256 != slot_value.get("source_sha256"):
                raise ValueError(f"{slot} source hash mismatch")
            identity_path = run_root / "slots" / slot / "identity.json"
            expected_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            actual_identity = validation.identity.as_dict()
            if expected_identity != actual_identity:
                raise ValueError(f"{slot} identity mismatch")
            expected_behavior = json.loads(
                (run_root / "slots" / slot / "behavior.json").read_text(encoding="utf-8")
            ).get("signature")
            actual_behavior, telemetry = _behavior(source, limits, smoke_calls)
            if expected_behavior != actual_behavior:
                raise ValueError(f"{slot} behavior signature mismatch")
            provenance_path = slot_root / "provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if not isinstance(provenance, dict) or provenance.get("model") != generation_config.get(
                "model"
            ):
                raise ValueError(f"{slot} model provenance mismatch")
            for key in ("request_id", "thread_id", "session_id", "turn_id", "transport_sha256"):
                if provenance.get(key) != final_turn.get(key):
                    raise ValueError(f"{slot} {key} provenance mismatch")
            if provenance.get("protocol_version") != generation_config.get("protocol_version"):
                raise ValueError(f"{slot} protocol provenance mismatch")
            if provenance.get("usage") != usage:
                raise ValueError(f"{slot} usage provenance mismatch")
            unique_hashes.add(
                (slot_value.get("source_sha256"), slot_value.get("normalized_ast_sha256"))
            )
            replayed.append(
                {
                    "slot": slot,
                    "source_sha256": source_sha256,
                    "normalized_ast_sha256": validation.identity.normalized_ast_sha256,
                    "behavior_signature_sha256": actual_behavior.get("signature_sha256"),
                    "smoke_calls": telemetry.get("smoke_calls"),
                }
            )
        if (
            int(summary.get("initial_turn_count", -1)) != 8
            or int(summary.get("repair_turn_count", -1)) != repair_count
        ):
            raise ValueError("turn count mismatch")
        if (
            int(summary.get("total_live_turns", -1)) != 8 + repair_count
            or int(summary.get("provider_attempts", -1)) != 8 + repair_count
        ):
            raise ValueError("live turn count mismatch")
        if any(
            int(summary.get(key, -1)) != 8 + repair_count
            for key in ("completed_turns", "model_turns", "accepted_model_turns")
        ):
            raise ValueError("completed turn count mismatch")
        if (
            summary.get("exact_usage_complete") is not True
            or summary.get("usage_totals") != usage_totals
        ):
            raise ValueError("usage totals mismatch")
        expected_unique = sum(1 for slot_value in slots if slot_value.get("status") == "accepted")
        if int(summary.get("unique_count", -1)) != expected_unique:
            raise ValueError("unique candidate count mismatch")
    except Exception as error:
        errors.append({"code": type(error).__name__, "message": str(error)[:512]})
    return {
        **summary,
        "replay_validated": not errors,
        "provider_calls": 0,
        "replayed_candidates": replayed,
        "replay_errors": errors,
        "replay_sha256": canonical_hash(replayed) if not errors else None,
    }
