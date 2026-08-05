"""Bounded, deduplicated exact-verification supervisor for Native v3."""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
import resource
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import ExactVerification, GraphState, JsonValue

from .canonical import canonical_json_bytes, domain_hash, json_value

VERIFICATION_PROTOCOL_ID = "native_v3_dual_exact_verification_v1"


class VerificationDecision(StrEnum):
    VERIFIED = "VERIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONFLICT = "CONFLICT"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class VerificationProfile:
    concurrency: int = 1
    queue_capacity: int = 16
    verifier_wall_timeout_seconds: float = 600.0
    verifier_memory_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        if self.concurrency != 1:
            raise ValueError("Native v3 verification profile requires concurrency=1")
        if self.queue_capacity != 16:
            raise ValueError("Native v3 verification profile requires queue_capacity=16")
        if self.verifier_wall_timeout_seconds != 600.0:
            raise ValueError("Native v3 verification profile requires a 600-second timeout")
        if self.verifier_memory_bytes != 4 * 1024**3:
            raise ValueError("Native v3 verification profile requires a 4 GiB limit")


@dataclass(frozen=True, slots=True)
class VerificationJob:
    graph_hash: str
    graph: GraphState
    provenance: Mapping[str, JsonValue]
    verification_protocol_id: str = VERIFICATION_PROTOCOL_ID

    @property
    def identity(self) -> tuple[str, str]:
        return (self.graph_hash, self.verification_protocol_id)


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    graph_hash: str
    verification_protocol_id: str
    decision: VerificationDecision
    primary: ExactVerification | None
    independent: ExactVerification | None
    artifact_directory: Path
    detail: str


@dataclass(slots=True)
class _QueuedJob:
    job: VerificationJob
    future: Future[VerificationOutcome]


def graph_content_hash(graph: GraphState) -> str:
    return domain_hash(
        b"mforge-native-v3-graph\0",
        canonical_json_bytes(
            {
                "order": graph.order,
                "edges": [list(edge) for edge in sorted(graph.edges)],
            }
        ),
    )


def verify_heg_primary(
    graph: GraphState,
    *,
    heg_repo: Path,
    graph_mode: str,
) -> ExactVerification:
    from mutation_forge.backends.heg import HegBackend

    backend = HegBackend(heg_repo, graph_mode=graph_mode)
    try:
        return backend.exact_verify(graph)
    finally:
        backend.close()


def _graph6_bytes(graph: GraphState) -> bytes:
    order = graph.order
    if order <= 62:
        prefix = bytes((order + 63,))
    elif order <= 258047:
        prefix = bytes(
            (
                126,
                ((order >> 12) & 63) + 63,
                ((order >> 6) & 63) + 63,
                (order & 63) + 63,
            )
        )
    else:
        raise ValueError("Native v3 graph order exceeds graph6 compact range")
    edge_set = set(graph.edges)
    bits = [int((low, high) in edge_set) for high in range(1, order) for low in range(high)]
    while len(bits) % 6:
        bits.append(0)
    encoded = bytearray(prefix)
    for offset in range(0, len(bits), 6):
        value = 0
        for bit in bits[offset : offset + 6]:
            value = (value << 1) | bit
        encoded.append(value + 63)
    return bytes(encoded)


def verify_independent_python(graph: GraphState) -> ExactVerification:
    from mutation_forge.independent_verifier import verify

    with tempfile.TemporaryDirectory(prefix="mforge-native-v3-verify-") as directory:
        path = Path(directory) / "candidate.graph6"
        path.write_bytes(_graph6_bytes(graph) + b"\n")
        payload = verify(path)
    witnesses_value = payload.get("witnesses", [])
    witnesses = tuple(
        (str(item[0]), tuple(int(vertex) for vertex in item[1]))
        for item in witnesses_value
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], list)
    )
    return ExactVerification(
        status=str(payload.get("status", "UNKNOWN")),
        complete=payload.get("complete") is True,
        message=str(payload.get("message", "")),
        implementation=str(payload.get("implementation", "mforge-independent-cycle-mitm-v2")),
        witnesses=witnesses,
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        implementation_sha256=(
            str(payload["implementation_sha256"])
            if isinstance(payload.get("implementation_sha256"), str)
            else None
        ),
        configuration={
            "target_forbidden_lengths": [
                int(value)
                for value in payload.get("target_forbidden_lengths", [])
                if isinstance(value, int)
            ],
            "process_isolated": True,
        },
    )


def _verifier_process(
    connection: Any,
    verifier: Callable[[GraphState], ExactVerification],
    graph: GraphState,
    memory_bytes: int,
) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        result = verifier(graph)
        connection.send(("ok", result))
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _run_isolated(
    verifier: Callable[[GraphState], ExactVerification],
    graph: GraphState,
    *,
    timeout_seconds: float,
    memory_bytes: int,
) -> tuple[ExactVerification | None, str]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_verifier_process,
        args=(child, verifier, graph, memory_bytes),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.kill()
            process.join(timeout=5)
            return None, "verifier wall timeout"
        status, value = parent.recv()
        process.join(timeout=5)
        if status != "ok":
            return None, str(value)
        if not isinstance(value, ExactVerification):
            return None, "verifier returned an invalid result type"
        return value, ""
    finally:
        parent.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


class VerificationSupervisor:
    """Persist apparent zeros and verify them on one bounded priority lane."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        primary_verifier: Callable[[GraphState], ExactVerification],
        independent_verifier: Callable[[GraphState], ExactVerification],
        profile: VerificationProfile | None = None,
        telemetry_sink: Callable[[str, Mapping[str, JsonValue]], None] | None = None,
    ) -> None:
        self.artifact_root = artifact_root
        self.primary_verifier = primary_verifier
        self.independent_verifier = independent_verifier
        self.profile = profile or VerificationProfile()
        self.telemetry_sink = telemetry_sink
        self._jobs: queue.Queue[_QueuedJob | None] = queue.Queue(self.profile.queue_capacity)
        self._lock = threading.Lock()
        self._futures: dict[tuple[str, str], Future[VerificationOutcome]] = {}
        self._thread = threading.Thread(
            target=self._run,
            name="native-v3-verification-supervisor",
            daemon=True,
        )
        self._thread.start()

    def submit(self, job: VerificationJob) -> Future[VerificationOutcome]:
        if graph_content_hash(job.graph) != job.graph_hash:
            raise ValueError("verification job graph hash mismatch")
        directory = self.artifact_root / job.verification_protocol_id / job.graph_hash
        with self._lock:
            existing = self._futures.get(job.identity)
            if existing is not None:
                self._append_provenance(directory, job.provenance)
                return existing
            # Persist the graph and provenance before it can be hidden behind
            # queue backpressure or handed to an exact verifier. A crash after
            # this write leaves a durable apparent-zero artifact for recovery.
            self._persist_candidate(directory, job)
            self._append_provenance(directory, job.provenance)
            future: Future[VerificationOutcome] = Future()
            self._futures[job.identity] = future
        # Blocking here is intentional. A full verification queue applies
        # backpressure to scoring; an apparent zero is never discarded.
        backpressure_started_ns: int | None = None
        if self._jobs.full():
            backpressure_started_ns = time.monotonic_ns()
            self._emit(
                "verification_backpressure_started",
                {"verification_queue_depth": self._jobs.qsize()},
            )
        self._jobs.put(_QueuedJob(job, future), block=True)
        if backpressure_started_ns is not None:
            self._emit(
                "verification_backpressure_ended",
                {
                    "verification_queue_depth": self._jobs.qsize(),
                    "idle_ns": time.monotonic_ns() - backpressure_started_ns,
                },
            )
        self._emit(
            "verification_job_queued",
            {
                "graph_hash": job.graph_hash,
                "verification_queue_depth": self._jobs.qsize(),
            },
        )
        return future

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join()

    def recover_pending(self) -> tuple[Future[VerificationOutcome], ...]:
        """Requeue durable candidates that have no terminal verification outcome."""

        protocol_root = self.artifact_root / VERIFICATION_PROTOCOL_ID
        if not protocol_root.is_dir():
            return ()
        recovered: list[Future[VerificationOutcome]] = []
        for candidate_path in sorted(protocol_root.glob("*/candidate.json")):
            directory = candidate_path.parent
            if (directory / "outcome.json").is_file():
                continue
            value = json.loads(candidate_path.read_text(encoding="utf-8"))
            graph_value = value.get("graph")
            if not isinstance(graph_value, dict):
                raise ValueError(f"invalid durable verification candidate: {candidate_path}")
            edges_value = graph_value.get("edges")
            if not isinstance(edges_value, list):
                raise ValueError(f"invalid durable verification graph: {candidate_path}")
            graph = GraphState(
                int(graph_value["order"]),
                tuple((int(edge[0]), int(edge[1])) for edge in edges_value),
            )
            provenance = value.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"invalid durable verification provenance: {candidate_path}")
            recovered.append(
                self.submit(
                    VerificationJob(
                        str(value["graph_hash"]),
                        graph,
                        cast(Mapping[str, JsonValue], json_value(provenance)),
                        str(value["verification_protocol_id"]),
                    )
                )
            )
        return tuple(recovered)

    def __enter__(self) -> VerificationSupervisor:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            queued = self._jobs.get()
            if queued is None:
                return
            try:
                outcome = self._verify(queued.job)
            except BaseException as error:
                queued.future.set_exception(error)
            else:
                self._persist_outcome(outcome)
                queued.future.set_result(outcome)
                self._emit(
                    "verification_job_completed",
                    {
                        "graph_hash": outcome.graph_hash,
                        "decision": outcome.decision.value,
                        "verification_queue_depth": self._jobs.qsize(),
                    },
                )

    def _emit(self, name: str, fields: Mapping[str, JsonValue]) -> None:
        if self.telemetry_sink is None:
            return
        try:
            self.telemetry_sink(name, fields)
        except Exception:
            # Observational telemetry must not alter scientific execution.
            return

    def _verify(self, job: VerificationJob) -> VerificationOutcome:
        directory = self.artifact_root / job.verification_protocol_id / job.graph_hash
        primary, primary_error = _run_isolated(
            self.primary_verifier,
            job.graph,
            timeout_seconds=self.profile.verifier_wall_timeout_seconds,
            memory_bytes=self.profile.verifier_memory_bytes,
        )
        self._persist_result(directory, "primary", primary, primary_error)
        if (
            primary is None
            or primary.status.upper() != VerificationDecision.VERIFIED
            or not primary.complete
        ):
            return VerificationOutcome(
                job.graph_hash,
                job.verification_protocol_id,
                VerificationDecision.INCONCLUSIVE,
                primary,
                None,
                directory,
                primary_error or "primary verifier did not return complete VERIFIED",
            )
        independent, independent_error = _run_isolated(
            self.independent_verifier,
            job.graph,
            timeout_seconds=self.profile.verifier_wall_timeout_seconds,
            memory_bytes=self.profile.verifier_memory_bytes,
        )
        self._persist_result(
            directory,
            "independent",
            independent,
            independent_error,
        )
        if independent is None or not independent.complete:
            return VerificationOutcome(
                job.graph_hash,
                job.verification_protocol_id,
                VerificationDecision.INCONCLUSIVE,
                primary,
                independent,
                directory,
                independent_error or "independent verifier was incomplete",
            )
        if independent.status.upper() != VerificationDecision.VERIFIED:
            return VerificationOutcome(
                job.graph_hash,
                job.verification_protocol_id,
                VerificationDecision.CONFLICT,
                primary,
                independent,
                directory,
                "exact verifiers disagree",
            )
        certificate = {
            "schema_version": "mforge.native.counterexample_certificate.v3",
            "graph_hash": job.graph_hash,
            "verification_protocol_id": job.verification_protocol_id,
            "primary_verified": True,
            "independent_verified": True,
        }
        self._write_durable(directory / "certificate.json", canonical_json_bytes(certificate))
        return VerificationOutcome(
            job.graph_hash,
            job.verification_protocol_id,
            VerificationDecision.VERIFIED,
            primary,
            independent,
            directory,
            "both exact verifiers returned complete VERIFIED",
        )

    @staticmethod
    def _persist_candidate(directory: Path, job: VerificationJob) -> None:
        payload = {
            "schema_version": "mforge.native.counterexample_candidate.v3",
            "graph_hash": job.graph_hash,
            "verification_protocol_id": job.verification_protocol_id,
            "graph": {
                "order": job.graph.order,
                "edges": [list(edge) for edge in sorted(job.graph.edges)],
            },
            "provenance": dict(job.provenance),
        }
        VerificationSupervisor._write_durable(
            directory / "candidate.json",
            canonical_json_bytes(payload),
        )

    @staticmethod
    def _append_provenance(directory: Path, provenance: Mapping[str, JsonValue]) -> None:
        """Retain every discovery of a deduplicated graph without rerunning it."""

        path = directory / "provenance.ndjson"
        payload = canonical_json_bytes({"provenance": dict(provenance)})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _persist_result(
        directory: Path,
        role: str,
        result: ExactVerification | None,
        error: str,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "schema_version": "mforge.native.counterexample_verification.v3",
            "role": role,
            "error": error,
        }
        if result is not None:
            payload["result"] = json_value(
                {
                    "status": result.status,
                    "complete": result.complete,
                    "message": result.message,
                    "implementation": result.implementation,
                    "witnesses": [[kind, list(vertices)] for kind, vertices in result.witnesses],
                    "implementation_sha256": result.implementation_sha256,
                    "configuration": dict(result.configuration),
                }
            )
        VerificationSupervisor._write_durable(
            directory / f"verification-{role}.json",
            canonical_json_bytes(payload),
        )

    @staticmethod
    def _persist_outcome(outcome: VerificationOutcome) -> None:
        VerificationSupervisor._write_durable(
            outcome.artifact_directory / "outcome.json",
            canonical_json_bytes(
                {
                    "schema_version": "mforge.native.counterexample_outcome.v3",
                    "graph_hash": outcome.graph_hash,
                    "verification_protocol_id": outcome.verification_protocol_id,
                    "decision": outcome.decision.value,
                    "detail": outcome.detail,
                }
            ),
        )

    @staticmethod
    def _write_durable(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
