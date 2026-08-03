from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from mutation_forge.artifacts import git_state
from mutation_forge.backends.base import GraphBackend
from mutation_forge.models import ExactVerification, GraphScore, GraphState
from mutation_forge.proposals.k_switch import EvaluationContractError

CANDIDATE_SCHEMA_VERSION = "mforge.counterexample.candidate.v2"
VERIFICATION_SCHEMA_VERSION = "mforge.counterexample.verification.v2"
CERTIFICATE_SCHEMA_VERSION = "mforge.counterexample.certificate.v2"
TARGET_CONTRACT_VERSION = "erdos_gyarfas.v2"


class CounterexampleDecision(Enum):
    CONTINUE = "continue"
    PRIMARY_REJECTED = "primary_rejected"
    PAUSE_INCONCLUSIVE = "pause_inconclusive"
    STOP_VERIFIED = "stop_verified"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CandidateProvenance:
    source_kind: str
    source_id: str
    generation: int | None = None
    slot: str | None = None
    episode_id: str | None = None
    graph_seed: int | None = None
    policy_seed: int | None = None
    evaluation_step: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "generation": self.generation,
            "slot": self.slot,
            "episode_id": self.episode_id,
            "graph_seed": self.graph_seed,
            "policy_seed": self.policy_seed,
            "evaluation_step": self.evaluation_step,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CounterexampleCandidate:
    candidate_id: str
    artifact_directory: Path
    graph_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class CounterexampleVerificationRecord:
    role: str
    status: str
    complete: bool
    verifier_id: str
    artifact_directory: Path
    artifact_path: Path
    message: str = ""


@dataclass(frozen=True, slots=True)
class CounterexampleCertificate:
    candidate_id: str
    certificate_id: str
    artifact_directory: Path
    artifact_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class CounterexampleOutcome:
    decision: CounterexampleDecision
    candidate: CounterexampleCandidate | None = None
    primary: CounterexampleVerificationRecord | None = None
    independent: CounterexampleVerificationRecord | None = None
    certificate: CounterexampleCertificate | None = None
    stop_reason: str | None = None


class CounterexamplePipelineError(RuntimeError):
    pass


class CounterexampleVerified(RuntimeError):
    def __init__(self, outcome: CounterexampleOutcome) -> None:
        self.outcome = outcome
        super().__init__(
            f"counterexample verified: "
            f"{outcome.candidate.candidate_id if outcome.candidate else 'unknown'}"
        )


class CounterexampleHalt(RuntimeError):
    def __init__(self, outcome: CounterexampleOutcome) -> None:
        self.outcome = outcome
        super().__init__(outcome.stop_reason or outcome.decision.value)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(path: Path, files: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}
        if actual != dict(files):
            raise CounterexamplePipelineError(f"immutable counterexample artifact differs: {path}")
        return
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        for name, payload in files.items():
            target = temporary / name
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if target.read_bytes() != payload:
                raise CounterexamplePipelineError(
                    f"counterexample artifact write verification failed: {name}"
                )
        _fsync_directory(temporary)
        try:
            os.rename(temporary, path)
        except FileExistsError:
            _publish_directory(path, files)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            for item in temporary.iterdir():
                item.unlink(missing_ok=True)
            temporary.rmdir()


def _graph_properties(graph: GraphState) -> tuple[int, bool]:
    adjacency = [set[int]() for _ in range(graph.order)]
    for u, v in graph.edges:
        if u == v or not (0 <= u < graph.order and 0 <= v < graph.order):
            return 0, False
        adjacency[u].add(v)
        adjacency[v].add(u)
    minimum_degree = min((len(neighbors) for neighbors in adjacency), default=0)
    if not adjacency:
        return minimum_degree, False
    seen = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return minimum_degree, len(seen) == graph.order


class CounterexamplePipeline:
    def __init__(
        self,
        *,
        backend: GraphBackend,
        artifact_root: str | Path,
        target_id: str = "erdos_gyarfas",
        target_contract_version: str = TARGET_CONTRACT_VERSION,
        independent_verifier: Callable[[Path], ExactVerification] | None = None,
        event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        independent_timeout_seconds: float = 300.0,
        independent_memory_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.backend = backend
        self.root = Path(artifact_root) / "counterexamples"
        self.target_id = target_id
        self.target_contract_version = target_contract_version
        self.independent_verifier = independent_verifier
        self.event_callback = event_callback
        self.independent_timeout_seconds = independent_timeout_seconds
        self.independent_memory_bytes = independent_memory_bytes

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, payload)

    def _candidate_id(self, graph6_bytes: bytes, lengths: tuple[int, ...]) -> str:
        fields = (
            self.target_id.encode("utf-8"),
            self.target_contract_version.encode("utf-8"),
            graph6_bytes,
            _canonical(list(lengths)),
        )
        framed = b"".join(len(value).to_bytes(8, "big") + value for value in fields)
        return f"cx-{_sha256(framed)}"

    def _repository_states(self) -> tuple[dict[str, Any], dict[str, Any]]:
        project_root = Path(__file__).resolve().parents[2]
        try:
            mutation_forge = dict(git_state(project_root))
        except (OSError, RuntimeError, subprocess.SubprocessError):
            mutation_forge = {"commit": None, "dirty": None}
        backend_repo = getattr(self.backend, "repo", None)
        heg = {
            "repo": str(backend_repo) if backend_repo is not None else None,
            "commit": getattr(self.backend, "commit", None),
            "dirty": getattr(self.backend, "dirty", None),
        }
        return mutation_forge, heg

    def _save_candidate(
        self,
        graph: GraphState,
        score: GraphScore,
        provenance: CandidateProvenance,
        lengths: tuple[int, ...],
        witness_cap: int,
    ) -> CounterexampleCandidate:
        graph6_bytes = self.backend.serialize_graph6(graph).rstrip("\n").encode("utf-8") + b"\n"
        candidate_id = self._candidate_id(graph6_bytes, lengths)
        candidate_root = self.root / candidate_id
        candidate_dir = candidate_root / "candidate"
        candidate = CounterexampleCandidate(
            candidate_id,
            candidate_dir,
            candidate_dir / "candidate.graph6",
            candidate_dir / "candidate.json",
        )
        if candidate_dir.exists():
            self._read_candidate(candidate)
            self._save_observation(candidate_root, provenance)
            return candidate
        minimum_degree, connected = _graph_properties(graph)
        mutation_forge, heg = self._repository_states()
        metadata = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "target_id": self.target_id,
            "target_contract_version": self.target_contract_version,
            "order": graph.order,
            "edge_count": len(graph.edges),
            "edges": [list(edge) for edge in graph.edges],
            "minimum_degree": minimum_degree,
            "connected": connected,
            "artifact_sha256": _sha256(graph6_bytes),
            "graph6_sha256": _sha256(graph6_bytes),
            "state_hash": self.backend.state_hash(graph),
            "canonical_hash": self.backend.canonical_hash(graph),
            "canonical_hash_authoritative": False,
            "target_forbidden_lengths": list(lengths),
            "graph_score": score.as_dict(),
            "witness_cap": witness_cap,
            "heuristic_complete": score.complete,
            "source_kind": provenance.source_kind,
            "source_id": provenance.source_id,
            "source_path": provenance.metadata.get("source_path"),
            "generation": provenance.generation,
            "slot": provenance.slot,
            "episode_id": provenance.episode_id,
            "policy_id": (
                provenance.source_id
                if provenance.source_kind == "generated_ranker"
                else None
            ),
            "baseline_id": (
                provenance.source_id if provenance.source_kind == "baseline" else None
            ),
            "graph_seed": provenance.graph_seed,
            "policy_seed": provenance.policy_seed,
            "evaluation_step": provenance.evaluation_step,
            "mutation_forge": mutation_forge,
            "heg": heg,
            "discovery": provenance.as_dict(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        candidate_payload = _canonical(metadata) + b"\n"
        seal = {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "files": {
                "candidate.graph6": _sha256(graph6_bytes),
                "candidate.json": _sha256(candidate_payload),
            },
        }
        _publish_directory(
            candidate_dir,
            {
                "candidate.graph6": graph6_bytes,
                "candidate.json": candidate_payload,
                "seal.json": _canonical(seal) + b"\n",
            },
        )
        self._save_observation(candidate_root, provenance)
        return candidate

    def _save_observation(
        self,
        candidate_root: Path,
        provenance: CandidateProvenance,
    ) -> None:
        observation = provenance.as_dict()
        mutation_forge, heg = self._repository_states()
        observation["mutation_forge"] = mutation_forge
        observation["backend"] = {
            "backend_id": getattr(self.backend, "backend_id", "unknown"),
            **heg,
        }
        observation["observed_at"] = datetime.now(UTC).isoformat()
        observation_id = _sha256(_canonical(observation))
        _publish_directory(
            candidate_root / "observations" / observation_id,
            {"observation.json": _canonical(observation) + b"\n"},
        )

    def _read_candidate(
        self, candidate: CounterexampleCandidate
    ) -> tuple[GraphState, dict[str, Any]]:
        graph_bytes = candidate.graph_path.read_bytes()
        metadata_bytes = candidate.metadata_path.read_bytes()
        seal_path = candidate.artifact_directory / "seal.json"
        try:
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CounterexamplePipelineError("candidate seal is unreadable") from exc
        expected_files = {
            "candidate.graph6": _sha256(graph_bytes),
            "candidate.json": _sha256(metadata_bytes),
        }
        if (
            not isinstance(seal, Mapping)
            or seal.get("schema_version") != CANDIDATE_SCHEMA_VERSION
            or seal.get("files") != expected_files
        ):
            raise CounterexamplePipelineError("candidate seal mismatch")
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise CounterexamplePipelineError("candidate metadata is not an object")
        if (
            metadata.get("schema_version") != CANDIDATE_SCHEMA_VERSION
            or metadata.get("candidate_id") != candidate.candidate_id
        ):
            raise CounterexamplePipelineError("candidate identity mismatch")
        graph_sha256 = _sha256(graph_bytes)
        if (
            graph_sha256 != metadata.get("artifact_sha256")
            or graph_sha256 != metadata.get("graph6_sha256")
        ):
            raise CounterexamplePipelineError("candidate graph6 hash mismatch")
        graph = self.backend.deserialize_graph6(graph_bytes.decode("utf-8").rstrip("\n"))
        validation = self.backend.validate(graph)
        if not validation.valid:
            raise CounterexamplePipelineError(
                f"candidate graph validation failed: {validation.errors}"
            )
        if graph.order != metadata.get("order"):
            raise CounterexamplePipelineError("candidate order mismatch")
        if [list(edge) for edge in graph.edges] != metadata.get("edges"):
            raise CounterexamplePipelineError("candidate edge list mismatch")
        if self.backend.state_hash(graph) != metadata.get("state_hash"):
            raise CounterexamplePipelineError("candidate state hash mismatch")
        expected = self.backend.target_forbidden_lengths(graph.order)
        if list(expected) != metadata.get("target_forbidden_lengths"):
            raise CounterexamplePipelineError("candidate target lengths mismatch")
        return graph, metadata

    def _verification_record(
        self,
        candidate: CounterexampleCandidate,
        role: str,
        result: ExactVerification,
    ) -> CounterexampleVerificationRecord:
        payload = {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "verification_role": role,
            "verifier_id": result.implementation,
            "status": result.status,
            "complete": result.complete,
            "message": result.message,
            "witnesses": [[kind, list(vertices)] for kind, vertices in result.witnesses],
            "elapsed_seconds": result.elapsed_seconds,
            "implementation_sha256": result.implementation_sha256,
            "configuration": dict(result.configuration),
            "candidate_graph6_sha256": _sha256(candidate.graph_path.read_bytes()),
        }
        attempt_id = _sha256(_canonical(payload))
        directory = candidate.artifact_directory.parent / "verifications" / role / attempt_id
        record_bytes = _canonical(payload) + b"\n"
        _publish_directory(
            directory,
            {
                "verification.json": record_bytes,
                "seal.json": _canonical({"verification.json": _sha256(record_bytes)}) + b"\n",
            },
        )
        return CounterexampleVerificationRecord(
            role,
            result.status,
            result.complete,
            result.implementation,
            directory,
            directory / "verification.json",
            result.message,
        )

    def _load_verification(
        self,
        candidate: CounterexampleCandidate,
        role: str,
    ) -> CounterexampleVerificationRecord | None:
        root = candidate.artifact_directory.parent / "verifications" / role
        if not root.is_dir():
            return None
        records: list[CounterexampleVerificationRecord] = []
        signatures: set[tuple[str, bool, str, str]] = set()
        candidate_sha256 = _sha256(candidate.graph_path.read_bytes())
        for directory in sorted(item for item in root.iterdir() if item.is_dir()):
            artifact_path = directory / "verification.json"
            seal_path = directory / "seal.json"
            try:
                artifact_bytes = artifact_path.read_bytes()
                payload = json.loads(artifact_bytes)
                seal = json.loads(seal_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CounterexamplePipelineError(
                    f"{role} verification record is unreadable"
                ) from exc
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema_version") != VERIFICATION_SCHEMA_VERSION
                or payload.get("candidate_id") != candidate.candidate_id
                or payload.get("verification_role") != role
                or payload.get("candidate_graph6_sha256") != candidate_sha256
                or seal != {"verification.json": _sha256(artifact_bytes)}
            ):
                raise CounterexamplePipelineError(
                    f"{role} verification record integrity mismatch"
                )
            status = payload.get("status")
            complete = payload.get("complete")
            verifier_id = payload.get("verifier_id")
            message = payload.get("message", "")
            if (
                not isinstance(status, str)
                or not isinstance(complete, bool)
                or not isinstance(verifier_id, str)
                or not isinstance(message, str)
            ):
                raise CounterexamplePipelineError(
                    f"{role} verification record contract mismatch"
                )
            signatures.add((status, complete, verifier_id, message))
            records.append(
                CounterexampleVerificationRecord(
                    role,
                    status,
                    complete,
                    verifier_id,
                    directory,
                    artifact_path,
                    message,
                )
            )
        if len(signatures) > 1:
            raise CounterexamplePipelineError(
                f"conflicting immutable {role} verification records"
            )
        return records[0] if records else None

    def _run_independent(
        self,
        graph_path: Path,
        expected_lengths: tuple[int, ...],
    ) -> ExactVerification:
        if self.independent_verifier is not None:
            result = self.independent_verifier(graph_path)
            if not isinstance(result, ExactVerification):
                raise CounterexamplePipelineError(
                    "independent verifier returned an invalid result"
                )
            configuration = dict(result.configuration)
            configuration.setdefault("target_forbidden_lengths", list(expected_lengths))
            configuration.setdefault("process_isolated", False)
            return replace(result, configuration=configuration)
        command = [
            sys.executable,
            "-m",
            "mutation_forge.independent_verifier",
            str(graph_path),
        ]

        def limits() -> None:
            import resource

            cpu = max(1, int(self.independent_timeout_seconds))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.independent_memory_bytes, self.independent_memory_bytes),
            )

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.independent_timeout_seconds,
                preexec_fn=limits if os.name == "posix" else None,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ExactVerification(
                "UNKNOWN",
                False,
                f"independent verifier unavailable: {exc}",
                "mforge-independent-cycle-mitm-v2",
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, Mapping):
            payload = {}
        observed_lengths = payload.get("target_forbidden_lengths")
        if (
            payload.get("status") in {"VERIFIED", "REJECTED"}
            and observed_lengths != list(expected_lengths)
        ):
            return ExactVerification(
                "INVALID",
                True,
                "independent verifier target lengths mismatch",
                str(payload.get("implementation", "mforge-independent-cycle-mitm-v2")),
                implementation_sha256=(
                    str(payload["implementation_sha256"])
                    if isinstance(payload.get("implementation_sha256"), str)
                    else None
                ),
            )
        return ExactVerification(
            status=str(payload.get("status", "UNKNOWN")),
            complete=payload.get("complete") is True,
            message=str(
                payload.get("message")
                or completed.stderr.strip()
                or f"independent verifier exited {completed.returncode}"
            ),
            implementation=str(
                payload.get(
                    "implementation",
                    "mforge-independent-cycle-mitm-v2",
                )
            ),
            witnesses=tuple(
                (
                    str(item[0]),
                    tuple(int(vertex) for vertex in item[1]),
                )
                for item in payload.get("witnesses", [])
                if isinstance(item, list) and len(item) == 2 and isinstance(item[1], list)
            ),
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            implementation_sha256=(
                str(payload["implementation_sha256"])
                if isinstance(payload.get("implementation_sha256"), str)
                else None
            ),
            configuration={
                "timeout_seconds": self.independent_timeout_seconds,
                "memory_bytes": self.independent_memory_bytes,
                "process_isolated": True,
            },
        )

    def _certificate(
        self,
        candidate: CounterexampleCandidate,
        lengths: tuple[int, ...],
        primary: CounterexampleVerificationRecord,
        independent: CounterexampleVerificationRecord,
    ) -> CounterexampleCertificate:
        evidence = {
            "candidate.graph6": _sha256(candidate.graph_path.read_bytes()),
            "candidate.json": _sha256(candidate.metadata_path.read_bytes()),
            "verification-primary.json": _sha256(primary.artifact_path.read_bytes()),
            "verification-independent.json": _sha256(independent.artifact_path.read_bytes()),
        }
        payload = {
            "schema_version": CERTIFICATE_SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "target_id": self.target_id,
            "target_contract_version": self.target_contract_version,
            "target_forbidden_lengths": list(lengths),
            "primary": {
                "status": primary.status,
                "complete": primary.complete,
                "verifier_id": primary.verifier_id,
                "artifact_ref": str(primary.artifact_path),
            },
            "independent": {
                "status": independent.status,
                "complete": independent.complete,
                "verifier_id": independent.verifier_id,
                "artifact_ref": str(independent.artifact_path),
            },
            "evidence_sha256": evidence,
        }
        certificate_bytes = _canonical(payload) + b"\n"
        certificate_id = _sha256(certificate_bytes)
        directory = candidate.artifact_directory.parent / "certificates" / certificate_id
        _publish_directory(
            directory,
            {
                "certificate.json": certificate_bytes,
                "seal.json": _canonical({"certificate.json": certificate_id}) + b"\n",
            },
        )
        return CounterexampleCertificate(
            candidate.candidate_id,
            certificate_id,
            directory,
            directory / "certificate.json",
            certificate_id,
        )

    def inspect(
        self,
        *,
        graph: GraphState,
        score: GraphScore,
        provenance: CandidateProvenance,
        witness_cap: int,
    ) -> CounterexampleOutcome:
        if witness_cap <= 0:
            raise EvaluationContractError("witness_cap must be positive")
        lengths = self.backend.target_forbidden_lengths(graph.order)
        observed = tuple(length for length, _ in score.capped_cycle_counts)
        counts = tuple(count for _, count in score.capped_cycle_counts)
        if observed != lengths:
            raise EvaluationContractError(
                f"score lengths {observed!r} do not match target lengths {lengths!r}"
            )
        if len(observed) != len(set(observed)) or any(count < 0 for count in counts):
            raise EvaluationContractError("invalid authoritative score count vector")
        if sum(counts) != score.total_capped_witnesses:
            raise EvaluationContractError("authoritative score total mismatch")
        if score.total_capped_witnesses != 0:
            return CounterexampleOutcome(CounterexampleDecision.CONTINUE)
        if not score.valid:
            raise EvaluationContractError("invalid score cannot submit a zero candidate")

        candidate = self._save_candidate(graph, score, provenance, lengths, witness_cap)
        minimum_degree, connected = _graph_properties(graph)
        self._emit(
            "counterexample_candidate_found",
            {
                "candidate_id": candidate.candidate_id,
                "artifact_ref": str(candidate.artifact_directory),
                "order": graph.order,
                "edge_count": len(graph.edges),
                "minimum_degree": minimum_degree,
                "connected": connected,
                "target_forbidden_lengths": list(lengths),
                "heuristic_complete": score.complete,
                "idempotency_key": f"{candidate.candidate_id}:candidate",
            },
        )
        reread, _ = self._read_candidate(candidate)
        primary = self._load_verification(candidate, "primary")
        if primary is None:
            self._emit(
                "counterexample_primary_verification_started",
                {
                    "candidate_id": candidate.candidate_id,
                    "verification_role": "primary",
                    "verifier_id": (
                        f"{getattr(self.backend, 'backend_id', 'backend')}.exact_verify"
                    ),
                    "status": "RUNNING",
                    "complete": False,
                    "artifact_ref": str(candidate.graph_path),
                    "idempotency_key": f"{candidate.candidate_id}:primary:start",
                },
            )
            primary = self._verification_record(
                candidate,
                "primary",
                self.backend.exact_verify(reread),
            )
        self._emit(
            "counterexample_primary_verification_completed",
            {
                "candidate_id": candidate.candidate_id,
                "verification_role": "primary",
                "verifier_id": primary.verifier_id,
                "status": primary.status,
                "complete": primary.complete,
                "artifact_ref": str(primary.artifact_path),
                "idempotency_key": (
                    f"{candidate.candidate_id}:primary:{primary.artifact_directory.name}"
                ),
            },
        )
        if primary.status == "REJECTED":
            return CounterexampleOutcome(
                CounterexampleDecision.PRIMARY_REJECTED,
                candidate,
                primary,
            )
        if primary.status == "INVALID":
            return CounterexampleOutcome(
                CounterexampleDecision.FAIL,
                candidate,
                primary,
                stop_reason="primary_verification_invariant_failure",
            )
        if primary.status != "VERIFIED" or not primary.complete:
            return CounterexampleOutcome(
                CounterexampleDecision.PAUSE_INCONCLUSIVE,
                candidate,
                primary,
                stop_reason="primary_verification_inconclusive",
            )

        independent = self._load_verification(candidate, "independent")
        if independent is None:
            self._emit(
                "counterexample_independent_verification_started",
                {
                    "candidate_id": candidate.candidate_id,
                    "verification_role": "independent",
                    "verifier_id": "mforge-independent-cycle-mitm-v2",
                    "status": "RUNNING",
                    "complete": False,
                    "artifact_ref": str(candidate.graph_path),
                    "idempotency_key": f"{candidate.candidate_id}:independent:start",
                },
            )
            independent = self._verification_record(
                candidate,
                "independent",
                self._run_independent(candidate.graph_path, lengths),
            )
        self._emit(
            "counterexample_independent_verification_completed",
            {
                "candidate_id": candidate.candidate_id,
                "verification_role": "independent",
                "verifier_id": independent.verifier_id,
                "status": independent.status,
                "complete": independent.complete,
                "artifact_ref": str(independent.artifact_path),
                "idempotency_key": (
                    f"{candidate.candidate_id}:independent:{independent.artifact_directory.name}"
                ),
            },
        )
        if independent.status in {"REJECTED", "INVALID"}:
            self._emit(
                "counterexample_verification_conflict",
                {
                    "candidate_id": candidate.candidate_id,
                    "verification_role": "independent",
                    "verifier_id": independent.verifier_id,
                    "status": independent.status,
                    "complete": independent.complete,
                    "artifact_ref": str(independent.artifact_path),
                    "idempotency_key": f"{candidate.candidate_id}:conflict",
                },
            )
            return CounterexampleOutcome(
                CounterexampleDecision.FAIL,
                candidate,
                primary,
                independent,
                stop_reason="verification_conflict",
            )
        if independent.status != "VERIFIED" or not independent.complete:
            return CounterexampleOutcome(
                CounterexampleDecision.PAUSE_INCONCLUSIVE,
                candidate,
                primary,
                independent,
                stop_reason="awaiting_independent_verification",
            )
        certificate = self._certificate(
            candidate,
            lengths,
            primary,
            independent,
        )
        return CounterexampleOutcome(
            CounterexampleDecision.STOP_VERIFIED,
            candidate,
            primary,
            independent,
            certificate,
            "counterexample_verified",
        )


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "CERTIFICATE_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "CandidateProvenance",
    "CounterexampleCandidate",
    "CounterexampleCertificate",
    "CounterexampleDecision",
    "CounterexampleHalt",
    "CounterexampleOutcome",
    "CounterexamplePipeline",
    "CounterexamplePipelineError",
    "CounterexampleVerificationRecord",
    "CounterexampleVerified",
]
