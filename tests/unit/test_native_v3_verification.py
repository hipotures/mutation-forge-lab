from __future__ import annotations

import time
from pathlib import Path

from mutation_forge.models import ExactVerification, GraphState
from mutation_forge.native_v3.verification import (
    VerificationDecision,
    VerificationJob,
    VerificationProfile,
    VerificationSupervisor,
    graph_content_hash,
    verify_independent_python,
)


def _verified(_graph: GraphState) -> ExactVerification:
    return ExactVerification(
        status="VERIFIED",
        complete=True,
        message="verified",
        implementation="test-primary",
    )


def _rejected(_graph: GraphState) -> ExactVerification:
    return ExactVerification(
        status="REJECTED",
        complete=True,
        message="witness found",
        implementation="test-primary",
        witnesses=(("C4", (0, 1, 2, 3)),),
    )


def _slow_verified(graph: GraphState) -> ExactVerification:
    time.sleep(0.2)
    return _verified(graph)


def _graph() -> GraphState:
    return GraphState(
        4,
        ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    )


def test_profile_locks_operator_selected_limits() -> None:
    profile = VerificationProfile()
    assert profile.concurrency == 1
    assert profile.queue_capacity == 16
    assert profile.verifier_wall_timeout_seconds == 600
    assert profile.verifier_memory_bytes == 4 * 1024**3


def test_supervisor_persists_then_runs_both_verifiers_and_deduplicates(
    tmp_path: Path,
) -> None:
    graph = _graph()
    job = VerificationJob(
        graph_content_hash(graph),
        graph,
        {"source": "generated", "episode_id": "episode"},
    )
    with VerificationSupervisor(
        artifact_root=tmp_path,
        primary_verifier=_verified,
        independent_verifier=_verified,
    ) as supervisor:
        first = supervisor.submit(job)
        second = supervisor.submit(job)
        assert first is second
        outcome = first.result(timeout=10)
    assert outcome.decision is VerificationDecision.VERIFIED
    assert (outcome.artifact_directory / "candidate.json").is_file()
    assert (outcome.artifact_directory / "verification-primary.json").is_file()
    assert (outcome.artifact_directory / "verification-independent.json").is_file()
    assert (outcome.artifact_directory / "certificate.json").is_file()
    assert (outcome.artifact_directory / "outcome.json").is_file()
    assert len((outcome.artifact_directory / "provenance.ndjson").read_text().splitlines()) == 2


def test_independent_verifier_is_not_run_after_primary_rejection(tmp_path: Path) -> None:
    graph = _graph()
    job = VerificationJob(graph_content_hash(graph), graph, {"source": "seed"})
    with VerificationSupervisor(
        artifact_root=tmp_path,
        primary_verifier=_rejected,
        independent_verifier=_verified,
    ) as supervisor:
        outcome = supervisor.submit(job).result(timeout=10)
    assert outcome.decision is VerificationDecision.INCONCLUSIVE
    assert (outcome.artifact_directory / "verification-primary.json").is_file()
    assert not (outcome.artifact_directory / "verification-independent.json").exists()
    assert not (outcome.artifact_directory / "certificate.json").exists()


def test_candidate_is_durable_before_verifier_completion(tmp_path: Path) -> None:
    graph = _graph()
    job = VerificationJob(graph_content_hash(graph), graph, {"source": "generated"})
    with VerificationSupervisor(
        artifact_root=tmp_path,
        primary_verifier=_slow_verified,
        independent_verifier=_verified,
    ) as supervisor:
        future = supervisor.submit(job)
        directory = tmp_path / job.verification_protocol_id / job.graph_hash
        assert (directory / "candidate.json").is_file()
        assert not future.done()
        assert future.result(timeout=10).decision is VerificationDecision.VERIFIED


def test_supervisor_recovers_a_durable_nonterminal_candidate(tmp_path: Path) -> None:
    graph = _graph()
    job = VerificationJob(graph_content_hash(graph), graph, {"source": "recovery"})
    directory = tmp_path / job.verification_protocol_id / job.graph_hash
    VerificationSupervisor._persist_candidate(directory, job)  # noqa: SLF001
    with VerificationSupervisor(
        artifact_root=tmp_path,
        primary_verifier=_verified,
        independent_verifier=_verified,
    ) as supervisor:
        recovered = supervisor.recover_pending()
        assert len(recovered) == 1
        assert recovered[0].result(timeout=10).decision is VerificationDecision.VERIFIED


def test_independent_verifier_rejects_a_graph_with_forbidden_cycle() -> None:
    result = verify_independent_python(_graph())
    assert result.complete
    assert result.status != "VERIFIED"
