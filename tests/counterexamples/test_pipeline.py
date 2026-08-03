from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.counterexamples import (
    CandidateProvenance,
    CounterexampleDecision,
    CounterexamplePipeline,
    CounterexamplePipelineError,
)
from mutation_forge.independent_verifier import verify
from mutation_forge.models import ExactVerification, GraphScore, GraphState
from mutation_forge.proposals.k_switch import EvaluationContractError


def _zero_score(*, complete: bool = False) -> GraphScore:
    return GraphScore(True, ((4, 0),), 0, 0, complete, (0,))


def _petersen() -> GraphState:
    return GraphState(
        10,
        (
            (0, 1),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 6),
            (2, 3),
            (2, 7),
            (3, 4),
            (3, 8),
            (4, 9),
            (5, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (7, 9),
        ),
    )


def _verified(_: Path) -> ExactVerification:
    return ExactVerification("VERIFIED", True, "independent", "test-independent")


def _graph6(graph: GraphState) -> bytes:
    assert graph.order <= 62
    bits = [
        int((low, high) in graph.edges) for high in range(1, graph.order) for low in range(high)
    ]
    while len(bits) % 6:
        bits.append(0)
    encoded = bytes(
        63 + sum(bit << (5 - offset) for offset, bit in enumerate(bits[index : index + 6]))
        for index in range(0, len(bits), 6)
    )
    return bytes([graph.order + 63]) + encoded + b"\n"


def test_pipeline_seals_candidate_and_two_verifications(tmp_path: Path) -> None:
    backend = ToyBackend()
    pipeline = CounterexamplePipeline(
        backend=backend,
        artifact_root=tmp_path,
        independent_verifier=_verified,
    )

    outcome = pipeline.inspect(
        graph=_petersen(),
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "test"),
    )

    assert outcome.decision is CounterexampleDecision.STOP_VERIFIED
    assert outcome.candidate is not None
    assert outcome.certificate is not None
    assert outcome.candidate.graph_path.is_file()
    assert (outcome.candidate.artifact_directory / "verification-primary.json").is_file()
    assert (outcome.candidate.artifact_directory / "verification-independent.json").is_file()
    assert (outcome.candidate.artifact_directory / "certificate.json").is_file()
    assert outcome.primary is not None and outcome.primary.status == "VERIFIED"
    assert outcome.independent is not None
    assert outcome.independent.status == "VERIFIED"
    assert outcome.certificate.artifact_path.is_file()
    metadata = json.loads(outcome.candidate.metadata_path.read_text(encoding="utf-8"))
    assert metadata["witness_cap"] == 64
    assert metadata["artifact_sha256"] == hashlib.sha256(
        outcome.candidate.graph_path.read_bytes()
    ).hexdigest()
    assert metadata["source_kind"] == "baseline"
    assert metadata["baseline_id"] == "test"
    assert metadata["policy_id"] is None
    assert set(metadata["mutation_forge"]) == {"commit", "dirty"}
    assert set(metadata["heg"]) == {"repo", "commit", "dirty"}


def test_pipeline_detects_candidate_tampering(tmp_path: Path) -> None:
    backend = ToyBackend()
    pipeline = CounterexamplePipeline(
        backend=backend,
        artifact_root=tmp_path,
        independent_verifier=_verified,
    )
    graph = _petersen()
    outcome = pipeline.inspect(
        graph=graph,
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "test"),
    )
    assert outcome.candidate is not None
    outcome.candidate.graph_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(CounterexamplePipelineError):
        pipeline.inspect(
            graph=graph,
            score=_zero_score(),
            witness_cap=64,
            provenance=CandidateProvenance("baseline", "test"),
        )


def test_primary_rejection_continues_search(tmp_path: Path) -> None:
    backend = ToyBackend()
    graph = GraphState(4, ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    outcome = CounterexamplePipeline(
        backend=backend,
        artifact_root=tmp_path,
        independent_verifier=_verified,
    ).inspect(
        graph=graph,
        score=_zero_score(complete=True),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "test"),
    )
    assert outcome.decision is CounterexampleDecision.PRIMARY_REJECTED


def test_independent_unknown_pauses_without_certificate(tmp_path: Path) -> None:
    outcome = CounterexamplePipeline(
        backend=ToyBackend(),
        artifact_root=tmp_path,
        independent_verifier=lambda _: ExactVerification(
            "UNKNOWN",
            False,
            "resource limit",
            "test-independent",
        ),
    ).inspect(
        graph=_petersen(),
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "test"),
    )
    assert outcome.decision is CounterexampleDecision.PAUSE_INCONCLUSIVE
    assert outcome.stop_reason == "awaiting_independent_verification"
    assert outcome.certificate is None


def test_primary_unknown_pauses_before_independent_verification(tmp_path: Path) -> None:
    class InconclusiveBackend(ToyBackend):
        def exact_verify(self, graph: GraphState) -> ExactVerification:
            return ExactVerification(
                "UNKNOWN",
                False,
                "primary resource limit",
                "test-primary",
            )

    independent_called = False

    def independent(_: Path) -> ExactVerification:
        nonlocal independent_called
        independent_called = True
        return _verified(_)

    outcome = CounterexamplePipeline(
        backend=InconclusiveBackend(),
        artifact_root=tmp_path,
        independent_verifier=independent,
    ).inspect(
        graph=_petersen(),
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "test"),
    )

    assert outcome.decision is CounterexampleDecision.PAUSE_INCONCLUSIVE
    assert outcome.stop_reason == "primary_verification_inconclusive"
    assert outcome.candidate is not None
    assert outcome.primary is not None and outcome.primary.status == "UNKNOWN"
    assert outcome.independent is None
    assert independent_called is False


def test_unknown_verification_is_retried_on_resume(tmp_path: Path) -> None:
    class RetryBackend(ToyBackend):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def exact_verify(self, graph: GraphState) -> ExactVerification:
            self.calls += 1
            if self.calls == 1:
                return ExactVerification("UNKNOWN", False, "busy", "test-primary")
            return ExactVerification("VERIFIED", True, "verified", "test-primary")

    backend = RetryBackend()
    pipeline = CounterexamplePipeline(
        backend=backend,
        artifact_root=tmp_path,
        independent_verifier=_verified,
    )
    first = pipeline.inspect(
        graph=_petersen(),
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "first"),
    )
    assert first.decision is CounterexampleDecision.PAUSE_INCONCLUSIVE
    resumed = pipeline.inspect(
        graph=_petersen(),
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "resume"),
    )
    assert resumed.decision is CounterexampleDecision.STOP_VERIFIED
    assert backend.calls == 2


def test_completed_verifications_are_reused_on_resume(tmp_path: Path) -> None:
    class CountingBackend(ToyBackend):
        def __init__(self) -> None:
            super().__init__()
            self.exact_calls = 0

        def exact_verify(self, graph: GraphState) -> ExactVerification:
            self.exact_calls += 1
            return super().exact_verify(graph)

    backend = CountingBackend()
    independent_calls = 0

    def independent(_: Path) -> ExactVerification:
        nonlocal independent_calls
        independent_calls += 1
        return ExactVerification("VERIFIED", True, "independent", "test-independent")

    pipeline = CounterexamplePipeline(
        backend=backend,
        artifact_root=tmp_path,
        independent_verifier=independent,
    )
    first = pipeline.inspect(
        graph=_petersen(),
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "first"),
    )
    resumed = pipeline.inspect(
        graph=_petersen(),
        score=_zero_score(),
        witness_cap=64,
        provenance=CandidateProvenance("baseline", "resume"),
    )

    assert first.decision is CounterexampleDecision.STOP_VERIFIED
    assert resumed.decision is CounterexampleDecision.STOP_VERIFIED
    assert resumed.certificate == first.certificate
    assert backend.exact_calls == 1
    assert independent_calls == 1


def test_score_length_or_total_mismatch_is_contract_failure(
    tmp_path: Path,
) -> None:
    pipeline = CounterexamplePipeline(
        backend=ToyBackend(),
        artifact_root=tmp_path,
        independent_verifier=_verified,
    )
    with pytest.raises(EvaluationContractError, match="score lengths"):
        pipeline.inspect(
            graph=_petersen(),
            score=GraphScore(True, ((8, 0),), 0, 0, True, (0,)),
            witness_cap=64,
            provenance=CandidateProvenance("baseline", "test"),
        )
    with pytest.raises(EvaluationContractError, match="total mismatch"):
        pipeline.inspect(
            graph=_petersen(),
            score=GraphScore(True, ((4, 1),), 0, 0, True, (0,)),
            witness_cap=64,
            provenance=CandidateProvenance("baseline", "test"),
        )


def test_invalid_zero_is_an_invariant_failure(tmp_path: Path) -> None:
    pipeline = CounterexamplePipeline(
        backend=ToyBackend(),
        artifact_root=tmp_path,
        independent_verifier=_verified,
    )
    with pytest.raises(EvaluationContractError, match="invalid score"):
        pipeline.inspect(
            graph=_petersen(),
            score=GraphScore(False, ((4, 0),), 0, 0, True, (1,)),
            witness_cap=64,
            provenance=CandidateProvenance("baseline", "test"),
        )


def test_independent_verifier_rejects_target_cycles(
    tmp_path: Path,
) -> None:
    petersen = tmp_path / "petersen.g6"
    petersen.write_bytes(_graph6(_petersen()))
    petersen_result = verify(petersen)
    assert petersen_result["status"] == "REJECTED"
    assert petersen_result["witnesses"][0][0] == "C8"

    k4 = tmp_path / "k4.g6"
    k4.write_bytes(
        _graph6(
            GraphState(
                4,
                ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
            )
        )
    )
    result = verify(k4)
    assert result["status"] == "REJECTED"
    assert result["complete"] is True
