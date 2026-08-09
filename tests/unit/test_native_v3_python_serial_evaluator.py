from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from mutation_forge.backends.base import (
    DeepProposalProfileRecorder,
    InvalidRewriteError,
    ProposalTimingRecorder,
    ScoreProfileRecorder,
)
from mutation_forge.counterexamples import (
    CandidateProvenance,
    CounterexampleDecision,
    CounterexampleOutcome,
)
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
    normalized_edge,
)
from mutation_forge.native_v3.scoring import (
    EXPANDED_NODE_BUDGET,
    INITIAL_NODE_BUDGET,
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EvidenceStatus,
    ScoreEvidence,
    ScoreTimeoutWithoutPartial,
)
from mutation_forge.native_v3.serial_evaluator import (
    CounterexampleInspector,
    SerialEvaluationStatus,
)
from mutation_forge.native_v3_python import (
    PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID,
    SEMANTIC_TRACE_PROTOCOL_ID,
    PolicyInfrastructureError,
    PolicyRuntimeLimitsV1,
    PythonSerialEpisodeConfigV1,
    evaluate_serial_python_policy,
    validate_python_policy_source,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "native_v3_python_m3"
)
TEST_IDENTITY = BackendIdentity(
    backend_id="python-serial-test",
    heg_commit="fixture",
    source_tree_sha256="a" * 64,
    binary_sha256="b" * 64,
    compiler_identity="fixture",
    build_flags=(),
    platform="fixture",
    architecture="fixture",
)


def _source(name: str) -> str:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8")


def _cubic_graph(order: int = 6) -> GraphState:
    edges = {
        normalized_edge((vertex, (vertex + 1) % order)) for vertex in range(order)
    }
    edges.update((vertex, vertex + order // 2) for vertex in range(order // 2))
    return GraphState(order, tuple(sorted(edges)))


def _degrees(graph: GraphState) -> tuple[int, ...]:
    result = [0] * graph.order
    for u, v in graph.edges:
        result[u] += 1
        result[v] += 1
    return tuple(result)


class _Backend:
    backend_id = "python-serial-test"

    def __init__(
        self,
        *,
        zero_after_edges: int | None = None,
        host_error: bool = False,
        interval_mode: str = "exact",
    ) -> None:
        self.zero_after_edges = zero_after_edges
        self.host_error = host_error
        self.interval_mode = interval_mode
        self.score_calls: list[tuple[GraphState, tuple[int, ...], AttemptKind]] = []
        self.apply_calls: list[RewritePlan] = []
        self.raw_graph_score_calls = 0
        self.unique_graph_scores = 0

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        assert order >= 4
        return (4,)

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        assert seed >= 0
        return _cubic_graph(order)

    def validate(self, graph: GraphState) -> GraphValidation:
        valid = (
            graph.order > 0
            and len(set(graph.edges)) == len(graph.edges)
            and all(
                u != v and 0 <= u < graph.order and 0 <= v < graph.order
                for u, v in graph.edges
            )
            and min(_degrees(graph)) >= 3
        )
        return GraphValidation(valid, () if valid else ("invalid graph",))

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: ScoreProfileRecorder | None = None,
    ) -> GraphScore | None:
        del cutoff, record_profile
        total = min(witness_cap, max(0, 20 - len(graph.edges)))
        return GraphScore(
            valid=True,
            capped_cycle_counts=((4, total),),
            total_capped_witnesses=total,
            weighted_penalty=total,
            complete=True,
            ordering_key=(total, len(graph.edges)),
        )

    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        lengths = tuple(forbidden_lengths or self.target_forbidden_lengths(graph.order))
        assert lengths == (4,)
        self.score_calls.append((graph, lengths, attempt_kind))
        self.raw_graph_score_calls += 1
        self.unique_graph_scores += 1
        candidate = len(graph.edges) > len(_cubic_graph(graph.order).edges)
        exact_total = min(witness_cap, max(0, 20 - len(graph.edges)))
        if self.zero_after_edges is not None and len(graph.edges) >= self.zero_after_edges:
            exact_total = 0
        lower = upper = exact_total
        status = EvidenceStatus.EXACT
        if self.interval_mode == "safe_partial" and candidate:
            lower, upper = 0, 1
            status = EvidenceStatus.SEARCH_TIMEOUT_WITH_SAFE_PARTIAL
        elif self.interval_mode == "overlap":
            lower = 4 if candidate else 5
            upper = 8 if attempt_kind is AttemptKind.EXPANDED else (9 if candidate else 10)
            status = (
                EvidenceStatus.SEARCH_TIMEOUT_WITH_SAFE_PARTIAL
                if attempt_kind is AttemptKind.EXPANDED
                else EvidenceStatus.SEARCH_BUDGET_EXHAUSTED
            )
        return ScoreEvidence(
            graph_content_hash=self.state_hash(graph),
            order=graph.order,
            edge_count=len(graph.edges),
            witness_cap=witness_cap,
            components=(
                CycleComponentEvidence(
                    forbidden_length=4,
                    observed_count=lower,
                    lower_bound=lower,
                    upper_bound=upper,
                    status=status,
                    node_budget=(
                        INITIAL_NODE_BUDGET
                        if attempt_kind is AttemptKind.INITIAL
                        else EXPANDED_NODE_BUDGET
                    ),
                    nodes_visited=1,
                    wall_time_ns=0,
                    attempt_kind=attempt_kind,
                    backend_identity=TEST_IDENTITY,
                ),
            ),
        )

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        del graph
        return ExactVerification("REJECTED", True, "fixture", "fixture")

    def canonical_hash(self, graph: GraphState) -> str:
        return self.state_hash(graph)

    def state_hash(self, graph: GraphState) -> str:
        return hashlib.sha256(repr((graph.order, graph.edges)).encode()).hexdigest()

    def serialize_graph6(self, graph: GraphState) -> str:
        return repr((graph.order, graph.edges))

    def deserialize_graph6(self, value: str) -> GraphState:
        raise NotImplementedError(value)

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: ScoreProfileRecorder | None = None,
    ) -> GraphState:
        del record_score_profile
        if self.host_error:
            raise RuntimeError("trusted backend failed")
        self.apply_calls.append(rewrite)
        current = set(graph.edges)
        removed = set(rewrite.removed_edges)
        added = set(rewrite.added_edges)
        if not removed.issubset(current) or added.intersection(current - removed):
            raise InvalidRewriteError("illegal rewrite delta")
        candidate = GraphState(
            graph.order,
            tuple(sorted((current - removed).union(added))),
        )
        validation = self.validate(candidate)
        if not validation.valid:
            raise InvalidRewriteError("illegal final graph")
        return candidate

    def propose_rewrite(
        self,
        graph: GraphState,
        *,
        operator_family: str,
        policy_seed: int,
        evaluation: int,
        record_timing: ProposalTimingRecorder | None = None,
        record_deep_profile: DeepProposalProfileRecorder | None = None,
    ) -> RewritePlan:
        del (
            graph,
            operator_family,
            policy_seed,
            evaluation,
            record_timing,
            record_deep_profile,
        )
        raise AssertionError("M3 must not call backend proposal generation")

    def close(self) -> None:
        return None


class _Inspector(CounterexampleInspector):
    def __init__(self) -> None:
        self.calls: list[CandidateProvenance] = []

    def inspect(
        self,
        *,
        graph: GraphState,
        score: GraphScore,
        provenance: CandidateProvenance,
        witness_cap: int,
    ) -> CounterexampleOutcome:
        del graph, witness_cap
        assert score.total_capped_witnesses == 0
        self.calls.append(provenance)
        return CounterexampleOutcome(CounterexampleDecision.CONTINUE)


def _config(*, horizon: int = 1, episode_id: str = "python-serial") -> PythonSerialEpisodeConfigV1:
    return PythonSerialEpisodeConfigV1(
        order=6,
        graph_seed=11,
        policy_seed=17,
        horizon=horizon,
        witness_cap=100,
        episode_id=episode_id,
        forbidden_lengths=(4,),
    )


def _evaluate(
    source: str,
    *,
    backend: _Backend | None = None,
    config: PythonSerialEpisodeConfigV1 | None = None,
    limits: PolicyRuntimeLimitsV1 | None = None,
    inspector: CounterexampleInspector | None = None,
):
    selected_backend = backend or _Backend()
    return evaluate_serial_python_policy(
        backend=selected_backend,
        scorer=selected_backend,
        source=source,
        config=config or _config(),
        runtime_limits=limits,
        counterexample_pipeline=inspector,
    )


def test_checked_in_fixture_manifest_is_complete_and_m1_validated() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "mforge.native.python_m3_fixture_manifest.v1"
    assert {case["id"] for case in manifest["cases"]} == {
        "no-plan",
        "add-edge",
        "no-effect",
        "remove-edge",
        "invalid-return",
        "timeout",
        "invalid-import",
    }
    for case in manifest["cases"]:
        validation = validate_python_policy_source(_source(case["path"]))
        assert validation.valid is (case["validation"] == "VALID")


def test_no_plan_consumes_every_step_without_candidate_scoring() -> None:
    backend = _Backend()
    result = _evaluate(
        _source("no_plan.py"),
        backend=backend,
        config=_config(horizon=2),
    )
    scientific = result.scientific_result
    assert scientific.status is SerialEvaluationStatus.COMPLETE
    assert [step.outcome for step in scientific.steps] == ["no_plan", "no_plan"]
    assert [step.no_plan_reason for step in scientific.steps] == [
        "EXPLICIT",
        "EXPLICIT",
    ]
    assert len(backend.score_calls) == 1
    assert not backend.apply_calls


def test_fixture_replay_preserves_science_trace_and_separate_behavior_identity() -> None:
    first = _evaluate(_source("add_edge.py"), backend=_Backend())
    second = _evaluate(_source("add_edge.py"), backend=_Backend())
    assert first.scientific_result.as_dict(include_telemetry=False) == (
        second.scientific_result.as_dict(include_telemetry=False)
    )
    assert first.behavior_identity == second.behavior_identity
    assert first.runtime_profile["sandbox_wall_seconds"] > 0
    assert first.runtime_profile["selector_wall_seconds"] > 0
    assert first.runtime_profile["action_wall_seconds"] > 0
    assert "runtime_profile" in first.as_dict()
    assert first.scientific_result.protocol_id == PYTHON_SERIAL_EVALUATOR_PROTOCOL_ID
    assert (
        first.scientific_result.execution_trace_protocol_id
        == SEMANTIC_TRACE_PROTOCOL_ID
    )
    alternate_source = _source("add_edge.py").replace("candidates", "options")
    alternate = _evaluate(alternate_source, backend=_Backend())
    assert (
        first.program_identity.program_hash
        != alternate.program_identity.program_hash
    )
    assert first.behavior_identity == alternate.behavior_identity


def test_one_host_minted_rewrite_gets_one_authoritative_candidate_score() -> None:
    backend = _Backend()
    result = _evaluate(_source("add_edge.py"), backend=backend)
    step = result.scientific_result.steps[0]
    assert step.outcome == "rewrite"
    assert step.rewrite is not None
    assert step.rewrite.operator_family == "native_v3_python_policy"
    assert step.candidate_evidence is not None
    assert step.accepted and step.acceptance_proved
    assert len(backend.apply_calls) == 1
    assert len(backend.score_calls) == 2
    assert all(lengths == (4,) for _, lengths, _ in backend.score_calls)


def test_candidate_scoring_timeout_is_inconclusive_without_safe_evidence() -> None:
    class CandidateTimeoutBackend(_Backend):
        def score_evidence(
            self,
            graph: GraphState,
            *,
            witness_cap: int,
            forbidden_lengths: Iterable[int] | None = None,
            attempt_kind: AttemptKind = AttemptKind.INITIAL,
        ) -> ScoreEvidence:
            if len(graph.edges) > len(_cubic_graph(graph.order).edges):
                self.raw_graph_score_calls += 1
                self.unique_graph_scores += 1
                raise ScoreTimeoutWithoutPartial("candidate score timed out")
            return super().score_evidence(
                graph,
                witness_cap=witness_cap,
                forbidden_lengths=forbidden_lengths,
                attempt_kind=attempt_kind,
            )

    result = _evaluate(_source("add_edge.py"), backend=CandidateTimeoutBackend())
    science = result.scientific_result

    assert science.status is SerialEvaluationStatus.INCONCLUSIVE_UNSAFE_TIMEOUT
    assert science.failure is None
    assert science.scientific_error == (
        "candidate scoring timed out without safe partial evidence"
    )
    assert science.fitness_interval.lower == 0
    assert science.fitness_interval.upper == 1
    assert science.accepted_rewrites == 0
    assert len(science.steps) == 1
    assert science.steps[0].outcome == "score_timeout_without_partial"
    assert science.steps[0].candidate_evidence is None
    assert not science.steps[0].accepted


def test_no_effect_and_illegal_final_state_do_not_score_candidates() -> None:
    no_effect_backend = _Backend()
    no_effect = _evaluate(_source("no_effect.py"), backend=no_effect_backend)
    assert no_effect.scientific_result.steps[0].no_plan_reason == "NO_EFFECT"
    assert len(no_effect_backend.score_calls) == 1

    class RejectingBackend(_Backend):
        def apply_rewrite(
            self,
            graph: GraphState,
            rewrite: RewritePlan,
            *,
            record_score_profile: ScoreProfileRecorder | None = None,
        ) -> GraphState:
            del graph, rewrite, record_score_profile
            raise InvalidRewriteError("fixture final-state rejection")

    rejecting_backend = RejectingBackend()
    illegal = _evaluate(_source("add_edge.py"), backend=rejecting_backend)
    assert illegal.scientific_result.steps[0].no_plan_reason == "ILLEGAL_FINAL_STATE"
    assert len(rejecting_backend.score_calls) == 1

    class MismatchedBackend(_Backend):
        def apply_rewrite(
            self,
            graph: GraphState,
            rewrite: RewritePlan,
            *,
            record_score_profile: ScoreProfileRecorder | None = None,
        ) -> GraphState:
            del rewrite, record_score_profile
            return graph

    mismatched_backend = MismatchedBackend()
    mismatched = _evaluate(_source("add_edge.py"), backend=mismatched_backend)
    assert (
        mismatched.scientific_result.steps[0].no_plan_reason
        == "ILLEGAL_FINAL_STATE"
    )
    assert len(mismatched_backend.score_calls) == 1


def test_program_failure_and_timeout_are_worst_candidate_fitness() -> None:
    invalid = _evaluate(_source("invalid_return.py"))
    invalid_science = invalid.scientific_result
    assert invalid_science.status is SerialEvaluationStatus.PROGRAM_FAILURE
    assert invalid_science.failure is not None
    assert invalid_science.failure.code == "INVALID_RETURN"
    assert invalid_science.fitness_interval.lower == 0
    assert invalid_science.fitness_interval.upper == 0

    timed = _evaluate(
        _source("timeout.py"),
        limits=PolicyRuntimeLimitsV1(propose_wall_seconds=0.0001),
    )
    timed_science = timed.scientific_result
    assert timed_science.status is SerialEvaluationStatus.PROGRAM_FAILURE
    assert timed_science.failure is not None
    assert timed_science.failure.code == "PROPOSE_TIMEOUT"
    assert timed_science.fitness_interval.lower == 0
    assert timed_science.fitness_interval.upper == 0


def test_invalid_source_and_infrastructure_fail_without_scientific_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mutation_forge.native_v3_python.serial_evaluator as serial_module

    worker_starts = 0
    original_worker = serial_module.IsolatedPolicyWorkerV1

    def counted_worker(*args: object, **kwargs: object):
        nonlocal worker_starts
        worker_starts += 1
        return original_worker(*args, **kwargs)

    monkeypatch.setattr(serial_module, "IsolatedPolicyWorkerV1", counted_worker)
    with pytest.raises(ValueError, match="failed M1 validation"):
        _evaluate(_source("invalid_import.py"))
    assert worker_starts == 0

    with pytest.raises(PolicyInfrastructureError, match="rewrite host failed"):
        _evaluate(_source("add_edge.py"), backend=_Backend(host_error=True))

    class BrokenValueBackend(_Backend):
        def apply_rewrite(
            self,
            graph: GraphState,
            rewrite: RewritePlan,
            *,
            record_score_profile: ScoreProfileRecorder | None = None,
        ) -> GraphState:
            del graph, rewrite, record_score_profile
            raise ValueError("trusted backend internal failure")

    with pytest.raises(PolicyInfrastructureError, match="rewrite host failed"):
        _evaluate(_source("add_edge.py"), backend=BrokenValueBackend())


def test_partial_interval_dominance_and_overlap_remain_conservative() -> None:
    safe = _evaluate(
        _source("add_edge.py"),
        backend=_Backend(interval_mode="safe_partial"),
    )
    assert safe.scientific_result.steps[0].acceptance_proved
    assert safe.scientific_result.steps[0].accepted

    overlap_backend = _Backend(interval_mode="overlap")
    overlap = _evaluate(_source("add_edge.py"), backend=overlap_backend)
    assert not overlap.scientific_result.steps[0].acceptance_proved
    assert not overlap.scientific_result.steps[0].accepted
    assert len(overlap_backend.score_calls) == 4


def test_heuristic_zero_only_submits_to_exact_verification_boundary() -> None:
    backend = _Backend(zero_after_edges=len(_cubic_graph().edges) + 1)
    inspector = _Inspector()
    result = _evaluate(
        _source("add_edge.py"),
        backend=backend,
        inspector=inspector,
    )
    counterexample = result.scientific_result.steps[0].counterexample
    assert counterexample is not None
    assert counterexample.decision == CounterexampleDecision.CONTINUE.value
    assert counterexample.primary_status is None
    assert counterexample.independent_status is None
    assert len(inspector.calls) == 1
    assert inspector.calls[0].source_kind == "native_v3_python_fixture"
    assert inspector.calls[0].source_id == result.program_identity.program_hash


def test_python_episode_rejects_non_authoritative_forbidden_lengths() -> None:
    with pytest.raises(ValueError, match="authoritative backend target"):
        _evaluate(
            _source("no_plan.py"),
            config=PythonSerialEpisodeConfigV1(
                order=6,
                graph_seed=11,
                policy_seed=17,
                horizon=1,
                witness_cap=100,
                episode_id="wrong-lengths",
                forbidden_lengths=(6,),
            ),
        )
