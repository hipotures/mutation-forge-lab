from __future__ import annotations

import hashlib
import json

from mutation_forge.backends.base import (
    DeepProposalProfileRecorder,
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
from mutation_forge.native_v3.contracts import (
    PROGRAM_SCHEMA_VERSION,
    ValidatedProgram,
    validate_program,
)
from mutation_forge.native_v3.interpreter import InterpreterLimits
from mutation_forge.native_v3.scoring import (
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EvidenceStatus,
    ScoreEvidence,
    ScoreTimeoutWithoutPartial,
)
from mutation_forge.native_v3.serial_evaluator import (
    CounterexampleInspector,
    SerialEpisodeConfig,
    SerialEvaluationStatus,
    evaluate_serial_program,
)

TEST_IDENTITY = BackendIdentity(
    backend_id="serial-test-backend",
    heg_commit="fixture",
    source_tree_sha256="a" * 64,
    binary_sha256="b" * 64,
    compiler_identity="fixture",
    build_flags=(),
    platform="fixture",
    architecture="fixture",
)


def _validated(entry: object) -> ValidatedProgram:
    validation = validate_program(
        json.dumps(
            {"schema_version": PROGRAM_SCHEMA_VERSION, "entry": entry},
            separators=(",", ":"),
        )
    )
    assert validation.valid
    assert validation.program is not None
    return validation.program


def _add_program() -> ValidatedProgram:
    return _validated(
        {
            "op": "let",
            "name": "edge",
            "value": {
                "op": "pick",
                "source": {
                    "op": "selector",
                    "selector_id": "non_edges_legal",
                    "arguments": {},
                },
                "mode": "seeded_uniform",
            },
            "body": {
                "op": "block",
                "children": [
                    {
                        "op": "apply",
                        "action_id": "add_edge",
                        "arguments": {"edge": {"op": "ref", "name": "edge"}},
                    },
                    {"op": "emit"},
                ],
            },
        }
    )


def _remove_program() -> ValidatedProgram:
    return _validated(
        {
            "op": "let",
            "name": "edge",
            "value": {
                "op": "pick",
                "source": {
                    "op": "selector",
                    "selector_id": "edges_removable",
                    "arguments": {},
                },
                "mode": "seeded_uniform",
            },
            "body": {
                "op": "block",
                "children": [
                    {
                        "op": "apply",
                        "action_id": "remove_edge",
                        "arguments": {"edge": {"op": "ref", "name": "edge"}},
                    },
                    {"op": "emit"},
                ],
            },
        }
    )


def _cubic_graph(order: int = 6) -> GraphState:
    edges = {
        normalized_edge((vertex, (vertex + 1) % order))
        for vertex in range(order)
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
    backend_id = "serial-test-backend"

    def __init__(self, *, zero_after_edges: int | None = None) -> None:
        self.score_calls: list[GraphState] = []
        self.apply_calls: list[RewritePlan] = []
        self.zero_after_edges = zero_after_edges
        self.raw_graph_score_calls = 0
        self.unique_graph_scores = 0

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        del order
        return (4,)

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        del seed
        return _cubic_graph(order)

    def validate(self, graph: GraphState) -> GraphValidation:
        edge_set = set(graph.edges)
        valid = (
            graph.order > 0
            and len(edge_set) == len(graph.edges)
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
        del witness_cap, cutoff, record_profile
        self.score_calls.append(graph)
        total = max(0, 20 - len(graph.edges))
        if self.zero_after_edges is not None and len(graph.edges) >= self.zero_after_edges:
            total = 0
        return GraphScore(
            valid=True,
            capped_cycle_counts=((4, total),),
            total_capped_witnesses=total,
            weighted_penalty=total,
            complete=True,
            ordering_key=(total, -len(graph.edges)),
        )

    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: object = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        del forbidden_lengths
        self.score_calls.append(graph)
        self.raw_graph_score_calls += 1
        self.unique_graph_scores += 1
        total = max(0, 20 - len(graph.edges))
        if (
            self.zero_after_edges is not None
            and len(graph.edges) >= self.zero_after_edges
        ):
            total = 0
        total = min(total, witness_cap)
        return ScoreEvidence(
            graph_content_hash=self.state_hash(graph),
            order=graph.order,
            edge_count=len(graph.edges),
            witness_cap=witness_cap,
            components=(
                CycleComponentEvidence(
                    forbidden_length=4,
                    observed_count=total,
                    lower_bound=total,
                    upper_bound=total,
                    status=EvidenceStatus.EXACT,
                    node_budget=(
                        50_000
                        if attempt_kind is AttemptKind.INITIAL
                        else 200_000
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
        return ExactVerification(
            status="REJECTED",
            complete=True,
            message="test",
            implementation="serial-test",
        )

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
        self.apply_calls.append(rewrite)
        current = set(graph.edges)
        if not set(rewrite.removed_edges).issubset(current):
            raise ValueError("missing removed edge")
        remaining = current.difference(rewrite.removed_edges)
        if set(rewrite.added_edges).intersection(remaining):
            raise ValueError("existing added edge")
        candidate = GraphState(
            graph.order,
            tuple(sorted(remaining.union(rewrite.added_edges))),
        )
        validation = self.validate(candidate)
        if not validation.valid:
            raise ValueError("invalid candidate")
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
        raise AssertionError("serial evaluator must not call backend proposal generation")

    def close(self) -> None:
        return None


class _Inspector(CounterexampleInspector):
    def __init__(self) -> None:
        self.calls: list[tuple[GraphState, int, CandidateProvenance]] = []

    def inspect(
        self,
        *,
        graph: GraphState,
        score: GraphScore,
        provenance: CandidateProvenance,
        witness_cap: int,
    ) -> CounterexampleOutcome:
        assert score.total_capped_witnesses == 0
        self.calls.append((graph, witness_cap, provenance))
        return CounterexampleOutcome(CounterexampleDecision.CONTINUE)


CONFIG = SerialEpisodeConfig(
    order=6,
    graph_seed=11,
    policy_seed=17,
    horizon=2,
    witness_cap=100,
    episode_id="serial-fixture",
)


def test_fixed_fixture_replays_identical_semantic_trace_and_hash() -> None:
    first_backend = _Backend()
    first = evaluate_serial_program(
        backend=first_backend,
        scorer=first_backend,
        program=_add_program(),
        config=CONFIG,
    )
    second_backend = _Backend()
    second = evaluate_serial_program(
        backend=second_backend,
        scorer=second_backend,
        program=_add_program(),
        config=CONFIG,
    )
    assert first == second
    assert first.semantic_trace_hash == second.semantic_trace_hash
    assert len(first.semantic_trace_hash) == 64
    binding_events = [
        event
        for step in first.steps
        for event in step.interpreter_trace
        if event.kind == "binding"
    ]
    assert binding_events
    assert binding_events[0].payload["value"] is not None


def test_no_plan_consumes_horizon_without_scoring_nonexistent_graph() -> None:
    backend = _Backend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_validated({"op": "no_plan", "reason": "EXPLICIT"}),
        config=CONFIG,
    )
    assert len(result.steps) == CONFIG.horizon
    assert all(step.outcome == "no_plan" for step in result.steps)
    assert all(step.no_plan_reason == "EXPLICIT" for step in result.steps)
    assert len(backend.score_calls) == 1
    assert not backend.apply_calls


def test_illegal_final_overlay_consumes_step_without_candidate_score() -> None:
    backend = _Backend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_remove_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="illegal-fixture",
        ),
    )
    assert result.steps[0].outcome == "no_plan"
    assert result.steps[0].no_plan_reason == "ILLEGAL_FINAL_STATE"
    assert len(backend.score_calls) == 1
    assert not backend.apply_calls


def test_legal_degree_change_uses_one_authoritative_candidate_score() -> None:
    backend = _Backend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_add_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="legal-fixture",
        ),
    )
    assert len(backend.score_calls) == 2
    assert len(backend.apply_calls) == 1
    assert result.steps[0].accepted
    assert result.accepted_rewrites == 1
    assert sorted(_degrees(backend.score_calls[1])) == [3, 3, 3, 3, 4, 4]


def test_apparent_zero_initial_and_proposal_reach_current_inspection_boundary() -> None:
    initial_backend = _Backend(zero_after_edges=len(_cubic_graph().edges))
    initial_inspector = _Inspector()
    initial = evaluate_serial_program(
        backend=initial_backend,
        scorer=initial_backend,
        program=_validated({"op": "no_plan", "reason": "EXPLICIT"}),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=0,
            witness_cap=100,
            episode_id="zero-initial",
        ),
        counterexample_pipeline=initial_inspector,
    )
    assert initial.initial_counterexample is not None
    assert len(initial_inspector.calls) == 1

    proposal_backend = _Backend(zero_after_edges=len(_cubic_graph().edges) + 1)
    proposal_inspector = _Inspector()
    proposal = evaluate_serial_program(
        backend=proposal_backend,
        scorer=proposal_backend,
        program=_add_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="zero-proposal",
        ),
        counterexample_pipeline=proposal_inspector,
    )
    assert proposal.steps[0].counterexample is not None
    assert len(proposal_inspector.calls) == 1
    assert proposal_inspector.calls[0][2].source_id == proposal.program_hash


def test_equal_score_is_not_accepted_under_strict_improvement() -> None:
    class ConstantBackend(_Backend):
        def score(
            self,
            graph: GraphState,
            *,
            witness_cap: int,
            cutoff: GraphScore | None = None,
            record_profile: ScoreProfileRecorder | None = None,
        ) -> GraphScore | None:
            del witness_cap, cutoff, record_profile
            self.score_calls.append(graph)
            return GraphScore(True, ((4, 1),), 1, 1, True, (1,))

        def score_evidence(
            self,
            graph: GraphState,
            *,
            witness_cap: int,
            forbidden_lengths: object = None,
            attempt_kind: AttemptKind = AttemptKind.INITIAL,
        ) -> ScoreEvidence:
            del forbidden_lengths
            self.score_calls.append(graph)
            self.raw_graph_score_calls += 1
            self.unique_graph_scores += 1
            return ScoreEvidence(
                graph_content_hash=self.state_hash(graph),
                order=graph.order,
                edge_count=len(_cubic_graph(graph.order).edges),
                witness_cap=witness_cap,
                components=(
                    CycleComponentEvidence(
                        forbidden_length=4,
                        observed_count=1,
                        lower_bound=1,
                        upper_bound=1,
                        status=EvidenceStatus.EXACT,
                        node_budget=50_000,
                        nodes_visited=1,
                        wall_time_ns=0,
                        attempt_kind=attempt_kind,
                        backend_identity=TEST_IDENTITY,
                    ),
                ),
            )

    backend = ConstantBackend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_add_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="strict",
        ),
    )
    assert not result.steps[0].accepted
    assert result.accepted_rewrites == 0
    assert result.terminal_identity == result.initial_identity


def test_overlapping_budget_bounds_retry_selected_length_and_reject() -> None:
    class BoundedBackend(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.attempts: list[tuple[int, AttemptKind, object]] = []

        def score_evidence(
            self,
            graph: GraphState,
            *,
            witness_cap: int,
            forbidden_lengths: object = None,
            attempt_kind: AttemptKind = AttemptKind.INITIAL,
        ) -> ScoreEvidence:
            self.score_calls.append(graph)
            self.raw_graph_score_calls += 1
            self.unique_graph_scores += 1
            self.attempts.append(
                (len(graph.edges), attempt_kind, forbidden_lengths)
            )
            candidate = len(graph.edges) > len(_cubic_graph(graph.order).edges)
            lower = 4 if candidate else 5
            upper = (
                8
                if attempt_kind is AttemptKind.EXPANDED
                else (9 if candidate else 10)
            )
            return ScoreEvidence(
                self.state_hash(graph),
                graph.order,
                len(graph.edges),
                witness_cap,
                (
                    CycleComponentEvidence(
                        4,
                        lower,
                        lower,
                        upper,
                        (
                            EvidenceStatus.SEARCH_TIMEOUT_WITH_SAFE_PARTIAL
                            if attempt_kind is AttemptKind.EXPANDED
                            else EvidenceStatus.SEARCH_BUDGET_EXHAUSTED
                        ),
                        (
                            200_000
                            if attempt_kind is AttemptKind.EXPANDED
                            else 50_000
                        ),
                        1,
                        0,
                        attempt_kind,
                        TEST_IDENTITY,
                    ),
                ),
            )

    backend = BoundedBackend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_add_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="bounded-overlap",
        ),
    )

    assert not result.steps[0].accepted
    assert not result.steps[0].acceptance_proved
    assert result.score_attempts == 4
    expanded = [
        attempt for attempt in backend.attempts
        if attempt[1] is AttemptKind.EXPANDED
    ]
    assert len(expanded) == 2
    assert all(attempt[2] == (4,) for attempt in expanded)


def test_safe_timeout_partial_can_win_only_by_proved_interval_dominance() -> None:
    class SafePartialBackend(_Backend):
        def score_evidence(
            self,
            graph: GraphState,
            *,
            witness_cap: int,
            forbidden_lengths: object = None,
            attempt_kind: AttemptKind = AttemptKind.INITIAL,
        ) -> ScoreEvidence:
            del forbidden_lengths
            self.score_calls.append(graph)
            self.raw_graph_score_calls += 1
            self.unique_graph_scores += 1
            candidate = len(graph.edges) > len(_cubic_graph(graph.order).edges)
            lower, upper = ((0, 1) if candidate else (10, 10))
            return ScoreEvidence(
                self.state_hash(graph),
                graph.order,
                len(graph.edges),
                witness_cap,
                (
                    CycleComponentEvidence(
                        4,
                        lower,
                        lower,
                        upper,
                        (
                            EvidenceStatus.SEARCH_TIMEOUT_WITH_SAFE_PARTIAL
                            if candidate
                            else EvidenceStatus.EXACT
                        ),
                        50_000,
                        1,
                        0,
                        attempt_kind,
                        TEST_IDENTITY,
                    ),
                ),
            )

    backend = SafePartialBackend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_add_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="safe-partial",
        ),
    )

    assert result.steps[0].acceptance_proved
    assert result.steps[0].accepted
    assert result.accepted_rewrites == 1


def test_unsafe_initial_timeout_is_inconclusive_not_fitness() -> None:
    class UnsafeTimeoutBackend(_Backend):
        def score_evidence(
            self,
            graph: GraphState,
            *,
            witness_cap: int,
            forbidden_lengths: object = None,
            attempt_kind: AttemptKind = AttemptKind.INITIAL,
        ) -> ScoreEvidence:
            del graph, witness_cap, forbidden_lengths, attempt_kind
            self.raw_graph_score_calls += 1
            self.unique_graph_scores += 1
            raise ScoreTimeoutWithoutPartial("unsafe timeout")

    backend = UnsafeTimeoutBackend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_add_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="unsafe-timeout",
        ),
    )

    assert result.status is SerialEvaluationStatus.INCONCLUSIVE_UNSAFE_TIMEOUT
    assert result.initial_evidence is None
    assert result.fitness_interval.lower == 0
    assert result.fitness_interval.upper == 1
    assert not result.steps


def test_program_failure_is_worst_fitness_not_infrastructure() -> None:
    backend = _Backend()
    result = evaluate_serial_program(
        backend=backend,
        scorer=backend,
        program=_add_program(),
        config=SerialEpisodeConfig(
            order=6,
            graph_seed=11,
            policy_seed=17,
            horizon=1,
            witness_cap=100,
            episode_id="program-failure",
        ),
        interpreter_limits=InterpreterLimits(maximum_steps=0),
    )

    assert result.status is SerialEvaluationStatus.PROGRAM_FAILURE
    assert result.failure is not None
    assert result.fitness_interval.lower == 0
    assert result.fitness_interval.upper == 0
