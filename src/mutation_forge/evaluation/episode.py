from __future__ import annotations

import time
from collections.abc import Callable

from mutation_forge.backends.base import GraphBackend
from mutation_forge.counterexamples import (
    CandidateProvenance,
    CounterexampleDecision,
    CounterexampleHalt,
    CounterexampleOutcome,
    CounterexamplePipeline,
    CounterexampleVerified,
)
from mutation_forge.evaluation.profiling import (
    DeepOperatorTimingAccumulator,
    DeepScoreTimingAccumulator,
    TimingAccumulator,
)
from mutation_forge.models import EpisodeResult, GraphScore, GraphState, JsonValue
from mutation_forge.policies.baselines import BaselinePolicy
from mutation_forge.proposals.two_switch import TwoSwitchProposalSource

ProgressCallback = Callable[[dict[str, JsonValue]], None]


def _normalized_auc(curve: list[int], initial: int) -> float:
    if not curve:
        return 1.0
    denominator = max(1, initial) * len(curve)
    return sum(curve) / denominator


def run_episode(
    *,
    backend: GraphBackend,
    initial_graph: GraphState,
    entry_id: str,
    graph_seed: int,
    policy_seed: int,
    run_seed: int = 0,
    baseline: BaselinePolicy,
    evaluations: int,
    witness_cap: int,
    deadline: float,
    progress: ProgressCallback | None = None,
    profiling_enabled: bool = True,
    deep_profiling_enabled: bool = False,
    score_cache_enabled: bool = True,
    counterexample_pipeline: CounterexamplePipeline | None = None,
) -> EpisodeResult:
    started = time.monotonic()
    timing = TimingAccumulator() if profiling_enabled else None
    deep_timing = DeepOperatorTimingAccumulator() if deep_profiling_enabled else None
    deep_score_timing = DeepScoreTimingAccumulator() if deep_profiling_enabled else None
    record_score_profile = deep_score_timing.record if deep_score_timing is not None else None
    timing_started_ns = time.perf_counter_ns() if timing is not None else 0
    phase_started_ns = time.perf_counter_ns() if timing is not None else 0
    current = initial_graph
    if deep_score_timing is not None:
        deep_score_timing.record("score_request", {"calls": 1})
    initial_score = backend.score(
        current,
        witness_cap=witness_cap,
        record_profile=record_score_profile,
    )
    if initial_score is None:
        raise RuntimeError("initial score cannot be cutoff-dominated")
    if deep_score_timing is not None:
        deep_score_timing.record("score_result", {"full_results": 1})
    if timing is not None:
        timing.scoring_ns += time.perf_counter_ns() - phase_started_ns
    current_score = initial_score
    best_graph = current
    best_score = initial_score
    curve: list[int] = []
    first_improvement: int | None = None
    counterexample_records: list[dict[str, JsonValue]] = []
    legal = 0
    invalid = 0
    noop = 0
    duplicate = 0
    score_failures = 0
    policy_call_ms = 0.0
    phase_started_ns = time.perf_counter_ns() if timing is not None else 0
    seen = {current}
    if timing is not None:
        timing.duplicate_detection_ns += time.perf_counter_ns() - phase_started_ns
    exact_submissions: set[GraphState] = set()
    score_cache: dict[GraphState, GraphScore] = {}
    if score_cache_enabled:
        score_cache[current] = initial_score
        if deep_score_timing is not None:
            deep_score_timing.record("score_cache", {"inserts": 1})
    source = TwoSwitchProposalSource(backend, baseline.operator_family)
    record_proposal_timing = timing.record_proposal_phase if timing is not None else None
    completed = 0
    timed_out = False

    def inspect_counterexample(
        graph: GraphState,
        score: GraphScore,
        evaluation: int,
    ) -> None:
        if (
            counterexample_pipeline is None
            or score.total_capped_witnesses != 0
            or graph in exact_submissions
        ):
            return
        exact_submissions.add(graph)
        outcome: CounterexampleOutcome = counterexample_pipeline.inspect(
            graph=graph,
            score=score,
            witness_cap=witness_cap,
            provenance=CandidateProvenance(
                source_kind="baseline",
                source_id=baseline.policy_id,
                episode_id=entry_id,
                graph_seed=graph_seed,
                policy_seed=policy_seed,
                evaluation_step=evaluation,
            ),
        )
        record: dict[str, JsonValue] = {
            "decision": outcome.decision.value,
            "stop_reason": outcome.stop_reason,
            "candidate_id": (
                outcome.candidate.candidate_id if outcome.candidate is not None else None
            ),
            "primary_status": (outcome.primary.status if outcome.primary is not None else None),
            "independent_status": (
                outcome.independent.status if outcome.independent is not None else None
            ),
            "certificate_sha256": (
                outcome.certificate.sha256 if outcome.certificate is not None else None
            ),
        }
        counterexample_records.append(record)
        if outcome.decision is CounterexampleDecision.STOP_VERIFIED:
            raise CounterexampleVerified(outcome)
        if outcome.decision in {
            CounterexampleDecision.PAUSE_INCONCLUSIVE,
            CounterexampleDecision.FAIL,
        }:
            raise CounterexampleHalt(outcome)

    phase_started_ns = time.perf_counter_ns() if timing is not None else 0
    inspect_counterexample(current, initial_score, 0)
    if timing is not None:
        timing.exact_verification_ns += time.perf_counter_ns() - phase_started_ns

    for evaluation in range(1, evaluations + 1):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        policy_started_ns = time.perf_counter_ns()
        effective_policy_seed = (run_seed << 32) ^ policy_seed
        rewrite = source.propose(
            current,
            policy_seed=effective_policy_seed,
            evaluation=evaluation,
            record_timing=record_proposal_timing,
            record_deep_profile=(deep_timing.record if deep_timing is not None else None),
        )
        policy_elapsed_ns = time.perf_counter_ns() - policy_started_ns
        policy_call_ms += policy_elapsed_ns / 1_000_000
        if timing is not None:
            timing.proposal_generation_ns += policy_elapsed_ns
            timing.proposal_generation_calls += 1
        if time.monotonic() >= deadline:
            timed_out = True
            break
        if not rewrite.removed_edges and not rewrite.added_edges:
            noop += 1
            phase_started_ns = time.perf_counter_ns() if timing is not None else 0
            curve.append(best_score.total_capped_witnesses)
            completed = evaluation
            if timing is not None:
                timing.controller_ns += time.perf_counter_ns() - phase_started_ns
            continue
        phase_started_ns = time.perf_counter_ns() if timing is not None else 0
        try:
            candidate = backend.apply_rewrite(
                current,
                rewrite,
                record_score_profile=record_score_profile,
            )
        except ValueError:
            if timing is not None:
                timing.rewrite_application_ns += time.perf_counter_ns() - phase_started_ns
            invalid += 1
            phase_started_ns = time.perf_counter_ns() if timing is not None else 0
            curve.append(best_score.total_capped_witnesses)
            completed = evaluation
            if timing is not None:
                timing.controller_ns += time.perf_counter_ns() - phase_started_ns
            if time.monotonic() >= deadline:
                timed_out = True
                break
            continue
        if timing is not None:
            timing.rewrite_application_ns += time.perf_counter_ns() - phase_started_ns
        if time.monotonic() >= deadline:
            timed_out = True
            break
        legal += 1
        phase_started_ns = time.perf_counter_ns() if timing is not None else 0
        if candidate in seen:
            duplicate += 1
        seen.add(candidate)
        if timing is not None:
            timing.duplicate_detection_ns += time.perf_counter_ns() - phase_started_ns
        phase_started_ns = time.perf_counter_ns() if timing is not None else 0
        if deep_score_timing is not None:
            deep_score_timing.record("score_request", {"calls": 1})
        candidate_score: GraphScore | None
        try:
            cache_hit = score_cache_enabled and candidate in score_cache
            if score_cache_enabled and deep_score_timing is not None:
                deep_score_timing.record(
                    "score_cache",
                    {
                        "lookups": 1,
                        "hits": int(cache_hit),
                        "misses": int(not cache_hit),
                    },
                )
            if cache_hit:
                candidate_score = score_cache[candidate]
            else:
                candidate_score = backend.score(
                    candidate,
                    witness_cap=witness_cap,
                    cutoff=(current_score if current_score.total_capped_witnesses > 0 else None),
                    record_profile=record_score_profile,
                )
        except (RuntimeError, TimeoutError):
            if timing is not None:
                timing.scoring_ns += time.perf_counter_ns() - phase_started_ns
            score_failures += 1
            if deep_score_timing is not None:
                deep_score_timing.record("score_result", {"failures": 1})
            phase_started_ns = time.perf_counter_ns() if timing is not None else 0
            curve.append(best_score.total_capped_witnesses)
            completed = evaluation
            if timing is not None:
                timing.controller_ns += time.perf_counter_ns() - phase_started_ns
            continue
        if timing is not None:
            timing.scoring_ns += time.perf_counter_ns() - phase_started_ns
        if deep_score_timing is not None:
            deep_score_timing.record(
                "score_result",
                {
                    "full_results": int(candidate_score is not None),
                    "dominated_results": int(candidate_score is None),
                },
            )
        if score_cache_enabled and not cache_hit and candidate_score is not None:
            score_cache[candidate] = candidate_score
            if deep_score_timing is not None:
                deep_score_timing.record("score_cache", {"inserts": 1})
        if time.monotonic() >= deadline:
            timed_out = True
            break
        phase_started_ns = time.perf_counter_ns() if timing is not None else 0
        if candidate_score is not None:
            if candidate_score.ordering_key < current_score.ordering_key:
                current = candidate
                current_score = candidate_score
            if candidate_score.ordering_key < best_score.ordering_key:
                best_graph = candidate
                best_score = candidate_score
                if first_improvement is None:
                    first_improvement = evaluation
        if timing is not None:
            timing.controller_ns += time.perf_counter_ns() - phase_started_ns
        if candidate_score is not None:
            phase_started_ns = time.perf_counter_ns() if timing is not None else 0
            inspect_counterexample(candidate, candidate_score, evaluation)
            if timing is not None:
                timing.exact_verification_ns += time.perf_counter_ns() - phase_started_ns
            if time.monotonic() >= deadline:
                timed_out = True
                break
        phase_started_ns = time.perf_counter_ns() if timing is not None else 0
        curve.append(best_score.total_capped_witnesses)
        completed = evaluation
        if timing is not None:
            timing.controller_ns += time.perf_counter_ns() - phase_started_ns
        if progress is not None and (evaluation == 1 or evaluation % 50 == 0):
            elapsed = max(time.monotonic() - started, 1e-9)
            phase_started_ns = time.perf_counter_ns() if timing is not None else 0
            progress(
                {
                    "baseline": baseline.policy_id,
                    "graph_seed": graph_seed,
                    "policy_seed": policy_seed,
                    "evaluations": evaluation,
                    "evaluations_per_second": evaluation / elapsed,
                    "initial_total": initial_score.total_capped_witnesses,
                    "current_total": current_score.total_capped_witnesses,
                    "best_total": best_score.total_capped_witnesses,
                    "legal_proposals": legal,
                    "invalid_proposals": invalid,
                }
            )
            if timing is not None:
                timing.progress_reporting_ns += time.perf_counter_ns() - phase_started_ns
        if time.monotonic() >= deadline:
            timed_out = True
            break

    phase_started_ns = time.perf_counter_ns() if timing is not None else 0
    final_graph6 = backend.serialize_graph6(best_graph)
    final_graph_hash = backend.canonical_hash(best_graph)
    if timing is not None:
        timing.finalization_ns += time.perf_counter_ns() - phase_started_ns
    elapsed_seconds = time.monotonic() - started
    timing_profile = (
        timing.finish(time.perf_counter_ns() - timing_started_ns) if timing is not None else None
    )
    return EpisodeResult(
        baseline=baseline.policy_id,
        entry_id=entry_id,
        graph_seed=graph_seed,
        policy_seed=policy_seed,
        evaluations=completed,
        initial_score=initial_score,
        best_score=best_score,
        final_score=current_score,
        best_curve=tuple(curve),
        normalized_best_auc=_normalized_auc(curve, initial_score.total_capped_witnesses),
        first_improvement_evaluation=first_improvement,
        counterexample_records=tuple(counterexample_records),
        legal_proposals=legal,
        invalid_proposals=invalid,
        noop_proposals=noop,
        duplicate_proposals=duplicate,
        score_failures=score_failures,
        timed_out=timed_out,
        policy_call_ms=policy_call_ms,
        elapsed_seconds=elapsed_seconds,
        final_graph6=final_graph6,
        final_graph_hash=final_graph_hash,
        timing_profile=timing_profile,
        deep_operator_profile=(deep_timing.finish() if deep_timing is not None else None),
        deep_score_profile=(deep_score_timing.finish() if deep_score_timing is not None else None),
    )
