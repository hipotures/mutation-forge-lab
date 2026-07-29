from __future__ import annotations

import time
from collections.abc import Callable

from mutation_forge.backends.base import GraphBackend
from mutation_forge.evaluation.profiling import TimingAccumulator
from mutation_forge.models import EpisodeResult, GraphState, JsonValue
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
) -> EpisodeResult:
    started = time.monotonic()
    timing = TimingAccumulator() if profiling_enabled else None
    timing_started_ns = time.perf_counter_ns() if timing is not None else 0
    phase_started_ns = time.perf_counter_ns() if timing is not None else 0
    current = initial_graph
    initial_score = backend.score(current, witness_cap=witness_cap)
    if timing is not None:
        timing.scoring_ns += time.perf_counter_ns() - phase_started_ns
    current_score = initial_score
    best_graph = current
    best_score = initial_score
    curve: list[int] = []
    first_improvement: int | None = None
    exact_zero_submissions = 0
    exact_verified_count = 0
    exact_verification_failures = 0
    legal = 0
    invalid = 0
    noop = 0
    duplicate = 0
    score_failures = 0
    policy_call_ms = 0.0
    phase_started_ns = time.perf_counter_ns() if timing is not None else 0
    seen = {backend.state_hash(current)}
    if timing is not None:
        timing.duplicate_detection_ns += time.perf_counter_ns() - phase_started_ns
    exact_submissions: set[str] = set()
    source = TwoSwitchProposalSource(backend, baseline.operator_family)
    record_proposal_timing = (
        timing.record_proposal_phase if timing is not None else None
    )
    completed = 0
    timed_out = False

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
            candidate = backend.apply_rewrite(current, rewrite)
        except ValueError:
            if timing is not None:
                timing.rewrite_application_ns += (
                    time.perf_counter_ns() - phase_started_ns
                )
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
        candidate_hash = backend.state_hash(candidate)
        if candidate_hash in seen:
            duplicate += 1
        seen.add(candidate_hash)
        if timing is not None:
            timing.duplicate_detection_ns += time.perf_counter_ns() - phase_started_ns
        phase_started_ns = time.perf_counter_ns() if timing is not None else 0
        try:
            candidate_score = backend.score(candidate, witness_cap=witness_cap)
        except (RuntimeError, TimeoutError):
            if timing is not None:
                timing.scoring_ns += time.perf_counter_ns() - phase_started_ns
            score_failures += 1
            phase_started_ns = time.perf_counter_ns() if timing is not None else 0
            curve.append(best_score.total_capped_witnesses)
            completed = evaluation
            if timing is not None:
                timing.controller_ns += time.perf_counter_ns() - phase_started_ns
            continue
        if timing is not None:
            timing.scoring_ns += time.perf_counter_ns() - phase_started_ns
        if time.monotonic() >= deadline:
            timed_out = True
            break
        phase_started_ns = time.perf_counter_ns() if timing is not None else 0
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
        if (
            candidate_score.total_capped_witnesses == 0
            and candidate_hash not in exact_submissions
        ):
            exact_submissions.add(candidate_hash)
            exact_zero_submissions += 1
            phase_started_ns = time.perf_counter_ns() if timing is not None else 0
            verification = backend.exact_verify(candidate)
            if timing is not None:
                timing.exact_verification_ns += (
                    time.perf_counter_ns() - phase_started_ns
                )
            if verification.status == "VERIFIED":
                exact_verified_count += 1
            elif verification.status != "REJECTED":
                exact_verification_failures += 1
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
                timing.progress_reporting_ns += (
                    time.perf_counter_ns() - phase_started_ns
                )
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
        timing.finish(time.perf_counter_ns() - timing_started_ns)
        if timing is not None
        else None
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
        exact_zero_submissions=exact_zero_submissions,
        exact_verified_count=exact_verified_count,
        exact_verification_failures=exact_verification_failures,
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
    )
