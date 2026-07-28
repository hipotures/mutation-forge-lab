from __future__ import annotations

import time
from collections.abc import Callable

from mutation_forge.backends.base import GraphBackend
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
) -> EpisodeResult:
    started = time.monotonic()
    current = initial_graph
    initial_score = backend.score(current, witness_cap=witness_cap)
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
    seen = {backend.canonical_hash(current)}
    exact_submissions: set[str] = set()
    source = TwoSwitchProposalSource(backend, baseline.operator_family)
    completed = 0
    timed_out = False

    for evaluation in range(1, evaluations + 1):
        if time.monotonic() >= deadline:
            timed_out = True
            break
        policy_started = time.perf_counter()
        effective_policy_seed = (run_seed << 32) ^ policy_seed
        rewrite = source.propose(
            current,
            policy_seed=effective_policy_seed,
            evaluation=evaluation,
        )
        policy_call_ms += (time.perf_counter() - policy_started) * 1000
        if time.monotonic() >= deadline:
            timed_out = True
            break
        if not rewrite.removed_edges and not rewrite.added_edges:
            noop += 1
            curve.append(best_score.total_capped_witnesses)
            completed = evaluation
            continue
        try:
            candidate = backend.apply_rewrite(current, rewrite)
        except ValueError:
            invalid += 1
            curve.append(best_score.total_capped_witnesses)
            completed = evaluation
            if time.monotonic() >= deadline:
                timed_out = True
                break
            continue
        if time.monotonic() >= deadline:
            timed_out = True
            break
        legal += 1
        candidate_hash = backend.canonical_hash(candidate)
        if candidate_hash in seen:
            duplicate += 1
        seen.add(candidate_hash)
        try:
            candidate_score = backend.score(candidate, witness_cap=witness_cap)
        except (RuntimeError, TimeoutError):
            score_failures += 1
            curve.append(best_score.total_capped_witnesses)
            completed = evaluation
            continue
        if time.monotonic() >= deadline:
            timed_out = True
            break
        if candidate_score.ordering_key < current_score.ordering_key:
            current = candidate
            current_score = candidate_score
        if candidate_score.ordering_key < best_score.ordering_key:
            best_graph = candidate
            best_score = candidate_score
            if first_improvement is None:
                first_improvement = evaluation
        if candidate_score.total_capped_witnesses == 0 and candidate_hash not in exact_submissions:
            exact_submissions.add(candidate_hash)
            exact_zero_submissions += 1
            verification = backend.exact_verify(candidate)
            if verification.status == "VERIFIED":
                exact_verified_count += 1
            elif verification.status != "REJECTED":
                exact_verification_failures += 1
            if time.monotonic() >= deadline:
                timed_out = True
                break
        curve.append(best_score.total_capped_witnesses)
        completed = evaluation
        if progress is not None and (evaluation == 1 or evaluation % 50 == 0):
            elapsed = max(time.monotonic() - started, 1e-9)
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
        if time.monotonic() >= deadline:
            timed_out = True
            break

    elapsed_seconds = time.monotonic() - started
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
        final_graph6=backend.serialize_graph6(best_graph),
        final_graph_hash=backend.canonical_hash(best_graph),
    )
