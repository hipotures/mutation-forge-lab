from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from mutation_forge.artifacts import canonical_json_hash, git_state
from mutation_forge.backends.base import GraphBackend
from mutation_forge.backends.heg import HegBackend
from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import GraphScore, GraphState, JsonValue
from mutation_forge.proposals.k_switch import (
    FEATURE_SCHEMA_VERSION,
    POOL_SCHEMA_VERSION,
    KSwitchPoolGenerator,
    ProposalCandidate,
    ProposalPool,
    make_scientific_context,
)
from mutation_forge.sandbox.contracts import (
    SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
    SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
)
from mutation_forge.sandbox.validation import validate_policy
from mutation_forge.stage2b.config import Stage2BConfig
from mutation_forge.stage2b.rankers import RankResult, SourceRanker

STAGE2B_ARTIFACT_VERSION = "stage2b.artifact.v1"
STAGE2B_BEHAVIOR_VERSION = "stage2b.behavior.v1"
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode()
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact {path.name} exceeds {MAX_ARTIFACT_BYTES} bytes")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded + b"\n")
    temporary.replace(path)


def _git_is_ancestor(repo: Path, ancestor: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, "HEAD"],
            check=False,
            capture_output=True,
            timeout=10,
        ).returncode
        == 0
    )


def _candidate(pool: ProposalPool, proposal_id: str | None) -> ProposalCandidate:
    for candidate in pool.candidates:
        if candidate.proposal_id == proposal_id:
            return candidate
    raise ValueError("ranker did not select a proposal from the shared pool")


def _score_selected(
    backend: GraphBackend,
    graph: GraphState,
    candidate: ProposalCandidate,
    *,
    witness_cap: int,
) -> tuple[GraphState, GraphScore, int]:
    candidate_graph = backend.apply_rewrite(graph, candidate.rewrite)
    validation = backend.validate(candidate_graph)
    if not validation.valid:
        raise ValueError(f"host-applied graph is invalid: {validation.errors}")
    started = time.perf_counter_ns()
    score = backend.score(candidate_graph, witness_cap=witness_cap)
    elapsed = time.perf_counter_ns() - started
    if score is None:
        raise RuntimeError("selected-plan scoring cannot be cutoff-dominated")
    return candidate_graph, score, elapsed


def _score_delta(initial: GraphScore, selected: GraphScore) -> int:
    return initial.total_capped_witnesses - selected.total_capped_witnesses


def _quality(initial: GraphScore, selected: GraphScore) -> float:
    return _score_delta(initial, selected) / max(
        1,
        initial.total_capped_witnesses,
    )


def _rank_record(
    rank: RankResult,
    candidate: ProposalCandidate,
    initial: GraphScore,
    selected: GraphScore,
) -> dict[str, JsonValue]:
    return {
        **rank.as_dict(),
        "selected_k": candidate.payload["k"],
        "selected_operator_family": candidate.payload["operator_family"],
        "selected_selector_tags": cast(
            list[JsonValue],
            candidate.payload["selector_tags"],
        ),
        "selected_score": selected.as_dict(),
        "selected_score_delta": _score_delta(initial, selected),
    }


def _paired_seed(
    backend: GraphBackend,
    graph: GraphState,
    *,
    initial_score: GraphScore,
    policy_seed: int,
    steps: int,
    witness_cap: int,
    generator: KSwitchPoolGenerator,
    random_ranker: SourceRanker,
    structural_ranker: SourceRanker,
    capture_rankings: bool,
) -> dict[str, JsonValue]:
    random_best = float("-inf")
    structural_best = float("-inf")
    random_curve: list[float] = []
    structural_curve: list[float] = []
    random_best_total = initial_score.total_capped_witnesses
    structural_best_total = initial_score.total_capped_witnesses
    pool_hashes: list[JsonValue] = []
    selection_trace: list[JsonValue] = []
    fixed_probe_rankings: list[JsonValue] = []
    selector_counts: Counter[str] = Counter()
    k_counts: Counter[str] = Counter()
    random_baseline_labels: Counter[str] = Counter()
    structural_selection_labels: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    feature_usage_totals: Counter[str] = Counter()
    feature_usage_maximums: Counter[str] = Counter()
    feature_budget_exhausted: dict[str, bool] = {
        "cycle_budget_exhausted": False,
        "distance_budget_exhausted": False,
        "local_risk_budget_exhausted": False,
    }
    pool_attempted = pool_deduplicated = pool_retained = 0
    invalid_graphs = 0
    score_calls = 1
    legality_ns = feature_ns = ranker_ns = scoring_ns = 0
    for step in range(steps):
        pool = generator.generate(graph, policy_seed=policy_seed, step=step)
        if not pool.candidates:
            raise RuntimeError("bounded generator produced an empty proposal pool")
        pool_hashes.append(pool.pool_hash)
        selector_counts.update(pool.selector_counts)
        k_counts.update(pool.k_counts)
        rejection_counts.update(pool.rejected)
        pool_attempted += pool.attempted
        pool_deduplicated += pool.deduplicated
        pool_retained += pool.retained
        for name, value in pool.feature_usage.items():
            if name in feature_budget_exhausted:
                feature_budget_exhausted[name] = feature_budget_exhausted[name] or cast(bool, value)
            else:
                feature_usage_totals[name] += cast(int, value)
                feature_usage_maximums[name] = max(
                    feature_usage_maximums[name],
                    cast(int, value),
                )
        legality_ns += pool.legality_elapsed_ns
        feature_ns += pool.feature_elapsed_ns
        context = make_scientific_context(
            graph,
            initial_score,
            forbidden_lengths=generator.feature_limits.forbidden_lengths,
            step=step,
            remaining_steps=steps - step - 1,
        )
        random_rank = random_ranker.rank(context, pool)
        structural_rank = structural_ranker.rank(context, pool)
        ranker_ns += random_rank.elapsed_ns + structural_rank.elapsed_ns
        if random_rank.pool_hash != structural_rank.pool_hash:
            raise RuntimeError("paired rankers did not receive the same pool")
        random_candidate = _candidate(pool, random_rank.selected_proposal_id)
        structural_candidate = _candidate(
            pool,
            structural_rank.selected_proposal_id,
        )
        random_label = (
            f"random_legal_k_switch:k={random_candidate.payload['k']}:"
            f"selector={random_candidate.payload['selector_tags'][0]}"
        )
        structural_label = (
            f"structural_legal_k_switch:k={structural_candidate.payload['k']}:"
            f"selector={structural_candidate.payload['selector_tags'][0]}"
        )
        random_baseline_labels[random_label] += 1
        structural_selection_labels[structural_label] += 1
        try:
            _, random_score, random_score_ns = _score_selected(
                backend,
                graph,
                random_candidate,
                witness_cap=witness_cap,
            )
            _, structural_score, structural_score_ns = _score_selected(
                backend,
                graph,
                structural_candidate,
                witness_cap=witness_cap,
            )
        except ValueError:
            invalid_graphs += 1
            raise
        score_calls += 2
        scoring_ns += random_score_ns + structural_score_ns
        random_quality = _quality(initial_score, random_score)
        structural_quality = _quality(initial_score, structural_score)
        random_best = max(random_best, random_quality)
        structural_best = max(structural_best, structural_quality)
        random_curve.append(random_best)
        structural_curve.append(structural_best)
        random_best_total = min(
            random_best_total,
            random_score.total_capped_witnesses,
        )
        structural_best_total = min(
            structural_best_total,
            structural_score.total_capped_witnesses,
        )
        random_record = _rank_record(
            random_rank,
            random_candidate,
            initial_score,
            random_score,
        )
        structural_record = _rank_record(
            structural_rank,
            structural_candidate,
            initial_score,
            structural_score,
        )
        selection_trace.append(
            {
                "step": step,
                "pool_hash": pool.pool_hash,
                "pool_size": pool.retained,
                "random_selected_id": random_candidate.proposal_id,
                "random_selected_k": random_candidate.payload["k"],
                "random_selected_operator_family": random_candidate.payload["operator_family"],
                "random_selected_selector_tags": cast(
                    list[JsonValue],
                    random_candidate.payload["selector_tags"],
                ),
                "random_score_delta": _score_delta(initial_score, random_score),
                "structural_selected_id": structural_candidate.proposal_id,
                "structural_selected_k": structural_candidate.payload["k"],
                "structural_selected_operator_family": (
                    structural_candidate.payload["operator_family"]
                ),
                "structural_selected_selector_tags": cast(
                    list[JsonValue],
                    structural_candidate.payload["selector_tags"],
                ),
                "structural_score_delta": _score_delta(
                    initial_score,
                    structural_score,
                ),
            }
        )
        if capture_rankings:
            fixed_probe_rankings.append(
                {
                    "step": step,
                    "pool_hash": pool.pool_hash,
                    "pool_k_counts": cast(
                        dict[str, JsonValue],
                        pool.k_counts,
                    ),
                    "pool_selector_counts": cast(
                        dict[str, JsonValue],
                        pool.selector_counts,
                    ),
                    "random": random_record,
                    "structural": structural_record,
                }
            )
    return {
        "policy_seed": policy_seed,
        "same_pool_proof": True,
        "pool_hashes": pool_hashes,
        "random": {
            "best_so_far_auc": statistics.fmean(random_curve),
            "best_total_witnesses": random_best_total,
            "curve": cast(list[JsonValue], random_curve),
        },
        "structural": {
            "best_so_far_auc": statistics.fmean(structural_curve),
            "best_total_witnesses": structural_best_total,
            "curve": cast(list[JsonValue], structural_curve),
        },
        "invalid_host_applied_graphs": invalid_graphs,
        "score_calls": score_calls,
        "pool_candidates_considered": pool_retained,
        "pool_generation": {
            "attempted": pool_attempted,
            "rejected": cast(
                dict[str, JsonValue],
                dict(sorted(rejection_counts.items())),
            ),
            "deduplicated": pool_deduplicated,
            "retained": pool_retained,
            "feature_usage_totals": cast(
                dict[str, JsonValue],
                dict(sorted(feature_usage_totals.items())),
            ),
            "feature_usage_maximums": cast(
                dict[str, JsonValue],
                dict(sorted(feature_usage_maximums.items())),
            ),
            "feature_budget_exhausted": cast(
                dict[str, JsonValue],
                feature_budget_exhausted,
            ),
        },
        "k_counts": dict(sorted(k_counts.items())),
        "selector_counts": dict(sorted(selector_counts.items())),
        "random_baselines_by_k_selector": dict(sorted(random_baseline_labels.items())),
        "structural_selections_by_k_selector": dict(sorted(structural_selection_labels.items())),
        "timing_ns": {
            "proposal_legality": legality_ns,
            "feature_work": feature_ns,
            "ranker": ranker_ns,
            "authoritative_selected_scoring": scoring_ns,
        },
        "selection_trace": selection_trace,
        "fixed_probe_rankings": fixed_probe_rankings,
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def _bootstrap_ci(
    differences: list[float],
    *,
    samples: int,
    confidence_level: float,
) -> tuple[float, float]:
    rng = random.Random(20260729)
    estimates = [
        statistics.median(
            differences[rng.randrange(len(differences))] for _ in range(len(differences))
        )
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return _percentile(estimates, alpha), _percentile(estimates, 1.0 - alpha)


def _strip_timing(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_strip_timing(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_timing(item)
            for key, item in value.items()
            if not key.endswith("_ns") and key != "timing_ns"
        }
    return value


def _canonical_paired_runs(runs: list[dict[str, JsonValue]]) -> list[JsonValue]:
    return [_strip_timing(run) for run in runs]


def run_toy_gate(
    config: Stage2BConfig,
    random_source: str,
    structural_source: str,
) -> dict[str, JsonValue]:
    backend = ToyBackend()
    graph = backend.generate_seed(
        order=config.toy_gate.order,
        seed=config.toy_gate.graph_seed,
    )
    initial_score = backend.score(
        graph,
        witness_cap=config.search.witness_cap,
    )
    if initial_score is None:
        raise RuntimeError("toy initial score unavailable")
    generator = KSwitchPoolGenerator(
        backend,
        pool_limits=config.pool,
        feature_limits=config.features,
    )
    runs: list[dict[str, JsonValue]] = []
    with (
        SourceRanker("random", random_source, config.sandbox) as random_ranker,
        SourceRanker(
            "structural",
            structural_source,
            config.sandbox,
        ) as structural_ranker,
    ):
        for index, policy_seed in enumerate(config.toy_gate.policy_seeds):
            runs.append(
                _paired_seed(
                    backend,
                    graph,
                    initial_score=initial_score,
                    policy_seed=policy_seed,
                    steps=config.search.steps,
                    witness_cap=config.search.witness_cap,
                    generator=generator,
                    random_ranker=random_ranker,
                    structural_ranker=structural_ranker,
                    capture_rankings=index < 2,
                )
            )
        worker_telemetry: dict[str, JsonValue] = {
            "random": random_ranker.telemetry(),
            "structural": structural_ranker.telemetry(),
        }
    random_auc = [
        cast(float, cast(dict[str, JsonValue], run["random"])["best_so_far_auc"]) for run in runs
    ]
    structural_auc = [
        cast(
            float,
            cast(dict[str, JsonValue], run["structural"])["best_so_far_auc"],
        )
        for run in runs
    ]
    random_best_total = [
        cast(
            int,
            cast(dict[str, JsonValue], run["random"])["best_total_witnesses"],
        )
        for run in runs
    ]
    structural_best_total = [
        cast(
            int,
            cast(dict[str, JsonValue], run["structural"])["best_total_witnesses"],
        )
        for run in runs
    ]
    differences = [
        structural - random_value
        for structural, random_value in zip(
            structural_auc,
            random_auc,
            strict=True,
        )
    ]
    ci_low, ci_high = _bootstrap_ci(
        differences,
        samples=config.toy_gate.bootstrap_samples,
        confidence_level=config.toy_gate.confidence_level,
    )
    median_random = statistics.median(random_auc)
    median_structural = statistics.median(structural_auc)
    relative = (median_structural - median_random) / max(
        abs(median_random),
        1.0e-12,
    )
    invalid = sum(cast(int, run["invalid_host_applied_graphs"]) for run in runs)
    structural_telemetry = cast(dict[str, JsonValue], worker_telemetry["structural"])
    failures = cast(int, structural_telemetry["failures"])
    criteria: dict[str, JsonValue] = {
        "paired_policy_seeds_at_least_32": len(runs) >= 32,
        "relative_auc_improvement_at_least_threshold": (
            relative >= config.toy_gate.auc_relative_improvement_threshold
        ),
        "paired_bootstrap_ci_excludes_zero": ci_low > 0.0,
        "structural_best_total_no_worse": (
            statistics.median(structural_best_total) <= statistics.median(random_best_total)
        ),
        "zero_invalid_host_applied_graphs": invalid == 0,
        "zero_structural_timeouts_or_crashes": failures == 0,
    }
    canonical_runs = _canonical_paired_runs(runs)
    behavior_base: dict[str, JsonValue] = {
        "schema_version": STAGE2B_BEHAVIOR_VERSION,
        "context_schema_version": SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
        "proposal_schema_version": SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
        "paired_runs": canonical_runs,
    }
    return {
        "status": (
            "completed" if all(cast(bool, value) for value in criteria.values()) else "failed"
        ),
        "preregistered": True,
        "dataset": {
            "backend": backend.backend_id,
            "order": config.toy_gate.order,
            "graph_seed": config.toy_gate.graph_seed,
            "policy_seeds": list(config.toy_gate.policy_seeds),
            "steps": config.search.steps,
        },
        "thresholds": {
            "minimum_policy_seeds": 32,
            "minimum_relative_auc_improvement": (
                config.toy_gate.auc_relative_improvement_threshold
            ),
            "confidence_level": config.toy_gate.confidence_level,
        },
        "metrics": {
            "median_random_best_so_far_auc": median_random,
            "median_structural_best_so_far_auc": median_structural,
            "relative_auc_improvement": relative,
            "paired_bootstrap_ci": [ci_low, ci_high],
            "median_random_best_total_witnesses": statistics.median(random_best_total),
            "median_structural_best_total_witnesses": statistics.median(structural_best_total),
            "invalid_host_applied_graphs": invalid,
        },
        "criteria": criteria,
        "paired_runs": cast(list[JsonValue], runs),
        "behavior_signature": {
            **behavior_base,
            "signature_sha256": canonical_json_hash(behavior_base),
        },
        "worker_telemetry": worker_telemetry,
    }


def _reference_policies(
    backend: HegBackend,
    graph: GraphState,
    *,
    witness_cap: int,
) -> list[dict[str, JsonValue]]:
    results: list[dict[str, JsonValue]] = []
    for index, operator in enumerate(("heg_uniform_two_switch", "heg_forbidden_cycle_break")):
        rewrite = backend.propose_rewrite(
            graph,
            operator_family=operator,
            policy_seed=1,
            evaluation=index,
        )
        candidate = backend.apply_rewrite(graph, rewrite)
        validation = backend.validate(candidate)
        score = backend.score(candidate, witness_cap=witness_cap)
        results.append(
            {
                "policy_id": operator,
                "valid": validation.valid,
                "score": score.as_dict() if score is not None else None,
            }
        )
    return results


def _pilot_once(
    config: Stage2BConfig,
    random_source: str,
    structural_source: str,
) -> dict[str, JsonValue]:
    backend = HegBackend(config.repositories.heg_repo)
    runs: list[dict[str, JsonValue]] = []
    references: list[dict[str, JsonValue]] = []
    try:
        generator = KSwitchPoolGenerator(
            backend,
            pool_limits=config.pool,
            feature_limits=config.features,
        )
        with (
            SourceRanker("random", random_source, config.sandbox) as random_ranker,
            SourceRanker(
                "structural",
                structural_source,
                config.sandbox,
            ) as structural_ranker,
        ):
            for graph_seed in config.heg_pilot.graph_seeds:
                graph = backend.generate_seed(
                    order=config.heg_pilot.order,
                    seed=graph_seed,
                )
                initial_score = backend.score(
                    graph,
                    witness_cap=config.search.witness_cap,
                )
                if initial_score is None:
                    raise RuntimeError("HEG initial score unavailable")
                references.extend(
                    _reference_policies(
                        backend,
                        graph,
                        witness_cap=config.search.witness_cap,
                    )
                )
                for policy_seed in config.heg_pilot.policy_seeds:
                    run = _paired_seed(
                        backend,
                        graph,
                        initial_score=initial_score,
                        policy_seed=policy_seed,
                        steps=config.heg_pilot.steps,
                        witness_cap=config.search.witness_cap,
                        generator=generator,
                        random_ranker=random_ranker,
                        structural_ranker=structural_ranker,
                        capture_rankings=True,
                    )
                    run["graph_seed"] = graph_seed
                    run["initial_canonical_hash"] = backend.canonical_hash(graph)
                    runs.append(run)
            telemetry: dict[str, JsonValue] = {
                "random": random_ranker.telemetry(),
                "structural": structural_ranker.telemetry(),
            }
        canonical = {
            "runs": _canonical_paired_runs(runs),
            "references": references,
        }
        return {
            "status": "completed",
            "backend": backend.backend_id,
            "heg_commit": backend.commit,
            "heg_dirty": backend.dirty,
            "runs": cast(list[JsonValue], runs),
            "reference_policies": cast(list[JsonValue], references),
            "worker_telemetry": telemetry,
            "canonical_hash": canonical_json_hash(canonical),
            "all_graphs_valid": all(
                cast(int, run["invalid_host_applied_graphs"]) == 0 for run in runs
            )
            and all(cast(bool, item["valid"]) for item in references),
            "hidden_best_of_k_scoring": False,
            "exact_verify_calls": 0,
            "score_calls_match_selected_only": all(
                cast(int, run["score_calls"]) == 1 + 2 * config.heg_pilot.steps for run in runs
            ),
            "rich_json_canonical_equal": True,
        }
    finally:
        backend.close()


def run_heg_pilot(
    config: Stage2BConfig,
    random_source: str,
    structural_source: str,
) -> dict[str, JsonValue]:
    if not config.heg_pilot.enabled:
        return {"status": "disabled"}
    first = _pilot_once(config, random_source, structural_source)
    second = _pilot_once(config, random_source, structural_source)
    replay_match = first["canonical_hash"] == second["canonical_hash"]
    first["replay_canonical_hash"] = second["canonical_hash"]
    first["deterministic_replay"] = replay_match
    if not replay_match:
        first["status"] = "failed"
    return first


def inspect_proposals(config: Stage2BConfig) -> dict[str, JsonValue]:
    backend = ToyBackend()
    graph = backend.generate_seed(
        order=config.toy_gate.order,
        seed=config.toy_gate.graph_seed,
    )
    generator = KSwitchPoolGenerator(
        backend,
        pool_limits=config.pool,
        feature_limits=config.features,
    )
    pool = generator.generate(
        graph,
        policy_seed=config.toy_gate.policy_seeds[0],
        step=0,
    )
    return {
        "status": "completed",
        "config_hash": config.stable_hash(),
        "graph_state_hash": backend.state_hash(graph),
        "pool": pool.as_dict(include_plans=True),
    }


def _schema_hashes(project_repo: Path) -> dict[str, JsonValue]:
    names = (
        "stage2a-probe.schema.json",
        "stage2b-context.schema.json",
        "stage2b-proposal.schema.json",
        "stage2b-config.schema.json",
    )
    root = project_repo / "configs" / "schemas"
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _new_run_path(config: Stage2BConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = config.run.run_root / f"stage2b-{timestamp}-{config.stable_hash()[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "artifacts" / "programs").mkdir(parents=True)
    return path


def run_stage2b_compare(
    random_policy_path: Path,
    structural_policy_path: Path,
    config: Stage2BConfig,
) -> dict[str, JsonValue]:
    random_source = random_policy_path.read_text()
    structural_source = structural_policy_path.read_text()
    random_validation = validate_policy(random_source, config.sandbox)
    structural_validation = validate_policy(structural_source, config.sandbox)
    if not random_validation.valid or not structural_validation.valid:
        raise ValueError("both Stage 2B policies must pass Stage 2A validation")
    run_path = _new_run_path(config)
    shutil.copy2(config.source_path, run_path / "stage2b_config.toml")
    shutil.copy2(
        random_policy_path,
        run_path / "artifacts" / "programs" / "random.py",
    )
    shutil.copy2(
        structural_policy_path,
        run_path / "artifacts" / "programs" / "structural.py",
    )
    terminal: dict[str, JsonValue] = {
        "schema_version": STAGE2B_ARTIFACT_VERSION,
        "status": "failed",
    }
    try:
        project_state = git_state(config.repositories.project_repo)
        heg_state = git_state(config.repositories.heg_repo)
        gates = {
            "project_base_verified": _git_is_ancestor(
                config.repositories.project_repo,
                config.repositories.frozen_project_commit,
            ),
            "heg_pin_verified": (
                heg_state["commit"] == config.repositories.frozen_heg_commit
                and heg_state["dirty"] is False
            ),
        }
        if not all(gates.values()):
            raise RuntimeError("frozen Stage 2B repository gate failed")
        toy = run_toy_gate(config, random_source, structural_source)
        pilot = run_heg_pilot(config, random_source, structural_source)
        behavior = cast(dict[str, JsonValue], toy["behavior_signature"])
        status = (
            "completed"
            if toy["status"] == "completed"
            and pilot["status"] in {"completed", "disabled"}
            and cast(bool, pilot.get("deterministic_replay", True))
            and cast(bool, pilot.get("all_graphs_valid", True))
            else "failed"
        )
        result: dict[str, JsonValue] = {
            "schema_version": STAGE2B_ARTIFACT_VERSION,
            "status": status,
            "run_path": str(run_path),
            "config_hash": config.stable_hash(),
            "schema_hashes": _schema_hashes(config.repositories.project_repo),
            "pool_schema_version": POOL_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "context_schema_version": SCIENTIFIC_CONTEXT_SCHEMA_VERSION,
            "proposal_schema_version": SCIENTIFIC_PROPOSAL_SCHEMA_VERSION,
            "budgets": {
                "pool": config.pool.as_dict(),
                "features": config.features.as_dict(),
                "sandbox": config.sandbox.as_dict(),
            },
            "policy_identities": {
                "random": random_validation.identity.as_dict(),
                "structural": structural_validation.identity.as_dict(),
            },
            "toy_gate": toy,
            "heg_pilot": pilot,
            "behavior_signature": behavior,
            "provenance": {
                "frozen_entry_point": {
                    "mutation_forge": config.repositories.frozen_project_commit,
                    "heg": config.repositories.frozen_heg_commit,
                },
                "observed": {
                    "mutation_forge": project_state,
                    "heg": heg_state,
                },
                **gates,
                "heg_read_only": True,
                "model_calls": 0,
                "network_calls": 0,
                "full_score_best_of_k_oracle_calls": 0,
            },
        }
        _write_json(run_path / "result.json", result)
        _write_json(run_path / "toy_benchmark.json", toy)
        _write_json(run_path / "heg_pilot.json", pilot)
        _write_json(run_path / "behavior_signature.json", behavior)
        _write_json(run_path / "budgets.json", result["budgets"])
        _write_json(run_path / "schema_hashes.json", result["schema_hashes"])
        _write_json(
            run_path / "policy_identities.json",
            result["policy_identities"],
        )
        _write_json(run_path / "provenance.json", result["provenance"])
        terminal["status"] = status
        _write_json(run_path / "terminal_status.json", terminal)
        return result
    except BaseException as error:
        terminal["error_type"] = type(error).__name__
        terminal["error"] = str(error)[:1024]
        _write_json(run_path / "terminal_status.json", terminal)
        raise


def evaluate_source_policy(
    policy_path: Path,
    config: Stage2BConfig,
) -> dict[str, JsonValue]:
    source = policy_path.read_text()
    validation = validate_policy(source, config.sandbox)
    run_path = _new_run_path(config)
    shutil.copy2(config.source_path, run_path / "stage2b_config.toml")
    shutil.copy2(
        policy_path,
        run_path / "artifacts" / "programs" / "candidate.py",
    )
    if not validation.valid:
        result: dict[str, JsonValue] = {
            "schema_version": STAGE2B_ARTIFACT_VERSION,
            "status": "invalid",
            "run_path": str(run_path),
            "validation": validation.as_dict(),
        }
        _write_json(run_path / "result.json", result)
        _write_json(
            run_path / "terminal_status.json",
            {"status": "invalid", "schema_version": STAGE2B_ARTIFACT_VERSION},
        )
        return result
    project_state = git_state(config.repositories.project_repo)
    heg_state = git_state(config.repositories.heg_repo)
    gates = {
        "project_base_verified": _git_is_ancestor(
            config.repositories.project_repo,
            config.repositories.frozen_project_commit,
        ),
        "heg_pin_verified": (
            heg_state["commit"] == config.repositories.frozen_heg_commit
            and heg_state["dirty"] is False
        ),
    }
    if not all(gates.values()):
        raise RuntimeError("frozen Stage 2B repository gate failed")
    backend = ToyBackend()
    graph = backend.generate_seed(
        order=config.toy_gate.order,
        seed=config.toy_gate.graph_seed,
    )
    score = backend.score(graph, witness_cap=config.search.witness_cap)
    if score is None:
        raise RuntimeError("toy score unavailable")
    pool = KSwitchPoolGenerator(
        backend,
        pool_limits=config.pool,
        feature_limits=config.features,
    ).generate(graph, policy_seed=config.toy_gate.policy_seeds[0], step=0)
    context = make_scientific_context(
        graph,
        score,
        forbidden_lengths=config.features.forbidden_lengths,
        step=0,
        remaining_steps=0,
    )
    with SourceRanker("candidate", source, config.sandbox) as ranker:
        ranking = ranker.rank(context, pool)
        telemetry = ranker.telemetry()
    ranking_ok = (
        ranking.selected_proposal_id is not None
        and not ranking.exception
        and not ranking.timeout
        and not ranking.crash
        and not ranking.protocol
    )
    signature_base: dict[str, JsonValue] = {
        "schema_version": STAGE2B_BEHAVIOR_VERSION,
        "pool_hash": pool.pool_hash,
        "ranking": ranking.as_dict(),
    }
    result = {
        "schema_version": STAGE2B_ARTIFACT_VERSION,
        "status": "completed" if ranking_ok else "failed",
        "run_path": str(run_path),
        "validation": validation.as_dict(),
        "identity": validation.identity.as_dict(),
        "pool_hash": pool.pool_hash,
        "ranking": ranking.as_dict(),
        "behavior_signature": {
            **signature_base,
            "signature_sha256": canonical_json_hash(signature_base),
        },
        "worker_telemetry": telemetry,
        "provenance": {
            "observed": {
                "mutation_forge": project_state,
                "heg": heg_state,
            },
            **gates,
            "heg_read_only": True,
            "model_calls": 0,
            "network_calls": 0,
        },
    }
    _write_json(run_path / "result.json", result)
    _write_json(run_path / "identity.json", result["identity"])
    _write_json(run_path / "pool.json", pool.as_dict(include_plans=True))
    _write_json(
        run_path / "behavior_signature.json",
        result["behavior_signature"],
    )
    _write_json(run_path / "worker_telemetry.json", telemetry)
    _write_json(run_path / "provenance.json", result["provenance"])
    _write_json(
        run_path / "terminal_status.json",
        {
            "status": "completed" if ranking_ok else "failed",
            "schema_version": STAGE2B_ARTIFACT_VERSION,
        },
    )
    return result
