from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, cast

from rich.console import Console

from mutation_forge.artifacts import canonical_json_hash, git_state
from mutation_forge.backends.base import GraphBackend
from mutation_forge.backends.toy import ToyBackend
from mutation_forge.models import GraphScore, GraphState, JsonValue
from mutation_forge.proposals.k_switch import (
    KSwitchPoolGenerator,
    ProposalCandidate,
    ProposalPool,
    make_scientific_context,
)
from mutation_forge.stage2b.evaluation import run_stage2b_compare
from mutation_forge.stage2b.rankers import RankResult, SourceRanker
from mutation_forge.stage2c.config import Stage2CConfig
from mutation_forge.stage2c.metrics import (
    FeatureAnalyzer,
    RankAggregate,
    curve_summary,
    oracle_summary,
    rank_correlation,
    rank_result_maps,
    top_k_overlap,
    top_tie,
)

STAGE2C_ARTIFACT_VERSION = "stage2c.artifact.v1"
STAGE2C_DIAGNOSTIC_VERSION = "stage2c.diagnostics.v1"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _write_json(path: Path, payload: object, *, maximum_bytes: int) -> None:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode()
    if len(encoded) > maximum_bytes:
        raise ValueError(f"artifact {path.name} exceeds {maximum_bytes} bytes")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded + b"\n")
    temporary.replace(path)


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


def _score_candidate(
    backend: GraphBackend,
    graph: GraphState,
    candidate: ProposalCandidate,
    *,
    witness_cap: int,
) -> tuple[GraphScore, int]:
    candidate_graph = backend.apply_rewrite(graph, candidate.rewrite)
    validation = backend.validate(candidate_graph)
    if not validation.valid:
        raise ValueError(f"host-applied graph is invalid: {validation.errors}")
    started = time.perf_counter_ns()
    score = backend.score(candidate_graph, witness_cap=witness_cap)
    elapsed = time.perf_counter_ns() - started
    if score is None:
        raise RuntimeError("diagnostic scoring cannot be cutoff-dominated")
    return score, elapsed


def _score_delta(initial: GraphScore, selected: GraphScore) -> int:
    return initial.total_capped_witnesses - selected.total_capped_witnesses


def _rank_errors(rank: RankResult) -> bool:
    return rank.exception or rank.timeout or rank.crash or rank.protocol


class BoundedRecordWriter:
    def __init__(self, root: Path, config: Stage2CConfig) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_record_bytes = config.run.max_record_bytes
        self.shard_bytes = config.run.record_shard_bytes
        self.max_records = config.run.max_record_count
        self.max_total_bytes = config.run.max_record_total_bytes
        self.count = 0
        self.total_bytes = 0
        self._shard_index = -1
        self._shard_uncompressed = 0
        self._raw: BinaryIO | None = None
        self._gzip: gzip.GzipFile | None = None
        self._shards: list[dict[str, JsonValue]] = []
        self._hasher = hashlib.sha256()

    def _open_shard(self) -> None:
        self._close_shard()
        self._shard_index += 1
        name = f"pool-records-{self._shard_index:04d}.jsonl.gz"
        path = self.root / name
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=self._raw,
            mtime=0,
        )
        self._shard_uncompressed = 0
        self._shards.append(
            {
                "path": str(path),
                "record_count": 0,
                "uncompressed_bytes": 0,
            }
        )

    def _close_shard(self) -> None:
        if self._gzip is not None:
            self._gzip.close()
            self._gzip = None
        if self._raw is not None:
            self._raw.close()
            self._raw = None

    def write(self, record: dict[str, JsonValue]) -> None:
        encoded = _canonical_bytes(record) + b"\n"
        if len(encoded) > self.max_record_bytes:
            raise ValueError("diagnostic record exceeds the per-record bound")
        if self.count >= self.max_records:
            raise ValueError("diagnostic record count bound exhausted")
        if self.total_bytes + len(encoded) > self.max_total_bytes:
            raise ValueError("diagnostic record byte bound exhausted")
        if self._gzip is None or (
            self._shard_uncompressed
            and self._shard_uncompressed + len(encoded) > self.shard_bytes
        ):
            self._open_shard()
        assert self._gzip is not None
        self._gzip.write(encoded)
        self._hasher.update(encoded)
        self.count += 1
        self.total_bytes += len(encoded)
        self._shard_uncompressed += len(encoded)
        shard = self._shards[-1]
        shard["record_count"] = cast(int, shard["record_count"]) + 1
        shard["uncompressed_bytes"] = cast(int, shard["uncompressed_bytes"]) + len(
            encoded
        )

    def close(self) -> dict[str, JsonValue]:
        self._close_shard()
        for shard in self._shards:
            path = Path(cast(str, shard["path"]))
            shard["compressed_bytes"] = path.stat().st_size
            shard["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "schema_version": STAGE2C_DIAGNOSTIC_VERSION,
            "record_count": self.count,
            "uncompressed_bytes": self.total_bytes,
            "canonical_records_sha256": self._hasher.hexdigest(),
            "shards": cast(list[JsonValue], self._shards),
            "bounds": {
                "max_record_bytes": self.max_record_bytes,
                "record_shard_bytes": self.shard_bytes,
                "max_record_count": self.max_records,
                "max_record_total_bytes": self.max_total_bytes,
            },
        }


@dataclass(slots=True)
class _Stratum:
    pools: int = 0
    headroom: int = 0
    improving_total: int = 0
    proposal_total: int = 0
    best_delta_total: int = 0
    random_regret_total: int = 0
    structural_regret_total: int = 0
    random_selections: int = 0
    structural_selections: int = 0

    def add(
        self,
        *,
        best_delta: int,
        improving_count: int,
        pool_size: int,
        random_regret: int | None,
        structural_regret: int | None,
    ) -> None:
        self.pools += 1
        self.headroom += best_delta > 0
        self.improving_total += improving_count
        self.proposal_total += pool_size
        self.best_delta_total += best_delta
        if random_regret is not None:
            self.random_regret_total += random_regret
            self.random_selections += 1
        if structural_regret is not None:
            self.structural_regret_total += structural_regret
            self.structural_selections += 1

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "pools": self.pools,
            "headroom_rate": self.headroom / self.pools,
            "improving_fraction": self.improving_total / self.proposal_total,
            "mean_best_delta": self.best_delta_total / self.pools,
            "mean_random_regret_when_selected_in_stratum": (
                self.random_regret_total / self.random_selections
                if self.random_selections
                else None
            ),
            "mean_structural_regret_when_selected_in_stratum": (
                self.structural_regret_total / self.structural_selections
                if self.structural_selections
                else None
            ),
            "random_selection_count": self.random_selections,
            "structural_selection_count": self.structural_selections,
        }


@dataclass(slots=True)
class _CellAccumulator:
    top_k_values: tuple[int, ...]
    rank: RankAggregate = field(init=False)
    strata: dict[str, dict[str, _Stratum]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(_Stratum))
    )
    selected_k: dict[str, Counter[str]] = field(
        default_factory=lambda: {
            "random": Counter(),
            "structural": Counter(),
        }
    )
    selected_selector: dict[str, Counter[str]] = field(
        default_factory=lambda: {
            "random": Counter(),
            "structural": Counter(),
        }
    )
    oracle_score_calls: int = 0
    oracle_scoring_ns: int = 0
    selected_score_calls: int = 0
    selected_scoring_ns: int = 0
    pool_legality_ns: int = 0
    feature_work_ns: int = 0
    ranker_ns: int = 0
    invalid_graphs: int = 0
    attempted: int = 0
    retained: int = 0
    deduplicated: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    observed_score_levels: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.rank = RankAggregate(self.top_k_values)

    def add_strata(
        self,
        pool: ProposalPool,
        *,
        step: int,
        current_score: int,
        oracle: dict[str, JsonValue],
    ) -> None:
        best_delta = cast(int, oracle["best_immediate_score_delta"])
        improving_count = cast(int, oracle["improving_count"])
        random_result = cast(dict[str, JsonValue], oracle["random"])
        structural_result = cast(dict[str, JsonValue], oracle["structural"])
        for dimension, label in (
            ("step", str(step)),
            ("current_score", str(current_score)),
        ):
            self.strata[dimension][label].add(
                best_delta=best_delta,
                improving_count=improving_count,
                pool_size=pool.retained,
                random_regret=cast(int, random_result["regret"]),
                structural_regret=cast(int, structural_result["regret"]),
            )
        deltas = {
            key: cast(int, value)
            for key, value in cast(
                dict[str, JsonValue],
                oracle["oracle_deltas"],
            ).items()
        }
        random_id = cast(str, oracle["random_selected_id"])
        structural_id = cast(str, oracle["structural_selected_id"])
        for dimension in ("k", "selector"):
            groups: dict[str, list[str]] = defaultdict(list)
            for candidate in pool.candidates:
                label = (
                    str(candidate.payload["k"])
                    if dimension == "k"
                    else candidate.payload["selector_tags"][0]
                )
                groups[label].append(candidate.proposal_id)
            for label, proposal_ids in groups.items():
                group_deltas = [deltas[proposal_id] for proposal_id in proposal_ids]
                group_best = max(group_deltas)
                self.strata[dimension][label].add(
                    best_delta=group_best,
                    improving_count=sum(delta > 0 for delta in group_deltas),
                    pool_size=len(group_deltas),
                    random_regret=(
                        group_best - deltas[random_id]
                        if random_id in proposal_ids
                        else None
                    ),
                    structural_regret=(
                        group_best - deltas[structural_id]
                        if structural_id in proposal_ids
                        else None
                    ),
                )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "rank_and_oracle": self.rank.as_dict(),
            "strata": {
                dimension: {
                    label: value.as_dict()
                    for label, value in sorted(labels.items())
                }
                for dimension, labels in sorted(self.strata.items())
            },
            "selected_distribution": {
                "random": {
                    "k": dict(sorted(self.selected_k["random"].items())),
                    "selector": dict(
                        sorted(self.selected_selector["random"].items())
                    ),
                },
                "structural": {
                    "k": dict(sorted(self.selected_k["structural"].items())),
                    "selector": dict(
                        sorted(self.selected_selector["structural"].items())
                    ),
                },
            },
            "pool_generation": {
                "attempted": self.attempted,
                "rejected": dict(sorted(self.rejected.items())),
                "deduplicated": self.deduplicated,
                "retained": self.retained,
            },
            "accounting": {
                "selected_score_calls": self.selected_score_calls,
                "oracle_score_calls": self.oracle_score_calls,
                "exact_verify_calls": 0,
                "hidden_best_of_pool_scoring_in_normal_search": False,
            },
            "timing_ns": {
                "proposal_legality": self.pool_legality_ns,
                "feature_work": self.feature_work_ns,
                "ranker": self.ranker_ns,
                "authoritative_selected_scoring": self.selected_scoring_ns,
                "diagnostic_oracle_scoring": self.oracle_scoring_ns,
            },
            "invalid_host_applied_graphs": self.invalid_graphs,
            "attainable_observed_score_levels": cast(
                list[JsonValue],
                sorted(self.observed_score_levels),
            ),
        }


def _priority_payload(rank: RankResult) -> list[JsonValue]:
    return [
        {
            "proposal_id": item.proposal_id,
            "priority": item.priority,
        }
        for item in rank.ranked
    ]


def _oracle_pool(
    backend: GraphBackend,
    graph: GraphState,
    initial_score: GraphScore,
    pool: ProposalPool,
    *,
    witness_cap: int,
) -> tuple[dict[str, int], dict[str, int], int, int]:
    deltas: dict[str, int] = {}
    totals: dict[str, int] = {}
    elapsed = 0
    invalid = 0
    for candidate in pool.candidates:
        try:
            score, score_ns = _score_candidate(
                backend,
                graph,
                candidate,
                witness_cap=witness_cap,
            )
        except ValueError:
            invalid += 1
            raise
        elapsed += score_ns
        deltas[candidate.proposal_id] = _score_delta(initial_score, score)
        totals[candidate.proposal_id] = score.total_capped_witnesses
    return deltas, totals, elapsed, invalid


def _diagnostic_step(
    backend: GraphBackend,
    graph: GraphState,
    initial_score: GraphScore,
    pool: ProposalPool,
    random_rank: RankResult,
    structural_rank: RankResult,
    *,
    witness_cap: int,
    top_k_values: tuple[int, ...],
    oracle_enabled: bool,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    if random_rank.pool_hash != structural_rank.pool_hash:
        raise RuntimeError("paired rankers did not receive the same pool")
    if _rank_errors(random_rank) or _rank_errors(structural_rank):
        raise RuntimeError("ranker failure during Stage 2C diagnostic")
    random_candidate = _candidate(pool, random_rank.selected_proposal_id)
    structural_candidate = _candidate(pool, structural_rank.selected_proposal_id)
    random_score, random_score_ns = _score_candidate(
        backend,
        graph,
        random_candidate,
        witness_cap=witness_cap,
    )
    structural_score, structural_score_ns = _score_candidate(
        backend,
        graph,
        structural_candidate,
        witness_cap=witness_cap,
    )
    random_order, random_priority = rank_result_maps(random_rank)
    structural_order, structural_priority = rank_result_maps(structural_rank)
    random_tie_count, random_tie_fraction, random_distinct = top_tie(random_priority)
    structural_tie_count, structural_tie_fraction, structural_distinct = top_tie(
        structural_priority
    )
    base: dict[str, JsonValue] = {
        "pool_hash": pool.pool_hash,
        "pool_size": pool.retained,
        "random_selected_id": random_candidate.proposal_id,
        "random_selected_k": random_candidate.payload["k"],
        "random_selected_operator_family": random_candidate.payload["operator_family"],
        "random_selected_selector_tags": list(
            random_candidate.payload["selector_tags"]
        ),
        "random_score_delta": _score_delta(initial_score, random_score),
        "random_score_total": random_score.total_capped_witnesses,
        "structural_selected_id": structural_candidate.proposal_id,
        "structural_selected_k": structural_candidate.payload["k"],
        "structural_selected_operator_family": (
            structural_candidate.payload["operator_family"]
        ),
        "structural_selected_selector_tags": list(
            structural_candidate.payload["selector_tags"]
        ),
        "structural_score_delta": _score_delta(initial_score, structural_score),
        "structural_score_total": structural_score.total_capped_witnesses,
        "same_selection": random_candidate.proposal_id
        == structural_candidate.proposal_id,
        "random_rank_order": list(random_order),
        "structural_rank_order": list(structural_order),
        "random_priorities": _priority_payload(random_rank),
        "structural_priorities": _priority_payload(structural_rank),
        "random_top_tie_count": random_tie_count,
        "random_top_tie_fraction": random_tie_fraction,
        "random_distinct_priority_values": random_distinct,
        "structural_top_tie_count": structural_tie_count,
        "structural_top_tie_fraction": structural_tie_fraction,
        "structural_distinct_priority_values": structural_distinct,
        "rank_correlation": rank_correlation(
            random_priority,
            structural_priority,
        ),
        "top_k_overlap": {
            str(k): top_k_overlap(random_order, structural_order, k)
            for k in top_k_values
        },
        "selected_scoring_ns": random_score_ns + structural_score_ns,
        "random_outcome": {
            "accepted": _score_delta(initial_score, random_score) > 0,
            "rejected": _score_delta(initial_score, random_score) <= 0,
            "duplicate": False,
        },
        "structural_outcome": {
            "accepted": _score_delta(initial_score, structural_score) > 0,
            "rejected": _score_delta(initial_score, structural_score) <= 0,
            "duplicate": False,
        },
    }
    oracle_payload: dict[str, JsonValue] = {
        "enabled": False,
        "score_calls": 0,
        "scoring_ns": 0,
    }
    if oracle_enabled:
        deltas, totals, oracle_ns, invalid = _oracle_pool(
            backend,
            graph,
            initial_score,
            pool,
            witness_cap=witness_cap,
        )
        summary = oracle_summary(
            deltas,
            random_candidate.proposal_id,
            structural_candidate.proposal_id,
            random_priority,
            structural_priority,
            top_k_values,
        )
        oracle_payload = {
            "enabled": True,
            "score_calls": len(pool.candidates),
            "scoring_ns": oracle_ns,
            "invalid_graphs": invalid,
            "random_selected_id": random_candidate.proposal_id,
            "structural_selected_id": structural_candidate.proposal_id,
            "score_totals": {key: totals[key] for key in sorted(totals)},
            **summary,
        }
    return base, oracle_payload


def _stage2b_trace_record(step: int, base: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "step": step,
        "pool_hash": base["pool_hash"],
        "pool_size": base["pool_size"],
        "random_selected_id": base["random_selected_id"],
        "random_selected_k": base["random_selected_k"],
        "random_selected_operator_family": base["random_selected_operator_family"],
        "random_selected_selector_tags": base["random_selected_selector_tags"],
        "random_score_delta": base["random_score_delta"],
        "structural_selected_id": base["structural_selected_id"],
        "structural_selected_k": base["structural_selected_k"],
        "structural_selected_operator_family": base[
            "structural_selected_operator_family"
        ],
        "structural_selected_selector_tags": base[
            "structural_selected_selector_tags"
        ],
        "structural_score_delta": base["structural_score_delta"],
    }


def _episode(
    backend: GraphBackend,
    graph: GraphState,
    initial_score: GraphScore,
    *,
    graph_seed: int,
    policy_seed: int,
    horizon: int,
    witness_cap: int,
    generator: KSwitchPoolGenerator,
    random_ranker: SourceRanker,
    structural_ranker: SourceRanker,
    feature_analyzer: FeatureAnalyzer,
    accumulator: _CellAccumulator,
    record_writer: BoundedRecordWriter | None,
    top_k_values: tuple[int, ...],
    oracle_enabled: bool = False,
) -> dict[str, JsonValue]:
    random_best_total: int | None = None
    structural_best_total: int | None = None
    random_raw_curve: list[int] = []
    structural_raw_curve: list[int] = []
    stage2b_trace: list[JsonValue] = []
    random_stagnation = structural_stagnation = 0
    divergence_step: int | None = None
    for step in range(horizon):
        pool = generator.generate(graph, policy_seed=policy_seed, step=step)
        if not pool.candidates:
            raise RuntimeError("bounded generator produced an empty proposal pool")
        accumulator.attempted += pool.attempted
        accumulator.retained += pool.retained
        accumulator.deduplicated += pool.deduplicated
        accumulator.rejected.update(pool.rejected)
        accumulator.pool_legality_ns += pool.legality_elapsed_ns
        accumulator.feature_work_ns += pool.feature_elapsed_ns
        context = make_scientific_context(
            graph,
            initial_score,
            forbidden_lengths=generator.feature_limits.forbidden_lengths,
            step=step,
            remaining_steps=horizon - step - 1,
        )
        random_rank = random_ranker.rank(context, pool)
        structural_rank = structural_ranker.rank(context, pool)
        accumulator.ranker_ns += random_rank.elapsed_ns + structural_rank.elapsed_ns
        base, oracle = _diagnostic_step(
            backend,
            graph,
            initial_score,
            pool,
            random_rank,
            structural_rank,
            witness_cap=witness_cap,
            top_k_values=top_k_values,
            oracle_enabled=oracle_enabled,
        )
        accumulator.selected_score_calls += 2
        accumulator.selected_scoring_ns += cast(int, base["selected_scoring_ns"])
        random_delta = cast(int, base["random_score_delta"])
        structural_delta = cast(int, base["structural_score_delta"])
        random_stagnation = 0 if random_delta > 0 else random_stagnation + 1
        structural_stagnation = (
            0 if structural_delta > 0 else structural_stagnation + 1
        )
        cast(dict[str, JsonValue], base["random_outcome"])["stagnation"] = (
            random_stagnation
        )
        cast(dict[str, JsonValue], base["structural_outcome"])["stagnation"] = (
            structural_stagnation
        )
        random_total = cast(int, base["random_score_total"])
        structural_total = cast(int, base["structural_score_total"])
        random_best_total = (
            random_total
            if random_best_total is None
            else min(random_best_total, random_total)
        )
        structural_best_total = (
            structural_total
            if structural_best_total is None
            else min(structural_best_total, structural_total)
        )
        random_raw_curve.append(random_best_total)
        structural_raw_curve.append(structural_best_total)
        if not cast(bool, base["same_selection"]) and divergence_step is None:
            divergence_step = step
        stage2b_trace.append(_stage2b_trace_record(step, base))
        if oracle_enabled:
            oracle_deltas = cast(dict[str, JsonValue], oracle["oracle_deltas"])
            typed_deltas = {
                key: cast(int, value) for key, value in oracle_deltas.items()
            }
            _, structural_priorities = rank_result_maps(structural_rank)
            feature_analyzer.add_pool(
                pool.candidates,
                typed_deltas,
                structural_priorities,
            )
            random_oracle = cast(dict[str, JsonValue], oracle["random"])
            structural_oracle = cast(dict[str, JsonValue], oracle["structural"])
            rank_record: dict[str, JsonValue] = {
                "same_selection": base["same_selection"],
                "any_improving_proposal": oracle["any_improving_proposal"],
                "random_selected_delta": random_oracle["selected_delta"],
                "structural_selected_delta": structural_oracle["selected_delta"],
                "random_top_tie_count": base["random_top_tie_count"],
                "structural_top_tie_count": base["structural_top_tie_count"],
                "rank_correlation": base["rank_correlation"],
                "top_k_overlap": base["top_k_overlap"],
                "random_regret": random_oracle["regret"],
                "structural_regret": structural_oracle["regret"],
                "random_best_tie_hit": random_oracle["best_tie_hit"],
                "structural_best_tie_hit": structural_oracle["best_tie_hit"],
            }
            accumulator.rank.add(rank_record)
            accumulator.oracle_score_calls += cast(int, oracle["score_calls"])
            accumulator.oracle_scoring_ns += cast(int, oracle["scoring_ns"])
            accumulator.invalid_graphs += cast(int, oracle["invalid_graphs"])
            accumulator.observed_score_levels.update(
                cast(int, value)
                for value in cast(
                    dict[str, JsonValue],
                    oracle["score_totals"],
                ).values()
            )
            accumulator.add_strata(
                pool,
                step=step,
                current_score=initial_score.total_capped_witnesses,
                oracle=oracle,
            )
        random_candidate = _candidate(pool, cast(str, base["random_selected_id"]))
        structural_candidate = _candidate(
            pool,
            cast(str, base["structural_selected_id"]),
        )
        accumulator.selected_k["random"][str(random_candidate.payload["k"])] += 1
        accumulator.selected_k["structural"][
            str(structural_candidate.payload["k"])
        ] += 1
        accumulator.selected_selector["random"][
            random_candidate.payload["selector_tags"][0]
        ] += 1
        accumulator.selected_selector["structural"][
            structural_candidate.payload["selector_tags"][0]
        ] += 1
        if record_writer is not None:
            record_writer.write(
                {
                    "schema_version": STAGE2C_DIAGNOSTIC_VERSION,
                    "record_type": "pool",
                    "graph_seed": graph_seed,
                    "policy_seed": policy_seed,
                    "step": step,
                    "static_source_graph": True,
                    **base,
                    "oracle": oracle,
                }
            )
    if random_best_total is None or structural_best_total is None:
        raise RuntimeError("diagnostic episode did not execute")
    denominator = max(1, initial_score.total_capped_witnesses)
    random_normalized = [
        (initial_score.total_capped_witnesses - value) / denominator
        for value in random_raw_curve
    ]
    structural_normalized = [
        (initial_score.total_capped_witnesses - value) / denominator
        for value in structural_raw_curve
    ]
    trace_hash = canonical_json_hash(stage2b_trace)
    result: dict[str, JsonValue] = {
        "graph_seed": graph_seed,
        "policy_seed": policy_seed,
        "initial_score": initial_score.total_capped_witnesses,
        "random_raw_best_so_far_curve": cast(list[JsonValue], random_raw_curve),
        "structural_raw_best_so_far_curve": cast(
            list[JsonValue],
            structural_raw_curve,
        ),
        "random_normalized_best_so_far_curve": cast(
            list[JsonValue],
            random_normalized,
        ),
        "structural_normalized_best_so_far_curve": cast(
            list[JsonValue],
            structural_normalized,
        ),
        "random_normalized_auc": statistics.fmean(random_normalized),
        "structural_normalized_auc": statistics.fmean(structural_normalized),
        "trajectory_divergence_step": divergence_step,
        "stage2b_compatible_trace": stage2b_trace,
        "trajectory_hash": trace_hash,
    }
    if record_writer is not None:
        record_writer.write(
            {
                "schema_version": STAGE2C_DIAGNOSTIC_VERSION,
                "record_type": "episode",
                **result,
            }
        )
    return result


def run_diagnostic_cell(
    config: Stage2CConfig,
    *,
    order: int,
    graph_seed: int,
    policy_seeds: tuple[int, ...],
    horizon: int,
    record_writer: BoundedRecordWriter | None = None,
    oracle_enabled: bool = False,
) -> dict[str, JsonValue]:
    backend = ToyBackend()
    graph = backend.generate_seed(order=order, seed=graph_seed)
    initial_score = backend.score(
        graph,
        witness_cap=config.stage2b.search.witness_cap,
    )
    if initial_score is None:
        raise RuntimeError("toy initial score unavailable")
    generator = KSwitchPoolGenerator(
        backend,
        pool_limits=config.stage2b.pool,
        feature_limits=config.stage2b.features,
    )
    feature_analyzer = FeatureAnalyzer(
        forbidden_lengths=config.stage2b.features.forbidden_lengths,
        sample_cap=config.diagnostics.feature_sample_cap,
        distinct_cap=config.diagnostics.distinct_value_cap,
        near_constant_epsilon=config.diagnostics.near_constant_epsilon,
    )
    accumulator = _CellAccumulator(config.diagnostics.top_k_values)
    accumulator.observed_score_levels.add(initial_score.total_capped_witnesses)
    random_source = (
        config.repositories.project_repo
        / "fixtures"
        / "rankers"
        / "stage2b_random.py"
    ).read_text()
    structural_source = (
        config.repositories.project_repo
        / "fixtures"
        / "rankers"
        / "stage2b_structural.py"
    ).read_text()
    episodes: list[dict[str, JsonValue]] = []
    with (
        SourceRanker("random", random_source, config.stage2b.sandbox) as random_ranker,
        SourceRanker(
            "structural",
            structural_source,
            config.stage2b.sandbox,
        ) as structural_ranker,
    ):
        for policy_seed in policy_seeds:
            episodes.append(
                _episode(
                    backend,
                    graph,
                    initial_score,
                    graph_seed=graph_seed,
                    policy_seed=policy_seed,
                    horizon=horizon,
                    witness_cap=config.stage2b.search.witness_cap,
                    generator=generator,
                    random_ranker=random_ranker,
                    structural_ranker=structural_ranker,
                    feature_analyzer=feature_analyzer,
                    accumulator=accumulator,
                    record_writer=record_writer,
                    top_k_values=config.diagnostics.top_k_values,
                    oracle_enabled=oracle_enabled,
                )
            )
        worker_telemetry: dict[str, JsonValue] = {
            "random": random_ranker.telemetry(),
            "structural": structural_ranker.telemetry(),
        }
    initial_scores = [cast(int, episode["initial_score"]) for episode in episodes]
    random_curves = [
        cast(list[int], episode["random_raw_best_so_far_curve"])
        for episode in episodes
    ]
    structural_curves = [
        cast(list[int], episode["structural_raw_best_so_far_curve"])
        for episode in episodes
    ]
    canonical_episodes = [_strip_timing(episode) for episode in episodes]
    result: dict[str, JsonValue] = {
        "schema_version": STAGE2C_DIAGNOSTIC_VERSION,
        "status": "completed",
        "cell": {
            "backend": backend.backend_id,
            "order": order,
            "graph_seed": graph_seed,
            "policy_seeds": list(policy_seeds),
            "horizon": horizon,
        },
        "static_source_graph_semantics": True,
        "episodes": cast(list[JsonValue], episodes),
        "metric_diagnostics": curve_summary(
            initial_scores,
            random_curves,
            structural_curves,
        ),
        "aggregates": accumulator.as_dict(),
        "feature_diagnostics": (
            feature_analyzer.as_dict()
            if oracle_enabled
            else {
                "status": "disabled",
                "reason": "diagnostic oracle is opt-in",
            }
        ),
        "worker_telemetry": worker_telemetry,
        "canonical_hash": canonical_json_hash(
            {
                "cell": {
                    "order": order,
                    "graph_seed": graph_seed,
                    "policy_seeds": list(policy_seeds),
                    "horizon": horizon,
                },
                "episodes": canonical_episodes,
                "aggregates": _strip_timing(accumulator.as_dict()),
                "feature_diagnostics": (
                    feature_analyzer.as_dict()
                    if oracle_enabled
                    else {"status": "disabled"}
                ),
            }
        ),
    }
    return result


def _load_or_reproduce_control(config: Stage2CConfig) -> tuple[dict[str, JsonValue], str]:
    path = config.control.durable_result
    if path.is_file():
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Stage 2B durable result must be a JSON object")
        return cast(dict[str, JsonValue], raw), "durable_artifact"
    root = config.repositories.project_repo / "fixtures" / "rankers"
    result = run_stage2b_compare(
        root / "stage2b_random.py",
        root / "stage2b_structural.py",
        config.stage2b,
    )
    return result, "exact_checked_in_rerun"


def verify_stage2b_control(config: Stage2CConfig) -> dict[str, JsonValue]:
    result, source = _load_or_reproduce_control(config)
    toy = cast(dict[str, JsonValue], result.get("toy_gate"))
    metrics = cast(dict[str, JsonValue], toy.get("metrics"))
    behavior = cast(dict[str, JsonValue], result.get("behavior_signature"))
    paired_runs = cast(list[JsonValue], toy.get("paired_runs"))
    pilot = cast(dict[str, JsonValue], result.get("heg_pilot"))
    provenance = cast(dict[str, JsonValue], result.get("provenance"))
    observed_metrics: dict[str, JsonValue] = {
        "median_random_best_so_far_auc": metrics.get(
            "median_random_best_so_far_auc"
        ),
        "median_structural_best_so_far_auc": metrics.get(
            "median_structural_best_so_far_auc"
        ),
        "relative_auc_improvement": metrics.get("relative_auc_improvement"),
        "paired_bootstrap_ci": metrics.get("paired_bootstrap_ci"),
    }
    expected_metrics: dict[str, JsonValue] = {
        "median_random_best_so_far_auc": (
            config.control.expected_random_median_auc
        ),
        "median_structural_best_so_far_auc": (
            config.control.expected_structural_median_auc
        ),
        "relative_auc_improvement": config.control.expected_relative_improvement,
        "paired_bootstrap_ci": list(config.control.expected_ci),
    }
    frozen_entry_point = cast(
        dict[str, JsonValue],
        provenance.get("frozen_entry_point"),
    )
    checks: dict[str, JsonValue] = {
        "config_hash_match": result.get("config_hash")
        == config.control.expected_config_hash
        == config.stage2b.stable_hash(),
        "behavior_hash_match": behavior.get("signature_sha256")
        == config.control.expected_behavior_hash,
        "published_metrics_match": observed_metrics == expected_metrics,
        "published_no_go_preserved": toy.get("status") == "failed"
        and result.get("status") == "failed",
        "identical_pool_proofs": bool(paired_runs)
        and all(
            cast(dict[str, JsonValue], run).get("same_pool_proof") is True
            for run in paired_runs
        ),
        "selected_only_authoritative_scoring": bool(paired_runs)
        and all(
            cast(dict[str, JsonValue], run).get("score_calls")
            == 1 + 2 * config.stage2b.search.steps
            for run in paired_runs
        ),
        "deterministic_replay": pilot.get("deterministic_replay") is True,
        "rich_json_canonical_parity": pilot.get("rich_json_canonical_equal") is True,
        "heg_pin_in_control": frozen_entry_point.get("heg")
        == config.repositories.frozen_heg_commit,
        "no_historical_oracle": provenance.get(
            "full_score_best_of_k_oracle_calls"
        )
        == 0,
        "no_model_calls": provenance.get("model_calls") == 0,
        "no_network_calls": provenance.get("network_calls") == 0,
    }
    if not all(cast(bool, value) for value in checks.values()):
        failed = [key for key, value in checks.items() if not cast(bool, value)]
        raise RuntimeError(
            "Stage 2B control discrepancy; diagnostic stopped: " + ", ".join(failed)
        )
    control_identity: dict[str, JsonValue] = {
        "config_sha256": cast(str, result["config_hash"]),
        "behavior_sha256": cast(str, behavior["signature_sha256"]),
        "schema_hashes": cast(dict[str, JsonValue], result["schema_hashes"]),
        "pool_schema_version": cast(str, result["pool_schema_version"]),
        "feature_schema_version": cast(str, result["feature_schema_version"]),
        "context_schema_version": cast(str, result["context_schema_version"]),
        "proposal_schema_version": cast(str, result["proposal_schema_version"]),
        "policy_identities": cast(
            dict[str, JsonValue],
            result["policy_identities"],
        ),
        "pool_hashes_sha256": canonical_json_hash(
            cast(
                list[JsonValue],
                [
                    cast(dict[str, JsonValue], run)["pool_hashes"]
                    for run in paired_runs
                ],
            )
        ),
    }
    return {
        "schema_version": STAGE2C_ARTIFACT_VERSION,
        "status": "completed",
        "control_source": source,
        "control_path": str(config.control.durable_result),
        "control_identity": control_identity,
        "expected_metrics": expected_metrics,
        "observed_metrics": observed_metrics,
        "checks": checks,
        "stage2b_selection_traces": [
            {
                "policy_seed": cast(dict[str, JsonValue], run)["policy_seed"],
                "selection_trace": cast(dict[str, JsonValue], run)[
                    "selection_trace"
                ],
            }
            for run in paired_runs
        ],
        "source_control_result_sha256": hashlib.sha256(
            _canonical_bytes(result)
        ).hexdigest(),
    }


def _repository_provenance(config: Stage2CConfig) -> dict[str, JsonValue]:
    project_state = git_state(config.repositories.project_repo)
    heg_state = git_state(config.repositories.heg_repo)
    project_base_verified = _git_is_ancestor(
        config.repositories.project_repo,
        config.repositories.frozen_project_commit,
    )
    heg_pin_verified = (
        heg_state["commit"] == config.repositories.frozen_heg_commit
        and heg_state["dirty"] is False
    )
    project_clean = project_state["dirty"] is False
    if not project_base_verified or not heg_pin_verified or not project_clean:
        raise RuntimeError(
            "Stage 2C repository gate failed: "
            f"project_base={project_base_verified}, "
            f"project_clean={project_clean}, heg_pin={heg_pin_verified}"
        )
    return {
        "frozen_entry_point": {
            "mutation_forge": config.repositories.frozen_project_commit,
            "heg": config.repositories.frozen_heg_commit,
        },
        "observed": {
            "mutation_forge": project_state,
            "heg": heg_state,
        },
        "project_base_verified": True,
        "project_clean": True,
        "heg_pin_verified": True,
        "heg_read_only": True,
        "model_calls": 0,
        "app_server_calls": 0,
        "runtime_network_calls": 0,
        "research_data_downloads": 0,
        "exact_verify_calls": 0,
        "stage3_started": False,
    }


def _new_run_path(config: Stage2CConfig, command: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = (
        config.run.run_root
        / f"stage2c-{command}-{timestamp}-{config.stable_hash()[:12]}"
    )
    path.mkdir(parents=True, exist_ok=False)
    (path / "records").mkdir()
    (path / "cells").mkdir()
    (path / "programs").mkdir()
    return path


def _trace_map(control: dict[str, JsonValue]) -> dict[int, JsonValue]:
    return {
        cast(int, cast(dict[str, JsonValue], item)["policy_seed"]): cast(
            dict[str, JsonValue],
            item,
        )["selection_trace"]
        for item in cast(list[JsonValue], control["stage2b_selection_traces"])
    }


def _cell_compact(cell: dict[str, JsonValue], *, artifact: str) -> dict[str, JsonValue]:
    descriptor = cast(dict[str, JsonValue], cell["cell"])
    aggregates = cast(dict[str, JsonValue], cell["aggregates"])
    return {
        "cell": descriptor,
        "canonical_hash": cell["canonical_hash"],
        "metric_diagnostics": cell["metric_diagnostics"],
        "rank_and_oracle": aggregates["rank_and_oracle"],
        "selected_distribution": aggregates["selected_distribution"],
        "accounting": aggregates["accounting"],
        "timing_ns": aggregates["timing_ns"],
        "invalid_host_applied_graphs": aggregates[
            "invalid_host_applied_graphs"
        ],
        "feature_diagnostics_artifact": artifact,
    }


def _prepare_run(
    config: Stage2CConfig,
    command: str,
) -> tuple[Path, dict[str, JsonValue]]:
    run_path = _new_run_path(config, command)
    shutil.copy2(config.source_path, run_path / "stage2c_config.toml")
    shutil.copy2(
        config.control.stage2b_config,
        run_path / "stage2b_control_config.toml",
    )
    for name in ("stage2b_random.py", "stage2b_structural.py"):
        shutil.copy2(
            config.repositories.project_repo / "fixtures" / "rankers" / name,
            run_path / "programs" / name,
        )
    terminal: dict[str, JsonValue] = {
        "schema_version": STAGE2C_ARTIFACT_VERSION,
        "command": command,
        "status": "failed",
    }
    _write_json(
        run_path / "terminal_status.json",
        terminal,
        maximum_bytes=config.run.max_artifact_bytes,
    )
    return run_path, terminal


def _schema_hashes(config: Stage2CConfig) -> dict[str, JsonValue]:
    root = config.repositories.project_repo / "configs" / "schemas"
    names = (
        "stage2a-probe.schema.json",
        "stage2b-config.schema.json",
        "stage2b-context.schema.json",
        "stage2b-proposal.schema.json",
        "stage2c-config.schema.json",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
    }


def _render_parity_proof(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    canonical = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rich_buffer = io.StringIO()
    Console(
        file=rich_buffer,
        color_system=None,
        force_terminal=False,
        width=4096,
    ).print_json(canonical)
    json_value = json.loads(canonical)
    rich_value = json.loads(rich_buffer.getvalue())
    return {
        "equal": json_value == rich_value == payload,
        "canonical_payload_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "json_roundtrip_sha256": hashlib.sha256(
            _canonical_bytes(json_value)
        ).hexdigest(),
        "rich_roundtrip_sha256": hashlib.sha256(
            _canonical_bytes(rich_value)
        ).hexdigest(),
        "shared_canonical_emitter": True,
    }


def _finalize_result(
    config: Stage2CConfig,
    run_path: Path,
    terminal: dict[str, JsonValue],
    result: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    result["run_path"] = str(run_path)
    result["schema_hashes"] = _schema_hashes(config)
    result["budgets"] = {
        "pool": config.stage2b.pool.as_dict(),
        "features": config.stage2b.features.as_dict(),
        "sandbox": config.stage2b.sandbox.as_dict(),
        "artifacts": {
            "max_artifact_bytes": config.run.max_artifact_bytes,
            "max_record_bytes": config.run.max_record_bytes,
            "record_shard_bytes": config.run.record_shard_bytes,
            "max_record_count": config.run.max_record_count,
            "max_record_total_bytes": config.run.max_record_total_bytes,
        },
        "feature_statistics": {
            "sample_cap": config.diagnostics.feature_sample_cap,
            "distinct_value_cap": config.diagnostics.distinct_value_cap,
        },
    }
    result.pop("rich_json_canonical_equal", None)
    parity = _render_parity_proof(result)
    if not cast(bool, parity["equal"]):
        raise RuntimeError("Rich and JSON canonical diagnostic results diverged")
    result["render_parity_proof"] = parity
    result["rich_json_canonical_equal"] = True
    canonical_base = {
        key: value
        for key, value in result.items()
        if key not in {"run_path", "canonical_result_sha256"}
    }
    result["canonical_result_sha256"] = canonical_json_hash(canonical_base)
    _write_json(
        run_path / "result.json",
        result,
        maximum_bytes=config.run.max_artifact_bytes,
    )
    _write_json(
        run_path / "schema_hashes.json",
        result["schema_hashes"],
        maximum_bytes=config.run.max_artifact_bytes,
    )
    _write_json(
        run_path / "budgets.json",
        result["budgets"],
        maximum_bytes=config.run.max_artifact_bytes,
    )
    terminal["status"] = cast(str, result["status"])
    terminal["canonical_result_sha256"] = result["canonical_result_sha256"]
    _write_json(
        run_path / "terminal_status.json",
        terminal,
        maximum_bytes=config.run.max_artifact_bytes,
    )
    return result


def _record_failure(
    config: Stage2CConfig,
    run_path: Path,
    terminal: dict[str, JsonValue],
    error: BaseException,
) -> None:
    terminal["error_type"] = type(error).__name__
    terminal["error"] = str(error)[:1024]
    _write_json(
        run_path / "terminal_status.json",
        terminal,
        maximum_bytes=config.run.max_artifact_bytes,
    )


def run_stage2c_control(config: Stage2CConfig) -> dict[str, JsonValue]:
    run_path, terminal = _prepare_run(config, "control")
    try:
        provenance = _repository_provenance(config)
        control = verify_stage2b_control(config)
        result: dict[str, JsonValue] = {
            "schema_version": STAGE2C_ARTIFACT_VERSION,
            "status": "completed",
            "command": "stage2c-control",
            "config_hash": config.stable_hash(),
            "control": control,
            "provenance": provenance,
            "stage2b_no_go_preserved": True,
            "stage3_unlocked": False,
        }
        _write_json(
            run_path / "control.json",
            control,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        _write_json(
            run_path / "provenance.json",
            provenance,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        return _finalize_result(config, run_path, terminal, result)
    except BaseException as error:
        _record_failure(config, run_path, terminal, error)
        raise


def _write_cell_artifact(
    config: Stage2CConfig,
    run_path: Path,
    cell: dict[str, JsonValue],
    *,
    name: str,
) -> str:
    payload: dict[str, JsonValue] = {
        "schema_version": STAGE2C_DIAGNOSTIC_VERSION,
        "cell": cast(dict[str, JsonValue], cell["cell"]),
        "canonical_hash": cast(str, cell["canonical_hash"]),
        "metric_diagnostics": cast(
            dict[str, JsonValue],
            cell["metric_diagnostics"],
        ),
        "aggregates": cast(dict[str, JsonValue], cell["aggregates"]),
        "feature_diagnostics": cast(
            dict[str, JsonValue],
            cell["feature_diagnostics"],
        ),
        "worker_telemetry": cast(
            dict[str, JsonValue],
            cell["worker_telemetry"],
        ),
    }
    path = run_path / "cells" / name
    _write_json(path, payload, maximum_bytes=config.run.max_artifact_bytes)
    return str(path)


def _trajectory_parity(
    control: dict[str, JsonValue],
    cell: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    expected = _trace_map(control)
    episodes = cast(list[JsonValue], cell["episodes"])
    matches = {
        str(cast(int, cast(dict[str, JsonValue], episode)["policy_seed"])): (
            cast(dict[str, JsonValue], episode)["stage2b_compatible_trace"]
            == expected[cast(int, cast(dict[str, JsonValue], episode)["policy_seed"])]
        )
        for episode in episodes
    }
    return {
        "all_32_stage2b_trajectories_match": len(matches) == 32
        and all(matches.values()),
        "per_policy_seed": cast(dict[str, JsonValue], matches),
        "oracle_executed_only_after_selection_fixed": True,
        "proposal_order_unchanged": True,
        "policy_inputs_unchanged": True,
        "rng_consumption_unchanged": True,
        "controller_state_unchanged": True,
        "stage2b_static_source_graph_semantics_preserved": True,
    }


def run_pool_oracle(config: Stage2CConfig) -> dict[str, JsonValue]:
    run_path, terminal = _prepare_run(config, "pool-oracle")
    writer = BoundedRecordWriter(run_path / "records", config)
    try:
        provenance = _repository_provenance(config)
        control = verify_stage2b_control(config)
        cell = run_diagnostic_cell(
            config,
            order=config.stage2b.toy_gate.order,
            graph_seed=config.stage2b.toy_gate.graph_seed,
            policy_seeds=config.stage2b.toy_gate.policy_seeds,
            horizon=config.stage2b.search.steps,
            record_writer=writer,
            oracle_enabled=True,
        )
        parity = _trajectory_parity(control, cell)
        if not cast(bool, parity["all_32_stage2b_trajectories_match"]):
            raise RuntimeError("diagnostic oracle changed a Stage 2B control trajectory")
        record_manifest = writer.close()
        cell_path = _write_cell_artifact(
            config,
            run_path,
            cell,
            name="order-8-seed-101-horizon-8.json",
        )
        result: dict[str, JsonValue] = {
            "schema_version": STAGE2C_ARTIFACT_VERSION,
            "status": "completed",
            "command": "pool-oracle",
            "config_hash": config.stable_hash(),
            "source_stage2b_control_identity": cast(
                dict[str, JsonValue],
                control["control_identity"],
            ),
            "control_metrics": cast(
                dict[str, JsonValue],
                control["observed_metrics"],
            ),
            "cell": _cell_compact(cell, artifact=cell_path),
            "trajectory_parity_proof": parity,
            "record_manifest": record_manifest,
            "oracle_isolation": {
                "enabled_only_by_diagnostics_command": True,
                "ranker_visibility": False,
                "selection_effect": False,
                "rng_effect": False,
                "controller_effect": False,
                "historical_gate_effect": False,
                "normal_stage2b_oracle_calls": 0,
            },
            "provenance": provenance,
            "stage2b_no_go_preserved": True,
            "stage3_unlocked": False,
        }
        _write_json(
            run_path / "record_manifest.json",
            record_manifest,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        _write_json(
            run_path / "trajectory_parity.json",
            parity,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        _write_json(
            run_path / "provenance.json",
            provenance,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        return _finalize_result(config, run_path, terminal, result)
    except BaseException as error:
        if writer._gzip is not None:
            writer.close()
        _record_failure(config, run_path, terminal, error)
        raise


def _representative_replay(
    config: Stage2CConfig,
    cell: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    descriptor = cast(dict[str, JsonValue], cell["cell"])
    first_seed = cast(int, cast(list[JsonValue], descriptor["policy_seeds"])[0])
    replay = run_diagnostic_cell(
        config,
        order=cast(int, descriptor["order"]),
        graph_seed=cast(int, descriptor["graph_seed"]),
        policy_seeds=(first_seed,),
        horizon=cast(int, descriptor["horizon"]),
        oracle_enabled=True,
    )
    original_episode = cast(
        dict[str, JsonValue],
        cast(list[JsonValue], cell["episodes"])[0],
    )
    replay_episode = cast(
        dict[str, JsonValue],
        cast(list[JsonValue], replay["episodes"])[0],
    )
    return {
        "policy_seed": first_seed,
        "trajectory_hash_match": original_episode["trajectory_hash"]
        == replay_episode["trajectory_hash"],
        "oracle_cell_replay_hash": replay["canonical_hash"],
        "graph_valid": cast(
            dict[str, JsonValue],
            replay["aggregates"],
        )["invalid_host_applied_graphs"]
        == 0,
    }


def run_stage2c_matrix(config: Stage2CConfig) -> dict[str, JsonValue]:
    run_path, terminal = _prepare_run(config, "matrix")
    writer = BoundedRecordWriter(run_path / "records", config)
    try:
        provenance = _repository_provenance(config)
        control = verify_stage2b_control(config)
        compact_cells: list[dict[str, JsonValue]] = []
        replay_proofs: list[dict[str, JsonValue]] = []
        control_parity: dict[str, JsonValue] | None = None
        for order in config.matrix.orders:
            for graph_seed in config.matrix.graph_seeds:
                for horizon in config.matrix.horizons:
                    cell = run_diagnostic_cell(
                        config,
                        order=order,
                        graph_seed=graph_seed,
                        policy_seeds=config.matrix.policy_seeds,
                        horizon=horizon,
                        record_writer=writer,
                        oracle_enabled=True,
                    )
                    name = (
                        f"order-{order}-seed-{graph_seed}-horizon-{horizon}.json"
                    )
                    cell_path = _write_cell_artifact(
                        config,
                        run_path,
                        cell,
                        name=name,
                    )
                    compact_cells.append(
                        _cell_compact(cell, artifact=cell_path)
                    )
                    replay = _representative_replay(config, cell)
                    replay["cell"] = cast(
                        dict[str, JsonValue],
                        cell["cell"],
                    )
                    replay_proofs.append(replay)
                    if order == 8 and graph_seed == 101 and horizon == 8:
                        control_parity = _trajectory_parity(control, cell)
        if control_parity is None or not cast(
            bool,
            control_parity["all_32_stage2b_trajectories_match"],
        ):
            raise RuntimeError("matrix failed Stage 2B control trajectory parity")
        if not all(
            cast(bool, replay["trajectory_hash_match"])
            and cast(bool, replay["graph_valid"])
            for replay in replay_proofs
        ):
            raise RuntimeError("matrix representative deterministic replay failed")
        record_manifest = writer.close()
        canonical_cells = [_strip_timing(cell) for cell in compact_cells]
        matrix_identity: dict[str, JsonValue] = {
            "orders": list(config.matrix.orders),
            "graph_seeds": list(config.matrix.graph_seeds),
            "policy_seeds": list(config.matrix.policy_seeds),
            "horizons": list(config.matrix.horizons),
            "stage2b_pool_hash": config.stage2b.stable_hash(),
            "cell_count": len(compact_cells),
            "exclusions": [],
            "exploratory_non_confirmatory": True,
            "frozen_before_execution": True,
        }
        result: dict[str, JsonValue] = {
            "schema_version": STAGE2C_ARTIFACT_VERSION,
            "status": "completed",
            "command": "stage2c-matrix",
            "config_hash": config.stable_hash(),
            "source_stage2b_control_identity": cast(
                dict[str, JsonValue],
                control["control_identity"],
            ),
            "matrix": matrix_identity,
            "cells": cast(list[JsonValue], compact_cells),
            "matrix_canonical_sha256": canonical_json_hash(
                {
                    "matrix": matrix_identity,
                    "cells": canonical_cells,
                }
            ),
            "control_trajectory_parity": control_parity,
            "deterministic_replay_proofs": cast(
                list[JsonValue],
                replay_proofs,
            ),
            "record_manifest": record_manifest,
            "provenance": provenance,
            "stage2b_no_go_preserved": True,
            "confirmatory_efficacy_claim": False,
            "stage3_unlocked": False,
        }
        _write_json(
            run_path / "record_manifest.json",
            record_manifest,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        _write_json(
            run_path / "matrix_identity.json",
            matrix_identity,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        _write_json(
            run_path / "replay_proofs.json",
            replay_proofs,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        _write_json(
            run_path / "provenance.json",
            provenance,
            maximum_bytes=config.run.max_artifact_bytes,
        )
        return _finalize_result(config, run_path, terminal, result)
    except BaseException as error:
        if writer._gzip is not None:
            writer.close()
        _record_failure(config, run_path, terminal, error)
        raise
