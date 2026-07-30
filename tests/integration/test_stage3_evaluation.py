from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import GraphScore, GraphState, RewritePlan
from mutation_forge.proposals.k_switch import ProposalCandidate, ProposalPool
from mutation_forge.stage2b.rankers import RankedProposal, RankResult
from mutation_forge.stage3 import commands, evaluation
from mutation_forge.stage3.config import load_stage3_config
from mutation_forge.stage3.manifest import load_manifest
from mutation_forge.stage3.replay import verify_replay


@dataclass
class _TinyBackend:
    """Small deterministic backend used to exercise trajectory accounting."""

    calls: int = 0

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        return GraphState(order, ((0, 1),))

    def validate(self, graph: GraphState) -> Any:
        return type("Validation", (), {"valid": True, "errors": ()})()

    def score(self, graph: GraphState, *, witness_cap: int, cutoff: Any = None) -> GraphScore:
        total = 1 if (0, 2) in graph.edges else 2
        return GraphScore(True, ((4, total),), total, total * 16, True, (0, total, total * 16))

    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState:
        self.calls += 1
        return GraphState(graph.order, tuple(sorted(set(graph.edges).union(rewrite.added_edges))))

    def state_hash(self, graph: GraphState) -> str:
        return f"tiny:{graph.edges}"

    def close(self) -> None:
        return None


class _TinyGenerator:
    def __init__(self, backend: _TinyBackend) -> None:
        self.backend = backend
        self.calls: list[GraphState] = []

    def generate(self, graph: GraphState, *, policy_seed: int, step: int) -> ProposalPool:
        self.calls.append(graph)
        candidates = (
            ProposalCandidate(
                RewritePlan((), ((0, 1),), "tiny"),
                cast(
                    Any,
                    {
                        "proposal_id": "first",
                        "k": 2,
                        "operator_family": "tiny",
                        "selector_tags": [],
                    },
                ),
            ),
            ProposalCandidate(
                RewritePlan((), ((0, 2),), "tiny"),
                cast(
                    Any,
                    {
                        "proposal_id": "second",
                        "k": 2,
                        "operator_family": "tiny",
                        "selector_tags": [],
                    },
                ),
            ),
        )
        return ProposalPool(
            "test", candidates, f"pool-{len(self.calls)}", 2, {}, 0, 2, {}, {}, {}, 0, 0
        )


@dataclass
class _TinyRanker:
    policy_id: str

    def rank(self, context: Any, pool: ProposalPool) -> RankResult:
        selected = "first" if self.policy_id == "random" else "second"
        candidate = next(item for item in pool.candidates if item.proposal_id == selected)
        ranked = RankedProposal(selected, 1, 2, "tiny", ())
        return RankResult(
            self.policy_id,
            pool.pool_hash,
            (ranked,),
            candidate.proposal_id,
            0,
            False,
            False,
            False,
            False,
            None,
        )

    def close(self) -> None:
        return None


def test_trajectories_diverge_then_generate_independent_pools(
    monkeypatch: Any, project_root: Path
) -> None:
    config = load_stage3_config(project_root / "configs" / "stage3-generation.toml")
    backend = _TinyBackend()
    generator = _TinyGenerator(backend)
    rankers = {"random": _TinyRanker("random"), "structural": _TinyRanker("structural")}
    monkeypatch.setattr(
        evaluation,
        "_rankers",
        lambda config, policies: (rankers, ["random", "structural"], []),
    )
    monkeypatch.setattr(evaluation, "_pool_generator", lambda backend, config: generator)
    record = evaluation.run_development_episode(
        config,
        {"episode_id": "tiny", "order": 4, "graph_seed": 1, "policy_seed": 2, "horizon": 3},
        rankers,
        backend=backend,
    )
    assert record["terminal_status"] == "completed"
    # Divergence is observed after the first shared step; independent pools
    # begin at the following step.
    assert record["divergence_step"] == 1
    assert record["shared_pool_steps"] == 1
    assert record["independent_pool_steps"] == 4
    assert record["initial_score_calls"] == 1
    assert record["selected_score_calls"] == 6
    assert record["oracle_score_calls"] == 0
    assert (
        record["model_calls"] == record["app_server_calls"] == record["runtime_network_calls"] == 0
    )
    # First pool is shared; after divergence each policy gets a fresh graph-specific pool.
    assert len(generator.calls) == 5
    assert generator.calls[1] != generator.calls[2]
    policies = cast(dict[str, dict[str, Any]], record["policies"])
    assert all("trace" not in policy for policy in policies.values())
    assert all("initial_score" not in policy for policy in policies.values())
    assert all("best_score" not in policy for policy in policies.values())
    assert "initial_score" in record
    steps = cast(list[dict[str, Any]], record["steps"])
    traces = [
        trace
        for step in steps
        for trace in cast(dict[str, dict[str, Any]], step["policies"]).values()
    ]
    assert len(traces) == 6
    for trace in traces:
        assert not {"selected_score", "current_score", "previous_score"} & set(trace)
        assert {
            "selected_ordering_key",
            "previous_ordering_key",
            "selected_total_witnesses",
            "previous_total_witnesses",
            "current_total_witnesses",
            "selected_witness_delta",
            "selected_penalty_delta",
        } <= set(trace)


def test_episode_failure_is_recorded_without_invalid_scoring(
    monkeypatch: Any, project_root: Path
) -> None:
    config = load_stage3_config(project_root / "configs" / "stage3-generation.toml")

    class _InvalidBackend(_TinyBackend):
        def validate(self, graph: GraphState) -> Any:
            return type("Validation", (), {"valid": False, "errors": ("invalid",)})()

    backend = _InvalidBackend()
    rankers = {"random": _TinyRanker("random"), "structural": _TinyRanker("structural")}
    monkeypatch.setattr(
        evaluation,
        "_rankers",
        lambda config, policies: (rankers, list(rankers), []),
    )
    record = evaluation.run_development_episode(
        config,
        {"episode_id": "failed", "order": 4, "graph_seed": 1, "policy_seed": 2, "horizon": 1},
        rankers,
        backend=backend,
    )
    assert record["terminal_status"] == "failure"
    assert record["selected_score_calls"] == 0
    assert record["oracle_score_calls"] == 0
    assert record["invalid_graphs"] == 1
    assert (
        record["model_calls"] == record["app_server_calls"] == record["runtime_network_calls"] == 0
    )


def test_evaluation_records_are_written_once_in_eight_bounded_shards(
    tmp_path: Path, project_root: Path
) -> None:
    config = load_stage3_config(project_root / "configs" / "stage3-generation.toml")
    manifest = load_manifest(config.manifest_path, config)
    records = [
        {
            "episode_id": episode["episode_id"],
            "order": episode["order"],
            "graph_seed": episode["graph_seed"],
            "policy_seed": episode["policy_seed"],
            "value": index,
        }
        for index, episode in enumerate(cast(list[dict[str, Any]], manifest["episodes"]))
    ]

    primary = commands._write_evaluation_record_shards(
        config, tmp_path, "primary", records
    )
    replay = commands._write_evaluation_record_shards(
        config, tmp_path, "replay", records
    )

    assert primary["shard_count"] == replay["shard_count"] == 8
    assert primary["record_count"] == replay["record_count"] == 128
    assert primary["episodes_per_shard"] == replay["episodes_per_shard"] == 16
    assert all(item["record_count"] == 16 for item in primary["shards"])
    assert all(
        item["uncompressed_bytes"] <= config.limits.artifact_bytes
        for item in primary["shards"]
    )
    assert len(list(tmp_path.glob("evaluation-primary-shard-*.jsonl.gz"))) == 8
    assert len(list(tmp_path.glob("evaluation-replay-shard-*.jsonl.gz"))) == 8
    assert not (tmp_path / "evaluation-primary.jsonl.gz").exists()
    assert not (tmp_path / "evaluation-replay.jsonl.gz").exists()
    verified = verify_replay(
        tmp_path / "evaluation-primary-shards.json",
        tmp_path / "evaluation-replay-shards.json",
    )
    assert verified["exact"] is True
