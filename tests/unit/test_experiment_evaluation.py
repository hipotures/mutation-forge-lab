from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge.experiment import evaluation
from mutation_forge.experiment.config import orders_for_generation
from mutation_forge.experiment.json_io import write_json
from mutation_forge.models import GraphScore, GraphState, RewritePlan
from mutation_forge.proposals.k_switch import ProposalCandidate, ProposalPool
from mutation_forge.stage2b.rankers import RankResult


class _Backend:
    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        return (4,) if order >= 4 else ()

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        return GraphState(order, ((0, 1),))

    def validate(self, graph: GraphState) -> Any:
        return type("Validation", (), {"valid": True, "errors": ()})()

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: Any = None,
    ) -> GraphScore:
        return GraphScore(True, ((4, 2),), 2, 32, True, (0, 2, 32))


class _PoolGenerator:
    def __init__(
        self,
        backend: _Backend,
        *,
        pool_limits: Any,
        feature_limits: Any,
    ) -> None:
        self.backend = backend

    def generate(self, graph: GraphState, *, policy_seed: int, step: int) -> ProposalPool:
        candidate = ProposalCandidate(
            RewritePlan((), (), "test"),
            cast(
                Any,
                {
                    "proposal_id": "proposal-0",
                    "k": 2,
                    "operator_family": "test",
                    "selector_tags": [],
                },
            ),
        )
        return ProposalPool(
            "test",
            (candidate,),
            "pool-0",
            1,
            {},
            0,
            1,
            {},
            {},
            {},
            0,
            0,
        )


class _TimedOutRanker:
    def rank(self, context: Any, pool: ProposalPool) -> RankResult:
        return RankResult(
            "candidate",
            pool.pool_hash,
            (),
            None,
            0,
            False,
            True,
            False,
            False,
            {"code": "worker_timeout", "message": "worker exceeded total wall limit"},
        )


def test_rank_failure_is_recorded_without_aborting_evaluation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(evaluation, "KSwitchPoolGenerator", _PoolGenerator)

    result = evaluation._trajectory(
        cast(Any, _Backend()),
        _TimedOutRanker(),
        {
            "horizon": 1,
            "proposal_pool_size": 1,
            "witness_cap": 64,
            "baselines": (),
        },
        order=4,
        graph_seed=1,
        policy_seed=2,
        candidate_id="candidate",
    )

    candidate = cast(dict[str, Any], result["policies"])["candidate"]
    assert candidate["failure_count"] == 1
    assert candidate["raw_best_so_far_curve"] == [2]
    trace = candidate["trace"]
    assert trace[0]["error"] == "ranker did not select a pool proposal"
    assert trace[0]["rank"]["timeout"] is True
    assert trace[0]["rank"]["error"]["code"] == "worker_timeout"


def test_run_once_resumes_from_last_completed_episode(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    class Ranker:
        source = "def priority(ctx, proposal):\n    return 0\n"

        def rank(self, _context: Any, _pool: Any) -> None:
            return None

    settings = {
        "orders": (4,),
        "graph_seeds": (1,),
        "policy_seeds": (10, 11),
        "horizon": 1,
        "proposal_pool_size": 1,
        "baselines": (),
        "workers": 1,
        "thread_count": 1,
        "witness_cap": 64,
    }
    first_calls: list[int] = []

    def interrupted(*_args: Any, policy_seed: int, **_kwargs: Any) -> dict[str, Any]:
        first_calls.append(policy_seed)
        if policy_seed == 11:
            raise KeyboardInterrupt
        return {
            "order": 4,
            "graph_seed": 1,
            "policy_seed": policy_seed,
            "policies": {
                "candidate": {
                    "auc": 0.25,
                    "accepted_count": 0,
                    "rejected_count": 1,
                    "failure_count": 0,
                }
            },
        }

    monkeypatch.setattr(evaluation, "_trajectory", interrupted)
    with pytest.raises(KeyboardInterrupt):
        evaluation._run_once(
            {},
            "candidate",
            Ranker(),
            settings,
            backend=cast(Any, object()),
            limits=cast(Any, object()),
            checkpoint_root=tmp_path,
        )
    assert first_calls == [10, 11]

    resumed_calls: list[int] = []

    def resumed(*_args: Any, policy_seed: int, **_kwargs: Any) -> dict[str, Any]:
        resumed_calls.append(policy_seed)
        return {
            "order": 4,
            "graph_seed": 1,
            "policy_seed": policy_seed,
            "policies": {
                "candidate": {
                    "auc": 0.5,
                    "accepted_count": 1,
                    "rejected_count": 0,
                    "failure_count": 0,
                }
            },
        }

    progress: list[dict[str, Any]] = []
    monkeypatch.setattr(evaluation, "_trajectory", resumed)
    result = evaluation._run_once(
        {},
        "candidate",
        Ranker(),
        settings,
        backend=cast(Any, object()),
        limits=cast(Any, object()),
        checkpoint_root=tmp_path,
        progress=lambda payload: progress.append(dict(payload)),
    )

    assert resumed_calls == [11]
    assert len(cast(list[Any], result["episodes"])) == 2
    assert progress[0]["restored"] is True
    assert "evaluations_per_second" not in progress[0]
    assert progress[1]["restored"] is False
    assert progress[1]["executed"] == 1
    assert progress[1]["restored_count"] == 1
    assert cast(dict[str, Any], result["runtime"])["executed_episodes"] == 1
    assert cast(dict[str, Any], result["runtime"])["restored_episodes"] == 1
    assert cast(dict[str, Any], result["summary"])["improvement_rate"] == pytest.approx(
        0.5
    )


def test_stale_episode_checkpoint_is_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "episode-000000.json.gz"
    write_json(
        path,
        {
            "schema_version": evaluation.EPISODE_CHECKPOINT_VERSION,
            "identity": "old-request",
            "index": 0,
            "episode": {
                "order": 4,
                "graph_seed": 1,
                "policy_seed": 10,
            },
        },
    )

    assert (
        evaluation._load_episode_checkpoint(
            path,
            identity="new-request",
            index=0,
            order=4,
            graph_seed=1,
            policy_seed=10,
        )
        is None
    )
    with pytest.raises(ValueError, match="does not match request"):
        evaluation._load_episode_checkpoint(
            path,
            identity="old-request",
            index=0,
            order=5,
            graph_seed=1,
            policy_seed=10,
        )


def test_candidate_publishes_development_objective_before_replay(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    class Backend:
        backend_id = "test"
        commit = "a" * 40
        dirty = False
        repo = tmp_path

        def close(self) -> None:
            pass

    config = {
        "evaluation": {
            "order_schedule": "static",
            "orders": [4],
            "graph_seeds": [1],
            "policy_seeds": [2],
            "horizon": 1,
            "proposal_pool_size": 1,
            "baselines": [],
            "replay": True,
        },
        "resources": {"workers": 1, "thread_count": 1},
    }

    def completed_pass(
        _config: object,
        candidate_id: str,
        _source: Any,
        settings: dict[str, Any],
        *,
        pass_name: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "schema_version": evaluation.SCHEMA_VERSION,
            "status": "completed",
            "candidate_id": candidate_id,
            "source_identity": {"source_sha256": "unused"},
            "settings": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in settings.items()
            },
            "episodes": [],
            "summary": {"mean_auc": 0.25, "best_auc": 0.5, "baseline_auc": {}},
            "runtime": {
                "elapsed_seconds": 2.0,
                "execution_seconds": 2.0,
                "executed_episodes": 1,
                "restored_episodes": 0,
            },
            "pass_name": pass_name,
        }

    monkeypatch.setattr(evaluation, "_run_once", completed_pass)
    progress: list[dict[str, Any]] = []
    evaluation.evaluate_candidate(
        config,
        "candidate",
        "def priority(ctx, proposal):\n    return 0\n",
        artifact_root=tmp_path / "artifacts",
        backend=cast(Any, Backend()),
        progress=lambda payload: progress.append(dict(payload)),
    )

    assert progress == [
        {
            "candidate_id": "candidate",
            "pass": "development",
            "pass_completed": True,
            "development_progress": 1.0,
            "replay_progress": 0.0,
            "current_objective": 0.25,
            "best_auc": 0.5,
        }
    ]


def test_settings_use_generation_specific_adaptive_orders() -> None:
    config = {
        "evaluation": {
            "graph_mode": "unrestricted_min_degree_3",
            "order_schedule": "adaptive",
            "min_order": 22,
            "max_order": 128,
            "orders_per_generation": 5,
            "graph_seeds": [1],
            "policy_seeds": [2],
            "horizon": 1,
            "proposal_pool_size": 1,
            "baselines": [],
            "replay": False,
        },
        "resources": {"workers": 1, "thread_count": 1},
    }

    evaluation_config = cast(dict[str, object], config["evaluation"])
    sampled = []
    for generation in (0, 1, 21):
        expected = orders_for_generation(evaluation_config, generation)
        assert evaluation._settings(config, generation)["orders"] == expected
        assert len(expected) == len(set(expected)) == 5
        assert all(22 <= order <= 128 for order in expected)
        sampled.append(expected)
    assert len(set(sampled)) == 3
