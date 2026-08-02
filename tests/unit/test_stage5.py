from __future__ import annotations

from fractions import Fraction

from mutation_forge import cli
from mutation_forge.backends.toy import ToyBackend
from mutation_forge.stage5_config import (
    CHAMPION_ID,
    RANDOM_ID,
    STAGE3_COMPARATOR_ID,
    STRUCTURAL_ID,
    build_manifest,
    load_stage5_config,
)
from mutation_forge.stage5_execution import timing_stripped_projection
from mutation_forge.stage5_metrics import (
    EFFECT_STAGE3,
    PolicyAreaEpisode,
    bootstrap_stage5,
    summarize_stage5,
)
from mutation_forge.stage5_relabel import (
    apply_permutation,
    canonical_unlabeled_identity,
    deterministic_permutation,
    graph_label_hash,
    relabel_contract_digest,
    relabel_graph,
)

POLICIES = (CHAMPION_ID, STAGE3_COMPARATOR_ID, RANDOM_ID, STRUCTURAL_ID)


def test_stage5_parser_exposes_frozen_commands() -> None:
    parser = cli._build_legacy_parser()
    for command in ("freeze", "generalize"):
        parsed = parser.parse_args(["stage5", command, "--json"])
        assert parsed.command == "stage5"
        assert parsed.stage5_command == command
    parsed = parser.parse_args(
        [
            "stage5",
            "finalize",
            "--preserved-evidence",
            "/tmp/evidence",
            "--evidence-manifest-sha256",
            "0" * 64,
            "--json",
        ]
    )
    assert parsed.command == "stage5"
    assert parsed.stage5_command == "finalize"


def test_stage5_manifest_is_exact_and_sharded() -> None:
    config = load_stage5_config("configs/stage5-generalization.toml")
    manifest = build_manifest(config)
    assert manifest["episode_count"] == 1536
    assert manifest["shard_count"] == 24
    assert all(item["episode_count"] == 64 for item in manifest["shards"])
    rows = manifest["episodes"]
    assert len({(row["order"], row["graph_seed"], row["policy_seed"]) for row in rows}) == 768
    assert len({row["episode_id"] for row in rows}) == 1536


def test_vertex_relabeling_is_deterministic_and_reversible() -> None:
    backend = ToyBackend()
    base = backend.generate_seed(order=14, seed=601)
    first, permutation = relabel_graph(base, graph_seed=601, relabeling_seed=6101)
    second, same_permutation = relabel_graph(base, graph_seed=601, relabeling_seed=6101)
    assert first == second
    assert permutation == same_permutation
    assert graph_label_hash(first) == graph_label_hash(second)
    assert canonical_unlabeled_identity(base) == canonical_unlabeled_identity(base)
    assert deterministic_permutation(14, 601, 6101) == permutation
    inverse = tuple(permutation.index(index) for index in range(len(permutation)))
    assert apply_permutation(first, inverse) == base
    other, _ = relabel_graph(base, graph_seed=601, relabeling_seed=6102)
    assert canonical_unlabeled_identity(first) == canonical_unlabeled_identity(other)
    assert graph_label_hash(first) != graph_label_hash(other)
    assert relabel_contract_digest((14,), (601,), (6101, 6102))


def test_stage5_hierarchical_pair_effect_uses_six_order_relabel_strata() -> None:
    episodes: list[PolicyAreaEpisode] = []
    for order in (14, 18, 22):
        for graph_seed in range(601, 617):
            for relabeling_seed in (6101, 6102):
                for policy_seed in range(6001, 6017):
                    graph_offset = Fraction(graph_seed - 601, 1000)
                    relabel_offset = Fraction(relabeling_seed - 6101, 100)
                    areas = {
                        CHAMPION_ID: Fraction(1, 2) + graph_offset + relabel_offset,
                        STAGE3_COMPARATOR_ID: Fraction(2, 5),
                        RANDOM_ID: Fraction(1, 5),
                        STRUCTURAL_ID: Fraction(1, 2),
                    }
                    episodes.append(
                        PolicyAreaEpisode(
                            order=order,
                            graph_seed=graph_seed,
                            relabeling_seed=relabeling_seed,
                            policy_seed=policy_seed,
                            episode_id=f"{order}-{graph_seed}-{relabeling_seed}-{policy_seed}",
                            areas=areas,
                        )
                    )
    summary = summarize_stage5(episodes, POLICIES)
    effect = summary.effects[EFFECT_STAGE3]
    assert len(effect.stratum_deltas) == 6
    assert effect.stratum_deltas[(14, 6101)] == (
        Fraction(1, 2) + Fraction(15, 2000) - Fraction(2, 5)
    )
    assert effect.stratum_deltas[(14, 6102)] == (
        Fraction(1, 2) + Fraction(15, 2000) + Fraction(1, 100) - Fraction(2, 5)
    )
    assert summary.policy_means[CHAMPION_ID] > summary.policy_means[STAGE3_COMPARATOR_ID]
    bootstrap = bootstrap_stage5(summary, samples=8, seed=2026080103)
    assert bootstrap.observed[EFFECT_STAGE3] == effect.theta
    assert bootstrap.support.episode_count == 1536


def test_timing_stripping_is_recursive_and_identity_stable() -> None:
    value = {
        "identity": "same",
        "timing_ns": {"total": 1},
        "nested": [
            {"elapsed_ns": 2, "custom_ns": 7, "value": 3},
            {"value": {"started_at": 4, "kept": True}},
        ],
    }
    changed = {
        "identity": "same",
        "timing_ns": {"total": 999},
        "nested": [
            {"elapsed_ns": 777, "custom_ns": 777, "value": 3},
            {"value": {"started_at": 888, "kept": True}},
        ],
    }
    assert timing_stripped_projection(value) == timing_stripped_projection(changed)
    assert timing_stripped_projection(value) == {
        "identity": "same",
        "nested": [{"value": 3}, {"value": {"kept": True}}],
    }
