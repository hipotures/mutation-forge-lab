from pathlib import Path

from mutation_forge.stage2b.config import load_stage2b_config
from mutation_forge.stage2b.rankers import SourceRanker
from mutation_forge.stage3.evaluation import (
    reduce_records,
    run_development_episode,
    validate_record,
)
from mutation_forge.stage3.manifest import build_manifest


def test_stage3_reduced_episode_accounting_and_reducer() -> None:
    config = load_stage2b_config("configs/stage2b-preregistered.toml")
    sources = {
        name: Path(f"fixtures/rankers/stage2b_{name}.py").read_text()
        for name in ("random", "structural")
    }
    rankers = {name: SourceRanker(name, source, config.sandbox) for name, source in sources.items()}
    try:
        episode = dict(build_manifest()["episodes"][0])
        episode["horizon"] = 2
        record = run_development_episode(config, episode, rankers)
        validate_record(record, {"random", "structural"})
        assert record["selected_score_calls"] == 2 * 2
        assert record["oracle_score_calls"] == 0
        reduced = reduce_records([record], {str(episode["episode_id"])}, {"random", "structural"})
        assert reduced[0]["episode_id"] == episode["episode_id"]
    finally:
        for ranker in rankers.values():
            ranker.close()
