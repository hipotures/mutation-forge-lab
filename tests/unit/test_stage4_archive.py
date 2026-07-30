from __future__ import annotations

import hashlib

import pytest

from mutation_forge.stage4.archive import ArchiveError, ProgramArchive, ProgramRecord
from mutation_forge.stage4.checkpoint import Checkpoint, CheckpointStore
from mutation_forge.stage4.selection import behavior_distance, select_parents


def record(
    program_id: str, generation: int, slot: str, *, ast: str, parent: str | None = None
) -> ProgramRecord:
    source = f"# {program_id}\n"
    return ProgramRecord(
        program_id=program_id,
        source_path=f"archive/sources/{program_id}.py",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        normalized_ast_sha256=ast,
        behavior_signature=hashlib.sha256(program_id.encode()).hexdigest(),
        generation=generation,
        slot=slot,
        parent_id=parent,
        request_id=None if generation == 0 else f"request-{program_id}",
        validation_status="valid",
        fitness_status="complete",
        search_metrics={
            "pooled_auc": generation,
            "order10_auc": generation,
            "best_total_witness": 1,
        },
    )


def append(archive: ProgramArchive, item: ProgramRecord) -> None:
    source = archive.root.parent / item.source_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"# {item.program_id}\n", encoding="utf-8")
    archive.append(item)


def test_append_reindex_and_lineage(tmp_path) -> None:
    archive = ProgramArchive(tmp_path / "archive")
    append(archive, record("seed", 0, "slot-00", ast="a" * 64))
    append(archive, record("child", 1, "slot-00", ast="b" * 64, parent="seed"))
    report = archive.reindex()
    assert report.ok
    assert archive.lineage("child") == ("child", "seed")
    assert archive.inspect()["counts"]["records"] == 2


def test_corrupt_record_is_reported(tmp_path) -> None:
    archive = ProgramArchive(tmp_path / "archive")
    (archive.root / "programs" / "bad.json").write_text("not json", encoding="utf-8")
    report = archive.reindex()
    assert report.corrupt_files == ("bad.json",)
    assert not report.ok


def test_lineage_cycle_is_rejected(tmp_path) -> None:
    archive = ProgramArchive(tmp_path / "archive")
    append(archive, record("a", 1, "slot-00", ast="a" * 64, parent="b"))
    append(archive, record("b", 1, "slot-01", ast="b" * 64, parent="a"))
    with pytest.raises(ArchiveError):
        archive.lineage("a")


def test_reindex_rejects_missing_or_mutated_source(tmp_path) -> None:
    archive = ProgramArchive(tmp_path / "archive")
    item = record("seed", 0, "slot-00", ast="a" * 64)
    archive.append(item)
    assert archive.reindex().missing_sources == ("seed",)
    source = archive.root.parent / item.source_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("changed\n", encoding="utf-8")
    assert archive.reindex().source_hash_mismatches == ("seed",)


def test_selection_and_behavior_distance() -> None:
    assert behavior_distance("0" * 64, "f" * 64) == 256
    candidates = [record(f"p{i}", 1, f"slot-{i:02d}", ast=f"{i:064x}") for i in range(8)]
    result = select_parents(candidates)
    assert len(result.parents) == 8
    assert [slot for slot, _ in result.slots] == [f"slot-{i:02d}" for i in range(8)]


def test_checkpoint_partial_resume_and_idempotency(tmp_path) -> None:
    store = CheckpointStore(tmp_path)
    checkpoint = store.save(
        Checkpoint(
            2,
            ("slot-00",),
            ("slot-01",),
            {"slot-00": "p"},
            request_idempotency_keys=("req-1",),
            token_count=12,
        )
    )
    assert not store.should_evaluate("slot-00", checkpoint)
    assert not store.should_request("req-1", checkpoint)
    assert store.should_request("req-2", checkpoint)
    assert store.resume()["token_count"] == 12


def test_checkpoint_sequence_remains_monotonic_after_generation_recovery(
    tmp_path,
) -> None:
    store = CheckpointStore(tmp_path)
    original = store.save(Checkpoint(3, completed_slots=("slot-00",)))
    recovery = store.save(Checkpoint(0, completed_slots=("slot-00",)))
    continued = store.save(
        Checkpoint(0, completed_slots=("slot-00", "slot-01"))
    )
    assert [original.sequence, recovery.sequence, continued.sequence] == [1, 2, 3]
    assert [checkpoint.sequence for checkpoint in store.list()] == [1, 2, 3]
    assert store.latest() == continued
