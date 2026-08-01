# ruff: noqa: E501
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mutation_forge.models import JsonValue
from mutation_forge.sandbox.contracts import (
    SandboxLimits,
    ScientificContext,
    ScientificProposal,
)
from mutation_forge.sandbox.worker import PolicyWorker
from mutation_forge.stage7_heg_bridge.contract import (
    FROZEN_IDENTITY,
    REPLAY_SCHEMA_VERSION,
    canonical_json_hash,
    catalog_source,
)
from mutation_forge.stage7_heg_bridge.fixtures import HEGFixture, build_fixtures

REPLAY_RECORD_COUNT = 2_048


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    record_id: str
    fixture_id: str
    pool_hash: str
    context: dict[str, JsonValue]
    proposal: dict[str, JsonValue]
    expected_priority: int | float
    expected_rank: int
    expected_selected_proposal_id: str
    expected_rank_order: tuple[str, ...]

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "record_id": self.record_id,
            "fixture_id": self.fixture_id,
            "pool_hash": self.pool_hash,
            "context": self.context,
            "proposal": self.proposal,
            "expected_priority": self.expected_priority,
            "expected_rank": self.expected_rank,
            "expected_selected_proposal_id": self.expected_selected_proposal_id,
            "expected_rank_order": list(self.expected_rank_order),
        }


@dataclass(frozen=True, slots=True)
class ReplayCorpus:
    records: tuple[ReplayRecord, ...]
    corpus_hash: str
    fixture_hashes: dict[str, str]

    def as_dict(self) -> dict[str, JsonValue]:
        records = [record.as_dict() for record in self.records]
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "policy_identity": FROZEN_IDENTITY.as_dict(),
            "record_count": len(records),
            "corpus_hash": self.corpus_hash,
            "fixture_hashes": dict(sorted(self.fixture_hashes.items())),
            "records": cast(list[JsonValue], records),
        }


def _fixture_hash(fixture: HEGFixture) -> str:
    return canonical_json_hash(fixture.as_dict(include_plans=True, include_timings=False))


def build_corpus(
    bridge: Any,
    *,
    record_count: int = REPLAY_RECORD_COUNT,
) -> ReplayCorpus:
    if record_count < REPLAY_RECORD_COUNT:
        raise ValueError(f"the frozen corpus requires at least {REPLAY_RECORD_COUNT} records")
    fixtures = build_fixtures(bridge)
    ranked_by_fixture: dict[str, Any] = {}
    for fixture in fixtures:
        ranked_by_fixture[fixture.spec.fixture_id] = bridge.ranker.rank(
            fixture.context,
            fixture.pool,
        )
    records: list[ReplayRecord] = []
    for index in range(record_count):
        fixture = fixtures[index % len(fixtures)]
        ranking = ranked_by_fixture[fixture.spec.fixture_id]
        candidate_index = (index * 5 + index // len(fixtures)) % len(fixture.pool.candidates)
        candidate = fixture.pool.candidates[candidate_index]
        by_id = {item.proposal_id: item for item in ranking.ranked}
        ranked_ids = tuple(item.proposal_id for item in ranking.ranked)
        ranked_item = by_id[candidate.proposal_id]
        records.append(
            ReplayRecord(
                record_id=f"replay-{index:06d}",
                fixture_id=fixture.spec.fixture_id,
                pool_hash=fixture.pool.pool_hash,
                context=cast(dict[str, JsonValue], fixture.context),
                proposal=cast(dict[str, JsonValue], candidate.payload),
                expected_priority=ranked_item.priority,
                expected_rank=ranked_ids.index(candidate.proposal_id),
                expected_selected_proposal_id=ranking.selected_proposal_id,
                expected_rank_order=ranked_ids,
            )
        )
    canonical_records = [record.as_dict() for record in records]
    return ReplayCorpus(
        records=tuple(records),
        corpus_hash=canonical_json_hash(canonical_records),
        fixture_hashes={fixture.spec.fixture_id: _fixture_hash(fixture) for fixture in fixtures},
    )


def write_corpus(path: Path, corpus: ReplayCorpus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(corpus.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def load_corpus(path: Path) -> ReplayCorpus:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported Stage 7 replay schema")
    records_raw = raw.get("records")
    if not isinstance(records_raw, list) or len(records_raw) < REPLAY_RECORD_COUNT:
        raise ValueError("replay corpus is smaller than the frozen minimum")
    records: list[ReplayRecord] = []
    for item in records_raw:
        if not isinstance(item, dict):
            raise ValueError("replay record must be an object")
        records.append(
            ReplayRecord(
                record_id=str(item["record_id"]),
                fixture_id=str(item["fixture_id"]),
                pool_hash=str(item["pool_hash"]),
                context=cast(dict[str, JsonValue], item["context"]),
                proposal=cast(dict[str, JsonValue], item["proposal"]),
                expected_priority=cast(int | float, item["expected_priority"]),
                expected_rank=int(item["expected_rank"]),
                expected_selected_proposal_id=str(item["expected_selected_proposal_id"]),
                expected_rank_order=tuple(cast(list[str], item["expected_rank_order"])),
            )
        )
    canonical_records = [record.as_dict() for record in records]
    corpus_hash = canonical_json_hash(canonical_records)
    if corpus_hash != raw.get("corpus_hash"):
        raise ValueError("replay corpus hash mismatch")
    return ReplayCorpus(
        records=tuple(records),
        corpus_hash=corpus_hash,
        fixture_hashes=cast(dict[str, str], raw.get("fixture_hashes", {})),
    )


def _record_result(worker: PolicyWorker, record: ReplayRecord) -> tuple[int | float, int]:
    started = time.perf_counter_ns()
    result = worker.call(
        cast(ScientificContext, record.context),
        cast(ScientificProposal, record.proposal),
    )
    elapsed = time.perf_counter_ns() - started
    if result.status != "ok" or result.priority is None:
        raise RuntimeError(f"replay policy call failed for {record.record_id}: {result.error}")
    return result.priority, elapsed


def run_replay(
    path: Path,
    *,
    limits: SandboxLimits | None = None,
) -> dict[str, JsonValue]:
    corpus = load_corpus(path)
    worker = PolicyWorker(catalog_source(), limits or SandboxLimits())
    priorities: list[JsonValue] = []
    elapsed: list[int] = []
    mismatches: list[str] = []
    try:
        if worker.identity.source_sha256 != FROZEN_IDENTITY.source_sha256:
            raise RuntimeError("worker source identity drift")
        for record in corpus.records:
            priority, call_elapsed = _record_result(worker, record)
            priorities.append(priority)
            elapsed.append(call_elapsed)
            if priority != record.expected_priority:
                mismatches.append(record.record_id)
    finally:
        telemetry = worker.telemetry()
        worker.close()
    replay_payload = {
        "record_priorities": priorities,
        "record_ids": [record.record_id for record in corpus.records],
    }
    return {
        "status": "passed" if not mismatches else "failed",
        "schema_version": REPLAY_SCHEMA_VERSION,
        "record_count": len(corpus.records),
        "policy_call_count": len(corpus.records),
        "expected_corpus_hash": corpus.corpus_hash,
        "observed_replay_hash": canonical_json_hash(replay_payload),
        "priority_mismatch_count": len(mismatches),
        "priority_mismatch_ids": cast(list[JsonValue], mismatches),
        "latency_ns": {
            "min": min(elapsed, default=0),
            "max": max(elapsed, default=0),
            "sum": sum(elapsed),
        },
        "worker_telemetry": telemetry,
        "identity": FROZEN_IDENTITY.as_dict(),
    }


def verify_ordering_from_fixtures(bridge: Any, fixtures: tuple[HEGFixture, ...]) -> dict[str, JsonValue]:
    mismatches: list[str] = []
    for fixture in fixtures:
        ranking = bridge.ranker.rank(fixture.context, fixture.pool)
        if ranking.selected_proposal_id is None:
            mismatches.append(fixture.spec.fixture_id)
            continue
        if ranking.ranked[0].proposal_id != ranking.selected_proposal_id:
            mismatches.append(fixture.spec.fixture_id)
    return {
        "fixture_count": len(fixtures),
        "ordering_mismatch_count": len(mismatches),
        "ordering_mismatch_ids": cast(list[JsonValue], mismatches),
        "status": "passed" if not mismatches else "failed",
    }
