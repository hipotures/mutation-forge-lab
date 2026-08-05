from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

import pytest

from mutation_forge.native_v3.canonical import canonical_json_bytes
from mutation_forge.native_v3.scheduler import (
    EpisodeShard,
    EpisodeSpec,
    EpisodeTask,
    EpochSnapshot,
    EpochStatus,
    GeneratedEntry,
    ProviderBatch,
    ProviderCall,
    RetryableShardFailure,
    SchedulerConfig,
    SlotStatus,
    StreamingEpochScheduler,
    build_episode_shards,
    split_residual_shard,
)


def _snapshot(slot_count: int = 8) -> EpochSnapshot:
    return EpochSnapshot(
        epoch_id="epoch-0001",
        epoch_number=1,
        parent_program_hashes=("parent",),
        archive_snapshot_hash="archive",
        development_manifest_hash="development",
        protocol_bundle_hash="protocol",
        planned_slot_ids=tuple(f"slot-{index:02d}" for index in range(slot_count)),
    )


def _task(program_hash: str, episode: EpisodeSpec) -> EpisodeTask:
    return EpisodeTask(
        program_hash=program_hash,
        manifest_hash="development",
        protocol_bundle_hash="protocol",
        initial_graph_hash=f"graph-{episode.order}-{episode.graph_seed}",
        horizon=4,
        interpreter_protocol="interpreter",
        selector_protocol="selectors",
        score_protocol="score",
        acceptance_protocol="acceptance",
        episode=episode,
    )


def _shards(program_hash: str, _program: str) -> Sequence[EpisodeShard]:
    episodes = (
        EpisodeSpec(order, graph_seed, policy_seed)
        for order in (8, 10)
        for graph_seed in (1, 2)
        for policy_seed in (3, 4)
    )
    return build_episode_shards(
        program_hash=program_hash,
        task_factory=lambda episode: _task(program_hash, episode),
        episodes=episodes,
        shard_size=1,
    )


def _config(*, provider_concurrency: int = 2, evaluator_workers: int = 4) -> SchedulerConfig:
    return SchedulerConfig(
        provider_concurrency=provider_concurrency,
        evaluator_workers=evaluator_workers,
        provider_batch_size=2,
        candidate_queue_capacity=8,
        evaluation_queue_capacity=16,
        target_evaluation_backlog=8,
    )


def test_streaming_starts_evaluation_before_slow_provider_call_finishes() -> None:
    late_provider_finished = threading.Event()
    evaluation_started = threading.Event()
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def provider(call: ProviderCall) -> ProviderBatch[str]:
        if call.call_id.endswith("0001"):
            time.sleep(0.08)
            late_provider_finished.set()
        else:
            time.sleep(0.005)
        entries = tuple(
            GeneratedEntry(slot, f"program:{slot}", f"hash:{slot}") for slot in call.slot_ids
        )
        return ProviderBatch(call.call_id, entries, 100)

    def evaluate(shard: EpisodeShard, _program: str) -> Sequence[str]:
        nonlocal active, peak_active
        assert not late_provider_finished.is_set() or evaluation_started.is_set()
        evaluation_started.set()
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return (shard.tasks[0].episode_id,)

    result = StreamingEpochScheduler(
        config=_config(),
        provider_call=provider,
        build_shards=_shards,
        evaluate_shard=evaluate,
    ).run(_snapshot())

    assert evaluation_started.is_set()
    assert peak_active == 4
    first_evaluation = next(
        event for event in result.telemetry if event.name == "evaluation_shard_started"
    )
    slow_provider = next(
        event
        for event in result.telemetry
        if event.name == "provider_call_completed"
        and event.fields["call_id"] == "epoch-0001:provider:0001"
    )
    assert first_evaluation.monotonic_ns < slow_provider.monotonic_ns
    assert result.epoch_status is EpochStatus.COMPLETE


def test_one_provider_batch_publishes_first_valid_entry_before_sibling_validation() -> None:
    evaluation_started = threading.Event()
    first_entry_waited_for_evaluator = threading.Event()

    def provider(call: ProviderCall) -> ProviderBatch[str]:
        raise AssertionError(f"non-streaming provider path used for {call.call_id}")

    def streaming_provider(
        call: ProviderCall,
        entry_sink: Callable[[GeneratedEntry[str]], None],
    ) -> ProviderBatch[str]:
        entries = tuple(
            GeneratedEntry(slot, f"program:{slot}", f"hash:{slot}") for slot in call.slot_ids
        )
        entry_sink(entries[0])
        assert evaluation_started.wait(timeout=2)
        first_entry_waited_for_evaluator.set()
        for entry in entries[1:]:
            entry_sink(entry)
        return ProviderBatch(call.call_id, entries, 100)

    def evaluate(shard: EpisodeShard, _program: str) -> Sequence[str]:
        evaluation_started.set()
        return (shard.tasks[0].episode_id,)

    result = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=provider,
        streaming_provider_call=streaming_provider,
        build_shards=_shards,
        evaluate_shard=evaluate,
    ).run(_snapshot())

    assert first_entry_waited_for_evaluator.is_set()
    assert result.epoch_status is EpochStatus.COMPLETE


def test_invalid_batch_entry_does_not_block_valid_sibling() -> None:
    def provider(call: ProviderCall) -> ProviderBatch[str]:
        first, second = call.slot_ids
        return ProviderBatch(
            call.call_id,
            (
                GeneratedEntry(first, f"program:{first}", f"hash:{first}"),
                GeneratedEntry(second, None, None, "invalid AST"),
            ),
            100,
        )

    result = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=provider,
        build_shards=_shards,
        evaluate_shard=lambda shard, _program: (shard.tasks[0].episode_id,),
    ).run(_snapshot())

    assert result.slot_statuses["slot-00"] is SlotStatus.COMPLETE
    assert result.slot_statuses["slot-01"] is SlotStatus.INVALID
    assert len(result.program_results["hash:slot-00"]) == 8
    assert result.epoch_status is EpochStatus.DEGRADED


def test_provider_failure_telemetry_preserves_safe_diagnostics() -> None:
    def provider(_call: ProviderCall) -> ProviderBatch[str]:
        raise RuntimeError("isolated Codex home is not authenticated")

    result = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=provider,
        build_shards=_shards,
        evaluate_shard=lambda shard, _program: (shard.tasks[0].episode_id,),
    ).run(_snapshot(slot_count=2))

    failure = next(event for event in result.telemetry if event.name == "provider_call_failed")
    assert failure.fields["error_type"] == "RuntimeError"
    assert failure.fields["error_message"] == "isolated Codex home is not authenticated"
    assert failure.fields["provider_calls_in_flight"] == 0
    assert result.epoch_status is EpochStatus.INCONCLUSIVE


def test_provider_call_activity_reports_slots_elapsed_time_and_locked_timeout() -> None:
    def provider(call: ProviderCall) -> ProviderBatch[str]:
        time.sleep(0.03)
        entries = tuple(
            GeneratedEntry(slot, f"program:{slot}", f"hash:{slot}")
            for slot in call.slot_ids
        )
        return ProviderBatch(call.call_id, entries, 100)

    result = StreamingEpochScheduler(
        config=SchedulerConfig(
            provider_concurrency=1,
            evaluator_workers=2,
            provider_batch_size=2,
            candidate_queue_capacity=8,
            evaluation_queue_capacity=16,
            target_evaluation_backlog=8,
            provider_call_timeout_seconds=600.0,
            provider_activity_interval_seconds=0.005,
        ),
        provider_call=provider,
        build_shards=_shards,
        evaluate_shard=lambda shard, _program: (shard.tasks[0].episode_id,),
    ).run(_snapshot(slot_count=2))

    started = next(event for event in result.telemetry if event.name == "provider_call_started")
    activities = [
        event for event in result.telemetry if event.name == "provider_call_activity"
    ]
    completed = next(
        event for event in result.telemetry if event.name == "provider_call_completed"
    )
    assert started.fields["slot_ids"] == "slot-00,slot-01"
    assert started.fields["timeout_ns"] == 600_000_000_000
    assert activities
    assert all(
        event.fields["operation_elapsed_ns"] > 0
        and event.fields["timeout_ns"] == 600_000_000_000
        for event in activities
    )
    assert completed.fields["slot_ids"] == "slot-00,slot-01"
    for event in result.telemetry:
        canonical_json_bytes(dict(event.fields))


def test_provider_is_not_invoked_until_started_event_is_accepted() -> None:
    provider_calls = 0

    def provider(call: ProviderCall) -> ProviderBatch[str]:
        nonlocal provider_calls
        provider_calls += 1
        return ProviderBatch(call.call_id, (), 0)

    def reject_started_event(event: object) -> None:
        if getattr(event, "name", None) == "provider_call_started":
            raise RuntimeError("semantic event persistence rejected")

    scheduler = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=provider,
        build_shards=_shards,
        evaluate_shard=lambda shard, _program: (shard.tasks[0].episode_id,),
        telemetry_sink=reject_started_event,
    )

    with pytest.raises(RuntimeError, match="semantic event persistence rejected"):
        scheduler.run(_snapshot(slot_count=2))

    assert provider_calls == 0


def test_late_duplicate_alias_joins_the_in_flight_canonical_evaluation() -> None:
    evaluation_started = threading.Event()

    def streaming_provider(
        call: ProviderCall,
        entry_sink: Callable[[GeneratedEntry[str]], None],
    ) -> ProviderBatch[str]:
        entries = tuple(GeneratedEntry(slot, "same-program", "same-hash") for slot in call.slot_ids)
        entry_sink(entries[0])
        assert evaluation_started.wait(timeout=2)
        entry_sink(entries[1])
        return ProviderBatch(call.call_id, entries, 100)

    def evaluate(shard: EpisodeShard, _program: str) -> Sequence[str]:
        evaluation_started.set()
        time.sleep(0.002)
        return (shard.tasks[0].episode_id,)

    result = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=lambda call: ProviderBatch(call.call_id, (), 0),
        streaming_provider_call=streaming_provider,
        build_shards=_shards,
        evaluate_shard=evaluate,
    ).run(_snapshot())

    assert result.program_aliases["same-hash"] == _snapshot().planned_slot_ids
    assert all(status is SlotStatus.COMPLETE for status in result.slot_statuses.values())
    assert len(result.program_results["same-hash"]) == 8


def test_semantic_result_is_independent_of_provider_and_shard_completion_order() -> None:
    def run(reverse: bool) -> tuple[object, object, object]:
        def provider(call: ProviderCall) -> ProviderBatch[str]:
            ordinal = int(call.call_id.rsplit(":", 1)[1])
            time.sleep((3 - ordinal if reverse else ordinal) * 0.001)
            return ProviderBatch(
                call.call_id,
                tuple(
                    GeneratedEntry(slot, f"program:{slot}", f"hash:{slot}")
                    for slot in call.slot_ids
                ),
                100,
            )

        def evaluate(shard: EpisodeShard, _program: str) -> Sequence[str]:
            ordinal = int(shard.shard_id[-2:], 16)
            time.sleep(((255 - ordinal) if reverse else ordinal) % 4 * 0.001)
            return (shard.tasks[0].episode_id,)

        result = StreamingEpochScheduler(
            config=_config(),
            provider_call=provider,
            build_shards=_shards,
            evaluate_shard=evaluate,
        ).run(_snapshot())
        return result.slot_statuses, result.program_aliases, result.program_results

    assert run(False) == run(True)


def test_timeout_residual_is_split_into_single_episode_shards() -> None:
    original = _shards("hash", "program")[0]
    combined = EpisodeShard(
        "combined",
        "hash",
        (original.tasks[0], _shards("hash", "program")[1].tasks[0]),
    )
    residuals = split_residual_shard(combined)
    assert len(residuals) == 2
    assert all(len(shard.tasks) == 1 for shard in residuals)
    assert all(shard.residual_of == "combined" for shard in residuals)


def test_infrastructure_failure_is_rescheduled_once_as_residual_work() -> None:
    attempts: dict[str, int] = {}

    def provider(call: ProviderCall) -> ProviderBatch[str]:
        return ProviderBatch(
            call.call_id,
            tuple(
                GeneratedEntry(slot, f"program:{slot}", f"hash:{slot}") for slot in call.slot_ids
            ),
            100,
        )

    def evaluate(shard: EpisodeShard, _program: str) -> Sequence[str]:
        root = shard.residual_of or shard.shard_id
        attempts[root] = attempts.get(root, 0) + 1
        if shard.residual_of is None:
            raise RetryableShardFailure("worker lost")
        return (shard.tasks[0].episode_id,)

    result = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=provider,
        build_shards=_shards,
        evaluate_shard=evaluate,
    ).run(_snapshot())
    assert result.epoch_status is EpochStatus.COMPLETE
    assert any(event.name == "evaluation_shard_rescheduled" for event in result.telemetry)


def test_recovered_provider_batch_is_not_called_again() -> None:
    called: list[str] = []

    def provider(call: ProviderCall) -> ProviderBatch[str]:
        called.append(call.call_id)
        return ProviderBatch(
            call.call_id,
            tuple(
                GeneratedEntry(slot, f"program:{slot}", f"hash:{slot}") for slot in call.slot_ids
            ),
            100,
        )

    recovered = (
        GeneratedEntry("slot-00", "program:slot-00", "hash:slot-00"),
        GeneratedEntry("slot-01", "program:slot-01", "hash:slot-01"),
    )
    result = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=provider,
        build_shards=_shards,
        evaluate_shard=lambda shard, _program: (shard.tasks[0].episode_id,),
    ).run(_snapshot(), recovered_entries=recovered)
    assert "epoch-0001:provider:0000" not in called
    assert called == [
        "epoch-0001:provider:0001",
        "epoch-0001:provider:0002",
        "epoch-0001:provider:0003",
    ]
    assert result.slot_statuses["slot-00"] is SlotStatus.COMPLETE


def test_partial_recovered_provider_batch_fails_closed() -> None:
    scheduler = StreamingEpochScheduler(
        config=_config(provider_concurrency=1, evaluator_workers=2),
        provider_call=lambda call: ProviderBatch(call.call_id, (), 0),
        build_shards=_shards,
        evaluate_shard=lambda _shard, _program: (),
    )
    with pytest.raises(ValueError, match="atomically committed"):
        scheduler.run(
            _snapshot(),
            recovered_entries=(GeneratedEntry("slot-00", "program:slot-00", "hash:slot-00"),),
        )
