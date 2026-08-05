"""Bounded streaming epoch scheduler for Native v3.

The scheduler is deliberately agnostic about providers and graph backends.  It
owns concurrency, backpressure, deterministic work identities, and epoch
barriers; provider parsing and episode evaluation are injected pure callables.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from .canonical import canonical_json_bytes, domain_hash

ProgramT = TypeVar("ProgramT")
ResultT = TypeVar("ResultT")

SCHEDULER_PROTOCOL_ID = "native_v3_streaming_scheduler_v1"
SHARD_PROTOCOL_ID = "native_v3_episode_interleave_v1"


class EpochStatus(StrEnum):
    COMPLETE = "COMPLETE"
    DEGRADED = "DEGRADED"
    INCONCLUSIVE = "INCONCLUSIVE"


class SlotStatus(StrEnum):
    PLANNED = "PLANNED"
    GENERATING = "GENERATING"
    VALID = "VALID"
    INVALID = "INVALID"
    EVALUATING = "EVALUATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class RetryableShardFailure(RuntimeError):
    """Infrastructure work may be rescheduled once as residual microshards."""


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    order: int
    graph_seed: int
    policy_seed: int
    validation_shard: str = "development"

    def sort_key(self) -> tuple[int, int, int, str]:
        return (self.order, self.graph_seed, self.policy_seed, self.validation_shard)


@dataclass(frozen=True, slots=True)
class EpisodeTask:
    program_hash: str
    manifest_hash: str
    protocol_bundle_hash: str
    initial_graph_hash: str
    horizon: int
    interpreter_protocol: str
    selector_protocol: str
    score_protocol: str
    acceptance_protocol: str
    episode: EpisodeSpec

    @property
    def episode_id(self) -> str:
        payload = {
            "program_hash": self.program_hash,
            "panel_manifest_hash": self.manifest_hash,
            "initial_graph_hash": self.initial_graph_hash,
            "order": self.episode.order,
            "graph_seed": self.episode.graph_seed,
            "policy_seed": self.episode.policy_seed,
            "horizon": self.horizon,
            "interpreter_protocol": self.interpreter_protocol,
            "selector_protocol": self.selector_protocol,
            "score_protocol": self.score_protocol,
            "acceptance_protocol": self.acceptance_protocol,
        }
        return domain_hash(
            b"mforge-native-v3-episode\0",
            canonical_json_bytes(payload),
        )


@dataclass(frozen=True, slots=True)
class EpisodeShard:
    shard_id: str
    program_hash: str
    tasks: tuple[EpisodeTask, ...]
    residual_of: str | None = None


@dataclass(frozen=True, slots=True)
class EpochSnapshot:
    epoch_id: str
    epoch_number: int
    parent_program_hashes: tuple[str, ...]
    archive_snapshot_hash: str
    development_manifest_hash: str
    protocol_bundle_hash: str
    planned_slot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.planned_slot_ids)) != len(self.planned_slot_ids):
            raise ValueError("epoch slot IDs must be unique")
        if not self.planned_slot_ids:
            raise ValueError("an epoch must plan at least one slot")


@dataclass(frozen=True, slots=True)
class ProviderCall:
    call_id: str
    slot_ids: tuple[str, ...]
    snapshot: EpochSnapshot
    repair_of: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratedEntry[ProgramT]:
    slot_id: str
    program: ProgramT | None
    program_hash: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        valid = self.program is not None and self.program_hash is not None
        if valid == (self.error is not None):
            raise ValueError("generated entry must contain either a program or an error")


@dataclass(frozen=True, slots=True)
class ProviderBatch[ProgramT]:
    call_id: str
    entries: tuple[GeneratedEntry[ProgramT], ...]
    response_bytes: int


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    provider_concurrency: int
    evaluator_workers: int
    provider_batch_size: int
    candidate_queue_capacity: int
    evaluation_queue_capacity: int
    target_evaluation_backlog: int
    candidate_shard_size: int = 1
    auxiliary_shard_size: int = 1

    def __post_init__(self) -> None:
        values = (
            self.provider_concurrency,
            self.evaluator_workers,
            self.provider_batch_size,
            self.candidate_queue_capacity,
            self.evaluation_queue_capacity,
            self.target_evaluation_backlog,
            self.candidate_shard_size,
            self.auxiliary_shard_size,
        )
        if any(value <= 0 for value in values):
            raise ValueError("scheduler limits must be positive")
        if self.target_evaluation_backlog > self.evaluation_queue_capacity:
            raise ValueError("target backlog exceeds evaluation queue capacity")
        if self.candidate_queue_capacity < self.provider_concurrency * self.provider_batch_size:
            raise ValueError(
                "candidate queue must hold every result from all in-flight provider calls"
            )


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    name: str
    monotonic_ns: int
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamingEpochResult[ResultT]:
    epoch_status: EpochStatus
    slot_statuses: Mapping[str, SlotStatus]
    program_results: Mapping[str, tuple[ResultT, ...]]
    program_aliases: Mapping[str, tuple[str, ...]]
    invalid_slots: Mapping[str, str]
    telemetry: tuple[TelemetryEvent, ...]


def deterministic_interleave(episodes: Iterable[EpisodeSpec]) -> tuple[EpisodeSpec, ...]:
    """Versioned round-robin minimizing concentration by order and seed class."""

    groups: dict[int, deque[EpisodeSpec]] = {}
    ordered = sorted(episodes, key=EpisodeSpec.sort_key)
    for episode in ordered:
        groups.setdefault(episode.order, deque()).append(episode)
    output: list[EpisodeSpec] = []
    while groups:
        for order in sorted(groups):
            group = groups[order]
            output.append(group.popleft())
        groups = {order: group for order, group in groups.items() if group}
    return tuple(output)


def build_episode_shards(
    *,
    program_hash: str,
    task_factory: Callable[[EpisodeSpec], EpisodeTask],
    episodes: Iterable[EpisodeSpec],
    shard_size: int,
    residual_of: str | None = None,
) -> tuple[EpisodeShard, ...]:
    if shard_size <= 0:
        raise ValueError("shard size must be positive")
    tasks = tuple(task_factory(episode) for episode in deterministic_interleave(episodes))
    shards: list[EpisodeShard] = []
    for ordinal, offset in enumerate(range(0, len(tasks), shard_size)):
        shard_tasks = tasks[offset : offset + shard_size]
        identity = canonical_json_bytes(
            {
                "protocol": SHARD_PROTOCOL_ID,
                "program_hash": program_hash,
                "episode_ids": [task.episode_id for task in shard_tasks],
                "residual_of": residual_of,
                "ordinal": ordinal,
            }
        )
        shards.append(
            EpisodeShard(
                shard_id=domain_hash(b"mforge-native-v3-shard\0", identity),
                program_hash=program_hash,
                tasks=shard_tasks,
                residual_of=residual_of,
            )
        )
    return tuple(shards)


def split_residual_shard(shard: EpisodeShard) -> tuple[EpisodeShard, ...]:
    """Split one timed-out shard into deterministic one-episode residual shards."""

    return build_episode_shards(
        program_hash=shard.program_hash,
        task_factory=lambda episode: next(task for task in shard.tasks if task.episode == episode),
        episodes=(task.episode for task in shard.tasks),
        shard_size=1,
        residual_of=shard.shard_id,
    )


class StreamingEpochScheduler[ProgramT, ResultT]:
    """Run one frozen epoch as a bounded producer-consumer pipeline."""

    def __init__(
        self,
        *,
        config: SchedulerConfig,
        provider_call: Callable[[ProviderCall], ProviderBatch[ProgramT]],
        streaming_provider_call: (
            Callable[
                [ProviderCall, Callable[[GeneratedEntry[ProgramT]], None]],
                ProviderBatch[ProgramT],
            ]
            | None
        ) = None,
        build_shards: Callable[[str, ProgramT], Sequence[EpisodeShard]],
        evaluate_shard: Callable[[EpisodeShard, ProgramT], Sequence[ResultT]],
        provider_executor: Executor | None = None,
        evaluator_executor: Executor | None = None,
        telemetry_sink: Callable[[TelemetryEvent], None] | None = None,
        result_sink: Callable[[EpisodeShard, tuple[ResultT, ...]], None] | None = None,
    ) -> None:
        self.config = config
        self.provider_call = provider_call
        self.streaming_provider_call = streaming_provider_call
        self.build_shards = build_shards
        self.evaluate_shard = evaluate_shard
        self._provider_executor = provider_executor
        self._evaluator_executor = evaluator_executor
        self.telemetry_sink = telemetry_sink
        self.result_sink = result_sink

    def run(
        self,
        snapshot: EpochSnapshot,
        *,
        auxiliary_programs: Mapping[str, ProgramT] | None = None,
        recovered_entries: Sequence[GeneratedEntry[ProgramT]] = (),
    ) -> StreamingEpochResult[ResultT]:
        recovered_slots = {entry.slot_id for entry in recovered_entries}
        if len(recovered_slots) != len(recovered_entries) or not recovered_slots.issubset(
            snapshot.planned_slot_ids
        ):
            raise ValueError("recovered provider entries must have unique planned slot IDs")
        provider_groups = tuple(
            tuple(snapshot.planned_slot_ids[offset : offset + self.config.provider_batch_size])
            for offset in range(
                0,
                len(snapshot.planned_slot_ids),
                self.config.provider_batch_size,
            )
        )
        partial_recovered_groups = [
            group
            for group in provider_groups
            if recovered_slots.intersection(group) and not set(group).issubset(recovered_slots)
        ]
        if partial_recovered_groups:
            raise ValueError("recovery requires atomically committed provider batches")
        recovered_program_count = len(
            {
                entry.program_hash
                for entry in recovered_entries
                if entry.program is not None and entry.program_hash is not None
            }
        )
        if (
            len(auxiliary_programs or {})
            + recovered_program_count
            + min(
                len(snapshot.planned_slot_ids) - len(recovered_slots),
                self.config.provider_concurrency * self.config.provider_batch_size,
            )
            > self.config.candidate_queue_capacity
        ):
            raise ValueError("candidate queue cannot hold auxiliary and provider results")
        started_ns = time.monotonic_ns()
        events: list[TelemetryEvent] = []
        events_lock = threading.Lock()

        def emit(name: str, **fields: object) -> None:
            event = TelemetryEvent(name, time.monotonic_ns(), fields)
            with events_lock:
                events.append(event)
            if self.telemetry_sink is not None:
                self.telemetry_sink(event)

        slot_statuses = {slot_id: SlotStatus.PLANNED for slot_id in snapshot.planned_slot_ids}
        invalid_slots: dict[str, str] = {}
        aliases: dict[str, list[str]] = defaultdict(list)
        programs: dict[str, ProgramT] = {}
        results: dict[str, dict[str, tuple[ResultT, ...]]] = defaultdict(dict)
        pending_shards: dict[str, int] = {}
        candidate_queue: queue.Queue[tuple[str, ProgramT, bool]] = queue.Queue(
            self.config.candidate_queue_capacity
        )
        evaluation_queue: queue.Queue[tuple[EpisodeShard, ProgramT, bool]] = queue.Queue(
            self.config.evaluation_queue_capacity
        )
        provider_groups_pending = deque(
            (ordinal, group)
            for ordinal, group in enumerate(provider_groups)
            if not set(group).issubset(recovered_slots)
        )
        provider_futures: dict[Future[ProviderBatch[ProgramT]], ProviderCall] = {}
        provider_entry_queue: queue.Queue[tuple[ProviderCall, GeneratedEntry[ProgramT]]] = (
            queue.Queue(self.config.candidate_queue_capacity)
        )
        streamed_slots: dict[str, set[str]] = defaultdict(set)
        evaluator_futures: dict[Future[Sequence[ResultT]], tuple[EpisodeShard, ProgramT, bool]] = {}
        candidate_shard_sources: deque[tuple[Iterator[EpisodeShard], ProgramT, bool]] = deque()
        auxiliary_shard_sources: deque[tuple[Iterator[EpisodeShard], ProgramT, bool]] = deque()
        owns_provider = self._provider_executor is None
        owns_evaluator = self._evaluator_executor is None
        provider_executor = self._provider_executor or ThreadPoolExecutor(
            max_workers=self.config.provider_concurrency,
            thread_name_prefix="native-v3-provider",
        )
        evaluator_executor = self._evaluator_executor or ThreadPoolExecutor(
            max_workers=self.config.evaluator_workers,
            thread_name_prefix="native-v3-evaluator",
        )
        provider_paused_since: int | None = None
        candidate_seen = False
        first_evaluation_started = False
        first_candidate_evaluation_started = False
        first_candidate_validated_ns: int | None = None
        first_candidate_half_workers_ns: int | None = None
        first_candidate_all_workers_ns: int | None = None
        starvation_since: int | None = None
        for program_hash, program in sorted((auxiliary_programs or {}).items()):
            programs[program_hash] = program
            candidate_queue.put((program_hash, program, True))
        for entry in sorted(recovered_entries, key=lambda value: value.slot_id):
            if entry.error is not None:
                slot_statuses[entry.slot_id] = SlotStatus.INVALID
                invalid_slots[entry.slot_id] = entry.error
                continue
            assert entry.program is not None and entry.program_hash is not None
            slot_statuses[entry.slot_id] = SlotStatus.VALID
            aliases[entry.program_hash].append(entry.slot_id)
            if entry.program_hash not in programs:
                if first_candidate_validated_ns is None:
                    first_candidate_validated_ns = time.monotonic_ns()
                programs[entry.program_hash] = entry.program
                candidate_queue.put((entry.program_hash, entry.program, False))
            emit(
                "candidate_recovered",
                slot_id=entry.slot_id,
                program_hash=entry.program_hash,
                candidate_queue_depth=candidate_queue.qsize(),
            )

        def evaluation_backlog() -> int:
            return evaluation_queue.qsize() + len(evaluator_futures)

        def submit_provider_calls() -> None:
            nonlocal provider_paused_since
            while (
                provider_groups_pending and len(provider_futures) < self.config.provider_concurrency
            ):
                if evaluation_backlog() >= self.config.target_evaluation_backlog:
                    if provider_paused_since is None:
                        provider_paused_since = time.monotonic_ns()
                        emit(
                            "provider_backpressure_started",
                            evaluation_shard_queue_depth=evaluation_queue.qsize(),
                        )
                    return
                if provider_paused_since is not None:
                    emit(
                        "provider_backpressure_ended",
                        idle_ns=time.monotonic_ns() - provider_paused_since,
                    )
                    provider_paused_since = None
                call_ordinal, slot_ids = provider_groups_pending.popleft()
                call_id = f"{snapshot.epoch_id}:provider:{call_ordinal:04d}"
                call = ProviderCall(call_id, slot_ids, snapshot)
                for slot_id in slot_ids:
                    slot_statuses[slot_id] = SlotStatus.GENERATING
                if self.streaming_provider_call is None:
                    future = provider_executor.submit(self.provider_call, call)
                else:

                    def entry_sink(
                        entry: GeneratedEntry[ProgramT],
                        *,
                        frozen_call: ProviderCall = call,
                    ) -> None:
                        provider_entry_queue.put((frozen_call, entry), block=True)

                    future = provider_executor.submit(
                        self.streaming_provider_call,
                        call,
                        entry_sink,
                    )
                provider_futures[future] = call
                emit(
                    "provider_call_started",
                    call_id=call_id,
                    provider_calls_in_flight=len(provider_futures),
                    slot_count=len(slot_ids),
                )

        def consume_entry(
            call: ProviderCall,
            entry: GeneratedEntry[ProgramT],
        ) -> None:
            nonlocal first_candidate_validated_ns
            returned_slots = streamed_slots[call.call_id]
            if entry.slot_id not in call.slot_ids:
                return
            if entry.slot_id in returned_slots:
                slot_statuses[entry.slot_id] = SlotStatus.INVALID
                invalid_slots[entry.slot_id] = "duplicate slot in provider batch"
                return
            returned_slots.add(entry.slot_id)
            if entry.error is not None:
                slot_statuses[entry.slot_id] = SlotStatus.INVALID
                invalid_slots[entry.slot_id] = entry.error
                return
            assert entry.program is not None and entry.program_hash is not None
            slot_statuses[entry.slot_id] = SlotStatus.VALID
            aliases[entry.program_hash].append(entry.slot_id)
            if entry.program_hash not in programs:
                if first_candidate_validated_ns is None:
                    first_candidate_validated_ns = time.monotonic_ns()
                programs[entry.program_hash] = entry.program
                candidate_queue.put((entry.program_hash, entry.program, False))
                emit(
                    "candidate_validated",
                    slot_id=entry.slot_id,
                    program_hash=entry.program_hash,
                    candidate_queue_depth=candidate_queue.qsize(),
                )
            elif entry.program_hash in pending_shards:
                slot_statuses[entry.slot_id] = (
                    SlotStatus.COMPLETE
                    if pending_shards[entry.program_hash] == 0
                    else SlotStatus.EVALUATING
                )

        def consume_streamed_entries() -> bool:
            changed = False
            while True:
                try:
                    call, entry = provider_entry_queue.get_nowait()
                except queue.Empty:
                    return changed
                consume_entry(call, entry)
                changed = True

        def consume_provider_futures() -> bool:
            changed = False
            for future in tuple(provider_futures):
                if not future.done():
                    continue
                consume_streamed_entries()
                changed = True
                call = provider_futures.pop(future)
                latency_ns = time.monotonic_ns() - next(
                    event.monotonic_ns
                    for event in reversed(events)
                    if event.name == "provider_call_started"
                    and event.fields.get("call_id") == call.call_id
                )
                try:
                    batch = future.result()
                except BaseException as error:
                    for slot_id in call.slot_ids:
                        if slot_id in streamed_slots[call.call_id]:
                            continue
                        slot_statuses[slot_id] = SlotStatus.INVALID
                        invalid_slots[slot_id] = (
                            f"provider failure: {type(error).__name__}: {error}"
                        )
                    emit(
                        "provider_call_failed",
                        call_id=call.call_id,
                        latency_ns=latency_ns,
                        error_type=type(error).__name__,
                        error_message=str(error)[:1000],
                        provider_calls_in_flight=len(provider_futures),
                    )
                    continue
                if self.streaming_provider_call is None:
                    for entry in batch.entries:
                        consume_entry(call, entry)
                returned_slots = streamed_slots[call.call_id]
                valid_count = sum(entry.program is not None for entry in batch.entries)
                for slot_id in call.slot_ids:
                    if slot_id not in returned_slots:
                        slot_statuses[slot_id] = SlotStatus.INVALID
                        invalid_slots[slot_id] = "provider omitted planned slot"
                emit(
                    "provider_call_completed",
                    call_id=call.call_id,
                    latency_ns=latency_ns,
                    programs_returned=len(batch.entries),
                    valid_programs=valid_count,
                    response_bytes=batch.response_bytes,
                    provider_calls_in_flight=len(provider_futures),
                )
            return changed

        def expand_candidates() -> bool:
            nonlocal candidate_seen
            changed = False
            while not candidate_queue.empty():
                program_hash, program, auxiliary = candidate_queue.get_nowait()
                if not auxiliary:
                    candidate_seen = True
                shards = tuple(self.build_shards(program_hash, program))
                pending_shards[program_hash] = len(shards)
                if not auxiliary:
                    for alias in aliases[program_hash]:
                        slot_statuses[alias] = SlotStatus.EVALUATING
                if shards:
                    target = auxiliary_shard_sources if auxiliary else candidate_shard_sources
                    target.append((iter(shards), program, auxiliary))
                emit(
                    ("auxiliary_candidate_expanded" if auxiliary else "candidate_expanded"),
                    program_hash=program_hash,
                    shard_count=len(shards),
                    evaluation_shard_queue_depth=evaluation_queue.qsize(),
                )
                if not shards and not auxiliary:
                    for alias in aliases[program_hash]:
                        slot_statuses[alias] = SlotStatus.COMPLETE
                changed = True
            return changed

        def fill_evaluation_queue() -> bool:
            changed = False
            while candidate_shard_sources and not evaluation_queue.full():
                source, program, auxiliary = candidate_shard_sources.popleft()
                try:
                    shard = next(source)
                except StopIteration:
                    continue
                evaluation_queue.put_nowait((shard, program, auxiliary))
                candidate_shard_sources.append((source, program, auxiliary))
                changed = True
            auxiliary_limit = max(0, self.config.evaluator_workers - 1)
            while (
                auxiliary_shard_sources
                and not evaluation_queue.full()
                and not candidate_shard_sources
                and (
                    candidate_seen
                    or evaluation_queue.qsize() + len(evaluator_futures) < auxiliary_limit
                )
            ):
                source, program, auxiliary = auxiliary_shard_sources.popleft()
                try:
                    shard = next(source)
                except StopIteration:
                    continue
                evaluation_queue.put_nowait((shard, program, auxiliary))
                auxiliary_shard_sources.append((source, program, auxiliary))
                changed = True
            return changed

        def dispatch_evaluators() -> bool:
            nonlocal first_candidate_all_workers_ns
            nonlocal first_candidate_evaluation_started
            nonlocal first_candidate_half_workers_ns
            nonlocal first_evaluation_started
            changed = False
            while (
                not evaluation_queue.empty()
                and len(evaluator_futures) < self.config.evaluator_workers
            ):
                shard, program, auxiliary = evaluation_queue.get_nowait()
                future = evaluator_executor.submit(self.evaluate_shard, shard, program)
                evaluator_futures[future] = (shard, program, auxiliary)
                first_start_ns = None
                if not first_evaluation_started:
                    first_evaluation_started = True
                    first_start_ns = time.monotonic_ns() - started_ns
                first_candidate_start_ns = None
                if not auxiliary and not first_candidate_evaluation_started:
                    first_candidate_evaluation_started = True
                    if first_candidate_validated_ns is not None:
                        first_candidate_start_ns = (
                            time.monotonic_ns() - first_candidate_validated_ns
                        )
                active_candidate_workers = sum(
                    not in_flight_auxiliary
                    for _in_flight_shard, _in_flight_program, in_flight_auxiliary in (
                        evaluator_futures.values()
                    )
                )
                first_half_ns = None
                half_target = (self.config.evaluator_workers + 1) // 2
                if (
                    first_candidate_validated_ns is not None
                    and first_candidate_half_workers_ns is None
                    and active_candidate_workers >= half_target
                ):
                    first_candidate_half_workers_ns = (
                        time.monotonic_ns() - first_candidate_validated_ns
                    )
                    first_half_ns = first_candidate_half_workers_ns
                first_all_ns = None
                if (
                    first_candidate_validated_ns is not None
                    and first_candidate_all_workers_ns is None
                    and active_candidate_workers >= self.config.evaluator_workers
                ):
                    first_candidate_all_workers_ns = (
                        time.monotonic_ns() - first_candidate_validated_ns
                    )
                    first_all_ns = first_candidate_all_workers_ns
                emit(
                    "evaluation_shard_started",
                    shard_id=shard.shard_id,
                    program_hash=shard.program_hash,
                    auxiliary=auxiliary,
                    evaluator_tasks_in_flight=len(evaluator_futures),
                    evaluation_shard_queue_depth=evaluation_queue.qsize(),
                    time_to_first_evaluation_ns=first_start_ns,
                    first_valid_ast_to_first_worker_ns=first_candidate_start_ns,
                    first_valid_ast_to_50_percent_workers_ns=first_half_ns,
                    first_valid_ast_to_all_workers_ns=first_all_ns,
                )
                changed = True
            return changed

        def consume_evaluator_futures() -> bool:
            changed = False
            for future in tuple(evaluator_futures):
                if not future.done():
                    continue
                changed = True
                shard, _program, auxiliary = evaluator_futures.pop(future)
                try:
                    shard_results = tuple(future.result())
                except BaseException as error:
                    if isinstance(error, RetryableShardFailure) and shard.residual_of is None:
                        residuals = split_residual_shard(shard)
                        pending_shards[shard.program_hash] = max(
                            0, pending_shards[shard.program_hash] - 1
                        ) + len(residuals)
                        target = auxiliary_shard_sources if auxiliary else candidate_shard_sources
                        target.appendleft((iter(residuals), _program, auxiliary))
                        emit(
                            "evaluation_shard_rescheduled",
                            shard_id=shard.shard_id,
                            program_hash=shard.program_hash,
                            residual_count=len(residuals),
                        )
                        continue
                    if not isinstance(error, RetryableShardFailure):
                        raise
                    for alias in aliases.get(shard.program_hash, ()):
                        slot_statuses[alias] = SlotStatus.FAILED
                        invalid_slots[alias] = (
                            f"evaluation failure: {type(error).__name__}: {error}"
                        )
                    pending_shards[shard.program_hash] = 0
                    emit(
                        "evaluation_shard_failed",
                        shard_id=shard.shard_id,
                        program_hash=shard.program_hash,
                    )
                    continue
                if self.result_sink is not None:
                    self.result_sink(shard, shard_results)
                results[shard.program_hash][shard.shard_id] = shard_results
                remaining = max(0, pending_shards[shard.program_hash] - 1)
                pending_shards[shard.program_hash] = remaining
                if remaining == 0:
                    for alias in aliases.get(shard.program_hash, ()):
                        if slot_statuses[alias] is SlotStatus.EVALUATING:
                            slot_statuses[alias] = SlotStatus.COMPLETE
                emit(
                    "evaluation_shard_completed",
                    shard_id=shard.shard_id,
                    program_hash=shard.program_hash,
                    result_count=len(shard_results),
                    evaluator_tasks_in_flight=len(evaluator_futures),
                )
            return changed

        try:
            submit_provider_calls()
            while True:
                changed = consume_streamed_entries()
                changed = consume_provider_futures() or changed
                changed = expand_candidates() or changed
                changed = consume_evaluator_futures() or changed
                changed = fill_evaluation_queue() or changed
                changed = dispatch_evaluators() or changed
                submit_provider_calls()
                all_slots_terminal = all(
                    status
                    in {
                        SlotStatus.INVALID,
                        SlotStatus.COMPLETE,
                        SlotStatus.FAILED,
                    }
                    for status in slot_statuses.values()
                )
                if (
                    all_slots_terminal
                    and not provider_futures
                    and provider_entry_queue.empty()
                    and candidate_queue.empty()
                    and not candidate_shard_sources
                    and not auxiliary_shard_sources
                    and evaluation_queue.empty()
                    and not evaluator_futures
                ):
                    break
                if not changed:
                    # Provider starvation includes the interval before the
                    # first valid AST when all bounded auxiliary work has
                    # drained. With no useful CPU work queued, the provider is
                    # the actual bottleneck and the dashboard must say so.
                    starved = bool(
                        not evaluator_futures
                        and evaluation_queue.empty()
                        and not candidate_shard_sources
                        and not auxiliary_shard_sources
                        and provider_futures
                    )
                    if starved and starvation_since is None:
                        starvation_since = time.monotonic_ns()
                        emit("cpu_idle_provider_starvation_started")
                    elif not starved and starvation_since is not None:
                        emit(
                            "cpu_idle_provider_starvation_ended",
                            idle_ns=time.monotonic_ns() - starvation_since,
                        )
                        starvation_since = None
                    time.sleep(0.001)
                elif starvation_since is not None:
                    emit(
                        "cpu_idle_provider_starvation_ended",
                        idle_ns=time.monotonic_ns() - starvation_since,
                    )
                    starvation_since = None
        finally:
            if owns_provider:
                provider_executor.shutdown(wait=True, cancel_futures=True)
            if owns_evaluator:
                evaluator_executor.shutdown(wait=True, cancel_futures=True)

        unique_valid = sum(
            bool(program_aliases)
            and all(slot_statuses[slot] is SlotStatus.COMPLETE for slot in program_aliases)
            for program_aliases in aliases.values()
        )
        if unique_valid >= 8:
            epoch_status = EpochStatus.COMPLETE
        elif unique_valid >= 4:
            epoch_status = EpochStatus.DEGRADED
        else:
            epoch_status = EpochStatus.INCONCLUSIVE
        emit(
            "epoch_terminal",
            status=epoch_status.value,
            unique_valid_programs=unique_valid,
        )
        return StreamingEpochResult(
            epoch_status=epoch_status,
            slot_statuses=dict(sorted(slot_statuses.items())),
            program_results={
                key: tuple(result for shard_id in sorted(values) for result in values[shard_id])
                for key, values in sorted(results.items())
                if pending_shards.get(key) == 0
                and (
                    key not in aliases
                    or all(slot_statuses[slot] is SlotStatus.COMPLETE for slot in aliases[key])
                )
            },
            program_aliases={key: tuple(sorted(values)) for key, values in sorted(aliases.items())},
            invalid_slots=dict(sorted(invalid_slots.items())),
            telemetry=tuple(events),
        )
