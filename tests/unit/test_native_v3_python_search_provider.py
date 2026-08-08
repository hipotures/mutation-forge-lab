from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.native_v3_python.search import M5InfrastructureError
from mutation_forge.native_v3_python.search_provider import (
    M5_PROVIDER_MAX_CAMPAIGNS,
    M5_PROVIDER_MAX_TURNS,
    M5_PROVIDER_STDOUT_BYTES,
    M5_PROVIDER_TRANSCRIPT_BYTES,
    CodexM5SearchProvider,
    specification_ack_schema,
)
from mutation_forge.stage3.app_server import (
    ForkResult,
    GenerationResult,
    ModelProfile,
    TokenUsage,
)
from mutation_forge.stage3.artifacts import TransportLogger


def test_m5_provider_transport_budget_is_explicit_and_bounded() -> None:
    assert M5_PROVIDER_MAX_TURNS == 40
    assert M5_PROVIDER_MAX_CAMPAIGNS == 40
    assert M5_PROVIDER_TRANSCRIPT_BYTES == 16 * 1024 * 1024
    assert M5_PROVIDER_STDOUT_BYTES == 16 * 1024 * 1024


def _source(label: str) -> str:
    return (
        "def propose(ctx, graph, api, seed):\n"
        f"    return api.no_plan(\"{label}\")\n"
    )


class _Adapter:
    def __init__(self) -> None:
        self.logger: TransportLogger | None = None
        self.current_thread = "anchor-thread"
        self.histories: dict[str, tuple[str, ...]] = {"anchor-thread": ()}
        self.forks: list[tuple[str, str, tuple[str, ...]]] = []
        self.turn_count = 0
        self.closed = False

    def rotate_logger(
        self,
        artifact_dir: Path,
        prefix: str,
        *,
        compress_json: bool,
    ) -> None:
        self.logger = TransportLogger(
            artifact_dir,
            prefix,
            compress_json=compress_json,
        )

    def inspect_metadata(self) -> dict[str, int]:
        return {"serverRetries": 0, "serverWarnings": 0}

    def generate_persistent(
        self,
        prompt: str,
        profile: ModelProfile,
        *,
        output_schema: Mapping[str, Any],
    ) -> GenerationResult:
        del prompt, profile, output_schema
        self.turn_count += 1
        turn_id = f"turn-{self.turn_count:02d}"
        history = self.histories[self.current_thread] + (turn_id,)
        self.histories[self.current_thread] = history
        response: dict[str, str]
        if self.turn_count == 1:
            response = {
                "schema_version": "mforge.native.python_m5_specification_ack.v1",
                "ack": "specification-retained",
            }
        else:
            response = {
                "schema_version": "mforge.native.python_policy_response.v1",
                "source": _source(f"policy-{self.turn_count}"),
            }
        return GenerationResult(
            text=json.dumps(response, separators=(",", ":")),
            usage=TokenUsage(
                input_tokens=1,
                cached_input_tokens=0,
                cache_write_input_tokens=0,
                output_tokens=1,
                reasoning_output_tokens=0,
                total_tokens=2,
                final=True,
                partial=False,
                raw={"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            ),
            thread_id=self.current_thread,
            session_id="session",
            turn_id=turn_id,
            request_id=self.turn_count,
            thread_path=f"/opaque/{self.current_thread}",
            duration_ms=1,
        )

    def fork_persistent_thread(
        self,
        profile: ModelProfile,
        *,
        last_turn_id: str,
        activate: bool,
    ) -> ForkResult:
        del profile
        source_thread = self.current_thread
        history = self.histories[source_thread]
        inclusive = history[: history.index(last_turn_id) + 1]
        child = f"fork-{len(self.forks):02d}"
        self.histories[child] = inclusive
        self.forks.append((source_thread, last_turn_id, inclusive))
        if activate:
            self.current_thread = child
        return ForkResult(
            source_thread_id=source_thread,
            child_thread_id=child,
            session_id="session",
            thread_path=f"/opaque/{child}",
            last_turn_id=last_turn_id,
            included_turn_ids=inclusive,
        )

    def activate_forked_thread(
        self,
        thread_id: str,
        *,
        completed_turn_ids: tuple[str, ...] | None = None,
    ) -> None:
        assert thread_id in self.histories
        if completed_turn_ids is not None:
            assert self.histories[thread_id] == completed_turn_ids
        self.current_thread = thread_id

    def close(self) -> None:
        self.closed = True


def test_provider_forks_roots_at_anchor_and_child_at_exact_parent(
    tmp_path: Path,
) -> None:
    adapter = _Adapter()
    provider = CodexM5SearchProvider(
        workspace=tmp_path / "runtime",
        model="fixture-model",
        effort="high",
        base_instructions="Return only the structured response.",
        adapter=cast(Any, adapter),
    )
    anchor_result = provider.ensure_specification_anchor(
        prompt="retain specification",
        system_prompt="system",
        output_schema=specification_ack_schema(),
        artifact_dir=tmp_path / "anchor",
    )
    anchor = anchor_result.context
    root_zero = provider.generate_root(
        anchor=anchor,
        generation=0,
        slot="slot-00",
        prompt="root zero",
        system_prompt="system",
        output_schema={"type": "object"},
        idempotency_key="root-zero",
        artifact_dir=tmp_path / "root-zero",
    )
    fork_count = len(adapter.forks)
    turn_count = adapter.turn_count
    assert (
        provider.generate_root(
            anchor=anchor,
            generation=0,
            slot="slot-00",
            prompt="root zero",
            system_prompt="system",
            output_schema={"type": "object"},
            idempotency_key="root-zero",
            artifact_dir=tmp_path / "root-zero",
        )
        == root_zero
    )
    assert len(adapter.forks) == fork_count
    assert adapter.turn_count == turn_count
    with pytest.raises(M5InfrastructureError, match="provider contract changed"):
        provider.generate_root(
            anchor=anchor,
            generation=0,
            slot="slot-00",
            prompt="changed root request",
            system_prompt="system",
            output_schema={"type": "object"},
            idempotency_key="root-zero",
            artifact_dir=tmp_path / "root-zero",
        )
    root_one = provider.generate_root(
        anchor=anchor,
        generation=0,
        slot="slot-01",
        prompt="root one",
        system_prompt="system",
        output_schema={"type": "object"},
        idempotency_key="root-one",
        artifact_dir=tmp_path / "root-one",
    )
    child = provider.generate_child(
        parent=root_zero.context,
        generation=1,
        slot="slot-00",
        prompt="child",
        system_prompt="system",
        output_schema={"type": "object"},
        idempotency_key="child-zero",
        artifact_dir=tmp_path / "child-zero",
    )
    assert root_zero.context.included_turn_ids == (
        anchor.turn_id,
        root_zero.context.turn_id,
    )
    assert root_one.context.included_turn_ids == (
        anchor.turn_id,
        root_one.context.turn_id,
    )
    assert child.context.included_turn_ids == (
        anchor.turn_id,
        root_zero.context.turn_id,
        child.context.turn_id,
    )
    assert root_one.context.turn_id not in child.context.included_turn_ids
    assert [item[1] for item in adapter.forks] == [
        anchor.turn_id,
        anchor.turn_id,
        root_zero.context.turn_id,
    ]
    assert [item[0] for item in adapter.forks[:2]] == [
        anchor.thread_id,
        anchor.thread_id,
    ]
    provider.close()
    assert adapter.closed is True


def test_retained_fork_mismatch_fails_closed(tmp_path: Path) -> None:
    adapter = _Adapter()
    provider = CodexM5SearchProvider(
        workspace=tmp_path / "runtime",
        model="fixture-model",
        effort="high",
        base_instructions="Return only the structured response.",
        adapter=cast(Any, adapter),
    )
    anchor = provider.ensure_specification_anchor(
        prompt="retain specification",
        system_prompt="system",
        output_schema=specification_ack_schema(),
        artifact_dir=tmp_path / "anchor",
    ).context
    provider.generate_root(
        anchor=anchor,
        generation=0,
        slot="slot-00",
        prompt="root zero",
        system_prompt="system",
        output_schema={"type": "object"},
        idempotency_key="root-zero",
        artifact_dir=tmp_path / "root-zero",
    )
    fork_path = tmp_path / "root-zero" / "fork" / "m5-fork-result.json.gz"
    retained = read_json(fork_path)
    assert isinstance(retained, dict)
    retained["last_turn_id"] = "wrong-turn"
    write_json(fork_path, retained)
    with pytest.raises(M5InfrastructureError, match="inclusive turn boundary"):
        provider.generate_root(
            anchor=anchor,
            generation=0,
            slot="slot-00",
            prompt="root zero",
            system_prompt="system",
            output_schema={"type": "object"},
            idempotency_key="root-zero",
            artifact_dir=tmp_path / "root-zero",
        )
