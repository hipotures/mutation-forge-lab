from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.native_v3.compaction_experiment import (
    build_reference_manifest,
)
from mutation_forge.native_v3.lineage_experiment import (
    ACK_SCHEMA_VERSION,
    _child_prompt,
    _child_repair_prompt,
    _feedback_prompt,
    _fixture_prompt,
    _fresh_prompt,
    _memory_prompt,
    build_search_memory,
    run_lineage_experiment,
)
from mutation_forge.native_v3.persistent_experiment import (
    BOOTSTRAP_ACK_SCHEMA_VERSION,
    BOOTSTRAP_ACK_VALUE,
    assert_model_facing_payload,
)
from mutation_forge.native_v3.search_memory import (
    MAX_PATTERNS_PER_OUTCOME,
    DuplicateCandidateError,
    PatternSummary,
    SearchMemoryError,
    SearchMemoryV1,
    reject_duplicate,
)
from mutation_forge.native_v3.single_program_contract import (
    validate_single_program_response,
)
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    IsolationError,
    ModelProfile,
    ProtocolError,
)

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location("step12d_fake_app_server", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario

FORBIDDEN_LENGTHS = (4, 8, 16)
PROFILE = ModelProfile("codex", "gpt-5.6-luna", "high")


def _candidate_responses() -> dict[str, dict[str, Any]]:
    value = json.loads(
        Path("tests/fixtures/native_v3_single_program_responses.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def _adapter(
    scenario: Any,
    *,
    process_holder: list[Any] | None = None,
    max_campaigns: int = 3,
) -> CodexAppServerAdapter:
    def factory(*_args: Any, **kwargs: Any) -> Any:
        process = FakeProcess(scenario, **kwargs)
        if process_holder is not None:
            process_holder.append(process)
        return process

    return CodexAppServerAdapter(
        process_factory=factory,
        auth_checker=lambda _capsule: True,
        limits=AppServerLimits(
            max_turns=8,
            max_campaigns=max_campaigns,
            turn_timeout=0.1,
            usage_grace=0.01,
        ),
        base_instructions="Return only the requested structured response.",
        compress_json_artifacts=True,
        sandbox_mode="read-only",
    )


def _novel_response(
    source: dict[str, Any],
    *,
    extra_fallbacks: int,
) -> dict[str, Any]:
    result = copy.deepcopy(source)
    entry = result["program"]["entry"]
    for _ in range(extra_fallbacks):
        entry = {
            "op": "try",
            "branches": [
                entry,
                {"op": "no_plan", "reason": "NO_MATCH"},
            ],
        }
    result["program"]["entry"] = entry
    return result


def _unterminated_response(source: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(source)
    block = result["program"]["entry"]["branches"][0]["body"]["body"]
    block["children"] = block["children"][:1]
    return result


def test_thread_fork_is_inclusive_and_excludes_later_turns() -> None:
    processes: list[Any] = []
    adapter = _adapter(FakeScenario(), process_holder=processes)
    try:
        first = adapter.generate_persistent("first", PROFILE)
        second = adapter.generate_persistent("second", PROFILE)
        fork = adapter.fork_persistent_thread(
            PROFILE,
            last_turn_id=first.turn_id,
        )
    finally:
        adapter.close()

    assert fork.source_thread_id == "thread-1"
    assert fork.child_thread_id == "fork-thread-1"
    assert fork.included_turn_ids == (first.turn_id,)
    assert second.turn_id not in fork.included_turn_ids
    request = next(
        item
        for item in processes[0].received_requests
        if item.get("method") == "thread/fork"
    )
    assert request["params"]["threadId"] == "thread-1"
    assert request["params"]["lastTurnId"] == first.turn_id
    assert request["params"]["excludeTurns"] is False


def test_second_fork_accepts_strictly_correlated_late_first_fork_notifications(
) -> None:
    adapter = _adapter(FakeScenario(fork_late_notifications=True))
    try:
        anchor = adapter.generate_persistent("anchor", PROFILE)
        parent = adapter.generate_persistent("parent", PROFILE)
        sibling = adapter.generate_persistent("sibling", PROFILE)
        child = adapter.fork_persistent_thread(
            PROFILE,
            last_turn_id=parent.turn_id,
        )
        fresh = adapter.fork_persistent_thread(
            PROFILE,
            last_turn_id=anchor.turn_id,
        )
        adapter.activate_forked_thread(child.child_thread_id)
        child_followup = adapter.generate_persistent("child followup", PROFILE)
    finally:
        adapter.close()

    assert child.included_turn_ids == (anchor.turn_id, parent.turn_id)
    assert sibling.turn_id not in child.included_turn_ids
    assert fresh.included_turn_ids == (anchor.turn_id,)
    assert child_followup.thread_id == child.child_thread_id


def test_thread_fork_rejects_invalid_and_in_progress_boundaries() -> None:
    invalid = _adapter(FakeScenario())
    try:
        invalid.generate_persistent("first", PROFILE)
        with pytest.raises(ProtocolError, match="request thread/fork failed"):
            invalid.fork_persistent_thread(
                PROFILE,
                last_turn_id="missing-turn",
            )
    finally:
        invalid.close()

    in_progress = _adapter(
        FakeScenario(fork_in_progress_last_turn_ids=["turn-1"])
    )
    try:
        turn = in_progress.generate_persistent("first", PROFILE)
        with pytest.raises(ProtocolError, match="request thread/fork failed"):
            in_progress.fork_persistent_thread(
                PROFILE,
                last_turn_id=turn.turn_id,
            )
    finally:
        in_progress.close()


def test_thread_fork_rejects_foreign_source_and_requires_durable_idle_thread() -> None:
    foreign = _adapter(FakeScenario(fork_foreign_source=True))
    try:
        turn = foreign.generate_persistent("first", PROFILE)
        with pytest.raises(ProtocolError, match="invalid child identity"):
            foreign.fork_persistent_thread(PROFILE, last_turn_id=turn.turn_id)
    finally:
        foreign.close()

    ephemeral = _adapter(FakeScenario())
    try:
        turn = ephemeral.generate_ephemeral_experiment("first", PROFILE)
        with pytest.raises(IsolationError, match="idle durable thread"):
            ephemeral.fork_persistent_thread(PROFILE, last_turn_id=turn.turn_id)
    finally:
        ephemeral.close()


def test_search_memory_is_canonical_bounded_and_contains_no_ast() -> None:
    fixtures = _candidate_responses()
    first = build_search_memory(fixtures, forbidden_lengths=FORBIDDEN_LENGTHS)
    second = SearchMemoryV1(
        protocol_hash=first.protocol_hash,
        seen_program_hashes=tuple(reversed(first.seen_program_hashes)),
        seen_behavior_signatures=tuple(
            reversed(first.seen_behavior_signatures)
        ),
        successful_patterns=tuple(reversed(first.successful_patterns)),
        failed_patterns=tuple(reversed(first.failed_patterns)),
        active_lineages=tuple(reversed(first.active_lineages)),
        validated_archive_ids=tuple(reversed(first.validated_archive_ids)),
        active_parent=first.active_parent,
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.sha256 == second.sha256
    encoded = first.canonical_bytes()
    assert b"canonical_ast" not in encoded
    assert b'"entry"' not in encoded
    assert len(first.successful_patterns) <= MAX_PATTERNS_PER_OUTCOME
    assert len(first.failed_patterns) <= MAX_PATTERNS_PER_OUTCOME
    host_memory = json.dumps(first.as_dict(), sort_keys=True)
    model_memory = json.dumps(first.model_facing_dict(), sort_keys=True)
    assert first.protocol_hash in host_memory
    assert first.seen_program_hashes[0] in host_memory
    assert first.seen_behavior_signatures[0] in host_memory
    assert first.protocol_hash not in model_memory
    assert first.seen_program_hashes[0] not in model_memory
    assert first.seen_behavior_signatures[0] not in model_memory
    assert '"selectors":' in model_memory
    assert '"actions":' in model_memory
    assert '"control_flow":' in model_memory

    reference = build_reference_manifest(
        fixtures,
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    parent = reference["candidates"][-1]
    prompts = (
        _fixture_prompt(parent, role="active-parent"),
        _feedback_prompt(parent),
        _child_prompt(parent),
        _child_repair_prompt(parent, "missing terminal"),
        _memory_prompt(first),
        _fresh_prompt(),
    )
    for prompt in prompts:
        assert_model_facing_payload(
            prompt=prompt,
            system_prompt="Return only the requested structured response.",
            schema={"type": "object"},
        )

    sample = first.successful_patterns[0]
    too_many = tuple(
        PatternSummary(
            pattern_id=f"extra-{index:02d}",
            selector_families=sample.selector_families,
            action_families=sample.action_families,
            control_flow=sample.control_flow,
            description=sample.description,
            description_source=sample.description_source,
            evaluation_outcome=sample.evaluation_outcome,
            evidence_kind=sample.evidence_kind,
            main_evidence=sample.main_evidence,
        )
        for index in range(MAX_PATTERNS_PER_OUTCOME + 1)
    )
    with pytest.raises(SearchMemoryError, match="successful_patterns exceeds"):
        SearchMemoryV1(
            protocol_hash=first.protocol_hash,
            seen_program_hashes=(),
            seen_behavior_signatures=(),
            successful_patterns=too_many,
            failed_patterns=(),
            active_lineages=(),
            validated_archive_ids=(),
        )


def test_host_duplicate_gate_rejects_hash_and_behavior_signature() -> None:
    memory = build_search_memory(
        _candidate_responses(),
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    with pytest.raises(DuplicateCandidateError, match="program hash"):
        reject_duplicate(
            memory,
            program_hash=memory.seen_program_hashes[0],
            behavior_signature="0" * 64,
        )
    with pytest.raises(DuplicateCandidateError, match="behavior signature"):
        reject_duplicate(
            memory,
            program_hash="0" * 64,
            behavior_signature=memory.seen_behavior_signatures[0],
        )


def test_fake_lineage_experiment_proves_both_boundaries_and_artifacts(
    tmp_path: Path,
) -> None:
    fixtures = _candidate_responses()
    child = _novel_response(fixtures["add-edge"], extra_fallbacks=1)
    fresh = _novel_response(fixtures["remove-edge"], extra_fallbacks=2)
    invalid_child = _unterminated_response(fixtures["add-edge"])
    validate_single_program_response(
        json.dumps(child, separators=(",", ":")),
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    validate_single_program_response(
        json.dumps(fresh, separators=(",", ":")),
        forbidden_lengths=FORBIDDEN_LENGTHS,
    )
    scenario = FakeScenario(
        fork_thread_ids=["child-thread", "fresh-thread"],
        final_texts=[
            json.dumps(
                {
                    "schema_version": BOOTSTRAP_ACK_SCHEMA_VERSION,
                    "ack": BOOTSTRAP_ACK_VALUE,
                },
                separators=(",", ":"),
            ),
            json.dumps(
                {"schema_version": ACK_SCHEMA_VERSION, "ack": "root-parent"},
                separators=(",", ":"),
            ),
            json.dumps(
                {"schema_version": ACK_SCHEMA_VERSION, "ack": "later-sibling"},
                separators=(",", ":"),
            ),
            json.dumps(
                {"schema_version": ACK_SCHEMA_VERSION, "ack": "child-feedback"},
                separators=(",", ":"),
            ),
            json.dumps(invalid_child, separators=(",", ":")),
            json.dumps(child, separators=(",", ":")),
            json.dumps(
                {"schema_version": ACK_SCHEMA_VERSION, "ack": "search-memory"},
                separators=(",", ":"),
            ),
            json.dumps(fresh, separators=(",", ":")),
        ],
    )

    workspace = tmp_path / "experiment"
    report = run_lineage_experiment(
        workspace,
        model="gpt-5.6-luna",
        effort="high",
        forbidden_lengths=FORBIDDEN_LENGTHS,
        candidate_responses=fixtures,
        adapter_factory=lambda base: _adapter(scenario),
    )

    assert report["status"] == "completed"
    assert report["compaction_used"] is False
    assert report["boundary_proofs"]["child_exact"] is True
    assert report["boundary_proofs"]["fresh_exact"] is True
    assert report["lineage_child"]["parent_id"] == "g0003-s04"
    assert report["lineage_child"]["fork_thread_id"] == "child-thread"
    assert report["fresh_root"]["parent_id"] is None
    assert report["fresh_root"]["fork_thread_id"] == "fresh-thread"
    assert report["fresh_root_diverse"] is True
    assert report["duplicate_rejections"] == 0
    assert report["provider_turns"] == 8
    assert report["fork_rpc_count"] == 2
    assert report["child_repair_used"] is True
    assert len(report["child_attempts"]) == 2
    assert (workspace / "lineage-report.json.gz").is_file()
    assert (workspace / "lineage-report.md").is_file()
    assert (workspace / "child-program.json.gz").is_file()
    assert (workspace / "fresh-root-program.json.gz").is_file()

    memory_prompt = (
        workspace / "provider-turns/07-search-memory.request.md"
    ).read_text(encoding="utf-8")
    json.loads(memory_prompt)
    assert "canonical_program" not in memory_prompt
    assert '"entry"' not in memory_prompt
    json.loads(
        (workspace / "provider-turns/00-spec-anchor.request.md").read_text(
            encoding="utf-8"
        )
    )

    expected_suffixes = {
        "codex-profile.json.gz",
        "codex-rpc.jsonl",
        "events.jsonl",
        "output-schema.json.gz",
        "provider-raw.json.gz",
        "request.json.gz",
        "request.md",
        "response.json.gz",
        "response.md",
        "response.raw.txt",
        "stderr.txt",
        "stdout.jsonl",
        "system-prompt.md",
        "transcript.sha256",
        "usage.json.gz",
        "wire.jsonl",
    }
    turns_dir = workspace / "provider-turns"
    for prefix in (
        "00-spec-anchor",
        "01-root-parent",
        "02-later-sibling",
        "03-child-fork",
        "04-fresh-fork",
        "05-child-feedback",
        "06-child-mutation",
        "06-child-repair",
        "07-search-memory",
        "08-fresh-root",
    ):
        suffixes = {
            path.name.removeprefix(f"{prefix}.")
            for path in turns_dir.iterdir()
            if path.name.startswith(f"{prefix}.")
        }
        assert suffixes == expected_suffixes
