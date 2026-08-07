from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)
from mutation_forge.native_v3.experiment import (
    V3Config,
    V3WorkspaceError,
    run_v3,
    v3_status,
)
from mutation_forge.native_v3.scoring import (
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EvidenceStatus,
    ScoreEvidence,
)

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location(
    "native_v3_route_fake",
    _FIXTURE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURE
_SPEC.loader.exec_module(_FIXTURE)
FakeProcess = _FIXTURE.FakeProcess
FakeScenario = _FIXTURE.FakeScenario
_SCORE_IDENTITY = BackendIdentity(
    backend_id="v3-recorded-backend",
    heg_commit="fixture",
    source_tree_sha256="a" * 64,
    binary_sha256="b" * 64,
    compiler_identity="fixture",
    build_flags=(),
    platform="fixture",
    architecture="fixture",
)


def _source(index: int) -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.program.v3",
            "entry": {
                "op": "if",
                "condition": {
                    "op": "less",
                    "left": {"op": "feature", "field": "order"},
                    "right": 30 + index,
                },
                "then": {"op": "no_plan", "reason": "NO_MATCH"},
                "else": {"op": "no_plan", "reason": "EXPLICIT"},
            },
        },
        separators=(",", ":"),
    )


def _response(
    slot_ids: tuple[str, ...],
    *,
    offset: int = 0,
    sources: dict[str, str] | None = None,
) -> str:
    batch = json.dumps(
        {
            "schema_version": "mforge.native.program_batch.v3",
            "programs": [
                {
                    "slot_id": slot_id,
                    "program_json_raw": (
                        sources[slot_id]
                        if sources is not None and slot_id in sources
                        else _source(offset + index)
                    ),
                    "design_summary": f"Bounded mechanism {offset + index}.",
                }
                for index, slot_id in enumerate(slot_ids)
            ],
        },
        separators=(",", ":"),
    )
    return json.dumps(
        {
            "schema_version": "mforge.native.generated_policy.v1",
            "source": batch,
            "design_summary": "Return four bounded mutation programs.",
            "hypothesis": "The cohort reaches authoritative serial evaluation.",
            "used_fields": [],
            "assumptions": [],
            "expected_failure_modes": [],
        },
        separators=(",", ":"),
    )


def _config(tmp_path: Path, *, extra: str = "") -> Path:
    heg_repo = tmp_path / "heg"
    (heg_repo / "src" / "sglab").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "experiment.toml"
    path.write_text(
        f'''schema_version = "mforge.experiment.v3"
protocol = "v3"
exp_id = "v3-run"
workspace = "{(tmp_path / "workspace").as_posix()}"
{extra}

[v3]
model = "gpt-5.6-luna"
effort = "high"
timeout_seconds = 30
heg_repo = "{heg_repo.as_posix()}"
communication_mode = "multi_program_batch"
''',
        encoding="utf-8",
    )
    return path


def _provider_with_responses(
    responses: list[str],
) -> LocalCodexAppServerProvider:
    process_index = 0

    def process_factory(*_: Any, **kwargs: Any) -> Any:
        nonlocal process_index
        response = responses[process_index]
        process_index += 1
        return FakeProcess(FakeScenario(final_text=response), **kwargs)

    return LocalCodexAppServerProvider(
        model="gpt-5.6-luna",
        effort="high",
        concurrency=1,
        max_repairs=1,
        turn_timeout_base_seconds=1,
        process_factory=process_factory,
        auth_checker=lambda _: True,
        persist_artifacts=False,
    )


def _provider(_: V3Config) -> LocalCodexAppServerProvider:
    return _provider_with_responses(
        [
            _response(("slot-00", "slot-01", "slot-02", "slot-03"), offset=0),
            _response(("slot-04", "slot-05", "slot-06", "slot-07"), offset=4),
        ]
    )


class _Backend:
    backend_id = "v3-recorded-backend"
    score_implementation = "v3-recorded-scorer"

    def __init__(self) -> None:
        self.raw_graph_score_calls = 0
        self.unique_graph_scores = 0

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        return (4,)

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        del seed
        edges = {tuple(sorted((vertex, (vertex + 1) % order))) for vertex in range(order)}
        edges.update((vertex, vertex + order // 2) for vertex in range(order // 2))
        return GraphState(order, tuple(sorted(edges)))

    def validate(self, graph: GraphState) -> GraphValidation:
        return GraphValidation(True)

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: Any = None,
    ) -> GraphScore:
        return GraphScore(True, ((4, 1),), 1, 1, True, (1, 1))

    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: object = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        del forbidden_lengths
        self.raw_graph_score_calls += 1
        self.unique_graph_scores += 1
        return ScoreEvidence(
            self.state_hash(graph),
            graph.order,
            len(graph.edges),
            witness_cap,
            (
                CycleComponentEvidence(
                    4,
                    1,
                    1,
                    1,
                    EvidenceStatus.EXACT,
                    50_000,
                    1,
                    0,
                    attempt_kind,
                    _SCORE_IDENTITY,
                ),
            ),
        )

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        raise AssertionError("nonzero score must not be verified")

    def canonical_hash(self, graph: GraphState) -> str:
        return self.state_hash(graph)

    def state_hash(self, graph: GraphState) -> str:
        return hashlib.sha256(repr(graph).encode()).hexdigest()

    def serialize_graph6(self, graph: GraphState) -> str:
        raise AssertionError("nonzero score must not be serialized")

    def deserialize_graph6(self, value: str) -> GraphState:
        raise AssertionError("not used")

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: Any = None,
    ) -> GraphState:
        raise AssertionError("no-plan program must not rewrite")

    def propose_rewrite(self, graph: GraphState, **_: Any) -> RewritePlan:
        raise AssertionError("v3 evaluator must not request backend proposals")

    def close(self) -> None:
        return None


def _backend(_: V3Config) -> _Backend:
    return _Backend()


def test_v3_routes_selected_preview_without_constructing_batch_provider(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'communication_mode = "multi_program_batch"',
            (
                'communication_mode = "persistent_single_ast"\n'
                'output_contract = "slot_specific"'
            ),
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def provider(_: V3Config) -> LocalCodexAppServerProvider:
        raise AssertionError("selected preview must not construct the batch provider")

    def preview_runner(
        experiment_root: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append({"experiment_root": experiment_root, **kwargs})
        return {
            "status": "completed",
            "cohort_outcome": "COMPLETE",
            "valid_ast": True,
            "valid_slots": 8,
            "unique_valid_programs": 8,
            "duplicate_aliases": 0,
            "model_turns": 9,
            "graph_evaluations": 8,
            "scientific_terminal_result": True,
            "selected_program_hash": "a" * 64,
            "usage": {},
            "epoch_manifest": str(tmp_path / "epoch-manifest.json.gz"),
            "cohort_report": str(tmp_path / "cohort-report.json.gz"),
        }

    result = run_v3(
        path,
        provider_factory=provider,
        backend_factory=_backend,
        auth_available=lambda _: True,
        preview_runner=preview_runner,
    )

    assert result["state"] == "completed"
    assert result["communication_mode"] == "persistent_single_ast"
    assert result["provider_mode"] == "persistent_single_ast"
    assert result["output_contract"] == "slot_specific"
    assert result["output_schema_sha256"]
    assert result["provider_turns"] == 9
    assert len(calls) == 1
    assert calls[0]["experiment_root"] == tmp_path / "workspace" / "v3-run"
    assert calls[0]["output_contract"] == "slot_specific"


def test_preview_provider_failure_preserves_attempt_and_retry_progress(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'communication_mode = "multi_program_batch"',
            (
                'communication_mode = "persistent_single_ast"\n'
                'output_contract = "slot_specific"'
            ),
        ),
        encoding="utf-8",
    )

    def preview_runner(
        experiment_root: Path,
        **_: object,
    ) -> dict[str, object]:
        state_path = (
            experiment_root / "native-v3-output" / "epoch-0000" / "communication-state.json.gz"
        )
        state_path.parent.mkdir(parents=True)
        write_json(
            state_path,
            {
                "model_turns": 2,
                "provider_attempts": 3,
                "failed_provider_attempts": 1,
                "provider_retries": 7,
                "provider_warnings": 1,
                "provider_process_restarts": 1,
                "thread_resume_attempts": 2,
                "failed_thread_resume_attempts": 0,
                "active_provider_attempt": None,
                "last_provider_attempt": {
                    "phase": "program",
                    "status": "failed",
                    "slot_id": "slot-01",
                    "worker_index": 1,
                    "thread_id": "worker-1",
                    "turn_id": "turn-3",
                    "provider_retries": 7,
                    "provider_warnings": 1,
                    "error": "TurnError: turn timed out",
                },
            },
        )
        raise RuntimeError("fixture provider failure")

    result = run_v3(
        path,
        backend_factory=_backend,
        auth_available=lambda _: True,
        preview_runner=preview_runner,
    )

    assert result["state"] == "blocked"
    assert result["provider_turns"] == 2
    assert result["provider_attempts"] == 3
    assert result["failed_provider_attempts"] == 1
    assert result["provider_retries"] == 7
    assert result["provider_warnings"] == 1
    assert result["provider_process_restarts"] == 1
    assert result["thread_resume_attempts"] == 2
    assert result["last_provider_attempt"]["slot_id"] == "slot-01"
    assert result["last_provider_attempt"]["turn_id"] == "turn-3"
    assert v3_status(path) == result


def test_selected_preview_auth_failure_is_resumable_before_scientific_work(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'communication_mode = "multi_program_batch"',
            (
                'communication_mode = "persistent_single_ast"\n'
                'output_contract = "slot_specific"'
            ),
        ),
        encoding="utf-8",
    )
    preview_calls = 0

    def preview_runner(*_: object, **__: object) -> dict[str, object]:
        nonlocal preview_calls
        preview_calls += 1
        raise AssertionError("auth preflight must run first")

    result = run_v3(
        path,
        backend_factory=_backend,
        auth_available=lambda _: False,
        preview_runner=preview_runner,
    )

    assert result["state"] == "blocked"
    assert result["resumable"] is True
    assert result["communication_mode"] == "persistent_single_ast"
    assert result["provider_mode"] == "persistent_single_ast"
    assert result["output_contract"] == "slot_specific"
    assert preview_calls == 0


def test_v3_completes_eight_slots_and_status_is_read_only(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)

    result = run_v3(
        path,
        provider_factory=_provider,
        backend_factory=_backend,
        auth_available=lambda _: True,
    )

    assert result["state"] == "completed"
    assert result["protocol"] == "v3"
    assert result["protocol_version"] == "v3"
    assert result["provider_turns"] == 2
    assert result["evaluation_count"] == 8
    assert result["cohort_outcome"] == "COMPLETE"
    assert result["unique_valid_programs"] == 8
    assert result["valid_slots"] == 8
    assert result["latest_scientific_stop_reason"] == "cohort_complete"
    manifest = Path(str(result["artifacts"]["epoch_manifest"]))
    report = Path(str(result["artifacts"]["cohort_report"]))
    assert manifest.is_file()
    assert report.is_file()
    report_payload = read_json(report)
    assert report_payload["selected_program_hash"] == (report_payload["canonical_program_order"][0])
    selected_evaluation = read_json(
        report.parent / "programs" / report_payload["selected_program_hash"] / "evaluation.json.gz"
    )
    assert selected_evaluation["protocols"]["score_evidence"] == "native_v3_score_50k_200k_v1"
    assert selected_evaluation["protocols"]["fitness"] == "native_v3_interval_utility_auc_v1"
    assert selected_evaluation["evaluation"]["status"] == "COMPLETE"
    fitness = selected_evaluation["evaluation"]["fitness_interval"]
    assert set(fitness) == {"lower", "upper"}
    assert set(fitness["lower"]) == {"numerator", "denominator"}
    assert set(fitness["upper"]) == {"numerator", "denominator"}
    for call_index, batch_report in enumerate(report_payload["batch_reports"]):
        turn = Path(batch_report["attempts"][0]["turn_directory"])
        prefix = f"slot-{call_index * 4:02d}"
        assert not (turn / "source.py").exists()
        assert (turn / "program-batch.json").is_file()
        turn_manifest = read_json(turn / "turn-manifest.json.gz")
        assert turn_manifest["source_artifact"] == "program-batch.json"
        assert turn_manifest["source_extraction"] is True
        response_markdown = (turn / f"{prefix}.response.md").read_text()
        assert response_markdown.startswith("# Native v3 program batch\n\n")
        assert "```json\n" in response_markdown
        assert "```python\n" not in response_markdown
        assert (turn / f"{prefix}.response.raw.txt").read_text() == _response(
            (
                f"slot-{call_index * 4:02d}",
                f"slot-{call_index * 4 + 1:02d}",
                f"slot-{call_index * 4 + 2:02d}",
                f"slot-{call_index * 4 + 3:02d}",
            ),
            offset=call_index * 4,
        )

    before = {
        item.relative_to(tmp_path): (item.stat().st_mtime_ns, item.read_bytes())
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    status = v3_status(path)
    after = {
        item.relative_to(tmp_path): (item.stat().st_mtime_ns, item.read_bytes())
        for item in tmp_path.rglob("*")
        if item.is_file()
    }
    assert status == result
    assert after == before


def test_auth_preflight_is_blocked_resumable_then_completes(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    provider_calls = 0
    backend_calls = 0

    def provider(config: V3Config) -> LocalCodexAppServerProvider:
        nonlocal provider_calls
        provider_calls += 1
        return _provider(config)

    def backend(config: V3Config) -> _Backend:
        nonlocal backend_calls
        backend_calls += 1
        return _backend(config)

    blocked = run_v3(
        path,
        provider_factory=provider,
        backend_factory=backend,
        auth_available=lambda _: False,
    )

    assert blocked["state"] == "blocked"
    assert blocked["resumable"] is True
    assert blocked["latest_infrastructure_stop_reason"] == "preflight_failed"
    assert blocked["latest_scientific_stop_reason"] is None
    assert provider_calls == 0
    assert backend_calls == 0

    completed = run_v3(
        path,
        provider_factory=provider,
        backend_factory=backend,
        auth_available=lambda _: True,
    )
    assert completed["state"] == "completed"
    assert provider_calls == 1
    assert backend_calls == 1


def test_partial_batch_is_not_repaired_and_duplicate_is_evaluated_once(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    first_slots = ("slot-00", "slot-01", "slot-02", "slot-03")
    responses = [
        _response(
            first_slots,
            sources={
                "slot-01": _source(0),
                "slot-02": "{}",
            },
        ),
        _response(("slot-04", "slot-05", "slot-06", "slot-07"), offset=4),
    ]

    result = run_v3(
        path,
        provider_factory=lambda _: _provider_with_responses(responses),
        backend_factory=_backend,
        auth_available=lambda _: True,
    )
    report = read_json(Path(str(result["artifacts"]["cohort_report"])))

    assert result["state"] == "completed"
    assert result["provider_turns"] == 2
    assert result["evaluation_count"] == 6
    assert report["cohort_outcome"] == "DEGRADED"
    assert report["unique_valid_programs"] == 6
    assert report["duplicate_aliases"] == 1
    assert report["batch_reports"][0]["repaired"] is False
    duplicate_hash = next(
        program_hash
        for program_hash, aliases in report["program_aliases"].items()
        if len(aliases) == 2
    )
    program_artifact = read_json(
        Path(str(result["artifacts"]["cohort_report"])).parent
        / "programs"
        / duplicate_hash
        / "program.json.gz"
    )
    assert [item["slot_id"] for item in program_artifact["lineage"]] == [
        "slot-00",
        "slot-01",
    ]
    assert all(
        item["provider_attempts"][0]["turn_directory"] for item in program_artifact["lineage"]
    )


def test_wholly_invalid_batch_gets_exactly_one_frozen_repair(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    first_slots = ("slot-00", "slot-01", "slot-02", "slot-03")
    responses = [
        _response(
            first_slots,
            sources={slot_id: "{}" for slot_id in first_slots},
        ),
        _response(first_slots, offset=0),
        _response(("slot-04", "slot-05", "slot-06", "slot-07"), offset=4),
    ]

    result = run_v3(
        path,
        provider_factory=lambda _: _provider_with_responses(responses),
        backend_factory=_backend,
        auth_available=lambda _: True,
    )
    report = read_json(Path(str(result["artifacts"]["cohort_report"])))

    assert result["state"] == "completed"
    assert result["provider_turns"] == 3
    assert result["evaluation_count"] == 8
    assert report["cohort_outcome"] == "COMPLETE"
    assert report["batch_reports"][0]["repaired"] is True
    assert report["batch_reports"][1]["repaired"] is False
    attempts = report["batch_reports"][0]["attempts"]
    assert [attempt["phase"] for attempt in attempts] == [
        "initial",
        "repair-01",
    ]
    initial_turn = Path(attempts[0]["turn_directory"])
    repair_turn = Path(attempts[1]["turn_directory"])
    initial_source = initial_turn / "program-batch.json"
    repair_source = repair_turn / "program-batch.json"
    assert initial_source.is_file()
    assert repair_source.is_file()
    assert not (initial_turn / "source.py").exists()
    assert not (repair_turn / "source.py").exists()
    assert initial_source.read_bytes() != repair_source.read_bytes()
    assert "```json\n" in (repair_turn / "slot-00.response.md").read_text()
    assert "```python\n" not in (repair_turn / "slot-00.response.md").read_text()
    assert all(attempt["provider_turn_id"] for attempt in attempts)


def test_v2_workspace_is_rejected_by_v3_without_mutation(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    root = tmp_path / "workspace" / "v3-run"
    root.mkdir(parents=True)
    (root / "state.sqlite3").write_bytes(b"v2-state-sentinel")
    (root / "experiment.toml").write_text(
        'schema_version = "mforge.experiment.v2"\n',
        encoding="utf-8",
    )
    before = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }

    with pytest.raises(
        V3WorkspaceError,
        match="never reinterpret a Native v2 workspace",
    ):
        run_v3(
            path,
            provider_factory=_provider,
            backend_factory=_backend,
            auth_available=lambda _: True,
        )
    status = v3_status(path)
    after = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }

    assert status["state"] == "failed"
    assert status["resumable"] is False
    assert after == before


def test_old_preview_workspace_marker_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    root = tmp_path / "workspace" / "v3-run"
    root.mkdir(parents=True)
    (root / "experiment.toml").write_bytes(path.read_bytes())
    write_json(
        root / "native-v3-preview-state.json.gz",
        {
            "schema_version": "mforge.experiment.status.v3-preview.v2",
            "protocol": "native-v3-preview",
            "protocol_version": "native-v3-preview.v2",
        },
    )
    before = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }

    with pytest.raises(V3WorkspaceError, match="not a v3 workspace"):
        run_v3(
            path,
            provider_factory=_provider,
            backend_factory=_backend,
            auth_available=lambda _: True,
        )

    after = {
        item.relative_to(root): (item.stat().st_mtime_ns, item.read_bytes())
        for item in root.rglob("*")
        if item.is_file()
    }
    assert after == before


def test_mixed_config_fails_before_workspace_provider_or_backend(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path, extra="\n[search]\npopulation_size = 8\n")
    provider_called = False
    backend_called = False

    def provider(_: V3Config) -> LocalCodexAppServerProvider:
        nonlocal provider_called
        provider_called = True
        return _provider(_)

    def backend(_: V3Config) -> _Backend:
        nonlocal backend_called
        backend_called = True
        return _Backend()

    with pytest.raises(ValueError, match="cannot contain Native v2 fields"):
        run_v3(
            path,
            provider_factory=provider,
            backend_factory=backend,
            auth_available=lambda _: True,
        )

    assert provider_called is False
    assert backend_called is False
    assert not (tmp_path / "workspace").exists()
