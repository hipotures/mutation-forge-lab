from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mutation_forge.experiment.artifacts import TurnArtifactStore
from mutation_forge.experiment.json_io import read_json
from mutation_forge.experiment.provider import (
    LocalCodexAppServerProvider,
    NativeProviderError,
)
from mutation_forge.models import (
    ExactVerification,
    GraphScore,
    GraphState,
    GraphValidation,
    RewritePlan,
)
from mutation_forge.native_v3.scoring import (
    AttemptKind,
    BackendIdentity,
    CycleComponentEvidence,
    EvidenceStatus,
    ScoreEvidence,
    ScoreTimeoutWithoutPartial,
)
from mutation_forge.native_v3_python import (
    API_METHODS,
    PythonSerialEpisodeConfigV1,
)
from mutation_forge.native_v3_python.provider_evaluation import (
    M4_REPORT_SCHEMA_VERSION,
    build_m4_request,
    run_m4_single_root,
)

_APP_SERVER_FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
)
_APP_SERVER_SPEC = importlib.util.spec_from_file_location(
    "m4_python_provider_fake",
    _APP_SERVER_FIXTURE_PATH,
)
assert _APP_SERVER_SPEC is not None and _APP_SERVER_SPEC.loader is not None
_APP_SERVER_FIXTURE = importlib.util.module_from_spec(_APP_SERVER_SPEC)
sys.modules[_APP_SERVER_SPEC.name] = _APP_SERVER_FIXTURE
_APP_SERVER_SPEC.loader.exec_module(_APP_SERVER_FIXTURE)
FakeProcess = _APP_SERVER_FIXTURE.FakeProcess
FakeScenario = _APP_SERVER_FIXTURE.FakeScenario

_NO_PLAN_SOURCE = """\
def propose(ctx, graph, api, seed):
    candidates = api.non_edges_local_cycle_risk(mode="max")
    selected = api.pick(candidates, seed, "cycle-risk")
    if not selected:
        return api.no_plan("NO_MATCH")
    if graph.edge_count > graph.order * 2:
        api.add_edge(selected)
        return api.emit()
    return api.no_plan("EXPLICIT")
"""
_INVALID_SOURCE = """\
import os

def propose(ctx, graph, api, seed):
    return api.no_plan("EXPLICIT")
"""
_SOURCE_WITH_REDACTION_LIKE_LITERAL = """\
def propose(ctx, graph, api, seed):
    candidates = api.non_edges_legal()
    selected = api.pick(candidates, seed, "secret=abc")
    if not selected:
        return api.no_plan("NO_MATCH")
    return api.no_plan("EXPLICIT")
"""
_IDENTITY = BackendIdentity(
    backend_id="m4-recorded",
    heg_commit="fixture",
    source_tree_sha256="a" * 64,
    binary_sha256="b" * 64,
    compiler_identity="fixture",
    build_flags=(),
    platform="fixture",
    architecture="fixture",
)


def _envelope(source: str) -> str:
    return json.dumps(
        {
            "schema_version": "mforge.native.python_policy_response.v1",
            "source": source,
        },
        separators=(",", ":"),
    )


class _Transport:
    def __init__(
        self,
        responses: Sequence[str],
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.responses = list(responses)
        self.failure = failure
        self.calls: list[tuple[str, Mapping[str, Any], object]] = []

    def _result(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.failure is not None:
            raise self.failure
        response = self.responses.pop(0)
        self.calls.append(("turn", dict(request), None))
        return {
            "status": "completed",
            "accepted": True,
            "charged": True,
            "content": True,
            "response": json.loads(response),
            "response_text": response,
            "response_projection_valid": False,
            "response_diagnostics": [],
            "provider_request_id": 10 + len(self.calls),
            "provider_thread_id": f"thread-{len(self.calls)}",
            "provider_turn_id": f"turn-{len(self.calls)}",
            "provider_duration_ms": 12,
            "model": "gpt-5.6-luna",
            "effort": "high",
            "usage": {
                "inputTokens": 100,
                "cachedInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 50,
                "reasoningOutputTokens": 10,
                "totalTokens": 150,
                "final": True,
                "partial": False,
            },
            "events": [
                {"method": "turn/started"},
                {"method": "turn/completed", "params": {"status": "completed"}},
            ],
        }

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._result(request)

    def repair(
        self,
        request: Mapping[str, Any],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        self.calls.append(("repair", dict(request), tuple(diagnostics)))
        return self._result(request)

    def close(self) -> None:
        return None


def _cubic_graph(order: int) -> GraphState:
    edges = {
        tuple(sorted((vertex, (vertex + 1) % order)))
        for vertex in range(order)
    }
    edges.update(
        (vertex, vertex + order // 2) for vertex in range(order // 2)
    )
    return GraphState(order, tuple(sorted(edges)))


class _Backend:
    backend_id = "m4-recorded"
    score_implementation = "m4-recorded-evidence"

    def __init__(self) -> None:
        self.closed = False
        self.raw_graph_score_calls = 0
        self.unique_graph_scores = 0

    def target_forbidden_lengths(self, order: int) -> tuple[int, ...]:
        assert order == 8
        return (4, 8)

    def generate_seed(self, *, order: int, seed: int) -> GraphState:
        assert seed == 101
        return _cubic_graph(order)

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
        del graph, witness_cap, cutoff, record_profile
        return GraphScore(True, ((4, 2), (8, 1)), 3, 3, True, (3, 1))

    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        lengths = tuple(forbidden_lengths or ())
        assert lengths == (4, 8)
        self.raw_graph_score_calls += 1
        self.unique_graph_scores += 1
        return ScoreEvidence(
            graph_content_hash=self.state_hash(graph),
            order=graph.order,
            edge_count=len(graph.edges),
            witness_cap=witness_cap,
            components=tuple(
                CycleComponentEvidence(
                    forbidden_length=length,
                    observed_count=1,
                    lower_bound=1,
                    upper_bound=1,
                    status=EvidenceStatus.EXACT,
                    node_budget=50_000,
                    nodes_visited=1,
                    wall_time_ns=0,
                    attempt_kind=attempt_kind,
                    backend_identity=_IDENTITY,
                )
                for length in lengths
            ),
        )

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        raise AssertionError(f"nonzero score must not verify {graph!r}")

    def canonical_hash(self, graph: GraphState) -> str:
        return self.state_hash(graph)

    def state_hash(self, graph: GraphState) -> str:
        return hashlib.sha256(repr(graph).encode()).hexdigest()

    def serialize_graph6(self, graph: GraphState) -> str:
        raise AssertionError(graph)

    def deserialize_graph6(self, value: str) -> GraphState:
        raise AssertionError(value)

    def apply_rewrite(
        self,
        graph: GraphState,
        rewrite: RewritePlan,
        *,
        record_score_profile: Any = None,
    ) -> GraphState:
        del graph, rewrite, record_score_profile
        raise AssertionError("fixture program must return NoPlan")

    def propose_rewrite(self, graph: GraphState, **kwargs: Any) -> RewritePlan:
        raise AssertionError((graph, kwargs))

    def close(self) -> None:
        self.closed = True


class _TimeoutBackend(_Backend):
    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        del graph, witness_cap, forbidden_lengths, attempt_kind
        raise ScoreTimeoutWithoutPartial("recorded unsafe score timeout")


class _VerifierFailureBackend(_Backend):
    def score_evidence(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        forbidden_lengths: Iterable[int] | None = None,
        attempt_kind: AttemptKind = AttemptKind.INITIAL,
    ) -> ScoreEvidence:
        lengths = tuple(forbidden_lengths or ())
        self.raw_graph_score_calls += 1
        self.unique_graph_scores += 1
        return ScoreEvidence(
            graph_content_hash=self.state_hash(graph),
            order=graph.order,
            edge_count=len(graph.edges),
            witness_cap=witness_cap,
            components=tuple(
                CycleComponentEvidence(
                    forbidden_length=length,
                    observed_count=0,
                    lower_bound=0,
                    upper_bound=0,
                    status=EvidenceStatus.EXACT,
                    node_budget=50_000,
                    nodes_visited=1,
                    wall_time_ns=0,
                    attempt_kind=attempt_kind,
                    backend_identity=_IDENTITY,
                )
                for length in lengths
            ),
        )

    def score(
        self,
        graph: GraphState,
        *,
        witness_cap: int,
        cutoff: GraphScore | None = None,
        record_profile: Any = None,
    ) -> GraphScore:
        del graph, witness_cap, cutoff, record_profile
        return GraphScore(True, ((4, 0), (8, 0)), 0, 0, True, (0, 0))

    def exact_verify(self, graph: GraphState) -> ExactVerification:
        raise RuntimeError(f"recorded exact verifier failure for {graph.order}")


def _config() -> PythonSerialEpisodeConfigV1:
    return PythonSerialEpisodeConfigV1(
        order=8,
        graph_seed=101,
        policy_seed=17,
        horizon=1,
        witness_cap=64,
        episode_id="m4-recorded-root",
        forbidden_lengths=(4, 8),
    )


def _provider(transport: _Transport) -> LocalCodexAppServerProvider:
    return LocalCodexAppServerProvider(
        model="gpt-5.6-luna",
        effort="high",
        concurrency=1,
        max_repairs=1,
        transport=transport,
        persist_artifacts=False,
    )


def test_m4_request_is_small_python_only_and_model_facing_hygienic(
    tmp_path: Path,
) -> None:
    request = build_m4_request(
        tmp_path,
        model="gpt-5.6-luna",
        effort="high",
    )

    assert request["response_projection"] == "native-v3-python-policy"
    assert request["output_schema"]["required"] == ["schema_version", "source"]
    assert request["output_schema"]["properties"]["schema_version"] == {
        "type": "string",
        "const": "mforge.native.python_policy_response.v1",
    }
    assert len(json.dumps(request["output_schema"]).encode()) < 1_024
    model_facing = (
        str(request["system_prompt"])
        + "\n"
        + str(request["prompt"])
        + "\n"
        + json.dumps(request["output_schema"])
    )
    assert "/home/" not in model_facing
    assert "program_hash" not in model_facing
    assert "thread_id" not in model_facing
    assert "native.program.v3" not in model_facing
    assert "def propose(ctx, graph, api, seed):" in model_facing
    assert "Erdős–Gyárfás" in model_facing
    assert all(f"api.{method}(" in model_facing for method in API_METHODS)


def test_app_server_transport_projects_two_field_python_response(
    tmp_path: Path,
) -> None:
    scenario = FakeScenario(final_text=_envelope(_NO_PLAN_SOURCE))

    def process_factory(*_: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(scenario, **kwargs)

    provider = LocalCodexAppServerProvider(
        model="gpt-5.6-luna",
        effort="high",
        concurrency=1,
        max_repairs=1,
        process_factory=process_factory,
        auth_checker=lambda _: True,
        persist_artifacts=False,
    )
    request = build_m4_request(
        tmp_path,
        model=provider.model,
        effort=provider.effort,
    )
    try:
        result = provider.generate(request)
    finally:
        provider.close()

    assert result["status"] == "completed"
    assert result["response_projection_valid"] is True
    assert result["response_diagnostics"] == []
    assert result["response"] == json.loads(_envelope(_NO_PLAN_SOURCE))
    turn = Path(str(request["artifact_dir"]))
    assert (turn / "slot-00.response.raw.txt").read_text(encoding="utf-8") == (
        _envelope(_NO_PLAN_SOURCE)
    )
    markdown = (turn / "slot-00.response.md").read_text(encoding="utf-8")
    assert "def propose(ctx, graph, api, seed):" in markdown
    assert (turn / "slot-00.output-schema.json.gz").is_file()


def test_app_server_turn_materializes_final_usage_before_manifest(
    tmp_path: Path,
) -> None:
    scenario = FakeScenario(final_text=_envelope(_NO_PLAN_SOURCE))

    def process_factory(*_: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(scenario, **kwargs)

    provider = LocalCodexAppServerProvider(
        model="gpt-5.6-luna",
        effort="high",
        concurrency=1,
        max_repairs=1,
        process_factory=process_factory,
        auth_checker=lambda _: True,
        persist_artifacts=False,
    )
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    turn = Path(str(report["provider_turn_directory"]))
    assert report["status"] == "completed"
    assert (turn / "slot-00.usage.json.gz").is_file()
    assert read_json(turn / "turn-manifest.json.gz")["artifact_complete"] is True


def test_recorded_model_python_runs_one_root_and_replays_offline(
    tmp_path: Path,
) -> None:
    semantic: list[dict[str, Any]] = []
    for name in ("first", "second"):
        backend = _Backend()
        transport = _Transport([_envelope(_NO_PLAN_SOURCE)])
        provider = _provider(transport)
        try:
            report = run_m4_single_root(
                provider,
                tmp_path / name,
                backend_factory=lambda backend=backend: backend,
                config=_config(),
            )
        finally:
            provider.close()
        assert report["status"] == "completed"
        assert report["root_count"] == 1
        assert report["programs_per_turn"] == 1
        assert report["model_turns"] == 1
        assert report["outcome"]["kind"] == "NO_PLAN"
        assert report["graph_score_attempts"] == 1
        assert report["dsl_runtime_used"] is False
        assert backend.closed is True
        turn = Path(str(report["provider_turn_directory"]))
        source_archive = Path(str(report["source_archive"]))
        evaluation_path = Path(str(report["evaluation_result"]))
        assert source_archive.read_text(encoding="utf-8") == _NO_PLAN_SOURCE
        assert source_archive.is_relative_to(tmp_path / name)
        assert not source_archive.is_relative_to(turn)
        assert not evaluation_path.is_relative_to(turn)
        assert TurnArtifactStore(tmp_path / name / "artifacts").verify_turn(turn)
        evaluation = read_json(evaluation_path)
        semantic.append(
            {
                "source": evaluation["source"],
                "program_identity": evaluation["program_identity"],
                "behavior_identity": evaluation["behavior_identity"],
                "scientific": evaluation["evaluation"]["scientific_result"],
                "outcome": evaluation["outcome"],
            }
        )
        assert evaluation["protocols"]["dsl_runtime_used"] is False
        assert evaluation["external_activity"] == {
            "provider_turns": 1,
            "model_turns": 1,
            "app_server_calls": 1,
        }
    assert semantic[0] == semantic[1]


def test_invalid_first_response_uses_one_bounded_repair(
    tmp_path: Path,
) -> None:
    transport = _Transport(
        [_envelope(_INVALID_SOURCE), _envelope(_NO_PLAN_SOURCE)]
    )
    backend = _Backend()
    provider = _provider(transport)
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=lambda: backend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "completed"
    assert report["model_turns"] == 2
    assert report["repair_attempts"] == 1
    assert len(report["usage"]) == 2
    assert [call[0] for call in transport.calls] == ["turn", "repair", "turn"]
    repair_request = transport.calls[1][1]
    assert repair_request["phase"] == "repair-01"
    assert "FORBIDDEN_AST_NODE" in str(repair_request["prompt"])
    initial = tmp_path / "artifacts/generations/generation-0000/slot-00/initial"
    repair = tmp_path / "artifacts/generations/generation-0000/slot-00/repair-01"
    assert TurnArtifactStore(tmp_path / "artifacts").verify_turn(initial)
    assert TurnArtifactStore(tmp_path / "artifacts").verify_turn(repair)


def test_two_invalid_responses_create_no_scientific_result(
    tmp_path: Path,
) -> None:
    backend_created = False

    def backend_factory() -> _Backend:
        nonlocal backend_created
        backend_created = True
        return _Backend()

    transport = _Transport(
        [_envelope(_INVALID_SOURCE), _envelope(_INVALID_SOURCE)]
    )
    provider = _provider(transport)
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=backend_factory,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "contract_invalid"
    assert report["model_turns"] == 2
    assert report["scientific_result"] is False
    assert backend_created is False
    assert not (
        tmp_path
        / "native-v3-python-output/root-0000/evaluation-result.json.gz"
    ).exists()


def test_provider_failure_creates_no_backend_or_scientific_result(
    tmp_path: Path,
) -> None:
    backend_created = False

    def backend_factory() -> _Backend:
        nonlocal backend_created
        backend_created = True
        return _Backend()

    provider = _provider(
        _Transport([], failure=NativeProviderError("recorded provider failure"))
    )
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=backend_factory,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "provider_error"
    assert report["model_turns"] == 0
    assert report["scientific_result"] is False
    assert backend_created is False
    assert not (
        tmp_path
        / "native-v3-python-output/root-0000/evaluation-result.json.gz"
    ).exists()


def test_provider_partial_artifact_failure_still_emits_m4_report(
    tmp_path: Path,
) -> None:
    class PartialArtifactFailureTransport(_Transport):
        def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            Path(str(request["artifact_dir"])).mkdir(parents=True)
            raise NativeProviderError("recorded failure after logger creation")

    provider = _provider(PartialArtifactFailureTransport([]))
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "provider_error"
    assert report["provider_attempts"] == 1
    assert report["scientific_result"] is False
    assert "ArtifactIncompleteError" in report["artifact_recording_error"]
    report_path = (
        tmp_path
        / "native-v3-python-output/root-0000/m4-report.json.gz"
    )
    assert read_json(report_path)["status"] == "provider_error"


def test_repair_provider_failure_reports_attempt_without_scientific_result(
    tmp_path: Path,
) -> None:
    class RepairFailureTransport(_Transport):
        def repair(
            self,
            request: Mapping[str, Any],
            diagnostics: Sequence[Mapping[str, Any]],
        ) -> Mapping[str, Any]:
            del request, diagnostics
            raise NativeProviderError("recorded repair failure")

    provider = _provider(
        RepairFailureTransport([_envelope(_INVALID_SOURCE)])
    )
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "provider_error"
    assert report["provider_completed"] is False
    assert report["provider_attempts"] == 2
    assert report["model_turns"] == 1
    assert report["scientific_result"] is False


def test_nonterminal_provider_mapping_fails_before_backend(
    tmp_path: Path,
) -> None:
    backend_created = False

    class FailedMappingTransport(_Transport):
        def _result(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            return {
                **dict(super()._result(request)),
                "status": "failed",
                "accepted": False,
            }

    transport = FailedMappingTransport([_envelope(_NO_PLAN_SOURCE)])

    def backend_factory() -> _Backend:
        nonlocal backend_created
        backend_created = True
        return _Backend()

    provider = _provider(transport)
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=backend_factory,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "provider_error"
    assert report["scientific_result"] is False
    assert backend_created is False


def test_unsafe_initial_scoring_timeout_is_inconclusive_not_program_failure(
    tmp_path: Path,
) -> None:
    provider = _provider(_Transport([_envelope(_NO_PLAN_SOURCE)]))
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_TimeoutBackend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "scoring_inconclusive"
    assert report["scientific_result"] is False
    assert report["verification_completed"] is False
    assert report["outcome"]["kind"] == "NO_INVOCATION"


def test_exact_verifier_failure_is_not_a_scientific_result(
    tmp_path: Path,
) -> None:
    provider = _provider(_Transport([_envelope(_NO_PLAN_SOURCE)]))
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_VerifierFailureBackend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "evaluation_error"
    assert report["error_classification"] == "verification"
    assert report["scientific_result"] is False
    assert report["graph_score_attempts"] == 0


def test_exact_source_bytes_are_not_redacted_in_scientific_archive(
    tmp_path: Path,
) -> None:
    provider = _provider(
        _Transport([_envelope(_SOURCE_WITH_REDACTION_LIKE_LITERAL)])
    )
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    archive = Path(str(report["source_archive"]))
    assert archive.read_text(encoding="utf-8") == _SOURCE_WITH_REDACTION_LIKE_LITERAL
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        report["validation"]["identity"]["source_sha256"]
    )
    turn = Path(str(report["provider_turn_directory"]))
    assert read_json(turn / "canonical_response.json.gz")["source"] == (
        _SOURCE_WITH_REDACTION_LIKE_LITERAL
    )
    assert read_json(turn / "validation.json.gz")["response"]["source"] == (
        _SOURCE_WITH_REDACTION_LIKE_LITERAL
    )


def test_malformed_transport_evidence_fails_closed_with_report(
    tmp_path: Path,
) -> None:
    class BadWireTransport(_Transport):
        def _result(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            return {
                **dict(super()._result(request)),
                "wire": [{"missing_direction": True}],
            }

    provider = _provider(BadWireTransport([_envelope(_NO_PLAN_SOURCE)]))
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "artifact_error"
    assert report["error_classification"] == "infrastructure"
    assert report["scientific_result"] is False
    assert report["evaluation_completed"] is False


def test_preexisting_source_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    class StaleSourceTransport(_Transport):
        def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            turn = Path(str(request["artifact_dir"]))
            turn.mkdir(parents=True)
            (turn / "source.py").write_text(
                'def propose(ctx, graph, api, seed):\n    return api.no_plan("NO_MATCH")\n',
                encoding="utf-8",
            )
            return super().generate(request)

    provider = _provider(StaleSourceTransport([_envelope(_NO_PLAN_SOURCE)]))
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "artifact_error"
    assert "does not match the validated response" in report["error"]
    assert report["scientific_result"] is False


def test_repair_diagnostics_are_fully_bounded(tmp_path: Path) -> None:
    request = build_m4_request(
        tmp_path,
        model="gpt-5.6-luna",
        effort="high",
        phase="repair-01",
        repair_attempt=1,
        diagnostics=[
            {
                "code": "X" * 10_000,
                "path": "Y" * 1_000_000,
                "message": "Z" * 1_000_000,
                "line": "not-an-integer",
                "column": ["not", "an", "integer"],
            }
        ]
        * 100,
    )

    prompt = str(request["prompt"])
    assert len(prompt.encode("utf-8")) < 100_000
    assert "X" * 129 not in prompt
    assert "Y" * 513 not in prompt
    assert "Z" * 513 not in prompt


def test_provider_repair_receives_only_bounded_diagnostics(
    tmp_path: Path,
) -> None:
    invalid = "\n".join(f"import forbidden_{index}" for index in range(100))
    invalid += (
        "\n\ndef propose(ctx, graph, api, seed):\n"
        '    return api.no_plan("EXPLICIT")\n'
    )
    transport = _Transport(
        [_envelope(invalid), _envelope(_NO_PLAN_SOURCE)]
    )
    provider = _provider(transport)
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["status"] == "completed"
    repair_diagnostics = transport.calls[1][2]
    assert isinstance(repair_diagnostics, tuple)
    assert 0 < len(repair_diagnostics) <= 32


def test_m4_report_status_contract_is_versioned(tmp_path: Path) -> None:
    provider = _provider(_Transport([_envelope(_NO_PLAN_SOURCE)]))
    try:
        report = run_m4_single_root(
            provider,
            tmp_path,
            backend_factory=_Backend,
            config=_config(),
        )
    finally:
        provider.close()

    assert report["schema_version"] == M4_REPORT_SCHEMA_VERSION
    assert report["provider_completed"] is True
    assert report["contract_valid"] is True
    assert report["sandbox_completed"] is True
    assert report["evaluation_completed"] is True
    assert report["scientific_result"] is True
    assert report["verification"]["authority"] == "exact_verifier_only"
    assert report["parent_count"] == 0
    assert report["lineage_count"] == 0
    assert report["generation_count"] == 0
    assert report["m5_features_used"] is False
