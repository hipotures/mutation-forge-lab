from __future__ import annotations

import io
import json
import multiprocessing
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.experiment.config import load_experiment_config
from mutation_forge.experiment.generation import (
    GenerationConfig,
    GenerationCoordinator,
    ProviderResult,
)
from mutation_forge.experiment.layout import ExperimentLayout
from mutation_forge.experiment.native import (
    NativeExperimentAdapter,
    _evaluate_candidate_process,
    _resume_parent_assignments,
)
from mutation_forge.experiment.service import ExperimentService
from mutation_forge.experiment.state import ExperimentStateStore
from mutation_forge.experiment.status import experiment_status
from mutation_forge.output.rich_live import RichLiveSink
from mutation_forge.sandbox.contracts import VALIDATOR_VERSION, SandboxLimits
from mutation_forge.sandbox.validation import (
    SAFE_BUILTINS,
    render_policy_validator_contract,
)

VALID_SOURCE = 'def priority(ctx, proposal):\n    return proposal["local_c4_risk"]\n'
INVALID_SOURCE = "def priority(ctx, proposal)\n    return 0\n"


def test_native_output_schema_uses_app_server_supported_keywords() -> None:
    schema = json.loads(
        Path("configs/native/generated-policy.schema.json").read_text(encoding="utf-8")
    )

    def contains_unique_items(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(
                key == "uniqueItems" or contains_unique_items(item) for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_unique_items(item) for item in value)
        return False

    assert not contains_unique_items(schema)


def _write_config(
    path: Path,
    *,
    workspace: Path,
    exp_id: str = "native-test",
    population: int = 1,
    generations: int = 1,
    max_turns: int = 2,
    concurrency: int = 1,
    max_repairs: int = 1,
    workers: int = 1,
    thread_count: int = 1,
    replay: bool = False,
    hourly_token_limit: int | None = None,
) -> Path:
    hourly_limit_line = (
        f"\nmax_total_tokens_per_hour = {hourly_token_limit}"
        if hourly_token_limit is not None
        else ""
    )
    path.write_text(
        f'''schema_version = "mforge.experiment.v2"
exp_id = "{exp_id}"
workspace = "{workspace.as_posix()}"
kind = "heg"
preset = "native"

[run]
wall_seconds = 30{hourly_limit_line}

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "high"
concurrency = {concurrency}
max_repairs = {max_repairs}

[search]
population_size = {population}
max_generations = {generations}
max_model_turns = {max_turns}
selection = "elite-diversity"

[evaluation]
orders = [4]
graph_seeds = [1]
policy_seeds = [2]
horizon = 1
proposal_pool_size = 2
baselines = ["random", "structural"]
replay = {str(replay).lower()}

[resources]
workers = {workers}
thread_count = {thread_count}
''',
        encoding="utf-8",
    )
    return path


class RecordingProvider:
    def __init__(self, source: str = VALID_SOURCE) -> None:
        self.source = source
        self.calls: list[dict[str, Any]] = []
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.calls.append(dict(request))
        try:
            response = {
                "schema_version": "mforge.native.generated_policy.v1",
                "source": self.source,
                "design_summary": "A deterministic test policy.",
                "hypothesis": "A bounded structural score is reproducible.",
                "used_fields": ["proposal.local_c4_risk"],
                "assumptions": ["The host supplies legal proposals."],
                "expected_failure_modes": ["The simple ranker may underperform."],
            }
            usage = {
                "inputTokens": 1,
                "cachedInputTokens": 0,
                "outputTokens": 1,
                "reasoningOutputTokens": 0,
                "totalTokens": 2,
                "final": True,
                "partial": False,
            }
            return {
                "status": "completed",
                "accepted": True,
                "content": True,
                "response": response,
                "response_text": json.dumps(response, sort_keys=True),
                "usage": usage,
                "provider_thread_id": "thread-native",
                "provider_turn_id": f"turn-{len(self.calls):04d}",
                "model": request.get("model"),
                "effort": request.get("effort"),
            }
        finally:
            with self._lock:
                self._active -= 1

    def close(self) -> None:
        return None


class RepairProvider(RecordingProvider):
    def __init__(self) -> None:
        super().__init__(INVALID_SOURCE)

    def repair(
        self, request: Mapping[str, Any], diagnostics: tuple[Mapping[str, Any], ...]
    ) -> Mapping[str, Any]:
        assert diagnostics
        self.source = VALID_SOURCE
        return self.generate(request)


class InvalidRepairProvider(RecordingProvider):
    def __init__(self, source: str = INVALID_SOURCE) -> None:
        super().__init__(source)

    def repair(
        self, request: Mapping[str, Any], diagnostics: tuple[Mapping[str, Any], ...]
    ) -> Mapping[str, Any]:
        assert diagnostics
        return self.generate(request)


class ChargedFailure(RuntimeError):
    def __init__(self) -> None:
        self.evidence = {
            "accepted": True,
            "charged": True,
            "content": False,
            "uncharged": False,
            "usage": {
                "inputTokens": 8_000,
                "cachedInputTokens": 0,
                "outputTokens": 4_000,
                "reasoningOutputTokens": 1_000,
                "totalTokens": 12_000,
                "final": False,
                "partial": True,
            },
            "provider_thread_id": "thread-failed",
            "provider_turn_id": "turn-failed",
        }
        super().__init__("transport timeout")


class ChargedFailureProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, _request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        raise ChargedFailure

    def close(self) -> None:
        return None


def _service(provider: Any, *, engine: Any | None = None) -> ExperimentService:
    return ExperimentService(
        adapter=NativeExperimentAdapter(
            provider=provider,
            engine=engine,
            backend=ToyBackend(),
        )
    )


def test_public_experiment_imports_do_not_load_historical_stage_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mutation_forge.cli, mutation_forge.experiment; "
                "print(sorted(name for name in sys.modules if name == 'mutation_forge.stage4' "
                "or name.startswith('mutation_forge.stage4.')))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "[]"


def test_service_defaults_to_native_adapter() -> None:
    assert isinstance(ExperimentService().adapter, NativeExperimentAdapter)


def test_public_experiment_sources_have_no_historical_stage4_references() -> None:
    forbidden = re.compile(
        r"LegacyStage4Adapter|legacy_stage4_config|stage4-search|search-freeze|"
        r"campaign_root|mutation_forge\.stage4"
    )
    for relative in ("src/mutation_forge/experiment", "src/mutation_forge/cli.py"):
        root = Path(relative)
        paths = root.rglob("*.py") if root.is_dir() else (root,)
        for path in paths:
            assert forbidden.search(path.read_text(encoding="utf-8")) is None, path


def test_fresh_finite_native_experiment_exhausts_and_next_run_is_a_noop(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path / "experiment.toml", workspace=tmp_path / "workspace")
    provider = RecordingProvider()
    service = _service(provider)

    first = service.run(config_path)
    root = tmp_path / "workspace" / "native-test"
    second = service.run(config_path)

    assert first["state"] == "exhausted"
    assert second["state"] == "exhausted"
    assert second["stop_reason"] == "generation_limit"
    assert [request["generation"] for request in provider.calls] == [0]
    assert (root / "experiment.toml").is_file()
    assert (root / "experiment.lock.json").is_file()
    assert (root / "state.sqlite3").is_file()
    assert list((root / "checkpoints").glob("checkpoint-*.json"))
    assert (root / "artifacts" / "archive" / "programs").is_dir()
    assert (root / "artifacts" / "evaluations" / "development").is_dir()
    assert list((root / "artifacts" / "evaluations" / "development").glob("*.json"))
    assert not any("stage4" in str(path).lower() for path in root.rglob("*"))
    assert not (tmp_path / "runs" / "stage4-search").exists()
    lock_text = (root / "experiment.lock.json").read_text(encoding="utf-8").lower()
    assert "search-freeze" not in lock_text
    assert "stage4" not in lock_text
    status = experiment_status(config_path)
    assert status["state"] == "exhausted"
    assert status["provider_turns"] == 1
    assert status["unique_candidate_count"] == 1
    assert status["best_program_id"] == "g0000-slot-00"


def test_native_live_progress_is_visible_before_blocked_provider_finishes(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        exp_id="native-blocked",
    )
    started = threading.Event()
    release = threading.Event()

    class BlockingProvider(RecordingProvider):
        def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            started.set()
            assert release.wait(5)
            return super().generate(request)

    sink = RichLiveSink(console=Console(file=io.StringIO(), force_terminal=False))
    service = ExperimentService(
        adapter=NativeExperimentAdapter(
            provider=BlockingProvider(),
            backend=ToyBackend(),
        ),
        event_sinks=[sink],
    )
    result: dict[str, Any] = {}
    errors: list[BaseException] = []

    def run() -> None:
        try:
            result.update(service.run(config_path))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert started.wait(5)
        snapshot = io.StringIO()
        Console(file=snapshot, force_terminal=False, width=180).print(sink._render())
        rendered = snapshot.getvalue()
        assert not errors
        assert sink.state["latest_event"] == "provider_turn_started"
        assert "active 1" in rendered
        assert "turns 0/1" in rendered
        assert "native-blocked" in rendered
    finally:
        release.set()
        worker.join(timeout=10)
        sink.close()
    assert not worker.is_alive()
    assert not errors
    assert result["state"] == "exhausted"


def test_native_config_values_reach_provider_and_evaluator(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        population=2,
        max_turns=3,
        concurrency=1,
        workers=1,
        thread_count=1,
    )
    provider = RecordingProvider()
    result = _service(provider).run(config_path)
    config = load_experiment_config(config_path)
    root = ExperimentLayout.from_config(config).root
    lock = json.loads((root / "experiment.lock.json").read_text(encoding="utf-8"))

    assert result["state"] == "exhausted"
    assert len(provider.calls) == 2
    assert provider.max_active == 1
    assert {request["model"] for request in provider.calls} == {"gpt-5.6-luna"}
    assert {request["effort"] for request in provider.calls} == {"high"}
    assert all(request["system_prompt"] for request in provider.calls)
    assert all(request["output_schema"]["type"] == "object" for request in provider.calls)
    prompt = provider.calls[0]["prompt"]
    assert prompt.startswith("# Mutation Forge native ranker task\n")
    assert "## Context contract" in prompt
    assert "## Proposal contract" in prompt
    assert "## Experiment configuration" in prompt
    assert '"parent_id"' not in prompt
    assert lock["model"]["concurrency"] == 1
    assert lock["search"]["population_size"] == 2
    assert lock["search"]["max_generations"] == 1
    assert lock["search"]["max_model_turns"] == 3
    evaluation = next((root / "artifacts" / "evaluations" / "development").glob("*.json"))
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    assert payload["settings"]["workers"] == 1
    assert payload["settings"]["thread_count"] == 1
    turn = root / "artifacts" / "generations" / "generation-0000" / "slot-00" / "initial"
    assert (turn / "slot-00.request.md").read_text(encoding="utf-8") == prompt
    request_envelope = json.loads((turn / "slot-00.request.json").read_text(encoding="utf-8"))
    assert request_envelope["prompt"] == prompt
    assert (turn / "slot-00.system-prompt.md").is_file()
    assert (turn / "slot-00.output-schema.json").is_file()


def test_native_evaluation_runs_in_spawned_process(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        population=1,
        generations=1,
        max_turns=1,
        thread_count=2,
    )
    config = load_experiment_config(config_path)
    receive_progress, send_progress = multiprocessing.Pipe(duplex=False)
    heg_repo = Path(__file__).resolve().parents[3] / "heg"
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future = executor.submit(
            _evaluate_candidate_process,
            config,
            "process-canary",
            VALID_SOURCE,
            tmp_path / "artifacts",
            heg_repo,
            False,
            send_progress,
        )
        messages: list[tuple[str, Mapping[str, Any]]] = []
        while True:
            kind, payload = receive_progress.recv()
            if kind == "done":
                break
            if isinstance(payload, Mapping):
                messages.append((kind, payload))
        result, elapsed = future.result(timeout=30)
    send_progress.close()
    receive_progress.close()

    assert result["status"] == "completed"
    assert elapsed > 0
    assert any(kind == "progress" for kind, _payload in messages)


def test_native_slot_events_do_not_report_queued_turns_as_active() -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []
    provider = RecordingProvider()

    GenerationCoordinator(
        provider,
        config=GenerationConfig(
            generations=1,
            population_size=2,
            concurrency=1,
            max_model_turns=2,
            max_repairs=0,
        ),
        observer=lambda event_type, payload: events.append((event_type, payload)),
    ).run()

    slot_events = [payload for event_type, payload in events if event_type == "slot_queued"]
    assert provider.max_active == 1
    assert slot_events
    assert all("active_model_turns" not in payload for payload in slot_events)


def test_native_recovery_event_replays_authoritative_usage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "native-generation-checkpoint.json"
    provider = RecordingProvider()

    def interrupt_after_generation(*_args: Any) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        GenerationCoordinator(
            provider,
            config=GenerationConfig(
                generations=1,
                population_size=1,
                concurrency=1,
                max_model_turns=1,
                max_repairs=0,
                checkpoint_path=checkpoint,
            ),
            selection_callback=interrupt_after_generation,
        ).run()

    events: list[tuple[str, Mapping[str, Any]]] = []
    GenerationCoordinator(
        provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=1,
            max_repairs=0,
            checkpoint_path=checkpoint,
        ),
        selection_callback=lambda *_args: None,
        observer=lambda event_type, payload: events.append((event_type, payload)),
    ).run(resume=True)

    recovered = next(
        payload
        for event_type, payload in events
        if event_type == "slot_queued" and payload.get("recovered") is True
    )
    assert len(provider.calls) == 1
    assert recovered["usage"] == {
        "inputTokens": 1,
        "cachedInputTokens": 0,
        "outputTokens": 1,
        "reasoningOutputTokens": 0,
        "totalTokens": 2,
        "final": True,
        "partial": False,
    }
    assert recovered["inputTokens"] == 1
    assert recovered["cachedInputTokens"] == 0
    assert recovered["outputTokens"] == 1
    assert recovered["reasoningOutputTokens"] == 0
    assert recovered["totalTokens"] == 2
    assert recovered["usage_quality"] == "exact"
    assert recovered["recovered_status"] == "accepted"
    assert recovered["candidate_id"] == "g0000-slot-00"
    assert recovered["validation_status"] == "passed"
    assert recovered["probe_status"] == "passed"
    assert recovered["charged"] is True


def test_native_repair_persists_separate_initial_and_repair_turns(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=2,
        max_repairs=1,
    )
    provider = RepairProvider()
    result = _service(provider).run(config_path)
    root = tmp_path / "workspace" / "native-test"
    initial = root / "artifacts" / "generations" / "generation-0000" / "slot-00" / "initial"
    repair = initial.parent / "repair-01"

    assert result["state"] == "exhausted"
    assert [request["phase"] for request in provider.calls] == ["initial", "repair"]
    assert json.loads((initial / "validation.json").read_text(encoding="utf-8"))["valid"] is False
    assert json.loads((repair / "validation.json").read_text(encoding="utf-8"))["valid"] is True
    assert (initial / "turn-manifest.json").is_file()
    assert (repair / "turn-manifest.json").is_file()
    assert (repair / "behavior.json").is_file()
    assert list((root / "artifacts" / "archive" / "programs").glob("*.json"))
    assert list((root / "artifacts" / "evaluations" / "development").glob("*.json"))


def test_invalid_final_repair_is_terminal_and_not_repeated_on_resume(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=2,
        max_repairs=1,
    )
    provider = InvalidRepairProvider()
    service = _service(provider)

    first = service.run(config_path)
    second = service.run(config_path)
    root = tmp_path / "workspace" / "native-test"
    checkpoint = json.loads(
        (root / "artifacts" / "native-generation-checkpoint.json").read_text(encoding="utf-8")
    )
    events = [
        json.loads(line)
        for line in (root / "artifacts" / "sessions" / "session-000001" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    slot_states = list(checkpoint["slots"].values())
    state = ExperimentStateStore(root / "state.sqlite3")
    try:
        assert state.counts()["provider_turns"] == 2
        assert state.cumulative()["total_tokens"] == 4
    finally:
        state.close()

    assert first["state"] == "exhausted"
    assert second["stop_reason"] == "generation_limit"
    assert [request["generation"] for request in provider.calls] == [0, 0]
    assert {item["status"] for item in slot_states} == {"invalid"}
    assert {item["repairs"] for item in slot_states} == {1}
    assert {item["remaining_repairs"] for item in slot_states} == {0}
    assert all(len(item["repair_idempotency_keys"]) == 1 for item in slot_states)
    assert any(
        event["event_type"] == "validation_completed"
        and "syntax_error" in event["validation_codes"]
        for event in events
    )
    assert any(
        event["event_type"] == "repair_completed"
        and event["status"] == "invalid"
        and event["repair_state"] == "repair_failed"
        for event in events
    )


def test_multiple_repairs_use_distinct_durable_attempts(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=3,
        max_repairs=2,
    )
    provider = InvalidRepairProvider()
    _service(provider).run(config_path)
    root = tmp_path / "workspace" / "native-test"
    checkpoint = json.loads(
        (root / "artifacts" / "native-generation-checkpoint.json").read_text(encoding="utf-8")
    )
    slot = next(iter(checkpoint["slots"].values()))
    phases = root / "artifacts" / "generations" / "generation-0000" / "slot-00"

    assert len(provider.calls) == 3
    assert slot["status"] == "invalid"
    assert slot["repairs"] == 2
    assert slot["remaining_repairs"] == 0
    assert len(slot["repair_idempotency_keys"]) == 2
    assert (phases / "repair-01" / "turn-manifest.json").is_file()
    assert (phases / "repair-02" / "turn-manifest.json").is_file()


class _CrashAfterRepairAssessmentCoordinator(GenerationCoordinator):
    def _assess(
        self,
        request: Any,
        raw: ProviderResult,
        *,
        repair: bool = False,
    ) -> Any:
        result = super()._assess(request, raw, repair=repair)
        if repair:
            raise KeyboardInterrupt
        return result


class CrashAfterFailedRepairEvidenceEngine:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        provider: Any,
        *,
        config: Any,
        layout: Any,
        **_: Any,
    ) -> Mapping[str, Any]:
        self.calls += 1
        coordinator_type = (
            _CrashAfterRepairAssessmentCoordinator if self.calls == 1 else GenerationCoordinator
        )
        generation = coordinator_type(
            provider,
            config=GenerationConfig(
                campaign_id=config.exp_id,
                generations=config.search.max_generations,
                population_size=config.search.population_size,
                concurrency=config.model.concurrency,
                max_model_turns=config.search.max_model_turns,
                max_repairs=config.model.max_repairs,
                model=config.model.name,
                effort=config.model.effort,
                checkpoint_path=layout.artifacts / "native-generation-checkpoint.json",
            ),
        ).run(resume=True)
        return {"status": generation.status, "generation": len(generation.generations)}


def test_resume_reuses_durable_failed_repair_evidence(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=2,
        max_repairs=1,
    )
    provider = InvalidRepairProvider()
    engine = CrashAfterFailedRepairEvidenceEngine()
    service = _service(provider, engine=engine)

    with pytest.raises(KeyboardInterrupt):
        service.run(config_path)
    root = tmp_path / "workspace" / "native-test"
    interrupted_state = ExperimentStateStore(root / "state.sqlite3")
    try:
        assert interrupted_state.counts()["provider_turns"] == 2
        assert interrupted_state.cumulative()["total_tokens"] == 4
    finally:
        interrupted_state.close()

    resumed = service.run(config_path)
    final_state = ExperimentStateStore(root / "state.sqlite3")
    try:
        assert final_state.counts()["provider_turns"] == 2
        assert final_state.cumulative()["total_tokens"] == 4
    finally:
        final_state.close()
    checkpoint = json.loads(
        (root / "artifacts" / "native-generation-checkpoint.json").read_text(encoding="utf-8")
    )

    assert resumed["state"] == "exhausted"
    assert len(provider.calls) == 2
    assert {item["status"] for item in checkpoint["slots"].values()} == {"invalid"}


def test_native_prompts_embed_exact_validator_contract_and_repair_budget(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=2,
        max_repairs=1,
    )
    provider = RepairProvider()
    _service(provider).run(config_path)

    contract = render_policy_validator_contract(SandboxLimits(), scientific=True)
    assert len(provider.calls) == 2
    assert all(contract in request["prompt"] for request in provider.calls)
    assert VALIDATOR_VERSION in contract
    assert ", ".join(f"`{name}`" for name in sorted(SAFE_BUILTINS)) in contract
    repair = provider.calls[1]
    assert repair["repair_attempt"] == 1
    assert repair["remaining_repairs"] == 0
    assert "Repair attempt 1 of 1; 0 repairs remain after this attempt." in repair["prompt"]
    assert INVALID_SOURCE.strip() in repair["prompt"]
    assert "syntax_error" in repair["prompt"]


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (
            "def helper():\n    return 1\n\ndef priority(ctx, proposal):\n    return helper()\n",
            "top_level_contract",
        ),
        (
            "def priority(ctx, proposal):\n    _score = 1\n    return _score\n",
            "private_name",
        ),
        (
            "def priority(ctx, proposal):\n    return proposal.get('k')\n",
            "forbidden_call",
        ),
        (
            "def priority(ctx, proposal):\n    return sorted(proposal['k'])\n",
            "forbidden_call",
        ),
    ],
)
def test_validator_contract_failures_use_at_most_configured_repairs(
    tmp_path: Path,
    source: str,
    expected_code: str,
) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=2,
        max_repairs=1,
    )
    provider = InvalidRepairProvider(source)
    _service(provider).run(config_path)
    checkpoint = json.loads(
        (
            tmp_path
            / "workspace"
            / "native-test"
            / "artifacts"
            / "native-generation-checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    slot = next(iter(checkpoint["slots"].values()))
    codes = {item["code"] for item in slot["errors"]}

    assert len(provider.calls) == 2
    assert slot["status"] == "invalid"
    assert expected_code in codes


class InterruptingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        provider: Any,
        *,
        config: Any,
        on_generation: Any,
        layout: Any,
        **_: Any,
    ) -> Mapping[str, Any]:
        self.calls += 1

        def select(*args: Any) -> Any:
            selected = on_generation(*args)
            if self.calls == 1:
                raise KeyboardInterrupt
            return selected

        generation = GenerationCoordinator(
            provider,
            config=GenerationConfig(
                campaign_id=config.exp_id,
                generations=config.search.max_generations,
                population_size=config.search.population_size,
                concurrency=config.model.concurrency,
                max_model_turns=config.search.max_model_turns,
                max_repairs=config.model.max_repairs,
                model=config.model.name,
                effort=config.model.effort,
                checkpoint_path=layout.artifacts / "native-generation-checkpoint.json",
            ),
            selection_callback=select,
        ).run(resume=True)
        return {"status": generation.status, "generation": len(generation.generations)}


def test_interrupt_resume_reuses_durable_turn_and_evaluation(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=1,
        max_repairs=0,
    )
    provider = RecordingProvider()
    engine = InterruptingEngine()
    service = _service(provider, engine=engine)

    with pytest.raises(KeyboardInterrupt):
        service.run(config_path)
    interrupted = experiment_status(config_path)
    assert interrupted["state"] == "interrupted"
    assert len(provider.calls) == 1

    resumed = service.run(config_path)
    status = experiment_status(config_path)
    assert resumed["state"] == "exhausted"
    assert status["state"] == "exhausted"
    assert len(provider.calls) == 1
    assert status["provider_turns"] == 1
    assert status["unique_candidate_count"] == 1


def test_charged_failed_turn_is_retained_without_replacement_call(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        max_turns=1,
        max_repairs=1,
    )
    provider = ChargedFailureProvider()
    service = _service(provider)

    first = service.run(config_path)
    second = service.run(config_path)
    config = load_experiment_config(config_path)
    layout = ExperimentLayout.from_config(config)
    state = ExperimentStateStore(layout.state)
    try:
        row = state.connection.execute("SELECT * FROM provider_turns LIMIT 1").fetchone()
        assert row is not None
        usage = json.loads(str(row["usage_json"]))
        assert row["state"] == "failed"
        assert usage["quality"] == "partial"
        assert usage["totalTokens"] == 12_000
        assert state.cumulative()["total_tokens"] == 12_000
    finally:
        state.close()
    assert first["state"] == "exhausted"
    assert second["stop_reason"] == "max_model_turns"
    assert provider.calls == 1


def test_uncharged_infrastructure_failure_retries_same_generation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "native-checkpoint.json"

    class InfrastructureFailureProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _request: Mapping[str, Any]) -> Mapping[str, Any]:
            self.calls += 1
            return {
                "status": "infrastructure",
                "accepted": False,
                "charged": False,
                "uncharged": True,
                "content": False,
                "usage": {"totalTokens": 0},
                "error": "pre-request failure",
            }

    failed_provider = InfrastructureFailureProvider()
    first = GenerationCoordinator(
        failed_provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=1,
            max_repairs=0,
            checkpoint_path=checkpoint,
        ),
    ).run()
    retained = json.loads(checkpoint.read_text(encoding="utf-8"))

    assert first.status == "infrastructure_failed"
    assert first.summary["completed_generation_count"] == 0
    assert retained["next_generation"] == 0

    successful_provider = RecordingProvider()
    second = GenerationCoordinator(
        successful_provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=1,
            max_repairs=0,
            checkpoint_path=checkpoint,
        ),
    ).run()

    assert second.status == "completed"
    assert second.summary["first_generation"] == 0
    assert second.summary["completed_generation_count"] == 1
    assert len(successful_provider.calls) == 1


def test_stale_retry_attempt_does_not_rewind_completed_generation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "native-checkpoint.json"
    first_provider = RecordingProvider()
    GenerationCoordinator(
        first_provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=2,
            max_repairs=0,
            checkpoint_path=checkpoint,
        ),
        selection_callback=lambda *_args: {"slot-00": "g0000-slot-00"},
    ).run()
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    accepted = next(
        value for value in saved["slots"].values() if value.get("status") == "accepted"
    )
    saved["slots"]["stale-retry"] = {
        **accepted,
        "status": "repair_pending",
        "candidate": None,
    }
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")

    resumed_provider = RecordingProvider()
    result = GenerationCoordinator(
        resumed_provider,
        config=GenerationConfig(
            generations=2,
            population_size=1,
            concurrency=1,
            max_model_turns=4,
            max_repairs=0,
            checkpoint_path=checkpoint,
        ),
    ).run()

    assert result.summary["first_generation"] == 1
    assert len(resumed_provider.calls) == 1


def test_resume_revalidates_repair_pending_source_under_current_limits(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "native-checkpoint.json"
    retained_provider = RecordingProvider()
    restrictive = GenerationCoordinator(
        retained_provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=1,
            max_repairs=1,
            checkpoint_path=checkpoint,
            sandbox_limits=SandboxLimits(max_ast_nodes=2),
        ),
    )
    request = restrictive.build_request(0, "slot-00", "parent-0-slot-00")
    retained_response = retained_provider.generate(request.as_dict())
    pending = restrictive.run_request(
        request,
        allow_repair=False,
        retained_result=retained_response,
    )
    assert pending.status == "repair_pending"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "mforge.experiment.generation.v2",
                "campaign_id": "native",
                "slots": {request.idempotency_key: pending.as_dict()},
                "callbacks": {},
                "next_generation": 0,
                "model_turns_used": 1,
            }
        ),
        encoding="utf-8",
    )

    class NoCallProvider:
        def generate(self, _request: Mapping[str, Any]) -> Mapping[str, Any]:
            raise AssertionError("retained source must be revalidated without a model call")

    resumed = GenerationCoordinator(
        NoCallProvider(),
        config=GenerationConfig(
            campaign_id="native",
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=2,
            prior_model_turns=1,
            max_repairs=1,
            checkpoint_path=checkpoint,
            sandbox_limits=SandboxLimits(max_ast_nodes=1000),
        ),
    ).run()

    assert resumed.status == "completed"
    assert resumed.generations[0][0].status == "accepted"
    assert resumed.generations[0][0].candidate is not None


def test_resume_parent_assignments_match_retained_generation(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.sqlite3"
    ExperimentStateStore.initialize(
        state_path,
        exp_id="resume-parents",
        lock_hash="0" * 64,
        root=tmp_path,
    )
    checkpoint = tmp_path / "native-checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": "mforge.experiment.generation.v2",
                "campaign_id": "resume-parents",
                "slots": {
                    "old-slot": {
                        "generation": 3,
                        "slot": "slot-00",
                        "parent_id": "g0002-slot-04",
                        "status": "accepted",
                    },
                    "new-slot": {
                        "generation": 3,
                        "slot": "slot-00",
                        "parent_id": "native-baseline",
                        "status": "failed",
                    },
                },
                "callbacks": {},
            }
        ),
        encoding="utf-8",
    )
    original = {
        "slot-00": "g0002-slot-04",
        "slot-01": "g0002-slot-06",
    }
    with ExperimentStateStore(state_path) as store:
        store.write_event(
            "selection_completed",
            {"generation": 2, "selected_parents": original},
        )
        store.write_event(
            "selection_completed",
            {
                "generation": 2,
                "selected_parents": {
                    "slot-00": "native-baseline",
                    "slot-01": "native-baseline",
                },
            },
        )

        recovered = _resume_parent_assignments(checkpoint, store)

    assert recovered == {3: original}


def test_unbounded_generation_canary_crosses_finite_defaults(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    completed_generations: list[int] = []

    def select(generation: int, *_args: object) -> None:
        completed_generations.append(generation)

    result = GenerationCoordinator(
        provider,
        config=GenerationConfig(
            generations=None,
            population_size=1,
            concurrency=1,
            max_model_turns=None,
            max_repairs=0,
            checkpoint_path=tmp_path / "unbounded-checkpoint.json",
        ),
        selection_callback=select,
        budget_exhausted=lambda: len(completed_generations) >= 8,
    ).run()

    assert completed_generations == list(range(8))
    assert len(provider.calls) == 8
    assert result.status == "budget_exhausted"
    assert result.summary["stop_reason"] == "wall_seconds"


def test_repair_infrastructure_failure_retries_same_request(
    tmp_path: Path,
) -> None:
    class FlakyRepairProvider(RecordingProvider):
        def __init__(self) -> None:
            super().__init__(INVALID_SOURCE)
            self.repair_attempts = 0

        def repair(
            self, request: Mapping[str, Any], diagnostics: tuple[Mapping[str, Any], ...]
        ) -> Mapping[str, Any]:
            self.repair_attempts += 1
            if self.repair_attempts <= 2:
                self.calls.append(dict(request))
                return {
                    "status": "infrastructure",
                    "accepted": False,
                    "charged": False,
                    "uncharged": True,
                    "content": False,
                    "usage": {"totalTokens": 0},
                    "error": "temporary app-server outage",
                }
            self.source = VALID_SOURCE
            return self.generate(request)

    provider = FlakyRepairProvider()
    result = GenerationCoordinator(
        provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=4,
            max_repairs=1,
            checkpoint_path=tmp_path / "native-checkpoint.json",
            infrastructure_retry_limit=2,
            infrastructure_retry_backoff_seconds=0,
        ),
        retry_infrastructure=True,
    ).run()

    assert result.status == "completed"
    assert provider.repair_attempts == 3
    assert [request["phase"] for request in provider.calls] == [
        "initial",
        "repair",
        "repair",
        "repair",
    ]
    assert provider.calls[1]["idempotency_key"] == provider.calls[2]["idempotency_key"]
    assert provider.calls[2]["idempotency_key"] == provider.calls[3]["idempotency_key"]


def test_infrastructure_retry_limit_bounds_initial_attempts(tmp_path: Path) -> None:
    class AlwaysFailingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _request: Mapping[str, Any]) -> Mapping[str, Any]:
            self.calls += 1
            return {
                "status": "infrastructure",
                "accepted": False,
                "charged": False,
                "uncharged": True,
                "content": False,
                "usage": {"totalTokens": 0},
                "error": "provider unavailable",
            }

    provider = AlwaysFailingProvider()
    result = GenerationCoordinator(
        provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=10,
            max_repairs=0,
            checkpoint_path=tmp_path / "native-checkpoint.json",
            infrastructure_retry_limit=2,
            infrastructure_retry_backoff_seconds=0,
        ),
        retry_infrastructure=True,
    ).run()

    assert result.status == "infrastructure_failed"
    assert provider.calls == 3


def test_repair_infrastructure_failure_resumes_same_attempt(
    tmp_path: Path,
) -> None:
    class RepairFailureProvider(RecordingProvider):
        def __init__(self, *, succeeds: bool) -> None:
            super().__init__(INVALID_SOURCE)
            self.succeeds = succeeds

        def repair(
            self, request: Mapping[str, Any], diagnostics: tuple[Mapping[str, Any], ...]
        ) -> Mapping[str, Any]:
            self.calls.append(dict(request))
            if not self.succeeds:
                return {
                    "status": "infrastructure",
                    "accepted": False,
                    "charged": False,
                    "uncharged": True,
                    "content": False,
                    "usage": {"totalTokens": 0},
                    "error": "temporary timeout",
                }
            self.source = VALID_SOURCE
            return self.generate(request)

    checkpoint = tmp_path / "native-checkpoint.json"
    first_provider = RepairFailureProvider(succeeds=False)
    first = GenerationCoordinator(
        first_provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=4,
            max_repairs=1,
            checkpoint_path=checkpoint,
            infrastructure_retry_limit=1,
            infrastructure_retry_backoff_seconds=0,
        ),
        retry_infrastructure=True,
    ).run()
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    repair_state = next(
        value for value in saved["slots"].values() if value.get("status") == "repair_running"
    )

    second_provider = RepairFailureProvider(succeeds=True)
    second = GenerationCoordinator(
        second_provider,
        config=GenerationConfig(
            generations=1,
            population_size=1,
            concurrency=1,
            max_model_turns=4,
            max_repairs=1,
            checkpoint_path=checkpoint,
            infrastructure_retry_limit=1,
            infrastructure_retry_backoff_seconds=0,
        ),
        retry_infrastructure=True,
    ).run()

    assert first.status == "infrastructure_failed"
    assert repair_state["repairs"] == 1
    assert second.status == "completed"
    assert [request["phase"] for request in second_provider.calls] == ["repair", "repair"]
    assert second_provider.calls[0]["idempotency_key"] == repair_state["request"]["idempotency_key"]


def test_native_wall_budget_is_resumable_not_an_interrupt(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        population=8,
        max_turns=2,
        max_repairs=0,
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "wall_seconds = 30", "wall_seconds = 0.001"
        ),
        encoding="utf-8",
    )

    def engine(provider: Any, **_: Any) -> Mapping[str, Any]:
        time.sleep(0.02)
        provider.generate(
            {
                "generation": 0,
                "slot": "slot-00",
                "phase": "initial",
                "idempotency_key": "budget-test",
                "prompt": "budget test",
            }
        )
        return {"status": "completed"}

    result = _service(
        RecordingProvider(),
        engine=engine,
    ).run(config_path)

    assert result["state"] == "idle"
    assert result["stop_reason"] == "session_wall_seconds"


def test_native_hourly_token_limit_stops_and_resumes_without_new_turns(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "experiment.toml",
        workspace=tmp_path / "workspace",
        population=2,
        max_turns=10,
        max_repairs=0,
        hourly_token_limit=2,
    )
    provider = RecordingProvider()

    def engine(wrapped: Any, **_: Any) -> Mapping[str, Any]:
        for index in range(2):
            wrapped.generate(
                {
                    "generation": 0,
                    "slot": f"slot-{index:02d}",
                    "phase": "initial",
                    "idempotency_key": f"hourly-{index}",
                    "prompt": "hourly limit test",
                }
            )
        raise AssertionError("the second provider turn must not start")

    service = _service(provider, engine=engine)
    first = service.run(config_path)
    second = service.run(config_path)
    status = experiment_status(config_path)
    events_path = (
        tmp_path
        / "workspace"
        / "native-test"
        / "artifacts"
        / "sessions"
        / "session-000001"
        / "events.jsonl"
    )
    event_types = [
        json.loads(line)["event_type"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert first["state"] == "idle"
    assert first["stop_reason"] == "hourly_token_limit"
    assert second["state"] == "idle"
    assert second["stop_reason"] == "hourly_token_limit"
    assert len(provider.calls) == 1
    assert status["hourly_token_limit"] == 2
    assert status["hourly_tokens_used"] == 2
    assert status["hourly_tokens_remaining"] == 0
    assert status["hourly_limit_reached"] is True
    assert status["hourly_retry_after"] is not None
    assert event_types.index("checkpoint_written") < event_types.index(
        "hourly_token_session_stopped"
    )
