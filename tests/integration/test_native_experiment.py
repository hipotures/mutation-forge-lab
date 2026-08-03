from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import threading
from collections.abc import Mapping
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
from mutation_forge.experiment.native import NativeExperimentAdapter
from mutation_forge.experiment.service import ExperimentService
from mutation_forge.experiment.state import ExperimentStateStore
from mutation_forge.experiment.status import experiment_status
from mutation_forge.output.rich_live import RichLiveSink
from mutation_forge.sandbox.contracts import VALIDATOR_VERSION, SandboxLimits
from mutation_forge.sandbox.validation import (
    SAFE_BUILTINS,
    render_policy_validator_contract,
)

VALID_SOURCE = "def priority(ctx, proposal):\n    return 0\n"
INVALID_SOURCE = "def priority(ctx, proposal)\n    return 0\n"


def test_native_output_schema_uses_app_server_supported_keywords() -> None:
    schema = json.loads(
        Path("configs/native/generated-policy.schema.json").read_text(encoding="utf-8")
    )

    def contains_unique_items(value: object) -> bool:
        if isinstance(value, Mapping):
            return any(
                key == "uniqueItems" or contains_unique_items(item)
                for key, item in value.items()
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
) -> Path:
    path.write_text(
        f'''schema_version = "mforge.experiment.v1"
exp_id = "{exp_id}"
workspace = "{workspace.as_posix()}"
kind = "ranker-search"
preset = "heg-ranker-evolution-v1"

[run]
wall_seconds = 30

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
                "used_fields": ["proposal.k"],
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


def test_fresh_native_experiment_creates_workspace_and_next_run_starts_next_batch(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path / "experiment.toml", workspace=tmp_path / "workspace")
    provider = RecordingProvider()
    service = _service(provider)

    first = service.run(config_path)
    root = tmp_path / "workspace" / "native-test"
    state = ExperimentStateStore(root / "state.sqlite3")
    try:
        checkpoint = state.checkpoint()
        state.set_state(
            "completed",
            stop_reason="generation_limit",
            checkpoint=str(checkpoint["checkpoint_id"]) if checkpoint is not None else None,
        )
    finally:
        state.close()
    second = service.run(config_path)

    assert first["state"] == "idle"
    assert second["state"] == "idle"
    assert second["stop_reason"] == "generation_batch_completed"
    assert [request["generation"] for request in provider.calls] == [0, 1]
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
    assert status["state"] == "idle"
    assert status["provider_turns"] == 2
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
    assert result["state"] == "idle"


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

    assert result["state"] == "idle"
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

    assert result["state"] == "idle"
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
        (root / "artifacts" / "native-generation-checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    events = [
        json.loads(line)
        for line in (
            root / "artifacts" / "sessions" / "session-000001" / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    slot_states = list(checkpoint["slots"].values())
    state = ExperimentStateStore(root / "state.sqlite3")
    try:
        assert state.counts()["provider_turns"] == 4
        assert state.cumulative()["total_tokens"] == 8
    finally:
        state.close()

    assert first["state"] == "idle"
    assert second["stop_reason"] == "generation_batch_completed"
    assert [request["generation"] for request in provider.calls] == [0, 0, 1, 1]
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
        (root / "artifacts" / "native-generation-checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    slot = next(iter(checkpoint["slots"].values()))
    phases = (
        root
        / "artifacts"
        / "generations"
        / "generation-0000"
        / "slot-00"
    )

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
            _CrashAfterRepairAssessmentCoordinator
            if self.calls == 1
            else GenerationCoordinator
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
        (root / "artifacts" / "native-generation-checkpoint.json").read_text(
            encoding="utf-8"
        )
    )

    assert resumed["state"] == "idle"
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

    contract = render_policy_validator_contract(SandboxLimits())
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
            "def helper():\n    return 1\n\n"
            "def priority(ctx, proposal):\n    return helper()\n",
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
    assert resumed["state"] == "idle"
    assert status["state"] == "idle"
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
        assert state.cumulative()["total_tokens"] == 24_000
    finally:
        state.close()
    assert first["state"] == "idle"
    assert second["stop_reason"] == "generation_batch_completed"
    assert provider.calls == 2


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
