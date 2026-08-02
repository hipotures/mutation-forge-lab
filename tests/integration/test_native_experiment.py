from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.backends.toy import ToyBackend
from mutation_forge.experiment.config import load_experiment_config
from mutation_forge.experiment.generation import GenerationConfig, GenerationCoordinator
from mutation_forge.experiment.layout import ExperimentLayout
from mutation_forge.experiment.native import NativeExperimentAdapter
from mutation_forge.experiment.service import ExperimentService
from mutation_forge.experiment.state import ExperimentStateStore
from mutation_forge.experiment.status import experiment_status

VALID_SOURCE = "def priority(ctx, proposal):\n    return 0\n"
INVALID_SOURCE = "def priority(ctx, proposal)\n    return 0\n"


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
            response = {"source": self.source}
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


def test_fresh_native_experiment_creates_native_workspace_and_noop_is_idempotent(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path / "experiment.toml", workspace=tmp_path / "workspace")
    provider = RecordingProvider()
    service = _service(provider)

    first = service.run(config_path)
    second = service.run(config_path)
    root = tmp_path / "workspace" / "native-test"

    assert first["state"] == "completed"
    assert second["stop_reason"] == "already_completed"
    assert len(provider.calls) == 1
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
    assert status["state"] == "completed"
    assert status["provider_turns"] == 1
    assert status["unique_candidate_count"] == 1
    assert status["best_program_id"] == "g0000-slot-00"


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

    assert result["state"] == "completed"
    assert len(provider.calls) == 2
    assert provider.max_active == 1
    assert {request["model"] for request in provider.calls} == {"gpt-5.6-luna"}
    assert {request["effort"] for request in provider.calls} == {"high"}
    assert all(request["system_prompt"] for request in provider.calls)
    assert all(request["output_schema"]["type"] == "object" for request in provider.calls)
    assert lock["model"]["concurrency"] == 1
    assert lock["search"]["population_size"] == 2
    assert lock["search"]["max_generations"] == 1
    assert lock["search"]["max_model_turns"] == 3
    evaluation = next((root / "artifacts" / "evaluations" / "development").glob("*.json"))
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    assert payload["settings"]["workers"] == 1
    assert payload["settings"]["thread_count"] == 1


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

    assert result["state"] == "completed"
    assert [request["phase"] for request in provider.calls] == ["initial", "repair"]
    assert json.loads((initial / "validation.json").read_text(encoding="utf-8"))["valid"] is False
    assert json.loads((repair / "validation.json").read_text(encoding="utf-8"))["valid"] is True
    assert (initial / "turn-manifest.json").is_file()
    assert (repair / "turn-manifest.json").is_file()
    assert (repair / "behavior.json").is_file()


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
    assert resumed["state"] == "completed"
    assert status["state"] == "completed"
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
    assert first["state"] == "completed"
    assert second["stop_reason"] == "already_completed"
    assert provider.calls == 1
