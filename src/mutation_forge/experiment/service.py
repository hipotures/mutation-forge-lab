"""Create-or-continue service for experiment workspaces."""

from __future__ import annotations

import inspect
import json
import shutil
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .artifacts import ArtifactIncompleteError, TurnArtifactStore
from .checkpoints import CheckpointStore
from .config import ExperimentConfig, load_experiment_config
from .layout import ExperimentLayout, WorkspaceError
from .lock import LockError, build_lock, load_lock, verify_lock
from .sessions import SessionContext, SessionManager
from .state import ExperimentStateStore, StateError


class ExperimentAdapter(Protocol):
    """Boundary implemented by the existing generation/evaluation engine."""

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class SessionBudget:
    wall_seconds: float
    started_monotonic: float

    @property
    def deadline(self) -> float:
        return self.started_monotonic + self.wall_seconds

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.deadline


class NullExperimentAdapter:
    """Explicit test adapter; never selected by the production service."""

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any]:
        del config, layout, state, session
        return {"state": "idle", "stop_reason": "budget_exhausted"}


class LegacyStage4Adapter:
    """Adapter around the existing Stage 4 generation/evaluation workflow.

    The engine and provider are injectable so tests can exercise continuation
    without a live account.  Normal runs resolve the frozen Stage 4 config
    from the preset and use the installed local Codex profile; credentials are
    never copied into the experiment workspace.
    """

    def __init__(self, *, provider: Any | None = None, engine: Any | None = None) -> None:
        self.provider = provider
        self.engine = engine

    def preflight(self, config: ExperimentConfig) -> Mapping[str, Any]:
        """Reject an unrunnable or semantically different Stage 4 invocation.

        The public experiment schema currently exposes a wider search surface
        than the frozen Stage 4 engine implements.  Until there is a native
        adapter, accepting different values would make the lock misleading.
        """

        from mutation_forge.stage4.commands import (
            _load_search_freeze,
            campaign_root,
            doctor,
        )
        from mutation_forge.stage4.config import load_stage4_config

        stage4_path = _resolve_stage4_config(config)
        stage4 = load_stage4_config(stage4_path)
        _require_stage4_compatibility(config, stage4)
        if self.engine is not None:
            return {"stage4_config": str(stage4_path), "injected_engine": True}
        freeze = campaign_root(stage4) / "search-freeze.json"
        if not freeze.is_file():
            raise WorkspaceError(
                "Stage 4 preset is not runnable: its frozen search metadata is missing at "
                f"{freeze}; run the private Stage 4 freeze workflow first"
            )
        try:
            _load_search_freeze(stage4)
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            raise WorkspaceError(
                "Stage 4 preset is not runnable: its frozen search metadata is invalid"
            ) from error
        auth_json = Path.home() / ".codex" / "auth.json"
        result = doctor(
            stage4_path,
            auth_json=auth_json if auth_json.is_file() else None,
            check_auth=True,
            write=False,
        )
        if (
            result.get("status") != "completed"
            or cast(Mapping[str, Any], result.get("auth", {})).get("authenticated") is not True
        ):
            raise WorkspaceError("Stage 4 App Server doctor did not authenticate a READY profile")
        return {"stage4_config": str(stage4_path), "doctor": dict(result)}

    def run(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any]:
        from mutation_forge.stage4.commands import evolve
        from mutation_forge.stage4.config import load_stage4_config

        stage4_config = _resolve_stage4_config(config)
        frozen_stage4 = load_stage4_config(stage4_config)
        engine = self.engine or evolve
        provider = self.provider or _build_local_stage4_provider(layout)
        run_override = layout.artifacts
        if self.engine is None:
            _prepare_stage4_workspace(stage4_config, run_override)
        adapter_provider = _WorkspaceStage4Provider(
            provider, layout, state, session, sandbox_limits=frozen_stage4.sandbox
        )

        def observe(event: Mapping[str, Any]) -> None:
            event_type = str(event.get("event", "adapter_event"))
            state.write_event(event_type, event, session_id=session.session_id)
            if session.budget_exhausted() and event_type == "generation_completed":
                raise _SessionBudgetExpired

        engine_kwargs: dict[str, Any] = {
            "provider": adapter_provider,
            # Compatibility was checked before workspace creation; both
            # values are now the same frozen scientific identity.
            "concurrency": frozen_stage4.model.concurrency,
            "resume": True,
            "observer": observe,
        }
        try:
            engine_parameters: Mapping[str, inspect.Parameter]
            try:
                engine_parameters = inspect.signature(engine).parameters
            except (TypeError, ValueError):
                engine_parameters = {}
            if "run_override" in engine_parameters:
                engine_kwargs["run_override"] = run_override
            result = engine(stage4_config, **engine_kwargs)
        except _SessionBudgetExpired:
            indexed = _index_legacy_run(run_override, state)
            session.candidates_created += indexed["candidates"]
            session.evaluations_completed += indexed["evaluations"]
            generation = _stage4_generation(run_override)
            return {
                "state": "idle",
                "stop_reason": "budget_exhausted",
                "generation": generation,
                "provider_turns": session.provider_turns_completed,
            }
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()

        if not isinstance(result, Mapping):
            raise RuntimeError("Stage 4 adapter returned a non-object result")
        run_path = result.get("run")
        if isinstance(run_path, str) and Path(run_path).is_dir():
            destination = layout.artifacts
            if Path(run_path).resolve() != destination.resolve():
                destination.mkdir(parents=True, exist_ok=True)
                for source in Path(run_path).iterdir():
                    target = destination / source.name
                    if source.is_dir():
                        shutil.copytree(source, target, dirs_exist_ok=True)
                    elif source.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            indexed = _index_legacy_run(destination, state)
            session.candidates_created += indexed["candidates"]
            session.evaluations_completed += indexed["evaluations"]
        status = str(result.get("status", "completed"))
        return {
            "state": "completed" if status == "completed" else "idle",
            "stop_reason": "engine_completed" if status == "completed" else "budget_exhausted",
            "generation": int(
                cast(Mapping[str, Any], result.get("generation", {})).get(
                    "generation_count", 0
                )
            )
            if isinstance(result.get("generation"), Mapping)
            else 0,
            "provider_turns": session.provider_turns_completed,
            "result": dict(result),
        }


class _SessionBudgetExpired(Exception):
    pass


def _resolve_stage4_config(config: ExperimentConfig) -> Path:
    configured = config.raw.get("legacy_stage4_config")
    if isinstance(configured, str) and configured:
        path = Path(configured)
        return (config.source_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if config.preset == "heg-ranker-evolution-v1":
        return Path(__file__).resolve().parents[3] / "configs" / "stage4-search.toml"
    raise WorkspaceError(
        "experiment preset has no Stage 4 adapter configuration; "
        "set legacy_stage4_config or use a supported preset"
    )


def _require_stage4_compatibility(config: ExperimentConfig, stage4: Any) -> None:
    """Fail closed when public config would not be executed by frozen Stage 4."""

    expected: dict[str, object] = {
        "kind": "ranker-search",
        "preset": "heg-ranker-evolution-v1",
        "model.provider": "codex",
        "model.name": stage4.model.name,
        "model.effort": stage4.model.effort,
        "model.concurrency": stage4.model.concurrency,
        "model.max_repairs": stage4.model.max_repairs,
        "search.population_size": stage4.model.slots,
        "search.max_generations": stage4.model.generations,
        "search.max_model_turns": stage4.model.max_accepted_turns,
        "search.selection": "elite-diversity",
        "evaluation.orders": stage4.experiment.orders,
        "evaluation.graph_seeds": stage4.experiment.graph_seeds,
        "evaluation.policy_seeds": stage4.experiment.policy_seeds,
        "evaluation.horizon": stage4.experiment.horizon,
        "evaluation.proposal_pool_size": stage4.stage2b.pool.pool_size,
        "evaluation.baselines": ("random", "structural"),
        "evaluation.replay": True,
        "resources.workers": stage4.limits.max_evaluation_workers,
        "resources.thread_count": stage4.limits.thread_count,
    }
    actual: dict[str, object] = {
        "kind": config.kind,
        "preset": config.preset,
        "model.provider": config.model.provider,
        "model.name": config.model.name,
        "model.effort": config.model.effort,
        "model.concurrency": config.model.concurrency,
        "model.max_repairs": config.model.max_repairs,
        "search.population_size": config.search.population_size,
        "search.max_generations": config.search.max_generations,
        "search.max_model_turns": config.search.max_model_turns,
        "search.selection": config.search.selection,
        "evaluation.orders": config.evaluation.orders,
        "evaluation.graph_seeds": config.evaluation.graph_seeds,
        "evaluation.policy_seeds": config.evaluation.policy_seeds,
        "evaluation.horizon": config.evaluation.horizon,
        "evaluation.proposal_pool_size": config.evaluation.proposal_pool_size,
        "evaluation.baselines": config.evaluation.baselines,
        "evaluation.replay": config.evaluation.replay,
        "resources.workers": config.resources.workers,
        "resources.thread_count": config.resources.thread_count,
    }
    mismatches = [
        f"{name}={actual[name]!r} (frozen Stage 4 requires {expected[name]!r})"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise WorkspaceError(
            "experiment.toml is incompatible with preset heg-ranker-evolution-v1: "
            + "; ".join(mismatches)
        )


def _stage4_generation(run: Path) -> int:
    try:
        value = json.loads((run / "generation-checkpoint.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0
    if not isinstance(value, Mapping):
        return 0
    generations = [
        int(slot.get("generation", 0))
        for slot in value.get("slots", {}).values()
        if isinstance(slot, Mapping) and isinstance(slot.get("generation"), int)
    ] if isinstance(value.get("slots"), Mapping) else []
    return max(generations, default=0)


def _prepare_stage4_workspace(config_path: Path, destination: Path) -> None:
    """Bring only the immutable Stage 4 freeze into the experiment root.

    Stage 4's scientific inputs remain owned by its frozen configuration.  The
    experiment owns the mutable campaign/checkpoint/evidence directory.  If a
    prior private Stage 4 campaign exists, copy its signed freeze metadata once
    so the legacy engine can verify the same identities under the workspace.
    Missing freeze metadata is a hard precondition failure, never an idle
    success that pretends a search happened.
    """

    from mutation_forge.stage4.commands import campaign_root
    from mutation_forge.stage4.config import load_stage4_config

    stage4 = load_stage4_config(config_path)
    destination.mkdir(parents=True, exist_ok=True)
    target_freeze = destination / "search-freeze.json"
    if target_freeze.is_file():
        return
    source_root = campaign_root(stage4)
    source_freeze = source_root / "search-freeze.json"
    if not source_freeze.is_file():
        raise WorkspaceError(
            "Stage 4 preset is not runnable: its frozen search metadata is missing at "
            f"{source_freeze}; run the private Stage 4 freeze workflow first"
        )
    shutil.copy2(source_freeze, target_freeze)
    # Technical/authentication amendments are part of the signed freeze chain.
    # Copy them only when present; never copy mutable generations or provider
    # artifacts from the historical campaign.
    for path in source_root.glob("search-freeze-pre-amendment*.json"):
        shutil.copy2(path, destination / path.name)
    amendment = source_root / "post-live-amendment.json"
    if amendment.is_file():
        shutil.copy2(amendment, destination / amendment.name)


def _build_local_stage4_provider(layout: ExperimentLayout) -> Any:
    from mutation_forge.stage4.app_server import Stage4AppServerProvider

    auth = Path.home() / ".codex" / "auth.json"
    return Stage4AppServerProvider(
        auth_json=auth if auth.is_file() else None,
        artifact_dir=layout.artifacts / "generations",
        artifact_root=layout.artifacts,
    )


class _WorkspaceStage4Provider:
    """Route Stage 4 transport artifacts and idempotency into one workspace."""

    def __init__(
        self,
        provider: Any,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
        *,
        sandbox_limits: Any,
    ) -> None:
        self.provider = provider
        self.layout = layout
        self.state = state
        self.session = session
        self.sandbox_limits = sandbox_limits
        self.turns = TurnArtifactStore(layout.artifacts)

    @staticmethod
    def _phase(request: Mapping[str, Any]) -> str:
        phase = str(request.get("phase", "initial"))
        if phase == "initial":
            return phase
        if phase == "repair":
            return "repair-01"
        return phase

    def _payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(request)
        generation = int(request.get("generation", 0))
        slot = str(request.get("slot", "slot-00"))
        phase = self._phase(request)
        directory = self.layout.generation_slot_phase(
            generation,
            slot,
            phase,
        )
        value.update(
            {
                "artifact_dir": str(directory),
                "artifact_root": str(self.layout.artifacts),
                "artifact_prefix": slot,
            }
        )
        return value

    def _record(self, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        generation = int(request.get("generation", 0))
        slot = str(request.get("slot", "slot-00"))
        phase = self._phase(request)
        directory = self.layout.generation_slot_phase(
            generation,
            slot,
            phase,
        )
        result = self._with_validation_evidence(result)
        status = str(result.get("status", "completed"))
        usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else None
        key = self._key(request)
        if not directory.is_dir():
            if status == "completed":
                raise ArtifactIncompleteError(
                    f"completed provider turn has no artifact directory: {directory}"
                )
        elif (directory / "turn-manifest.json").exists():
            self.turns.verify_turn(directory)
        else:
            self.turns.record_existing_turn(
                directory,
                generation=generation,
                slot=slot,
                phase=phase,
                request=request,
                result=result,
            )
            self.turns.verify_turn(directory)

        # Commit logical completion only after the durable evidence package
        # has been assembled and its exact hashes have been verified.
        self.state.record_provider_turn(
            idempotency_key=key,
            generation=generation,
            slot=slot,
            phase=phase,
            state="completed" if status == "completed" else "failed",
            artifact_path=str(directory),
            usage=cast(Mapping[str, Any], usage or {}),
            provider_thread_id=(
                str(result["provider_thread_id"])
                if result.get("provider_thread_id") is not None
                else None
            ),
            provider_turn_id=(
                str(result["provider_turn_id"])
                if result.get("provider_turn_id") is not None
                else None
            ),
            error=str(result.get("error")) if result.get("error") else None,
        )
        self.session.provider_turns_attempted += 1
        if status == "completed":
            self.session.provider_turns_completed += 1
        if isinstance(usage, Mapping):
            total = usage.get("totalTokens")
            if isinstance(total, int) and not isinstance(total, bool):
                self.session.token_usage_delta += total

    def _with_validation_evidence(self, result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Persist the same validation/probe boundary before accepting a turn."""

        response = result.get("response")
        source = response.get("source") if isinstance(response, Mapping) else None
        if not isinstance(source, str):
            return result
        from mutation_forge.sandbox.validation import validate_policy
        from mutation_forge.stage4.generation import _behavior

        value = dict(result)
        value["canonical_response"] = dict(cast(Mapping[str, Any], response))
        value["provenance"] = {
            key: value.get(key)
            for key in (
                "provider_request_id",
                "provider_thread_id",
                "provider_turn_id",
                "model",
                "effort",
                "prompt_hashes",
                "appserver_doctor_sha256",
            )
        }
        validation = validate_policy(source, self.sandbox_limits)
        value["validation"] = validation.as_dict()
        value["identity"] = validation.identity.as_dict()
        value["validation_completed"] = True
        if validation.valid:
            try:
                behavior, telemetry = _behavior(source, self.sandbox_limits, 10_000)
            except Exception as error:
                behavior, telemetry = (
                    {"status": "failed", "error": f"{type(error).__name__}: {error}"},
                    {},
                )
            value["behavior"] = behavior
            value["worker_telemetry"] = telemetry
        return value

    @staticmethod
    def _key(request: Mapping[str, Any]) -> str:
        value = request.get("idempotency_key", request.get("request_idempotency_key"))
        if isinstance(value, str) and value:
            return value
        # A provider failure can happen before the generation engine has
        # assembled its normal request identity.  Keep that failure indexed
        # under a deterministic, non-empty primary key rather than merging
        # every pre-request failure into one SQLite row.
        return "pre-request:" + ":".join(
            (
                str(request.get("campaign_id", "experiment")),
                str(request.get("generation", 0)),
                str(request.get("slot", "slot-00")),
                str(request.get("phase", "initial")),
            )
        )

    def _record_failure(self, request: Mapping[str, Any], error: BaseException) -> None:
        evidence = getattr(error, "evidence", {})
        result: dict[str, Any] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        if isinstance(evidence, Mapping):
            result.update(dict(evidence))
        with suppress(Exception):
            self._record(request, result)

    def _retained_result(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Recover a completed provider envelope before issuing a duplicate call."""

        key = self._key(request)
        existing = self.state.provider_turn(key)
        directory = self.layout.generation_slot_phase(
            int(request.get("generation", 0)),
            str(request.get("slot", "slot-00")),
            self._phase(request),
        )
        manifest_path = directory / "turn-manifest.json"
        if existing is None and not manifest_path.is_file():
            return None
        if existing is not None and existing.get("state") != "completed":
            return None
        self.turns.verify_turn(directory)
        try:
            manifest = json.loads((directory / "turn-manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactIncompleteError(
                "completed provider turn has no readable manifest"
            ) from exc
        if not isinstance(manifest, Mapping) or manifest.get("request_idempotency_key") != key:
            raise ArtifactIncompleteError(
                "completed provider turn idempotency key does not match request"
            )
        if manifest.get("usage_final_exact") is not True:
            raise ArtifactIncompleteError("completed provider turn has non-exact usage")
        response_paths = sorted(directory.glob("*.response.json")) if directory.is_dir() else []
        for response_path in reversed(response_paths):
            try:
                raw = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(raw, Mapping):
                continue
            raw_status = raw.get("status")
            if raw_status is not None and raw_status not in {"completed", "success"}:
                continue
            response = raw.get("response", raw)
            response_text = raw.get("response_text")
            if not isinstance(response_text, str):
                text_path = response_path.with_name(
                    response_path.name.removesuffix(".response.json") + ".response.md"
                )
                try:
                    response_text = text_path.read_text(encoding="utf-8")
                except OSError:
                    response_text = None
            usage: Mapping[str, Any] = {}
            usage_path = response_path.with_name(
                response_path.name.removesuffix(".response.json") + ".usage.json"
            )
            try:
                value = json.loads(usage_path.read_text(encoding="utf-8"))
                if isinstance(value, Mapping):
                    usage = cast(Mapping[str, Any], value)
            except (OSError, UnicodeError, json.JSONDecodeError):
                value = raw.get("usage")
                if isinstance(value, Mapping):
                    usage = cast(Mapping[str, Any], value)
            result: dict[str, Any] = {
                "status": "completed",
                "accepted": True,
                "accepted_turn": True,
                "content": bool(response_text),
                "response": response,
                "response_text": response_text,
                "usage": dict(usage),
                "provider_request_id": raw.get("provider_request_id", raw.get("request_id")),
                "provider_thread_id": raw.get("provider_thread_id", raw.get("thread_id")),
                "provider_turn_id": raw.get("provider_turn_id", raw.get("turn_id")),
                "retained": True,
            }
            if existing is None:
                recovered = self.state.record_provider_turn(
                    idempotency_key=key,
                    generation=int(request.get("generation", 0)),
                    slot=str(request.get("slot", "slot-00")),
                    phase=self._phase(request),
                    state="completed",
                    artifact_path=str(directory),
                    usage=usage,
                    provider_thread_id=(
                        str(result["provider_thread_id"])
                        if result.get("provider_thread_id") is not None
                        else None
                    ),
                    provider_turn_id=(
                        str(result["provider_turn_id"])
                        if result.get("provider_turn_id") is not None
                        else None
                    ),
                )
                if recovered:
                    self.session.provider_turns_attempted += 1
                    self.session.provider_turns_completed += 1
                    total = usage.get("totalTokens")
                    if isinstance(total, int) and not isinstance(total, bool):
                        self.session.token_usage_delta += total
            return result
        return None

    def generate(self, request: Mapping[str, Any]) -> Any:
        retained = self._retained_result(request)
        if retained is not None:
            return retained
        if self.session.budget_exhausted():
            raise _SessionBudgetExpired
        payload = self._payload(request)
        try:
            value = self.provider.generate(payload)
        except BaseException as error:
            self._record_failure(request, error)
            raise
        result = value if isinstance(value, Mapping) else {"response": value}
        self._record(request, cast(Mapping[str, Any], result))
        return value

    def repair(self, request: Mapping[str, Any], diagnostics: Any) -> Any:
        retained = self._retained_result(request)
        if retained is not None:
            return retained
        if self.session.budget_exhausted():
            raise _SessionBudgetExpired
        payload = self._payload(request)
        payload["diagnostics"] = list(diagnostics)
        try:
            method = getattr(self.provider, "repair", None)
            if callable(method):
                value = method(payload, diagnostics)
            else:
                value = self.provider.generate(payload)
        except BaseException as error:
            self._record_failure(request, error)
            raise
        result = value if isinstance(value, Mapping) else {"response": value}
        self._record(request, cast(Mapping[str, Any], result))
        return value

    def load_retained_result(self, request: Mapping[str, Any]) -> Any:
        retained = self._retained_result(request)
        if retained is not None:
            return retained
        method = getattr(self.provider, "load_retained_result", None)
        if callable(method):
            value = method(self._payload(request))
            if isinstance(value, Mapping):
                self._record(request, cast(Mapping[str, Any], value))
            return value
        return None

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


def _index_legacy_run(run: Path, state: ExperimentStateStore) -> dict[str, int]:
    """Project the legacy filesystem authorities into experiment SQLite.

    Stage 4 deliberately keeps JSON archives and evaluation summaries as the
    scientific source of truth.  The experiment database is an operational
    index, so re-reading those immutable files after every resumed engine run
    is safe and makes status useful without rerunning scoring.
    """

    counts = {"candidates": 0, "evaluations": 0, "generation": 0}
    known_candidates = {
        str(row[0]) for row in state.connection.execute("SELECT candidate_id FROM candidates")
    }
    known_evaluations = {
        str(row[0]) for row in state.connection.execute("SELECT identity FROM evaluations")
    }
    archive_root = run / "archive"
    if archive_root.is_dir():
        try:
            from mutation_forge.stage4.archive import ProgramArchive

            archive = ProgramArchive(archive_root)
            records = archive.records()
        except (OSError, ValueError, TypeError):
            records = ()
        for record in records:
            metadata = {
                "normalized_ast_sha256": record.normalized_ast_sha256,
                "behavior_signature_sha256": record.behavior_signature_sha256,
                "validation_status": record.validation_status,
                "probe_status": record.probe_status,
                "smoke_10k_status": record.smoke_10k_status,
                "replay_status": record.replay_status,
                "fitness_status": record.fitness_status,
                "search_metrics": dict(record.search_metrics),
                "usage": dict(record.usage),
                "error": record.error,
            }
            state.record_candidate(
                record.program_id,
                source_sha256=record.source_sha256,
                archive_path=str(run / record.source_path) if record.source_path else None,
                generation=record.generation,
                slot=record.slot,
                status=("duplicate" if record.duplicate_of else "created"),
                metadata=metadata,
            )
            if record.program_id not in known_candidates:
                counts["candidates"] += 1
                known_candidates.add(record.program_id)
            counts["generation"] = max(counts["generation"], record.generation)

    for summary_path in sorted(run.rglob("*-summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(summary, Mapping)
            or summary.get("schema_version") != "stage4.evaluation.v1"
        ):
            continue
        candidate_key = summary.get("candidate_cache_key")
        pass_name = summary.get("pass")
        if not isinstance(candidate_key, str) or not isinstance(pass_name, str):
            continue
        identity = f"{candidate_key}:{pass_name}"
        compact = {
            key: value
            for key, value in summary.items()
            if key not in {"shards", "manifest_shards", "manifest_episode_ids"}
        }
        state.record_evaluation(
            identity,
            candidate_id=str(summary.get("candidate_id", "roster")),
            kind=pass_name,
            state="completed",
            result=compact,
        )
        if identity not in known_evaluations:
            counts["evaluations"] += 1
            known_evaluations.add(identity)
    return counts


class ExperimentService:
    def __init__(self, *, adapter: ExperimentAdapter | None = None) -> None:
        self.adapter = adapter or LegacyStage4Adapter()

    def run(self, config_path: str | Path = "experiment.toml") -> dict[str, Any]:
        config = load_experiment_config(config_path)
        layout = ExperimentLayout.from_config(config)
        created = not layout.root.exists()
        if created:
            preflight = self._preflight(config)
            lock = build_lock(config, layout, preflight=preflight)
            lock_hash = str(lock["immutable_config_sha256"])
            layout.initialize_atomic(
                config,
                lock_payload=lock,
                state_initializer=lambda state_path: ExperimentStateStore.initialize(
                    state_path,
                    exp_id=config.exp_id,
                    lock_hash=lock_hash,
                    root=layout.root,
                ),
            )
        else:
            layout.verify_root()
            lock = load_lock(layout.lock)
            verify_lock(lock, config, layout)
            self._verify_root_config(layout, config, lock)

        state = ExperimentStateStore(layout.state)
        checkpoints = CheckpointStore(layout.checkpoints)
        try:
            checkpoints.verify()
            self._verify_state_checkpoint(state, checkpoints)
            # A crash may leave new, fsynced files outside the last manifest.
            # Reconciliation accepts only those append-only additions; it
            # first verifies every previously committed digest.
            layout.reconcile_artifact_manifest()
            current_state = state.state()
            if current_state == "completed":
                return self._record_completed_session(config, layout, state)

            number = state.next_session_number()
            session_id = f"session-{number:06d}"
            state.acquire_owner(exp_id=config.exp_id, session_id=session_id)
            manager = SessionManager(layout, state)
            session: SessionContext | None = None
            try:
                session = manager.start(config)
                starting_checkpoint = checkpoints.latest()
                if starting_checkpoint is None:
                    first = checkpoints.save(
                        {
                            "experiment_id": config.exp_id,
                            "state": "running",
                            "generation": 0,
                            "completed_slots": [],
                            "provider_turns": [],
                            "evaluations": [],
                        }
                    )
                    state.record_checkpoint(
                        sequence=int(first["sequence"]),
                        checkpoint_id=str(first["checkpoint_id"]),
                        path=str(layout.checkpoint_path(int(first["sequence"]))),
                        sha256=str(first["checkpoint_sha256"]),
                        generation=0,
                        completed_slots=0,
                    )
                    session.starting_checkpoint = str(first["checkpoint_id"])
                else:
                    session.starting_checkpoint = str(starting_checkpoint["checkpoint_id"])
                manager.event(session, "session_started", checkpoint=session.starting_checkpoint)
                result = self._invoke_adapter(config, layout, state, session)
                outcome = self._normalize_outcome(result, session)
                final_checkpoint = checkpoints.save(
                    {
                        "experiment_id": config.exp_id,
                        "state": outcome["state"],
                        "generation": int(outcome.get("generation", 0)),
                        "completed_slots": list(outcome.get("completed_slots", [])),
                        "provider_turns": list(outcome.get("provider_turns", [])),
                        "evaluations": list(outcome.get("evaluations", [])),
                        "result": dict(outcome),
                    }
                )
                state.record_checkpoint(
                    sequence=int(final_checkpoint["sequence"]),
                    checkpoint_id=str(final_checkpoint["checkpoint_id"]),
                    path=str(layout.checkpoint_path(int(final_checkpoint["sequence"]))),
                    sha256=str(final_checkpoint["checkpoint_sha256"]),
                    generation=int(outcome.get("generation", 0)),
                    completed_slots=len(cast(list[Any], outcome.get("completed_slots", []))),
                )
                session.ending_checkpoint = str(final_checkpoint["checkpoint_id"])
                manager.event(session, "checkpoint_written", checkpoint=session.ending_checkpoint)
                state.set_state(
                    outcome["state"],
                    error=outcome.get("last_error"),
                    stop_reason=outcome.get("stop_reason"),
                    checkpoint=session.ending_checkpoint,
                )
                session_summary = manager.finish(
                    session,
                    state=outcome["state"],
                    stop_reason=str(outcome.get("stop_reason", "budget_exhausted")),
                    summary={**outcome, "result": outcome.get("result")},
                )
                layout.reconcile_artifact_manifest()
                return self._run_result(config, layout, state, session_summary, outcome)
            except BaseException as error:
                if session is not None:
                    session.stop_reason = (
                        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    )
                    with suppress(Exception):
                        manager.finish(
                            session,
                            state="interrupted"
                            if isinstance(error, KeyboardInterrupt)
                            else "failed",
                            stop_reason=session.stop_reason,
                            exit_status=130 if isinstance(error, KeyboardInterrupt) else 1,
                            summary={"last_error": f"{type(error).__name__}: {error}"},
                        )
                    with suppress(Exception):
                        layout.reconcile_artifact_manifest()
                state.set_state(
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=f"{type(error).__name__}: {error}",
                )
                raise
            finally:
                state.release_owner(session_id)
        finally:
            state.close()

    @staticmethod
    def _verify_state_checkpoint(state: ExperimentStateStore, checkpoints: CheckpointStore) -> None:
        file_checkpoint = checkpoints.latest()
        db_checkpoint = state.checkpoint()
        if file_checkpoint is None and db_checkpoint is None:
            return
        if file_checkpoint is None or db_checkpoint is None:
            raise StateError("state/checkpoint indexes disagree")
        if (
            db_checkpoint.get("checkpoint_id") != file_checkpoint.get("checkpoint_id")
            or db_checkpoint.get("sha256") != file_checkpoint.get("checkpoint_sha256")
            or Path(str(db_checkpoint.get("path", ""))).resolve()
            != checkpoints.root / f"checkpoint-{int(file_checkpoint['sequence']):012d}.json"
        ):
            raise StateError("state database checkpoint digest does not match checkpoint chain")

    @staticmethod
    def _verify_root_config(
        layout: ExperimentLayout,
        config: ExperimentConfig,
        lock: Mapping[str, Any],
    ) -> None:
        try:
            stored = layout.experiment_config.read_bytes()
        except OSError as exc:
            raise WorkspaceError(
                f"cannot read immutable experiment.toml: {layout.experiment_config}"
            ) from exc
        import hashlib

        if hashlib.sha256(stored).hexdigest() != lock.get("source_config_sha256"):
            raise LockError("immutable root experiment.toml does not match experiment.lock.json")
        if stored != config.source_bytes and not _same_immutable_config(stored, config):
            raise LockError("immutable root experiment.toml differs from the locked specification")

    def _invoke_adapter(
        self,
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: SessionContext,
    ) -> Mapping[str, Any] | None:
        # Adapters from early experiments sometimes accepted an explicit
        # budget.  Support that additive form without changing the core API.
        run = self.adapter.run
        parameters: Mapping[str, inspect.Parameter]
        try:
            parameters = inspect.signature(run).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "budget" in parameters:
            return run(
                config,
                layout,
                state,
                session,
                budget=SessionBudget(config.run.wall_seconds, session.monotonic_started),
            )  # type: ignore[call-arg]
        return run(config, layout, state, session)

    def _preflight(self, config: ExperimentConfig) -> Mapping[str, Any] | None:
        preflight = getattr(self.adapter, "preflight", None)
        if not callable(preflight):
            return None
        value = preflight(config)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise WorkspaceError("experiment adapter preflight returned a non-object result")
        return value

    @staticmethod
    def _normalize_outcome(
        result: Mapping[str, Any] | None, session: SessionContext
    ) -> dict[str, Any]:
        outcome = dict(result or {})
        requested_state = str(outcome.get("state", "idle"))
        state = (
            requested_state
            if requested_state in {"idle", "interrupted", "failed", "completed"}
            else "idle"
        )
        if session.budget_exhausted() and state == "running":
            state = "idle"
        outcome["state"] = state
        # Checkpoint fields are append-only identity lists.  Adapters may
        # expose convenient numeric counters for their result envelope; do
        # not let that presentation detail corrupt checkpoint serialization.
        for field in ("completed_slots", "provider_turns", "evaluations"):
            if not isinstance(outcome.get(field), list):
                outcome[field] = []
        outcome.setdefault("stop_reason", "budget_exhausted" if state == "idle" else "completed")
        return outcome

    @staticmethod
    def _record_completed_session(
        config: ExperimentConfig, layout: ExperimentLayout, state: ExperimentStateStore
    ) -> dict[str, Any]:
        number = state.next_session_number()
        session_id = f"session-{number:06d}"
        state.acquire_owner(exp_id=config.exp_id, session_id=session_id)
        manager = SessionManager(layout, state)
        try:
            session = manager.start(config)
            latest_checkpoint = state.checkpoint()
            session.ending_checkpoint = (
                str(latest_checkpoint["checkpoint_id"]) if latest_checkpoint is not None else None
            )
            summary = manager.finish(
                session,
                state="completed",
                stop_reason="already_completed",
                summary={"provider_calls": 0, "evaluation_calls": 0},
            )
            layout.write_artifact_manifest()
            return {
                "schema_version": "mforge.experiment.run.v1",
                "status": "completed",
                "exp_id": config.exp_id,
                "state": "completed",
                "workspace": str(layout.root),
                "session_id": session_id,
                "stop_reason": "already_completed",
                "checkpoint": session.ending_checkpoint,
                "provider_calls": 0,
                "evaluation_calls": 0,
                "session": summary,
            }
        finally:
            state.release_owner(session_id)

    @staticmethod
    def _run_result(
        config: ExperimentConfig,
        layout: ExperimentLayout,
        state: ExperimentStateStore,
        session: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "mforge.experiment.run.v1",
            "status": "completed" if outcome.get("state") in {"idle", "completed"} else "failed",
            "exp_id": config.exp_id,
            "state": outcome.get("state"),
            "workspace": str(layout.root),
            "session_id": session.get("session_id"),
            "stop_reason": outcome.get("stop_reason"),
            "checkpoint": session.get("ending_checkpoint"),
            "provider_turns": state.counts().get("provider_turns", 0),
            "evaluation_count": state.counts().get("evaluation_count", 0),
        }
        if "result" in outcome:
            result["result"] = outcome["result"]
        if outcome.get("last_error"):
            result["last_error"] = outcome["last_error"]
        return result


def _same_immutable_config(stored_bytes: bytes, current: ExperimentConfig) -> bool:
    try:
        import tomllib

        raw = tomllib.loads(stored_bytes.decode("utf-8"))
        current_raw = dict(current.raw)
        raw.pop("run", None)
        current_raw.pop("run", None)
        return raw == current_raw
    except (UnicodeError, tomllib.TOMLDecodeError):
        return False


def run_experiment(
    config_path: str | Path = "experiment.toml",
    *,
    adapter: ExperimentAdapter | None = None,
) -> dict[str, Any]:
    return ExperimentService(adapter=adapter).run(config_path)


__all__ = [
    "ExperimentAdapter",
    "ExperimentService",
    "LegacyStage4Adapter",
    "NullExperimentAdapter",
    "SessionBudget",
    "run_experiment",
]
