"""Persistent Codex App Server and scientific adapters for M5."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.counterexamples import CounterexamplePipeline
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue
from mutation_forge.native_v3.canonical import canonical_json_bytes
from mutation_forge.native_v3.heg_scoring import scorer_for_backend
from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    ForkResult,
    GenerationResult,
    ModelProfile,
    ProtocolError,
)
from mutation_forge.stage3.isolation import IsolatedCapsule, secure_capsule_parent

from .runtime_contracts import PolicyRuntimeLimitsV1
from .search import (
    DevelopmentCaseV1,
    M5InfrastructureError,
    M5ProviderContextV1,
    M5ProviderResultV1,
)
from .serial_evaluator import (
    PythonSerialEpisodeConfigV1,
    evaluate_serial_builtin_baseline,
    evaluate_serial_python_policy,
)

_ANCHOR_ACK = {
    "schema_version": "mforge.native.python_m5_specification_ack.v1",
    "ack": "specification-retained",
}
M5_PROVIDER_TRANSCRIPT_BYTES = 16 * 1024 * 1024
M5_PROVIDER_STDOUT_BYTES = 16 * 1024 * 1024
M10_PROVIDER_MAX_EVENTS = 100_000
M10_PROVIDER_TRANSCRIPT_BYTES = 64 * 1024 * 1024
M10_PROVIDER_STDOUT_BYTES = 64 * 1024 * 1024
_M10_POOL_STATE_PROTOCOL = "mforge.native.python_provider_pool.v1"


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        retained = read_json(path)
        if not isinstance(retained, Mapping) or canonical_json_bytes(
            retained
        ) != canonical_json_bytes(value):
            raise M5InfrastructureError(f"immutable provider contract changed: {path}")
        return
    write_json(path, value, exclusive=True)


def specification_ack_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "ack"],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": _ANCHOR_ACK["schema_version"],
            },
            "ack": {"type": "string", "const": _ANCHOR_ACK["ack"]},
        },
    }


def _app_server_limits(
    *,
    program_turn_limit: int | None,
    turn_timeout_seconds: float,
) -> AppServerLimits:
    max_turns = None if program_turn_limit is None else program_turn_limit + 1
    return AppServerLimits(
        max_turns=max_turns,
        max_campaigns=(None if program_turn_limit is None else max_turns),
        max_events=(10_000 if program_turn_limit is None else M10_PROVIDER_MAX_EVENTS),
        response_bytes=64 * 1024,
        transcript_bytes=(
            M5_PROVIDER_TRANSCRIPT_BYTES
            if program_turn_limit is None
            else M10_PROVIDER_TRANSCRIPT_BYTES
        ),
        stdout_bytes=(
            M5_PROVIDER_STDOUT_BYTES if program_turn_limit is None else M10_PROVIDER_STDOUT_BYTES
        ),
        turn_timeout=turn_timeout_seconds,
        resource_cpu_seconds=600,
    )


class PythonPanelScientificEvaluator:
    """Run every panel case through the accepted M1/M2/M3 path."""

    def __init__(
        self,
        *,
        backend: GraphBackend,
        artifact_root: str | Path,
        runtime_limits: PolicyRuntimeLimitsV1 | None = None,
    ) -> None:
        self.backend = backend
        self.artifact_root = Path(artifact_root)
        self.runtime_limits = runtime_limits

    def evaluate(
        self,
        *,
        source: str,
        case: DevelopmentCaseV1,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]:
        scorer = scorer_for_backend(self.backend)
        pipeline = CounterexamplePipeline(
            backend=self.backend,
            artifact_root=self.artifact_root / candidate_id / case.case_id / "counterexamples",
        )
        result = evaluate_serial_python_policy(
            backend=self.backend,
            scorer=scorer,
            source=source,
            config=PythonSerialEpisodeConfigV1(
                order=case.order,
                graph_seed=case.graph_seed,
                policy_seed=case.policy_seed,
                horizon=case.horizon,
                witness_cap=case.witness_cap,
                episode_id=f"native-v3-python-m5/{candidate_id}/{case.case_id}",
                forbidden_lengths=case.forbidden_lengths,
                graph_mode=case.graph_mode,
            ),
            runtime_limits=self.runtime_limits,
            counterexample_pipeline=pipeline,
            provenance_source_kind="native_v3_python_model_search",
        )
        return result.as_dict(
            include_telemetry=True,
            include_external_activity=True,
        )

    def evaluate_baseline(
        self,
        *,
        baseline: str,
        case: DevelopmentCaseV1,
        generation: int,
    ) -> Mapping[str, JsonValue]:
        scorer = scorer_for_backend(self.backend)
        pipeline = CounterexamplePipeline(
            backend=self.backend,
            artifact_root=self.artifact_root
            / f"generation-{generation:04d}"
            / "baselines"
            / baseline
            / case.case_id
            / "counterexamples",
        )
        return evaluate_serial_builtin_baseline(
            backend=self.backend,
            scorer=scorer,
            baseline=baseline,
            config=PythonSerialEpisodeConfigV1(
                order=case.order,
                graph_seed=case.graph_seed,
                policy_seed=case.policy_seed,
                horizon=case.horizon,
                witness_cap=case.witness_cap,
                episode_id=(f"native-v3-python-baseline/{generation}/{baseline}/{case.case_id}"),
                forbidden_lengths=case.forbidden_lengths,
                graph_mode=case.graph_mode,
            ),
            counterexample_pipeline=pipeline,
        )


class CodexM5SearchProvider:
    """One durable specification thread with exact root and parent forks."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        model: str,
        effort: str,
        base_instructions: str,
        auth_json: str | Path | None = None,
        adapter: CodexAppServerAdapter | None = None,
        capsule: IsolatedCapsule | None = None,
        turn_timeout_seconds: float = 300.0,
        cleanup_capsule: bool = True,
        program_turn_limit: int | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.effort = effort
        self.profile = ModelProfile("codex", model, effort)
        self._state_path = self.workspace / "provider-state.json.gz"
        self._cleanup_capsule = cleanup_capsule
        self._capsule: IsolatedCapsule | None = capsule
        self._owns_adapter = adapter is None
        self._resumed_thread_ids: set[str] = set()
        self._anchor_can_activate = False
        if program_turn_limit is not None and program_turn_limit < 1:
            raise ValueError("program_turn_limit must be positive")
        if adapter is not None:
            self.adapter = adapter
            if self._state_path.exists():
                self._resume_state()
                self._anchor_can_activate = True
            return
        if self._state_path.exists():
            raw = read_json(self._state_path)
            if not isinstance(raw, Mapping) or not isinstance(raw.get("capsule_root"), str):
                raise M5InfrastructureError("retained provider state is malformed")
            retained_root = str(raw["capsule_root"])
            if self._capsule is None:
                self._capsule = IsolatedCapsule.reopen(retained_root)
            elif self._capsule.root.resolve() != Path(retained_root).resolve():
                raise M5InfrastructureError("shared provider capsule changed on resume")
        else:
            self._capsule = self._capsule or IsolatedCapsule.create(
                secure_capsule_parent(),
                auth_json=auth_json,
                sandbox_mode="read-only",
                approval_policy="never",
            )
        self.adapter = CodexAppServerAdapter(
            capsule=self._capsule,
            limits=_app_server_limits(
                program_turn_limit=program_turn_limit,
                turn_timeout_seconds=turn_timeout_seconds,
            ),
            base_instructions=base_instructions,
            sandbox_mode="read-only",
            approval_policy="never",
            compress_json_artifacts=True,
            copy_rollout_artifact=True,
        )
        if self._state_path.exists():
            self._resume_state()
            self._anchor_can_activate = True

    @property
    def capsule(self) -> IsolatedCapsule:
        if self._capsule is None:
            raise M5InfrastructureError("provider capsule is unavailable")
        return self._capsule

    def _state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {
                "schema_version": "mforge.native.python_m5_provider_state.v1",
                "capsule_root": (str(self._capsule.root) if self._capsule is not None else None),
                "anchor": None,
                "threads": {},
                "telemetry": {
                    "transport_retries": 0,
                    "process_restarts": 0,
                    "thread_resume_attempts": 0,
                },
            }
        raw = read_json(self._state_path)
        if not isinstance(raw, Mapping):
            raise M5InfrastructureError("provider state is not an object")
        return dict(raw)

    def _increment_telemetry(self, field: str, amount: int = 1) -> None:
        state = self._state()
        telemetry = state.setdefault("telemetry", {})
        if not isinstance(telemetry, dict):
            raise M5InfrastructureError("provider telemetry state is malformed")
        current = telemetry.get(field, 0)
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            raise M5InfrastructureError("provider telemetry counter is malformed")
        telemetry[field] = current + amount
        write_json(self._state_path, state)

    def _save_context(
        self,
        *,
        anchor: M5ProviderContextV1 | None = None,
        context: M5ProviderContextV1 | None = None,
    ) -> None:
        state = self._state()
        if anchor is not None:
            state["anchor"] = anchor.as_dict()
        if context is not None:
            threads = state.setdefault("threads", {})
            if not isinstance(threads, dict):
                raise M5InfrastructureError("provider thread state is malformed")
            threads[context.thread_id] = context.as_dict()
        write_json(self._state_path, state)

    def _resume_state(self) -> None:
        state = self._state()
        anchor_raw = state.get("anchor")
        if not isinstance(anchor_raw, Mapping):
            return
        anchor = M5ProviderContextV1.from_dict(anchor_raw)
        self._increment_telemetry("thread_resume_attempts")
        self.adapter.resume_thread(
            self.profile,
            thread_id=anchor.thread_id,
            thread_path=anchor.thread_path,
        )
        self._resumed_thread_ids.add(anchor.thread_id)
        threads = state.get("threads", {})
        if not isinstance(threads, Mapping):
            raise M5InfrastructureError("provider threads are malformed")
        for thread_id, raw in sorted(threads.items()):
            if thread_id == anchor.thread_id:
                continue
            if not isinstance(raw, Mapping):
                raise M5InfrastructureError("provider thread context is malformed")
            context = M5ProviderContextV1.from_dict(raw)
            self._increment_telemetry("thread_resume_attempts")
            self.adapter.resume_forked_thread(
                self.profile,
                thread_id=context.thread_id,
                thread_path=context.thread_path,
            )
            self._resumed_thread_ids.add(context.thread_id)

    def ensure_context(self, context: M5ProviderContextV1) -> None:
        """Register one exact durable thread on this provider process."""

        if context.thread_id in self._resumed_thread_ids:
            return
        self.adapter.resume_forked_thread(
            self.profile,
            thread_id=context.thread_id,
            thread_path=context.thread_path,
        )
        self._resumed_thread_ids.add(context.thread_id)
        self._save_context(context=context)

    def ensure_anchor_context(self, context: M5ProviderContextV1) -> None:
        """Register the one specification thread on this provider process."""

        if context.thread_id in self._resumed_thread_ids:
            return
        self.adapter.resume_thread(
            self.profile,
            thread_id=context.thread_id,
            thread_path=context.thread_path,
        )
        self._resumed_thread_ids.add(context.thread_id)
        self._save_context(anchor=context, context=context)

    @staticmethod
    def _usage(result: GenerationResult) -> dict[str, JsonValue]:
        return {
            "inputTokens": result.usage.input_tokens,
            "cachedInputTokens": result.usage.cached_input_tokens,
            "cacheWriteInputTokens": result.usage.cache_write_input_tokens,
            "outputTokens": result.usage.output_tokens,
            "reasoningOutputTokens": result.usage.reasoning_output_tokens,
            "totalTokens": result.usage.total_tokens,
            "final": result.usage.final,
            "partial": result.usage.partial,
            "raw": cast(JsonValue, dict(result.usage.raw)),
        }

    def _record_request(
        self,
        *,
        artifact_dir: Path,
        prefix: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.adapter.rotate_logger(artifact_dir, prefix, compress_json=True)
        logger = self.adapter.logger
        if logger is None:
            raise M5InfrastructureError("provider logger is unavailable")
        logger.profile(
            {
                "model": self.model,
                "effort": self.effort,
                "ephemeral": False,
                "artifactPrefix": prefix,
            }
        )
        logger.raw_text("request.md", prompt)
        logger.document(
            "request.json",
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "output_schema": dict(output_schema),
                "idempotency_key": idempotency_key,
            },
        )
        logger.raw_text("system-prompt.md", system_prompt)
        logger.document("output-schema.json", dict(output_schema))

    def _turn(
        self,
        *,
        artifact_dir: Path,
        prefix: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        history: tuple[str, ...],
    ) -> M5ProviderResultV1:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_or_verify(
            artifact_dir / "m5-turn-contract.json.gz",
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "output_schema": dict(output_schema),
                "idempotency_key": idempotency_key,
                "history": list(history),
            },
        )
        retained = artifact_dir / "m5-provider-result.json.gz"
        if retained.exists():
            value = M5ProviderResultV1.from_dict(cast(Mapping[str, Any], read_json(retained)))
            self._save_context(context=value.context)
            return value
        self._record_request(
            artifact_dir=artifact_dir,
            prefix=prefix,
            prompt=prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            idempotency_key=idempotency_key,
        )
        metadata_before = self.adapter.inspect_metadata()
        warnings_before = int(metadata_before["serverWarnings"])
        retries_before = int(metadata_before["serverRetries"])
        result = self.adapter.generate_persistent(
            prompt,
            self.profile,
            output_schema=output_schema,
        )
        if not result.usage.final or result.usage.partial:
            raise M5InfrastructureError("provider turn omitted exact final usage")
        logger = self.adapter.logger
        if logger is None:
            raise M5InfrastructureError("provider logger disappeared")
        logger.raw_text("response.raw.txt", result.text)
        logger.raw_text("response.md", result.text)
        try:
            response = json.loads(result.text)
        except (ValueError, RecursionError):
            response = result.text
        logger.document("response.json", response)
        logger.document("usage.json", self._usage(result))
        logger.document(
            "provider-raw.json",
            {
                "response_text": result.text,
                "thread_id": result.thread_id,
                "session_id": result.session_id,
                "turn_id": result.turn_id,
                "thread_path": result.thread_path,
                "request_id": result.request_id,
                "usage": self._usage(result),
                "diagnostics": list(result.diagnostics),
            },
        )
        context = M5ProviderContextV1(
            thread_id=result.thread_id,
            turn_id=result.turn_id,
            thread_path=result.thread_path,
            included_turn_ids=history + (result.turn_id,),
        )
        metadata_after = self.adapter.inspect_metadata()
        retries = int(metadata_after["serverRetries"]) - retries_before
        if retries:
            self._increment_telemetry("transport_retries", retries)
        value = M5ProviderResultV1(
            response_text=result.text,
            context=context,
            usage=self._usage(result),
            duration_ms=result.duration_ms or 0,
            warnings=(int(metadata_after["serverWarnings"]) - warnings_before),
        )
        write_json(retained, value.as_dict(), exclusive=True)
        self._save_context(context=context)
        return value

    def _fork(
        self,
        *,
        last_turn_id: str,
        expected_history: tuple[str, ...],
        artifact_dir: Path,
        prefix: str,
    ) -> ForkResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_or_verify(
            artifact_dir / "m5-fork-contract.json.gz",
            {
                "last_turn_id": last_turn_id,
                "expected_history": list(expected_history),
                "prefix": prefix,
            },
        )
        fork_path = artifact_dir / "m5-fork-result.json.gz"
        if fork_path.exists():
            raw = read_json(fork_path)
            if not isinstance(raw, Mapping):
                raise M5InfrastructureError("retained fork result is malformed")
            retained = ForkResult(
                source_thread_id=str(raw["source_thread_id"]),
                child_thread_id=str(raw["child_thread_id"]),
                session_id=str(raw["session_id"]),
                thread_path=str(raw["thread_path"]),
                last_turn_id=str(raw["last_turn_id"]),
                included_turn_ids=tuple(
                    str(item) for item in cast(Sequence[object], raw["included_turn_ids"])
                ),
            )
            if (
                retained.last_turn_id != last_turn_id
                or retained.included_turn_ids != expected_history
            ):
                raise M5InfrastructureError(
                    "retained fork crossed the exact inclusive turn boundary"
                )
            self._save_context(
                context=M5ProviderContextV1(
                    thread_id=retained.child_thread_id,
                    turn_id=retained.last_turn_id,
                    thread_path=retained.thread_path,
                    included_turn_ids=retained.included_turn_ids,
                )
            )
            return retained
        self.adapter.rotate_logger(artifact_dir, prefix, compress_json=True)
        logger = self.adapter.logger
        if logger is None:
            raise M5InfrastructureError("fork logger is unavailable")
        logger.raw_text(
            "request.md",
            json.dumps(
                {"method": "thread/fork", "lastTurnId": last_turn_id},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        result = self.adapter.fork_persistent_thread(
            self.profile,
            last_turn_id=last_turn_id,
            activate=False,
        )
        if result.included_turn_ids != expected_history:
            raise M5InfrastructureError("thread/fork crossed the exact inclusive turn boundary")
        payload = {
            "source_thread_id": result.source_thread_id,
            "child_thread_id": result.child_thread_id,
            "session_id": result.session_id,
            "thread_path": result.thread_path,
            "last_turn_id": result.last_turn_id,
            "included_turn_ids": list(result.included_turn_ids),
        }
        logger.raw_text(
            "response.raw.txt",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        logger.document("response.json", payload)
        logger.document("provider-raw.json", payload)
        logger.document("usage.json", {"totalTokens": 0, "final": True, "partial": False})
        write_json(fork_path, payload, exclusive=True)
        self._save_context(
            context=M5ProviderContextV1(
                thread_id=result.child_thread_id,
                turn_id=result.last_turn_id,
                thread_path=result.thread_path,
                included_turn_ids=result.included_turn_ids,
            )
        )
        return result

    def ensure_specification_anchor(
        self,
        *,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        state = self._state()
        retained = state.get("anchor")
        if isinstance(retained, Mapping):
            result_path = artifact_dir / "m5-provider-result.json.gz"
            if not result_path.exists():
                raise M5InfrastructureError(
                    "provider anchor state exists without its durable result"
                )
            result = M5ProviderResultV1.from_dict(cast(Mapping[str, Any], read_json(result_path)))
            if result.context != M5ProviderContextV1.from_dict(retained):
                raise M5InfrastructureError("retained specification anchor changed")
            return result
        result = self._turn(
            artifact_dir=artifact_dir,
            prefix="specification-anchor",
            prompt=prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            idempotency_key="m5-specification-anchor",
            history=(),
        )
        try:
            decoded = json.loads(result.response_text)
        except ValueError as error:
            raise M5InfrastructureError("specification anchor returned invalid JSON") from error
        if decoded != _ANCHOR_ACK:
            raise M5InfrastructureError("specification anchor acknowledgement changed")
        self._save_context(anchor=result.context, context=result.context)
        return result

    def generate_root(
        self,
        *,
        anchor: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        if self._anchor_can_activate:
            self.adapter.activate_forked_thread(
                anchor.thread_id,
                completed_turn_ids=anchor.included_turn_ids,
            )
        fork = self._fork(
            last_turn_id=anchor.turn_id,
            expected_history=anchor.included_turn_ids,
            artifact_dir=artifact_dir / "fork",
            prefix=f"g{generation:04d}-{slot}-root-fork",
        )
        self._anchor_can_activate = True
        self.adapter.activate_forked_thread(fork.child_thread_id)
        return self._turn(
            artifact_dir=artifact_dir,
            prefix=f"g{generation:04d}-{slot}-root",
            prompt=prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            idempotency_key=idempotency_key,
            history=fork.included_turn_ids,
        )

    def prepare_root_worker(
        self,
        *,
        anchor: M5ProviderContextV1,
        worker: int,
        artifact_dir: Path,
    ) -> M5ProviderContextV1:
        """Fork one persistent root worker from the exact specification turn."""

        self.ensure_anchor_context(anchor)
        self.adapter.activate_forked_thread(
            anchor.thread_id,
            completed_turn_ids=anchor.included_turn_ids,
        )
        fork = self._fork(
            last_turn_id=anchor.turn_id,
            expected_history=anchor.included_turn_ids,
            artifact_dir=artifact_dir,
            prefix=f"root-worker-{worker:02d}-fork",
        )
        context = M5ProviderContextV1(
            thread_id=fork.child_thread_id,
            turn_id=fork.last_turn_id,
            thread_path=fork.thread_path,
            included_turn_ids=fork.included_turn_ids,
        )
        self._resumed_thread_ids.add(context.thread_id)
        return context

    def fork_root_worker_from_active_anchor(
        self,
        *,
        anchor: M5ProviderContextV1,
        worker: int,
        artifact_dir: Path,
    ) -> M5ProviderContextV1:
        """Fork a root worker while this process exclusively owns the anchor."""

        state = self._state()
        retained_anchor = state.get("anchor")
        if (
            not isinstance(retained_anchor, Mapping)
            or M5ProviderContextV1.from_dict(retained_anchor) != anchor
        ):
            raise M5InfrastructureError("root worker uses a foreign specification anchor")
        fork = self._fork(
            last_turn_id=anchor.turn_id,
            expected_history=anchor.included_turn_ids,
            artifact_dir=artifact_dir,
            prefix=f"root-worker-{worker:02d}-fork",
        )
        context = M5ProviderContextV1(
            thread_id=fork.child_thread_id,
            turn_id=fork.last_turn_id,
            thread_path=fork.thread_path,
            included_turn_ids=fork.included_turn_ids,
        )
        self._save_context(context=context)
        return context

    def generate_root_on_worker(
        self,
        *,
        worker_context: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        """Run one fresh-root request on a persistent root-worker thread."""

        self.ensure_context(worker_context)
        self.adapter.activate_forked_thread(
            worker_context.thread_id,
            completed_turn_ids=worker_context.included_turn_ids,
        )
        return self._turn(
            artifact_dir=artifact_dir,
            prefix=f"g{generation:04d}-{slot}-root",
            prompt=prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            idempotency_key=idempotency_key,
            history=worker_context.included_turn_ids,
        )

    def generate_child(
        self,
        *,
        parent: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        self.ensure_context(parent)
        self.adapter.activate_forked_thread(
            parent.thread_id,
            completed_turn_ids=parent.included_turn_ids,
        )
        fork = self._fork(
            last_turn_id=parent.turn_id,
            expected_history=parent.included_turn_ids,
            artifact_dir=artifact_dir / "fork",
            prefix=f"g{generation:04d}-{slot}-child-fork",
        )
        self.adapter.activate_forked_thread(fork.child_thread_id)
        return self._turn(
            artifact_dir=artifact_dir,
            prefix=f"g{generation:04d}-{slot}-child",
            prompt=prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            idempotency_key=idempotency_key,
            history=fork.included_turn_ids,
        )

    def repair(
        self,
        *,
        previous: M5ProviderResultV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        self.ensure_context(previous.context)
        self.adapter.activate_forked_thread(
            previous.context.thread_id,
            completed_turn_ids=previous.context.included_turn_ids,
        )
        return self._turn(
            artifact_dir=artifact_dir,
            prefix=f"g{generation:04d}-{slot}-repair-01",
            prompt=prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            idempotency_key=idempotency_key,
            history=previous.context.included_turn_ids,
        )

    def close(self, *, cleanup_capsule: bool | None = None) -> None:
        self.adapter.close()
        cleanup = self._cleanup_capsule if cleanup_capsule is None else cleanup_capsule
        if self._owns_adapter and cleanup and self._capsule is not None:
            self._capsule.cleanup()


class CodexM10SearchProvider:
    """Four bounded app-server workers sharing one durable specification anchor."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        model: str,
        effort: str,
        base_instructions: str,
        auth_json: str | Path,
        turn_timeout_seconds: float,
        provider_concurrency: int,
        provider_total_turn_limit: int | None,
    ) -> None:
        if not 1 <= provider_concurrency <= 4:
            raise ValueError("provider_concurrency must be between 1 and 4")
        if provider_total_turn_limit is not None and provider_total_turn_limit < 1:
            raise ValueError("provider_total_turn_limit must be positive")
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.effort = effort
        self.provider_concurrency = provider_concurrency
        self._base_instructions = base_instructions
        self._turn_timeout_seconds = turn_timeout_seconds
        self._provider_total_turn_limit = provider_total_turn_limit
        self._state_path = self.workspace / "provider-pool-state.json.gz"
        self._state_lock = threading.RLock()
        self._turn_condition = threading.Condition(self._state_lock)
        self._coordinator = CodexM5SearchProvider(
            workspace=self.workspace / "coordinator",
            model=model,
            effort=effort,
            base_instructions=base_instructions,
            auth_json=auth_json,
            turn_timeout_seconds=turn_timeout_seconds,
            cleanup_capsule=True,
            program_turn_limit=provider_total_turn_limit,
        )
        self._workers: list[CodexM5SearchProvider] = []
        self._worker_locks = [threading.Lock() for _ in range(provider_concurrency)]
        self._anchor: M5ProviderContextV1 | None = None
        self._root_workers: dict[int, M5ProviderContextV1] = {}
        self._thread_owners: dict[str, int] = {}
        self._primary_slot_owners: dict[str, int] = {}
        self._completed_primary_slots: set[str] = set()
        self._released_primary_slots: set[str] = set()
        self._coordinator_released = False
        self._load_state()

    def _load_state(self) -> None:
        if not self._state_path.is_file():
            return
        raw = read_json(self._state_path)
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != _M10_POOL_STATE_PROTOCOL
            or raw.get("model") != self.model
            or raw.get("effort") != self.effort
            or raw.get("provider_concurrency") != self.provider_concurrency
        ):
            raise M5InfrastructureError("provider pool state changed on resume")
        anchor = raw.get("anchor")
        if isinstance(anchor, Mapping):
            self._anchor = M5ProviderContextV1.from_dict(anchor)
        root_workers = raw.get("root_workers", {})
        owners = raw.get("thread_owners", {})
        completed = raw.get("completed_primary_slots", [])
        released = raw.get("released_primary_slots", [])
        if (
            not isinstance(root_workers, Mapping)
            or not isinstance(owners, Mapping)
            or not isinstance(completed, Sequence)
            or isinstance(completed, str | bytes)
            or not all(isinstance(item, str) and item for item in completed)
            or not isinstance(released, Sequence)
            or isinstance(released, str | bytes)
            or not all(isinstance(item, str) and item for item in released)
        ):
            raise M5InfrastructureError("provider pool topology is malformed")
        self._root_workers = {
            int(worker): M5ProviderContextV1.from_dict(cast(Mapping[str, Any], value))
            for worker, value in root_workers.items()
            if isinstance(value, Mapping)
        }
        self._thread_owners = {
            str(thread_id): int(worker)
            for thread_id, worker in owners.items()
            if isinstance(thread_id, str)
            and isinstance(worker, int)
            and not isinstance(worker, bool)
        }
        self._completed_primary_slots = set(cast(Sequence[str], completed))
        self._released_primary_slots = set(cast(Sequence[str], released))
        if not self._released_primary_slots <= self._completed_primary_slots:
            raise M5InfrastructureError("released provider slots are not completed")
        if any(
            worker not in range(self.provider_concurrency)
            for worker in (
                *self._root_workers,
                *self._thread_owners.values(),
            )
        ):
            raise M5InfrastructureError("provider pool worker assignment changed")

    def _save_state(self) -> None:
        with self._state_lock:
            write_json(
                self._state_path,
                {
                    "schema_version": _M10_POOL_STATE_PROTOCOL,
                    "model": self.model,
                    "effort": self.effort,
                    "provider_concurrency": self.provider_concurrency,
                    "anchor": (self._anchor.as_dict() if self._anchor is not None else None),
                    "root_workers": {
                        str(worker): context.as_dict()
                        for worker, context in sorted(self._root_workers.items())
                    },
                    "thread_owners": dict(sorted(self._thread_owners.items())),
                    "completed_primary_slots": sorted(self._completed_primary_slots),
                    "released_primary_slots": sorted(self._released_primary_slots),
                },
            )

    def _ensure_workers(self) -> None:
        if self._workers:
            return
        for worker in range(self.provider_concurrency):
            self._workers.append(self._new_worker(worker))

    def _new_worker(self, worker: int) -> CodexM5SearchProvider:
        capsule = self._coordinator.capsule
        adapter = CodexAppServerAdapter(
            capsule=capsule,
            limits=_app_server_limits(
                program_turn_limit=self._provider_total_turn_limit,
                turn_timeout_seconds=self._turn_timeout_seconds,
            ),
            base_instructions=self._base_instructions,
            sandbox_mode="read-only",
            approval_policy="never",
            compress_json_artifacts=True,
            copy_rollout_artifact=True,
        )
        return CodexM5SearchProvider(
            workspace=self.workspace / "workers" / f"worker-{worker:02d}",
            model=self.model,
            effort=self.effort,
            base_instructions=self._base_instructions,
            adapter=adapter,
            capsule=capsule,
            cleanup_capsule=False,
        )

    def ensure_specification_anchor(
        self,
        **kwargs: Any,
    ) -> M5ProviderResultV1:
        result = self._coordinator.ensure_specification_anchor(**kwargs)
        if self._anchor is not None and self._anchor != result.context:
            raise M5InfrastructureError("provider pool anchor changed")
        self._anchor = result.context
        self._save_state()
        return result

    def prepare_generation(
        self,
        *,
        snapshot: Mapping[str, Any],
        anchor: M5ProviderContextV1,
        artifact_dir: Path,
    ) -> None:
        """Start workers sequentially and freeze their exact root forks."""

        _write_or_verify(artifact_dir / "provider-snapshot.json.gz", snapshot)
        if self._anchor != anchor:
            raise M5InfrastructureError("generation uses a foreign anchor")
        if not self._root_workers:
            for worker in range(self.provider_concurrency):
                context = self._coordinator.fork_root_worker_from_active_anchor(
                    anchor=anchor,
                    worker=worker,
                    artifact_dir=(
                        self.workspace / "root-workers" / f"worker-{worker:02d}" / "fork"
                    ),
                )
                self._root_workers[worker] = context
                self._thread_owners[context.thread_id] = worker
            self._save_state()
        if not self._coordinator_released:
            # A durable thread may be resumed by a replacement app-server
            # process only after the process that created it has released it.
            # The coordinator is specification-only after the anchor exists.
            self._coordinator.close(cleanup_capsule=False)
            self._coordinator_released = True
        self._ensure_workers()
        for worker, provider in enumerate(self._workers):
            context = self._root_workers[worker]
            try:
                provider.ensure_anchor_context(context)
            except ProtocolError as error:
                if str(error) != "request thread/resume failed":
                    raise
                provider._increment_telemetry("process_restarts")
                provider.close(cleanup_capsule=False)
                replacement = self._new_worker(worker)
                self._workers[worker] = replacement
                replacement.ensure_anchor_context(context)
        slots = snapshot.get("slots")
        generation = snapshot.get("generation")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or not isinstance(slots, Sequence)
            or isinstance(slots, str | bytes)
        ):
            raise M5InfrastructureError("generation provider snapshot is malformed")
        slot_owners: dict[str, int] = {}
        for raw_slot in slots:
            if not isinstance(raw_slot, Mapping):
                raise M5InfrastructureError("generation provider slot is malformed")
            slot = raw_slot.get("slot")
            kind = raw_slot.get("kind")
            parent_thread_id = raw_slot.get("parent_thread_id")
            if not isinstance(slot, str) or kind not in {"root", "child"}:
                raise M5InfrastructureError("generation provider slot identity changed")
            candidate_id = f"g{generation:04d}-{slot}"
            if kind == "root":
                owner = self._slot_index(slot) % self.provider_concurrency
            else:
                if (
                    not isinstance(parent_thread_id, str)
                    or parent_thread_id not in self._thread_owners
                ):
                    raise M5InfrastructureError("child provider owner is unavailable")
                owner = self._thread_owners[parent_thread_id]
            slot_owners[candidate_id] = owner
        self._primary_slot_owners.update(slot_owners)
        self._save_state()

    @staticmethod
    def _slot_index(slot: str) -> int:
        try:
            value = int(slot.removeprefix("slot-"))
        except ValueError as error:
            raise M5InfrastructureError("invalid provider slot identity") from error
        if not 0 <= value < 8:
            raise M5InfrastructureError("invalid provider slot identity")
        return value

    def _owner_for_context(
        self,
        context: M5ProviderContextV1,
        *,
        slot: str,
    ) -> int:
        return self._thread_owners.get(
            context.thread_id,
            self._slot_index(slot) % self.provider_concurrency,
        )

    def _record_owner(
        self,
        *,
        worker: int,
        context: M5ProviderContextV1,
        root_worker: bool = False,
    ) -> None:
        with self._state_lock:
            self._thread_owners[context.thread_id] = worker
            if root_worker:
                self._root_workers[worker] = context
            self._save_state()

    def _run_primary_turn(
        self,
        *,
        generation: int,
        slot: str,
        worker: int,
        operation: Callable[[], M5ProviderResultV1],
    ) -> M5ProviderResultV1:
        candidate_id = f"g{generation:04d}-{slot}"
        if self._primary_slot_owners.get(candidate_id) != worker:
            raise M5InfrastructureError("provider worker assignment changed")
        self.await_primary_slot(generation=generation, slot=slot)
        try:
            return operation()
        finally:
            with self._turn_condition:
                self._completed_primary_slots.add(candidate_id)
                self._save_state()
                self._turn_condition.notify_all()

    def await_primary_slot(self, *, generation: int, slot: str) -> None:
        """Wait outside provider telemetry until the lane is admissible."""

        candidate_id = f"g{generation:04d}-{slot}"
        with self._turn_condition:
            worker = self._primary_slot_owners.get(candidate_id)
            if worker is None:
                raise M5InfrastructureError("provider generation was not prepared")
            preceding = sorted(
                item
                for item, owner in self._primary_slot_owners.items()
                if owner == worker
                and item.startswith(f"g{generation:04d}-")
                and item < candidate_id
            )
            self._turn_condition.wait_for(
                lambda: all(item in self._released_primary_slots for item in preceding)
            )

    def release_primary_slot(self, *, generation: int, slot: str) -> None:
        """Advance the lane only after validation or same-thread repair."""

        candidate_id = f"g{generation:04d}-{slot}"
        with self._turn_condition:
            if candidate_id not in self._completed_primary_slots:
                raise M5InfrastructureError("provider slot was released before its turn completed")
            if candidate_id not in self._primary_slot_owners:
                raise M5InfrastructureError("provider slot release has no frozen owner")
            self._released_primary_slots.add(candidate_id)
            self._save_state()
            self._turn_condition.notify_all()

    def primary_lane(self, *, generation: int, slot: str) -> int:
        candidate_id = f"g{generation:04d}-{slot}"
        try:
            return self._primary_slot_owners[candidate_id]
        except KeyError as error:
            raise M5InfrastructureError("provider generation was not prepared") from error

    def generate_root(
        self,
        *,
        anchor: M5ProviderContextV1,
        generation: int,
        slot: str,
        **kwargs: Any,
    ) -> M5ProviderResultV1:
        self._ensure_workers()
        if self._anchor != anchor:
            raise M5InfrastructureError("fresh root uses a foreign anchor")
        worker = self._slot_index(slot) % self.provider_concurrency

        def operation() -> M5ProviderResultV1:
            with self._worker_locks[worker]:
                context = self._root_workers[worker]
                result = self._workers[worker].generate_root_on_worker(
                    worker_context=context,
                    generation=generation,
                    slot=slot,
                    **kwargs,
                )
                self._record_owner(
                    worker=worker,
                    context=result.context,
                    root_worker=True,
                )
                return result

        return self._run_primary_turn(
            generation=generation,
            slot=slot,
            worker=worker,
            operation=operation,
        )

    def generate_child(
        self,
        *,
        parent: M5ProviderContextV1,
        generation: int,
        slot: str,
        **kwargs: Any,
    ) -> M5ProviderResultV1:
        self._ensure_workers()
        worker = self._owner_for_context(parent, slot=slot)

        def operation() -> M5ProviderResultV1:
            with self._worker_locks[worker]:
                result = self._workers[worker].generate_child(
                    parent=parent,
                    generation=generation,
                    slot=slot,
                    **kwargs,
                )
                self._record_owner(worker=worker, context=result.context)
                return result

        return self._run_primary_turn(
            generation=generation,
            slot=slot,
            worker=worker,
            operation=operation,
        )

    def repair(
        self,
        *,
        previous: M5ProviderResultV1,
        generation: int,
        slot: str,
        **kwargs: Any,
    ) -> M5ProviderResultV1:
        self._ensure_workers()
        worker = self._owner_for_context(previous.context, slot=slot)
        with self._worker_locks[worker]:
            result = self._workers[worker].repair(
                previous=previous,
                generation=generation,
                slot=slot,
                **kwargs,
            )
            root_context = self._root_workers.get(worker)
            self._record_owner(
                worker=worker,
                context=result.context,
                root_worker=(
                    root_context is not None
                    and root_context.thread_id == previous.context.thread_id
                ),
            )
            return result

    def close(self, *, cleanup_capsule: bool = True) -> None:
        primary_error: Exception | None = None
        for worker in self._workers:
            try:
                worker.close(cleanup_capsule=False)
            except Exception as error:
                primary_error = primary_error or error
        try:
            self._coordinator.close(cleanup_capsule=cleanup_capsule)
        except Exception as error:
            primary_error = primary_error or error
        if primary_error is not None:
            raise primary_error


__all__ = [
    "CodexM5SearchProvider",
    "CodexM10SearchProvider",
    "M10_PROVIDER_MAX_EVENTS",
    "M10_PROVIDER_STDOUT_BYTES",
    "M10_PROVIDER_TRANSCRIPT_BYTES",
    "PythonPanelScientificEvaluator",
    "specification_ack_schema",
]
