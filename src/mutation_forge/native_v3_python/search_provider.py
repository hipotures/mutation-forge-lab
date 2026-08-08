"""Persistent Codex App Server and scientific adapters for M5."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
    evaluate_serial_python_policy,
)

_ANCHOR_ACK = {
    "schema_version": "mforge.native.python_m5_specification_ack.v1",
    "ack": "specification-retained",
}
M5_PROVIDER_MAX_TURNS = 40
M5_PROVIDER_MAX_CAMPAIGNS = 40
M5_PROVIDER_TRANSCRIPT_BYTES = 16 * 1024 * 1024
M5_PROVIDER_STDOUT_BYTES = 16 * 1024 * 1024


def _write_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        retained = read_json(path)
        if (
            not isinstance(retained, Mapping)
            or canonical_json_bytes(retained) != canonical_json_bytes(value)
        ):
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
            artifact_root=self.artifact_root
            / candidate_id
            / case.case_id
            / "counterexamples",
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
            ),
            runtime_limits=self.runtime_limits,
            counterexample_pipeline=pipeline,
            provenance_source_kind="native_v3_python_model_search",
        )
        return result.as_dict(
            include_telemetry=True,
            include_external_activity=True,
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
        turn_timeout_seconds: float = 300.0,
        cleanup_capsule: bool = True,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.effort = effort
        self.profile = ModelProfile("codex", model, effort)
        self._state_path = self.workspace / "provider-state.json.gz"
        self._cleanup_capsule = cleanup_capsule
        self._capsule: IsolatedCapsule | None = None
        self._owns_adapter = adapter is None
        self._anchor_can_activate = False
        if adapter is not None:
            self.adapter = adapter
            if self._state_path.exists():
                self._resume_state()
                self._anchor_can_activate = True
            return
        if self._state_path.exists():
            raw = read_json(self._state_path)
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("capsule_root"), str
            ):
                raise M5InfrastructureError("retained provider state is malformed")
            self._capsule = IsolatedCapsule.reopen(str(raw["capsule_root"]))
        else:
            self._capsule = IsolatedCapsule.create(
                secure_capsule_parent(),
                auth_json=auth_json,
                sandbox_mode="read-only",
                approval_policy="never",
            )
        self.adapter = CodexAppServerAdapter(
            capsule=self._capsule,
            limits=AppServerLimits(
                max_turns=M5_PROVIDER_MAX_TURNS,
                max_campaigns=M5_PROVIDER_MAX_CAMPAIGNS,
                response_bytes=64 * 1024,
                transcript_bytes=M5_PROVIDER_TRANSCRIPT_BYTES,
                stdout_bytes=M5_PROVIDER_STDOUT_BYTES,
                turn_timeout=turn_timeout_seconds,
                resource_cpu_seconds=600,
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

    def _state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {
                "schema_version": "mforge.native.python_m5_provider_state.v1",
                "capsule_root": (
                    str(self._capsule.root) if self._capsule is not None else None
                ),
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
            value = M5ProviderResultV1.from_dict(
                cast(Mapping[str, Any], read_json(retained))
            )
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
            warnings=(
                int(metadata_after["serverWarnings"])
                - warnings_before
            ),
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
                    str(item)
                    for item in cast(Sequence[object], raw["included_turn_ids"])
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
            raise M5InfrastructureError(
                "thread/fork crossed the exact inclusive turn boundary"
            )
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
            result = M5ProviderResultV1.from_dict(
                cast(Mapping[str, Any], read_json(result_path))
            )
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
        cleanup = (
            self._cleanup_capsule
            if cleanup_capsule is None
            else cleanup_capsule
        )
        if self._owns_adapter and cleanup and self._capsule is not None:
            self._capsule.cleanup()


__all__ = [
    "CodexM5SearchProvider",
    "PythonPanelScientificEvaluator",
    "specification_ack_schema",
]
