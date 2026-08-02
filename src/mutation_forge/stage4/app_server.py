"""Stage 4 Codex App Server provider.

Stage 4 deliberately keeps the transport implementation in Stage 3.  This
module is a small, stateless boundary around that implementation: one call
gets one authenticated capsule, one private thread and one turn.  Keeping
the capsule on the call (rather than on the provider) is what makes eight
slots safe to run concurrently.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, cast

from mutation_forge.stage3.app_server import (
    AppServerLimits,
    CodexAppServerAdapter,
    ModelProfile,
    ProcessFactory,
    resolve_model_profile,
)
from mutation_forge.stage3.artifacts import canonical_hash, safe_value
from mutation_forge.stage3.isolation import (
    APP_SERVER_APPROVAL_POLICIES,
    APP_SERVER_SANDBOX_MODES,
    IsolatedCapsule,
    IsolationError,
)

FROZEN_STAGE4_MODEL: Final = "gpt-5.6-luna"
FROZEN_STAGE4_EFFORT: Final = "high"
STAGE4_TURN_TIMEOUT_SECONDS: Final = 600.0
INITIAL_TURN_BUDGET: Final = 32
REPAIR_TURN_BUDGET: Final = 32
TOTAL_TURN_BUDGET: Final = INITIAL_TURN_BUDGET + REPAIR_TURN_BUDGET
MAX_DIAGNOSTIC_BYTES: Final = 16 * 1024
_RECOVERY_PASSIVE_ITEMS: Final = {"userMessage", "agentMessage", "reasoning"}
_RECOVERY_GLOBAL_EVENTS: Final = {
    "account/updated",
    "account/rateLimits/updated",
    "configWarning",
    "remoteControl/status/changed",
}
_RECOVERY_DELTAS: Final = {
    "item/agentMessage/delta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded",
    "item/reasoning/textDelta",
}


class Stage4ProviderError(RuntimeError):
    """A bounded provider boundary error."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence = dict(evidence or {})


def _hash(value: object) -> str:
    """Hash canonical JSON, falling back to UTF-8 text for unusual values."""
    try:
        return canonical_hash(value)
    except Exception:
        return hashlib.sha256(repr(value).encode("utf-8", "replace")).hexdigest()


def _contains_unique_items(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            key == "uniqueItems" or _contains_unique_items(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_unique_items(child) for child in value)
    return False


def _codex_transport_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Add the redundant type required by Codex without changing the frozen schema."""
    value = cast(dict[str, Any], json.loads(json.dumps(schema)))
    properties = value.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Stage 4 output schema properties are missing")
    schema_version = properties.get("schema_version")
    if not isinstance(schema_version, dict):
        raise ValueError("Stage 4 schema_version contract is missing")
    if schema_version.get("const") != "stage4.generated_policy.v1":
        raise ValueError("Stage 4 schema_version constant drifted")
    declared_type = schema_version.get("type")
    if declared_type not in (None, "string"):
        raise ValueError("Stage 4 schema_version type is incompatible")
    schema_version["type"] = "string"
    return value


def _read_jsonl_mappings(
    path: Path,
    *,
    max_bytes: int,
    max_lines: int = 10_000,
) -> tuple[Mapping[str, Any], ...] | None:
    """Read one bounded retained JSONL stream without accepting partial lines."""

    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or len(lines) > max_lines:
            return None
        values = tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    if not all(isinstance(value, Mapping) for value in values):
        return None
    return cast(tuple[Mapping[str, Any], ...], values)


def _derived_artifact_prefix(request: Mapping[str, Any]) -> str:
    """Build a deterministic, bounded filename prefix for one generation turn."""
    identity = {
        key: request.get(key, "")
        for key in ("campaign_id", "generation", "slot", "phase", "idempotency_key")
    }
    digest = _hash(identity)[:16]

    def clean(value: object, fallback: str, limit: int) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
        text = text.strip("._-")[:limit]
        return text or fallback

    campaign = clean(request.get("campaign_id", ""), "campaign", 16)
    generation = clean(request.get("generation", ""), "g", 8)
    slot = clean(request.get("slot", ""), "slot", 12)
    phase = clean(request.get("phase", ""), "initial", 10)
    # Keep the digest at the end so truncating a human-readable component
    # cannot make two idempotency keys share an artifact path.
    return f"s4-{campaign}-{generation}-{slot}-{phase}-{digest}"


def _bounded_diagnostics(
    values: Any, *, limit: int = MAX_DIAGNOSTIC_BYTES
) -> tuple[Mapping[str, Any], ...]:
    """Return redacted, bounded diagnostic records from the Stage 3 adapter."""
    if not isinstance(values, (list, tuple)):
        return ()
    result: list[Mapping[str, Any]] = []
    used = 0
    for raw in values[:200]:
        value = safe_value(raw)
        if not isinstance(value, Mapping):
            value = {"value": value}
        # Diagnostics are telemetry only.  Do not let one malformed server
        # field consume the artifact budget or leak a private path.
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        remaining = limit - used
        if remaining <= 0:
            break
        if len(payload.encode()) > remaining:
            payload = payload.encode()[: max(0, remaining - 1)].decode("utf-8", "ignore")
            payload += "…"
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            item = {"diagnostic": payload}
        result.append(cast(Mapping[str, Any], item))
        used += len(payload.encode())
    return tuple(result)


def infrastructure_retry_eligible(evidence: Mapping[str, Any] | None) -> bool:
    """Return whether host evidence permits one replacement request.

    A retry is safe only when the App Server explicitly reports an uncharged
    attempt.  Missing or ambiguous evidence is fail-closed; in particular a
    turn id, content, or any non-zero usage prevents replacement.
    """
    if not isinstance(evidence, Mapping):
        return False
    if evidence.get("accepted") is not False or evidence.get("content") is not False:
        return False
    if evidence.get("charged") is True or evidence.get("accepted_turn") is True:
        return False
    if evidence.get("unauthorized_tool_approval") is True:
        return False
    if evidence.get("uncharged") is not True and evidence.get("app_server_uncharged") is not True:
        return False
    usage = evidence.get("usage")
    if usage is None:
        usage = evidence.get("usage_raw")
    if usage is not None and not isinstance(usage, Mapping):
        return False
    if isinstance(usage, Mapping):
        for value in usage.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value != 0:
                return False
    # App Server's accepted-turn evidence is authoritative.  A server may
    # expose an id for a rejected request, so only content/accepted flags gate.
    return True


# Short aliases used by campaign code and downstream callers.
retry_eligible = infrastructure_retry_eligible
can_retry_infrastructure = infrastructure_retry_eligible


def _artifact_refs(
    directory: str | Path | None,
    prefix: str,
    *,
    include_rollout: bool = True,
) -> tuple[str, ...]:
    """Return stable relative artifact names, never capsule/home paths."""
    if directory is None:
        return ()
    names = [
        "request.md",
        "request.json",
        "response.md",
        "response.json",
        "provider-raw.json",
        "usage.json",
        "codex-rpc.jsonl",
        "events.jsonl",
        "wire.jsonl",
        "stdout.jsonl",
        "stderr.txt",
        "transcript.sha256",
    ]
    if include_rollout:
        names.append("rollout.jsonl")
    clean = prefix.rstrip(".")
    return tuple(f"{clean}.{name}" if clean else name for name in names)


def _available_artifact_prefix(
    directory: str | Path | None,
    prefix: str,
) -> str:
    """Choose an additive retry prefix without overwriting retained evidence."""
    if directory is None:
        return prefix
    root = Path(directory)
    names = ("request.json", "response.json", "wire.jsonl", "transcript.sha256")

    def occupied(candidate: str) -> bool:
        return any((root / f"{candidate}.{name}").exists() for name in names)

    if not occupied(prefix):
        return prefix
    for attempt in range(1, 65):
        candidate = f"{prefix}.retry-{attempt:02d}"
        if not occupied(candidate):
            return candidate
    raise Stage4ProviderError("App Server artifact retry namespace is exhausted")


class Stage4CodexAppServerAdapter(CodexAppServerAdapter):
    """Stage 4 named adapter, inheriting all Stage 3 transport safeguards."""

    @staticmethod
    def _check_profile(profile: ModelProfile | str) -> None:
        selected = resolve_model_profile(profile) if isinstance(profile, str) else profile
        if selected.provider != "codex" or selected.model != FROZEN_STAGE4_MODEL:
            raise IsolationError(f"Stage 4 requires frozen model {FROZEN_STAGE4_MODEL}")
        if selected.effort != FROZEN_STAGE4_EFFORT:
            raise IsolationError("Stage 4 requires high reasoning effort")

    def start_thread(
        self, profile: ModelProfile | str, *, ephemeral: bool = True
    ) -> dict[str, Any]:
        self._check_profile(profile)
        return super().start_thread(profile, ephemeral=ephemeral)

    def generate(
        self,
        prompt: str,
        profile: ModelProfile | str,
        *,
        output_schema: Mapping[str, Any] | None = None,
    ) -> Any:
        self._check_profile(profile)
        if not isinstance(output_schema, Mapping) or not output_schema:
            raise ValueError("Stage 4 requires a non-empty structured output schema")
        return super().generate(prompt, profile, output_schema=output_schema)


# The longer name reads naturally in callers and is kept as a public alias.
Stage4AppServerAdapter = Stage4CodexAppServerAdapter


class Stage4AppServerProvider:
    """One-shot provider used by each independent Stage 4 slot.

    No adapter or capsule is retained between calls.  Consequently sharing a
    provider object across eight worker threads still creates eight isolated
    Codex homes and eight independent app-server processes.
    """

    def __init__(
        self,
        *,
        process_factory: ProcessFactory | None = None,
        auth_checker: Callable[[IsolatedCapsule], bool] | None = None,
        auth_json: str | Path | None = None,
        limits: AppServerLimits | None = None,
        artifact_dir: str | Path | None = None,
        artifact_prefix: str = "",
        artifact_root: str | Path | None = None,
        artifact_max_bytes: int = 32 * 1024 * 1024,
        sandbox_mode: str = "danger-full-access",
        approval_policy: str = "never",
    ) -> None:
        if sandbox_mode not in APP_SERVER_SANDBOX_MODES:
            raise ValueError("unsupported app-server sandbox mode")
        if approval_policy not in APP_SERVER_APPROVAL_POLICIES:
            raise ValueError("unsupported app-server approval policy")
        if process_factory is not None and auth_checker is None:
            raise ValueError("an injected process_factory requires an explicit auth_checker")
        self.process_factory = process_factory
        self.auth_checker = auth_checker
        self.auth_json = auth_json
        self.limits = limits or AppServerLimits(
            turn_timeout=STAGE4_TURN_TIMEOUT_SECONDS,
        )
        self.artifact_dir = artifact_dir
        self.artifact_prefix = artifact_prefix
        self.artifact_root = artifact_root
        self.artifact_max_bytes = artifact_max_bytes
        self.sandbox_mode = sandbox_mode
        self.approval_policy = approval_policy

    @staticmethod
    def _usage(raw: Mapping[str, Any]) -> dict[str, Any]:
        # Preserve every field emitted by this App Server version.  The Stage 3
        # TokenUsage object validates the required fields, while ``raw`` keeps
        # newly added usage counters intact for exact accounting.
        return dict(raw)

    def _request_values(
        self, request: Mapping[str, Any]
    ) -> tuple[str, str, Mapping[str, Any], str, str]:
        prompt = request.get("prompt")
        system = request.get("system_prompt", request.get("systemPrompt"))
        schema = request.get("output_schema", request.get("outputSchema"))
        model = request.get("model", FROZEN_STAGE4_MODEL)
        effort = request.get("effort", request.get("reasoning_effort", FROZEN_STAGE4_EFFORT))
        # GenerationRequest is transport-independent and supplies only the
        # rendered prompt.  Use the frozen Stage 4 bundle when it omits the
        # transport fields; explicit values are still validated above.
        if system is None or schema is None:
            from mutation_forge.stage4.prompts import load_prompt_bundle

            bundle = load_prompt_bundle()
            system = bundle.system if system is None else system
            schema = bundle.output_schema if schema is None else schema
        if (
            not isinstance(prompt, str)
            or not prompt
            or not isinstance(system, str)
            or not system
            or not isinstance(schema, Mapping)
            or not isinstance(model, str)
            or not isinstance(effort, str)
        ):
            raise ValueError("invalid Stage 4 generation request")
        if _contains_unique_items(schema):
            raise ValueError("Stage 4 output schema cannot contain uniqueItems")
        if model != FROZEN_STAGE4_MODEL or effort != FROZEN_STAGE4_EFFORT:
            raise Stage4ProviderError(
                f"Stage 4 generation requires {FROZEN_STAGE4_MODEL}:{FROZEN_STAGE4_EFFORT}"
            )
        return prompt, system, schema, model, effort

    def _load_completed_timeout_result(
        self,
        *,
        root: Path,
        retained_prefix: str,
        response_value: Mapping[str, Any],
        request: Mapping[str, Any],
        model: str,
        effort: str,
    ) -> Mapping[str, Any] | None:
        """Recover a remotely completed turn that the host timed out while logging."""

        if (
            response_value.get("status") != "error"
            or response_value.get("error_type") != "TurnError"
            or response_value.get("error") != "turn timed out"
            or response_value.get("accepted") is not True
            or response_value.get("content") is not False
            or response_value.get("usage") != {}
        ):
            return None
        events = _read_jsonl_mappings(
            root / f"{retained_prefix}.events.jsonl",
            max_bytes=self.limits.transcript_limit,
        )
        rpc = _read_jsonl_mappings(
            root / f"{retained_prefix}.codex-rpc.jsonl",
            max_bytes=self.limits.transcript_limit,
        )
        profile_path = root / f"{retained_prefix}.codex-profile.json"
        transcript_path = root / f"{retained_prefix}.transcript.sha256"
        if events is None or rpc is None:
            return None
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            transcript_sha256 = transcript_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return None
        doctor = request.get("appserver_doctor_sha256")
        if (
            not isinstance(profile, Mapping)
            or profile.get("model") != model
            or profile.get("effort") != effort
            or profile.get("protocolAuditSha256") != doctor
            or profile.get("artifactPrefix") != retained_prefix
            or re.fullmatch(r"[0-9a-f]{64}", transcript_sha256) is None
        ):
            return None

        thread_responses: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        turn_responses: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for message in rpc:
            result = message.get("result")
            if not isinstance(result, Mapping):
                continue
            thread = result.get("thread")
            turn = result.get("turn")
            if isinstance(thread, Mapping):
                thread_responses.append((message, thread))
            if isinstance(turn, Mapping):
                turn_responses.append((message, turn))
        if len(thread_responses) != 1 or len(turn_responses) != 1:
            return None
        thread_message, thread = thread_responses[0]
        turn_message, started_turn = turn_responses[0]
        thread_id = thread.get("id")
        turn_id = started_turn.get("id")
        request_id = turn_message.get("id")
        if (
            not isinstance(thread_id, str)
            or not thread_id
            or response_value.get("thread_id") != thread_id
            or thread.get("ephemeral") is not True
            or not isinstance(turn_id, str)
            or not turn_id
            or started_turn.get("status") != "inProgress"
            or not isinstance(started_turn.get("items"), list)
            or not isinstance(request_id, (int, str))
            or isinstance(request_id, bool)
        ):
            return None

        active: dict[str, str] = {}
        completed: set[str] = set()
        final_items: list[tuple[str, str]] = []
        usage_values: list[Mapping[str, Any]] = []
        completed_turns: list[Mapping[str, Any]] = []
        turn_started = 0
        for message in events:
            method = message.get("method")
            params = message.get("params")
            if not isinstance(method, str) or not isinstance(params, Mapping):
                return None
            event_thread = params.get("threadId", params.get("thread_id"))
            nested_turn = params.get("turn")
            event_turn = params.get("turnId", params.get("turn_id"))
            if event_turn is None and isinstance(nested_turn, Mapping):
                event_turn = nested_turn.get("id")
            if event_thread is not None and event_thread != thread_id:
                return None
            if event_turn is not None and event_turn != turn_id:
                return None
            if method in _RECOVERY_GLOBAL_EVENTS:
                continue
            if method in {"error", "model/rerouted"}:
                return None
            if method == "thread/started":
                nested_thread = params.get("thread")
                observed_thread = (
                    nested_thread.get("id")
                    if isinstance(nested_thread, Mapping)
                    else event_thread
                )
                if observed_thread != thread_id:
                    return None
                continue
            if method == "turn/started":
                if (
                    not isinstance(nested_turn, Mapping)
                    or nested_turn.get("id") != turn_id
                    or nested_turn.get("status") != "inProgress"
                    or not isinstance(nested_turn.get("items"), list)
                ):
                    return None
                turn_started += 1
                continue
            if method in {"item/started", "item/completed"}:
                item = params.get("item")
                if (
                    not isinstance(item, Mapping)
                    or not isinstance(item.get("id"), str)
                    or not isinstance(item.get("type"), str)
                    or item.get("type") not in _RECOVERY_PASSIVE_ITEMS
                ):
                    return None
                item_id = cast(str, item["id"])
                item_type = cast(str, item["type"])
                if method == "item/started":
                    if item_id in active or item_id in completed:
                        return None
                    active[item_id] = item_type
                    continue
                if active.pop(item_id, None) != item_type or item_id in completed:
                    return None
                completed.add(item_id)
                if item_type == "agentMessage" and item.get("phase") == "final_answer":
                    text = item.get("text")
                    if not isinstance(text, str) or not text:
                        return None
                    final_items.append((item_id, text))
                continue
            if method in _RECOVERY_DELTAS:
                item_id = params.get("itemId", params.get("item_id"))
                expected_type = (
                    "agentMessage"
                    if method == "item/agentMessage/delta"
                    else "reasoning"
                )
                if not isinstance(item_id, str) or active.get(item_id) != expected_type:
                    return None
                continue
            if method == "thread/tokenUsage/updated":
                token_usage = params.get("tokenUsage")
                last = (
                    token_usage.get("last")
                    if isinstance(token_usage, Mapping)
                    else None
                )
                if not isinstance(last, Mapping):
                    return None
                usage_values.append(last)
                continue
            if method == "thread/status/changed":
                status = params.get("status")
                status_type = (
                    status.get("type") if isinstance(status, Mapping) else None
                )
                if not isinstance(status_type, str) or status_type in {
                    "systemError",
                    "failed",
                    "interrupted",
                    "cancelled",
                }:
                    return None
                continue
            if method == "turn/completed":
                if not isinstance(nested_turn, Mapping):
                    return None
                completed_turns.append(nested_turn)
                continue
            return None

        if (
            turn_started != 1
            or active
            or len(final_items) != 1
            or len(completed_turns) != 1
            or not usage_values
        ):
            return None
        final_item_id, response_text = final_items[0]
        completed_turn = completed_turns[0]
        completed_items = completed_turn.get("items")
        if (
            completed_turn.get("id") != turn_id
            or completed_turn.get("status") != "completed"
            or not isinstance(completed_items, list)
        ):
            return None
        completed_ids = {
            item.get("id")
            for item in completed_items
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        if (
            completed_turn.get("itemsView") != "notLoaded"
            and final_item_id not in completed_ids
        ):
            return None

        usage_raw = usage_values[-1]
        required_usage = (
            "inputTokens",
            "cachedInputTokens",
            "outputTokens",
            "reasoningOutputTokens",
            "totalTokens",
        )
        if any(
            not isinstance(usage_raw.get(key), int)
            or isinstance(usage_raw.get(key), bool)
            or cast(int, usage_raw[key]) < 0
            for key in required_usage
        ):
            return None
        cache_write = usage_raw.get("cacheWriteInputTokens", 0)
        if (
            not isinstance(cache_write, int)
            or isinstance(cache_write, bool)
            or cache_write < 0
        ):
            return None
        usage = {
            **dict(usage_raw),
            "cacheWriteInputTokens": cache_write,
            "final": True,
            "partial": False,
        }
        response: Any = response_text
        try:
            decoded = json.loads(response_text)
        except (TypeError, ValueError):
            pass
        else:
            if isinstance(decoded, Mapping):
                response = dict(decoded)
        denied = response_value.get("unauthorized_tool_approval")
        if denied is not False:
            return None
        prompt_hashes = response_value.get("prompt_hashes")
        if not isinstance(prompt_hashes, Mapping):
            prompt_hashes = {
                key: response_value.get(key)
                for key in (
                    "prompt_sha256",
                    "system_prompt_sha256",
                    "output_schema_sha256",
                    "schema_sha256",
                    "transport_output_schema_sha256",
                    "request_sha256",
                )
            }
        total_tokens = cast(int, usage["totalTokens"])
        return {
            **dict(response_value),
            "response": response,
            "response_text": response_text,
            "raw_response": response_text,
            "status": "completed",
            "accepted": True,
            "accepted_turn": True,
            "charged": total_tokens > 0,
            "content": True,
            "content_proof": {
                "present": True,
                "bytes": len(response_text.encode()),
                "sha256": _hash(response_text),
            },
            "usage": usage,
            "usage_proof": {
                "complete": True,
                "charged": total_tokens > 0,
                "totalTokens": total_tokens,
            },
            "request_id": request_id,
            "thread_id": thread_id,
            "session_id": thread.get("sessionId"),
            "turn_id": turn_id,
            "provider_request_id": str(request_id),
            "provider_thread_id": thread_id,
            "provider_turn_id": turn_id,
            "model": model,
            "effort": effort,
            "prompt_hashes": dict(prompt_hashes),
            "transcript_sha256": transcript_sha256,
            "transport_sha256": transcript_sha256,
            "appserver_doctor_sha256": doctor,
            "retained_completed_turn_recovery": retained_prefix,
            "host_timeout_after_remote_completion": True,
            "original_error": {
                "error_type": response_value.get("error_type"),
                "error": response_value.get("error"),
            },
        }

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt, system, schema, model, effort = self._request_values(request)
        transport_schema = _codex_transport_schema(schema)
        prefix = (
            str(request["artifact_prefix"])
            if "artifact_prefix" in request
            else _derived_artifact_prefix(request)
        )
        directory = request.get("artifact_dir", self.artifact_dir)
        prefix = _available_artifact_prefix(
            directory if isinstance(directory, (str, Path)) else None,
            prefix,
        )
        root = request.get("artifact_root", self.artifact_root)
        doctor = request.get("appserver_doctor_sha256")
        adapter = Stage4CodexAppServerAdapter(
            process_factory=self.process_factory,
            auth_checker=self.auth_checker,
            auth_json=self.auth_json,
            limits=self.limits,
            base_instructions=system,
            artifact_dir=directory,
            artifact_prefix=prefix,
            artifact_root=root if isinstance(root, (str, Path)) else None,
            artifact_max_bytes=self.artifact_max_bytes,
            protocol_audit_sha256=doctor if isinstance(doctor, str) else None,
            sandbox_mode=self.sandbox_mode,
            approval_policy=self.approval_policy,
        )
        prompt_hashes = {
            "prompt_sha256": _hash(prompt),
            "system_prompt_sha256": _hash(system),
            "output_schema_sha256": _hash(schema),
            "schema_sha256": _hash(schema),
            "transport_output_schema_sha256": _hash(transport_schema),
            "request_sha256": _hash(
                {
                    k: request[k]
                    for k in sorted(request)
                    if k not in {"artifact_dir", "artifact_root", "artifact_prefix"}
                }
            ),
        }
        if adapter.logger:
            adapter.logger.text("request.md", prompt)
            adapter.logger.document(
                "request.json",
                {
                    "model": model,
                    "effort": effort,
                    "prompt": prompt,
                    "system_prompt": system,
                    "output_schema": transport_schema,
                    "frozen_output_schema": dict(schema),
                    "transport_schema_projected": transport_schema != schema,
                    **prompt_hashes,
                },
            )
        try:
            result = adapter.generate(
                prompt,
                ModelProfile("codex", model, effort),
                output_schema=transport_schema,
            )
            response_text = result.text
            response_value: Any = response_text
            try:
                decoded = json.loads(response_text)
            except (TypeError, ValueError):
                pass
            else:
                if isinstance(decoded, Mapping):
                    response_value = dict(decoded)
            usage = self._usage(result.usage.raw)
            usage.setdefault("inputTokens", result.usage.input_tokens)
            usage.setdefault("cachedInputTokens", result.usage.cached_input_tokens)
            usage.setdefault("cacheWriteInputTokens", result.usage.cache_write_input_tokens)
            usage.setdefault("outputTokens", result.usage.output_tokens)
            usage.setdefault("reasoningOutputTokens", result.usage.reasoning_output_tokens)
            usage.setdefault("totalTokens", result.usage.total_tokens)
            usage.update({"final": result.usage.final, "partial": result.usage.partial})
            diagnostics = _bounded_diagnostics(result.diagnostics)
            denied = any(d.get("event") == "denied_server_request" for d in diagnostics)
            payload: dict[str, Any] = {
                # GenerationCoordinator consumes ``response`` as the strict
                # decoded object; retain the exact App Server text beside it
                # for replay/provenance and malformed-output diagnostics.
                "response": response_value,
                "response_text": response_text,
                "raw_response": response_text,
                "accepted": True,
                "accepted_turn": True,
                "charged": result.usage.total_tokens > 0,
                "content": bool(result.text),
                "content_proof": {
                    "present": bool(result.text),
                    "bytes": len(result.text.encode()),
                    "sha256": _hash(result.text),
                },
                "usage": usage,
                "usage_proof": {
                    "complete": result.usage.final and not result.usage.partial,
                    "charged": result.usage.total_tokens > 0,
                    "totalTokens": result.usage.total_tokens,
                },
                "status": "completed",
                "request_id": result.request_id,
                "thread_id": result.thread_id,
                "session_id": result.session_id,
                "turn_id": result.turn_id,
                "provider_request_id": str(result.request_id),
                "provider_thread_id": result.thread_id,
                "provider_turn_id": result.turn_id,
                "model": model,
                "effort": effort,
                "unauthorized_tool_approval": denied,
                "tool_approval_requested": denied,
                "tool_approval_granted": False,
                "diagnostics": diagnostics,
                "artifact_refs": _artifact_refs(directory, prefix),
                "transcript_sha256": adapter.logger.transcript_sha256 if adapter.logger else None,
                "transport_sha256": adapter.logger.transcript_sha256 if adapter.logger else None,
                "prompt_hashes": prompt_hashes,
                **prompt_hashes,
                "appserver_doctor_sha256": doctor,
            }
            if adapter.logger:
                adapter.logger.text("response.md", result.text)
                adapter.logger.document("response.json", payload)
                adapter.logger.document("usage.json", usage)
                adapter.logger.document(
                    "provider-raw.json",
                    {
                        "usage": usage,
                        "diagnostics": list(diagnostics),
                        **prompt_hashes,
                        "thread_id": result.thread_id,
                        "turn_id": result.turn_id,
                        "request_id": result.request_id,
                    },
                )
            return payload
        except Exception as error:
            diagnostics = _bounded_diagnostics(adapter.diagnostics)
            denied = any(d.get("event") == "denied_server_request" for d in diagnostics)
            partial = adapter.partial_result
            metadata = adapter.inspect_metadata()
            usage_state = adapter.inspect_usage()
            observed_usage = usage_state.get("raw")
            partial_usage = (
                self._usage(partial.usage.raw)
                if partial is not None
                else self._usage(cast(Mapping[str, Any], observed_usage))
                if isinstance(observed_usage, Mapping)
                else {}
            )
            if partial is not None:
                partial_usage.setdefault("inputTokens", partial.usage.input_tokens)
                partial_usage.setdefault(
                    "cachedInputTokens",
                    partial.usage.cached_input_tokens,
                )
                partial_usage.setdefault("outputTokens", partial.usage.output_tokens)
                partial_usage.setdefault(
                    "reasoningOutputTokens",
                    partial.usage.reasoning_output_tokens,
                )
                partial_usage.setdefault("totalTokens", partial.usage.total_tokens)
                partial_usage.update(
                    {"final": partial.usage.final, "partial": partial.usage.partial}
                )
            elif isinstance(observed_usage, Mapping):
                partial_usage.update(
                    {
                        "final": usage_state.get("final") is True,
                        "partial": usage_state.get("partial") is True,
                    }
                )
            observed_total = partial_usage.get("totalTokens")
            observed_charged = (
                isinstance(observed_total, int)
                and not isinstance(observed_total, bool)
                and observed_total > 0
            )
            thread_id = (
                partial.thread_id
                if partial is not None
                else metadata.get("threadId")
            )
            turn_evidence_observed = (
                partial is not None
                or isinstance(thread_id, str)
                or isinstance(observed_usage, Mapping)
            )
            error_evidence = {
                # Once a private thread or usage event exists, replacement is
                # fail-closed even if no final turn envelope was assembled.
                "accepted": turn_evidence_observed,
                "charged": observed_charged,
                "content": bool(partial is not None and partial.text),
                "uncharged": not turn_evidence_observed,
                "usage": partial_usage,
                "request_id": partial.request_id if partial is not None else None,
                "thread_id": thread_id,
                "turn_id": partial.turn_id if partial is not None else None,
                "unauthorized_tool_approval": denied,
            }
            if adapter.logger:
                adapter.logger.document(
                    "response.json",
                    {
                        **error_evidence,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error)[:512],
                        "diagnostics": list(diagnostics),
                        "unauthorized_tool_approval": denied,
                        "tool_approval_requested": denied,
                        "tool_approval_granted": False,
                        "artifact_refs": _artifact_refs(directory, prefix),
                        **prompt_hashes,
                    },
                )
                adapter.logger.document("usage.json", partial_usage)
            raise Stage4ProviderError(
                f"{type(error).__name__}: {str(error)[:512]}",
                evidence=error_evidence,
            ) from error
        finally:
            adapter.close()

    def load_retained_result(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Load an accepted additive retry artifact after checkpoint persistence failed."""
        prompt, system, schema, model, effort = self._request_values(request)
        transport_schema = _codex_transport_schema(schema)
        directory = request.get("artifact_dir", self.artifact_dir)
        if not isinstance(directory, (str, Path)):
            return None
        root = Path(directory)
        prefix = (
            str(request["artifact_prefix"])
            if "artifact_prefix" in request
            else _derived_artifact_prefix(request)
        )
        expected_request_sha256 = _hash(
            {
                key: request[key]
                for key in sorted(request)
                if key not in {"artifact_dir", "artifact_root", "artifact_prefix"}
            }
        )
        candidates = sorted(
            root.glob(f"{prefix}.retry-*.response.json"),
            reverse=True,
        )
        for response_path in candidates:
            try:
                response_path.resolve().relative_to(root.resolve())
                retained_prefix = response_path.name.removesuffix(".response.json")
                request_path = root / f"{retained_prefix}.request.json"
                request_path.resolve().relative_to(root.resolve())
                response_value = json.loads(response_path.read_text(encoding="utf-8"))
                request_value = json.loads(request_path.read_text(encoding="utf-8"))
            except (
                OSError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            if not isinstance(response_value, Mapping) or not isinstance(
                request_value,
                Mapping,
            ):
                continue
            usage = response_value.get("usage")
            expected_hashes = {
                "prompt_sha256": _hash(prompt),
                "system_prompt_sha256": _hash(system),
                "output_schema_sha256": _hash(schema),
                "schema_sha256": _hash(schema),
                "request_sha256": expected_request_sha256,
            }
            projected_metadata = (
                request_value.get("output_schema") == transport_schema
                and request_value.get("frozen_output_schema") == schema
                and request_value.get("transport_output_schema_sha256")
                == _hash(transport_schema)
                and response_value.get("transport_output_schema_sha256")
                == _hash(transport_schema)
            )
            legacy_rejected_metadata = (
                response_value.get("status") == "error"
                and request_value.get("output_schema") == schema
                and "frozen_output_schema" not in request_value
                and "transport_output_schema_sha256" not in request_value
                and "transport_output_schema_sha256" not in response_value
            )
            if (
                response_value.get("accepted") is not True
                or not isinstance(response_value.get("content"), bool)
                or not isinstance(response_value.get("charged"), bool)
                or not isinstance(usage, Mapping)
                or response_value.get("unauthorized_tool_approval") is not False
                or request_value.get("model") != model
                or request_value.get("effort") != effort
                or request_value.get("prompt") != prompt
                or request_value.get("system_prompt") != system
                or not (projected_metadata or legacy_rejected_metadata)
                or any(
                    response_value.get(key) != expected
                    or request_value.get(key) != expected
                    for key, expected in expected_hashes.items()
                )
            ):
                continue
            status = response_value.get("status")
            if status == "completed":
                response_text = response_value.get("response_text")
                content_proof = response_value.get("content_proof")
                if (
                    not isinstance(response_text, str)
                    or not isinstance(content_proof, Mapping)
                    or content_proof.get("sha256") != _hash(response_text)
                    or content_proof.get("bytes") != len(response_text.encode())
                ):
                    continue
                return dict(response_value)
            if (
                status == "error"
                and isinstance(response_value.get("error_type"), str)
                and isinstance(response_value.get("error"), str)
            ):
                recovered = self._load_completed_timeout_result(
                    root=root,
                    retained_prefix=retained_prefix,
                    response_value=response_value,
                    request=request,
                    model=model,
                    effort=effort,
                )
                if recovered is not None:
                    return recovered
                return {
                    **dict(response_value),
                    "status": "infrastructure",
                    "error": (
                        f"{response_value['error_type']}: "
                        f"{response_value['error']}"
                    ),
                    "retained_artifact_recovery": response_path.name,
                }
        return None

    def repair(
        self, request: Mapping[str, Any], diagnostics: tuple[Mapping[str, Any], ...]
    ) -> Mapping[str, Any]:
        if not diagnostics:
            raise ValueError("repair requires bounded diagnostics")
        prompt = request.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("repair request prompt must be a string")
        bounded = _bounded_diagnostics(diagnostics)
        prefix = str(
            request.get("artifact_prefix")
            or self.artifact_prefix
            or _derived_artifact_prefix(request)
        ).rstrip(".")
        repair_prompt = (
            prompt
            + "\n\nRepair only the output listed below.\n"
            + json.dumps(list(bounded), sort_keys=True, separators=(",", ":"))
        )
        return self.generate(
            {
                **request,
                "prompt": repair_prompt,
                "artifact_prefix": f"{prefix}.repair" if prefix else "repair",
            }
        )


# Protocol-friendly names used by Stage 3-compatible generation code.
AppServerGenerationProvider = Stage4AppServerProvider
CodexAppServerProvider = Stage4AppServerProvider


__all__ = [
    "AppServerLimits",
    "IsolatedCapsule",
    "Stage4AppServerAdapter",
    "Stage4CodexAppServerAdapter",
    "Stage4AppServerProvider",
    "Stage4ProviderError",
    "AppServerGenerationProvider",
    "CodexAppServerProvider",
    "FROZEN_STAGE4_MODEL",
    "FROZEN_STAGE4_EFFORT",
    "STAGE4_TURN_TIMEOUT_SECONDS",
    "INITIAL_TURN_BUDGET",
    "REPAIR_TURN_BUDGET",
    "TOTAL_TURN_BUDGET",
    "infrastructure_retry_eligible",
    "retry_eligible",
    "can_retry_infrastructure",
]
