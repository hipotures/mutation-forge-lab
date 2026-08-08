"""Explicit guarded experiment route for the ordinary-Python preview."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from mutation_forge.backends.base import GraphBackend
from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.config import validate_experiment_id
from mutation_forge.experiment.json_io import read_json, write_json
from mutation_forge.models import JsonValue

from .contracts import (
    PYTHON_EXPERIMENT_PROTOCOL_ID,
    PYTHON_WORKSPACE_SCHEMA_VERSION,
    PythonWorkspaceProtocolError,
    require_python_workspace_schema_version,
)
from .provenance import M5_PROVENANCE_FILENAME, ensure_m5_acceptance_provenance
from .runtime_contracts import PolicyRuntimeLimitsV1
from .search import (
    M5_REPORT_PROTOCOL_ID,
    DevelopmentCaseV1,
    M5ScientificEvaluator,
    M5SearchProvider,
    run_m5_search,
)
from .search_provider import (
    CodexM5SearchProvider,
    PythonPanelScientificEvaluator,
    specification_ack_schema,
)

PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION = (
    "mforge.experiment.native_python_preview_config.v1"
)
PYTHON_PREVIEW_STATE_SCHEMA_VERSION = (
    "mforge.experiment.status.native_python_preview.v1"
)
PYTHON_PREVIEW_PROTOCOL_VERSION = "mforge.native.python_preview.v1"
PYTHON_PREVIEW_MODE = "ordinary-python"
_STATE_NAME = "python-preview-state.json.gz"
_CONFIG_NAME = "python-preview.toml"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TERMINAL_CANDIDATE_STATUSES = frozenset(
    {
        "evaluated",
        "contract_invalid",
        "duplicate",
        "provider_failed",
        "missing",
        "evaluation_infrastructure_failure",
    }
)
_V2_TOP_LEVEL_FIELDS = frozenset(
    {"kind", "preset", "run", "model", "search", "evaluation", "resources"}
)
_PUBLIC_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "workspace_schema_version",
        "protocol",
        "protocol_version",
        "preview_mode",
        "preview_active",
        "native_v2_default",
        "dsl_runtime_used",
        "safe_api_expanded",
        "state",
        "resumable",
        "run_terminal",
        "terminal_reason",
        "scientific_result_kind",
        "scientific_success",
        "last_error",
        "last_boundary",
        "resume_attempts",
    }
)

type ProviderFactory = Callable[
    ["PythonPreviewConfig", str],
    M5SearchProvider,
]
type BackendFactory = Callable[["PythonPreviewConfig"], GraphBackend]
type EvaluatorFactory = Callable[
    ["PythonPreviewConfig", GraphBackend],
    M5ScientificEvaluator,
]
type ProvenanceGuard = Callable[..., Mapping[str, JsonValue]]


@dataclass(frozen=True, slots=True)
class PythonPreviewConfig:
    """Strict opt-in configuration for the bounded Python preview."""

    schema_version: str
    protocol: str
    exp_id: str
    workspace: Path
    model: str
    effort: str
    timeout_seconds: float
    heg_repo: Path
    source_path: Path
    source_bytes: bytes = field(repr=False, compare=False)

    @property
    def experiment_root(self) -> Path:
        return self.workspace / self.exp_id

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source_bytes).hexdigest()


class PythonPreviewWorkspaceError(RuntimeError):
    """A workspace cannot be interpreted by the Python preview protocol."""


def _raw_config(path: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    source_path = Path(path).resolve()
    source_bytes = source_path.read_bytes()
    value = tomllib.loads(source_bytes.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Python preview configuration must be a TOML table")
    return source_path, source_bytes, value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _resolved_path(value: object, name: str, source_dir: Path) -> Path:
    raw = Path(_string(value, name))
    return (source_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()


def load_python_preview_config(
    path: str | Path,
) -> PythonPreviewConfig:
    """Parse the separate fail-closed Python preview configuration."""

    source_path, source_bytes, raw = _raw_config(path)
    mixed = sorted(_V2_TOP_LEVEL_FIELDS.intersection(raw))
    if mixed:
        raise ValueError(
            f"Python preview configuration cannot contain Native v2 fields: {mixed}"
        )
    allowed = {
        "schema_version",
        "protocol",
        "exp_id",
        "workspace",
        "python_preview",
    }
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported top-level fields: {unknown}")
    if raw.get("schema_version") != PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Python preview requires schema_version "
            f"{PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION!r}"
        )
    if raw.get("protocol") != PYTHON_EXPERIMENT_PROTOCOL_ID:
        raise ValueError(
            f"Python preview requires protocol {PYTHON_EXPERIMENT_PROTOCOL_ID!r}"
        )
    exp_id = validate_experiment_id(raw.get("exp_id"))
    workspace = _resolved_path(
        raw.get("workspace"),
        "workspace",
        source_path.parent,
    )
    preview_value = raw.get("python_preview")
    if not isinstance(preview_value, dict):
        raise ValueError("[python_preview] must be a table")
    preview = cast(dict[str, Any], preview_value)
    allowed_preview = {"model", "effort", "timeout_seconds", "heg_repo"}
    unknown_preview = sorted(set(preview).difference(allowed_preview))
    if unknown_preview:
        raise ValueError(
            f"unsupported [python_preview] fields: {unknown_preview}"
        )
    model = _string(preview.get("model"), "python_preview.model")
    effort = _string(preview.get("effort"), "python_preview.effort")
    if effort not in {"minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("python_preview.effort is unsupported")
    timeout_seconds = _positive_number(
        preview.get("timeout_seconds"),
        "python_preview.timeout_seconds",
    )
    heg_repo = _resolved_path(
        preview.get("heg_repo"),
        "python_preview.heg_repo",
        source_path.parent,
    )
    return PythonPreviewConfig(
        PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION,
        PYTHON_EXPERIMENT_PROTOCOL_ID,
        exp_id,
        workspace,
        model,
        effort,
        timeout_seconds,
        heg_repo,
        source_path,
        source_bytes,
    )


def _state_path(config: PythonPreviewConfig) -> Path:
    return config.experiment_root / _STATE_NAME


def _stored_config_path(config: PythonPreviewConfig) -> Path:
    return config.experiment_root / _CONFIG_NAME


def _base_state(config: PythonPreviewConfig) -> dict[str, Any]:
    return {
        "schema_version": PYTHON_PREVIEW_STATE_SCHEMA_VERSION,
        "workspace_schema_version": PYTHON_WORKSPACE_SCHEMA_VERSION,
        "protocol": PYTHON_EXPERIMENT_PROTOCOL_ID,
        "protocol_version": PYTHON_PREVIEW_PROTOCOL_VERSION,
        "preview_mode": PYTHON_PREVIEW_MODE,
        "preview_active": True,
        "native_v2_default": True,
        "dsl_runtime_used": False,
        "safe_api_expanded": False,
        "state": "not_created",
        "resumable": False,
        "run_terminal": False,
        "terminal_reason": None,
        "scientific_result_kind": "NONE",
        "scientific_success": False,
        "last_error": None,
        "last_boundary": None,
        "resume_attempts": 0,
        "config_sha256": config.source_sha256,
    }


def _public(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if key in _PUBLIC_STATE_FIELDS
    }


def _safe_error(error: Exception, config: PythonPreviewConfig) -> str:
    del config
    return type(error).__name__


def _write_state(
    config: PythonPreviewConfig,
    state: Mapping[str, Any],
) -> None:
    write_json(_state_path(config), dict(state))


def _load_state(config: PythonPreviewConfig) -> dict[str, Any]:
    path = _state_path(config)
    if not path.is_file():
        raise PythonPreviewWorkspaceError(
            "existing workspace is not an ordinary-Python preview workspace; "
            "JSON-DSL and Native v2 workspaces cannot be migrated or reinterpreted"
        )
    raw = read_json(path)
    if not isinstance(raw, Mapping):
        raise PythonPreviewWorkspaceError("Python preview state is not an object")
    unknown = sorted(set(raw).difference(_base_state(config)))
    if unknown:
        raise PythonPreviewWorkspaceError(
            f"Python preview state contains unsupported fields: {unknown}"
        )
    try:
        require_python_workspace_schema_version(raw.get("workspace_schema_version"))
    except PythonWorkspaceProtocolError as error:
        raise PythonPreviewWorkspaceError(str(error)) from error
    if (
        raw.get("schema_version") != PYTHON_PREVIEW_STATE_SCHEMA_VERSION
        or raw.get("protocol") != PYTHON_EXPERIMENT_PROTOCOL_ID
        or raw.get("protocol_version") != PYTHON_PREVIEW_PROTOCOL_VERSION
        or raw.get("preview_mode") != PYTHON_PREVIEW_MODE
    ):
        raise PythonPreviewWorkspaceError(
            "Python preview workspace protocol does not match this runtime"
        )
    if raw.get("config_sha256") != config.source_sha256:
        raise PythonPreviewWorkspaceError(
            "Python preview configuration changed; create a fresh workspace"
        )
    stored = _stored_config_path(config)
    if (
        not stored.is_file()
        or hashlib.sha256(stored.read_bytes()).hexdigest() != config.source_sha256
    ):
        raise PythonPreviewWorkspaceError(
            "Python preview workspace configuration identity mismatch"
        )
    return dict(raw)


def _initialize_workspace(config: PythonPreviewConfig) -> dict[str, Any]:
    config.workspace.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{config.exp_id}.native-python.",
            dir=config.workspace,
        )
    )
    initial = {**_base_state(config), "state": "ready", "resumable": True}
    try:
        (temporary / _CONFIG_NAME).write_bytes(config.source_bytes)
        write_json(temporary / _STATE_NAME, initial)
        os.replace(temporary, config.experiment_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return initial


def _load_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path)
    except (OSError, ValueError, RecursionError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _candidates(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(
        root.glob("generations/generation-*/slot-*/candidate.json.gz")
    ):
        value = _load_mapping(path)
        if value is not None:
            values.append(value)
    return values


def _manifests(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(
        root.glob("generations/generation-*/manifest.json.gz")
    ):
        value = _load_mapping(path)
        if value is not None:
            values.append(value)
    return values


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _usage_total(
    candidates: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any] | None,
) -> dict[str, int]:
    keys = (
        "inputTokens",
        "cachedInputTokens",
        "cacheWriteInputTokens",
        "outputTokens",
        "reasoningOutputTokens",
        "totalTokens",
    )
    totals = {key: 0 for key in keys}
    attempts: list[Mapping[str, Any]] = []
    if anchor is not None:
        attempts.append(anchor)
    for candidate in candidates:
        raw_attempts = candidate.get("provider_attempts")
        if isinstance(raw_attempts, Sequence) and not isinstance(
            raw_attempts, str | bytes
        ):
            attempts.extend(
                item for item in raw_attempts if isinstance(item, Mapping)
            )
    for attempt in attempts:
        usage = attempt.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in keys:
            totals[key] += _nonnegative_int(usage.get(key))
    return totals


def _evaluation_telemetry(root: Path) -> dict[str, Any]:
    worker_starts = 0
    worker_rotations = 0
    worker_failures = 0
    worker_timeouts = 0
    maximum_rss_kib = 0
    policy_invocations = 0
    graph_score_attempts = 0
    unique_graph_scores = 0
    for path in sorted(
        root.glob(
            "generations/generation-*/slot-*/evaluations/*.json.gz"
        )
    ):
        evaluation = _load_mapping(path)
        if evaluation is None:
            continue
        worker = evaluation.get("worker_telemetry")
        if isinstance(worker, Mapping):
            worker_starts += 1
            worker_rotations += _nonnegative_int(worker.get("rotations"))
            worker_failures += _nonnegative_int(worker.get("failures"))
            maximum_rss_kib = max(
                maximum_rss_kib,
                _nonnegative_int(worker.get("worker_rss_kib")),
            )
        scientific = evaluation.get("scientific_result")
        if not isinstance(scientific, Mapping):
            continue
        steps = scientific.get("steps")
        if isinstance(steps, Sequence) and not isinstance(steps, str | bytes):
            policy_invocations += len(steps)
        graph_score_attempts += _nonnegative_int(
            scientific.get("score_attempts")
        )
        unique_graph_scores += _nonnegative_int(
            scientific.get("unique_graph_scores")
        )
        failure = scientific.get("failure")
        if isinstance(failure, Mapping) and failure.get("code") == "PROPOSE_TIMEOUT":
            worker_timeouts += 1
    return {
        "starts": worker_starts,
        "rotations": worker_rotations,
        "failures": worker_failures,
        "timeouts": worker_timeouts,
        "maximum_rss_kib": maximum_rss_kib,
        "policy_invocations": policy_invocations,
        "graph_score_attempts": graph_score_attempts,
        "unique_graph_scores": unique_graph_scores,
    }


def _provider_telemetry(
    root: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor = _load_mapping(root / "anchor.json.gz")
    attempts = sum(
        len(raw)
        for candidate in candidates
        if isinstance((raw := candidate.get("provider_attempts")), Sequence)
        and not isinstance(raw, str | bytes)
    )
    repairs = sum(_nonnegative_int(item.get("repairs")) for item in candidates)
    warnings = sum(_nonnegative_int(item.get("warnings")) for item in candidates)
    provider_state = _load_mapping(
        root / "provider-runtime" / "provider-state.json.gz"
    )
    counters = (
        provider_state.get("telemetry")
        if isinstance(provider_state, Mapping)
        and isinstance(provider_state.get("telemetry"), Mapping)
        else {}
    )
    return {
        "turns": attempts + (1 if anchor is not None else 0),
        "candidate_turns": attempts,
        "contract_repairs": repairs,
        "transport_retries": _nonnegative_int(
            cast(Mapping[str, Any], counters).get("transport_retries")
        ),
        "warnings": warnings
        + _nonnegative_int(
            cast(Mapping[str, Any], anchor or {}).get("warnings")
        ),
        "process_restarts": _nonnegative_int(
            cast(Mapping[str, Any], counters).get("process_restarts")
        ),
        "thread_resume_attempts": _nonnegative_int(
            cast(Mapping[str, Any], counters).get("thread_resume_attempts")
        ),
        "forks": len(
            list(
                root.glob(
                    "generations/generation-*/slot-*/provider-*/"
                    "fork/m5-fork-result.json.gz"
                )
            )
        ),
        "usage": _usage_total(candidates, anchor),
    }


def _program_projection(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates[:16]:
        profile = candidate.get("behavior_profile")
        result.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "generation": candidate.get("generation"),
                "slot": candidate.get("slot"),
                "kind": candidate.get("kind"),
                "status": candidate.get("status"),
                "parent_candidate_id": candidate.get("parent_candidate_id"),
                "program_hash": candidate.get("program_hash"),
                "behavior_signature": candidate.get("behavior_signature"),
                "fitness_interval": (
                    profile.get("fitness_interval")
                    if isinstance(profile, Mapping)
                    else None
                ),
                "selector_frequencies": (
                    profile.get("selector_frequencies")
                    if isinstance(profile, Mapping)
                    else {}
                ),
                "action_frequencies": (
                    profile.get("action_frequencies")
                    if isinstance(profile, Mapping)
                    else {}
                ),
                "accepted_rewrites": (
                    _nonnegative_int(profile.get("accepted_rewrite_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "rejected_rewrites": (
                    _nonnegative_int(profile.get("rejected_rewrite_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "no_plan": (
                    _nonnegative_int(profile.get("no_plan_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "program_failures": (
                    _nonnegative_int(profile.get("program_failure_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
                "illegal_final_states": (
                    _nonnegative_int(profile.get("illegal_final_state_count"))
                    if isinstance(profile, Mapping)
                    else 0
                ),
            }
        )
    return result


def _progress(
    config: PythonPreviewConfig,
    retained_state: Mapping[str, Any],
) -> dict[str, Any]:
    root = config.experiment_root
    manifests = _manifests(root)
    candidates = _candidates(root)
    statuses = Counter(str(item.get("status", "unknown")) for item in candidates)
    planned = sum(
        len(cast(Sequence[object], manifest.get("slots", ())))
        for manifest in manifests
        if isinstance(manifest.get("slots"), Sequence)
        and not isinstance(manifest.get("slots"), str | bytes)
    )
    terminal = sum(
        count for status, count in statuses.items() if status in _TERMINAL_CANDIDATE_STATUSES
    )
    pending = max(0, planned - terminal)
    allocations = Counter(
        str(slot.get("kind"))
        for manifest in manifests
        for slot in cast(Sequence[Mapping[str, Any]], manifest.get("slots", ()))
        if isinstance(slot, Mapping)
    )
    report = _load_mapping(root / "m5-report.json.gz")
    stop = _load_mapping(root / "m5-stop.json.gz")
    state = str(retained_state.get("state", "ready"))
    resumable = bool(retained_state.get("resumable", True))
    run_terminal = retained_state.get("run_terminal") is True
    terminal_reason = (
        str(retained_state["terminal_reason"])
        if retained_state.get("terminal_reason") is not None
        else None
    )
    result_kind = str(
        retained_state.get("scientific_result_kind", "NONE")
    )
    exact = retained_state.get("scientific_success") is True
    if (
        report is not None
        and report.get("protocol_id") == M5_REPORT_PROTOCOL_ID
        and retained_state.get("state") == "completed"
    ):
        state = "completed"
        resumable = False
        run_terminal = True
        terminal_reason = str(report.get("stop_reason", "generation_budget"))
        exact = report.get("exact_verified") is True
        result_kind = (
            "VERIFIED_COUNTEREXAMPLE"
            if exact
            else "DEVELOPMENT_SEARCH_EVIDENCE"
        )
    elif stop is not None:
        state = "blocked"
        resumable = stop.get("resumable") is True
        result_kind = "NO_SCIENTIFIC_RESULT"
    generations = [
        _nonnegative_int(manifest.get("generation"))
        for manifest in manifests
    ]
    telemetry = _evaluation_telemetry(root)
    profiles = _program_projection(candidates)
    verifier_submissions = sum(
        _nonnegative_int(
            cast(Mapping[str, Any], item.get("behavior_profile", {})).get(
                "exact_verifier_submissions"
            )
        )
        for item in candidates
        if isinstance(item.get("behavior_profile"), Mapping)
    )
    verifier_records = sum(
        _nonnegative_int(
            cast(Mapping[str, Any], item.get("behavior_profile", {})).get(
                "exact_verifier_records"
            )
        )
        for item in candidates
        if isinstance(item.get("behavior_profile"), Mapping)
    )
    return {
        **_public(retained_state),
        "state": state,
        "resumable": resumable,
        "run_terminal": run_terminal,
        "terminal_reason": terminal_reason,
        "scientific_result_kind": result_kind,
        "scientific_success": exact,
        "generation_index": max(generations, default=None),
        "generation_manifest_hashes": [
            manifest.get("sha256") for manifest in manifests
        ],
        "counts": {
            "planned": planned,
            "terminal": terminal,
            "pending": pending,
            "valid": statuses["evaluated"] + statuses["duplicate"],
            "contract_invalid": statuses["contract_invalid"],
            "duplicate": statuses["duplicate"],
            "provider_failed": statuses["provider_failed"],
            "evaluation_infrastructure_failure": statuses[
                "evaluation_infrastructure_failure"
            ],
            "evaluated": statuses["evaluated"],
            "missing": statuses["missing"],
            "roots": allocations["root"],
            "children": allocations["child"],
        },
        "candidate_status_counts": dict(sorted(statuses.items())),
        "provider": _provider_telemetry(root, candidates),
        "sandbox": {
            key: telemetry[key]
            for key in (
                "starts",
                "rotations",
                "failures",
                "timeouts",
                "maximum_rss_kib",
            )
        },
        "policy_invocations": telemetry["policy_invocations"],
        "graph_scores": {
            "attempts": telemetry["graph_score_attempts"],
            "unique_graphs": telemetry["unique_graph_scores"],
        },
        "programs": profiles,
        "recovery": {
            "state": (
                "terminal"
                if run_terminal
                else "resumable"
                if resumable and candidates
                else "fresh"
            ),
            "resume_attempts": _nonnegative_int(
                retained_state.get("resume_attempts")
            ),
            "last_boundary": retained_state.get("last_boundary"),
            "completed_slots_will_not_repeat": True,
        },
        "exact_verification": {
            "authority": "exact_verifier_only",
            "submissions": verifier_submissions,
            "records": verifier_records,
            "verified": exact,
        },
        "equal_development_panel": (
            _load_mapping(root / "protocol.json.gz") or {}
        ).get("panel_hash"),
    }


def python_preview_status(
    config_path: str | Path,
) -> dict[str, Any]:
    """Return a bounded read-only status projection without provider contact."""

    config = load_python_preview_config(config_path)
    if not config.experiment_root.exists():
        return _progress(config, _base_state(config))
    try:
        state = _load_state(config)
    except (OSError, ValueError, PythonPreviewWorkspaceError) as error:
        return {
            **_public(_base_state(config)),
            "state": "failed",
            "run_terminal": True,
            "terminal_reason": "workspace_mismatch",
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "last_error": _safe_error(error, config),
        }
    return _progress(config, state)


def _prompt_inputs() -> tuple[str, str, str, dict[str, Any], dict[str, Any]]:
    system_prompt = (
        _PROJECT_ROOT / "prompts/native-v3-python/m5-system.md"
    ).read_text(encoding="utf-8").strip()
    request_template = (
        _PROJECT_ROOT / "prompts/native-v3-python/m4-request.md"
    ).read_text(encoding="utf-8")
    specification_prompt = (
        "Retain the complete policy specification below for later root and "
        "mutation turns. Do not generate a policy on this turn. Return only "
        "the required specification acknowledgement.\n\n"
        + request_template
    )
    policy_schema = json.loads(
        (
            _PROJECT_ROOT
            / "configs/native/native-v3-python-policy-response.schema.json"
        ).read_text(encoding="utf-8")
    )
    if not isinstance(policy_schema, dict):
        raise ValueError("Python policy output schema must be an object")
    return (
        system_prompt,
        request_template,
        specification_prompt,
        policy_schema,
        specification_ack_schema(),
    )


def _panel(backend: GraphBackend) -> tuple[DevelopmentCaseV1, ...]:
    forbidden_lengths = backend.target_forbidden_lengths(30)
    return (
        DevelopmentCaseV1(
            case_id="order-30-seed-101",
            order=30,
            graph_seed=101,
            policy_seed=17,
            horizon=1,
            witness_cap=64,
            forbidden_lengths=forbidden_lengths,
        ),
        DevelopmentCaseV1(
            case_id="order-30-seed-103",
            order=30,
            graph_seed=103,
            policy_seed=19,
            horizon=1,
            witness_cap=64,
            forbidden_lengths=forbidden_lengths,
        ),
    )


def _default_provider(
    config: PythonPreviewConfig,
    system_prompt: str,
) -> M5SearchProvider:
    return CodexM5SearchProvider(
        workspace=config.experiment_root / "provider-runtime",
        model=config.model,
        effort=config.effort,
        base_instructions=system_prompt,
        auth_json=Path.home() / ".codex" / "auth.json",
        turn_timeout_seconds=config.timeout_seconds,
    )


def _default_backend(config: PythonPreviewConfig) -> GraphBackend:
    return HegBackend(config.heg_repo)


def _default_evaluator(
    config: PythonPreviewConfig,
    backend: GraphBackend,
) -> M5ScientificEvaluator:
    return PythonPanelScientificEvaluator(
        backend=backend,
        artifact_root=config.experiment_root / "scientific-artifacts",
        runtime_limits=PolicyRuntimeLimitsV1(),
    )


def run_python_preview(
    config_path: str | Path,
    *,
    provider_factory: ProviderFactory = _default_provider,
    backend_factory: BackendFactory = _default_backend,
    evaluator_factory: EvaluatorFactory = _default_evaluator,
    provenance_guard: ProvenanceGuard = ensure_m5_acceptance_provenance,
    auth_available: Callable[[Path], bool] = Path.is_file,
) -> dict[str, Any]:
    """Run or resume the explicit two-generation ordinary-Python preview."""

    config = load_python_preview_config(config_path)
    existed = config.experiment_root.exists()
    state = _load_state(config) if existed else _initialize_workspace(config)
    if state.get("state") == "completed":
        return _progress(config, state)
    if state.get("state") == "failed" and state.get("resumable") is not True:
        raise PythonPreviewWorkspaceError(
            "Python preview workspace is terminal and cannot be resumed"
        )
    if not (config.heg_repo / "src" / "sglab").is_dir():
        blocked = {
            **state,
            "state": "blocked",
            "resumable": True,
            "last_error": "HEG repository is unavailable",
        }
        _write_state(config, blocked)
        return _progress(config, blocked)
    (
        system_prompt,
        request_template,
        specification_prompt,
        policy_schema,
        ack_schema,
    ) = _prompt_inputs()
    try:
        provenance_guard(
            workspace=config.experiment_root,
            resume=(
                config.experiment_root / M5_PROVENANCE_FILENAME
            ).is_file(),
            repository_root=_PROJECT_ROOT,
            heg_root=config.heg_repo,
            experiment_config=config.source_path,
            model=config.model,
            effort=config.effort,
            system_prompt=system_prompt,
            request_template=request_template,
            specification_prompt=specification_prompt,
            output_schema=policy_schema,
            specification_ack_schema=ack_schema,
            runtime_limits=PolicyRuntimeLimitsV1(),
        )
    except Exception as error:
        failed = {
            **state,
            "state": "failed",
            "resumable": False,
            "run_terminal": True,
            "terminal_reason": "provenance_mismatch",
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": _safe_error(error, config),
        }
        _write_state(config, failed)
        return _progress(config, failed)
    auth_json = Path.home() / ".codex" / "auth.json"
    if not auth_available(auth_json):
        blocked = {
            **state,
            "state": "blocked",
            "resumable": True,
            "last_error": "local Codex authentication is unavailable",
        }
        _write_state(config, blocked)
        return _progress(config, blocked)
    running = {
        **state,
        "state": "running",
        "resumable": True,
        "run_terminal": False,
        "last_error": None,
        "resume_attempts": _nonnegative_int(state.get("resume_attempts"))
        + int(existed),
    }
    _write_state(config, running)

    def boundary_hook(boundary: str) -> None:
        running["last_boundary"] = boundary
        _write_state(config, running)

    provider: M5SearchProvider | None = None
    backend: GraphBackend | None = None
    final_state: dict[str, Any]
    primary_error: Exception | None = None
    try:
        backend = backend_factory(config)
        panel = _panel(backend)
        evaluator = evaluator_factory(config, backend)
        provider = provider_factory(config, system_prompt)
        report = run_m5_search(
            provider=provider,
            evaluator=evaluator,
            workspace=config.experiment_root,
            panel=panel,
            system_prompt=system_prompt,
            specification_prompt=specification_prompt,
            specification_ack_schema=ack_schema,
            policy_schema=policy_schema,
            preview_active=True,
            close_provider=False,
            boundary_hook=boundary_hook,
        )
        final_state = {
            **running,
            "state": "completed",
            "resumable": False,
            "run_terminal": True,
            "terminal_reason": report.get("stop_reason"),
            "scientific_result_kind": (
                "VERIFIED_COUNTEREXAMPLE"
                if report.get("exact_verified") is True
                else "DEVELOPMENT_SEARCH_EVIDENCE"
            ),
            "scientific_success": report.get("exact_verified") is True,
        }
    except Exception as error:
        primary_error = error
        final_state = {
            **running,
            "state": "blocked",
            "resumable": True,
            "run_terminal": False,
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": _safe_error(error, config),
        }
    cleanup_errors: list[Exception] = []
    if provider is not None:
        try:
            if isinstance(provider, CodexM5SearchProvider):
                provider.close(
                    cleanup_capsule=(
                        primary_error is None
                        and final_state.get("state") == "completed"
                    )
                )
            else:
                provider.close()
        except Exception as error:
            cleanup_errors.append(error)
    if backend is not None:
        try:
            backend.close()
        except Exception as error:
            cleanup_errors.append(error)
    if cleanup_errors and primary_error is None:
        final_state = {
            **running,
            "state": "blocked",
            "resumable": True,
            "run_terminal": False,
            "scientific_result_kind": "NO_SCIENTIFIC_RESULT",
            "scientific_success": False,
            "last_error": f"{type(cleanup_errors[0]).__name__}_during_cleanup",
        }
    _write_state(config, final_state)
    return _progress(config, final_state)


__all__ = [
    "PYTHON_PREVIEW_CONFIG_SCHEMA_VERSION",
    "PYTHON_PREVIEW_MODE",
    "PYTHON_PREVIEW_PROTOCOL_VERSION",
    "PYTHON_PREVIEW_STATE_SCHEMA_VERSION",
    "PythonPreviewConfig",
    "PythonPreviewWorkspaceError",
    "load_python_preview_config",
    "python_preview_status",
    "run_python_preview",
]
