"""Offline tests for the Stage 4 provider boundary.

The fake peer is an in-memory JSONL process; no Codex executable or model call
is started by this module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from mutation_forge.stage3.app_server import IsolationError
from mutation_forge.stage4.app_server import (
    FROZEN_STAGE4_EFFORT,
    FROZEN_STAGE4_MODEL,
    STAGE4_TURN_TIMEOUT_SECONDS,
    Stage4AppServerProvider,
    Stage4ProviderError,
    _available_artifact_prefix,
    _codex_transport_schema,
    infrastructure_retry_eligible,
)

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "fake_stage3_app_server.py"
_SPEC = importlib.util.spec_from_file_location("stage4_fake_app_server", _FIXTURE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
FakeProcess = _MODULE.FakeProcess
FakeScenario = _MODULE.FakeScenario


def _provider(tmp_path: Path, scenario: Any | None = None) -> Stage4AppServerProvider:
    def factory(*args: Any, **kwargs: Any) -> Any:
        return FakeProcess(scenario, **kwargs)

    return Stage4AppServerProvider(
        process_factory=factory,
        auth_checker=lambda capsule: bool(capsule.codex_home),
        artifact_dir=tmp_path,
    )


def _request(**extra: Any) -> dict[str, Any]:
    return {
        "prompt": "change the policy",
        "system_prompt": "Return one structured Stage 4 policy.",
        "output_schema": {
            "type": "object",
            "required": ["schema_version", "source"],
            "properties": {
                "schema_version": {"const": "stage4.generated_policy.v1"},
                "source": {"type": "string"},
            },
        },
        "model": FROZEN_STAGE4_MODEL,
        "effort": FROZEN_STAGE4_EFFORT,
        **extra,
    }


def test_frozen_model_and_effort_are_enforced(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    with pytest.raises(Exception, match="requires gpt-5.6-luna:high"):
        provider.generate(_request(model="other"))
    with pytest.raises(Exception, match="requires gpt-5.6-luna:high"):
        provider.generate(_request(effort="low"))


def test_stage4_uses_the_proven_six_hundred_second_turn_timeout(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    assert provider.limits.turn_timeout == STAGE4_TURN_TIMEOUT_SECONDS == 600.0


def test_schema_prompt_ids_usage_and_artifact_refs_are_preserved(tmp_path: Path) -> None:
    result = _provider(tmp_path).generate(_request(artifact_prefix="slot-00"))
    assert result["response"] == "fixture answer"
    assert result["thread_id"] == "thread-1"
    assert result["turn_id"] == "turn-1"
    assert result["request_id"] >= 0
    assert result["usage"]["totalTokens"] == 5
    assert result["usage"]["reasoningOutputTokens"] == 1
    assert len(result["prompt_sha256"]) == 64
    assert len(result["schema_sha256"]) == 64
    request_artifact = json.loads(
        (tmp_path / "slot-00.request.json").read_text(encoding="utf-8")
    )
    assert request_artifact["output_schema"]["properties"]["schema_version"][
        "type"
    ] == "string"
    assert "type" not in request_artifact["frozen_output_schema"]["properties"][
        "schema_version"
    ]
    assert result["transport_output_schema_sha256"] != result["schema_sha256"]
    assert "slot-00.response.json" in result["artifact_refs"]
    assert not any("codex-home" in str(ref) for ref in result["artifact_refs"])


def test_valid_json_response_is_decoded_for_generation_and_raw_text_is_retained(
    tmp_path: Path,
) -> None:
    source = (Path(__file__).parents[2] / "fixtures" / "stage4-seeds" / "slot-00.py").read_text()
    envelope = {
        "schema_version": "stage4.generated_policy.v1",
        "source": source,
        "design_summary": "A deterministic policy.",
        "change_summary": "Retains the parent ranking signals.",
        "hypothesis": "The parent remains a useful baseline.",
        "used_fields": [],
        "assumptions": ["Inputs satisfy the policy contract."],
        "expected_failure_modes": ["Malformed inputs are rejected."],
    }
    text = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    result = _provider(tmp_path, FakeScenario(final_text=text)).generate(_request())
    assert result["response"] == envelope
    assert result["response_text"] == text
    assert result["raw_response"] == text


def test_unique_items_schema_is_rejected_before_transport(tmp_path: Path) -> None:
    calls = 0

    def factory(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return FakeProcess(FakeScenario(), **kwargs)

    provider = Stage4AppServerProvider(
        process_factory=factory,
        auth_checker=lambda capsule: True,
    )
    with pytest.raises(ValueError, match="uniqueItems"):
        provider.generate(_request(output_schema={"type": "array", "uniqueItems": False}))
    assert calls == 0


def test_codex_transport_schema_does_not_mutate_frozen_contract() -> None:
    frozen = {
        "type": "object",
        "properties": {
            "schema_version": {"const": "stage4.generated_policy.v1"},
        },
    }
    projected = _codex_transport_schema(frozen)
    assert projected["properties"]["schema_version"] == {
        "const": "stage4.generated_policy.v1",
        "type": "string",
    }
    assert frozen["properties"]["schema_version"] == {
        "const": "stage4.generated_policy.v1"
    }


def test_eight_concurrent_calls_use_independent_process_targets(tmp_path: Path) -> None:
    seen: list[str] = []

    def factory(*args: Any, **kwargs: Any) -> Any:
        seen.append(str(kwargs["env"]["CODEX_HOME"]))
        return FakeProcess(FakeScenario(), **kwargs)

    provider = Stage4AppServerProvider(
        process_factory=factory,
        auth_checker=lambda capsule: True,
        artifact_dir=tmp_path,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda i: provider.generate(_request(artifact_prefix=f"s{i}")), range(8)
            )
        )
    assert len(results) == 8
    assert len(set(seen)) == 8
    response_logs = tuple(tmp_path.glob("*.response.json"))
    assert len(response_logs) == 8
    assert len({path.name.split(".response.json")[0] for path in response_logs}) == 8


def test_generation_request_fields_derive_distinct_artifact_prefixes(
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    requests = [
        _request(
            campaign_id="campaign-1",
            generation=2,
            slot=f"slot-{index:02d}",
            phase="initial",
            idempotency_key=f"request-{index}",
        )
        for index in range(8)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(provider.generate, requests))
    assert len(results) == 8
    assert len({result["artifact_refs"][0] for result in results}) == 8
    assert all(len(result["artifact_refs"][0]) <= 128 for result in results)


def test_repair_requests_derive_distinct_artifact_prefixes(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    results = [
        provider.repair(
            _request(
                campaign_id="campaign-1",
                generation=2,
                slot=slot,
                phase="repair",
                idempotency_key=f"repair-{slot}",
            ),
            ({"code": "syntax_error", "message": "invalid syntax"},),
        )
        for slot in ("slot-00", "slot-01")
    ]
    assert len({result["artifact_refs"][0] for result in results}) == 2
    assert len(tuple(tmp_path.glob("*.response.json"))) == 2


def test_retry_artifact_prefix_never_overwrites_prior_attempt(tmp_path: Path) -> None:
    prefix = "slot-00"
    assert _available_artifact_prefix(tmp_path, prefix) == prefix
    (tmp_path / f"{prefix}.request.json").write_text("retained", encoding="utf-8")
    assert _available_artifact_prefix(tmp_path, prefix) == f"{prefix}.retry-01"
    (tmp_path / f"{prefix}.retry-01.response.json").write_text(
        "retained",
        encoding="utf-8",
    )
    assert _available_artifact_prefix(tmp_path, prefix) == f"{prefix}.retry-02"


def test_pre_turn_provider_failure_is_proven_uncharged(tmp_path: Path) -> None:
    provider = Stage4AppServerProvider(
        process_factory=lambda *args, **kwargs: FakeProcess(FakeScenario(), **kwargs),
        auth_checker=lambda capsule: False,
        artifact_dir=tmp_path,
    )
    with pytest.raises(Stage4ProviderError) as raised:
        provider.generate(_request(artifact_prefix="pre-turn"))
    evidence = raised.value.evidence
    assert evidence["accepted"] is False
    assert evidence["charged"] is False
    assert evidence["content"] is False
    assert evidence["uncharged"] is True
    assert evidence["usage"] == {}
    assert infrastructure_retry_eligible(evidence)


def test_post_thread_provider_failure_is_not_retried(tmp_path: Path) -> None:
    provider = _provider(tmp_path, FakeScenario(terminal_status="failed"))
    with pytest.raises(Stage4ProviderError) as raised:
        provider.generate(_request(artifact_prefix="post-thread"))
    evidence = raised.value.evidence
    assert evidence["accepted"] is True
    assert evidence["thread_id"] == "thread-1"
    assert evidence["uncharged"] is False
    assert not infrastructure_retry_eligible(evidence)


def test_accepted_retry_artifact_can_be_recovered_without_a_new_turn(
    tmp_path: Path,
) -> None:
    request = _request(artifact_prefix="retained")
    (tmp_path / "retained.response.json").write_text("{}", encoding="utf-8")
    provider = _provider(tmp_path)
    generated = provider.generate(request)
    assert "retained.retry-01.response.json" in generated["artifact_refs"]

    recovered = provider.load_retained_result(request)
    assert recovered is not None
    assert recovered["accepted"] is True
    assert recovered["thread_id"] == generated["thread_id"]
    assert recovered["turn_id"] == generated["turn_id"]
    assert recovered["request_sha256"] == generated["request_sha256"]


def test_accepted_error_artifact_is_recovered_as_terminal_infrastructure(
    tmp_path: Path,
) -> None:
    request = _request(artifact_prefix="retained-error")
    (tmp_path / "retained-error.response.json").write_text("{}", encoding="utf-8")
    provider = _provider(tmp_path, FakeScenario(terminal_status="failed"))
    with pytest.raises(Stage4ProviderError):
        provider.generate(request)

    recovered = provider.load_retained_result(request)
    assert recovered is not None
    assert recovered["status"] == "infrastructure"
    assert recovered["accepted"] is True
    assert recovered["content"] is False
    assert recovered["uncharged"] is False
    assert recovered["retained_artifact_recovery"].endswith(
        ".retry-01.response.json"
    )


def test_legacy_rejected_schema_artifact_is_recovered_without_resubmission(
    tmp_path: Path,
) -> None:
    request = _request(artifact_prefix="legacy-error")
    (tmp_path / "legacy-error.response.json").write_text("{}", encoding="utf-8")
    provider = _provider(tmp_path, FakeScenario(terminal_status="failed"))
    with pytest.raises(Stage4ProviderError):
        provider.generate(request)
    prefix = "legacy-error.retry-01"
    request_path = tmp_path / f"{prefix}.request.json"
    response_path = tmp_path / f"{prefix}.response.json"
    request_value = json.loads(request_path.read_text(encoding="utf-8"))
    response_value = json.loads(response_path.read_text(encoding="utf-8"))
    request_value["output_schema"] = request_value.pop("frozen_output_schema")
    request_value.pop("transport_output_schema_sha256")
    response_value.pop("transport_output_schema_sha256")
    request_path.write_text(json.dumps(request_value), encoding="utf-8")
    response_path.write_text(json.dumps(response_value), encoding="utf-8")

    recovered = provider.load_retained_result(request)
    assert recovered is not None
    assert recovered["status"] == "infrastructure"
    assert recovered["accepted"] is True


def test_completed_remote_turn_is_recovered_after_host_logging_timeout(
    tmp_path: Path,
) -> None:
    request = _request(artifact_prefix="completed-timeout")
    (tmp_path / "completed-timeout.response.json").write_text(
        "{}",
        encoding="utf-8",
    )
    provider = _provider(tmp_path)
    generated = provider.generate(request)
    prefix = "completed-timeout.retry-01"
    response_path = tmp_path / f"{prefix}.response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response.update(
        {
            "status": "error",
            "error_type": "TurnError",
            "error": "turn timed out",
            "accepted": True,
            "charged": False,
            "content": False,
            "usage": {},
            "request_id": None,
            "turn_id": None,
            "provider_request_id": None,
            "provider_turn_id": None,
        }
    )
    response_path.write_text(json.dumps(response), encoding="utf-8")

    recovered = provider.load_retained_result(request)
    assert recovered is not None
    assert recovered["status"] == "completed"
    assert recovered["accepted"] is True
    assert recovered["content"] is True
    assert recovered["response"] == generated["response"]
    assert recovered["turn_id"] == generated["turn_id"]
    assert recovered["usage"] == generated["usage"]
    assert recovered["retained_completed_turn_recovery"] == prefix
    assert recovered["host_timeout_after_remote_completion"] is True


def test_logs_are_redacted_and_bounded(tmp_path: Path) -> None:
    result = _provider(tmp_path).generate(_request(artifact_prefix="slot"))
    assert len(result["diagnostics"]) <= 200
    raw = (tmp_path / "slot.response.json").read_text()
    assert "auth_token" not in raw
    assert len(raw.encode()) < 1_048_576


def test_unauthorized_tool_request_is_not_approved(tmp_path: Path) -> None:
    provider = _provider(tmp_path, FakeScenario(server_request=True))
    with pytest.raises(Stage4ProviderError, match="unsupported server request") as raised:
        provider.generate(_request())
    assert raised.value.evidence["unauthorized_tool_approval"] is True


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({"accepted": False, "content": False, "uncharged": True, "usage": {}}, True),
        (
            {"accepted": False, "content": False, "uncharged": True, "usage": {"totalTokens": 1}},
            False,
        ),
        ({"accepted": False, "content": False, "usage": {}}, False),
        ({"accepted": True, "content": False, "uncharged": True, "usage": {}}, False),
        (
            {
                "accepted": False,
                "content": False,
                "uncharged": True,
                "usage": {},
                "unauthorized_tool_approval": True,
            },
            False,
        ),
    ],
)
def test_infrastructure_retry_requires_host_zero_evidence(
    evidence: dict[str, Any], expected: bool
) -> None:
    assert infrastructure_retry_eligible(evidence) is expected


def test_authenticated_capsule_is_seen_without_reading_credentials(tmp_path: Path) -> None:
    seen: list[tuple[str, str]] = []

    def checker(capsule: Any) -> bool:
        seen.append((str(capsule.codex_home), str(capsule.sqlite_home)))
        return True

    def factory(*args: Any, **kwargs: Any) -> Any:
        return FakeProcess(FakeScenario(), **kwargs)

    provider = Stage4AppServerProvider(
        process_factory=factory,
        auth_checker=checker,
        auth_json=tmp_path / "authorized-auth.json",
    )
    # The fake checker is intentionally the only observer of capsule identity;
    # credentials are never opened or serialized by this test.
    with pytest.raises(IsolationError):
        provider.generate(_request())
    assert seen == []  # unavailable auth source fails before capsule use
