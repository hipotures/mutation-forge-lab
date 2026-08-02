from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mutation_forge import cli
from mutation_forge.stage4 import commands
from mutation_forge.stage4.archive import ProgramArchive, ProgramRecord
from mutation_forge.stage4.generation import SlotResult


def test_stage4_parser_exposes_every_frozen_command(tmp_path: Path) -> None:
    config = Path("configs/stage4-search.toml")
    run = tmp_path / "run"
    cases = (
        [
            "stage4",
            "doctor",
            "--config",
            str(config),
            "--auth-json",
            str(tmp_path / "auth.json"),
            "--json",
        ],
        [
            "stage4",
            "freeze",
            "--config",
            str(config),
            "--auth-json",
            str(tmp_path / "auth.json"),
            "--json",
        ],
        [
            "stage4",
            "evolve",
            "--config",
            str(config),
            "--concurrency",
            "8",
            "--json",
        ],
        ["stage4", "resume", str(run), "--json"],
        ["stage4", "archive", "inspect", str(run), "--json"],
        ["stage4", "archive", "reindex", str(run), "--json"],
        [
            "stage4",
            "evaluate-candidate",
            str(run),
            "program-1",
            "--pass",
            "primary",
            "--workers",
            "8",
            "--json",
        ],
        ["stage4", "freeze-validation", str(run), "--json"],
        ["stage4", "validate", str(run), "--workers", "8", "--json"],
        ["stage4", "verify-replay", str(run), "--json"],
    )
    parser = cli._build_legacy_parser()
    for arguments in cases:
        parsed = parser.parse_args(arguments)
        assert parsed.command == "stage4"


def test_stage4_json_cli_is_jsonl_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = {
        "schema_version": "stage4.doctor.v1",
        "status": "completed",
        "decision": "READY",
    }
    observed: dict[str, object] = {}

    def fake_doctor(config: Path, *, auth_json: Path | None = None) -> dict[str, object]:
        observed.update({"config": config, "auth_json": auth_json})
        return result

    monkeypatch.setattr(commands, "doctor", fake_doctor)
    assert (
        cli.legacy_main(
            [
                "stage4",
                "doctor",
                "--config",
                "configs/stage4-search.toml",
                "--auth-json",
                "/tmp/authorized-auth.json",
                "--json",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == result
    assert observed["auth_json"] == Path("/tmp/authorized-auth.json")


def test_stage4_json_failure_is_one_json_object(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_: Path, *, auth_json: Path | None = None) -> dict[str, object]:
        assert auth_json == Path("/tmp/authorized-auth.json")
        raise RuntimeError("freeze unavailable")

    monkeypatch.setattr(commands, "freeze", fail)
    assert (
        cli.legacy_main(
            [
                "stage4",
                "freeze",
                "--config",
                "configs/stage4-search.toml",
                "--auth-json",
                "/tmp/authorized-auth.json",
                "--json",
            ]
        )
        == 1
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    value = json.loads(lines[0])
    assert value["event_type"] == "run_failed"
    assert value["error_type"] == "RuntimeError"


def test_archive_commands_share_authoritative_files(tmp_path: Path) -> None:
    run = tmp_path / "campaign"
    archive = ProgramArchive(run / "archive")
    source = run / "archive" / "sources" / "stage3-slot-00.py"
    source.parent.mkdir(parents=True)
    source.write_text("seed\n", encoding="utf-8")
    archive.append(
        ProgramRecord(
            program_id="stage3-slot-00",
            source_path="archive/sources/stage3-slot-00.py",
            source_sha256=hashlib.sha256(b"seed\n").hexdigest(),
            normalized_ast_sha256="b" * 64,
            behavior_signature="c" * 64,
            generation=0,
            slot="slot-00",
            validation_status="valid",
            probe_status="passed",
            smoke_10k_status="passed",
            replay_status="verified",
            fitness_status="verified",
            seed_id="stage3-slot-00",
        )
    )
    inspection = commands.archive_inspect(run)
    reindex = commands.archive_reindex(run)
    assert inspection["counts"]["records"] == 1
    assert inspection["archive_hash"] == reindex["archive_hash"]
    assert reindex["status"] == "completed"


def test_doctor_runs_full_shape_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Path("configs/stage4-search.toml").resolve()
    monkeypatch.setattr(commands, "campaign_root", lambda _: tmp_path / "campaign")
    original_git_state = commands._git_state

    def clean_git_state(repo: Path) -> dict[str, object]:
        return {**original_git_state(repo), "dirty": False}

    monkeypatch.setattr(commands, "_git_state", clean_git_state)
    result = commands.doctor(config, check_auth=False, write=False)
    assert result["status"] == "completed"
    assert result["inference"] is False
    assert result["live_model_results_observed"] is False
    assert result["checks"]["manifest_matrix"] is True
    assert result["projection"]["counts"]["search_policies"] == 40
    assert result["projection"]["counts"]["search_records"] == 10_240
    assert result["projection"]["counts"]["validation_records"] == 512


def test_appserver_profile_audits_auth_inside_supplied_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "authorized-auth.json"
    capsule_environment = {
        "PATH": "/usr/bin",
        "HOME": str(tmp_path / "capsule"),
        "CODEX_HOME": str(tmp_path / "capsule" / "codex-home"),
    }
    observed: dict[str, object] = {}

    class FakeCapsule:
        env = capsule_environment

        @staticmethod
        def create(
            root: Path,
            *,
            auth_json: Path,
            sandbox_mode: str,
            approval_policy: str,
        ) -> FakeCapsule:
            observed.update(
                {
                    "root": root,
                    "auth_json": auth_json,
                    "sandbox_mode": sandbox_mode,
                    "approval_policy": approval_policy,
                }
            )
            return FakeCapsule()

        def cleanup(self) -> None:
            observed["cleaned"] = True

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            observed["capsule"] = kwargs["capsule"]

        def model_catalog(self) -> tuple[dict[str, object], ...]:
            return (
                {
                    "model": "gpt-5.6-luna",
                    "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
                },
            )

        def close(self) -> None:
            observed["closed"] = True

    def fake_auth_status(*, environment: object = None) -> dict[str, object]:
        observed["environment"] = environment
        return {
            "ok": True,
            "authenticated": True,
            "source": "private capsule codex login status",
        }

    monkeypatch.setattr(commands, "IsolatedCapsule", FakeCapsule)
    monkeypatch.setattr(commands, "Stage4AppServerAdapter", FakeAdapter)
    monkeypatch.setattr(commands, "_auth_status", fake_auth_status)
    monkeypatch.setattr(commands, "secure_capsule_parent", lambda: tmp_path)

    result = commands._appserver_profile_status(tmp_path / "run", auth_json=auth_path)
    assert result["ok"] is True
    assert result["auth"]["authenticated"] is True
    assert observed["auth_json"] == auth_path
    assert observed["environment"] is capsule_environment
    assert observed["sandbox_mode"] == "danger-full-access"
    assert observed["approval_policy"] == "never"
    assert observed["cleaned"] is True
    assert observed["closed"] is True


def test_validation_and_generation_worker_counts_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        commands.evolve(
            "configs/stage4-search.toml",
            concurrency=7,
            provider=object(),
        )
    with pytest.raises(ValueError, match="workers"):
        commands.validate(tmp_path, workers=7)


def test_exact_usage_rejects_partial_or_incomplete_envelopes() -> None:
    complete = {
        "inputTokens": 1,
        "cachedInputTokens": 0,
        "outputTokens": 1,
        "reasoningOutputTokens": 0,
        "totalTokens": 2,
        "final": True,
        "partial": False,
    }
    assert commands._usage_complete(complete)
    assert not commands._usage_complete({**complete, "partial": True})
    assert not commands._usage_complete({"totalTokens": 2, "final": True})


def test_generation_request_identity_remains_pinned_after_protocol_amendment(
    tmp_path: Path,
) -> None:
    freeze_doctor = "a" * 64
    authenticated_doctor = "b" * 64
    protocol_doctor = "c" * 64
    assert (
        commands._generation_request_doctor_sha256(
            tmp_path,
            {"doctor_sha256": freeze_doctor},
        )
        == freeze_doctor
    )
    (tmp_path / commands.POST_LIVE_AMENDMENT_PATH).write_text(
        json.dumps({"authenticated_doctor_sha256": authenticated_doctor}),
        encoding="utf-8",
    )
    (tmp_path / commands.PROTOCOL_AMENDMENT_PATH).write_text(
        json.dumps({"authenticated_doctor_sha256": protocol_doctor}),
        encoding="utf-8",
    )
    assert (
        commands._generation_request_doctor_sha256(
            tmp_path,
            {"doctor_sha256": freeze_doctor},
        )
        == authenticated_doctor
    )


def test_seed_evaluation_failure_is_not_live_model_evidence(tmp_path: Path) -> None:
    failed_seed = tmp_path / "evaluations" / "search-seeds" / "failure.json"
    failed_seed.parent.mkdir(parents=True)
    failed_seed.write_text('{"error":"worker_timeout"}', encoding="utf-8")
    evidence = commands._live_model_result_evidence(tmp_path)
    assert evidence["observed"] is False
    (tmp_path / "generation-checkpoint.json").write_text("{}", encoding="utf-8")
    assert commands._live_model_result_evidence(tmp_path)["observed"] is True


def test_amendment_freeze_requires_retained_scientific_identity(tmp_path: Path) -> None:
    original = {
        "config_sha256": "a" * 64,
        "frozen_hashes": {"manifest": "b" * 64},
    }
    original["freeze_sha256"] = commands._freeze_digest(original)
    amendment_v1 = {
        "name": "stage4-search-amendment-v1",
        "type": "tag",
        "commit": "1" * 40,
    }
    previous = {
        **original,
        "amendment_tag": amendment_v1,
        "amendment_category": "evaluation_worker_process_isolation",
        "previous_freeze_sha256": original["freeze_sha256"],
        "scientific_identity_unchanged": True,
        "pre_amendment_live_model_evidence": {"observed": False},
    }
    previous["freeze_sha256"] = commands._freeze_digest(previous)
    amendment_v2 = {
        "name": "stage4-search-amendment-v2",
        "type": "tag",
        "commit": "2" * 40,
    }
    active = {
        **original,
        "amendment_tag": amendment_v2,
        "amendment_tags": [amendment_v1, amendment_v2],
        "amendment_category": "replay_metrics_timing_projection",
        "previous_freeze_sha256": previous["freeze_sha256"],
        "previous_freeze_path": "search-freeze-pre-amendment-v2.json",
        "scientific_identity_unchanged": True,
        "pre_amendment_live_model_evidence": {"observed": False},
    }
    active["freeze_sha256"] = commands._freeze_digest(active)
    assert not commands._amendment_freeze_valid(tmp_path, active)
    (tmp_path / "search-freeze-pre-amendment.json").write_text(
        json.dumps(original),
        encoding="utf-8",
    )
    (tmp_path / "search-freeze-pre-amendment-v2.json").write_text(
        json.dumps(previous),
        encoding="utf-8",
    )
    assert commands._amendment_freeze_valid(tmp_path, active)
    active["frozen_hashes"] = {"manifest": "d" * 64}
    assert not commands._amendment_freeze_valid(tmp_path, active)


def test_authentication_recovery_is_additive_and_idempotent(tmp_path: Path) -> None:
    run = tmp_path / "campaign"
    archive = ProgramArchive(run / "archive")
    seed_ids: list[str] = []
    for index in range(8):
        program_id = f"seed-{index:02d}"
        seed_ids.append(program_id)
        source = f"def priority(ctx, proposal):\n    return {index}.0\n"
        source_path = run / "archive" / "sources" / f"{program_id}.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        archive.append(
            ProgramRecord(
                program_id=program_id,
                source_path=f"archive/sources/{program_id}.py",
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                normalized_ast_sha256=hashlib.sha256(f"ast-{index}".encode()).hexdigest(),
                behavior_signature=hashlib.sha256(f"behavior-{index}".encode()).hexdigest(),
                generation=0,
                slot=f"slot-{index:02d}",
                validation_status="valid",
                probe_status="passed",
                smoke_10k_status="passed",
                replay_status="verified",
                fitness_status="verified",
                seed_id=program_id,
            )
        )

    slots: dict[str, object] = {}
    for generation in range(4):
        for index in range(8):
            key = hashlib.sha256(f"{generation}:{index}".encode()).hexdigest()
            envelope = {
                "response": None,
                "status": "infrastructure",
                "accepted": False,
                "charged": False,
                "content": False,
                "uncharged": False,
                "usage": {},
                "request_id": None,
                "thread_id": None,
                "turn_id": None,
                "session_id": None,
                "provider_request_id": None,
                "provider_thread_id": None,
                "provider_turn_id": None,
                "error": "IsolationError: isolated Codex home is not authenticated",
            }
            slots[key] = {
                "generation": generation,
                "slot": f"slot-{index:02d}",
                "parent_id": seed_ids[index],
                "status": "failed",
                "candidate": None,
                "errors": [],
                "repairs": 0,
                "initial": envelope,
                "repair": None,
                "request": {"idempotency_key": key},
                "raw_result": envelope,
                "duplicate_of": None,
            }
            digest = hashlib.sha256(f"tombstone:{key}".encode()).hexdigest()
            archive.append(
                ProgramRecord(
                    program_id=f"tombstone-{generation}-{index}",
                    source_path="",
                    source_sha256=digest,
                    normalized_ast_sha256=digest,
                    behavior_signature=digest,
                    generation=generation + 1,
                    slot=f"slot-{index:02d}",
                    parent_id=seed_ids[index],
                    request_id=key,
                    tombstone=True,
                    validation_status="failed",
                    probe_status="failed",
                    smoke_10k_status="failed",
                    replay_status="not_evaluated",
                    fitness_status="failed",
                    metadata={
                        "repairs": 0,
                        "status": "failed",
                        "usage_complete": False,
                        "unauthorized_tool_approval": False,
                        "accepted_turn_count": 0,
                    },
                )
            )

    checkpoint = {
        "schema_version": "stage4.checkpoint.v1",
        "campaign_id": "stage4-test",
        "slots": slots,
        "callbacks": {str(index): {"status": "completed"} for index in range(4)},
        "summary": {"initial_turn_count": 32},
    }
    summary = {
        "schema_version": "stage4.campaign.v1",
        "decision": "NO_GO",
        "decision_reason": "minimum_unique_offspring_not_met",
        "initial_turns": 32,
        "repair_turns": 0,
        "accepted_live_turns": 0,
        "new_unique_valid_offspring": 0,
        "exact_usage": False,
        "unauthorized_tool_approval": False,
        "live_stage4_model_results_observed": True,
    }
    (run / "generation-checkpoint.json").write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )
    (run / "search-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "EVIDENCE_MANIFEST.json").write_text('{"retained":true}', encoding="utf-8")
    for relative in (
        "appserver/attempt.response.json",
        "raw/generation-0001/slot-0000.json.gz",
        "selection/generation-01.json",
    ):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"retained:{relative}".encode())

    recovery = commands._prepare_authentication_recovery(run)
    assert recovery["phase"] == "completed"
    assert recovery["slot_count"] == 32
    assert recovery["replacement_requests_authorized"] is True
    assert len(recovery["moved_tombstones"]) == 32
    assert len(ProgramArchive(run / "archive").records()) == 8
    active_checkpoint = json.loads((run / "generation-checkpoint.json").read_text(encoding="utf-8"))
    assert active_checkpoint["slots"] == {}
    assert active_checkpoint["callbacks"] == {}
    assert active_checkpoint["authentication_recovery"]["replacement_requests_authorized"] is True
    retained_root = run / commands.AUTH_RECOVERY_ROOT
    assert (retained_root / "generation-checkpoint.json").read_text(encoding="utf-8") == json.dumps(
        checkpoint
    )
    assert (retained_root / "search-summary.json").read_text(encoding="utf-8") == json.dumps(
        summary
    )
    assert len(tuple((retained_root / "archive" / "programs").glob("*.json"))) == 32
    assert commands._prepare_authentication_recovery(run) == recovery


def test_archive_uses_global_idempotency_key_not_process_local_rpc_id() -> None:
    slot = SlotResult(
        generation=1,
        slot="slot-01",
        parent_id="parent",
        status="failed",
        repairs=1,
        initial={
            "request_id": 10,
            "thread_id": "initial-thread",
            "turn_id": "initial-turn",
        },
        repair={
            "request_id": 10,
            "provider_request_id": "10",
            "thread_id": "repair-thread",
            "turn_id": "repair-turn",
        },
        request={
            "idempotency_key": "globally-unique-repair-key",
            "request_idempotency_key": "globally-unique-repair-key",
        },
    )

    assert commands._slot_archive_transport_identity(slot) == (
        "globally-unique-repair-key",
        "10",
        "repair-thread",
        "repair-turn",
    )


def test_completed_turn_partial_callback_recovery_is_additive_and_idempotent(
    tmp_path: Path,
) -> None:
    run = tmp_path / "campaign"
    archive = ProgramArchive(run / "archive")
    seed_ids: list[str] = []
    for index in range(8):
        seed_id = f"seed-{index:02d}"
        seed_ids.append(seed_id)
        source = f"def priority(ctx, proposal):\n    return {index}.0\n"
        source_path = run / "archive" / "sources" / f"{seed_id}.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        archive.append(
            ProgramRecord(
                program_id=seed_id,
                source_path=f"archive/sources/{seed_id}.py",
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                normalized_ast_sha256=hashlib.sha256(f"seed-ast-{index}".encode()).hexdigest(),
                behavior_signature=hashlib.sha256(f"seed-behavior-{index}".encode()).hexdigest(),
                generation=0,
                slot=f"slot-{index:02d}",
                validation_status="valid",
                probe_status="passed",
                smoke_10k_status="passed",
                replay_status="verified",
                fitness_status="verified",
                seed_id=seed_id,
            )
        )
        digest = hashlib.sha256(f"protocol-{index}".encode()).hexdigest()
        archive.append(
            ProgramRecord(
                program_id=f"protocol-{index:02d}",
                source_path="",
                source_sha256=digest,
                normalized_ast_sha256=digest,
                behavior_signature=digest,
                generation=1,
                slot=f"slot-{index:02d}",
                parent_id=seed_id,
                request_id=f"initial-{index:02d}",
                validation_status="failed",
                probe_status="failed",
                smoke_10k_status="failed",
                replay_status="not_evaluated",
                fitness_status="failed",
                tombstone=True,
            )
        )
        partial_source = f"def priority(ctx, proposal):\n    return {index + 8}.0\n"
        partial_source_path = run / "archive" / "sources" / f"partial-{index:02d}.py"
        partial_source_path.write_text(partial_source, encoding="utf-8")
        archive.append(
            ProgramRecord(
                program_id=f"partial-{index:02d}",
                source_path=f"archive/sources/partial-{index:02d}.py",
                source_sha256=hashlib.sha256(partial_source.encode()).hexdigest(),
                normalized_ast_sha256=hashlib.sha256(f"partial-ast-{index}".encode()).hexdigest(),
                behavior_signature=hashlib.sha256(f"partial-behavior-{index}".encode()).hexdigest(),
                generation=2,
                slot=f"slot-{index:02d}",
                parent_id=seed_id,
                request_id="10",
                validation_status="valid",
                probe_status="passed",
                smoke_10k_status="passed",
                replay_status="verified",
                fitness_status="verified",
                seed_id=seed_id,
            )
        )

    source_report = archive.reindex()
    assert source_report.duplicate_requests == ("10",)
    checkpoint = {
        "schema_version": "stage4.checkpoint.v1",
        "campaign_id": "stage4-test",
        "slots": {},
        "callbacks": {"0": {"status": "completed"}},
    }
    (run / "generation-checkpoint.json").write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )
    parent_recovery = {"source_checkpoint_sha256": "a" * 64}

    recovery = commands._prepare_completed_turn_callback_recovery(
        run,
        parent_recovery,
    )
    assert recovery is not None
    assert recovery["phase"] == "completed"
    assert recovery["partial_generation"] == 2
    assert recovery["partial_record_count"] == 8
    assert len(recovery["retained_files"]) == 16
    reconciled = archive.reindex()
    assert reconciled.ok
    assert len(reconciled.records) == 16
    assert not tuple((run / "archive" / "programs").glob("partial-*.json"))

    for index in range(8):
        partial_source = f"def priority(ctx, proposal):\n    return {index + 8}.0\n"
        partial_source_path = run / "archive" / "sources" / f"partial-{index:02d}.py"
        partial_source_path.write_text(partial_source, encoding="utf-8")
        archive.append(
            ProgramRecord(
                program_id=f"partial-{index:02d}",
                source_path=f"archive/sources/partial-{index:02d}.py",
                source_sha256=hashlib.sha256(partial_source.encode()).hexdigest(),
                normalized_ast_sha256=hashlib.sha256(f"partial-ast-{index}".encode()).hexdigest(),
                behavior_signature=hashlib.sha256(f"partial-behavior-{index}".encode()).hexdigest(),
                generation=2,
                slot=f"slot-{index:02d}",
                parent_id=seed_ids[index],
                request_id=f"globally-unique-{index:02d}",
                validation_status="valid",
                probe_status="passed",
                smoke_10k_status="passed",
                replay_status="verified",
                fitness_status="verified",
                seed_id=seed_ids[index],
            )
        )

    assert (
        commands._prepare_completed_turn_callback_recovery(
            run,
            parent_recovery,
        )
        == recovery
    )
    rebuilt = archive.reindex()
    assert rebuilt.ok
    assert len(rebuilt.records) == 24
