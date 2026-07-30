from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from mutation_forge.sandbox.contracts import ProbeContext, ProbeProposal, SandboxLimits
from mutation_forge.sandbox.errors import (
    ProtocolError,
    WorkerCrashError,
    WorkerTimeoutError,
)
from mutation_forge.sandbox.worker import PolicyWorker, _encode_frame


def _ctx() -> ProbeContext:
    return {
        "probe_id": "worker",
        "step": 0,
        "budget_remaining": 10000,
        "features": {},
    }


def _proposal(identifier: str = "p") -> ProbeProposal:
    return {
        "proposal_id": identifier,
        "kind": "probe",
        "features": {"weight": 2.0, "penalty": 1, "values": [1, 2]},
    }


def _source(project_root: Path, fixture: str) -> str:
    root = "rankers" if fixture in {
        "constant.py",
        "weighted.py",
        "bounded_loop.py",
        "conditional.py",
    } else "adversarial"
    return (project_root / "fixtures" / root / fixture).read_text()


def test_persistent_worker_completes_10000_bounded_calls(project_root: Path) -> None:
    with PolicyWorker(_source(project_root, "constant.py")) as worker:
        for index in range(10_000):
            result = worker.call(_ctx(), _proposal(f"p-{index}"))
            assert result.status == "ok"
            assert result.priority == 1
        telemetry = worker.telemetry()
        assert telemetry["calls"] == 10_000
        assert telemetry["failures"] == 0
        assert telemetry["captured_stderr_bytes"] <= 64 * 1024
        controls = telemetry["controls"]
        assert isinstance(controls, dict)
        assert controls["stdin_mode"] == "protocol_pipe"
        assert controls["process_group_isolated"]
        assert controls["rlimits"] == {
            "cpu": [60, 61],
            "address_space": [128 * 1024 * 1024, 128 * 1024 * 1024],
            "file_size": [64 * 1024, 64 * 1024],
            "open_files": [16, 16],
            "process_count": [1, 1],
        }
        assert set(controls["environment_keys"]) <= {
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
        }
        assert worker.usable


@pytest.mark.parametrize(
    "fixture",
    ["large_output.py", "non_finite.py", "exception.py"],
)
def test_runtime_output_and_exception_abuse_fails_only_candidate(
    project_root: Path,
    fixture: str,
) -> None:
    with PolicyWorker(_source(project_root, fixture)) as worker:
        result = worker.call(_ctx(), _proposal())
        assert result.status == "exception"
        assert result.priority is None
        assert not worker.usable
    with PolicyWorker(_source(project_root, "constant.py")) as healthy:
        assert healthy.call(_ctx(), _proposal()).priority == 1


@pytest.mark.parametrize(
    "source",
    [
        "def priority(ctx, proposal):\n    return True\n",
        (
            "def priority(ctx, proposal):\n"
            "    return 2 ** proposal['features']['exponent']\n"
        ),
    ],
)
def test_bool_and_oversized_numeric_outputs_are_rejected(source: str) -> None:
    proposal = _proposal()
    proposal["features"]["exponent"] = 5000
    with PolicyWorker(source) as worker:
        result = worker.call(_ctx(), proposal)
        assert result.status == "exception"
        assert result.priority is None
        assert not worker.usable


@pytest.mark.parametrize("fixture", ["large_allocation.py", "large_integer.py"])
def test_memory_or_cpu_abuse_is_bounded(project_root: Path, fixture: str) -> None:
    started = time.monotonic()
    worker = PolicyWorker(
        _source(project_root, fixture),
        SandboxLimits(per_call_wall_seconds=0.05),
    )
    try:
        try:
            result = worker.call(_ctx(), _proposal())
            assert result.status == "exception"
        except (WorkerTimeoutError, WorkerCrashError):
            pass
        assert not worker.usable
    finally:
        worker.close()
    assert time.monotonic() - started < 3.0


def test_worker_crash_is_reaped_and_not_reused(project_root: Path) -> None:
    worker = PolicyWorker(_source(project_root, "constant.py"))
    os.kill(worker._process.pid, signal.SIGKILL)
    with pytest.raises(WorkerCrashError):
        worker.call(_ctx(), _proposal())
    assert not worker.usable
    worker.close()


def test_protocol_failure_marks_worker_unusable(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = PolicyWorker(_source(project_root, "constant.py"))

    def corrupt(_wall_seconds: float) -> dict[str, object]:
        raise ProtocolError("corrupt frame")

    monkeypatch.setattr(worker, "_receive", corrupt)
    with pytest.raises(ProtocolError, match="corrupt"):
        worker.call(_ctx(), _proposal())
    assert not worker.usable
    worker.close()


def test_protocol_frames_are_size_bounded() -> None:
    with pytest.raises(ProtocolError, match="exceeds"):
        _encode_frame({"message": "x" * 100}, 16)


def test_shutdown_reaps_process_group_and_removes_isolated_cwd(
    project_root: Path,
) -> None:
    worker = PolicyWorker(_source(project_root, "constant.py"))
    process = worker._process
    isolated_cwd = Path(worker._temporary.name)
    assert isolated_cwd.is_dir()
    assert process.poll() is None
    worker.close()
    assert process.poll() is not None
    assert not isolated_cwd.exists()


def test_total_wall_limit_is_parent_controlled(project_root: Path) -> None:
    worker = PolicyWorker(_source(project_root, "constant.py"))
    worker._started -= worker.limits.total_wall_seconds + 1
    with pytest.raises(WorkerTimeoutError, match="total wall"):
        worker.call(_ctx(), _proposal())
    worker.close()


def test_infinite_loop_is_accepted_statically_and_stopped_by_worker_timeout() -> None:
    source = (
        "def priority(ctx, proposal):\n"
        "    total = 0\n"
        "    while True:\n"
        "        total += 1\n"
        "    return total\n"
    )
    limits = SandboxLimits(per_call_wall_seconds=0.05)
    worker = PolicyWorker(source, limits)
    started = time.monotonic()
    try:
        with pytest.raises(WorkerTimeoutError):
            worker.call(_ctx(), _proposal())
        assert not worker.usable
    finally:
        worker.close()
    assert time.monotonic() - started < 3.0
