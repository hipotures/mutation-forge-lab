from __future__ import annotations

import ast
import math
import os
import signal
import time
from pathlib import Path

import pytest

from mutation_forge.models import GraphState, RewritePlan, normalized_edge
from mutation_forge.native_v3_python import (
    IllegalRewriteError,
    IsolatedPolicyWorkerV1,
    NoPlan,
    PolicyContextV1,
    PolicyInfrastructureError,
    PolicyProtocolError,
    PolicyRuntimeLimitsV1,
    UnsupportedPolicySandboxError,
)
from mutation_forge.native_v3_python.runner import (
    _canonical_frame,
    _strict_json_object,
    _WorkerCrash,
    _WorkerTimeout,
)
from mutation_forge.native_v3_python.safe_api import SafeGraphSessionV1, graph_view_v1

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "mutation_forge" / "native_v3_python"

NO_PLAN_SOURCE = """\
def propose(ctx, graph, api, seed):
    return api.no_plan()
"""

ADD_EDGE_SOURCE = """\
def propose(ctx, graph, api, seed) -> RewritePlan | NoPlan:
    candidates = api.non_edges_legal()
    if not candidates:
        return api.no_plan(reason="NO_MATCH")
    edge = api.pick(candidates, seed, "m2-add")
    if edge == None:
        return api.no_plan(reason="NO_MATCH")
    api.add_edge(edge)
    return api.emit()
"""


def _cubic_graph(order: int = 6) -> GraphState:
    edges = {normalized_edge((vertex, (vertex + 1) % order)) for vertex in range(order)}
    edges.update((vertex, vertex + order // 2) for vertex in range(order // 2))
    return GraphState(order, tuple(sorted(edges)))


def _degrees(graph: GraphState) -> tuple[int, ...]:
    values = [0] * graph.order
    for u, v in graph.edges:
        values[u] += 1
        values[v] += 1
    return tuple(values)


def _connected(graph: GraphState) -> bool:
    adjacency = [set[int]() for _ in range(graph.order)]
    for u, v in graph.edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    seen = {0}
    pending = [0]
    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return len(seen) == graph.order


class _Host:
    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState:
        current = set(graph.edges)
        removed = set(rewrite.removed_edges)
        added = set(rewrite.added_edges)
        if not removed.issubset(current) or added & (current - removed):
            raise IllegalRewriteError("invalid delta")
        candidate = GraphState(graph.order, tuple(sorted((current - removed) | added)))
        if min(_degrees(candidate)) < 3 or not _connected(candidate):
            raise IllegalRewriteError("illegal final graph")
        return candidate


HOST = _Host()


class _BrokenHost:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def apply_rewrite(self, graph: GraphState, rewrite: RewritePlan) -> GraphState:
        del graph, rewrite
        raise self.error


def _context(invocation_ordinal: int = 0) -> PolicyContextV1:
    return PolicyContextV1(
        step_index=0,
        horizon=8,
        acceptance_profile_id="m2-worker",
        stagnation_steps=0,
        exploration_window_index=0,
        accepted_rewrites=0,
        accepted_non_improving_rewrites=0,
        consecutive_non_improving_rewrites=0,
        witness_cap=100,
        invocation_ordinal=invocation_ordinal,
        forbidden_lengths=(4, 6),
    )


def test_worker_proves_required_os_sandbox_controls_and_no_plan() -> None:
    with IsolatedPolicyWorkerV1(NO_PLAN_SOURCE) as worker:
        result = worker.invoke(
            context=_context(),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
        telemetry = worker.telemetry()

    assert result.outcome == "NO_PLAN"
    assert result.no_plan == NoPlan("EXPLICIT")
    assert result.failure is None
    controls = telemetry["controls"]
    assert isinstance(controls, dict)
    assert controls["cwd"] == "/work"
    assert controls["cwd_empty"] is True
    assert controls["environment"] == {
        "HOME": "/work",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin",
        "PWD": "/work",
    }
    assert all(controls["namespaces"].values())
    assert all(controls["seccomp"].values())
    assert controls["no_new_privileges"] is True
    assert controls["rlimits"]["address_space"] == [256 * 1024 * 1024] * 2
    assert telemetry["captured_stderr_bytes"] == 0


def test_worker_projects_immutable_policy_context_and_label_opaque_graph_view() -> None:
    source = """\
def propose(ctx, graph, api, seed):
    if ctx.forbidden_lengths != (4, 6):
        return api.no_plan(reason="EXPLICIT")
    if ctx.invocation_ordinal != 7:
        return api.no_plan(reason="EXPLICIT")
    if graph.order != 6:
        return api.no_plan(reason="EXPLICIT")
    if graph.edge_count != 9:
        return api.no_plan(reason="EXPLICIT")
    if graph.minimum_degree != 3:
        return api.no_plan(reason="EXPLICIT")
    if graph.maximum_degree != 3:
        return api.no_plan(reason="EXPLICIT")
    return api.no_plan(reason="NO_MATCH")
"""
    with IsolatedPolicyWorkerV1(source) as worker:
        result = worker.invoke(
            context=_context(7),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
    assert result.no_plan == NoPlan("NO_MATCH")


def test_worker_returns_only_host_minted_plan_and_replays_trace() -> None:
    graph = _cubic_graph()
    traces = []
    plans = []
    with IsolatedPolicyWorkerV1(ADD_EDGE_SOURCE) as worker:
        for _ in range(2):
            result = worker.invoke(
                context=_context(),
                graph=graph,
                rewrite_host=HOST,
                seed=11,
            )
            assert result.outcome == "REWRITE_PLAN"
            assert result.rewrite_plan is not None
            plans.append(result.rewrite_plan)
            traces.append(tuple(event.as_dict() for event in result.semantic_trace))
        assert worker.telemetry()["calls"] == 2
        assert worker.usable

    assert plans[0] == plans[1]
    assert traces[0] == traces[1]
    assert [event["method"] for event in traces[0]] == [
        "non_edges_legal",
        "pick",
        "add_edge",
        "emit",
    ]
    assert "$ref" not in repr(traces)
    assert all("wall" not in repr(event) for trace in traces for event in trace)


@pytest.mark.parametrize(
    ("source", "code"),
    (
        (
            """\
def propose(ctx, graph, api, seed):
    value = 1 // 0
    return api.no_plan()
""",
            "POLICY_EXCEPTION",
        ),
        (
            """\
def propose(ctx, graph, api, seed):
    value = [0] * 100000000
    return api.no_plan()
""",
            "MEMORY_LIMIT_EXCEEDED",
        ),
        (
            """\
def propose(ctx, graph, api, seed):
    return 7
""",
            "INVALID_RETURN",
        ),
    ),
)
def test_policy_exception_memory_and_invalid_return_are_program_failures(
    source: str,
    code: str,
) -> None:
    worker = IsolatedPolicyWorkerV1(source)
    result = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    assert result.outcome == "PROGRAM_FAILURE"
    assert result.failure is not None
    assert result.failure.classification == "PROGRAM_FAILURE"
    assert result.failure.code == code
    assert not worker.usable
    with pytest.raises(PolicyInfrastructureError, match="cannot be reused"):
        worker.invoke(
            context=_context(1),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
    worker.close()


def test_propose_timeout_kills_and_reaps_worker() -> None:
    source = """\
def propose(ctx, graph, api, seed):
    value = 0
    for first in range(64):
        for second in range(64):
            for third in range(64):
                for fourth in range(64):
                    value = value + first + second + third + fourth
    return api.no_plan()
"""
    worker = IsolatedPolicyWorkerV1(
        source,
        PolicyRuntimeLimitsV1(
            propose_wall_seconds=0.0001,
        ),
    )
    result = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    assert result.outcome == "PROGRAM_FAILURE"
    assert result.failure is not None
    assert result.failure.code == "PROPOSE_TIMEOUT"
    assert not worker.usable
    worker.close()


def test_worker_process_crash_is_a_program_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = IsolatedPolicyWorkerV1(NO_PLAN_SOURCE)
    send = worker._send  # noqa: SLF001 - deterministic crash injection

    def crash_before_request(payload: object, limit: int) -> None:
        if isinstance(payload, dict) and payload.get("type") == "invoke":
            os.killpg(worker._process.pid, signal.SIGKILL)  # noqa: SLF001
            worker._process.wait(timeout=1.0)  # noqa: SLF001
        send(payload, limit)

    monkeypatch.setattr(worker, "_send", crash_before_request)
    result = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    assert result.outcome == "PROGRAM_FAILURE"
    assert result.failure is not None
    assert result.failure.code == "WORKER_CRASH"
    assert not worker.usable
    worker.close()


def test_host_api_time_is_included_in_propose_wall_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SafeGraphSessionV1.handle_call

    def slow_call(
        session: SafeGraphSessionV1,
        method: str,
        arguments: dict[str, object],
    ) -> object:
        time.sleep(0.02)
        return original(session, method, arguments)

    monkeypatch.setattr(SafeGraphSessionV1, "handle_call", slow_call)
    worker = IsolatedPolicyWorkerV1(
        NO_PLAN_SOURCE,
        PolicyRuntimeLimitsV1(propose_wall_seconds=0.01),
    )
    result = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    assert result.outcome == "PROGRAM_FAILURE"
    assert result.failure is not None
    assert result.failure.code == "PROPOSE_TIMEOUT"
    assert not worker.usable
    worker.close()


def test_worker_lifetime_rotates_between_calls_without_program_failure() -> None:
    worker = IsolatedPolicyWorkerV1(
        ADD_EDGE_SOURCE,
        PolicyRuntimeLimitsV1(
            propose_wall_seconds=0.05,
            worker_lifetime_seconds=0.25,
        ),
    )
    first = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    first_pid = worker.telemetry()["pid"]
    first_controls = worker.telemetry()["controls"]
    first_process = worker._process  # noqa: SLF001 - process-rotation evidence
    worker._started_at -= 0.24  # noqa: SLF001 - deterministic host-idle simulation
    second = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    second_process = worker._process  # noqa: SLF001 - process-rotation evidence
    worker._started_at -= 0.24  # noqa: SLF001 - deterministic host-idle simulation
    result = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    telemetry = worker.telemetry()
    assert first.outcome == second.outcome == result.outcome == "REWRITE_PLAN"
    assert first.failure is second.failure is result.failure is None
    assert first.rewrite_plan == second.rewrite_plan == result.rewrite_plan
    assert first.semantic_trace == second.semantic_trace == result.semantic_trace
    assert telemetry["pid"] != first_pid
    assert telemetry["rotations"] == 2
    assert telemetry["calls"] == 3
    assert telemetry["failures"] == 0
    assert telemetry["controls"] == first_controls
    assert first_process.poll() is not None
    assert first_process.stdin is not None and first_process.stdin.closed
    assert first_process.stdout is not None and first_process.stdout.closed
    assert second_process.poll() is not None
    assert second_process.stdin is not None and second_process.stdin.closed
    assert second_process.stdout is not None and second_process.stdout.closed
    assert worker.usable
    final_process = worker._process  # noqa: SLF001 - cleanup evidence
    worker.close()
    assert final_process.stdin is not None and final_process.stdin.closed
    assert final_process.stdout is not None and final_process.stdout.closed
    worker.close()
    assert worker.telemetry()["usable"] is False
    assert worker.captured_stderr() == ""


def test_rotated_worker_receives_the_full_propose_wall_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = SafeGraphSessionV1.handle_call

    def slow_call(
        session: SafeGraphSessionV1,
        method: str,
        arguments: dict[str, object],
    ) -> object:
        time.sleep(0.05)
        return original(session, method, arguments)

    monkeypatch.setattr(SafeGraphSessionV1, "handle_call", slow_call)
    worker = IsolatedPolicyWorkerV1(
        NO_PLAN_SOURCE,
        PolicyRuntimeLimitsV1(
            propose_wall_seconds=0.15,
            worker_lifetime_seconds=0.5,
        ),
    )
    worker._started_at = time.monotonic() - 0.49  # noqa: SLF001
    result = worker.invoke(
        context=_context(),
        graph=_cubic_graph(),
        rewrite_host=HOST,
        seed=11,
    )
    assert result.outcome == "NO_PLAN"
    assert result.failure is None
    assert 0.05 <= result.wall_seconds < 0.15
    assert worker.telemetry()["rotations"] == 1
    worker.close()


def test_rotation_startup_failure_is_infrastructure_not_program_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = IsolatedPolicyWorkerV1(
        NO_PLAN_SOURCE,
        PolicyRuntimeLimitsV1(
            propose_wall_seconds=0.05,
            worker_lifetime_seconds=0.25,
        ),
    )
    process = worker._process  # noqa: SLF001 - process-rotation evidence
    worker._started_at -= 0.24  # noqa: SLF001 - deterministic host-idle simulation

    def fail_spawn() -> None:
        raise UnsupportedPolicySandboxError("fixture re-attestation failure")

    monkeypatch.setattr(worker, "_spawn_process", fail_spawn)
    with pytest.raises(
        UnsupportedPolicySandboxError,
        match="fixture re-attestation failure",
    ):
        worker.invoke(
            context=_context(),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
    assert process.poll() is not None
    assert worker.telemetry()["failures"] == 0
    assert not worker.usable
    assert worker.captured_stderr() == ""
    worker.close()


@pytest.mark.parametrize(
    "startup_error",
    (
        _WorkerTimeout("fixture startup timeout"),
        _WorkerCrash("fixture startup crash"),
    ),
)
def test_initial_worker_startup_failures_are_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
    startup_error: Exception,
) -> None:
    def fail_receive(
        _worker: IsolatedPolicyWorkerV1,
        _deadline: float,
    ) -> dict[str, object]:
        raise startup_error

    monkeypatch.setattr(IsolatedPolicyWorkerV1, "_receive", fail_receive)
    with pytest.raises(PolicyInfrastructureError, match="worker initialization failed"):
        IsolatedPolicyWorkerV1(NO_PLAN_SOURCE)


def test_failed_rotation_shutdown_is_infrastructure_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = IsolatedPolicyWorkerV1(
        NO_PLAN_SOURCE,
        PolicyRuntimeLimitsV1(
            propose_wall_seconds=0.05,
            worker_lifetime_seconds=0.25,
        ),
    )
    process = worker._process  # noqa: SLF001 - rotation failure evidence
    worker._started_at -= 0.24  # noqa: SLF001 - deterministic host-idle simulation
    monkeypatch.setattr(
        worker,
        "_receive",
        lambda _deadline: {"status": "invalid", "type": "shutdown"},
    )
    with pytest.raises(PolicyInfrastructureError, match="cleanly rotated"):
        worker.invoke(
            context=_context(),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
    assert process.poll() is not None
    assert worker.telemetry()["rotations"] == 0
    assert worker.telemetry()["failures"] == 0
    assert not worker.usable
    worker.close()


def test_worker_crash_while_host_is_idle_is_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = IsolatedPolicyWorkerV1(NO_PLAN_SOURCE)

    def crash_during_host_setup(graph: GraphState) -> object:
        os.killpg(worker._process.pid, signal.SIGKILL)  # noqa: SLF001
        worker._process.wait(timeout=1.0)  # noqa: SLF001
        return graph_view_v1(graph)

    monkeypatch.setattr(
        "mutation_forge.native_v3_python.runner.graph_view_v1",
        crash_during_host_setup,
    )
    with pytest.raises(PolicyInfrastructureError, match="exited while idle"):
        worker.invoke(
            context=_context(),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
    assert worker.telemetry()["failures"] == 0
    assert not worker.usable
    worker.close()


def test_worker_crash_at_rotation_boundary_is_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = IsolatedPolicyWorkerV1(
        NO_PLAN_SOURCE,
        PolicyRuntimeLimitsV1(
            propose_wall_seconds=0.05,
            worker_lifetime_seconds=0.25,
        ),
    )
    process = worker._process  # noqa: SLF001 - rotation race evidence
    worker._started_at -= 0.24  # noqa: SLF001 - deterministic host-idle simulation
    rotate = worker._rotate_process  # noqa: SLF001 - deterministic race injection

    def crash_at_rotation_boundary() -> None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=1.0)
        rotate()

    monkeypatch.setattr(worker, "_rotate_process", crash_at_rotation_boundary)
    with pytest.raises(PolicyInfrastructureError, match="cleanly rotated"):
        worker.invoke(
            context=_context(),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
    assert process.poll() is not None
    assert worker.telemetry()["rotations"] == 0
    assert worker.telemetry()["failures"] == 0
    assert not worker.usable
    worker.close()


@pytest.mark.parametrize(
    ("source", "limits", "code"),
    (
        (
            """\
def propose(ctx, graph, api, seed):
    for value in range(2):
        pass
    return api.no_plan()
""",
            PolicyRuntimeLimitsV1(loop_body_entries=1),
            "LOOP_BUDGET_EXCEEDED",
        ),
        (
            """\
def helper_value(value):
    return value

def propose(ctx, graph, api, seed):
    first = helper_value(seed)
    second = helper_value(first)
    return api.no_plan()
""",
            PolicyRuntimeLimitsV1(helper_invocations=1),
            "HELPER_BUDGET_EXCEEDED",
        ),
        (
            """\
def helper_second(value):
    return value

def helper_first(value):
    return helper_second(value)

def propose(ctx, graph, api, seed):
    value = helper_first(seed)
    return api.no_plan()
""",
            PolicyRuntimeLimitsV1(helper_call_depth=1),
            "HELPER_DEPTH_EXCEEDED",
        ),
        (
            """\
def propose(ctx, graph, api, seed):
    candidates = api.non_edges_legal()
    return api.no_plan()
""",
            PolicyRuntimeLimitsV1(total_api_calls=1),
            "API_CALL_BUDGET_EXCEEDED",
        ),
    ),
)
def test_loop_helper_and_api_budgets_fail_closed(
    source: str,
    limits: PolicyRuntimeLimitsV1,
    code: str,
) -> None:
    with IsolatedPolicyWorkerV1(source, limits) as worker:
        result = worker.invoke(
            context=_context(),
            graph=_cubic_graph(),
            rewrite_host=HOST,
            seed=11,
        )
    assert result.outcome == "PROGRAM_FAILURE"
    assert result.failure is not None
    assert result.failure.code == code


def test_contract_invalid_source_never_starts_worker() -> None:
    with pytest.raises(ValueError, match="CONTRACT_INVALID"):
        IsolatedPolicyWorkerV1(
            """\
import os
def propose(ctx, graph, api, seed):
    return api.no_plan()
"""
        )
    with pytest.raises(ValueError, match="CONTRACT_INVALID"):
        IsolatedPolicyWorkerV1(
            """\
def propose(ctx, graph, api, seed):
    return RewritePlan((), (), "forged")
"""
        )


@pytest.mark.parametrize(
    "error",
    (
        KeyError("trusted host defect"),
        RuntimeError("trusted host defect"),
        ValueError("trusted host defect"),
    ),
)
def test_trusted_rewrite_host_failure_is_infrastructure_not_program_fitness(
    error: Exception,
) -> None:
    worker = IsolatedPolicyWorkerV1(ADD_EDGE_SOURCE)
    with pytest.raises(PolicyInfrastructureError, match="rewrite host"):
        worker.invoke(
            context=_context(),
            graph=_cubic_graph(),
            rewrite_host=_BrokenHost(error),
            seed=11,
        )
    assert not worker.usable
    worker.close()


def test_protocol_is_canonical_framed_json_and_rejects_oversize() -> None:
    first = _canonical_frame({"z": 1, "a": 2}, 1_024)
    second = _canonical_frame({"a": 2, "z": 1}, 1_024)
    assert first == second
    assert first[4:] == b'{"a":2,"z":1}'
    with pytest.raises(PolicyProtocolError, match="exceeds"):
        _canonical_frame({"value": "x" * 100}, 16)
    assert _strict_json_object(b'{"a":1,"z":2}') == {"a": 1, "z": 2}
    for malformed in (
        b'{"a":1,"a":2}',
        b'{"value":NaN}',
        b'{ "a": 1 }',
        b'{"z":2,"a":1}',
    ):
        with pytest.raises(PolicyProtocolError):
            _strict_json_object(malformed)


@pytest.mark.parametrize(
    "changes",
    (
        {"propose_wall_seconds": math.nan},
        {"propose_wall_seconds": math.inf},
        {"worker_lifetime_seconds": math.nan},
        {"propose_wall_seconds": 1.01},
        {"address_space_bytes": 256 * 1024 * 1024 + 1},
        {"process_count": 2},
        {"graph_order": 129},
    ),
)
def test_runtime_limits_reject_nonfinite_and_above_cap_values(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PolicyRuntimeLimitsV1(**changes)  # type: ignore[arg-type]


def test_graph_order_cap_is_host_infrastructure_failure() -> None:
    with (
        IsolatedPolicyWorkerV1(NO_PLAN_SOURCE) as worker,
        pytest.raises(PolicyInfrastructureError, match="graph order"),
    ):
        worker.invoke(
            context=_context(),
            graph=GraphState(129, ()),
            rewrite_host=HOST,
            seed=11,
        )


def test_worker_is_only_generated_source_execution_module() -> None:
    calls: dict[str, set[str]] = {}
    imports: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls[path.name] = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } & {"compile", "exec", "eval"}
        imports[path.name] = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
    assert calls["worker_main.py"] == {"compile", "exec"}
    assert all(not value for name, value in calls.items() if name != "worker_main.py")
    assert all(
        not any(module == "pickle" or module.startswith("importlib") for module in modules)
        for modules in imports.values()
    )


def test_missing_bubblewrap_fails_before_worker_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mutation_forge.native_v3_python.runner.shutil.which",
        lambda _name: None,
    )
    with pytest.raises(UnsupportedPolicySandboxError, match="bubblewrap"):
        IsolatedPolicyWorkerV1(NO_PLAN_SOURCE)
