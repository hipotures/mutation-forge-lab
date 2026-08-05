"""Deterministic in-memory JSONL app-server fixture (never performs inference)."""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class _Out:
    def __init__(self):
        self.q: queue.Queue[bytes] = queue.Queue()

    def put(self, v: dict[str, Any] | bytes):
        self.q.put(
            v if isinstance(v, bytes) else json.dumps(v, separators=(",", ":")).encode() + b"\n"
        )

    def readline(self, _size: int = -1):
        return self.q.get(timeout=0.05)

    def close(self):
        self.q.put(b"")


class _In:
    def __init__(self, p):
        self.p = p

    def write(self, b):
        for line in b.splitlines():
            self.p.receive(line)
        return len(b)

    def flush(self):
        pass

    def close(self):
        pass


@dataclass
class FakeScenario:
    enabled_skills: list[str] = field(default_factory=list)
    final_text: str = "fixture answer"
    usage: dict[str, Any] | None = field(
        default_factory=lambda: {
            "inputTokens": 2,
            "cachedInputTokens": 0,
            "cacheWriteInputTokens": 0,
            "outputTokens": 3,
            "reasoningOutputTokens": 1,
            "totalTokens": 5,
        }
    )
    terminal_status: str = "completed"
    interleave: bool = True
    server_request: bool = False
    malformed: bool = False
    oversized: bool = False
    crash: bool = False
    unknown_notification: bool = False
    model_rerouted: bool = False
    item_type: str = "agentMessage"
    item_id: str = "item-1"
    delta_item_id: str | None = None
    completed_item_id: str | None = None
    late_item: bool = False
    global_notification: str | None = None
    global_after_completion: bool = False
    error_will_retry: bool | None = None
    thread_started_notification: str | None = None
    turn_started_before_response: bool = False
    item_started_before_response: bool = False
    turn_completed_item_id: str | None = None
    completed_items_view: str = "loaded"


class FakeProcess:
    def __init__(self, scenario: FakeScenario | None = None, **kwargs: Any):
        self.scenario = scenario or FakeScenario()
        self.environment = kwargs.get("env", {})
        self.stdout = _Out()
        self.stderr = _Out()
        self.stdin = _In(self)
        self.returncode = None

    def receive(self, line: bytes):
        if self.scenario.crash:
            self.returncode = 1
            return
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            self.stdout.put(b"not-json\n")
            return
        m, i = r.get("method"), r.get("id")
        p = r.get("params", {})
        s = self.scenario
        q = self.stdout

        def response(result):
            q.put({"id": i, "result": result})

        if m == "initialize":
            response({"serverInfo": {"name": "fake"}})
        elif m == "initialized":
            return
        elif m == "skills/list":
            response(
                {
                    "data": [
                        {
                            "cwd": p.get("cwds", [""])[0],
                            "skills": [{"path": x, "enabled": True} for x in s.enabled_skills],
                            "errors": [],
                        }
                    ]
                }
            )
        elif m == "skills/config/write":
            s.enabled_skills = [x for x in s.enabled_skills if x != p.get("path")]
            response({"effectiveEnabled": False})
        elif m == "model/list":
            response(
                {
                    "data": [
                        {
                            "model": "gpt-5.6-luna",
                            "supportedReasoningEfforts": [{"reasoningEffort": "high"}],
                        }
                    ]
                }
            )
        elif m == "thread/start":
            if s.thread_started_notification == "nested-before-thread-response":
                q.put({"method": "thread/started", "params": {"thread": {"id": "thread-1"}}})
            sandbox = (
                {"type": "readOnly", "networkAccess": False}
                if p["sandbox"] == "read-only"
                else {"type": "dangerFullAccess"}
            )
            response(
                {
                    "approvalPolicy": "never",
                    "approvalsReviewer": "user",
                    "cwd": p["cwd"],
                    "instructionSources": [],
                    "model": p["model"],
                    "modelProvider": "openai",
                    "reasoningEffort": p["config"]["model_reasoning_effort"],
                    "runtimeWorkspaceRoots": [],
                    "sandbox": sandbox,
                    "thread": {
                        "id": "thread-1",
                        "sessionId": "session-1",
                        "path": str(
                            Path(self.environment["CODEX_HOME"]) / "rollout.jsonl"
                        ),
                        "cwd": p["cwd"],
                        "ephemeral": p["ephemeral"],
                    },
                }
            )
            if s.thread_started_notification in {"top-level", "nested"}:
                q.put(
                    {
                        "method": "thread/started",
                        "params": {"threadId": "thread-1"}
                        if s.thread_started_notification == "top-level"
                        else {"thread": {"id": "thread-1"}},
                    }
                )
        elif m == "turn/start":
            tid = "turn-1"
            iid = s.item_id
            if s.server_request:
                q.put({"id": 901, "method": "item/toolCall", "params": {}})
            if s.turn_started_before_response:
                q.put(
                    {
                        "method": "turn/started",
                        "params": {
                            "threadId": "thread-1",
                            "turn": {"id": tid, "items": [], "status": "inProgress"},
                        },
                    }
                )
            if s.item_started_before_response:
                q.put(
                    {
                        "method": "item/started",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": tid,
                            "item": {"id": iid, "type": s.item_type},
                        },
                    }
                )
            response({"turn": {"id": tid, "items": [], "status": "inProgress"}})
            if s.thread_started_notification == "nested-after-turn-response":
                q.put({"method": "thread/started", "params": {"thread": {"id": "thread-1"}}})
            if not s.turn_started_before_response:
                q.put(
                    {
                        "method": "turn/started",
                        "params": {
                            "threadId": "thread-1",
                            "turn": {"id": tid, "items": [], "status": "inProgress"},
                        },
                    }
                )
            q.put(
                {
                    "method": "thread/status/changed",
                    "params": {"threadId": "thread-1", "status": {"type": "active"}},
                }
            )
            if s.global_notification and not s.global_after_completion:
                q.put(
                    {
                        "method": s.global_notification,
                        "params": {"rateLimits": {}}
                        if s.global_notification.endswith("rateLimits/updated")
                        else {},
                    }
                )
            if not s.item_started_before_response:
                q.put(
                    {
                        "method": "item/started",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": tid,
                            "item": {"id": iid, "type": s.item_type},
                        },
                    }
                )
            if s.unknown_notification:
                q.put(
                    {"method": "fixture/unknown", "params": {"threadId": "thread-1", "turnId": tid}}
                )
            if s.model_rerouted:
                q.put(
                    {"method": "model/rerouted", "params": {"threadId": "thread-1", "turnId": tid}}
                )
            if s.error_will_retry is not None:
                q.put(
                    {
                        "method": "error",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": tid,
                            "error": {},
                            "willRetry": s.error_will_retry,
                        },
                    }
                )
            if s.interleave:
                q.put(
                    {
                        "method": "item/agentMessage/delta",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": tid,
                            "itemId": s.delta_item_id or iid,
                            "delta": "fixture",
                        },
                    }
                )
            q.put(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": tid,
                        "item": {
                            "id": s.completed_item_id or iid,
                            "type": s.item_type,
                            "phase": "final_answer",
                            "text": s.final_text,
                        },
                    },
                }
            )
            q.put(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {
                            "id": tid,
                            "items": (
                                []
                                if s.completed_items_view == "notLoaded"
                                else [
                                    {
                                        "id": (
                                            s.turn_completed_item_id
                                            or s.completed_item_id
                                            or iid
                                        ),
                                        "type": s.item_type,
                                    }
                                ]
                            ),
                            "itemsView": s.completed_items_view,
                            "status": s.terminal_status,
                        },
                    },
                }
            )
            if s.late_item:
                q.put(
                    {
                        "method": "item/started",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": tid,
                            "item": {"id": iid, "type": s.item_type},
                        },
                    }
                )
            if s.global_notification and s.global_after_completion:
                q.put(
                    {
                        "method": s.global_notification,
                        "params": {"rateLimits": {}}
                        if s.global_notification.endswith("rateLimits/updated")
                        else {},
                    }
                )
            if s.usage:
                q.put(
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": tid,
                            "tokenUsage": {"last": s.usage},
                        },
                    }
                )
            if s.malformed:
                q.put(b"not-json\n")
            if s.oversized:
                q.put(b"{" + b"x" * 1024 * 1024 + b"}\n")
        elif m == "turn/interrupt":
            response({})

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def fake_process_factory(scenario: FakeScenario | None = None):
    return lambda *a, **k: FakeProcess(scenario, **k)
