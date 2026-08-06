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
        return self.q.get()

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
    terminal_statuses: list[str] | None = None
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
    resume_status_before_response: str | None = None
    resume_status_thread_id: str | None = None
    resume_usage_after_response: bool = False
    resume_usage_turn_id: str = "resume-history-turn"
    turn_started_before_response: bool = False
    item_started_before_response: bool = False
    turn_completed_item_id: str | None = None
    completed_items_view: str = "loaded"
    final_texts: list[str] | None = None
    dangling_reasoning: bool = False
    warning_message: str | None = None
    thread_id: str = "thread-1"
    compaction_terminal_status: str = "completed"
    compaction_item_type: str = "contextCompaction"
    compaction_omit_item_completion: bool = False
    compaction_omit_turn_completion: bool = False
    compaction_close_after_events: bool = False
    compaction_hang: bool = False
    compaction_usage_before_completion: bool = False
    compaction_deprecated_notification: bool = False
    fork_thread_ids: list[str] | None = None
    fork_invalid_last_turn_ids: list[str] = field(default_factory=list)
    fork_in_progress_last_turn_ids: list[str] = field(default_factory=list)
    fork_foreign_source: bool = False
    fork_late_notifications: bool = False


class FakeProcess:
    def __init__(self, scenario: FakeScenario | None = None, **kwargs: Any):
        self.scenario = scenario or FakeScenario()
        self.environment = kwargs.get("env", {})
        self.stdout = _Out()
        self.stderr = _Out()
        self.stdin = _In(self)
        self.returncode = None
        self.turn_index = 0
        self.compaction_index = 0
        self.fork_index = 0
        self.received_requests: list[dict[str, Any]] = []
        self.thread_turns: dict[str, list[dict[str, Any]]] = {
            self.scenario.thread_id: []
        }

    def receive(self, line: bytes):
        if self.scenario.crash:
            self.returncode = 1
            self.stdout.close()
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
        thread_id = s.thread_id
        self.received_requests.append(r)

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
                q.put({"method": "thread/started", "params": {"thread": {"id": thread_id}}})
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
                        "id": thread_id,
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
                        "params": {"threadId": thread_id}
                        if s.thread_started_notification == "top-level"
                        else {"thread": {"id": thread_id}},
                    }
                )
        elif m == "thread/resume":
            resumed_thread_id = p["threadId"]
            if s.resume_status_before_response is not None:
                q.put(
                    {
                        "method": "thread/status/changed",
                        "params": {
                            "threadId": (
                                s.resume_status_thread_id or resumed_thread_id
                            ),
                            "status": {
                                "type": s.resume_status_before_response,
                            },
                        },
                    }
                )
            response(
                {
                    "thread": {
                        "id": resumed_thread_id,
                        "sessionId": "session-1",
                        "path": p.get(
                            "path",
                            str(Path(self.environment["CODEX_HOME"]) / "rollout.jsonl"),
                        ),
                        "cwd": p["cwd"],
                        "ephemeral": False,
                    }
                }
            )
            if s.resume_usage_after_response:
                usage = s.usage or {
                    "inputTokens": 0,
                    "cachedInputTokens": 0,
                    "cacheWriteInputTokens": 0,
                    "outputTokens": 0,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 0,
                }
                q.put(
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {
                            "threadId": resumed_thread_id,
                            "turnId": s.resume_usage_turn_id,
                            "tokenUsage": {
                                "last": usage,
                                "total": usage,
                            },
                        },
                    }
                )
        elif m == "thread/fork":
            source_thread_id = p.get("threadId")
            last_turn_id = p.get("lastTurnId")
            history = self.thread_turns.get(source_thread_id, [])
            matching = [
                index
                for index, turn in enumerate(history)
                if turn.get("id") == last_turn_id
            ]
            if (
                last_turn_id in s.fork_invalid_last_turn_ids
                or not isinstance(source_thread_id, str)
                or not isinstance(last_turn_id, str)
                or len(matching) != 1
            ):
                q.put(
                    {
                        "id": i,
                        "error": {
                            "code": -32600,
                            "message": "invalid fork boundary",
                        },
                    }
                )
                return
            if last_turn_id in s.fork_in_progress_last_turn_ids:
                q.put(
                    {
                        "id": i,
                        "error": {
                            "code": -32600,
                            "message": "fork boundary turn is in progress",
                        },
                    }
                )
                return
            self.fork_index += 1
            child_thread_id = (
                s.fork_thread_ids[self.fork_index - 1]
                if s.fork_thread_ids is not None
                else f"fork-thread-{self.fork_index}"
            )
            included = [
                json.loads(json.dumps(turn))
                for turn in history[: matching[0] + 1]
            ]
            self.thread_turns[child_thread_id] = included
            sandbox = (
                {"type": "readOnly", "networkAccess": False}
                if p["sandbox"] == "read-only"
                else {"type": "dangerFullAccess"}
            )
            child_thread = {
                "cliVersion": "0.146.0",
                "createdAt": 1,
                "cwd": p["cwd"],
                "ephemeral": False,
                "forkedFromId": (
                    "foreign-thread"
                    if s.fork_foreign_source
                    else source_thread_id
                ),
                "historyMode": "full",
                "id": child_thread_id,
                "modelProvider": "openai",
                "path": str(
                    Path(self.environment["CODEX_HOME"])
                    / f"{child_thread_id}.jsonl"
                ),
                "preview": "",
                "sessionId": "session-1",
                "source": "forked",
                "status": {"type": "idle"},
                "turns": included,
                "updatedAt": 1,
            }
            response(
                {
                    "approvalPolicy": p["approvalPolicy"],
                    "approvalsReviewer": "user",
                    "cwd": p["cwd"],
                    "instructionSources": [],
                    "model": p["model"],
                    "modelProvider": "openai",
                    "runtimeWorkspaceRoots": [],
                    "sandbox": sandbox,
                    "thread": child_thread,
                }
            )
            if s.fork_late_notifications:
                usage = s.usage or {
                    "inputTokens": 0,
                    "cachedInputTokens": 0,
                    "cacheWriteInputTokens": 0,
                    "outputTokens": 0,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 0,
                }
                q.put(
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {
                            "threadId": child_thread_id,
                            "turnId": included[-1]["id"],
                            "tokenUsage": {"last": usage, "total": usage},
                        },
                    }
                )
                q.put(
                    {
                        "method": "thread/started",
                        "params": {"thread": child_thread},
                    }
                )
        elif m == "thread/compact/start":
            self.compaction_index += 1
            turn_id = f"compact-{self.compaction_index}"
            item_id = f"context-compaction-{self.compaction_index}"
            response({})
            if s.compaction_hang:
                return
            q.put(
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": thread_id,
                        "turn": {
                            "id": turn_id,
                            "items": [],
                            "status": "inProgress",
                        },
                    },
                }
            )
            q.put(
                {
                    "method": "item/started",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "item": {
                            "id": item_id,
                            "type": s.compaction_item_type,
                        },
                    },
                }
            )
            if s.compaction_usage_before_completion and s.usage:
                q.put(
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {
                            "threadId": thread_id,
                            "turnId": turn_id,
                            "tokenUsage": {"last": s.usage},
                        },
                    }
                )
            if not s.compaction_omit_item_completion:
                q.put(
                    {
                        "method": "item/completed",
                        "params": {
                            "threadId": thread_id,
                            "turnId": turn_id,
                            "item": {
                                "id": item_id,
                                "type": s.compaction_item_type,
                            },
                        },
                    }
                )
            if s.compaction_deprecated_notification:
                q.put(
                    {
                        "method": "thread/compacted",
                        "params": {"threadId": thread_id},
                    }
                )
            if not s.compaction_omit_turn_completion:
                q.put(
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": thread_id,
                            "turn": {
                                "id": turn_id,
                                "items": [
                                    {
                                        "id": item_id,
                                        "type": s.compaction_item_type,
                                    }
                                ],
                                "itemsView": "loaded",
                                "status": s.compaction_terminal_status,
                            },
                        },
                    }
                )
            if not s.compaction_usage_before_completion and s.usage:
                q.put(
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {
                            "threadId": thread_id,
                            "turnId": turn_id,
                            "tokenUsage": {"last": s.usage},
                        },
                    }
                )
            if s.compaction_close_after_events:
                self.returncode = 0
                q.close()
        elif m == "turn/start":
            self.turn_index += 1
            tid = f"turn-{self.turn_index}"
            iid = s.item_id
            thread_id = p.get("threadId", thread_id)
            if s.server_request:
                q.put({"id": 901, "method": "item/toolCall", "params": {}})
            if s.turn_started_before_response:
                q.put(
                    {
                        "method": "turn/started",
                        "params": {
                            "threadId": thread_id,
                            "turn": {"id": tid, "items": [], "status": "inProgress"},
                        },
                    }
                )
            if s.item_started_before_response:
                q.put(
                    {
                        "method": "item/started",
                        "params": {
                            "threadId": thread_id,
                            "turnId": tid,
                            "item": {"id": iid, "type": s.item_type},
                        },
                    }
                )
            response({"turn": {"id": tid, "items": [], "status": "inProgress"}})
            if s.thread_started_notification == "nested-after-turn-response":
                q.put({"method": "thread/started", "params": {"thread": {"id": thread_id}}})
            if not s.turn_started_before_response:
                q.put(
                    {
                        "method": "turn/started",
                        "params": {
                            "threadId": thread_id,
                            "turn": {"id": tid, "items": [], "status": "inProgress"},
                        },
                    }
                )
            q.put(
                {
                    "method": "thread/status/changed",
                    "params": {"threadId": thread_id, "status": {"type": "active"}},
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
                if s.dangling_reasoning:
                    q.put(
                        {
                            "method": "item/started",
                            "params": {
                                "threadId": thread_id,
                                "turnId": tid,
                                "item": {"id": "reasoning-pending", "type": "reasoning"},
                            },
                        }
                    )
                q.put(
                    {
                        "method": "item/started",
                        "params": {
                            "threadId": thread_id,
                            "turnId": tid,
                            "item": {"id": iid, "type": s.item_type},
                        },
                    }
                )
            if s.unknown_notification:
                q.put(
                    {"method": "fixture/unknown", "params": {"threadId": thread_id, "turnId": tid}}
                )
            if s.model_rerouted:
                q.put(
                    {"method": "model/rerouted", "params": {"threadId": thread_id, "turnId": tid}}
                )
            if s.error_will_retry is not None:
                q.put(
                    {
                        "method": "error",
                        "params": {
                            "threadId": thread_id,
                            "turnId": tid,
                            "error": {},
                            "willRetry": s.error_will_retry,
                        },
                    }
                )
            if s.warning_message is not None:
                q.put(
                    {
                        "method": "warning",
                        "params": {
                            "threadId": thread_id,
                            "message": s.warning_message,
                        },
                    }
                )
            if s.interleave:
                q.put(
                    {
                        "method": "item/agentMessage/delta",
                        "params": {
                            "threadId": thread_id,
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
                        "threadId": thread_id,
                        "turnId": tid,
                        "item": {
                            "id": s.completed_item_id or iid,
                            "type": s.item_type,
                            "phase": "final_answer",
                            "text": (
                                s.final_texts[self.turn_index - 1]
                                if s.final_texts is not None
                                else s.final_text
                            ),
                        },
                    },
                }
            )
            q.put(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": thread_id,
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
                            "status": (
                                s.terminal_statuses[self.turn_index - 1]
                                if s.terminal_statuses is not None
                                else s.terminal_status
                            ),
                        },
                    },
                }
            )
            self.thread_turns.setdefault(thread_id, []).append(
                {
                    "id": tid,
                    "items": [
                        {
                            "id": (
                                s.turn_completed_item_id
                                or s.completed_item_id
                                or iid
                            ),
                            "type": s.item_type,
                        }
                    ],
                    "itemsView": s.completed_items_view,
                    "status": (
                        s.terminal_statuses[self.turn_index - 1]
                        if s.terminal_statuses is not None
                        else s.terminal_status
                    ),
                }
            )
            if s.late_item:
                q.put(
                    {
                        "method": "item/started",
                        "params": {
                            "threadId": thread_id,
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
                            "threadId": thread_id,
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
