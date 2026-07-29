"""Deterministic in-memory app-server process used by Stage 3 unit tests."""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field
from typing import Any


class _Input:
    def __init__(self, owner: FakeProcess) -> None:
        self.owner = owner

    def write(self, data: bytes) -> int:
        for line in data.splitlines():
            self.owner.receive(line)
        return len(data)

    def flush(self) -> None:
        return None


class _Output:
    def __init__(self) -> None:
        self.items: queue.Queue[bytes] = queue.Queue()

    def put(self, value: dict[str, Any] | bytes) -> None:
        self.items.put(
            value
            if isinstance(value, bytes)
            else (json.dumps(value, separators=(",", ":")).encode() + b"\n")
        )

    def readline(self) -> bytes:
        return self.items.get(timeout=2.0)


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


class FakeProcess:
    def __init__(self, scenario: FakeScenario | None = None, **_: Any) -> None:
        self.scenario = scenario or FakeScenario()
        self.stdout = _Output()
        self.stderr = _Output()
        self.stdin = _Input(self)
        self.returncode: int | None = None
        self._turn = 0

    def receive(self, line: bytes) -> None:
        if self.scenario.crash:
            self.returncode = 1
            return
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            self.stdout.put(b"not-json\n")
            return
        method, request_id = request.get("method"), request.get("id")
        params = request.get("params", {})
        if method == "initialize":
            self._response(request_id, {"serverInfo": {"name": "fake"}})
        elif method == "skills/list":
            self._response(
                request_id,
                {
                    "data": [
                        {
                            "cwd": params.get("cwds", [""])[0],
                            "skills": [
                                {"name": "fixture", "path": path, "enabled": True}
                                for path in self.scenario.enabled_skills
                            ],
                            "errors": [],
                        }
                    ]
                },
            )
        elif method == "skills/config/write":
            self.scenario.enabled_skills = [
                p for p in self.scenario.enabled_skills if p != params.get("path")
            ]
            self._response(request_id, {"effectiveEnabled": False})
        elif method == "model/list":
            self._response(
                request_id,
                {
                    "data": [
                        {
                            "id": "gpt-5.6-luna",
                            "model": "gpt-5.6-luna",
                            "displayName": "GPT-5.6 Luna",
                            "description": "fixture",
                            "hidden": False,
                            "isDefault": False,
                            "defaultReasoningEffort": "medium",
                            "supportedReasoningEfforts": [
                                {"reasoningEffort": "high", "description": "fixture"}
                            ],
                        }
                    ],
                    "nextCursor": None,
                },
            )
        elif method == "thread/start":
            self._response(
                request_id,
                {
                    "approvalPolicy": "never",
                    "approvalsReviewer": "user",
                    "cwd": params["cwd"],
                    "instructionSources": [],
                    "model": params["model"],
                    "modelProvider": "openai",
                    "runtimeWorkspaceRoots": [],
                    "sandbox": {"type": "readOnly", "networkAccess": False},
                    "thread": {
                        "id": "thread-1",
                        "sessionId": "session-1",
                        "path": "/private/rollout.jsonl",
                    },
                },
            )
        elif method == "turn/start":
            self._turn += 1
            if self.scenario.server_request:
                self.stdout.put({"id": 901, "method": "item/toolCall", "params": {}})
            self._response(request_id, {"turn": {"id": "turn-1"}})
            if self.scenario.unknown_notification:
                self.stdout.put(
                    {
                        "method": "fixture/unknown",
                        "params": {"threadId": "thread-1", "turnId": "turn-1"},
                    }
                )
            if self.scenario.interleave:
                self.stdout.put(
                    {
                        "method": "item/agentMessage/delta",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": "turn-1",
                            "itemId": "item-1",
                            "delta": "fixture",
                        },
                    }
                )
            self.stdout.put(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "item": {
                            "id": "item-1",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": self.scenario.final_text,
                        },
                    },
                }
            )
            self.stdout.put(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "turn": {"id": "turn-1", "status": self.scenario.terminal_status},
                    },
                }
            )
            if self.scenario.usage is not None:
                self.stdout.put(
                    {
                        "method": "thread/tokenUsage/updated",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": "turn-1",
                            "tokenUsage": {"last": self.scenario.usage},
                        },
                    }
                )
            if self.scenario.malformed:
                self.stdout.put(b"not-json\n")
            if self.scenario.oversized:
                self.stdout.put(b"{" + b"x" * (1024 * 1024) + b"}\n")

    def _response(self, request_id: Any, result: dict[str, Any]) -> None:
        self.stdout.put({"id": request_id, "result": result})

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def fake_process_factory(scenario: FakeScenario | None = None):
    def factory(*args: Any, **kwargs: Any) -> FakeProcess:
        return FakeProcess(scenario, **kwargs)

    return factory
