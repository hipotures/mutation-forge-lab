from __future__ import annotations

from typing import Protocol


class FixedPolicy(Protocol):
    policy_id: str
    operator_family: str
