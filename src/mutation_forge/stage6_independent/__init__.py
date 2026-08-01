"""Independent Stage 6 verification, red-team, and replication tools.

The modules in this package deliberately avoid the Stage 5 analysis and
orchestration implementations.  They operate on frozen configuration and raw
artifacts, and expose deterministic, machine-readable results for the Stage 6
CLI.
"""

from __future__ import annotations

from .orchestrator import (
    audit,
    freeze,
    plan_fresh,
    redteam,
    reduce,
    run_fresh,
)

DECISIONS = (
    "GO_TO_STAGE_7",
    "NO_GO",
    "INCONCLUSIVE_INFRASTRUCTURE_FAILURE",
)

__all__ = ["DECISIONS", "audit", "freeze", "plan_fresh", "redteam", "reduce", "run_fresh"]
