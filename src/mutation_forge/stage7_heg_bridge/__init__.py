"""Out-of-tree Stage 7 HEG integration decision bridge.

This namespace is deliberately a reference harness.  It never patches or
writes the sibling HEG checkout and it only loads the reviewed catalog entry
for the frozen Stage 4R policy.
"""

from mutation_forge.stage7_heg_bridge.contract import (
    CATALOG_ID,
    FROZEN_POLICY_ID,
    HEG_COMMIT,
    POLICY_BEHAVIOR_SHA256,
    POLICY_SOURCE_SHA256,
    verify_frozen_policy,
)

__all__ = [
    "CATALOG_ID",
    "FROZEN_POLICY_ID",
    "HEG_COMMIT",
    "POLICY_BEHAVIOR_SHA256",
    "POLICY_SOURCE_SHA256",
    "verify_frozen_policy",
]
