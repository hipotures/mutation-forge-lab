from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaselinePolicy:
    policy_id: str
    operator_family: str


HEG_UNIFORM_TWO_SWITCH = BaselinePolicy(
    policy_id="heg_uniform_two_switch",
    operator_family="heg_uniform_two_switch",
)
HEG_FORBIDDEN_CYCLE_BREAK = BaselinePolicy(
    policy_id="heg_forbidden_cycle_break",
    operator_family="heg_forbidden_cycle_break",
)

BASELINES = {
    policy.policy_id: policy
    for policy in (HEG_UNIFORM_TWO_SWITCH, HEG_FORBIDDEN_CYCLE_BREAK)
}
