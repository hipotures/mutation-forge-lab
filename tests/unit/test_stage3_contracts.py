from __future__ import annotations

import pytest

from mutation_forge.stage3.contracts import (
    GeneratedPolicy,
    Stage3ContractError,
    parse_generated_policy,
)


def test_generated_policy_is_strict_and_immutable() -> None:
    value = {
        "schema_version": "stage3.generated_policy.v1",
        "source": "def priority(ctx, proposal):\n    return 0.0",
        "design_summary": "finite policy",
        "used_fields": ["proposal.k"],
        "assumptions": ["legal inputs"],
    }
    result = parse_generated_policy(value)
    assert isinstance(result, GeneratedPolicy)
    assert result.used_fields == ("proposal.k",)
    with pytest.raises(AttributeError):
        result.source = "changed"  # type: ignore[misc]
    with pytest.raises(Stage3ContractError):
        parse_generated_policy({**value, "extra": True})
    with pytest.raises(Stage3ContractError):
        parse_generated_policy({**value, "used_fields": ["ctx.unknown"]})
