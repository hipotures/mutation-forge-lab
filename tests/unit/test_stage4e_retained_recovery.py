# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

from mutation_forge import stage4e_retained_recovery as recovery


def test_recursive_projection_removes_nested_timing_ns_only() -> None:
    primary = {
        "episode_id": "episode-1",
        "nested": {"timing_ns": {"total": 10}, "scientific": 7},
        "steps": [{"timing_ns": 11, "value": 3}],
    }
    replay = {
        "episode_id": "episode-1",
        "nested": {"timing_ns": {"total": 99}, "scientific": 7},
        "steps": [{"timing_ns": 111, "value": 3}],
    }

    comparison = recovery.compare_canonical_rows({"episode-1": primary}, {"episode-1": replay})

    assert recovery.timing_stripped_projection(primary) == recovery.timing_stripped_projection(replay)
    assert comparison["rows_exact"] is True
    assert comparison["non_timing_differences"] == []


def test_non_timing_difference_is_reported_with_exact_path() -> None:
    primary = {"episode_id": "episode-1", "nested": {"score": 7, "timing_ns": 1}}
    replay = {"episode_id": "episode-1", "nested": {"score": 8, "timing_ns": 2}}

    comparison = recovery.compare_canonical_rows({"episode-1": primary}, {"episode-1": replay})

    assert comparison["rows_exact"] is False
    assert comparison["non_timing_differences"] == [
        {
            "episode_id": "episode-1",
            "fields": [{"path": "$.nested.score", "primary": 7, "replay": 8}],
        }
    ]


def test_recovery_command_is_artifact_only_and_does_not_evaluate(tmp_path: Path) -> None:
    # The recovery module has no evaluator import or callable.  A missing run
    # must fail closed without creating an episode or invoking Stage 4E code.
    assert "run_development_episode" not in recovery.__dict__
    assert "stage4e_execution" not in recovery.__dict__

    result = recovery.recover_retained(
        tmp_path / "missing-run",
        output_dir=tmp_path / "result",
        report_path=tmp_path / "report.md",
    )

    assert result["decision"] == recovery.RETAINED_EVIDENCE_FAILURE
    assert result["provider_calls"] == 0
    assert not (tmp_path / "missing-run").exists()
