"""Deterministic synthetic Stage 6 evidence tampering checks."""

from mutation_forge.stage6_independent.redteam import (
    CASE_NAMES,
    METAMORPHIC_ACCEPTED_CASES,
    make_fixture,
    run_redteam,
    tamper_fixture,
    verify_fixture,
)


def test_base_fixture_is_valid_and_tamper_corpus_is_complete() -> None:
    assert verify_fixture(make_fixture())["exact"]
    result = run_redteam()
    assert result["status"] == "passed"
    assert len(result["findings"]) == 30


def test_corruptions_reject_and_metamorphic_changes_pass() -> None:
    fixture = make_fixture()
    for case in CASE_NAMES:
        assert not verify_fixture(tamper_fixture(fixture, case))["exact"], case
    for case in METAMORPHIC_ACCEPTED_CASES:
        assert verify_fixture(tamper_fixture(fixture, case))["exact"], case
    assert not verify_fixture(tamper_fixture(fixture, "fraction_float_drift"))["exact"]
