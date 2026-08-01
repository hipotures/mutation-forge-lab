# Stage 6 tampering corpus

The synthetic corpus is generated deterministically by
`mutation_forge.stage6_independent.redteam.write_fixture_set`.  It contains a
valid paired two-shard evidence envelope plus 25 corruption cases and four
metamorphic cases (shard/record permutation, timing-only changes, and an
equivalent vertex relabeling).  The fixture is intentionally tiny so it can be
used by unit tests without touching a Stage 5 run or the network.

Case names are exported as `CASE_NAMES` and `METAMORPHIC_CASES`; the verifier
must reject every corruption case and accept every metamorphic case.
