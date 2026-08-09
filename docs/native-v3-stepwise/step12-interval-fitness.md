# Native v3 step 12: component evidence and interval fitness

Step 12 makes bounded scoring uncertainty part of the scientific result. It
does not change the Native v2 provider transport, add exploratory acceptance,
or introduce graph concurrency.

## Locked scoring protocol

Native v3 uses the existing mandatory HEG C++ score worker through a dedicated
evidence adapter. There is no Python scoring fallback.

- Initial attempt: 50,000 search nodes and 5 seconds.
- Expanded attempt: 200,000 search nodes and 20 seconds.
- Expanded attempts request only unresolved forbidden-cycle lengths.
- Witness-cap saturation is an exact result for the capped objective.
- Search-budget exhaustion and a timeout with safe partial evidence retain
  sound lower and upper bounds.
- A timeout without safe partial evidence is scientifically inconclusive.
- Infrastructure failure and scorer contract violation fail closed.

The protocol identity is `native_v3_score_50k_200k_v1`; serialized component
evidence uses `mforge.native.score_evidence.v3`.

## Exact interval decisions

Count and weighted-penalty intervals are encoded into one mixed-radix energy
interval preserving the tuple:

1. total capped witnesses;
2. weighted capped penalty;
3. edge count.

Utility, best-so-far trajectories, episode AUC, and candidate fitness use
`fractions.Fraction` throughout. JSON artifacts serialize each rational bound
as a numerator and denominator. Floats and interval midpoints are not used for
scientific decisions.

A proposed graph is accepted only when:

`candidate_energy.upper < incumbent_energy.lower`

If initial intervals overlap, the evaluator retries only their unresolved
component lengths with the expanded budget and applies the same strict proof
after merging non-weaker bounds. A proposal timeout without safe partial
evidence is rejected without changing the incumbent. An initial unsafe timeout
makes the episode inconclusive and excludes it from cohort selection.

Program failure is distinct from infrastructure failure: it receives the
locked worst candidate fitness, while infrastructure and contract failures do
not become fitness values.

## Cohort behavior

The eight-slot cohort remains serial. One evidence scorer is shared only for
deterministic caching, and every program result records its own scoring-attempt
counts. Cohort ranking uses the conservative fitness lower bound, then
exactness, interval width, upper bound, and the frozen canonical program order.
If any program lacks scientifically comparable evidence, the cohort produces
no selected program.

Wall-clock duration is retained as telemetry but excluded from evidence and
episode semantic hashes.

## Donor review

The evidence types, locked HEG adapter, and rational interval formulas were
ported selectively from commit
`ae527b7c1eb18f6fb83eac9a1f8af548b5935cc0` on
`native-v3-wip-71a9cc2`. Its Metropolis exploration, scheduler, process pool,
persistence, and validation-panel code were not ported.
