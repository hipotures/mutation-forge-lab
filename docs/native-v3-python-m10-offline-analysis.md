# M10 offline analysis and resume budget

This report analyzes the preserved
`native-v3-python-m10-sustained-v1` workspace without resuming the
campaign. No App Server, provider, or model call was made. The preserved
workspace fingerprint was unchanged. The machine-readable extraction is
preserved at
`/home/user/DEV/mutation-forge-lab-evidence/m10/offline-analysis/workspace-analysis-v1.json`
(SHA-256
`ac0205896c6e91f3baeb8526e691da9ee786d2c3f90f819da090dba481b39da0`).

## Evaluated-program ranking

The frozen development panel contains two order-30 cases. Ranking uses the
canonical exact aggregate fitness; equal fractions share a rank.

| Rank | Candidate | Kind | Parent | Program | Exact fitness | Decimal |
|---:|---|---|---|---|---:|---:|
| 1 | `g0000-slot-01` | root | — | `19f705303dfa…` | `157427937/270610316` | 0.5817514252 |
| 2 | `g0000-slot-07` | root | — | `fc0d5659ce14…` | `157426373/270610316` | 0.5817456456 |
| 3 | `g0000-slot-05` | root | — | `cdf7d460d397…` | `314148555/541220632` | 0.5804445293 |
| 4 | `g0001-slot-03` | child | `g0000-slot-06` | `0e231fa9680b…` | `78361091/135305158` | 0.5791434130 |
| 5 | `g0001-slot-06` | root | — | `b80e3809a26a…` | `311334919/541220632` | 0.5752458435 |
| 6 | `g0000-slot-06` | root | — | `f777e0ff9a74…` | `38828841/67652579` | 0.5739447272 |
| 7 | `g0001-slot-04` | root | — | `39edfea48b5f…` | `38828450/67652579` | 0.5739389477 |
| 8 | `g0000-slot-04` | root | — | `355123509a0e…` | `38827668/67652579` | 0.5739273886 |
| 9 | `g0001-slot-05` | root | — | `22b199ccf1c8…` | `309926537/541220632` | 0.5726436109 |
| 10 | `g0000-slot-00` | root | — | `5af521c6228f…` | `154609609/270610316` | 0.5713367150 |
| 10 | `g0000-slot-02` | root | — | `5aca3c2f44ff…` | `154609609/270610316` | 0.5713367150 |
| 10 | `g0000-slot-03` | root | — | `4a490e7a4309…` | `154609609/270610316` | 0.5713367150 |
| 10 | `g0001-slot-02` | child | `g0000-slot-04` | `090047dcb993…` | `154609609/270610316` | 0.5713367150 |
| 10 | `g0001-slot-07` | root | — | `57561fafaf84…` | `154609609/270610316` | 0.5713367150 |

Three terminal slots were provider failures and produced no program:
`g0001-slot-00`, `g0001-slot-01`, and `g0002-slot-00`. The seven
interrupted or unstarted generation-2 slots remain nonterminal and are not
scientific failures.

## Children and retained parents

Two children reached evaluation:

- `g0001-slot-03` improved over `g0000-slot-06` by
  `703409/135305158` (`+0.0051986858`) and was retained.
- `g0001-slot-02` was worse than `g0000-slot-04` by
  `701063/270610316` (`-0.0025906736`) but remained one of the four
  deterministic parents selected from generation 1.

The other generated children did not yield comparable programs:
`g0001-slot-00`, `g0001-slot-01`, and `g0002-slot-00` ended in provider
failure.

Generation 2 froze these retained parents:

| Global rank | Parent | Fitness | Assigned child | Current child state |
|---:|---|---:|---|---|
| 4 | `g0001-slot-03` | 0.5791434130 | `g0002-slot-00` | provider failure |
| 7 | `g0001-slot-04` | 0.5739389477 | `g0002-slot-02` | interrupted |
| 9 | `g0001-slot-05` | 0.5726436109 | `g0002-slot-01` | interrupted |
| 10 | `g0001-slot-02` | 0.5713367150 | `g0002-slot-03` | queued |

The remaining pending identities, `g0002-slot-04` through
`g0002-slot-07`, are fresh roots.

## Actual policy behavior

Across 28 durable evaluation cases, programs made 28 proposals:

- 26 legal rewrite plans: 12 accepted and 14 rejected;
- 2 `NoPlan` outcomes, both `ILLEGAL_FINAL_STATE`;
- 0 program/runtime failures;
- 0 contract-invalid programs;
- 0 duplicates.

Actual relation-aware selector calls were:

- `edges_witness_load_extreme`: 26;
- `matching_k_switch_reconnections_for_edge`: 22;
- `edge_fanouts_legal_for_edge`: 4;
- `relocations_legal_for_edge`: 2;
- `edges_bridge_risk`: 2.

Actual attempted action families were 22 `k_switch`, 4 `edge_fanout`, and
2 `relocate_endpoint`. The best program used
`edges_witness_load_extreme`,
`matching_k_switch_reconnections_for_edge`, and `k_switch` in both
development cases; both rewrites were accepted.

There were zero apparent-zero events, zero exact-verifier submissions, zero
exact-verifier results, and no verified counterexample.

## Frozen offline generalization panel

No historical held-out panel was compatible with M10: the older panels use
the removed JSON DSL and Toy backend. A four-case M10-compatible panel was
therefore frozen before evaluation with new graph seeds
`107, 109, 113, 127` and policy seeds `23, 29, 31, 37`. It keeps the M10
HEG commit, graph mode, order 30, horizon 1, witness cap 64, and forbidden
lengths `(4, 8, 16)`.

- Panel:
  `/home/user/DEV/mutation-forge-lab-evidence/m10/offline-analysis/generalization-panel-v1.json`
- Panel SHA-256:
  `9c6c813e8fadb96dc2824428ad887582f84f8b4ae4d3ec87bf5211cd652c0059`
- Result:
  `/home/user/DEV/mutation-forge-lab-evidence/m10/offline-analysis/generalization-result-v1.json`
- Result SHA-256:
  `7fbba021a09be1d4b85b2658f6ad97de94e12026af2e677990b1f880b806718b`

All four cases completed with the same relation-aware selector/action
family, no `NoPlan`, illegal-final-state, or program failure, and no external
activity. There was 1 accepted and 3 rejected legal rewrites.

The exact held-out aggregate was
`623367773/1082441264` (`0.5758906222`). The same-cases no-action initial
graph baseline was `311331791/541220632` (`0.5752400640`), so the program
improved it by `704191/1082441264` (`+0.0006505582`). Held-out fitness was
below development fitness by `6343975/1082441264` (`-0.0058608030`).

The result passes the criterion frozen before evaluation and supports
limited exploratory seed generalization. It is not preregistered
confirmatory evidence, does not establish broader order generalization, and
is not an exact counterexample certificate.

## Existing resume behavior

The old resume command is not budget-safe for this state. Four pending slots
have interrupted provider reservations with no durable result, so the
current implementation fails closed instead of retrying them. Three
unstarted slots could make calls. If generation 2 completed, the configured
12-generation loop would continue through generations 3–11.

The frozen configuration permits 96 primary and 24 repair turns. With 21
primary and 1 repair already submitted, the old command can still admit up
to 75 primary and 23 repair calls. A runtime-only resume guard is therefore
required. It must validate the seven pending identities, allow one new
primary attempt for each (including explicit retries of the four
interrupted slots), admit at most two additional repair calls, and return a
resumable budget stop immediately after generation 2 commits.

The implemented guard keeps the frozen scientific configuration unchanged.
It fails closed unless generation 2 is the sole incomplete, latest
generation and its pending identities are exactly `g0002-slot-01` through
`g0002-slot-07`. Four interrupted initial turns receive distinct explicit
retry identities; terminal slots are never scheduled again. It cannot
create generation 3.

Future authorized resume command:

```console
uv run mforge experiment run --config configs/scientific/native-v3-python-m10-v1.toml --json --resume-current-generation 7 --max-new-repair-turns 2
```

This invocation can make at most 9 token-consuming calls: exactly 7 primary
attempts and at most 2 additional repairs. The persisted anchor is reused
without another provider call. Completing the current generation returns a
resumable `resume_generation_complete` budget stop; it does not schedule the
next generation.
