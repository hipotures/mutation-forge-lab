# Native v3 Step 12E0: transport-stable single-program schema

## Scope and decision

Issue #48 compared two non-recursive, one-program Structured Output contracts
without changing the Native v3 preview default:

1. a schema generated for one concrete brief/operator family;
2. a generic flat bounded IR compiled by the host.

Both candidates passed the medium and gated max criteria. The result is:

> **GO — select `slot_specific` for operator acceptance.**

Both candidates were eligible. `slot_specific` was selected by the documented
tie-break: smaller mean canonical schema, then lower medium token cost. This
does not activate it in preview. Issue #47 remains blocked until the operator
accepts this result.

## Implementation boundary

The model-facing formats are experimental adapters only. They do not replace
the internal `mforge.native.program.v3` AST or its interpreter.

`slot_specific` contains exactly the selector, literal arguments, pick mode,
relation-preserving action input, and terminal needed by its brief. Its
compiler constructs the existing:

```text
try
  let candidates = selector(...)
    let selected = pick(candidates)
      apply(...)
      emit
  no_plan(NO_MATCH)
```

`flat_ir` contains bounded `bindings[]`, bounded `steps[]`, and one explicit
`terminal`. The compiler resolves bindings sequentially, checks selector and
action types against the existing validator registry, limits total logical work
to eight steps, and rejects unresolved bindings, incompatible references,
aliases, hidden work under `no_plan`, and unterminated/no-op programs.

Every compiled result is passed to the existing `validate_program` path.
Canonicalization, program hashes, behavior signatures, interpreter semantics,
and scientific evaluation remain unchanged.

The provider-accepted schema subset required two details:

- arrays use object-valued `items`; `prefixItems` and boolean `items` are absent;
- discriminated alternatives use non-overlapping `anyOf`; `oneOf` is absent.

The host still enforces that `active_forbidden_lengths` equals the complete
ordered active set. The schema constrains every element to that active domain.

## Deterministic schema-complexity inventory

All system prompts were 673 bytes with SHA-256
`0774ad57d3e6f9ec3aaf1abeeed07f8dacbc1c640e341a83b4ff16ad38f3f380`.

| Candidate / brief | Canonical schema SHA-256 | Bytes | Objects | `anyOf` keywords / variants | `oneOf` | `$ref` / recursive | Max depth | Required / optional | Enum values / consts | Max steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rich recursive control (all briefs) | `47507acbc08c997b9270480702b422bd5de4d2ad03f633b4ea54aabd7e6a0314` | 12,125 | 61 | 2 / 40 | 0 | 29 / 17 | 11 | 131 / 0 | 57 / 58 | 256 |
| flat IR (all briefs) | `2906f517c522e5ee97596c5c90d104fa1a8bda2a1b443467cf37df45f105e5c8` | 4,744 | 20 | 3 / 11 | 0 | 0 / 0 | 12 | 50 / 0 | 17 / 20 | 8 |
| slot-specific add-edge | `ee716c45d85014445854432a3ac13dec7e790b93b3ce2c40124a46ff583ee3af` | 2,186 | 10 | 1 / 2 | 0 | 0 / 0 | 13 | 22 / 0 | 11 / 7 | 1 |
| slot-specific remove-edge | `f5bbe1483f8a999a63c4b16e913d61eebfc99d81a5d5f2f1e195757bb44f1f94` | 2,189 | 10 | 1 / 2 | 0 | 0 / 0 | 13 | 22 / 0 | 11 / 7 | 1 |
| slot-specific relocation | `4d77a3c84a23ff5a59cb44ceee90b069c88fef15f0543c1421ff8ef63454ff56` | 2,153 | 10 | 1 / 2 | 0 | 0 / 0 | 12 | 21 / 0 | 9 / 7 | 1 |
| slot-specific fanout | `307e53ea8225897b57236a559f60b4a30e0d687748efdfc9d37fa38992636087` | 2,128 | 10 | 1 / 2 | 0 | 0 / 0 | 12 | 21 / 0 | 9 / 7 | 1 |

The rich schema is 2.56 times the flat schema and 5.55–5.70 times the
slot-specific schemas. Only the rich control is recursive.

### Prompt identities

| Candidate | Brief | Prompt bytes | Prompt SHA-256 |
| --- | --- | ---: | --- |
| rich | add-edge | 6,175 | `79dee5c03daee74d9a261952bff493b613fbab8383cfb823110af10a6dceee80` |
| rich | remove-edge | 6,188 | `1dd566dd765bbebda717acec7316e57a56ba05dd8a4ff975e7fc6f7333cb4534` |
| rich | relocation | 6,177 | `f7c4b704ba63035d81c5d2e38f773483e6d06338add3ba32758665666019f322` |
| rich | fanout | 6,174 | `bfb3c757550dd975c89fedaf7e2c00bdea8a050b12ee50edbad060eda38bfc89` |
| flat IR | add-edge | 755 | `3836f14693f1e7dd6a59502beb602cd356529898165a1649617b5ab0c6af5de3` |
| flat IR | remove-edge | 729 | `6f51b3a0e0310f4c1a7b885f1b0922194c7f126ec8cf8db18d446f9c26f7c5a7` |
| flat IR | relocation | 792 | `c20a9e787a30caaf22b7280c08fdb11a14829d243592940ea71d54f1826a4859` |
| flat IR | fanout | 756 | `623cdc54a93ba1672b43fb220be0eae1dfbff145a2df5f8632ef8a826be29a79` |
| slot-specific | add-edge | 761 | `1a1a4b5f7ea746256ff675a26d345f449b206106096e77753a67b2dedea5faf1` |
| slot-specific | remove-edge | 735 | `d4d7be901d285d4722116a4d00cb7d9486ead2c06585b5d5e1a1af02ffd9c927` |
| slot-specific | relocation | 798 | `085366cf17fd8923ff1374b09ddeff251d4e5b7e6a1f599611e6d17179bf6f89` |
| slot-specific | fanout | 762 | `1d110a8ceef0b3036f70767f458d776a7fdd3f3b6054473b121fa6d7831b0636` |

## Offline conformance

Focused tests cover:

- deterministic schema hashes for all four slot-specific schemas and the flat
  schema;
- Draft 2020-12 schema validity;
- compiler and existing-validator acceptance for all four briefs;
- canonical compile/serialize/parse identity;
- missing terminal and missing rewrite fields;
- invalid enum aliases and forbidden lengths;
- unresolved and incompatible bindings;
- excessive work and no-op/unterminated programs;
- explicit `no_plan` without hidden work;
- fake persistent App Server lifecycle, independent medium gates, max gating,
  repeated-schema reporting, and exact 16-file successful artifact parity.

The candidate keyword set is a subset of the provider-accepted rich schema
keyword set. Every candidate object has `additionalProperties: false`, every
property is required, and every `anyOf` alternative has a disjoint `const`
discriminator.

## Live benchmark method

The final run used:

- model `gpt-5.6-luna`;
- one persistent App Server process per candidate and effort gate;
- one shared non-recursive specification anchor per process;
- brief order add-edge, remove-edge, relocation, fanout at `medium`;
- no repair in the primary measurement;
- add-edge and relational relocation only at `max`;
- no rich recursive max rerun;
- the same system prompt, forbidden lengths `(4, 8, 16)`, timeout, compiler,
  semantic validator, and artifact writer for both candidates.

The final canonical report is:

```text
/tmp/mforge-step12e0-final.AoHkh4/benchmark/benchmark-report.json.gz
```

The generated English projection is:

```text
/tmp/mforge-step12e0-final.AoHkh4/benchmark/benchmark-report.md
```

Three preflight workspaces are retained separately and are not benchmark
samples. They contain no generated candidate program response:

1. a raw bootstrap-ack key-order comparison defect;
2. provider HTTP 400 rejection of boolean `items`;
3. provider HTTP 400 rejection of `oneOf`.

The final schemas include the regression fixes for all three defects.

## Gate results

| Gate | Candidate | Turns | Transport | Reconnects / warnings | Schema | Compiler | Semantic | Canonical / behavior duplicates | Artifact parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | slot-specific | 4 | 4/4 | 0 / 0 | 4/4 | 4/4 | 4/4 | 0 / 0 | 4/4 |
| medium | flat IR | 4 | 4/4 | 0 / 0 | 4/4 | 4/4 | 4/4 | 0 / 0 | 4/4 |
| max | slot-specific | 2 | 2/2 | 0 / 0 | 2/2 | 2/2 | 2/2 | 0 / 0 | 2/2 |
| max | flat IR | 2 | 2/2 | 0 / 0 | 2/2 | 2/2 | 2/2 | 0 / 0 | 2/2 |

Both medium cohorts passed the eligibility gate. Exactly four max turns ran,
which is the issue limit.

## Per-turn results and token usage

All successful program turns had:

- no provider warning, reconnect, or upstream request ID;
- `json_decode`, schema conformance, compiler, and host semantic validation
  equal to `true`;
- complete Native v2 artifact parity;
- `cacheWriteInputTokens = 0`, `final = true`, and `partial = false`.

| Effort | Candidate | Brief | Wall ms | Input | Cached | Output | Reasoning | Total | First schema use |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| medium | flat IR | add-edge | 19,276 | 3,812 | 0 | 363 | 182 | 4,175 | yes |
| medium | flat IR | remove-edge | 14,301 | 4,374 | 2,816 | 192 | 20 | 4,566 | no |
| medium | flat IR | relocation | 14,189 | 4,813 | 0 | 181 | 14 | 4,994 | no |
| medium | flat IR | fanout | 14,231 | 5,284 | 3,840 | 182 | 6 | 5,466 | no |
| medium | slot-specific | add-edge | 14,042 | 3,445 | 0 | 181 | 35 | 3,626 | yes |
| medium | slot-specific | remove-edge | 13,993 | 3,821 | 0 | 176 | 36 | 3,997 | yes |
| medium | slot-specific | relocation | 13,551 | 4,228 | 0 | 154 | 23 | 4,382 | yes |
| medium | slot-specific | fanout | 13,452 | 4,667 | 0 | 145 | 7 | 4,812 | yes |
| max | flat IR | add-edge | 24,926 | 3,836 | 0 | 678 | 516 | 4,514 | yes |
| max | flat IR | relocation | 15,989 | 4,723 | 0 | 279 | 115 | 5,002 | no |
| max | slot-specific | add-edge | 21,813 | 3,467 | 0 | 597 | 459 | 4,064 | yes |
| max | slot-specific | relocation | 17,430 | 4,258 | 0 | 217 | 78 | 4,475 | yes |

### First-event timing

Times are milliseconds from the server's `turn/started.emittedAtMs`. `Reasoning`
is the first reasoning `item/started`, `agent` is the first
`item/agentMessage/delta`, and `usage` is the first
`thread/tokenUsage/updated`.

| Effort | Candidate | Brief | Turn start epoch ms | Reasoning | Agent | Usage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| medium | flat IR | add-edge | 1,786,061,127,640 | 834 | 4,265 | 9,239 |
| medium | flat IR | remove-edge | 1,786,061,146,936 | 788 | 1,292 | 4,263 |
| medium | flat IR | relocation | 1,786,061,161,259 | 820 | 1,232 | 4,152 |
| medium | flat IR | fanout | 1,786,061,175,469 | 829 | 1,094 | 4,194 |
| medium | slot-specific | add-edge | 1,786,061,058,108 | 704 | 1,493 | 4,015 |
| medium | slot-specific | remove-edge | 1,786,061,072,161 | 709 | 1,527 | 3,963 |
| medium | slot-specific | relocation | 1,786,061,086,164 | 717 | 1,613 | 3,523 |
| medium | slot-specific | fanout | 1,786,061,099,728 | 800 | 1,142 | 3,421 |
| max | flat IR | add-edge | 1,786,061,258,776 | 2,612 | 12,047 | 14,879 |
| max | flat IR | relocation | 1,786,061,283,733 | 849 | 3,076 | 5,938 |
| max | slot-specific | add-edge | 1,786,061,204,618 | 876 | 9,277 | 11,773 |
| max | slot-specific | relocation | 1,786,061,226,457 | 1,004 | 3,593 | 7,374 |

### Canonical identities

| Effort | Candidate | Brief | Program SHA-256 | Behavior signature |
| --- | --- | --- | --- | --- |
| medium | flat IR | add-edge | `54a1fc4bbd01f7fa0cb13f84e84c17c9e817f82cce7fc843d022cf4ea73333f8` | `cfb74846175f3aa30d8f636379765cb7be8e19a00994e64a0fdbcd4f54988e6e` |
| medium | flat IR | remove-edge | `d898f3ba697b6ed4b16c39b0c57486fbd2b40566c52a17217e33afa940eadaf6` | `4a55655cef448315d53d35b3c2aba809055979a1f1f483b3fc0e7d65a83ddc13` |
| medium | flat IR | relocation | `83e227114453eb9ae3f4a7bf3f5b7695a8b7379a4c7519c0f02dd6a234f096fe` | `3e094b69c720cb6ef3f097f280ba6ddc5fce8d396177d4699ad3e70490bc0efb` |
| medium | flat IR | fanout | `cc4c39893b9c5f5672d4ca5999c9ff813404d38d2d2183f015c4cec75c077410` | `7aef576dff1821847540e2e2d416082b2911bbd43288d3f3887f503f2cb3ce3b` |
| medium | slot-specific | add-edge | `4bf5450d808a33e802b3e56488b90a22ebe5bffaee9d3bbc1ce2c1e3848e0201` | `0990d5f95af231a8a04deb380a378f8c5e2d0a60850c1041256004a702d90a5d` |
| medium | slot-specific | remove-edge | `f598a10b271b04f11412b0a8e260da49a905277a9b57ac5850c5bdff5592c0cb` | `4539904b2b47354806946ee28e8e941fccbf3e0707e32bdab73c47379bfbaac1` |
| medium | slot-specific | relocation | `ee5e8458f76158e49b297e3c207cb0e0048d0636fd016ea25af266d073041ff7` | `c935f352b3d8b474019a52aa4921d122d4be312dadbfae4712fc175577811bfe` |
| medium | slot-specific | fanout | `ca87be577a7fdaa2f5557a0510acbfe4ca9404f15fe5afd72ef3c7f6246be312` | `975b8ba51416049ca4aaacec07689e883e25eed91c6c7adc45c3a38a564a0c73` |
| max | flat IR | add-edge | `0bbaa2dae37908c19a1de212e6880c4311d59e2e4dc330a3fe32d2adfaa592b9` | `0990d5f95af231a8a04deb380a378f8c5e2d0a60850c1041256004a702d90a5d` |
| max | flat IR | relocation | `dfe7cd6b4575876a50b6555a6fdc7ea093dea82ed41f01a5378ca2829f15e63e` | `c935f352b3d8b474019a52aa4921d122d4be312dadbfae4712fc175577811bfe` |
| max | slot-specific | add-edge | `4bf5450d808a33e802b3e56488b90a22ebe5bffaee9d3bbc1ce2c1e3848e0201` | `0990d5f95af231a8a04deb380a378f8c5e2d0a60850c1041256004a702d90a5d` |
| max | slot-specific | relocation | `ee5e8458f76158e49b297e3c207cb0e0048d0636fd016ea25af266d073041ff7` | `c935f352b3d8b474019a52aa4921d122d4be312dadbfae4712fc175577811bfe` |

Duplicate gates are evaluated within each candidate cohort. Similar behavior
between different representations or effort gates is expected and is not
representation collapse within a cohort.

### Aggregate usage

| Gate | Candidate | Input | Cached input | Cache-write input | Output | Reasoning output | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | slot-specific | 16,161 | 0 | 0 | 656 | 101 | 16,817 |
| medium | flat IR | 18,283 | 6,656 | 0 | 918 | 222 | 19,201 |
| max | slot-specific | 7,725 | 0 | 0 | 814 | 537 | 8,539 |
| max | flat IR | 8,559 | 0 | 0 | 957 | 631 | 9,516 |

## Bootstrap and persistence evidence

| Effort | Candidate | Wall ms | Reasoning / agent / usage ms | Input / output / reasoning / total | App Server request | Thread | Turn | Ack / parity |
| --- | --- | ---: | --- | --- | ---: | --- | --- | --- |
| medium | flat IR | 14,393 | 2,005 / 2,408 / 3,569 | 2,954 / 83 / 13 / 3,037 | 10 | `019fd989-ca6d-7250-afd6-82c765d1c097` | `019fd989-caa0-78e1-a8c1-6478f6a0d75c` | yes / yes |
| medium | slot-specific | 14,079 | 1,760 / 2,219 / 3,371 | 2,954 / 87 / 17 / 3,041 | 10 | `019fd988-bbab-7c41-8c13-c6914a6fca62` | `019fd988-bbdc-7532-a291-8704a9a6615c` | yes / yes |
| max | flat IR | 14,791 | 1,976 / 2,790 / 3,946 | 2,954 / 107 / 37 / 3,061 | 10 | `019fd98b-c919-7ae2-a5fe-a719b86839e1` | `019fd98b-c952-7122-9a22-fb6069ce991f` | yes / yes |
| max | slot-specific | 14,834 | 2,032 / 2,860 / 4,029 | 2,954 / 109 / 39 / 3,063 | 10 | `019fd98a-f547-71f0-a15c-fb92f1723153` | `019fd98a-f57c-79c0-80e7-9634a8b7c742` | yes / yes |

Each process kept one durable thread across its bootstrap and candidate turns.
Every successful prefix contained the exact 16-file Native v2 artifact set.

## Repeated-schema and cache observation

The generic flat IR supplied the same canonical schema hash on later turns in
the same App Server process.

At medium:

| Use | Wall ms | First agent delta ms | Reconnects | Input | Cached input |
| --- | ---: | ---: | ---: | ---: | ---: |
| first: add-edge | 19,276 | 4,265 | 0 | 3,812 | 0 |
| repeated: remove-edge | 14,301 | 1,292 | 0 | 4,374 | 2,816 |
| repeated: relocation | 14,189 | 1,232 | 0 | 4,813 | 0 |
| repeated: fanout | 14,231 | 1,094 | 0 | 5,284 | 3,840 |

At max:

| Use | Wall ms | First agent delta ms | Reconnects | Input | Cached input |
| --- | ---: | ---: | ---: | ---: | ---: |
| first: add-edge | 24,926 | 12,047 | 0 | 3,836 | 0 |
| repeated: relocation | 15,989 | 3,076 | 0 | 4,723 | 0 |

The repeated turns were faster in this bounded sample. `cachedInputTokens`
varied independently and is not evidence of schema-grammar caching.
Slot-specific schemas intentionally have distinct hashes, so they provide no
same-hash cache observation.

## Recommendation

`slot_specific` satisfies every selection condition:

1. 4/4 medium transport completion with zero reconnects;
2. 4/4 schema, compiler, and semantic validity;
3. no canonical or behavior-signature duplicates within the cohort;
4. 2/2 clean and semantically valid max turns;
5. no model-facing recursion or `$ref`;
6. 2,128–2,189 byte schemas versus the 12,125 byte control;
7. deterministic compilation into the existing internal AST;
8. unchanged Native v2 provider artifacts and scientific semantics.

The flat IR also passed, but its 4,744 byte schema and medium token total were
larger. It remains a useful general experimental representation, not the
selected integration contract.

## Repository and product invariants

The final live report was produced from repository head
`0307b935464d1eb0ade695fd39cdf2d44e1fffdf`. Repository status was byte-for-byte
unchanged by the benchmark. The pre-existing local `experiment.toml`
modification was not read, changed, staged, or included.

This issue did not change:

- the production preview default;
- `persistent_single_ast` topology or its operator decision;
- Native v2 transport, authentication, process lifecycle, retry accounting, or
  provider-turn artifact names;
- scoring, interval fitness, evaluation, SQLite, dashboard, or HEG;
- issue #47 or Step 13.

## Known limitations

- Each medium candidate has one bounded four-brief cohort.
- Each eligible max candidate has only two turns.
- Successful CLI notifications expose reasoning-item start but no
  reasoning-token delta event. The report records reasoning-item, first-agent,
  first-usage, and total wall timing; token-level reasoning latency is
  unavailable.
- Successful turns emitted no upstream error request IDs; arrays are empty.
- Cached input tokens do not identify schema-grammar cache behavior.
- App Server retains its small platform-owned runtime wrapper.
- The canonical machine report is in a disposable `/tmp` workspace and is not
  a committed production artifact.

## Commands

The focused implementation checks were:

```bash
uv run pytest -q tests/unit/test_native_v3_single_program_ir.py \
  tests/unit/test_native_v3_transport_schema_experiment.py
uv run ruff check src/mutation_forge/native_v3/single_program_ir.py \
  scripts/native_v3_transport_schema_experiment.py \
  tests/unit/test_native_v3_single_program_ir.py \
  tests/unit/test_native_v3_transport_schema_experiment.py
uv run mypy src/mutation_forge/native_v3/single_program_ir.py \
  scripts/native_v3_transport_schema_experiment.py
```

The final live command was:

```bash
uv run python scripts/native_v3_transport_schema_experiment.py \
  --workspace /tmp/mforge-step12e0-final.AoHkh4/benchmark \
  --auth-json /home/user/.codex/auth.json \
  --model gpt-5.6-luna \
  --turn-timeout 600
```

The final regression commands and commit SHA are added to the issue completion
comment after repository validation.

STOP — waiting for operator acceptance
