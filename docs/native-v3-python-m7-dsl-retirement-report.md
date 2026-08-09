# Native v3 ordinary-Python M7 DSL retirement report

Date: 2026-08-09

Issue: `#56`

## Outcome

M7 removes the superseded recursive JSON mutation language from production
while preserving Native v2 as the default and the explicit ordinary-Python
preview as the only Native v3 preview route.

The ordinary-Python path remains:

```text
Codex App Server
-> two-field Python response
-> M1 validation and identity
-> M2 isolated worker
-> M3 serial scientific evaluator
-> M5 generation, selection, and exact fork loop
-> current HEG evidence and interval fitness
-> exact-verification seam
```

No generated source executes outside the accepted M2 worker.

## Archival gate

The accepted M6 head is archived at the remote branch:

```text
archive/native-v3-python-m6-ee3284d7
ee3284d7d993c715a1b77ab2097dea0228995074
```

The original accepted M6 workspace under `/tmp` was lost during the disclosed
power outage. It was not reconstructed from hashes. A new bounded pre-cleanup
campaign was therefore run from the accepted M6 head into the durable root:

```text
/home/user/DEV/mutation-forge-evidence/
  native-v3-python-m6-precleanup-ee3284d7/
```

The first process stopped at the App Server event cap with five pending slots.
One exact resume completed only those five slots. The final durable cohort is:

- generation 0: eight roots;
- generation 1: four children and four fresh roots;
- sixteen terminal slots;
- fourteen evaluated programs;
- one contract-invalid slot;
- one provider-failed slot;
- twenty-eight development evaluations;
- fourteen exact Python source files with source, canonical AST, program, and
  behavior identities;
- two immutable generation manifests and two Search Memory projections;
- final status `completed`, stop reason `generation_budget`;
- one resume, with all pre-resume candidate, evaluation, provider, manifest,
  Search Memory, and provenance artifacts byte-identical;
- exact-verifier authority `exact_verifier_only`, zero submissions, zero
  records, and no verified counterexample.

The deterministic archive is:

```text
/home/user/DEV/mutation-forge-evidence/
  native-v3-python-m6-precleanup-ee3284d7.tar.gz
```

Archive SHA-256:

```text
ab09a13773623a325f05a132c6bba6d9d088b60f565be488ae8195a9f18bf856
```

It contains 636 files and excludes Codex authentication, capsule credentials,
and private provider state outside the retained experiment tree.

## Durable replay baseline

`scripts/native_v3_python_replay_evidence.py` revalidates every retained
source, recomputes identities, executes each evaluated program through M2 and
M3 on the immutable HEG panel, and compares timing-free scientific and semantic
projections.

The pre-cleanup replay contains fourteen candidates and twenty-eight cases:

```text
all_semantic_replays_match = true
compressed SHA-256 =
  5efa8b9b7f80fd6fa294b7b628c9f5a5720ee7c3bde9daa179b0ec16871a9a8d
internal report SHA-256 =
  eca92a0a699f33f03119cadc23ed33d51b34a9662d9c23670f95a57dc8ae2cce
```

After DSL removal, the same replay command produced a byte-identical gzip
report. Program/source/AST/behavior identities, manifest hashes, Search Memory
hashes, lineage, panel, budgets, failure taxonomy, fitness, semantic traces,
and exact-verifier state therefore did not change.

## Removed production surface

Cleanup commit:

```text
1c12c8b9c29411698e183c41db324450612329c1
```

Removed:

- recursive JSON program contracts and validator;
- JSON-AST graph runtime and interpreter;
- flat and slot-specific IR compiler;
- cohort, single-program, persistent, compaction, lineage, and preview DSL
  orchestration;
- DSL provider smoke/evaluation routes;
- DSL Search Memory implementation;
- DSL-only prompts and response schemas;
- DSL-only scripts, fixtures, integrations, and unit tests;
- the `protocol = "v3"` CLI route;
- the `native-v3-program-batch` provider/artifact projection.

Preserved:

- Native v2 assets, prompts, schema, lifecycle, and default routing;
- App Server transport and frozen artifact contract;
- representation-independent canonicalization, execution records, scoring,
  HEG adapter, randomness, and serial scientific core;
- ordinary-Python M1 through M6 implementation;
- historical reports and the exact M6 archival branch.

The explicit Python selector remains `native-v3-python-v1`. Omitting
`protocol` still selects Native v2. The removed `v3` selector fails closed and
cannot reinterpret an old JSON-DSL workspace.

## Post-cleanup guarded preview

The post-cleanup campaign is retained at:

```text
/home/user/DEV/mutation-forge-evidence/
  native-v3-python-m7-postcleanup-1c12c8b/
```

Its first process reached the cumulative App Server event limit after seven
generation-0 slots. One resume used the same `exp_id`, immutable manifests,
panel, budgets, prompts, schema, provenance, and capsule. It submitted only the
pending work.

Final result:

```text
planned                       16
terminal                      16
evaluated                     10
contract_invalid               5
provider_failed                1
pending                        0
generation 0 roots             8
generation 1 children          4
generation 1 fresh roots       4
resume attempts                1
terminal reason                generation_budget
```

All four children have valid exact-parent lineage and differ from their parent
in source, program identity, and behavior signature. Every evaluated candidate
received the same two-case, non-held-out order-30 panel, seeds, horizon,
witness cap, HEG scorer, and budget. All pre-resume immutable artifacts remained
byte-identical.

Runtime evidence records:

- `ordinary-python` preview mode;
- `dsl_runtime_used = false`;
- frozen M2 worker protocol and limits;
- Linux bubblewrap/seccomp/rlimit sandbox controls;
- complete-under-cap HEG score evidence;
- no scorer or model activity inside the evaluator;
- exact-verifier authority only, with zero submissions and no heuristic
  `VERIFIED` result.

## No-DSL proof

Repository and fresh-process import checks prove that production cannot import
or dispatch:

- `mutation_forge.native_v3.contracts`;
- `mutation_forge.native_v3.graph_runtime`;
- `mutation_forge.native_v3.interpreter`;
- `mutation_forge.native_v3.single_program_ir`;
- the deleted single-program, cohort, persistent, compaction, lineage,
  Search Memory, provider, or experiment routes.

The removed module specs resolve to `None`. The Python preview imports none of
them, and production contains no `invoke_program`, `evaluate_serial_program`,
or `compile_program` dispatch. The old DSL prompts and schemas are absent.

## Verification

The focused cleanup packet passed:

- Ruff;
- mypy over 155 source files;
- 204 Python-preview, provider, safe-API, cleanup, and App Server tests;
- fresh-process import and removed-module probes;
- repository reference scan;
- `git diff --check`;
- unchanged `experiment.toml`.

The final Native v2 smoke, App Server artifact parity, focused M1-M7 suite, and
full-suite zero-regression delta are recorded in the issue report produced from
the final synchronized head.
